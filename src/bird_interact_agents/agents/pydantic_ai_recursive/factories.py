"""Agent factories + tool registrars for the recursive adapter.

Three agent roles, three different tool sets:

* **Root clarifier**: `spawn_subagent` only. No `ask_user`, no
  `submit_query`, and no MCP toolset — the root's job is slicing the
  user's question into logical units, not looking up datasource
  entities; sub-clarifiers do the search / inspect work themselves.
* **Sub-clarifier**: `search`/`inspect_model`/etc. via MCP, plus
  `ask_user` and `spawn_subagent` (for compound replies). No
  `submit_query`, no `query`.
* **Query-constructor**: `search`/`inspect_model`/`query`/etc. via
  MCP, plus `ask_user` and `submit_query`. No `spawn_subagent`.

The `spawn_subagent` tool is registered with `sequential=True` so a
model batch emitting multiple spawns runs them serially — parallel
execution would race on `shared.agent_records.append` and corrupt the
`parent_idx` topology.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from bird_interact_agents.agents._run_capture import (
    _count_turns,
    _extract_tool_stats,
    _serialize_messages,
)
from bird_interact_agents.agents._submit import (
    ask_user_impl,
    submit_slayer_query,
)
from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
    AgentRecord,
    TaskDeps,
    _LegacyAdapter,
)
from bird_interact_agents.agents.pydantic_ai_recursive.prompts import (
    SUB_CLARIFIER_PROMPT,
)
from bird_interact_agents.harness import MAX_MODEL_TURNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prepare_tools shim — same idea as pydantic_ai/agent.py:_make_prepare_tools.
# ---------------------------------------------------------------------------


def _make_prepare_tools(strict_value: bool):
    """Force a uniform `strict` on every tool definition right before
    each model request. Cerebras's OpenAI-compatible API rejects requests
    with inconsistent `strict` values; pydantic-ai's default is None,
    and MCP-server tools also come in as None — the merged list violates
    the constraint. This callback enforces uniformity.
    """

    async def _force_strict(
        ctx: RunContext, tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [replace(td, strict=strict_value) for td in tool_defs]

    return _force_strict


# ---------------------------------------------------------------------------
# SLayer client lazy init — routed through _LegacyAdapter so the
# existing submit_slayer_query helper can reuse the cache.
# ---------------------------------------------------------------------------


def _slayer_client_factory(adapter: _LegacyAdapter):
    """Build (or return cached) SlayerClient. Cached on the shared
    state so root / sub-clarifiers / constructor share one client."""
    if adapter._slayer_client is None:
        from slayer.client.slayer_client import SlayerClient
        from slayer.storage.yaml_storage import YAMLStorage

        storage = YAMLStorage(base_dir=adapter.slayer_storage_dir)
        adapter._slayer_storage = storage
        adapter._slayer_client = SlayerClient(storage=storage)
    return adapter._slayer_client


# ---------------------------------------------------------------------------
# Tool registrars — share signatures with the existing pydantic_ai adapter
# so _submit.* duck-typing keeps working.
# ---------------------------------------------------------------------------


def _register_ask_user(agent: Agent) -> None:
    @agent.tool
    async def ask_user(ctx: RunContext[TaskDeps], question: str) -> str:
        """Ask the user a clarification question about their query."""
        return await ask_user_impl(_LegacyAdapter(ctx.deps), question, "slayer")


_WRAPPED_KEYS = ("queries", "nested_queries")


def _projection_count(parsed: Any) -> int | None:
    """Return the count of output columns in a parsed SLayer submission,
    or None if the shape can't be counted defensively.

    For a single-stage dict, count = len(dimensions) + len(measures).
    For a nested-DAG list, count = same applied to the LAST stage (the
    DAG root). For anything else — top-level scalar, empty list,
    non-dict last-stage, wrapped variants like `{queries: [...]}`,
    or non-list dimensions/measures values — return None so the
    closure-bound check skips and lets `submit_slayer_query` produce
    its canonical shape-error from DEV-1435.

    Total + non-raising: every code path returns either int or None;
    no exception escapes. The closure-bound gate depends on this
    contract — an exception here would surface as a tool failure
    instead of a clean shape-error.
    """
    # Pull last stage out of a nested list first.
    if isinstance(parsed, list):
        if not parsed:
            return None
        last = parsed[-1]
        if not isinstance(last, dict):
            return None
        parsed = last
    if not isinstance(parsed, dict):
        return None
    # Wrapped variants — these should hit the helper's sharp shape-
    # error path, not be counted as zero-column submissions.
    if any(isinstance(parsed.get(k), list) for k in _WRAPPED_KEYS):
        return None
    dims = parsed.get("dimensions", [])
    measures = parsed.get("measures", [])
    if dims is None:
        dims = []
    if measures is None:
        measures = []
    if not isinstance(dims, list) or not isinstance(measures, list):
        return None
    return len(dims) + len(measures)


def _register_submit_query(
    agent: Agent, confirmed_projection: tuple[str, ...],
) -> None:
    """Register the constructor's `submit_query` with a closure-bound
    count check. `confirmed_projection` is captured by reference; on
    each submit call the tool counts the draft's dims+measures
    against `len(confirmed_projection)` and raises `ModelRetry` on
    mismatch BEFORE calling `submit_slayer_query`.

    The pre-helper rejection means a count-mismatch submission
    consumes NO bird-coin budget — per the DEV-1432 broader rule,
    only calls that reach `execute_submit_action` charge submit_query
    cost. The constructor can retry until pydantic-ai's `retries`
    budget exhausts without burning coins.
    """
    expected_count = len(confirmed_projection)
    confirmed_list = list(confirmed_projection)
    # Format the confirmed list the same way the prompt does — one
    # numbered line per column — so the agent sees the same shape in
    # both the prompt and the ModelRetry message and doesn't have to
    # translate between Python repr and numbered text.
    confirmed_block = "\n".join(
        f"  {i + 1}. {name}" for i, name in enumerate(confirmed_list)
    )

    @agent.tool
    async def submit_query(ctx: RunContext[TaskDeps], query_json: str) -> str:
        """Submit your final SLayer query for evaluation.

        `query_json` is a JSON string whose top-level value is either a
        single SlayerQuery object (e.g. `{"source_model": "orders",
        "measures": ["amount:sum"]}`) or a nested-DAG array of stage
        objects (same shape `query_nested` accepts; last element is the
        DAG root). The chosen query is translated to SQL deterministically
        and tested against the ground truth.

        Your submission must project EXACTLY the columns the
        projection-resolver confirmed (in the prompt as CONFIRMED
        PROJECTION). A count mismatch is hard-rejected here before
        the helper is called — no budget is charged on rejection.

        If the rendered SQL has a SQLite-side error (missing function /
        missing column / window misuse), the dry-run gate returns the
        error to you WITHOUT charging the submit cost. Use
        `inspect_model` or `models_summary` (1 coin each) to verify
        schema before resubmitting.
        """
        # Closure-bound count check. Skip for unparseable JSON (the
        # helper owns the "Invalid JSON" message) and for shapes that
        # `_projection_count` can't safely count (wrapped variants,
        # scalars, empty lists — all handed off to the helper).
        try:
            parsed = json.loads(query_json)
        except json.JSONDecodeError:
            adapter = _LegacyAdapter(ctx.deps)
            return submit_slayer_query(
                adapter, query_json, _slayer_client_factory,
            )

        observed = _projection_count(parsed)
        if observed is not None and observed != expected_count:
            raise ModelRetry(
                f"Your submission has {observed} projected column(s), "
                f"but the projection-resolver confirmed exactly "
                f"{expected_count}:\n{confirmed_block}\n"
                f"Align your dimensions+measures to that count and "
                f"order, or call ask_user to surface the disagreement "
                f"with the user before retrying. This rejection "
                f"consumed no budget."
            )

        adapter = _LegacyAdapter(ctx.deps)
        return submit_slayer_query(
            adapter, query_json, _slayer_client_factory,
        )


def _register_spawn_subagent(
    agent: Agent,
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str,
) -> None:
    """Register `spawn_subagent` on the given agent.

    Important properties pinned by tests:

    * `sequential=True` so a model batch emitting multiple spawn calls
      runs them serially (preventing index races on agent_records).
    * Depth check fires BEFORE child construction — no child agent is
      built when `depth >= max_depth`.
    * Pre-reserves the child's record slot synchronously BEFORE awaiting
      the child's run, so grandchildren's `parent_idx` lands on the
      correct slot regardless of completion order.
    * Captures the child's own agent-side tokens via an explicit
      `add_call(scope='agent', ...)` from `child_run.usage()`.
    """

    @agent.tool(sequential=True)
    async def spawn_subagent(
        ctx: RunContext[TaskDeps],
        focus: str,
        instruction: str,
    ) -> str:
        """Spawn a sub-clarifier agent focused on ONE logical block of
        the user's question. Sibling spawn calls in one model batch run
        sequentially. The sub-agent returns a description of its slice."""
        deps = ctx.deps
        if deps.depth >= deps.max_depth:
            return (
                f"max_depth={deps.max_depth} reached at depth "
                f"{deps.depth}; answer from what you already know."
            )

        # Pre-reserve THIS spawn's record slot synchronously, BEFORE any
        # await — otherwise concurrent sibling completions interleave
        # the appends and break parent_idx.
        parent_record_idx = deps.self_record_idx
        record_partial = AgentRecord(
            role="sub_clarifier",
            depth=deps.depth + 1,
            parent_idx=parent_record_idx,
            focus=focus,
            instruction=instruction,
            started_at=time.monotonic(),
        )
        deps.shared.agent_records.append(record_partial)
        my_child_idx = len(deps.shared.agent_records) - 1

        child = _build_sub_clarifier(
            model=model,
            model_settings=model_settings,
            shared_slayer_server=shared_slayer_server,
            self_model_id=self_model_id,
        )
        child_deps = TaskDeps(
            shared=deps.shared,
            depth=deps.depth + 1,
            max_depth=deps.max_depth,
            self_record_idx=my_child_idx,
        )
        child_prompt = SUB_CLARIFIER_PROMPT.format(
            budget=deps.shared.status.remaining_budget,
            db_name=deps.shared.db_name,
            focus=focus,
            instruction=instruction,
        )

        try:
            run = await child.run(
                instruction,
                instructions=child_prompt,
                deps=child_deps,
                usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
            )
        except Exception as e:  # noqa: BLE001 — defensive
            logger.exception(
                "Subagent failed at depth %d focus=%r",
                child_deps.depth, focus,
            )
            record_partial.error = f"{type(e).__name__}: {e}"
            record_partial.user_sim_transcript = list(
                child_deps.user_sim_transcript,
            )
            record_partial.usage = child_deps.usage
            record_partial.ended_at = time.monotonic()
            return f"Subagent error: {type(e).__name__}: {e}"

        # Capture child's own agent-side tokens — pydantic-ai records
        # them on run.usage(), but they're NOT auto-merged into
        # child_deps.usage (which only holds the user-sim path tokens
        # written by acompletion_tracked).
        run_usage = run.usage()
        child_deps.usage.add_call(
            scope="agent",
            model=self_model_id,
            prompt=getattr(run_usage, "input_tokens", 0) or 0,
            completion=getattr(run_usage, "output_tokens", 0) or 0,
            cache_read=getattr(run_usage, "cache_read_tokens", 0) or 0,
            cache_write=getattr(run_usage, "cache_write_tokens", 0) or 0,
        )
        record_partial.output = str(run.output)
        record_partial.user_sim_transcript = list(
            child_deps.user_sim_transcript,
        )
        record_partial.usage = child_deps.usage
        record_partial.messages = _serialize_messages(run)
        record_partial.tool_call_stats = _extract_tool_stats(run)
        record_partial.n_agent_turns = _count_turns(run)
        record_partial.ended_at = time.monotonic()
        return str(run.output)


# ---------------------------------------------------------------------------
# Agent factories — one per role.
# ---------------------------------------------------------------------------


def _build_root_clarifier(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    max_depth: int,
    self_model_id: str = "unknown",
) -> Agent:
    """Root clarifier: spawn_subagent only. NO SLayer toolset, NO ask_user,
    NO submit_query.

    `shared_slayer_server` is still accepted because it must be
    forwarded into the sub-clarifiers spawned by this root — but it is
    deliberately NOT wired into the root's own Agent. The root's job is
    to slice the user's question into logical blocks; giving it
    `search` / `help` / `inspect_model` tempts the model into looking
    up tables and naming them in the handoff, starving the sub-
    clarifier's table-family disambiguation step of candidates.
    """
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_spawn_subagent(
        agent,
        model=model,
        model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
    )
    return agent


def _build_sub_clarifier(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """Sub-clarifier: ask_user + spawn_subagent (for compound replies).
    NO submit_query, NO query (the latter belongs to the constructor)."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent)
    _register_spawn_subagent(
        agent,
        model=model,
        model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
    )
    return agent


def _build_query_constructor(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    confirmed_projection: tuple[str, ...],
    self_model_id: str = "unknown",
) -> Agent:
    """Query-constructor: ask_user + submit_query. NO spawn_subagent.

    `confirmed_projection` is closure-captured into the constructor's
    `submit_query` wrapper to enforce a count-match against the
    Stage-2 projection-resolver's output. A required (no-default)
    argument: there's no safe placeholder — agent.py must invoke
    Stage 2 first and pass that list here.
    """
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent)
    _register_submit_query(agent, confirmed_projection)
    return agent


def _build_projection_resolver(
    *,
    model: Any,
    model_settings: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """Projection-resolver (Stage 2): `ask_user` only, structured
    output `list[str]`. NO submit_query, NO query, NO spawn_subagent,
    NO MCP toolset.

    Stage 2's job is narrow: read the user question + the clarifier-
    tree spec, ask the user-sim to confirm an ordered list of output
    column names, return the confirmed list. The closure-bound check
    on the constructor's submit_query depends on this list's length —
    so this agent has no business doing anything else.
    """
    kwargs: dict[str, Any] = dict(
        model=model,
        deps_type=TaskDeps,
        output_type=list[str],
        retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent)
    return agent
