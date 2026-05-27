"""Agent factories + tool registrars for the on-the-fly KB-encode adapter.

Three pieces sit alongside the recursive-adapter shape:

* `_parse_kb_row_from_memory` + `_ensure_kb_rows_loaded` — pull KB row
  dicts out of the per-task `YAMLStorage`'s `<db>_kb_<n>` memory bodies
  on demand. Codex finding 6: dependency edges come from each memory's
  surviving `entities` refs (already filtered by DEV-1455), NOT from
  the raw `children_knowledge` field in the YAML body.

* `_walk_children` + `_topo_sort` — deterministic graph helpers used by
  `_register_kb_to_slayer`. Cycles short-circuit (Codex finding 5).

* `_build_kb_encoder` + `_run_kb_encoder` + `_register_kb_to_slayer` —
  the new encoder agent + the sub-clarifier tool that orchestrates
  topo-sorted encoder dispatch, dedup, verification, and per-id error
  surfacing.

The four legacy agent factories (root / sub / projection-resolver /
constructor) are local copies of the recursive-adapter implementations
rebound onto this package's `TaskDeps` / `AgentRecord` types, per the
plan's sibling-not-subclass decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import replace
from typing import Any, Awaitable, Callable

from pydantic import ValidationError
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
from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
    AgentRecord,
    EncodedEntity,
    EncoderResult,
    SharedTaskState,
    TaskDeps,
    _LegacyAdapter,
)
from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
    KB_ENCODER_ONESHOT_PROMPT,
    KB_ENCODER_PROMPT,
    SUB_CLARIFIER_PROMPT,
    SUB_EXPLORER_PROMPT,
)
from bird_interact_agents.harness import MAX_MODEL_TURNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prepare_tools shim — identical to the recursive adapter's.
# ---------------------------------------------------------------------------


def _make_prepare_tools(strict_value: bool):
    """Force a uniform `strict` on every tool definition right before
    each model request. See the recursive adapter for the rationale."""

    async def _force_strict(
        ctx: RunContext, tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        return [replace(td, strict=strict_value) for td in tool_defs]

    return _force_strict


# ---------------------------------------------------------------------------
# SLayer client lazy init — identical to the recursive adapter.
# ---------------------------------------------------------------------------


def _slayer_client_factory(adapter: _LegacyAdapter):
    if adapter._slayer_client is None:
        from slayer.client.slayer_client import SlayerClient
        from slayer.storage.yaml_storage import YAMLStorage

        storage = YAMLStorage(base_dir=adapter.slayer_storage_dir)
        adapter._slayer_storage = storage
        adapter._slayer_client = SlayerClient(storage=storage)
    return adapter._slayer_client


# ---------------------------------------------------------------------------
# Tool registrars shared with the recursive adapter (ask_user, submit_query,
# spawn_subagent). Re-implemented here to bind to THIS module's TaskDeps.
# ---------------------------------------------------------------------------


def _register_ask_user(agent: Agent) -> None:
    @agent.tool
    async def ask_user(ctx: RunContext[TaskDeps], question: str) -> str:
        """Ask the user a clarification question about their query."""
        return await ask_user_impl(_LegacyAdapter(ctx.deps), question, "slayer")


_WRAPPED_KEYS = ("queries", "nested_queries")


def _projection_count(parsed: Any) -> int | None:
    """Count of output columns in a parsed SLayer submission, or None
    when the shape can't be counted defensively. Identical contract to
    the recursive adapter's same-named helper."""
    if isinstance(parsed, list):
        if not parsed:
            return None
        last = parsed[-1]
        if not isinstance(last, dict):
            return None
        parsed = last
    if not isinstance(parsed, dict):
        return None
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
    expected_count = len(confirmed_projection)
    confirmed_list = list(confirmed_projection)
    confirmed_block = "\n".join(
        f"  {i + 1}. {name}" for i, name in enumerate(confirmed_list)
    )

    @agent.tool
    async def submit_query(ctx: RunContext[TaskDeps], query_json: str) -> str:
        """Submit your final SLayer query for evaluation. The submission
        must project EXACTLY the columns the projection-resolver
        confirmed. A count mismatch is hard-rejected before the helper
        is called — no budget is charged on rejection."""
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


def _require_submission(attr: str):
    """Output-validator factory (DEV-1454): until ``ctx.deps.<attr>`` is set
    (the agent has called its submit_* tool), reject any final text response
    with ``ModelRetry``. This is what lets the agent reason in free text between
    tool calls (no structured ``output_type``, so ``tool_choice='auto'``)
    WITHOUT a bare-text reply ending the run before the result is delivered.
    Note: a submitted empty list is "set" (not None), so it passes the gate."""

    async def _validate(ctx: RunContext[Any], output: Any) -> Any:
        if getattr(ctx.deps, attr, None) is None:
            raise ModelRetry(
                "You have not delivered your result yet. Call the submit tool "
                "with your final result, then reply briefly to finish."
            )
        return output

    return _validate


def _register_submit_encoding(agent: Agent) -> None:
    """Register ``submit_encoding`` (captures the EncoderResult into per-run
    deps) plus the must-submit gate. Mirrors ``submit_query``: the encoder
    reasons in text, then calls this once with its final EncoderResult JSON.
    Works for both the task-time ``kb_encoder`` (deps=TaskDeps) and the
    build-time ``setup_encoder`` (deps=EncoderCaptureDeps) — both expose
    ``.encoder_submission``."""

    @agent.tool
    async def submit_encoding(ctx: RunContext[Any], result_json: str) -> str:
        """Submit your final result for this KB item as a JSON EncoderResult:
        {"kb_id": <int>, "status": "encoded"|"deferred", "entities": [...],
         "notes": "...", "clarifying_questions": [...]}. Reason first, then call
        this exactly once when done; reply briefly afterwards to finish."""
        try:
            result = EncoderResult.model_validate(json.loads(result_json))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ModelRetry(
                f"submit_encoding could not parse an EncoderResult: {exc}. "
                "Fix the JSON and call submit_encoding again."
            )
        ctx.deps.encoder_submission = result
        return (
            f"Recorded EncoderResult for KB {result.kb_id} "
            f"(status={result.status}). Reply 'done' to finish."
        )

    agent.output_validator(_require_submission("encoder_submission"))


def _register_submit_projection(agent: Agent) -> None:
    """Register ``submit_projection`` (captures the confirmed output-column list
    into per-run deps) plus the must-submit gate. An empty array is a valid
    submission (it triggers the resolver's recovery pass)."""

    @agent.tool
    async def submit_projection(ctx: RunContext[Any], columns_json: str) -> str:
        """Submit the confirmed output columns as a JSON array of name strings,
        e.g. ["region", "total_revenue"]. An empty array [] is allowed. Reason
        first, then call this once when done; reply briefly afterwards."""
        try:
            columns = json.loads(columns_json)
        except json.JSONDecodeError as exc:
            raise ModelRetry(
                f"submit_projection could not parse JSON: {exc}. "
                "Pass a JSON array of column-name strings."
            )
        if not isinstance(columns, list) or not all(
            isinstance(c, str) for c in columns
        ):
            raise ModelRetry(
                "submit_projection expects a JSON array of column-name strings, "
                'e.g. ["region", "revenue"]; got: ' + columns_json
            )
        ctx.deps.projection_submission = columns
        return (
            f"Recorded projection ({len(columns)} column(s)). "
            "Reply 'done' to finish."
        )

    agent.output_validator(_require_submission("projection_submission"))


def _register_spawn_subagent(
    agent: Agent,
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str,
    eval_mode: str = "a-interact",
) -> None:
    """Identical to the recursive adapter's spawn_subagent, but the
    spawned child is THIS module's `_build_sub_clarifier`
    (a-interact) or :func:`_build_sub_explorer` (one-shot, DEV-1462)."""
    use_explorer = eval_mode == "one-shot"

    @agent.tool(sequential=True)
    async def spawn_subagent(
        ctx: RunContext[TaskDeps], focus: str, instruction: str,
    ) -> str:
        """Spawn a sub-agent focused on ONE logical block of the
        user's question. Sibling spawn calls in one model batch run
        sequentially. The sub-agent returns a description of its slice."""
        deps = ctx.deps
        if deps.depth >= deps.max_depth:
            return (
                f"max_depth={deps.max_depth} reached at depth "
                f"{deps.depth}; answer from what you already know."
            )

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

        child_kwargs: dict[str, Any] = dict(
            model=model, model_settings=model_settings,
            shared_slayer_server=shared_slayer_server,
            self_model_id=self_model_id,
        )
        # Resolve child builder at CALL time (not registration time) so
        # tests that re-patch `_build_sub_clarifier` between building this
        # agent and invoking its spawn still pick up the new stub.
        if use_explorer:
            child_kwargs["eval_mode"] = eval_mode
            child = _build_sub_explorer(**child_kwargs)
        else:
            child = _build_sub_clarifier(**child_kwargs)
        child_deps = TaskDeps(
            shared=deps.shared, depth=deps.depth + 1,
            max_depth=deps.max_depth, self_record_idx=my_child_idx,
        )
        prompt_template = (
            SUB_EXPLORER_PROMPT if use_explorer else SUB_CLARIFIER_PROMPT
        )
        child_prompt = prompt_template.format(
            budget=deps.shared.status.remaining_budget,
            db_name=deps.shared.db_name,
            focus=focus,
            instruction=instruction,
        )

        try:
            run = await child.run(
                instruction, instructions=child_prompt, deps=child_deps,
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

        run_usage = run.usage()
        child_deps.usage.add_call(
            scope="agent", model=self_model_id,
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
# KB-row loader — Codex findings 6 + 7.
# ---------------------------------------------------------------------------


# Memory ids of the form `<db>_kb_<int>` are the only ones the loader
# treats as KB items. Anything else (agent-saved notes) is ignored.
_KB_MEMORY_ID_RE = re.compile(r"^(.+)_kb_(\d+)$")


async def _ensure_kb_rows_loaded(
    shared: SharedTaskState,
) -> dict[int, dict]:
    """Lazy-load the per-task KB map from the SLayer storage's `<db>_kb_<n>`
    memories — straight from the memory API, NO YAML re-parse (DEV-1454).

    Each "row" carries only what the rest of the tool needs: the memory's
    ``learning`` text (fed verbatim to the encoder) and its surviving
    ``entities`` refs (the dependency-edge + setup-encoded-ref source).
    HARD-8-deleted KBs are simply absent from ``list_memories``, so they never
    load and surface as per-id errors when requested.
    """
    if shared._kb_rows_by_id is not None:
        return shared._kb_rows_by_id

    storage = shared._slayer_storage
    if storage is None:
        raise RuntimeError(
            "kb_loader: SharedTaskState._slayer_storage is None; kb_to_slayer "
            "ran before the SLayer storage was initialised."
        )
    memories = await storage.list_memories(entities=[shared.db_name])

    rows: dict[int, dict] = {}
    seen_ids: set[int] = set()
    for mem in memories:
        m_id = getattr(mem, "id", None)
        if not m_id:
            continue
        match = _KB_MEMORY_ID_RE.match(m_id)
        if not match or match.group(1) != shared.db_name:
            continue
        kb_id = int(match.group(2))
        if kb_id in seen_ids:
            raise ValueError(
                f"kb_loader: duplicate KB memory id "
                f"`{shared.db_name}_kb_{kb_id}` — "
                f"data-integrity bug (DEV-1428 upsert should prevent this)"
            )
        seen_ids.add(kb_id)
        rows[kb_id] = {
            "id": kb_id,
            "_memory_entities": list(getattr(mem, "entities", []) or []),
            "_learning": getattr(mem, "learning", "") or "",
        }

    shared._kb_rows_by_id = rows
    return rows


# ---------------------------------------------------------------------------
# Topo sort — Codex finding 5.
# ---------------------------------------------------------------------------


def _entities_by_id(kb_rows_by_id: dict[int, dict]) -> dict[int, list[int]]:
    """Extract dep edges from each row's surviving memory `entities`
    refs (Codex finding 6). Each `memory:<db>_kb_<n>` token in
    `entities` becomes an int edge."""
    edges: dict[int, list[int]] = {}
    for kb_id, row in kb_rows_by_id.items():
        refs: list[int] = []
        for ent in row.get("_memory_entities", []):
            if not isinstance(ent, str):
                continue
            m = re.match(r"^memory:(.+)_kb_(\d+)$", ent)
            if m and m.group(2).isdigit():
                refs.append(int(m.group(2)))
        edges[kb_id] = refs
    return edges


def _walk_children(
    seed: set[int], entities_by_id: dict[int, list[int]],
) -> set[int]:
    """Return seed ∪ all transitive children reachable through
    `entities_by_id`. Children absent from the map are silently
    dropped (defensive shim on top of DEV-1455's filtering)."""
    if not seed:
        return set()
    out: set[int] = set()
    stack = list(seed)
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        if cur not in entities_by_id:
            raise KeyError(
                f"_walk_children: seed/intermediate id {cur} not in "
                f"entities_by_id"
            )
        out.add(cur)
        for child in entities_by_id[cur]:
            if child in entities_by_id and child not in out:
                stack.append(child)
            # else: silent drop (DEV-1455's filter already removed
            # deleted refs; an absent id means schema drift handled
            # upstream).
    return out


def _topo_sort(
    ids: set[int], entities_by_id: dict[int, list[int]],
) -> tuple[list[int], set[int]]:
    """Topologically sort `ids`. Returns `(acyclic_order, cycle_ids)`.

    `acyclic_order` lists deps-before-dependents with ascending-id
    tie-break (deterministic). `cycle_ids` contains every id that
    participates in or transitively depends on a strongly-connected
    component — these must be surfaced as per-kb errors by the
    caller (Codex finding 5).
    """
    if not ids:
        return [], set()

    # Build dep-set restricted to `ids` (edges to outside ids are
    # treated as already-satisfied).
    deps: dict[int, set[int]] = {
        i: {c for c in entities_by_id.get(i, []) if c in ids}
        for i in ids
    }
    # Reverse adj for cycle-affected propagation.
    dependents: dict[int, set[int]] = {i: set() for i in ids}
    for i, d in deps.items():
        for c in d:
            dependents[c].add(i)

    # Kahn's algorithm with ascending-id tie-break.
    ready = sorted(i for i, d in deps.items() if not d)
    order: list[int] = []
    visited: set[int] = set()
    while ready:
        n = ready.pop(0)
        if n in visited:
            continue
        visited.add(n)
        order.append(n)
        # Reduce dependents' deps.
        for dep_holder in dependents[n]:
            deps[dep_holder].discard(n)
            if not deps[dep_holder] and dep_holder not in visited:
                # Insert preserving ascending-id order.
                # Using bisect would be slightly faster; insort/sort
                # is fine at our scale.
                ready.append(dep_holder)
                ready.sort()

    cycle_seed = ids - visited
    if not cycle_seed:
        return order, set()

    # Anything touching the cycle (the SCC itself + transitive
    # dependents that couldn't be encoded) is reported as cyclic.
    cycle_ids = set(cycle_seed)
    # Walk dependents to include downstream-only ids.
    stack = list(cycle_seed)
    while stack:
        cur = stack.pop()
        for d in dependents.get(cur, set()):
            if d not in cycle_ids:
                cycle_ids.add(d)
                stack.append(d)
                # Also remove from order if it slipped in.
                if d in order:
                    order.remove(d)

    logger.warning(
        "kb_to_slayer: dependency cycle detected, affecting kb_ids %s",
        sorted(cycle_ids),
    )
    return order, cycle_ids


# ---------------------------------------------------------------------------
# KB encoder agent
# ---------------------------------------------------------------------------


def _build_kb_encoder(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """Build the encoder sub-agent (a-interact).

    Tool surface (Codex test-review finding 1): `ask_user` + `submit_encoding`
    on the native function-toolset. SLayer MCP write/read tools come via the
    shared MCP toolset (`shared_slayer_server`). NO `submit_query`, NO
    `spawn_subagent`, NO `kb_to_slayer`.

    DEV-1454: NO structured ``output_type`` — the encoder reasons in text and
    delivers its ``EncoderResult`` via ``submit_encoding`` (a structured
    output_type forces tool_choice='any', forbidding any reasoning text)."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps,
        retries=2, prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent)
    _register_submit_encoding(agent)
    return agent


def _build_kb_encoder_oneshot(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """DEV-1462 — one-shot variant of :func:`_build_kb_encoder`.

    Same tool surface MINUS ``ask_user``. Reasons in text and delivers
    its ``EncoderResult`` via ``submit_encoding``; SLayer MCP write/read
    tools come through the shared MCP toolset. The encoder decides every
    KB-encoding choice autonomously — there is no user-sim in the
    one-shot pipeline."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps,
        retries=2, prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_submit_encoding(agent)
    return agent


async def _run_kb_encoder_default(
    *,
    kb_id: int,
    row: dict,
    deps_map: list[EncoderResult],
    ctx: RunContext[TaskDeps],
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str,
    eval_mode: str = "a-interact",
) -> EncoderResult:
    """Default `_encoder_runner` for `_register_kb_to_slayer`. Builds
    the encoder agent, formats its prompt, runs it, captures the
    AgentRecord, and returns the typed result.

    `deps_map` is the list of already-encoded dependencies (their
    EncoderResults). It is formatted into the encoder prompt's
    `deps_block` so the encoder can reference dep entity_refs in
    R-RESOLVE-style formulas instead of re-encoding them."""
    deps_block = await _format_deps_block(
        deps_map,
        getattr(ctx.deps.shared, "_slayer_storage", None),
        ctx.deps.shared.db_name,
    )
    # DEV-1454: feed the memory's `learning` body verbatim (knowledge +
    # verbatim KB block) — no YAML re-parse round-trip.
    # DEV-1462: in one-shot mode the encoder must not be steered toward
    # an ask_user tool it doesn't have, so swap in the one-shot prompt.
    is_one_shot = eval_mode == "one-shot"
    kb_body = row.get("_learning") if isinstance(row, dict) else str(row)
    prompt_template = KB_ENCODER_ONESHOT_PROMPT if is_one_shot else KB_ENCODER_PROMPT
    prompt = prompt_template.format(
        db_name=ctx.deps.shared.db_name,
        kb_id=kb_id,
        kb_row_yaml=kb_body or "",
        deps_block=deps_block,
        budget=ctx.deps.shared.status.remaining_budget,
    )

    # Pre-reserve the kb_encoder record slot so the trajectory's
    # parent_idx survives any completion order.
    parent_record_idx = ctx.deps.self_record_idx
    record = AgentRecord(
        role="kb_encoder",
        depth=ctx.deps.depth + 1,
        parent_idx=parent_record_idx,
        instruction=f"encode kb {kb_id}",
        kb_id=kb_id,
        started_at=time.monotonic(),
    )
    ctx.deps.shared.agent_records.append(record)
    my_idx = len(ctx.deps.shared.agent_records) - 1

    encoder_builder = _build_kb_encoder_oneshot if is_one_shot else _build_kb_encoder
    encoder = encoder_builder(
        model=model, model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
    )
    encoder_deps = TaskDeps(
        shared=ctx.deps.shared,
        depth=ctx.deps.depth + 1,
        max_depth=ctx.deps.max_depth,
        self_record_idx=my_idx,
    )

    # 10-round ask_user cap (one model turn per ask + one per reply).
    # Plan finding #4 + Codex test-review finding 2.
    cap = 10 * 2

    # DEV-1454: the encoder reasons in text and delivers its EncoderResult via
    # the submit_encoding tool into encoder_deps (not pydantic-ai structured
    # output). Capture the run, but read the result from deps so a valid
    # submission survives a post-submit failure (e.g. cap hit chasing the final
    # text) — Codex finding.
    run: Any = None
    err: str | None = None
    try:
        run = await encoder.run(
            user_prompt=f"Encode KB {kb_id} per the recipe book.",
            instructions=prompt,
            deps=encoder_deps,
            usage_limits=UsageLimits(request_limit=cap),
        )
    except Exception as exc:  # noqa: BLE001 — defensive
        logger.exception("kb_encoder failed for kb_id=%s: %s", kb_id, exc)
        err = f"{type(exc).__name__}: {exc}"

    # Record finalisation from whatever the run produced.
    if run is not None:
        run_usage = run.usage()
        encoder_deps.usage.add_call(
            scope="agent", model=self_model_id,
            prompt=getattr(run_usage, "input_tokens", 0) or 0,
            completion=getattr(run_usage, "output_tokens", 0) or 0,
            cache_read=getattr(run_usage, "cache_read_tokens", 0) or 0,
            cache_write=getattr(run_usage, "cache_write_tokens", 0) or 0,
        )
        record.output = str(run.output)
        record.messages = _serialize_messages(run)
        record.tool_call_stats = _extract_tool_stats(run)
        record.n_agent_turns = _count_turns(run)
    if err is not None:
        record.error = err
    record.user_sim_transcript = list(encoder_deps.user_sim_transcript)
    record.usage = encoder_deps.usage
    record.ended_at = time.monotonic()

    # Prefer the submitted EncoderResult (force the kb_id against invention).
    submission = encoder_deps.encoder_submission
    if submission is not None:
        return (
            submission if submission.kb_id == kb_id
            else submission.model_copy(update={"kb_id": kb_id})
        )
    return EncoderResult(
        kb_id=kb_id, status="error", entities=[], notes="",
        error=err or "agent never called submit_encoding",
    )


def _meta_has_kb_id(meta: Any, kb_id: int) -> bool:
    """True iff ``meta`` carries ``kb_id`` (single ``kb_id`` or within ``kb_ids``)."""
    if not meta:
        return False
    try:
        if meta.get("kb_id") is not None and int(meta["kb_id"]) == kb_id:
            return True
        kb_ids = meta.get("kb_ids")
        return bool(kb_ids) and kb_id in {int(x) for x in kb_ids}
    except Exception:  # noqa: BLE001 — malformed meta is simply "not tagged"
        return False


async def _live_tagged_entities(
    storage: Any, db: str, kb_id: int,
) -> list[tuple[str, str]]:
    """Scan ``db``'s models for any entity (the model itself, a column, measure,
    or aggregation) carrying ``meta.kb_id == kb_id``; return ``(entity_ref,
    kind)`` pairs. Best-effort: a storage hiccup yields ``[]`` (the caller then
    falls back to the in-memory status)."""
    out: list[tuple[str, str]] = []
    if storage is None:
        return out
    try:
        names = await storage.list_models(data_source=db)
    except Exception:  # noqa: BLE001
        return out
    for name in names:
        try:
            model = await storage.get_model(name, data_source=db)
        except Exception:  # noqa: BLE001
            continue
        if model is None:
            continue
        if _meta_has_kb_id(getattr(model, "meta", None), kb_id):
            out.append((f"{db}.{name}", "model"))
        for kind, items in (
            ("column", getattr(model, "columns", None)),
            ("measure", getattr(model, "measures", None)),
            ("aggregation", getattr(model, "aggregations", None)),
        ):
            for item in items or []:
                if _meta_has_kb_id(getattr(item, "meta", None), kb_id):
                    out.append((f"{db}.{name}.{item.name}", kind))
    return out


async def _format_deps_block(
    deps_map: list[EncoderResult], storage: Any = None, db: str = "",
) -> str:
    """One line per dep: ``KB <id> -> <entity_ref> (kind=...)``.

    DEV-1454: prefer the LIVE datasource over the (possibly stale or
    concurrently-raced) in-memory ``EncoderResult.status``. A dep whose entity
    physically exists + is tagged is reported as encoded with its real ref even
    if its status says ``deferred`` — this unblocks a dependent KB whose dep was
    falsely downgraded by a verify torn-read race (the KB-13 deadlock)."""
    lines: list[str] = []
    for dep in deps_map:
        live = await _live_tagged_entities(storage, db, dep.kb_id)
        if live:
            for ref, kind in live:
                lines.append(f"  - KB {dep.kb_id} -> {ref} (kind={kind})")
        elif dep.status == "encoded":
            for ent in dep.entities:
                lines.append(
                    f"  - KB {dep.kb_id} -> {ent.entity_ref} (kind={ent.kind})"
                )
        else:
            lines.append(
                f"  - KB {dep.kb_id} -> NOT encoded ({dep.error or dep.status})"
            )
    return "\n".join(lines) if lines else "(none)"


# Trim long parent definitions so a KB with many parents can't blow up the
# prompt; the GUARD only needs enough of the definition to judge ownership.
_REVERSE_DEP_DEF_MAXLEN = 200


def _format_reverse_deps_block(reverse_deps: list[dict] | None) -> str:
    """One line per KB that REFERENCES the current KB (its parents in the KB
    graph), sorted by kb id, deduped by id (DEV-1466). Each line carries the
    parent's ``type`` and ``knowledge`` + a trimmed ``definition`` so the setup
    encoder can decide whether to defer an embedded scoring scheme to a
    ``calculation_knowledge`` parent that owns that score (KB-6 → KB-44) versus
    keep a component score its parent merely averages (KB-3/4/5 → KB-13).

    Pure / synchronous — unlike :func:`_format_deps_block` it needs no storage,
    only the raw KB rows. Tolerant of rows missing ``type`` / ``knowledge`` /
    ``definition`` / a non-int ``id``. Returns ``"(none)"`` when empty."""
    if not reverse_deps:
        return "(none)"
    seen: set[int] = set()
    rows: list[tuple[int, dict]] = []
    for r in reverse_deps:
        rid_val = r.get("id")
        if rid_val is None:
            continue
        try:
            rid = int(rid_val)
        except (TypeError, ValueError):
            continue
        if rid in seen:
            continue
        seen.add(rid)
        rows.append((rid, r))
    if not rows:
        return "(none)"
    rows.sort(key=lambda x: x[0])
    lines: list[str] = []
    for rid, r in rows:
        head = f"  - KB {rid}"
        ktype = r.get("type")
        if ktype:
            head += f" ({ktype})"
        knowledge = r.get("knowledge")
        if knowledge:
            head += f' "{knowledge}"'
        definition = r.get("definition")
        if definition:
            d = str(definition).strip()
            if len(d) > _REVERSE_DEP_DEF_MAXLEN:
                d = d[:_REVERSE_DEP_DEF_MAXLEN].rstrip() + "…"
            head += f": {d}"
        lines.append(head)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-encode verification — Codex finding 3.
# ---------------------------------------------------------------------------


async def _verify_entities_exist(
    result: EncoderResult, storage: Any, db_name: str,
) -> EncoderResult:
    """Verify each claimed entity_ref actually exists in storage.
    Returns the result unchanged on success; downgrades to
    `status='error'` if any ref is missing."""
    if result.status != "encoded" or not result.entities:
        return result

    missing: list[str] = []
    for ent in result.entities:
        ok = await _entity_exists(ent, storage, db_name)
        if not ok:
            missing.append(ent.entity_ref)
    if missing:
        return result.model_copy(update={
            "status": "error",
            "error": (
                f"claimed entities not found in storage: "
                f"{', '.join(missing)}"
            ),
        })
    return result


async def _entity_exists(
    ent: EncodedEntity, storage: Any, db_name: str,
) -> bool:
    """Single-entity existence check via direct storage lookup."""
    try:
        if ent.kind == "model":
            model = await storage.get_model(ent.name)
            return model is not None and model.data_source == db_name
        # column / measure / aggregation live on a host model.
        if ent.host_model is None:
            return False
        model = await storage.get_model(ent.host_model)
        if model is None or model.data_source != db_name:
            return False
        if ent.kind == "column":
            return any(c.name == ent.name for c in (model.columns or []))
        if ent.kind == "measure":
            return any(m.name == ent.name for m in (model.measures or []))
        if ent.kind == "aggregation":
            return any(
                a.name == ent.name for a in (model.aggregations or [])
            )
    except Exception:  # noqa: BLE001 — storage may raise on missing
        return False
    return False


# ---------------------------------------------------------------------------
# kb_to_slayer tool — Codex finding 5/6/8 + test seam from plan.
# ---------------------------------------------------------------------------


_EncoderRunner = Callable[..., Awaitable[EncoderResult]]


def _register_kb_to_slayer(
    agent: Agent,
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str,
    eval_mode: str = "a-interact",
    _encoder_runner: _EncoderRunner | None = None,
) -> None:
    """Register `kb_to_slayer(kb_ids: list[int])` on `agent`.

    `_encoder_runner` is a testability seam (underscore-prefixed) —
    tests inject a stub that returns canned EncoderResults without
    spinning up a real pydantic-ai Agent. Production callers omit
    this kwarg and get `_run_kb_encoder_default`.

    DEV-1462: ``eval_mode`` selects the per-KB encoder. In ``one-shot``
    mode the dispatched encoder is the one-shot variant (no ask_user)
    so the kb_to_slayer call chain stays free of user-sim turns.
    """

    async def _default_runner(*, kb_id, row, deps_map, ctx):
        return await _run_kb_encoder_default(
            kb_id=kb_id, row=row, deps_map=deps_map, ctx=ctx,
            model=model, model_settings=model_settings,
            shared_slayer_server=shared_slayer_server,
            self_model_id=self_model_id,
            eval_mode=eval_mode,
        )

    runner: _EncoderRunner = _encoder_runner or _default_runner

    @agent.tool(sequential=True)
    async def kb_to_slayer(
        ctx: RunContext[TaskDeps], kb_ids: list[int],
    ) -> str:
        """Elevate one or more KB items into first-class SLayer
        entities. Pass the integer ids you discovered via `search`
        and confirmed relevant with the user. Returns a JSON map
        keyed by kb_id; each value carries `status`, `entities`
        (list of `{kind, host_model, name, entity_ref}`), and
        `notes`. Use the returned `entity_ref` names in your slice
        output so the constructor can compose them into the final
        query."""
        if not kb_ids:
            return "{}"

        requested = set(kb_ids)
        shared = ctx.deps.shared

        # Lazy-load KB rows (idempotent).
        kb_rows = await _ensure_kb_rows_loaded(shared)
        edges = _entities_by_id(kb_rows)

        # Compute the union of requested ids + their transitive deps.
        # Ids that failed to load OR are unknown surface as per-id
        # errors below; we walk only the ones we can.
        walkable_seed: set[int] = set()
        unknown_or_failed: dict[int, str] = {}
        for kb_id in requested:
            if kb_id in kb_rows:
                walkable_seed.add(kb_id)
            elif kb_id in shared._kb_load_failures:
                unknown_or_failed[kb_id] = (
                    f"malformed KB memory: {shared._kb_load_failures[kb_id]}"
                )
            else:
                unknown_or_failed[kb_id] = (
                    f"unknown kb_id: no memory `{shared.db_name}_kb_{kb_id}`"
                )

        full_set = _walk_children(walkable_seed, edges)
        order, cycle_ids = _topo_sort(full_set, edges)

        # Short-circuit cycle members: per-id error, no encoder run.
        for kb_id in cycle_ids:
            if not _find_cached(shared.kb_encoded, kb_id):
                shared.kb_encoded.append(EncoderResult(
                    kb_id=kb_id, status="error",
                    entities=[], notes="",
                    error=f"dependency cycle: kb_id {kb_id} participates "
                          f"in cycle {sorted(cycle_ids)}",
                ))

        # Surface load-failure ids as per-id errors too.
        for kb_id, reason in unknown_or_failed.items():
            if not _find_cached(shared.kb_encoded, kb_id):
                shared.kb_encoded.append(EncoderResult(
                    kb_id=kb_id, status="error",
                    entities=[], notes="", error=reason,
                ))

        # Encode in topo order, with per-kb lock + dedup.
        for kb_id in order:
            cached = _find_cached(shared.kb_encoded, kb_id)
            if cached is not None:
                continue

            lock = shared._kb_locks.setdefault(kb_id, asyncio.Lock())
            async with lock:
                # Double-check under the lock.
                cached = _find_cached(shared.kb_encoded, kb_id)
                if cached is not None:
                    continue

                row = kb_rows[kb_id]

                # Short-circuit ids the setup pass already encoded: if the
                # memory records concrete entity refs and they ALL still exist
                # in the task storage, reuse them — don't re-run the encoder.
                storage = shared._slayer_storage
                existing = (
                    await _find_existing_entities_for_kb(
                        storage, shared.db_name,
                        row.get("_memory_entities", []),
                    )
                    if storage is not None else None
                )
                if existing is not None:
                    shared.kb_encoded.append(EncoderResult(
                        kb_id=kb_id, status="encoded", entities=existing,
                        notes="from setup reference",
                    ))
                    continue

                # Build deps_map from already-cached results.
                child_ids = [c for c in edges.get(kb_id, []) if c in kb_rows]
                deps_map = [
                    _find_cached(shared.kb_encoded, c)
                    for c in child_ids
                ]
                deps_map = [d for d in deps_map if d is not None]

                result = await runner(
                    kb_id=kb_id, row=row, deps_map=deps_map, ctx=ctx,
                )
                # Post-encode verification (Codex finding 3).
                storage = shared._slayer_storage
                if storage is not None:
                    result = await _verify_entities_exist(
                        result, storage, shared.db_name,
                    )
                shared.kb_encoded.append(result)

        # Return only requested ids, in stable order.
        payload: dict[str, dict] = {}
        for kb_id in kb_ids:
            cached = _find_cached(shared.kb_encoded, kb_id)
            if cached is not None:
                payload[str(kb_id)] = cached.model_dump()
        return json.dumps(payload, indent=2, default=str)


def _find_cached(
    cache: list[EncoderResult], kb_id: int,
) -> EncoderResult | None:
    for r in cache:
        if r.kb_id == kb_id:
            return r
    return None


# ---------------------------------------------------------------------------
# Setup-encoded short-circuit (DEV-1454).
# ---------------------------------------------------------------------------


def _setup_encoded_refs(memory_entities: list, db: str) -> list[str]:
    """Concrete entity refs the setup pass recorded on a KB's own memory:
    ``<db>.<...>`` entries (not the bare datasource, not ``memory:`` refs)."""
    return [
        ent for ent in (memory_entities or [])
        if isinstance(ent, str) and ent.startswith(f"{db}.")
    ]


async def _resolve_ref(storage: Any, db: str, ref: str) -> EncodedEntity | None:
    """Resolve a ``<db>.<model>[.<leaf>]`` ref to an EncodedEntity iff it
    exists in storage, else None."""
    parts = ref.split(".")
    try:
        if len(parts) == 2:
            model = await storage.get_model(parts[1])
            if model is not None and getattr(model, "data_source", None) == db:
                return EncodedEntity(
                    kind="model", host_model=None, name=parts[1], entity_ref=ref,
                )
            return None
        if len(parts) >= 3:
            model_name, leaf = parts[1], parts[2]
            model = await storage.get_model(model_name)
            if model is None or getattr(model, "data_source", None) != db:
                return None
            for kind, bucket in (
                ("column", model.columns),
                ("measure", model.measures),
                ("aggregation", model.aggregations),
            ):
                for item in (bucket or []):
                    if item.name == leaf:
                        return EncodedEntity(
                            kind=kind, host_model=model_name, name=leaf,
                            entity_ref=ref,
                        )
    except Exception:  # noqa: BLE001 — storage may raise on a missing model
        return None
    return None


async def _find_existing_entities_for_kb(
    storage: Any, db: str, memory_entities: list,
) -> list[EncodedEntity] | None:
    """If a KB's memory records concrete entity refs (setup pass) AND ALL of
    them still exist in storage, return them so ``kb_to_slayer`` can reuse the
    setup encode without re-running the encoder. Returns None when there are
    no recorded refs or the recorded set is incomplete (⇒ re-encode)."""
    refs = _setup_encoded_refs(memory_entities, db)
    if not refs:
        return None
    out: list[EncodedEntity] = []
    for ref in refs:
        ent = await _resolve_ref(storage, db, ref)
        if ent is None:
            return None
        out.append(ent)
    return out


# ---------------------------------------------------------------------------
# Agent factories — root / sub / projection-resolver / constructor.
# ---------------------------------------------------------------------------


def _build_root_clarifier(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    max_depth: int,
    self_model_id: str = "unknown",
    eval_mode: str = "a-interact",
) -> Agent:
    """Root clarifier: spawn_subagent only. NO SLayer toolset,
    NO ask_user, NO submit_query, NO kb_to_slayer.

    Identical contract to the recursive adapter's root_clarifier.
    DEV-1462: ``eval_mode`` selects sub-explorer vs sub-clarifier at
    spawn time (same as the recursive adapter)."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_spawn_subagent(
        agent, model=model, model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
        eval_mode=eval_mode,
    )
    return agent


def _build_sub_clarifier(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """Sub-clarifier: ask_user + spawn_subagent + kb_to_slayer.

    `kb_to_slayer` is registered here (and NOT on root or the
    constructor) per the plan."""
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
        agent, model=model, model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
    )
    _register_kb_to_slayer(
        agent, model=model, model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
    )
    return agent


# ---------------------------------------------------------------------------
# DEV-1462: one-shot factories — `ask_user`-free flavors. The
# orchestration shape is identical to a-interact (root → sub-explorer
# tree with kb_to_slayer → projection-resolver → query-constructor)
# but every role drops ask_user. The constructor's closure-bound count
# check survives in the one-shot variant; only its ModelRetry text is
# rewritten to drop the "call ask_user" suggestion.
# ---------------------------------------------------------------------------


def _build_sub_explorer(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str = "unknown",
    eval_mode: str = "one-shot",
) -> Agent:
    """Sub-explorer (one-shot): SLayer MCP toolset + spawn_subagent +
    kb_to_slayer. NO ask_user.

    Mirrors the a-interact :func:`_build_sub_clarifier` MINUS ask_user.
    The kb_to_slayer wrapper is dispatched in one-shot mode so its
    spawned encoder is also ask_user-free."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_spawn_subagent(
        agent, model=model, model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
        eval_mode=eval_mode,
    )
    _register_kb_to_slayer(
        agent, model=model, model_settings=model_settings,
        shared_slayer_server=shared_slayer_server,
        self_model_id=self_model_id,
        eval_mode=eval_mode,
    )
    return agent


def _build_projection_resolver_oneshot(
    *,
    model: Any,
    model_settings: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """Projection-resolver (Stage 2, one-shot): ``submit_projection``
    only. NO ask_user, NO query, NO MCP toolset.

    Like the a-interact variant, the resolver reasons in text and
    delivers its confirmed column list via ``submit_projection`` (a
    structured output_type would forbid the reasoning step). The
    ``_require_submission`` gate fires until the agent has actually
    called the tool."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps,
        retries=2, prepare_tools=_make_prepare_tools(False),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_submit_projection(agent)
    return agent


def _build_query_constructor_oneshot(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    confirmed_projection: tuple[str, ...],
    self_model_id: str = "unknown",
) -> Agent:
    """Query-constructor (Stage 3, one-shot): SLayer MCP toolset +
    ``submit_query``. NO ask_user.

    Mirrors the a-interact constructor (incl. the
    validate-before-persist hook that lives on the shared MCP server
    layer, not the native function-toolset) but drops ask_user. The
    closure-bound count check on submit_query is preserved with a
    one-shot ModelRetry text (no "call ask_user")."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_submit_query_oneshot(agent, confirmed_projection)
    return agent


def _register_submit_query_oneshot(
    agent: Agent, confirmed_projection: tuple[str, ...],
) -> None:
    """One-shot variant of :func:`_register_submit_query` — same
    closure-bound count check, NO ask_user reference in the ModelRetry
    text (the one-shot constructor has no ask_user tool)."""
    expected_count = len(confirmed_projection)
    confirmed_list = list(confirmed_projection)
    confirmed_block = "\n".join(
        f"  {i + 1}. {name}" for i, name in enumerate(confirmed_list)
    )

    @agent.tool
    async def submit_query(ctx: RunContext[TaskDeps], query_json: str) -> str:
        """Submit your final SLayer query for evaluation. The submission
        must project EXACTLY the columns the projection-resolver
        confirmed. A count mismatch is hard-rejected before the helper
        is called — no budget is charged on rejection. Align your draft
        with the confirmed list and resubmit."""
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
                f"order, then resubmit. This rejection consumed no "
                f"budget."
            )

        adapter = _LegacyAdapter(ctx.deps)
        return submit_slayer_query(
            adapter, query_json, _slayer_client_factory,
        )


def _build_query_constructor(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    confirmed_projection: tuple[str, ...],
    self_model_id: str = "unknown",
) -> Agent:
    """Query-constructor: ask_user + submit_query. NO spawn_subagent,
    NO kb_to_slayer. Identical contract to the recursive adapter."""
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
    """Projection-resolver (Stage 2): ask_user + submit_projection.

    DEV-1454: NO structured ``output_type=list[str]`` — the resolver reasons in
    text and delivers its confirmed column list via ``submit_projection`` (a
    structured output_type forces tool_choice='any', forbidding reasoning)."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps,
        retries=2, prepare_tools=_make_prepare_tools(False),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent)
    _register_submit_projection(agent)
    return agent
