"""Public class + run_task entry point for the recursive adapter.

``PydanticAIRecursiveAgent`` is the SLayer-only / a-interact-only
adapter selectable via ``--framework pydantic_ai_recursive`` in the
benchmark runner. Other query/eval modes raise ``ValueError``.

Internals:

* One ``MCPServerStdio`` per task, shared across root → every sub-agent
  → query-constructor. Slayer MCP startup is up to 300s with
  ``--ingest-on-startup``; per-agent spawn would dominate wall time.
  pydantic-ai's underlying ClientSession is multiplexed by request_id
  so concurrent tool calls on one shared session are safe.
* Constructor budget reservation: snapshot total budget, decrement
  ``status.remaining_budget`` by ``CONSTRUCTOR_RESERVE`` before the
  clarifier phase. After the clarifier phase returns, restore so the
  constructor sees at least the reserve plus any clarifier underspend.
* Per-agent records pre-reserved synchronously inside spawn — so
  ``parent_idx`` topology survives any completion order.
* On any top-level exception in root or constructor, the partial
  trajectory (whichever ``AgentRecord``s already completed) is
  preserved in the result row.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits

from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.agents._run_capture import (
    _count_turns,
    _extract_tool_stats,
    _serialize_messages,
)
from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
    AgentRecord,
    SharedTaskState,
    TaskDeps,
)
from bird_interact_agents.agents.pydantic_ai_recursive.factories import (
    _build_projection_resolver,
    _build_projection_resolver_oneshot,
    _build_query_constructor,
    _build_query_constructor_oneshot,
    _build_root_clarifier,
)
from bird_interact_agents.agents.pydantic_ai_recursive.prompts import (
    PROJECTION_RESOLVER_ONESHOT_PROMPT,
    PROJECTION_RESOLVER_PROMPT,
    QUERY_CONSTRUCTOR_ONESHOT_PROMPT,
    QUERY_CONSTRUCTOR_PROMPT,
    ROOT_CLARIFIER_PROMPT,
    ROOT_EXPLORER_PROMPT,
)
from bird_interact_agents.harness import (
    ACTION_COSTS,
    MAX_MODEL_TURNS,
    SampleStatus,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
    resolve_task_storage_dir,
    slayer_mcp_stdio_config,
)
from bird_interact_agents.hard8_preprocessor import extract_deleted_kb_ids
from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_tool_filters,
)
from bird_interact_agents.usage import TokenUsage
from bird_interact_agents.slayer_otf import (
    ensure_db_cache,
    prepare_task_storage,
)
from bird_interact_agents import paths as _paths

logger = logging.getLogger(__name__)


# Cap on the number of error-sample blobs persisted per task across the
# whole spawn tree. Matches the per-task budget used by the existing
# pydantic_ai adapter.
_TOOL_ERROR_SAMPLES_PER_TASK = 10


def _constructor_reserve(eval_mode: str = "a-interact") -> float:
    """Bird-coin reserve held back from the clarifier phase so the
    constructor's mandatory tool calls aren't rejected by
    ``gate_or_none``.

    * ``a-interact``: ``2 * ask_user + submit_query`` — the constructor
      may call ask_user twice for projection-mismatch surfacing before
      its final submit.
    * ``one-shot`` (DEV-1462): ``submit_query`` only — there is no
      ask_user anywhere in the spawn tree.

    ``help``, ``query``, and the rest of the SLayer MCP tool surface do
    not flow through ``update_budget`` in this adapter (they are not
    wrapped natively), so they cost nothing against the bird-coin pool
    — only ``ask_user`` (a-interact only) and ``submit_query`` decrement
    it.
    """
    if eval_mode == "one-shot":
        return ACTION_COSTS["submit_query"]
    return (
        2 * ACTION_COSTS["ask_user"]
        + ACTION_COSTS["submit_query"]
    )


def _build_shared_slayer_server(slayer_storage_dir: str) -> MCPServerStdio:
    """One MCPServerStdio per task, shared across the spawn tree.

    DEV-1478: a thin ``process_tool_call`` hook normalizes text-equality FILTER
    predicates (lower(trim) + lowercased literal) on the agent's exploratory
    ``query``/``query_nested`` calls, then forwards. Keeps the hand-audited
    (pre-encoded) eval path's case/whitespace handling identical to the
    OTF-encode path so the comparison stays apples-to-apples. No validator here
    (this adapter doesn't write models)."""
    cfg = slayer_mcp_stdio_config(slayer_storage_dir)

    async def _process_tool_call(ctx, call_tool, name, tool_args):
        return await call_tool(name, normalize_tool_filters(name, tool_args), None)

    return MCPServerStdio(
        command=cfg["command"], args=cfg["args"], env=cfg["env"],
        max_retries=100,
        timeout=300,
        process_tool_call=_process_tool_call,
    )


def _otf_work_dir(instance_id: str) -> Path:
    """Per-task scratch dir for the on-the-fly setup mode.

    Mirrors the layout of the HARD-8 variant dir
    (``$TMPDIR/bird_interact_w5_variants/<instance_id>/``) but under a
    distinct prefix so the two modes don't accidentally share scratch
    state.
    """
    p = (
        Path(tempfile.gettempdir())
        / "bird_interact_slayer_otf"
        / instance_id
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _resolve_otf_task_storage_dir(
    *,
    db_name: str,
    task_data: dict,
    data_path_base: str,
    benchmark: str,
) -> tuple[str, list[int]]:
    """On-the-fly equivalent of ``resolve_task_storage_dir``.

    Materialises the per-DB orchestrator cache (idempotent) and copies
    it into a per-task scratch dir with freshly-encoded KB memories.

    The mini-interact root is derived from ``data_path_base`` (i.e.
    the harness's ``--db-path``) — Codex flagged that hardcoding
    ``paths.mini_interact_root()`` would silently ignore an overridden
    DB path.

    DEV-1462: ``benchmark`` selects the per-benchmark scoped cache root
    so a LiveSQLBench task's cache lands at
    ``slayer_otf_cache_livesqlbench/<db>/`` instead of colliding with
    a same-named mini-interact DB at ``slayer_otf_cache/<db>/``.
    ``benchmark`` is REQUIRED; ``"mini_interact"`` keeps the legacy root.
    """
    deleted = sorted(extract_deleted_kb_ids(task_data))
    instance_id = task_data["instance_id"]
    # ``.resolve()`` is load-bearing: ``_phase1_ingest`` formats the
    # sqlite path into a 4-slash absolute URL (``sqlite:////<path>``),
    # so a relative ``--db-path mini-interact`` would otherwise become
    # rooted at ``/mini-interact/...`` and either fail or ingest the
    # wrong file (Codex finding on PR #19).
    mini_interact_root = Path(data_path_base).resolve()
    cache_entry = await ensure_db_cache(
        db_name,
        cache_root=_paths.slayer_otf_cache_root(benchmark=benchmark),
        mini_interact_root=mini_interact_root,
    )
    scratch = await prepare_task_storage(
        db=db_name,
        deleted_kb_ids=set(deleted),
        cache_entry=cache_entry,
        work_dir=_otf_work_dir(instance_id),
        mini_interact_root=mini_interact_root,
        # DEV-1462: pass the resolved --db-path as the authoritative
        # db_root so it overrides $BIRD_DB_PATH when re-anchoring the
        # per-task datasource. Without this, a LiveSQLBench DB whose name
        # collides with a mini-interact DB (e.g. `alien`) would silently
        # re-anchor to the mini-interact sqlite ($BIRD_DB_PATH default).
        # Mirrors the otf_encode adapter's _resolve_otf_task_storage_dir.
        db_root=mini_interact_root,
    )
    return str(scratch), deleted


def _merge_tool_stats(parts: list[dict | None]) -> dict | None:
    """Merge per-agent tool_call_stats into one summary. Sums n_calls /
    n_errors per tool name; concatenates error_samples up to the
    per-task cap; sums totals. Returns None if no part is populated."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    per_tool_map: dict[str, dict[str, Any]] = {}
    error_samples: list[dict[str, str]] = []
    total_calls = 0
    total_errors = 0
    for p in parts:
        for entry in p.get("per_tool", []):
            name = entry["tool"]
            agg = per_tool_map.setdefault(
                name, {"tool": name, "n_calls": 0, "n_errors": 0},
            )
            agg["n_calls"] += entry.get("n_calls", 0)
            agg["n_errors"] += entry.get("n_errors", 0)
        total_calls += p.get("total_calls", 0)
        total_errors += p.get("total_errors", 0)
        for sample in p.get("error_samples", []):
            if len(error_samples) >= _TOOL_ERROR_SAMPLES_PER_TASK:
                break
            error_samples.append(sample)
    per_tool = sorted(
        per_tool_map.values(),
        key=lambda x: (-x["n_calls"], x["tool"]),
    )
    return {
        "per_tool": per_tool,
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_samples": error_samples,
    }


class PydanticAIRecursiveAgent:
    """SLayer a-interact-only adapter with a recursive clarifier tree
    and a separate query-constructor agent."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        max_depth: int = 3,
        prompt_cache: bool = True,
        slayer_setup: str = "pre-encoded",
    ) -> None:
        # Reuse the existing pydantic_ai adapter's model-construction
        # helpers verbatim — every model/provider quirk is identical.
        from bird_interact_agents.agents.pydantic_ai.agent import (
            _anthropic_cache_settings,
            _build_anthropic_model_with_retries,
        )
        from bird_interact_agents.model_string import (
            build_pydantic_ai_model,
            is_anthropic,
            native_model_id,
        )

        if slayer_setup not in ("pre-encoded", "on-the-fly"):
            raise ValueError(
                f"slayer_setup must be 'pre-encoded' or 'on-the-fly'; "
                f"got {slayer_setup!r}"
            )

        self.slayer_storage_root = slayer_storage_root
        self.model_id = model
        self.slayer_setup = slayer_setup
        anthropic_model = (
            _build_anthropic_model_with_retries(native_model_id(model))
            if is_anthropic(model) else None
        )
        self.model = anthropic_model or build_pydantic_ai_model(model)
        self.max_depth = max_depth
        self._model_settings = (
            _anthropic_cache_settings()
            if (prompt_cache and is_anthropic(model)) else None
        )

    async def run_task(
        self,
        task_data: dict,
        data_path_base: str,
        budget: float,
        query_mode: str,
        eval_mode: str = "a-interact",
        user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version: str = "v2",
    ) -> dict:
        if query_mode != "slayer":
            raise ValueError(
                "pydantic_ai_recursive supports only --query-mode slayer; "
                f"got {query_mode!r}"
            )
        if eval_mode not in ("a-interact", "one-shot"):
            raise ValueError(
                "pydantic_ai_recursive supports only --mode a-interact "
                f"or --mode one-shot; got {eval_mode!r}"
            )

        is_one_shot = eval_mode == "one-shot"
        # DEV-1462 — Codex #1 programmatic-bypass close: one-shot REQUIRES
        # the loader-stamped ``dataset='livesqlbench'`` marker. A caller
        # that bypasses ``load_livesqlbench_tasks`` (cloud actor, custom
        # driver) MUST NOT silently get a one-shot run on un-marked data.
        if is_one_shot and not get_benchmark(
            task_data.get("dataset") or "mini_interact"
        ).one_shot:
            raise ValueError(
                "--mode one-shot requires a task whose benchmark declares "
                "one_shot=True (its loader stamps task_data['dataset']); got "
                f"dataset={task_data.get('dataset')!r}",
            )
        # DEV-1462 — CodeRabbit close: ``_validate_slayer_setup`` rejects
        # one-shot + pre-encoded at the CLI / ``run_evaluation`` /
        # ``make_runner`` / ``run_one_task`` boundaries, but a caller that
        # constructs ``PydanticAIRecursiveAgent(slayer_setup="pre-encoded")``
        # directly and invokes ``run_task(eval_mode="one-shot", ...)`` would
        # otherwise route LiveSQLBench through the legacy pre-encoded
        # ``slayer_models/`` path. Belt-and-suspenders defensive check.
        if is_one_shot and self.slayer_setup != "on-the-fly":
            raise ValueError(
                "--mode one-shot requires slayer_setup='on-the-fly'; "
                f"got {self.slayer_setup!r}",
            )

        db_name = task_data["selected_database"]
        instance_id = task_data["instance_id"]
        benchmark: str = get_benchmark(
            task_data.get("dataset") or "mini_interact"
        ).name

        load_db_data_if_needed(db_name, data_path_base)
        # DEV-1462 B0: LiveSQLBench tasks get a per-task isolated
        # `db_file_path`, so the upstream `reset_and_restore_database`
        # never touches the stable dataset `<db>.sqlite` that the OTF
        # cache reads. No-op for any non-livesqlbench task (mini-interact
        # path is unchanged).
        materialize_task_db(task_data, data_path_base)

        status = SampleStatus(
            idx=0,
            original_data=task_data,
            remaining_budget=budget,
            total_budget=budget,
        )

        if self.slayer_setup == "on-the-fly":
            slayer_storage_dir, deleted_kb_ids = (
                await _resolve_otf_task_storage_dir(
                    db_name=db_name,
                    task_data=task_data,
                    data_path_base=data_path_base,
                    benchmark=benchmark,
                )
            )
        else:
            slayer_storage_dir, deleted_kb_ids = await resolve_task_storage_dir(
                slayer_storage_root=self.slayer_storage_root,
                db_name=db_name,
                task_data=task_data,
                query_mode=query_mode,
            )

        shared = SharedTaskState(
            status=status,
            data_path_base=data_path_base,
            db_name=db_name,
            amb_user_query=task_data["amb_user_query"],
            slayer_storage_dir=slayer_storage_dir,
            user_sim_model=user_sim_model,
            user_sim_prompt_version=user_sim_prompt_version,
        )

        # Constructor budget reservation. Decrement the pool BEFORE the
        # clarifier phase so the sub-tree can't drain budget below the
        # constructor's mandatory tool costs. One-shot reserve = submit_query
        # only (no ask_user anywhere in the spawn tree).
        reserve = _constructor_reserve(eval_mode)
        total_budget = status.remaining_budget
        status.remaining_budget = max(
            0.0, status.remaining_budget - reserve,
        )

        slayer_server = (
            _build_shared_slayer_server(slayer_storage_dir)
            if slayer_storage_dir else None
        )

        # Pre-reserve the root's record slot synchronously so any spawn
        # inside the root sees a stable index in shared.agent_records.
        root_record = AgentRecord(
            role="root_clarifier",
            depth=0,
            parent_idx=None,
            instruction=task_data["amb_user_query"],
            started_at=time.monotonic(),
        )
        shared.agent_records.append(root_record)
        root_idx = len(shared.agent_records) - 1
        root_deps = TaskDeps(
            shared=shared, depth=0, max_depth=self.max_depth,
            self_record_idx=root_idx,
        )

        # Track the currently-running top-level record + its deps so the
        # except path can stamp the AgentRecord with the error rather
        # than leaving error=None, ended_at=0 (Codex finding #3).
        current_record: AgentRecord = root_record
        current_deps: TaskDeps = root_deps

        try:
            async with (slayer_server if slayer_server is not None
                        else _null_async_context()):
                # ----- ROOT PHASE -----
                # One-shot threads ``eval_mode`` into the root's
                # ``spawn_subagent`` so it builds sub-explorers (no
                # ask_user), not sub-clarifiers.
                root_agent = _build_root_clarifier(
                    model=self.model,
                    model_settings=self._model_settings,
                    shared_slayer_server=slayer_server,
                    max_depth=self.max_depth,
                    self_model_id=self.model_id,
                    eval_mode=eval_mode,
                )
                root_template = (
                    ROOT_EXPLORER_PROMPT if is_one_shot
                    else ROOT_CLARIFIER_PROMPT
                )
                root_prompt = root_template.format(
                    budget=shared.status.remaining_budget,
                    db_name=db_name,
                    user_query=task_data["amb_user_query"],
                )
                root_run = await root_agent.run(
                    user_prompt=task_data["amb_user_query"],
                    instructions=root_prompt,
                    deps=root_deps,
                    usage_limits=UsageLimits(
                        request_limit=MAX_MODEL_TURNS * 2,
                    ),
                )
                _fill_record_from_run(
                    root_record, root_run, root_deps, self.model_id,
                )
                spec = str(root_run.output) or task_data["amb_user_query"]

                # Restore budget for the constructor — both the numeric
                # pool AND the force_submit flag. A clarifier ask_user
                # that ran against the post-reserve budget may have
                # tripped `force_submit=True` when remaining dipped to
                # the submit cost. `gate_or_none` checks force_submit
                # FIRST and would reject the constructor's mandatory
                # ask_user even if the numeric budget is restored. Clear
                # the flag whenever the restored pool can again afford
                # the submit cost.
                shared.status.remaining_budget = min(
                    total_budget,
                    shared.status.remaining_budget + reserve,
                )
                if shared.status.remaining_budget > ACTION_COSTS["submit_query"]:
                    shared.status.force_submit = False

                # ----- PROJECTION-RESOLVER (STAGE 2) PHASE -----
                resolver_record = AgentRecord(
                    role="projection_resolver",
                    depth=0,
                    parent_idx=None,
                    instruction="resolve projection",
                    started_at=time.monotonic(),
                )
                shared.agent_records.append(resolver_record)
                resolver_idx = len(shared.agent_records) - 1
                resolver_deps = TaskDeps(
                    shared=shared, depth=0, max_depth=0,
                    self_record_idx=resolver_idx,
                )
                current_record = resolver_record
                current_deps = resolver_deps
                resolver_builder = (
                    _build_projection_resolver_oneshot if is_one_shot
                    else _build_projection_resolver
                )
                resolver_agent = resolver_builder(
                    model=self.model,
                    model_settings=self._model_settings,
                    self_model_id=self.model_id,
                )
                resolver_template = (
                    PROJECTION_RESOLVER_ONESHOT_PROMPT if is_one_shot
                    else PROJECTION_RESOLVER_PROMPT
                )
                resolver_prompt = resolver_template.format(
                    amb_user_query=task_data["amb_user_query"],
                    spec=spec,
                    budget=shared.status.remaining_budget,
                    db_name=db_name,
                )
                resolver_recovery = _ONE_SHOT_RECOVERY_PROMPT if is_one_shot else None
                resolver_result = await _run_projection_resolver(
                    resolver_agent=resolver_agent,
                    instructions=resolver_prompt,
                    user_prompt=task_data["amb_user_query"],
                    deps=resolver_deps,
                    model_id=self.model_id,
                    recovery_prompt=resolver_recovery,
                )
                # Record resolver output verbatim so trajectories show
                # what got passed to the constructor. Fold every
                # resolver-run's messages + tool_call_stats + turn
                # counts in so per-task totals don't lose Stage 2's
                # contribution (the helper aggregates them across the
                # initial run + the optional empty-list-guard retry).
                resolver_record.output = repr(resolver_result.projection)
                resolver_record.user_sim_transcript = list(
                    resolver_deps.user_sim_transcript,
                )
                resolver_record.usage = resolver_deps.usage
                resolver_record.messages = resolver_result.messages
                resolver_record.tool_call_stats = resolver_result.tool_call_stats
                resolver_record.n_agent_turns = resolver_result.n_agent_turns
                resolver_record.ended_at = time.monotonic()

                # Empty-after-guard: skip the constructor and finalize
                # as never_submitted with a diagnostic that FMA can
                # filter on.
                if resolver_result.status == "empty_after_guard":
                    shared.submitter_result = {
                        "phase1_passed": False,
                        "phase2_passed": False,
                        "total_reward": 0.0,
                        "finished": False,
                        "submitted_sql": None,
                        "submitted_query": None,
                        "submission_status": "never_submitted",
                    }
                    return _finalize(
                        shared=shared,
                        instance_id=instance_id,
                        db_name=db_name,
                        deleted_kb_ids=deleted_kb_ids,
                        slayer_storage_dir=slayer_storage_dir,
                        final_output_excerpt="",
                        error=None,
                        projection_resolver_status="empty_after_guard",
                    )

                # ----- CONSTRUCTOR PHASE -----
                constructor_record = AgentRecord(
                    role="query_constructor",
                    depth=0,
                    parent_idx=None,
                    instruction="assemble + submit",
                    started_at=time.monotonic(),
                )
                shared.agent_records.append(constructor_record)
                constructor_idx = len(shared.agent_records) - 1
                constructor_deps = TaskDeps(
                    shared=shared, depth=0, max_depth=0,
                    self_record_idx=constructor_idx,
                )
                current_record = constructor_record
                current_deps = constructor_deps
                confirmed_projection_tuple = tuple(resolver_result.projection)
                constructor_builder = (
                    _build_query_constructor_oneshot if is_one_shot
                    else _build_query_constructor
                )
                constructor_agent = constructor_builder(
                    model=self.model,
                    model_settings=self._model_settings,
                    shared_slayer_server=slayer_server,
                    confirmed_projection=confirmed_projection_tuple,
                    self_model_id=self.model_id,
                )
                # Render the confirmed list as a numbered block so the
                # prompt template shows it cleanly. Same source object
                # as the closure — order and content cannot drift.
                confirmed_projection_block = "\n".join(
                    f"  {i + 1}. {name}"
                    for i, name in enumerate(confirmed_projection_tuple)
                )
                constructor_template = (
                    QUERY_CONSTRUCTOR_ONESHOT_PROMPT if is_one_shot
                    else QUERY_CONSTRUCTOR_PROMPT
                )
                constructor_prompt = constructor_template.format(
                    amb_user_query=task_data["amb_user_query"],
                    spec=spec,
                    confirmed_projection=confirmed_projection_block,
                    budget=shared.status.remaining_budget,
                    db_name=db_name,
                )
                constructor_run = await constructor_agent.run(
                    user_prompt=task_data["amb_user_query"],
                    instructions=constructor_prompt,
                    deps=constructor_deps,
                    usage_limits=UsageLimits(
                        request_limit=MAX_MODEL_TURNS * 2,
                    ),
                )
                _fill_record_from_run(
                    constructor_record, constructor_run, constructor_deps,
                    self.model_id,
                )
                constructor_output = str(constructor_run.output)
        except Exception as e:
            logger.exception("Recursive agent error on %s: %s", instance_id, e)
            subs = getattr(e, "exceptions", None)
            if subs:
                for i, sub in enumerate(subs):
                    logger.error("  sub-exception %d: %r", i, sub)
            # Stamp the currently-running top-level record with the
            # error so the trajectory identifies which agent failed.
            # The partial sub-agent records (whichever already completed
            # inside spawn_subagent) keep their fully-populated state.
            if current_record.error is None:
                current_record.error = f"{type(e).__name__}: {e}"
                current_record.usage = current_deps.usage
                current_record.user_sim_transcript = list(
                    current_deps.user_sim_transcript,
                )
                current_record.ended_at = time.monotonic()
            return _finalize(
                shared=shared,
                instance_id=instance_id,
                db_name=db_name,
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
                final_output_excerpt="",
                error=str(e),
            )

        return _finalize(
            shared=shared,
            instance_id=instance_id,
            db_name=db_name,
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
            final_output_excerpt=constructor_output[:500],
            error=None,
        )


class _null_async_context:
    """No-op async context for tests / runs without an MCP server."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _ResolverResult:
    """Stage 2 result: the confirmed projection list + a status string
    indicating whether the user-sim confirmed (`confirmed`) or whether
    the empty-list guard fell through (`empty_after_guard`).

    Also carries `messages`, `tool_call_stats`, and `n_agent_turns`
    aggregated across the initial run + the optional empty-list-guard
    retry — so the AgentRecord written by run_task can surface Stage 2
    in per-task tool stats / turn counts the same way the root and
    constructor agents do.
    """

    __slots__ = (
        "messages", "n_agent_turns", "projection",
        "status", "tool_call_stats",
    )

    def __init__(
        self,
        projection: list[str],
        status: str,
        *,
        messages: list | None = None,
        tool_call_stats: dict | None = None,
        n_agent_turns: int | None = None,
    ) -> None:
        self.projection = projection
        self.status = status
        self.messages = messages or []
        self.tool_call_stats = tool_call_stats
        self.n_agent_turns = n_agent_turns


def _fold_run_usage_into_deps(run: Any, deps: Any, model_id: str) -> None:
    """Fold a pydantic-ai run's `.usage()` into deps.usage as one
    `scope=agent` call entry. Mirrors what `_fill_record_from_run` does
    for the root and constructor runs — extracted so the resolver
    wrapper can call it once per `agent.run` (possibly twice with the
    empty-list guard) without duplicating record-shaping logic."""
    run_usage = run.usage()
    deps.usage.add_call(
        scope="agent",
        model=model_id,
        prompt=getattr(run_usage, "input_tokens", 0) or 0,
        completion=getattr(run_usage, "output_tokens", 0) or 0,
        cache_read=getattr(run_usage, "cache_read_tokens", 0) or 0,
        cache_write=getattr(run_usage, "cache_write_tokens", 0) or 0,
    )


def _aggregate_runs(
    runs: list[Any], status: str, projection: list[str],
) -> _ResolverResult:
    """Merge messages + tool_call_stats + turn counts across one or
    two resolver runs into a single result, so the AgentRecord can
    carry the same per-agent metrics the root/constructor records do.
    Tool stats use the shared `_merge_tool_stats` so the per-task
    aggregate seen in `_finalize` is consistent."""
    messages: list = []
    turns: int = 0
    saw_turns = False
    stat_parts: list[dict | None] = []
    for run in runs:
        messages.extend(_serialize_messages(run))
        n = _count_turns(run)
        if n is not None:
            saw_turns = True
            turns += n
        stat_parts.append(_extract_tool_stats(run))
    merged_stats = _merge_tool_stats(stat_parts)
    return _ResolverResult(
        projection=projection,
        status=status,
        messages=messages,
        tool_call_stats=merged_stats,
        n_agent_turns=turns if saw_turns else None,
    )


_ONE_SHOT_RECOVERY_PROMPT = (
    "Your previous output was an empty list. Re-read the user's question "
    "and the specification, propose at least one output column you can "
    "derive from them, and return the list. There is no user simulator "
    "to consult — decide the projection autonomously and finalise."
)


async def _run_projection_resolver(
    *,
    resolver_agent: Any,
    instructions: str,
    user_prompt: str,
    deps: Any,
    model_id: str,
    recovery_prompt: str | None = None,
) -> _ResolverResult:
    """Run Stage 2 with an empty-list guard.

    First attempt: run the resolver. If the returned `list[str]` is
    non-empty, return `_ResolverResult(projection=..., status='confirmed')`.

    If the first attempt is empty, run ONCE MORE with a recovery
    user_prompt — passing the first run's `message_history` so the
    recovery turn continues the same conversation rather than starting
    fresh and losing the first attempt's ask_user exchanges. If the
    second attempt also returns empty, return
    `_ResolverResult(projection=[], status='empty_after_guard')` —
    the caller (run_task) skips Stage 3 and finalizes with never_submitted.

    DEV-1462: ``recovery_prompt`` lets the caller swap in a one-shot
    recovery message (no "ask the user to confirm") so the model isn't
    steered toward a tool the one-shot resolver doesn't have. Default
    is the a-interact recovery text.

    Each agent.run's `.usage()` is folded into `deps.usage` so the
    resolver's AgentRecord carries its share of tokens; messages +
    tool stats + turn counts are aggregated into the returned result
    for the caller to write onto the AgentRecord.
    """
    first_run = await resolver_agent.run(
        user_prompt=user_prompt,
        instructions=instructions,
        deps=deps,
        usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
    )
    _fold_run_usage_into_deps(first_run, deps, model_id)
    projection = list(first_run.output or [])
    if projection:
        return _aggregate_runs([first_run], "confirmed", projection)

    # Empty-list guard: one more attempt, continuing the same
    # conversation via message_history so the model sees its prior
    # turn's context.
    if recovery_prompt is None:
        recovery_prompt = (
            "Your previous output was an empty list. Propose at least one "
            "output column you derive from the user's question and the "
            "specification, then ask the user to confirm or refine."
        )
    recovery_run = await resolver_agent.run(
        user_prompt=recovery_prompt,
        instructions=instructions,
        deps=deps,
        message_history=first_run.all_messages(),
        usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
    )
    _fold_run_usage_into_deps(recovery_run, deps, model_id)
    projection = list(recovery_run.output or [])
    if projection:
        return _aggregate_runs([first_run, recovery_run], "confirmed", projection)
    return _aggregate_runs(
        [first_run, recovery_run], "empty_after_guard", [],
    )


def _fill_record_from_run(
    record: AgentRecord, run: Any, deps: TaskDeps, self_model_id: str,
) -> None:
    """Mirror the pydantic_ai adapter's post-run capture: read
    ``run.usage()`` and fold the agent-side tokens explicitly into
    ``deps.usage`` (they're NOT auto-merged from the run object), then
    record output / messages / tool stats / turn count / end time."""
    run_usage = run.usage()
    deps.usage.add_call(
        scope="agent",
        model=self_model_id,
        prompt=getattr(run_usage, "input_tokens", 0) or 0,
        completion=getattr(run_usage, "output_tokens", 0) or 0,
        cache_read=getattr(run_usage, "cache_read_tokens", 0) or 0,
        cache_write=getattr(run_usage, "cache_write_tokens", 0) or 0,
    )
    record.output = str(run.output)
    record.user_sim_transcript = list(deps.user_sim_transcript)
    record.usage = deps.usage
    record.messages = _serialize_messages(run)
    record.tool_call_stats = _extract_tool_stats(run)
    record.n_agent_turns = _count_turns(run)
    record.ended_at = time.monotonic()


def _finalize(
    *,
    shared: SharedTaskState,
    instance_id: str,
    db_name: str,
    deleted_kb_ids: list[int],
    slayer_storage_dir: str,
    final_output_excerpt: str,
    error: str | None,
    projection_resolver_status: str | None = None,
) -> dict:
    """Build the result row from the shared state. Aggregates per-agent
    usage + tool stats + turn counts; preserves whatever AgentRecords
    completed (even on error)."""
    submitter = shared.submitter_result or {}

    # Aggregate usage across all agent records.
    total_usage = TokenUsage()
    n_turns_total: int | None = None
    for rec in shared.agent_records:
        total_usage.merge(rec.usage)
        if rec.n_agent_turns is not None:
            n_turns_total = (n_turns_total or 0) + rec.n_agent_turns
    tool_stats = _merge_tool_stats(
        [r.tool_call_stats for r in shared.agent_records],
    )

    trajectory = {
        "final_output_excerpt": final_output_excerpt,
        "agents": [r.model_dump() for r in shared.agent_records],
    }

    row = {
        "task_id": instance_id,
        "instance_id": instance_id,
        "database": db_name,
        "phase1_passed": submitter.get("phase1_passed", False),
        "phase2_passed": submitter.get("phase2_passed", False),
        "total_reward": submitter.get("total_reward", 0.0),
        "submitted_sql": submitter.get("submitted_sql"),
        "submitted_query": submitter.get("submitted_query"),
        "trajectory": trajectory,
        "error": error,
        "usage": total_usage.model_dump(),
        "submission_status": submitter.get(
            "submission_status", "never_submitted",
        ),
        "phase1_observation": submitter.get("phase1_observation"),
        "phase2_observation": submitter.get("phase2_observation"),
        "predicted_result_json": submitter.get("predicted_result_json"),
        "gold_result_json": submitter.get("gold_result_json"),
        "n_agent_turns": n_turns_total,
        "tool_call_stats": tool_stats,
        # Dual-eval fields — populated only when --use-audited-gold-sql
        # is on AND the overlay applied; NULL elsewhere.
        "phase1_passed_audited": submitter.get("phase1_passed_audited"),
        "phase1_passed_original": submitter.get("phase1_passed_original"),
        "phase1_observation_audited": submitter.get("phase1_observation_audited"),
        "phase1_observation_original": submitter.get("phase1_observation_original"),
    }
    if projection_resolver_status is not None:
        row["projection_resolver_status"] = projection_resolver_status
    return finalize_result_row(
        row,
        deleted_kb_ids=deleted_kb_ids,
        slayer_storage_dir=slayer_storage_dir,
    )
