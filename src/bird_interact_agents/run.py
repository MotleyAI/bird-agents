"""CLI entry point for running BIRD-Interact evaluations."""

import argparse
import asyncio
import json
import logging
import shutil
import statistics
import time
from pathlib import Path

import sqlite3 as _sqlite3
from typing import Any as _Any

from bird_interact_agents import paths
from bird_interact_agents.benchmark import cli_dataset_tokens, get_benchmark
from bird_interact_agents.eval.cascading_report import emit_cascading_eval_json
from bird_interact_agents.eval.grade_in_place import (
    grade_one_submission,
    write_failed_submission_annotation,
)
from bird_interact_agents.harness import (
    apply_audited_gold_overlay,
    calculate_budget,
    execute_submit_action,
    finalize_result_row,
    load_benchmark_tasks,
    load_db_data_if_needed,
    materialize_task_db,
    SampleStatus,
)
from bird_interact_agents.results_db import (
    TaskResultRow,
    insert_run_metadata,
    insert_task_result,
    open_db,
)
from bird_interact_agents.usage import TokenUsage

def build_aggregate_eval(*, db_path: Path | str) -> dict[str, _Any]:
    """Read task_results from a results.db and emit aggregate phase-1
    metrics. DEV-1515: the legacy dual-eval bool columns are gone;
    every per-task cascade verdict now lives in the SubmissionAnnotation
    written inline by ``grade_and_write``. The cascading_phase1 block
    in ``eval.json`` is built by
    :func:`bird_interact_agents.eval.cascading_report.emit_cascading_eval_json`
    over the per-row annotation files; this helper only returns the
    simple ``phase1_count`` / ``phase1_rate`` totals used by
    back-compat consumers of the results DB."""
    conn = _sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT phase1_passed FROM task_results"
        ).fetchall()
    finally:
        conn.close()
    n = len(rows)
    p1 = sum(1 for r in rows if r[0])
    return {
        "total_tasks": n,
        "phase1_count": p1,
        "phase1_rate": p1 / n if n else 0.0,
    }


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def _validate_slayer_setup(
    *, slayer_setup: str, framework: str, query_mode: str, mode: str,
) -> None:
    """Reject ``slayer_setup`` combinations the on-the-fly path doesn't
    support, before any task starts. Used by both the CLI parser and
    ``run_evaluation`` so direct programmatic callers can't bypass the
    guard.

    Raises ``ValueError`` (programmatic callers); ``main`` catches and
    re-raises via ``parser.error`` so the CLI gets the standard
    argparse exit-2 + stderr behaviour.
    """
    # DEV-1462: one-shot REQUIRES on-the-fly. Pre-encoded one-shot would
    # silently target the committed `slayer_models/` which has no
    # LiveSQLBench coverage; fail fast.
    if mode == "one-shot" and slayer_setup != "on-the-fly":
        raise ValueError(
            "--mode one-shot requires --slayer-setup on-the-fly; "
            f"got --slayer-setup {slayer_setup!r}",
        )
    # pydantic_ai_otf_encode and the two claude_sdk_otf flavors are
    # on-the-fly-only adapters (DEV-1454 / DEV-1505 / DEV-1507).
    if (
        framework in (
            "pydantic_ai_otf_encode", "claude_sdk_otf",
            "claude_sdk_otf_ainteract",
        )
        and slayer_setup != "on-the-fly"
    ):
        raise ValueError(
            f"--framework {framework} requires "
            "--slayer-setup on-the-fly; "
            f"got --slayer-setup {slayer_setup}"
        )
    if slayer_setup == "pre-encoded":
        return
    if slayer_setup != "on-the-fly":
        raise ValueError(
            f"--slayer-setup must be 'pre-encoded' or 'on-the-fly'; "
            f"got {slayer_setup!r}"
        )
    if framework not in (
        "pydantic_ai_recursive", "pydantic_ai_otf_encode", "claude_sdk_otf",
        "claude_sdk_otf_ainteract",
    ):
        raise ValueError(
            "--slayer-setup on-the-fly is only supported with "
            "--framework pydantic_ai_recursive, "
            "--framework pydantic_ai_otf_encode, "
            "--framework claude_sdk_otf, or "
            "--framework claude_sdk_otf_ainteract; "
            f"got --framework {framework}"
        )
    if query_mode != "slayer":
        raise ValueError(
            "--slayer-setup on-the-fly is only supported with "
            "--query-mode slayer; "
            f"got --query-mode {query_mode}"
        )
    # DEV-1462: on-the-fly now allows {a-interact, one-shot}.
    if mode not in ("a-interact", "one-shot"):
        raise ValueError(
            "--slayer-setup on-the-fly is only supported with "
            "--mode a-interact or --mode one-shot; "
            f"got --mode {mode}"
        )


def _validate_dataset_mode(dataset: str, mode: str) -> None:
    """Gate ``--mode`` against the benchmark's declared ``supported_modes``
    (registry-driven, so adding a benchmark needs no edit here).

    This single membership check enforces both directions of the old
    hand-written gate: e.g. ``one-shot`` is rejected for mini-interact (not in
    its modes) and ``a-interact`` is rejected for livesqlbench — because each
    benchmark's ``supported_modes`` encodes exactly what it accepts.
    """
    b = get_benchmark(dataset)
    if mode not in b.supported_modes:
        raise ValueError(
            f"--mode {mode!r} is not supported by --dataset {dataset!r} "
            f"(benchmark {b.name!r}); supported modes: "
            f"{', '.join(b.supported_modes)}.",
        )


def _validate_one_shot_framework(*, mode: str, query_mode: str, framework: str) -> None:
    """DEV-1462: one-shot dispatch is recursive + otf_encode + claude_sdk_otf
    only.

    ``oracle`` stays framework-agnostic; ``a-interact``/``c-interact`` keep
    their existing per-framework dispatch. After DEV-1507
    ``claude_sdk_otf_ainteract`` is a-interact only — explicitly NOT on this
    list so the user-facing error names the right framework.
    """
    if mode != "one-shot":
        return
    if query_mode != "slayer":
        raise ValueError(
            "--mode one-shot requires --query-mode slayer; "
            f"got --query-mode {query_mode!r}",
        )
    if framework not in (
        "pydantic_ai_recursive", "pydantic_ai_otf_encode", "claude_sdk_otf",
    ):
        raise ValueError(
            "--mode one-shot is only supported with --framework "
            "pydantic_ai_recursive, --framework pydantic_ai_otf_encode, or "
            f"--framework claude_sdk_otf; got --framework {framework!r}",
        )


# (framework -> (bound dataset, bound mode)). DEV-1507: claude_sdk_otf is
# the livesqlbench/one-shot flavor; claude_sdk_otf_ainteract is the
# mini_interact/a-interact flavor. Every other framework is unbound —
# they don't appear here.
_FRAMEWORK_DATASET_MODE_BINDING = {
    "claude_sdk_otf": ("livesqlbench", "one-shot"),
    "claude_sdk_otf_ainteract": ("mini_interact", "a-interact"),
}


def _validate_framework_dataset_mode(
    *, framework: str, dataset: str, mode: str,
) -> None:
    """DEV-1507: reject any (framework, dataset, mode) combo that violates a
    framework's hard binding.

    Oracle does NOT bypass — picking the bound framework signals intent to
    use that flavor's agent, and ``run_oracle_task`` short-circuits
    framework dispatch entirely. A user who wants to oracle-eval a
    different benchmark should drop the framework binding (oracle is
    framework-agnostic in dispatch).

    Frameworks not listed in ``_FRAMEWORK_DATASET_MODE_BINDING`` are
    unbound and pass through silently.
    """
    bound = _FRAMEWORK_DATASET_MODE_BINDING.get(framework)
    if bound is None:
        return
    # Canonicalize the dataset token before comparison so the documented
    # ``mini-interact`` alias (and any future alias) is accepted —
    # otherwise a programmatic ``run_evaluation`` / ``run_one_task`` call
    # using the alias against the canonical binding would fail with a
    # confusing error (Codex + CodeRabbit on PR #10). The CLI normalises
    # at the argparse boundary, but the programmatic surface doesn't.
    canonical_dataset = get_benchmark(dataset).name
    bound_dataset, bound_mode = bound
    if canonical_dataset != bound_dataset:
        raise ValueError(
            f"--framework {framework} is bound to --dataset {bound_dataset!r}; "
            f"got --dataset {dataset!r}",
        )
    if mode != bound_mode:
        raise ValueError(
            f"--framework {framework} is bound to --mode {bound_mode!r}; "
            f"got --mode {mode!r}",
        )


def _maybe_force_wipe_otf(
    *, otf_rebuild: bool, framework: str, dbs,
    benchmark: str,
) -> None:
    """``--otf-rebuild`` force-wipe: drop BOTH on-the-fly layers (the phase-1-3
    cache AND the KB-encoded reference) for ``dbs``, for either on-the-fly
    framework. No-op when the flag is off or the framework isn't on-the-fly.

    Wiping both layers together is load-bearing: wiping only the reference
    would let a stale cache be re-encoded into a "fresh" reference (Codex r2
    High#3).

    DEV-1462: ``benchmark`` (REQUIRED, explicit) selects the per-benchmark
    scoped roots so a LiveSQLBench ``--otf-rebuild`` never wipes the
    mini-interact cache (and vice versa). ``"mini_interact"`` maps to the
    legacy roots.
    """
    if not otf_rebuild:
        return
    if framework not in (
        "pydantic_ai_recursive", "pydantic_ai_otf_encode", "claude_sdk_otf",
        "claude_sdk_otf_ainteract",
    ):
        return
    from bird_interact_agents.slayer_otf.reference_build import (
        purge_caches,
        purge_references,
    )

    dbs = set(dbs)
    removed_cache = purge_caches(
        paths.slayer_otf_cache_root(benchmark=benchmark), dbs,
    )
    removed_ref = purge_references(
        paths.slayer_models_otf_root(benchmark=benchmark), dbs,
    )
    logger.info(
        "--otf-rebuild: wiped OTF cache for %s and reference for %s "
        "(will rebuild from scratch)",
        sorted(removed_cache) or "none present",
        sorted(removed_ref) or "none present",
    )


async def run_oracle_task(task_data: dict, data_path_base: str) -> dict:
    """Submit ground-truth SQL directly — no LLM, validates eval pipeline."""
    from bird_interact_agents.agents._submit import (
        capture_result_snapshot,
        classify_submission,
    )
    import json as _json

    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    sol_sql = task_data.get("sol_sql", [])
    if isinstance(sol_sql, list) and sol_sql:
        sol_sql = sol_sql[0]
    elif isinstance(sol_sql, list):
        sol_sql = ""

    load_db_data_if_needed(db_name, data_path_base)
    # DEV-1462 B0: LiveSQLBench oracle runs need per-task DB isolation too —
    # at --concurrency > 1, multiple oracle tasks on the same DB would
    # otherwise race the shared <db>.sqlite that the OTF cache reads.
    # No-op for mini-interact (no `dataset` marker).
    materialize_task_db(task_data, data_path_base)
    status = SampleStatus(idx=0, original_data=task_data)

    observation, reward, p1, p2, finished = execute_submit_action(
        sol_sql, status, data_path_base
    )

    predicted = capture_result_snapshot(sol_sql, db_name, data_path_base)
    gold = capture_result_snapshot(sol_sql, db_name, data_path_base)
    return finalize_result_row(
        {
            "task_id": instance_id,
            "instance_id": instance_id,
            "database": db_name,
            "phase1_passed": p1,
            "phase2_passed": p2,
            "total_reward": reward if reward is not None else 0.0,
            "submitted_sql": sol_sql,
            "submitted_query": None,
            "trajectory": [],
            "error": None,
            "submission_status": classify_submission(
                p1=p1, p2=p2, observation=observation,
            ),
            "phase1_observation": observation,
            "phase2_observation": None,
            "predicted_result_json": (
                _json.dumps(predicted, default=str)
                if predicted is not None else None
            ),
            "gold_result_json": (
                _json.dumps(gold, default=str) if gold is not None else None
            ),
            "n_agent_turns": 0,
        },
        deleted_kb_ids=[],
        slayer_storage_dir="",
    )


def make_runner(
    *,
    framework: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    strict: bool,
    prompt_cache: bool,
    max_depth: int,
    slayer_storage_root: str | None,
    slayer_setup: str = "pre-encoded",
    reasoning_effort: str | None = None,
):
    """Public alias for `_make_runner` — the cloud actor (and other
    throughput-sensitive callers) call this once at startup and reuse the
    returned closure across tasks to avoid per-task agent reconstruction.

    Validates ``slayer_setup`` against the framework/query_mode/mode tuple
    upfront so a programmatic caller (cloud actor, custom driver) that
    bypasses the CLI parser can't silently get a runner that ignores the
    setting (Codex finding on PR #19).

    DEV-1462: also validates the one-shot ⟹ slayer + framework∈{recursive,
    otf_encode} dispatch. ``make_runner`` has no ``dataset`` argument (it
    is a per-task runner factory); the one-shot ⟹ livesqlbench guard
    lives further down in ``run_task`` itself, keyed on the task's
    loader-stamped ``dataset`` marker (Codex #1).

    Guard order: one-shot dispatch FIRST so a one-shot-with-wrong-framework
    surfaces a "one-shot requires …" error, not the more generic
    on-the-fly-framework error from ``_validate_slayer_setup``."""
    _validate_one_shot_framework(
        mode=mode, query_mode=query_mode, framework=framework,
    )
    _validate_slayer_setup(
        slayer_setup=slayer_setup, framework=framework,
        query_mode=query_mode, mode=mode,
    )
    return _make_runner(
        framework=framework, query_mode=query_mode, mode=mode,
        agent_model=agent_model, strict=strict, prompt_cache=prompt_cache,
        max_depth=max_depth, slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup, reasoning_effort=reasoning_effort,
    )


async def run_one_task_with_runner(
    runner,
    task_data: dict,
    *,
    data_dir: str,
    patience: int,
    user_sim_model: str,
) -> dict:
    """Run a single task with a pre-built runner, returning a
    `_persist`-consumable dict. Same try/except + duration + default-key
    semantics as `run_one_task`, but skips the per-call agent
    construction — costly for the cloud actor on long runs (CR#14)."""
    instance_id = str(task_data.get("instance_id") or "")
    t_start = time.perf_counter()
    try:
        r = await runner(task_data, data_dir, patience, user_sim_model)
        # A misbehaving custom runner could return None or some other
        # non-dict type. Treat that as a per-task failure rather than a
        # NameError outside the try/except down the chain.
        if not isinstance(r, dict):
            raise TypeError(
                f"runner returned {type(r).__name__}; expected dict"
            )
    except Exception as e:  # noqa: BLE001
        logger.error("Error on %s: %s", instance_id, e)
        r = finalize_result_row(
            {
                "task_id": instance_id,
                "instance_id": instance_id,
                "database": task_data.get("selected_database", ""),
                "phase1_passed": False,
                "phase2_passed": False,
                "total_reward": 0.0,
                "trajectory": [],
                "error": str(e),
            },
            deleted_kb_ids=[],
            slayer_storage_dir="",
        )
    r["duration_s"] = time.perf_counter() - t_start
    sol = task_data.get("sol_sql")
    if isinstance(sol, list) and sol:
        gt = sol[0]
    elif isinstance(sol, str):
        gt = sol
    else:
        gt = None
    r.setdefault("ground_truth_sql", gt)
    for key, default in (
        ("submitted_sql", None),
        ("submitted_query", None),
        ("error", None),
        ("submission_status", "never_submitted"),
        ("phase1_observation", None),
        ("phase2_observation", None),
        ("predicted_result_json", None),
        ("gold_result_json", None),
        ("n_agent_turns", None),
        ("instance_id", instance_id),
        ("database", task_data.get("selected_database", "")),
        ("phase1_passed", False),
        ("phase2_passed", False),
        ("total_reward", 0.0),
    ):
        r.setdefault(key, default)
    return r


def _make_runner(
    *,
    framework: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    strict: bool,
    prompt_cache: bool,
    max_depth: int,
    slayer_storage_root: str | None,
    slayer_setup: str = "pre-encoded",
    reasoning_effort: str | None = None,
):
    """Construct the per-task runner closure for the given config.

    Returns an `async (task_data, data_dir, patience, user_sim_model) -> dict`
    callable. The agent (if any) is constructed *once* and captured in the
    closure — callers that want one-shot semantics should call this factory
    per task. Callers that need throughput (`run_evaluation`, in-cluster
    actor) call it once and reuse the closure.
    """
    if mode == "oracle":
        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            return await run_oracle_task(td, data_dir)
        return run_one
    if framework == "claude_sdk":
        from bird_interact_agents.agents.claude_sdk.agent import ClaudeSDKAgent

        if strict:
            logger.warning(
                "[claude_sdk] --strict is a no-op for Anthropic models; ignored."
            )
        agent = ClaudeSDKAgent(
            slayer_storage_root=slayer_storage_root, model=agent_model,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "claude_sdk_otf":
        from bird_interact_agents.agents.claude_sdk_otf import ClaudeSDKOtfAgent

        if strict:
            logger.warning(
                "[claude_sdk_otf] --strict is a no-op for Anthropic models; "
                "ignored."
            )
        agent_cso = ClaudeSDKOtfAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            slayer_setup=slayer_setup,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_cso.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "claude_sdk_otf_ainteract":
        from bird_interact_agents.agents.claude_sdk_otf_ainteract import (
            ClaudeSDKOtfAInteractAgent,
        )

        if strict:
            logger.warning(
                "[claude_sdk_otf_ainteract] --strict is a no-op for "
                "Anthropic models; ignored."
            )
        agent_csoa = ClaudeSDKOtfAInteractAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            slayer_setup=slayer_setup,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csoa.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "pydantic_ai":
        from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

        agent_pa = PydanticAIAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            strict=strict,
            prompt_cache=prompt_cache,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_pa.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "pydantic_ai_recursive":
        from bird_interact_agents.agents.pydantic_ai_recursive import (
            PydanticAIRecursiveAgent,
        )

        if strict:
            logger.warning(
                "[pydantic_ai_recursive] --strict is unsupported; ignored.",
            )
        agent_par = PydanticAIRecursiveAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            max_depth=max_depth,
            prompt_cache=prompt_cache,
            slayer_setup=slayer_setup,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_par.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "pydantic_ai_otf_encode":
        from bird_interact_agents.agents.pydantic_ai_otf_encode import (
            PydanticAIOtfEncodeAgent,
        )

        if strict:
            logger.warning(
                "[pydantic_ai_otf_encode] --strict is unsupported; ignored.",
            )
        agent_otf = PydanticAIOtfEncodeAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            max_depth=max_depth,
            prompt_cache=prompt_cache,
            slayer_setup=slayer_setup,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_otf.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "mcp_agent":
        from bird_interact_agents.agents.mcp_agent.agent import McpAgentAgent

        agent_mcp = McpAgentAgent(
            slayer_storage_root=slayer_storage_root, model=agent_model,
            strict=strict,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_mcp.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "agno":
        from bird_interact_agents.agents.agno.agent import AgnoAgent

        agent_agno = AgnoAgent(
            slayer_storage_root=slayer_storage_root, model_id=agent_model,
            strict=strict,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_agno.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    if framework == "smolagents":
        from bird_interact_agents.agents.smolagents.agent import SmolagentsAgent

        agent_sa = SmolagentsAgent(
            slayer_storage_root=slayer_storage_root, model_id=agent_model,
            strict=strict,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_sa.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            )
        return run_one
    raise ValueError(f"Unknown framework: {framework}")


async def run_one_task(
    task_data: dict,
    *,
    data_dir: str,
    framework: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    user_sim_model: str,
    patience: int,
    strict: bool,
    use_audited_gold_sql: bool,  # noqa: ARG001 - accepted for API symmetry; overlay applied caller-side
    prompt_cache: bool,
    max_depth: int,
    slayer_storage_root: str | None,
    slayer_setup: str = "pre-encoded",
    reasoning_effort: str | None = None,
) -> dict:
    """Run a single per-task evaluation and return a `_persist`-consumable dict.

    Extracted from `run_evaluation`'s inline loop body so the cloud actor
    (and other one-shot callers) can run individual tasks without
    re-implementing the per-framework agent dispatch + try/except + timing
    that `run_evaluation` already does.

    `use_audited_gold_sql` is accepted for API symmetry. The audited-gold
    overlay is *not* applied here — callers should overlay the dataset
    before calling this function (`run_evaluation` does this once at the
    top of the run; the cloud driver mirrors it).

    ``slayer_setup`` is validated against framework/query_mode/mode and
    threaded into the runner factory, so the cloud path can opt into
    on-the-fly setup (Codex finding on PR #19).

    DEV-1462: when ``mode="one-shot"``, the task MUST carry the
    ``dataset="livesqlbench"`` marker (the loader stamps it). A
    programmatic caller that bypasses the loader can't silently get a
    one-shot run on un-marked task data (Codex #1 — programmatic-bypass
    close, complementary to ``_validate_dataset_mode`` on the CLI side).
    """
    # DEV-1507: enforce the framework-bound (dataset, mode) gate at the
    # per-task programmatic surface too — `make_runner` has no dataset
    # arg and `_make_runner` short-circuits to `run_oracle_task` regardless
    # of framework, so a direct caller invoking
    # `run_one_task(framework="claude_sdk_otf_ainteract", mode="oracle")`
    # would otherwise bypass the "no oracle bypass" contract.
    _validate_framework_dataset_mode(
        framework=framework,
        dataset=task_data.get("dataset") or "mini_interact",
        mode=mode,
    )
    _validate_one_shot_framework(
        mode=mode, query_mode=query_mode, framework=framework,
    )
    _validate_slayer_setup(
        slayer_setup=slayer_setup, framework=framework,
        query_mode=query_mode, mode=mode,
    )
    if mode == "one-shot" and not get_benchmark(
        task_data.get("dataset") or "mini_interact"
    ).one_shot:
        raise ValueError(
            "--mode one-shot requires a task whose benchmark declares "
            "one_shot=True (its loader stamps task_data['dataset']); got "
            f"dataset={task_data.get('dataset')!r}. This guard catches "
            "programmatic callers that bypass the one-shot loader.",
        )
    runner = _make_runner(
        framework=framework,
        query_mode=query_mode,
        mode=mode,
        agent_model=agent_model,
        strict=strict,
        prompt_cache=prompt_cache,
        max_depth=max_depth,
        slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup,
        reasoning_effort=reasoning_effort,
    )
    instance_id = str(task_data.get("instance_id") or "")
    t_start = time.perf_counter()
    try:
        r = await runner(task_data, data_dir, patience, user_sim_model)
    except Exception as e:  # noqa: BLE001 — same catch-all as the old inline loop
        logger.error("Error on %s: %s", instance_id, e)
        r = finalize_result_row(
            {
                "task_id": instance_id,
                "instance_id": instance_id,
                "database": task_data.get("selected_database", ""),
                "phase1_passed": False,
                "phase2_passed": False,
                "total_reward": 0.0,
                "trajectory": [],
                "error": str(e),
            },
            deleted_kb_ids=[],
            slayer_storage_dir="",
        )
    r["duration_s"] = time.perf_counter() - t_start

    # Ensure `_persist`-consumable keys are present even if the runner
    # didn't populate them (e.g. early-aborted oracle tasks).
    sol = task_data.get("sol_sql")
    if isinstance(sol, list) and sol:
        gt = sol[0]
    elif isinstance(sol, str):
        gt = sol
    else:
        gt = None
    r.setdefault("ground_truth_sql", gt)
    for key, default in (
        ("submitted_sql", None),
        ("submitted_query", None),
        ("error", None),
        ("submission_status", "never_submitted"),
        ("phase1_observation", None),
        ("phase2_observation", None),
        ("predicted_result_json", None),
        ("gold_result_json", None),
        ("n_agent_turns", None),
        ("instance_id", instance_id),
        ("database", task_data.get("selected_database", "")),
        ("phase1_passed", False),
        ("phase2_passed", False),
        ("total_reward", 0.0),
    ):
        r.setdefault(key, default)
    return r


async def run_evaluation(
    data_path: str,
    data_dir: str,
    output_path: str,
    mode: str,
    query_mode: str,
    framework: str,
    limit: int | None = None,
    concurrency: int = 3,
    patience: int = 3,
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
    slayer_storage_root: str | None = None,
    filter_ids: list[str] | None = None,
    agent_model: str = "anthropic/claude-sonnet-4-5",
    strict: bool = False,
    use_audited_gold_sql: bool = False,
    prompt_cache: bool = True,
    max_depth: int = 3,
    slayer_setup: str = "pre-encoded",
    otf_rebuild: bool = False,
    dataset: str = "mini-interact",
    gold_file: str | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Run full evaluation across all tasks."""
    # Programmatic-caller mirror of the CLI fail-fast guards. The CLI
    # parser rejects the same combinations in main(), but
    # ``run_evaluation`` is also called directly from tests and other
    # entry points; without these checks those callers would silently get
    # unsupported behaviour (Codex finding on DEV-1455 PR #19 +
    # DEV-1462 Codex round-2).
    _validate_dataset_mode(dataset=dataset, mode=mode)
    _validate_framework_dataset_mode(
        framework=framework, dataset=dataset, mode=mode,
    )
    _validate_one_shot_framework(
        mode=mode, query_mode=query_mode, framework=framework,
    )
    _validate_slayer_setup(
        slayer_setup=slayer_setup, framework=framework,
        query_mode=query_mode, mode=mode,
    )
    b = get_benchmark(dataset)
    if b.gold_required and not gold_file:
        raise ValueError(
            f"--dataset {b.name} requires --gold-file (the gated sidecar "
            "carrying sol_sql / external_knowledge / test_cases keyed by "
            "instance_id)",
        )

    # B3 empty-filter footgun hardening: a caller that passes
    # `filter_ids=[]` (e.g. `--instance-id ",,, "` collapsing to empty)
    # would, with the legacy truthy-check, fall through to running the
    # FULL task set. Treat "filter requested" as `is not None`.
    if filter_ids is not None and len(filter_ids) == 0:
        raise ValueError(
            "filter_ids was explicitly empty (zero matching ids). "
            "If you meant to run the full set, pass filter_ids=None.",
        )

    # Benchmark-aware loader dispatch (single source of truth in harness):
    # a gold-required benchmark merges its gated sidecar + stamps the marker +
    # SELECT-filters; otherwise plain load + optional instance-id filter.
    tasks = load_benchmark_tasks(
        dataset, data_path, gold_file, limit=limit, filter_ids=filter_ids,
    )

    # --otf-rebuild: force-wipe BOTH on-the-fly layers (cache + reference) for
    # the DBs in this run ONCE, before the (possibly concurrent) task loop, so
    # the lazy build regenerates each exactly once and all tasks reuse the fresh
    # copy. Default off reuses whatever is present. On-the-fly frameworks only.
    # DEV-1462: pass the per-benchmark scope so a livesqlbench rebuild never
    # wipes the mini-interact roots (and vice versa).
    benchmark_for_paths = b.name
    _maybe_force_wipe_otf(
        otf_rebuild=otf_rebuild,
        framework=framework,
        dbs={t.get("selected_database") for t in tasks if t.get("selected_database")},
        benchmark=benchmark_for_paths,
    )

    # DEV-1510: the audited-gold overlay now fires for ALL benchmarks. The
    # per-benchmark `audited_gold_layout` (per_db / single_file) on the
    # `Benchmark` descriptor selects the on-disk layout, so livesqlbench
    # picks up `audited_gold/livesqlbench_audited.jsonl` while mini-interact
    # keeps the per-db sidecar layout — both flow through the same call.
    audited_overlay_log: dict[str, str] = {}
    if use_audited_gold_sql:
        audited_overlay_log = apply_audited_gold_overlay(
            tasks, paths.audited_gold_root(), benchmark=b,
        )
        logger.info(
            "audited-gold overlay applied: %s",
            {s: sum(1 for v in audited_overlay_log.values() if v == s)
             for s in ("edited", "unrecoverable", "clean", "missing-row", "missing-file")},
        )
    logger.info(
        "%s/%s: Evaluating %d tasks (concurrency=%d)",
        mode, query_mode, len(tasks), concurrency,
    )

    # Build the runner once and reuse across tasks (avoids re-constructing
    # the per-framework agent N times).
    runner = _make_runner(
        framework=framework,
        query_mode=query_mode,
        mode=mode,
        agent_model=agent_model,
        strict=strict,
        prompt_cache=prompt_cache,
        max_depth=max_depth,
        slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup,
        reasoning_effort=reasoning_effort,
    )

    # Open the per-run results.db (lives next to eval.json) and write
    # the run-metadata header before any task starts.
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if audited_overlay_log:
        (output_dir / "audited_gold_overlay.json").write_text(
            json.dumps(audited_overlay_log, indent=2) + "\n"
        )
    db_path = output_dir / "results.db"
    db_conn = open_db(db_path)
    run_id = output_dir.name or "default"
    insert_run_metadata(
        db_conn,
        run_id=run_id,
        agent_model=agent_model,
        user_sim_model=user_sim_model,
        framework=framework,
        mode=mode,
        started_at=time.time(),
    )

    def _persist(td: dict, r: dict, started_at: float) -> None:
        """Insert one task result into the DB. Called immediately after
        each task — both successes and failures — so a mid-run crash
        never throws away completed-task data."""
        usage_blob = r.get("usage")
        usage_json = json.dumps(usage_blob) if usage_blob is not None else "{}"
        sol = td.get("sol_sql")
        if isinstance(sol, list) and sol:
            ground_truth = sol[0]
        elif isinstance(sol, str):
            ground_truth = sol
        else:
            ground_truth = None
        n_turns = r.get("n_agent_turns")
        stats_blob = r.get("tool_call_stats")
        tool_call_stats_json = (
            json.dumps(stats_blob) if stats_blob is not None else None
        )
        insert_task_result(db_conn, TaskResultRow(
            run_id=run_id,
            framework=framework,
            mode=mode,
            query_mode=query_mode,
            instance_id=str(r.get("instance_id") or td.get("instance_id") or ""),
            database=str(r.get("database") or td.get("selected_database") or ""),
            started_at=started_at,
            duration_s=float(r.get("duration_s") or 0.0),
            phase1_passed=bool(r.get("phase1_passed")),
            phase2_passed=bool(r.get("phase2_passed")),
            total_reward=float(r.get("total_reward") or 0.0),
            submitted_sql=r.get("submitted_sql"),
            submitted_query=r.get("submitted_query"),
            ground_truth_sql=r.get("ground_truth_sql") or ground_truth,
            error=r.get("error"),
            usage_json=usage_json,
            user_query=td.get("amb_user_query"),
            submission_status=str(
                r.get("submission_status") or "never_submitted"
            ),
            phase1_observation=r.get("phase1_observation"),
            phase2_observation=r.get("phase2_observation"),
            predicted_result_json=r.get("predicted_result_json"),
            gold_result_json=r.get("gold_result_json"),
            n_agent_turns=int(n_turns) if isinstance(n_turns, int) else None,
            tool_call_stats_json=tool_call_stats_json,
            phase1_observation_audited=r.get("phase1_observation_audited"),
            phase1_observation_original=r.get("phase1_observation_original"),
        ))

    # Run tasks with concurrency limiter
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    total_reward = 0.0
    p1_count = 0
    p2_count = 0

    # DEV-1515: inline-grade every task into ``<rows_dir>/<inst>/`` so
    # the existing aggregator at the bottom of ``run_evaluation`` can
    # emit the ``cascading_phase1`` block in ``eval.json``. Mirrors the
    # cloud worker (``cloud.ray_app._grade_one_submission``) — without
    # this, local runs would silently lose the N1-N9 cascade metrics
    # whenever audited gold / per-task annotations are present.
    #
    # Codex r9: ``aggregate_cascading_phase1`` walks EVERY subdir under
    # ``rows_dir`` to compute ``n_dual_eval_tasks``. Reusing the same
    # ``output_dir`` for a fresh run (different ``--limit`` /
    # ``--instance-id`` subset) would otherwise carry forward stale
    # annotations from the prior pass, inflating the denominator and
    # rewriting ``phase1_count`` / ``phase1_rate`` from the union of
    # old + new. Wipe per-instance subdirs that THIS run is about to
    # touch (or the whole rows dir when no filter is set) so the
    # aggregator only sees fresh annotations. Mirrors the round-2
    # regrade.py reset pattern.
    rows_dir = output_dir / "rows"
    if rows_dir.exists():
        if filter_ids is None:
            # Full run — wipe everything.
            shutil.rmtree(rows_dir, ignore_errors=True)
        else:
            # Filtered run — reset ONLY the subdirs this run will
            # overwrite, so unrelated instances from a prior pass
            # survive (and still contribute to the cascade block).
            _wanted = {str(t.get("instance_id") or "") for t in tasks}
            for sub in list(rows_dir.iterdir()):
                if sub.is_dir() and sub.name in _wanted:
                    shutil.rmtree(sub, ignore_errors=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    _benchmark_canonical = b.name

    def _grade_local_row(td: dict, r: dict) -> None:
        """Persist the per-row ``submission_annotation.json`` mirroring
        cloud's ``_grade_one_submission``. Best-effort: a grader raise
        on one instance must NOT take down the whole run loop — the row
        was already inserted into the results DB by ``_persist``.

        For tasks the grader can't run on (no ``submitted_sql``, agent
        crashed before submit, grader itself raised) we still write a
        ``fail-everything`` annotation so the aggregator's denominator
        (``cascading_phase1.n_dual_eval_tasks``) stays at
        ``len(tasks)`` and the reported rates aren't inflated by
        silently dropped rows.
        """
        instance_id_for_log = td.get("instance_id", "<unknown>")
        submitted_sql = r.get("submitted_sql")
        selected_database = (
            r.get("database") or td.get("selected_database") or ""
        )
        usage_blob = r.get("usage") or {}
        common_failed_kwargs = dict(
            rows_dir=rows_dir,
            instance_id=instance_id_for_log,
            selected_database=selected_database or "<unknown>",
            benchmark=_benchmark_canonical,
            run_id=run_id,
            trajectory_path=f"rows/{instance_id_for_log}/attempt-1.json",
            duration_s=r.get("duration_s"),
            n_agent_turns=usage_blob.get("n_agent_turns"),
            n_ask_user_calls=usage_blob.get("n_ask_user_calls"),
            predicted_row_count=r.get("predicted_row_count"),
        )
        if not submitted_sql or not selected_database:
            write_failed_submission_annotation(
                **common_failed_kwargs,
                failure_details=(
                    "no submitted_sql / selected_database — task "
                    "errored before reaching submit; counted as 0-pass "
                    "row at every cascade tier."
                ),
            )
            return
        # For postgres, db_path is used only as a db-name carrier
        # (executor uses db_path.stem). For SQLite, root the path at
        # the caller-provided data_dir — NOT paths.benchmark_data_root
        # — so alternate checkouts / fixtures / BIRD_DB_PATH overrides
        # don't cause agent/grader disagreement (Codex r7).
        if getattr(b, "db_backend", "sqlite") == "postgres":
            per_task_db = Path(selected_database)
        else:
            per_task_db = (
                Path(data_dir)
                / selected_database
                / f"{selected_database}.sqlite"
            )
        try:
            grade_one_submission(
                task_data=td,
                submitted_sql=submitted_sql,
                rows_dir=rows_dir,
                run_id=run_id,
                benchmark=_benchmark_canonical,
                db_path=per_task_db,
                duration_s=r.get("duration_s"),
                n_agent_turns=usage_blob.get("n_agent_turns"),
                n_ask_user_calls=usage_blob.get("n_ask_user_calls"),
                predicted_row_count=r.get("predicted_row_count"),
            )
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            logger.exception(
                "inline grader raised on instance=%s; writing "
                "fail-everything annotation so the cascade denominator "
                "stays honest",
                instance_id_for_log,
            )
            write_failed_submission_annotation(
                **common_failed_kwargs,
                failure_details=(
                    f"inline grader raised: {type(exc).__name__}: {exc}"
                )[:200],
            )

    async def _run_with_sem(i: int, td: dict) -> None:
        nonlocal total_reward, p1_count, p2_count
        async with semaphore:
            instance_id = td["instance_id"]
            logger.info("Task %d/%d: %s", i + 1, len(tasks), instance_id)
            started_at = time.time()
            t_start = time.perf_counter()
            try:
                r = await runner(td, data_dir, patience, user_sim_model)
            except Exception as e:
                logger.error("Error on %s: %s", instance_id, e)
                r = finalize_result_row(
                    {
                        "task_id": instance_id,
                        "instance_id": instance_id,
                        "database": td.get("selected_database", ""),
                        "phase1_passed": False,
                        "phase2_passed": False,
                        "total_reward": 0.0,
                        "trajectory": [],
                        "error": str(e),
                    },
                    deleted_kb_ids=[],
                    slayer_storage_dir="",
                )
            r["duration_s"] = time.perf_counter() - t_start
            _persist(td, r, started_at)
            results.append(r)
            total_reward += r.get("total_reward", 0)
            if r.get("phase1_passed"):
                p1_count += 1
            if r.get("phase2_passed"):
                p2_count += 1
            _grade_local_row(td, r)

    try:
        await asyncio.gather(*[_run_with_sem(i, td) for i, td in enumerate(tasks)])
    finally:
        db_conn.close()

    # Sum per-task usage blocks into a top-level total. Any task missing
    # `usage` (e.g. oracle pre-instrumentation) is skipped without error.
    total_usage = TokenUsage()
    for r in results:
        u_blob = r.get("usage")
        if u_blob is not None:
            total_usage.merge(TokenUsage.model_validate(u_blob))

    durations = [float(r.get("duration_s") or 0.0) for r in results]
    timing = {
        "total_duration_s": sum(durations),
        "avg_duration_s": (sum(durations) / len(durations)) if durations else 0.0,
        "p50_duration_s": statistics.median(durations) if durations else 0.0,
        "max_duration_s": max(durations) if durations else 0.0,
    }

    # Build metrics
    n = len(tasks)

    # DEV-1515: the legacy dual-eval breakdown (`phase1_count_audited`,
    # `phase1_count_original`, `n_dual_eval_tasks`, the two rates) has
    # been REPLACED by the cascading_phase1 block. The block is computed
    # downstream by `emit_cascading_eval_json` over per-row
    # submission_annotation.json files. `phase1_count` / `phase1_rate`
    # stay as back-compat aliases for N1.
    metrics = {
        "mode": mode,
        "query_mode": query_mode,
        "framework": framework,
        "total_tasks": n,
        "phase1_count": p1_count,
        "phase1_rate": p1_count / n if n else 0,
        "phase2_count": p2_count,
        "phase2_rate": p2_count / n if n else 0,
        "total_reward": total_reward,
        "average_reward": total_reward / n if n else 0,
        "total_usage": total_usage.model_dump(),
        **timing,
        "results": results,
    }

    # Save. If a local-mode rows tree carrying per-row
    # ``submission_annotation.json`` files exists alongside the eval
    # output (cloud convention: ``<output_dir>/rows/<inst>/``), enrich
    # eval.json with the freshly-aggregated ``cascading_phase1`` block
    # so the headline N1..N9 metrics aren't silently lost when local
    # runs DO have annotations (e.g. via ``grade_in_place.grade_and_write``
    # or the convert scripts). Local runs without that tree keep the
    # documented behaviour: omit the block, ship only the N1 aliases.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    rows_dir = Path(output_path).parent / "rows"
    if rows_dir.exists() and any(
        (sub / "submission_annotation.json").exists()
        for sub in rows_dir.iterdir() if sub.is_dir()
    ):
        # Codex r11: scope the cascade aggregation to the CURRENT run's
        # instance set. Filtered reruns preserve unrelated prior
        # annotations on disk (round 10 design), but the published
        # ``eval.json`` must describe ONLY the current run's row set —
        # otherwise ``cascading_phase1.n_dual_eval_tasks`` (union) would
        # exceed ``eval.total_tasks`` (filtered count) and the rewritten
        # ``phase1_count`` / ``phase1_rate`` would become uninterpretable.
        _current_iids = {
            str(td.get("instance_id") or "") for td in tasks
        } - {""}
        metrics = emit_cascading_eval_json(
            rows_dir, Path(output_path), base_metrics=metrics,
            instance_filter=_current_iids,
        )

    logger.info(
        "Done. Tasks: %d, P1: %d/%d (%.1f%%), Avg Reward: %.4f",
        n, p1_count, n, (p1_count / n * 100) if n else 0,
        (total_reward / n) if n else 0,
    )
    return metrics


def _apply_price_overrides(path: str) -> None:
    """Merge a JSON price-overrides file into litellm's built-in pricing
    table. Entries are `{"name": str, "input_per_m": float, "output_per_m": float}`.

    Per-million → per-token conversion happens here.
    """
    import litellm

    with open(path) as f:
        entries = json.load(f)

    for e in entries:
        litellm.model_cost[e["name"]] = {
            "input_cost_per_token": float(e["input_per_m"]) / 1_000_000,
            "output_cost_per_token": float(e["output_per_m"]) / 1_000_000,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BIRD-Interact benchmark runner with pluggable agents"
    )
    parser.add_argument(
        "--framework",
        choices=[
            "claude_sdk", "claude_sdk_otf", "claude_sdk_otf_ainteract",
            "pydantic_ai",
            "pydantic_ai_recursive", "pydantic_ai_otf_encode",
            "mcp_agent", "agno", "smolagents",
        ],
        default="claude_sdk",
        help="Agent framework to use",
    )
    parser.add_argument(
        "--mode",
        choices=["a-interact", "c-interact", "oracle", "one-shot"],
        default="a-interact",
        help=(
            "Evaluation mode. ``one-shot`` (DEV-1462) is the non-interactive "
            "path used by --dataset livesqlbench: no user-sim, no ask_user."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=cli_dataset_tokens(),
        required=True,
        help=(
            "Which benchmark to load (from the benchmark registry). "
            "REQUIRED — no default, to prevent silently running the wrong "
            "benchmark when --mode/--instance-ids happen to be consistent "
            "with both. Pick ``mini_interact`` (``mini-interact`` accepted "
            "as an alias) or ``livesqlbench``. ``livesqlbench`` loads "
            "LiveSQLBench-Base-Lite-SQLite and REQUIRES --gold-file; gated to "
            "--mode {one-shot, oracle}."
        ),
    )
    parser.add_argument(
        "--gold-file",
        default=None,
        help=(
            "Path to the gated LiveSQLBench gold sidecar "
            "(`*_gt_kg_testcases_*.jsonl`) — required when "
            "--dataset livesqlbench. Carries sol_sql / external_knowledge / "
            "test_cases keyed by instance_id."
        ),
    )
    parser.add_argument(
        "--query-mode",
        choices=["slayer", "raw"],
        default="raw",
        help="Query mode: slayer (semantic layer) or raw (direct SQL)",
    )
    parser.add_argument(
        "--data", required=True, help="Path to mini_interact.jsonl"
    )
    parser.add_argument(
        "--db-path", required=True, help="Path to mini-interact/ with SQLite DBs"
    )
    parser.add_argument(
        "--output",
        default=str(paths.results_root() / "eval.json"),
        help="Output JSON path (default: <main_checkout>/results/eval.json)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3, help="User patience budget")
    parser.add_argument(
        "--agent-model",
        default="anthropic/claude-sonnet-4-5",
        help=(
            "LiteLLM-style PROVIDER/MODEL_ID for the system agent. "
            "Examples: cerebras/zai-glm-4.7, openrouter/z-ai/glm-4.7-flash, "
            "anthropic/claude-sonnet-4-5, fireworks_ai/glm-4p7. The matching "
            "API-key env var (CEREBRAS_API_KEY, OPENROUTER_API_KEY, "
            "ANTHROPIC_API_KEY, FIREWORKS_API_KEY) must be set. The "
            "claude_sdk, claude_sdk_otf, and claude_sdk_otf_ainteract "
            "frameworks are locked to Anthropic and will skip with a "
            "warning if given a non-Anthropic model."
        ),
    )
    parser.add_argument(
        "--user-sim-model",
        default="anthropic/claude-haiku-4-5-20251001",
        help="LiteLLM model for user simulator",
    )
    parser.add_argument(
        "--slayer-storage-root",
        default="./slayer_storage",
        help="Root dir of per-DB SLayer model stores (only used in --query-mode slayer)",
    )
    parser.add_argument(
        "--filter-ids",
        default=None,
        help=(
            "Path to a text file with one instance_id per line; only tasks "
            "with these IDs are evaluated. Use to align with the original "
            "harness in 3-way comparison runs."
        ),
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help=(
            "Run a single task by its instance_id, or a comma-separated "
            "list of instance_ids (e.g. `households_5,households_6`). "
            "Mutually exclusive with --filter-ids and --limit. Use for "
            "one-shot debugging with full transcript capture."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force every tool definition to carry strict=True (OpenAI "
            "strict structured-output mode). Default False matches the "
            "non-strict, non-constrained-decoding behaviour of all "
            "frameworks. claude_sdk, claude_sdk_otf, and "
            "claude_sdk_otf_ainteract silently ignore the flag (Anthropic "
            "has no tool-level strict). mcp_agent doesn't expose a hook "
            "for it and exits with a clear error when --strict is given."
        ),
    )
    parser.add_argument(
        "--price-overrides",
        default=None,
        help=(
            "Optional JSON file with price overrides merged into litellm's "
            "built-in pricing table. Format: a list of "
            '{"name": "<model>", "input_per_m": <float>, '
            '"output_per_m": <float>} entries.'
        ),
    )
    parser.add_argument(
        "--use-audited-gold-sql",
        action="store_true",
        default=False,
        help=(
            "Swap each task's gold sol_sql for the audited version from "
            "audited_gold/<db>/<db>_audited.jsonl when available (status "
            "in {edited, unrecoverable}). Tasks marked 'clean' or missing "
            "from the sidecar use the original gold. The overlay log is "
            "written to <output_dir>/audited_gold_overlay.json."
        ),
    )
    parser.add_argument(
        "--no-prompt-cache",
        action="store_false",
        dest="prompt_cache",
        default=True,
        help=(
            "Disable Anthropic prompt caching on the agent (pydantic_ai "
            "framework only). Default is enabled: the agent's system "
            "prompt and tool definitions are sent with cache_control, "
            "and subsequent tool-call round-trips within the 5-minute "
            "TTL re-read from cache at 10%% of input cost. Doesn't "
            "affect the user-sim, which always uses its own model."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help=(
            "Max recursion depth for pydantic_ai_recursive's clarifier "
            "tree (ignored by other frameworks). Increase if you expect "
            "deeply compound user replies; decrease to cap LLM-call fanout."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help=(
            "Reasoning-effort level for the claude_sdk_otf and "
            "claude_sdk_otf_ainteract agents (maps to the Claude Agent SDK's "
            "ClaudeAgentOptions.effort). Ignored by other frameworks. Unset "
            "uses the SDK default."
        ),
    )
    parser.add_argument(
        "--otf-rebuild",
        dest="otf_rebuild",
        action="store_true",
        default=False,
        help=(
            "On-the-fly slayer setup only. Force-wipe BOTH on-the-fly layers "
            "(the phase-1-3 cache AND the KB-encoded reference) for each run "
            "DB ONCE before the task loop, so they rebuild from scratch. "
            "Default OFF reuses whatever is present (a changed KB/schema/DB is "
            "NOT auto-detected — reingest is explicit via this flag)."
        ),
    )
    parser.add_argument(
        # Backwards-compatible hidden alias for git/script continuity.
        "--otf-rebuild-reference",
        dest="otf_rebuild",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--slayer-setup",
        choices=["pre-encoded", "on-the-fly"],
        default="pre-encoded",
        help=(
            "How SLayer storage is provisioned for each task. "
            "'pre-encoded' (default) uses the committed slayer_models/ "
            "as today. 'on-the-fly' (DEV-1455) ingests the relevant DB "
            "into SLayer at task setup time and encodes each KB item "
            "as a SLayer memory, preserving cross-references as "
            "memory:<id> entity tokens. Valid with --query-mode slayer "
            "and --framework pydantic_ai_recursive, pydantic_ai_otf_encode, "
            "claude_sdk_otf, or claude_sdk_otf_ainteract, under --mode "
            "a-interact or one-shot. (pydantic_ai_otf_encode, "
            "claude_sdk_otf, and claude_sdk_otf_ainteract REQUIRE "
            "on-the-fly.)"
        ),
    )
    args = parser.parse_args()

    # Resolve --db-path to an absolute path ONCE at the CLI boundary. Every
    # downstream consumer (orchestrator ingest, on-the-fly cache/reference,
    # per-task DB materialisation) then receives an absolute root, so a
    # relative `--db-path ../livesqlbench-base-lite-sqlite/` (the README form)
    # cannot produce a broken `sqlite:////../…` connection string.
    args.db_path = str(Path(args.db_path).resolve())

    if args.instance_id is not None and (args.filter_ids or args.limit is not None):
        parser.error("--instance-id cannot be combined with --filter-ids or --limit")

    # Normalize the benchmark token to its canonical underscore form once, so
    # every downstream consumer (gates, loader dispatch, path roots) sees the
    # canonical name regardless of which alias the user typed.
    args.dataset = get_benchmark(args.dataset).name

    # Fail-fast: --slayer-setup on-the-fly is only valid for the
    # pydantic_ai_recursive + slayer + a-interact|one-shot tuple. Validate
    # before any task starts so the user doesn't get a results.db full of
    # bogus rows (Codex finding on DEV-1455). The same check lives in
    # ``run_evaluation`` for programmatic callers; here we translate its
    # ValueError into argparse's standard stderr + exit-2 path.
    try:
        # DEV-1462: dataset ⟺ mode gates + one-shot dispatch + gold-file
        # presence — same fail-fast pattern (CLI + programmatic mirror in
        # run_evaluation). The one-shot dispatch fires FIRST so a one-shot
        # + wrong-framework surfaces a "--mode one-shot requires …" error,
        # not the more generic on-the-fly-framework error.
        _validate_dataset_mode(dataset=args.dataset, mode=args.mode)
        # DEV-1507: framework-bound (dataset, mode) gate. Fires right after
        # `_validate_dataset_mode` so the error message names the framework
        # binding, not a downstream mode/slayer-setup gate.
        _validate_framework_dataset_mode(
            framework=args.framework, dataset=args.dataset, mode=args.mode,
        )
        _validate_one_shot_framework(
            mode=args.mode, query_mode=args.query_mode,
            framework=args.framework,
        )
        _validate_slayer_setup(
            slayer_setup=args.slayer_setup,
            framework=args.framework,
            query_mode=args.query_mode,
            mode=args.mode,
        )
        _b = get_benchmark(args.dataset)
        if _b.gold_required and not args.gold_file:
            raise ValueError(
                f"--dataset {_b.name} requires --gold-file (the gated "
                "sidecar carrying sol_sql / external_knowledge / test_cases "
                "keyed by instance_id).",
            )
    except ValueError as e:
        parser.error(str(e))

    if args.price_overrides:
        _apply_price_overrides(args.price_overrides)

    filter_ids: list[str] | None = None
    if args.instance_id:
        # Accept either a single id or a comma-separated list. Whitespace
        # around items is trimmed; empty tokens are dropped.
        filter_ids = [s.strip() for s in args.instance_id.split(",") if s.strip()]
        # Reject input that parses to an empty list (e.g. ",,, "). Without
        # this, filter_ids=[] later falls back to running the full
        # benchmark — a silent expansion of scope from a malformed flag.
        if not filter_ids:
            parser.error(
                "--instance-id must include at least one non-empty id",
            )
    elif args.filter_ids:
        with open(args.filter_ids) as f:
            filter_ids = [line.strip() for line in f if line.strip()]

    asyncio.run(
        run_evaluation(
            data_path=args.data,
            data_dir=args.db_path,
            output_path=args.output,
            mode=args.mode,
            query_mode=args.query_mode,
            framework=args.framework,
            limit=args.limit,
            concurrency=args.concurrency,
            patience=args.patience,
            user_sim_model=args.user_sim_model,
            slayer_storage_root=args.slayer_storage_root,
            filter_ids=filter_ids,
            agent_model=args.agent_model,
            strict=args.strict,
            use_audited_gold_sql=args.use_audited_gold_sql,
            prompt_cache=args.prompt_cache,
            max_depth=args.max_depth,
            slayer_setup=args.slayer_setup,
            otf_rebuild=args.otf_rebuild,
            dataset=args.dataset,
            gold_file=args.gold_file,
            reasoning_effort=args.reasoning_effort,
        )
    )


if __name__ == "__main__":
    main()
