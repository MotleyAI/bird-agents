"""CLI entry point for running BIRD-Interact evaluations."""

import argparse
import asyncio
import json
import logging
import statistics
import time
from pathlib import Path

import sqlite3 as _sqlite3
from typing import Any as _Any

from bird_interact_agents import paths
from bird_interact_agents.benchmark import cli_dataset_tokens, get_benchmark
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
    """Read task_results from a results.db and emit aggregate dual-eval
    metrics. Used by run.py at end-of-run and by tests for round-trip
    verification. Returns a dict with keys: ``phase1_count``,
    ``phase1_rate``, ``phase1_count_audited``, ``phase1_count_original``,
    ``phase1_rate_audited``, ``phase1_rate_original``, ``n_dual_eval_tasks``,
    ``total_tasks``."""
    conn = _sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT phase1_passed, phase1_passed_audited, phase1_passed_original "
            "FROM task_results"
        ).fetchall()
    finally:
        conn.close()
    n = len(rows)
    p1 = sum(1 for r in rows if r[0])
    dual_rows = [r for r in rows if r[1] is not None]
    n_dual = len(dual_rows)
    p1_aud = sum(1 for r in dual_rows if r[1])
    p1_orig = sum(1 for r in dual_rows if r[2])
    return {
        "total_tasks": n,
        "phase1_count": p1,
        "phase1_rate": p1 / n if n else 0.0,
        "n_dual_eval_tasks": n_dual,
        "phase1_count_audited": p1_aud,
        "phase1_count_original": p1_orig,
        "phase1_rate_audited": p1_aud / n_dual if n_dual else 0.0,
        "phase1_rate_original": p1_orig / n_dual if n_dual else 0.0,
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
    # pydantic_ai_otf_encode is an on-the-fly-only adapter (DEV-1454).
    if framework == "pydantic_ai_otf_encode" and slayer_setup != "on-the-fly":
        raise ValueError(
            "--framework pydantic_ai_otf_encode requires "
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
    if framework not in ("pydantic_ai_recursive", "pydantic_ai_otf_encode"):
        raise ValueError(
            "--slayer-setup on-the-fly is only supported with "
            "--framework pydantic_ai_recursive or "
            "--framework pydantic_ai_otf_encode; "
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
    """DEV-1462: one-shot dispatch is recursive + otf_encode only.

    ``oracle`` stays framework-agnostic; ``a-interact``/``c-interact`` keep
    their existing per-framework dispatch.
    """
    if mode != "one-shot":
        return
    if query_mode != "slayer":
        raise ValueError(
            "--mode one-shot requires --query-mode slayer; "
            f"got --query-mode {query_mode!r}",
        )
    if framework not in ("pydantic_ai_recursive", "pydantic_ai_otf_encode"):
        raise ValueError(
            "--mode one-shot is only supported with --framework "
            "pydantic_ai_recursive or --framework pydantic_ai_otf_encode; "
            f"got --framework {framework!r}",
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
    if framework not in ("pydantic_ai_recursive", "pydantic_ai_otf_encode"):
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
        slayer_setup=slayer_setup,
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
) -> dict:
    """Run full evaluation across all tasks."""
    # Programmatic-caller mirror of the CLI fail-fast guards. The CLI
    # parser rejects the same combinations in main(), but
    # ``run_evaluation`` is also called directly from tests and other
    # entry points; without these checks those callers would silently get
    # unsupported behaviour (Codex finding on DEV-1455 PR #19 +
    # DEV-1462 Codex round-2).
    _validate_dataset_mode(dataset=dataset, mode=mode)
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

    # The audited-gold overlay is a mini-interact concept: gold inline in the
    # data JSONL + a separate audited_gold/<db> sidecar. A gold_required
    # benchmark (livesqlbench) carries its gold in the gated sidecar instead
    # and has no audited_gold/, so skip the overlay there — mirrors the cloud
    # `_load_task_data` gate and avoids overlaying an unrelated audited_gold/<db>
    # row onto the gated gold on an instance_id clash (Codex).
    audited_overlay_log: dict[str, str] = {}
    if use_audited_gold_sql and not b.gold_required:
        audited_overlay_log = apply_audited_gold_overlay(
            tasks, paths.audited_gold_root(),
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
            phase1_passed_audited=r.get("phase1_passed_audited"),
            phase1_passed_original=r.get("phase1_passed_original"),
            phase1_observation_audited=r.get("phase1_observation_audited"),
            phase1_observation_original=r.get("phase1_observation_original"),
        ))

    # Run tasks with concurrency limiter
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    total_reward = 0.0
    p1_count = 0
    p2_count = 0

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

    # Dual-evaluation counts (NULL on single-eval runs — only populated
    # when the overlay applied AND evaluate_dual_gold ran). Counting
    # over the in-memory results list so we don't have to round-trip
    # through the DB just for aggregation.
    dual_audited = [r.get("phase1_passed_audited") for r in results]
    dual_original = [r.get("phase1_passed_original") for r in results]
    n_dual = sum(1 for x in dual_audited if x is not None)
    p1_audited = sum(1 for x in dual_audited if x)
    p1_original = sum(1 for x in dual_original if x)

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
        # Dual-eval breakdown (only meaningful when --use-audited-gold-sql
        # is on; equal to the single-eval counts otherwise).
        "n_dual_eval_tasks": n_dual,
        "phase1_count_audited": p1_audited,
        "phase1_count_original": p1_original,
        "phase1_rate_audited": p1_audited / n_dual if n_dual else 0,
        "phase1_rate_original": p1_original / n_dual if n_dual else 0,
        "results": results,
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

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
            "claude_sdk", "pydantic_ai", "pydantic_ai_recursive",
            "pydantic_ai_otf_encode",
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
        default="mini_interact",
        help=(
            "Which benchmark to load (from the benchmark registry). "
            "``mini_interact`` (default; ``mini-interact`` accepted as an "
            "alias) keeps the existing behaviour. ``livesqlbench`` loads "
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
            "claude_sdk framework is locked to Anthropic and will skip "
            "with a warning if given a non-Anthropic model."
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
            "frameworks. claude_sdk silently ignores the flag (Anthropic "
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
            "memory:<id> entity tokens. Only valid with "
            "--framework pydantic_ai_recursive --query-mode slayer "
            "--mode a-interact."
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
        )
    )


if __name__ == "__main__":
    main()
