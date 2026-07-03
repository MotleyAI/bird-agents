"""CLI entry point for running BIRD-Interact evaluations."""

import argparse
import asyncio
import datetime
import json
import logging
import os
import shutil
import statistics
import time
from pathlib import Path

import sqlite3 as _sqlite3
from typing import Any as _Any

from bird_interact_agents import paths
# DEV-1638: the local postgres bootstrap + annotation sync + dotenv loader moved
# into the package so this installed console script can compose them. Imported
# at module top level so tests monkeypatch `run.<name>`.
from bird_interact_agents.env_file import load_env_file
from bird_interact_agents.local_annotations import sync_annotations
from bird_interact_agents.local_postgres import DEFAULT_PORT, provision_and_export
from bird_interact_agents.provider_registry import agent_needs_bridge, get_provider
# DEV-1604: imported as a MODULE so `_maybe_start_bridge_proxy` and its test
# monkeypatch share the `run.bridge_proxy.ensure_bridge_proxy_for_actor` target.
from bird_interact_agents.cloud import bridge_proxy
from bird_interact_agents.benchmark import cli_dataset_tokens, get_benchmark
from bird_interact_agents.eval.cascading_report import emit_cascading_eval_json
from bird_interact_agents.eval.annotation_schema import SubmissionConfig
from bird_interact_agents.eval.grade_in_place import (
    decode_result_json as _decode_result_json,
    extract_usage_costs,
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


# DEV-1586: frameworks whose agents implement the read-only pre-encoded mode
# (the SLayer claude_sdk consumers — aggregators + the four direct OTF slayer
# flavors). Raw flavors never reach the slayer branch of the validator.
_PRE_ENCODED_FRAMEWORKS = frozenset({
    "claude_sdk",
    "claude_sdk_v1",
    "claude_sdk_otf",
    "claude_sdk_otf_ainteract",
    "claude_sdk_otf_v1",
    "claude_sdk_otf_ainteract_v1",
})


def _validate_slayer_setup(
    *, slayer_setup: str, framework: str, query_mode: str, mode: str,
    pre_encoded_source: str | None = None,
) -> None:
    """Reject inconsistent ``slayer_setup`` / ``pre_encoded_source`` combos.

    DEV-1586: ``slayer_setup`` is no longer user-set — it is DERIVED from the
    user-facing ``--pre-encoded-models`` flag (``"pre-encoded"`` when a source
    is set, else ``"on-the-fly"``). This guard just enforces that the derived
    value is internally consistent with the source, so a hand-built manifest
    or a stale resubmit can't carry a contradictory pair.

    ``raw`` query_mode has no SLayer dependency, but a stray
    ``--pre-encoded-models`` there is still nonsensical and rejected (the
    flag is slayer-only). ``slayer_setup`` itself is ignored for raw.
    """
    from bird_interact_agents.agents._pre_encoded import (
        derive_slayer_setup,
        validate_pre_encoded_source,
    )

    # DEV-1609: the encoder builds the SLayer reference, so it REQUIRES
    # --query-mode slayer (agent.py guardrail). Reject raw at CLI/cloud
    # validation — BEFORE the raw early-return below — rather than failing every
    # task after setup is built/uploaded (Codex review).
    if framework == "claude_sdk_otf_encode" and query_mode != "slayer":
        raise ValueError(
            "--framework claude_sdk_otf_encode requires --query-mode slayer; "
            f"got {query_mode!r}. An encode run builds the SLayer reference."
        )

    # Always validate the source vocabulary + framework gate, BEFORE the
    # raw early-return (Codex DEV-1586 r2 #2 — otherwise `--query-mode raw
    # --framework pydantic_ai --pre-encoded-models otf` slips through).
    validate_pre_encoded_source(pre_encoded_source)
    if pre_encoded_source is not None:
        # The flag only controls the SLayer datasource source, so it is
        # meaningless in raw mode.
        if query_mode != "slayer":
            raise ValueError(
                "--pre-encoded-models is only valid with --query-mode slayer; "
                f"got --query-mode {query_mode!r}."
            )
        # pre-encoded mode is implemented only by the read-only SLayer
        # claude_sdk consumers. Routing it to the encoder
        # (pydantic_ai_otf_encode) or the recursive/plain pydantic agents
        # would mis-route cloud artifacts and silently ignore the flag.
        if framework not in _PRE_ENCODED_FRAMEWORKS:
            raise ValueError(
                f"--pre-encoded-models is only supported for the SLayer "
                f"claude_sdk frameworks {sorted(_PRE_ENCODED_FRAMEWORKS)}; got "
                f"--framework {framework!r}. Omit --pre-encoded-models to "
                "encode on the fly."
            )

    if query_mode == "raw":
        return
    expected = derive_slayer_setup(pre_encoded_source)
    if slayer_setup != expected:
        raise ValueError(
            f"--query-mode slayer with pre_encoded_source="
            f"{pre_encoded_source!r} requires slayer_setup={expected!r}; "
            f"got {slayer_setup!r}. (slayer_setup is derived from "
            "--pre-encoded-models; do not set it directly.)"
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


def _validate_framework_mode(
    *, framework: str, dataset: str, mode: str,
) -> None:
    """Validate that ``mode`` is supported by the active framework's dispatch.

    ``_validate_dataset_mode`` checks the benchmark's declared ``supported_modes``,
    but ``claude_sdk`` agents only implement a subset:
    - one-shot benchmarks → ``one-shot`` only (ClaudeSDKOtf*)
    - multi-turn benchmarks → ``a-interact`` only (ClaudeSDKOtfAInteract*)

    Without this check, ``c-interact`` or ``oracle`` with a non-one-shot
    benchmark (or ``oracle`` with a one-shot benchmark) would pass the
    dataset-mode gate but fail deep inside the agent at task runtime.
    """
    # DEV-1609: the claude_sdk_otf_encode agent builds the reference and accepts
    # ONLY a-interact / one-shot (agent.py guardrail) regardless of benchmark —
    # reject c-interact / oracle at CLI/cloud validation rather than failing
    # per-task after setup (Codex review).
    if framework == "claude_sdk_otf_encode":
        if mode not in ("a-interact", "one-shot"):
            raise ValueError(
                "--framework claude_sdk_otf_encode only supports "
                f"--mode a-interact / one-shot; got {mode!r}. An encode run "
                "builds the canonical reference; c-interact / oracle are "
                "unsupported."
            )
        return
    # DEV-1555 v0/v1: validate both aggregator tokens.
    if framework not in ("claude_sdk", "claude_sdk_v1"):
        return
    if mode == "oracle":
        return  # oracle bypasses the agent entirely (run_oracle_task)
    b = get_benchmark(dataset)
    supported = ("one-shot",) if b.one_shot else ("a-interact",)
    if mode not in supported:
        raise ValueError(
            f"--framework {framework} with {b.name!r} only supports "
            f"--mode {' / '.join(supported)}; got {mode!r}. "
            f"The modes {set(b.supported_modes) - set(supported)} listed in "
            f"the benchmark spec are not yet wired to a claude_sdk agent variant."
        )


def _maybe_force_wipe_otf(
    *, otf_rebuild: bool, framework: str, dbs,
    benchmark: str,
    pre_encoded_source: str | None = None,
) -> None:
    """``--otf-rebuild`` force-wipe: drop BOTH on-the-fly layers (the phase-1-3
    cache AND the KB-encoded reference) for ``dbs``, for either on-the-fly
    framework. No-op when the flag is off or the framework isn't on-the-fly.

    Wiping both layers together is load-bearing: wiping only the reference
    would let a stale cache be re-encoded into a "fresh" reference (Codex r2
    High#3).

    DEV-1462: ``benchmark`` (REQUIRED, explicit) selects the per-benchmark
    scoped roots so a LiveSQLBench ``--otf-rebuild`` never wipes the
    mini-interact cache (and vice versa).

    DEV-1586: NO-OP in pre-encoded mode. The on-the-fly cache/reference are
    not owned by a pre-encoded run — and for ``--pre-encoded-models otf`` the
    reference IS the thing the read-only agent consumes, so wiping it here
    would delete the input and the agent could never rebuild it. (For
    ``custom`` it would needlessly purge unrelated OTF references for the
    selected DBs.)
    """
    if not otf_rebuild:
        return
    if pre_encoded_source is not None:
        return
    # DEV-1555 v0/v1: both aggregator tokens dispatch to on-the-fly agents.
    # DEV-1609: the claude_sdk OTF encoder rebuilds the reference from scratch,
    # so `--otf-rebuild` MUST wipe both layers for it too — otherwise a stale
    # cache would be re-encoded into a "fresh" reference.
    if framework not in (
        "claude_sdk", "claude_sdk_v1", "claude_sdk_otf_encode",
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
    dataset: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    strict: bool,
    prompt_cache: bool,
    max_depth: int,
    slayer_storage_root: str | None,
    slayer_setup: str | None = None,
    reasoning_effort: str | None = None,
    user_sim_prompt_version: str | None = None,
    pre_encoded_source: str | None = None,
):
    """Public alias for `_make_runner`. The cloud actor (and other
    throughput-sensitive callers) call this once at startup and reuse the
    returned closure across tasks to avoid per-task agent reconstruction.

    ``dataset`` drives the dispatch for the ``claude_sdk`` framework
    (selects the right OTF agent flavor based on benchmark.one_shot and
    query_mode).

    ``user_sim_prompt_version`` (DEV-1545): None at the public API
    layer; `_make_runner` normalises None → "v2" before threading into
    each framework's `run_task` closure (so explicit-None does not
    shadow the agent class's Python "v2" default).
    """
    # DEV-1586: slayer_setup is a pure function of the pre-encoded source.
    # Omitted (None) ⇒ derive (on-the-fly when no source). An explicit value
    # (the cloud actor passes the manifest's derived value) is honored and
    # consistency-checked below.
    from bird_interact_agents.agents._pre_encoded import derive_slayer_setup
    if slayer_setup is None:
        slayer_setup = derive_slayer_setup(pre_encoded_source)
    _validate_slayer_setup(
        slayer_setup=slayer_setup, framework=framework,
        query_mode=query_mode, mode=mode,
        pre_encoded_source=pre_encoded_source,
    )
    return _make_runner(
        framework=framework, dataset=dataset, query_mode=query_mode, mode=mode,
        agent_model=agent_model, strict=strict, prompt_cache=prompt_cache,
        max_depth=max_depth, slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup, reasoning_effort=reasoning_effort,
        user_sim_prompt_version=user_sim_prompt_version,
        pre_encoded_source=pre_encoded_source,
    )


_DEFAULT_PER_TASK_TIMEOUT_S = 0.0
"""Per-task wall-clock cap. 0 / negative = no cap (the default).

Originally landed under DEV-1535 at 900 s after a 76-task sweep showed
every `correct` verdict at <= 891 s and every `valid_interpretation`
at <= 659 s, so a 15-min cap killed `agent_miss` thrash with a 0%
false-negative rate on the correct/valid buckets. The cap was removed
as a default after rate-limited cloud runs revealed that LLM-side
back-offs (subscription throttles, provider 429s) routinely push
otherwise-correct tasks past the cap, turning recoverable retries into
permanent `eval_failed`s. Re-enable for a specific run via the
BIRD_INTERACT_PER_TASK_TIMEOUT_S env var (set to the desired seconds)."""

# DEV-1555 follow-up: the SDK agents now enforce the same budget
# AGENT-SIDE via wall-clock hooks (warn at 80%/90%, deny non-submit
# tools at 100%). The agent-side enforcement preserves the trajectory
# because submit happens inside the SDK session; an outer
# asyncio.wait_for at the raw budget would kill the receive loop
# mid-stream and lose the trajectory (last seen on Kimi r7). We keep
# asyncio.wait_for as a RUNAWAY SAFETY NET at budget + grace so a
# model that ignores the deny still terminates eventually.
_DEFAULT_RUNAWAY_GRACE_S = 120.0


def _runaway_grace_s() -> float:
    """Override for ``_DEFAULT_RUNAWAY_GRACE_S`` via
    ``BIRD_INTERACT_RUNAWAY_GRACE_S``. Tests for the runaway path set
    this to 0 so they can exercise the outer cap in seconds."""
    raw = os.environ.get("BIRD_INTERACT_RUNAWAY_GRACE_S")
    if raw is None:
        return _DEFAULT_RUNAWAY_GRACE_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_RUNAWAY_GRACE_S


def _per_task_timeout_s() -> float:
    """Outer ``asyncio.wait_for`` cap = agent budget + runaway grace.

    The raw env-var budget IS the agent's soft target (see
    ``context_budget.per_task_timeout_s``); this function returns the
    HARD ceiling at which the outer wait_for will rip the task — used
    as the last-resort safety net for a model that ignores the
    agent-side deny."""
    raw = os.environ.get("BIRD_INTERACT_PER_TASK_TIMEOUT_S")
    grace = _runaway_grace_s()
    # Default is "no cap" (0.0). Only add the runaway grace when the
    # operator explicitly opted in to a positive cap.
    if raw is None:
        return _DEFAULT_PER_TASK_TIMEOUT_S
    try:
        agent_budget = float(raw)
    except ValueError:
        return _DEFAULT_PER_TASK_TIMEOUT_S
    if agent_budget <= 0:
        return agent_budget  # 0 / negative still disables both caps
    return agent_budget + grace


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
    timeout = _per_task_timeout_s()
    try:
        coro = runner(task_data, data_dir, patience, user_sim_model)
        if timeout > 0:
            try:
                r = await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError as te:
                # asyncio.TimeoutError carries no message; raise a clearer
                # one so the error row records what actually happened
                # (DEV-1535 wall-clock cap).
                raise TimeoutError(
                    f"per-task runaway-grace ceiling of {timeout:.0f}s exceeded — the "
                    f"agent ignored the wall-clock budget hook's deny "
                    f"(BIRD_INTERACT_PER_TASK_TIMEOUT_S=0 to disable)"
                ) from te
        else:
            r = await coro
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
    dataset: str = "",
    query_mode: str,
    mode: str,
    agent_model: str,
    strict: bool,
    prompt_cache: bool,
    max_depth: int,
    slayer_storage_root: str | None,
    slayer_setup: str | None = None,
    reasoning_effort: str | None = None,
    user_sim_prompt_version: str | None = None,
    pre_encoded_source: str | None = None,
):
    """Construct the per-task runner closure for the given config.

    Returns an `async (task_data, data_dir, patience, user_sim_model) -> dict`
    callable. The agent (if any) is constructed *once* and captured in the
    closure — callers that want one-shot semantics should call this factory
    per task. Callers that need throughput (`run_evaluation`, in-cluster
    actor) call it once and reuse the closure.

    DEV-1545: ``user_sim_prompt_version`` flows from the CLI through
    ``cloud/cli.py`` → manifest → ``ray_app.run_pool`` → here. The CLI
    flag defaults to None to keep manifest serialisation
    unambiguous (None vs "v2"); we normalise None → "v2" exactly ONCE
    here so all 11+ closures below thread the same string into their
    agents' ``run_task``. Passing explicit None would shadow the agent
    class's Python "v2" default (Python defaults only apply to omitted
    args, not explicit-None) — leading to a KeyError on
    ``USER_SIMULATOR_ENCODER[None]`` in the user-sim invocation site.
    """
    _v = user_sim_prompt_version or "v2"
    # DEV-1586: derive slayer_setup from the pre-encoded source when omitted
    # (None), so direct callers get the same "omitted ⇒ on-the-fly" default
    # as the CLIs / make_runner.
    if slayer_setup is None:
        from bird_interact_agents.agents._pre_encoded import derive_slayer_setup
        slayer_setup = derive_slayer_setup(pre_encoded_source)
    if mode == "oracle":
        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            return await run_oracle_task(td, data_dir)
        return run_one
    if framework == "claude_sdk":
        b = get_benchmark(dataset)
        if strict:
            logger.warning(
                "[claude_sdk] --strict is a no-op for Anthropic models; ignored."
            )
        # DEV-1579: the v0 claude_sdk agents now carry the provider-aware
        # hermetic session env (registry base-url + auth + thinking) via
        # `hermetic_claude_sdk_session`, so registry open-weight models
        # (moonshot/, zai/, …) run on v0 too — no Anthropic-only rejection
        # here anymore.
        if b.one_shot and query_mode == "slayer":
            from bird_interact_agents.agents.claude_sdk_otf import ClaudeSDKOtfAgent
            _agent: object = ClaudeSDKOtfAgent(
                slayer_storage_root=slayer_storage_root,
                model=agent_model,
                slayer_setup=slayer_setup,
                pre_encoded_source=pre_encoded_source,
                reasoning_effort=reasoning_effort,
            )
        elif not b.one_shot and query_mode == "slayer":
            from bird_interact_agents.agents.claude_sdk_otf_ainteract import (
                ClaudeSDKOtfAInteractAgent,
            )
            _agent = ClaudeSDKOtfAInteractAgent(
                slayer_storage_root=slayer_storage_root,
                model=agent_model,
                slayer_setup=slayer_setup,
                pre_encoded_source=pre_encoded_source,
                reasoning_effort=reasoning_effort,
            )
        elif b.one_shot and query_mode == "raw":
            from bird_interact_agents.agents.claude_sdk_otf_raw import ClaudeSDKOtfRawAgent
            _agent = ClaudeSDKOtfRawAgent(
                model=agent_model,
                reasoning_effort=reasoning_effort,
            )
        else:  # not one_shot, raw
            from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import (
                ClaudeSDKOtfAInteractRawAgent,
            )
            _agent = ClaudeSDKOtfAInteractRawAgent(
                model=agent_model,
                reasoning_effort=reasoning_effort,
            )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await _agent.run_task(  # type: ignore[attr-defined]
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
                user_sim_prompt_version=_v,
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
            pre_encoded_source=pre_encoded_source,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_cso.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
                user_sim_prompt_version=_v,
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
            pre_encoded_source=pre_encoded_source,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csoa.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
                user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_raw":
        from bird_interact_agents.agents.claude_sdk_otf_raw import ClaudeSDKOtfRawAgent

        if strict:
            logger.warning(
                "[claude_sdk_otf_raw] --strict is a no-op for Anthropic models; "
                "ignored."
            )
        agent_csor = ClaudeSDKOtfRawAgent(
            model=agent_model,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csor.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
                user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_ainteract_raw":
        from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import (
            ClaudeSDKOtfAInteractRawAgent,
        )

        if strict:
            logger.warning(
                "[claude_sdk_otf_ainteract_raw] --strict is a no-op for "
                "Anthropic models; ignored."
            )
        agent_csoar = ClaudeSDKOtfAInteractRawAgent(
            model=agent_model,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csoar.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
                user_sim_prompt_version=_v,
            )
        return run_one
    # ---------- v1 = this branch's shape, opt-in via `_v1` tokens ----------
    if framework == "claude_sdk_v1":
        b = get_benchmark(dataset)
        if strict:
            logger.warning(
                "[claude_sdk_v1] --strict is a no-op for Anthropic models; ignored."
            )
        if b.one_shot and query_mode == "slayer":
            from bird_interact_agents.agents.claude_sdk_otf_v1 import (
                ClaudeSDKOtfAgent,
            )
            _agent_v1: object = ClaudeSDKOtfAgent(
                slayer_storage_root=slayer_storage_root,
                model=agent_model,
                slayer_setup=slayer_setup,
                pre_encoded_source=pre_encoded_source,
                reasoning_effort=reasoning_effort,
            )
        elif not b.one_shot and query_mode == "slayer":
            from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import (
                ClaudeSDKOtfAInteractAgent,
            )
            _agent_v1 = ClaudeSDKOtfAInteractAgent(
                slayer_storage_root=slayer_storage_root,
                model=agent_model,
                slayer_setup=slayer_setup,
                pre_encoded_source=pre_encoded_source,
                reasoning_effort=reasoning_effort,
            )
        elif b.one_shot and query_mode == "raw":
            from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import (
                ClaudeSDKOtfRawAgent,
            )
            _agent_v1 = ClaudeSDKOtfRawAgent(
                model=agent_model,
                reasoning_effort=reasoning_effort,
            )
        else:  # not one_shot, raw
            from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1 import (
                ClaudeSDKOtfAInteractRawAgent,
            )
            _agent_v1 = ClaudeSDKOtfAInteractRawAgent(
                model=agent_model,
                reasoning_effort=reasoning_effort,
            )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await _agent_v1.run_task(  # type: ignore[attr-defined]
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_v1":
        from bird_interact_agents.agents.claude_sdk_otf_v1 import ClaudeSDKOtfAgent

        if strict:
            logger.warning(
                "[claude_sdk_otf_v1] --strict is a no-op for Anthropic models; "
                "ignored."
            )
        agent_cso_v1 = ClaudeSDKOtfAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            slayer_setup=slayer_setup,
            pre_encoded_source=pre_encoded_source,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_cso_v1.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_ainteract_v1":
        from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import (
            ClaudeSDKOtfAInteractAgent,
        )

        if strict:
            logger.warning(
                "[claude_sdk_otf_ainteract_v1] --strict is a no-op for "
                "Anthropic models; ignored."
            )
        agent_csoa_v1 = ClaudeSDKOtfAInteractAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            slayer_setup=slayer_setup,
            pre_encoded_source=pre_encoded_source,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csoa_v1.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_raw_v1":
        from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import (
            ClaudeSDKOtfRawAgent,
        )

        if strict:
            logger.warning(
                "[claude_sdk_otf_raw_v1] --strict is a no-op for Anthropic "
                "models; ignored."
            )
        agent_csor_v1 = ClaudeSDKOtfRawAgent(
            model=agent_model,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csor_v1.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_ainteract_raw_v1":
        from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1 import (
            ClaudeSDKOtfAInteractRawAgent,
        )

        if strict:
            logger.warning(
                "[claude_sdk_otf_ainteract_raw_v1] --strict is a no-op for "
                "Anthropic models; ignored."
            )
        agent_csoar_v1 = ClaudeSDKOtfAInteractRawAgent(
            model=agent_model,
            reasoning_effort=reasoning_effort,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csoar_v1.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
            user_sim_prompt_version=_v,
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
                user_sim_prompt_version=_v,
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
                user_sim_prompt_version=_v,
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
                user_sim_prompt_version=_v,
            )
        return run_one
    if framework == "claude_sdk_otf_encode":
        # DEV-1609: the default OTF *reference* encoder. Build-only — it
        # constructs the claude_sdk build-encoder (DEV-1589) and builds the
        # canonical per-DB reference via `ensure_db_reference`, which the
        # cloud merge-back uploads home. No per-task masking / eval loop.
        from bird_interact_agents.agents.claude_sdk_otf_encode import (
            ClaudeSDKOtfEncodeAgent,
        )

        if strict:
            logger.warning(
                "[claude_sdk_otf_encode] --strict is unsupported; ignored.",
            )
        agent_csenc = ClaudeSDKOtfEncodeAgent(
            slayer_storage_root=slayer_storage_root,
            model=agent_model,
            reasoning_effort=reasoning_effort,
            slayer_setup=slayer_setup,
        )

        async def run_one(td: dict, data_dir: str, patience: int,
                          user_sim_model: str) -> dict:
            budget = calculate_budget(td, patience, mode=mode)
            return await agent_csenc.run_task(
                td, data_dir, budget, query_mode,
                eval_mode=mode,
                user_sim_model=user_sim_model,
                user_sim_prompt_version=_v,
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
                user_sim_prompt_version=_v,
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
                user_sim_prompt_version=_v,
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
                user_sim_prompt_version=_v,
            )
        return run_one
    raise ValueError(f"Unknown framework: {framework}")


async def run_one_task(
    task_data: dict,
    *,
    data_dir: str,
    framework: str,
    dataset: str,
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
    slayer_setup: str | None = None,
    reasoning_effort: str | None = None,
    user_sim_prompt_version: str | None = None,
    pre_encoded_source: str | None = None,
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
    from bird_interact_agents.agents._pre_encoded import derive_slayer_setup
    if slayer_setup is None:
        slayer_setup = derive_slayer_setup(pre_encoded_source)
    _validate_slayer_setup(
        slayer_setup=slayer_setup, framework=framework,
        query_mode=query_mode, mode=mode,
        pre_encoded_source=pre_encoded_source,
    )
    runner = _make_runner(
        framework=framework,
        dataset=dataset,
        query_mode=query_mode,
        mode=mode,
        agent_model=agent_model,
        strict=strict,
        prompt_cache=prompt_cache,
        max_depth=max_depth,
        slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup,
        reasoning_effort=reasoning_effort,
        user_sim_prompt_version=user_sim_prompt_version,
        pre_encoded_source=pre_encoded_source,
    )
    instance_id = str(task_data.get("instance_id") or "")
    t_start = time.perf_counter()
    timeout = _per_task_timeout_s()
    try:
        coro = runner(task_data, data_dir, patience, user_sim_model)
        if timeout > 0:
            try:
                r = await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError as te:
                # Same explicit message as in run_one_task_with_runner —
                # asyncio.TimeoutError has no message, so the per-task
                # error row would record the empty string. DEV-1535 r3
                # (Codex): the wall-clock cap was previously wrapped
                # ONLY in run_one_task_with_runner (the `cached_runner`
                # path), so SLayer / non-raw runs that fall through to
                # this function ran uncapped — defeating the cap on
                # exactly the runs most likely to thrash.
                raise TimeoutError(
                    f"per-task runaway-grace ceiling of {timeout:.0f}s exceeded — the "
                    f"agent ignored the wall-clock budget hook's deny "
                    f"(BIRD_INTERACT_PER_TASK_TIMEOUT_S=0 to disable)"
                ) from te
        else:
            r = await coro
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


def write_local_attempt_row(rows_dir: Path, instance_id: str, row: dict) -> None:
    """Persist a finalized per-task row to ``rows/<iid>/attempt-1.json``,
    the local mirror of the cloud actor's ``_gcs.write_row`` (which writes
    the same blob to GCS at ``ray_app.py``). This is what makes local runs
    capture the SAME raw per-turn ``trajectory`` (and the ``tool_call_stats``
    derived from it) that cloud runs do — local/cloud capability parity.

    The caller pops the non-serialisable Pydantic objects (``_task_annotation``
    / ``_autopsy``) from ``row`` first, exactly as the cloud path does before
    its ``json.dumps(row)``; ``default=str`` is a belt-and-braces guard so a
    stray object can never take down the run loop. Best-effort: a write
    failure is logged, never raised (the row is already in results.db).
    """
    try:
        dest = rows_dir / instance_id / "attempt-1.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write (tmp + rename) so a crash/kill mid-write can never
        # leave a truncated attempt-1.json that downstream consumers
        # (autopsy regen, tool-stats re-derivation) fail to parse — matches
        # the atomicity of the cloud path's GCS blob upload.
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, default=str))
        tmp.replace(dest)
    except Exception:  # noqa: BLE001 — persistence is best-effort
        logger.exception(
            "failed to write local attempt row for %s", instance_id
        )


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
    slayer_setup: str | None = None,
    otf_rebuild: bool = False,
    dataset: str = "mini-interact",
    reasoning_effort: str | None = None,
    user_sim_prompt_version: str | None = None,
    pre_encoded_source: str | None = None,
) -> dict:
    """Run full evaluation across all tasks."""
    from bird_interact_agents.agents._pre_encoded import derive_slayer_setup
    if slayer_setup is None:
        slayer_setup = derive_slayer_setup(pre_encoded_source)
    _validate_dataset_mode(dataset=dataset, mode=mode)
    _validate_framework_mode(framework=framework, dataset=dataset, mode=mode)
    _validate_slayer_setup(
        slayer_setup=slayer_setup, framework=framework,
        query_mode=query_mode, mode=mode,
        pre_encoded_source=pre_encoded_source,
    )
    b = get_benchmark(dataset)

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
        dataset, data_path, limit=limit, filter_ids=filter_ids,
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
        pre_encoded_source=pre_encoded_source,
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
        dataset=dataset,
        query_mode=query_mode,
        mode=mode,
        agent_model=agent_model,
        strict=strict,
        prompt_cache=prompt_cache,
        max_depth=max_depth,
        slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup,
        reasoning_effort=reasoning_effort,
        user_sim_prompt_version=user_sim_prompt_version,
        pre_encoded_source=pre_encoded_source,
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
        query_mode=query_mode,
        slayer_setup=slayer_setup,
        pre_encoded_source=pre_encoded_source,
        patience=patience,
        max_depth=max_depth,
        reasoning_effort=reasoning_effort,
        dataset=dataset,
        strict=strict,
        use_audited_gold_sql=use_audited_gold_sql,
        prompt_cache=prompt_cache,
    )

    # Per-task `SubmissionConfig` — duplicated into every annotation per
    # DEV-1535 design choice (B3 = both run_metadata AND annotation). The
    # config is identical across tasks in a single run, so one instance
    # is shared.
    submission_config = SubmissionConfig(
        framework=framework,
        mode=mode,
        query_mode=query_mode,
        agent_model=agent_model,
        user_sim_model=user_sim_model,
        slayer_setup=slayer_setup,
        pre_encoded_source=pre_encoded_source,
        reasoning_effort=reasoning_effort,
        patience=patience,
        max_depth=max_depth,
        dataset=dataset,
        strict=strict,
        use_audited_gold_sql=use_audited_gold_sql,
        prompt_cache=prompt_cache,
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
        _agent_cost, _sim_cost = extract_usage_costs(usage_blob)
        common_failed_kwargs = dict(
            rows_dir=rows_dir,
            instance_id=instance_id_for_log,
            selected_database=selected_database or "<unknown>",
            benchmark=_benchmark_canonical,
            run_id=run_id,
            trajectory_path=f"rows/{instance_id_for_log}/attempt-1.json",
            duration_s=r.get("duration_s"),
            cost_usd_agent=_agent_cost,
            cost_usd_user_sim=_sim_cost,
            n_agent_turns=usage_blob.get("n_agent_turns"),
            n_ask_user_calls=usage_blob.get("n_ask_user_calls"),
            predicted_row_count=r.get("predicted_row_count"),
            config=submission_config,
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
        # For postgres, db_path is a db-name carrier (executor uses
        # db_path.stem). For SQLite, prefer the materialized per-task copy
        # when available (LiveSQLBench tasks: materialize_task_db sets
        # db_file_path to an isolated $TMPDIR copy so concurrent runs don't
        # race the shared <db>.sqlite); fall back to data_dir/<db>/<db>.sqlite.
        if getattr(b, "db_backend", "sqlite") == "postgres":
            per_task_db = Path(selected_database)
        else:
            _db_file_path = td.get("db_file_path")
            per_task_db = (
                Path(_db_file_path)
                if _db_file_path
                else Path(data_dir) / selected_database / f"{selected_database}.sqlite"
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
                cost_usd_agent=_agent_cost,
                cost_usd_user_sim=_sim_cost,
                n_agent_turns=usage_blob.get("n_agent_turns"),
                n_ask_user_calls=usage_blob.get("n_ask_user_calls"),
                predicted_row_count=r.get("predicted_row_count"),
                config=submission_config,
                task_annotation=r.get("_task_annotation"),
                autopsy_result=r.get("_autopsy"),
                harness_passed=r.get("phase1_passed") is True,
                predicted_result=_decode_result_json(r.get("predicted_result_json")),
                gold_result=_decode_result_json(r.get("gold_result_json")),
                # DEV-1613: build the N5 insufficient-task judge from the
                # run's agent_model so the cascade can fire it inline.
                agent_model=agent_model,
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
            _timeout = _per_task_timeout_s()
            try:
                _coro = runner(td, data_dir, patience, user_sim_model)
                if _timeout > 0:
                    # DEV-1535 r5 (Codex): the local `run_evaluation`
                    # loop awaited `runner(...)` directly, so runaway
                    # tasks ran uncapped. The cloud (`run_one_task` /
                    # `run_one_task_with_runner`) entry points both
                    # already wrap with `asyncio.wait_for`; mirror it
                    # here so `BIRD_INTERACT_PER_TASK_TIMEOUT_S` binds
                    # the local CLI path too.
                    try:
                        r = await asyncio.wait_for(_coro, timeout=_timeout)
                    except asyncio.TimeoutError as te:
                        raise TimeoutError(
                            f"per-task runaway-grace ceiling of {_timeout:.0f}s "
                            f"exceeded — the agent ignored the "
                            f"wall-clock budget hook's deny "
                            f"(BIRD_INTERACT_PER_TASK_TIMEOUT_S=0 "
                            f"to disable)"
                        ) from te
                else:
                    r = await _coro
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
            r.pop("_autopsy", None)
            r.pop("_task_annotation", None)
            # Parity with the cloud actor (which writes this row to GCS via
            # `_gcs.write_row`): persist the raw trajectory + tool_call_stats
            # locally so `submission.trajectory_path` resolves and local runs
            # capture the same data as cloud.
            write_local_attempt_row(rows_dir, instance_id, r)

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

    # DEV-1533: emit the cascading_phase1 block from the runs/ golden
    # store. grade_and_write already wrote each task's annotation there
    # during execution, so runs/ is populated by the time we reach here.
    # Codex r11: scope to the CURRENT run's instance set so filtered
    # reruns don't mix stale annotations into the published metrics.
    _current_iids = {
        str(td.get("instance_id") or "") for td in tasks
    } - {""}
    try:
        metrics = emit_cascading_eval_json(
            _benchmark_canonical, run_id, Path(output_path),
            base_metrics=metrics, instance_filter=_current_iids,
        )
    except Exception as exc:  # noqa: BLE001
        # Mirror cloud/driver._emit_cascading_phase1_on_fetch: surface the
        # failure on the metrics dict + log a warning instead of silently
        # dropping it. Otherwise local runs ship an eval.json missing the
        # PR's primary metric block with no operator signal.
        logger.warning("cascading_phase1 aggregation failed: %s", exc)
        metrics["cascading_phase1_error"] = str(exc)

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


def _maybe_start_bridge_proxy(*, agent_model: str, subscription_auth, error) -> None:
    """DEV-1604: local-run wiring for the Anthropic⇄OpenAI bridge proxy.

    Recycles ``--subscription-auth``: for z.ai, ``--subscription-auth`` selects
    the direct coding-plan Anthropic endpoint (no bridge) and the default /
    ``--no-subscription-auth`` uses the per-token bridge. Doubleword is
    OpenAI-only (always bridged; ``--subscription-auth`` rejected); Moonshot is
    provider-key-only (``--subscription-auth`` rejected). When the agent needs
    the bridge, start the loopback proxy and point ``ANTHROPIC_BASE_URL``'s
    override at it. Called from ``main`` BEFORE any runner is built. ``error``
    is ``parser.error`` (exit-2 on misuse)."""
    spec = get_provider(agent_model)
    if spec is None:  # Anthropic (or unknown) — never bridges.
        return
    if bool(subscription_auth) and spec.key != "zai":
        _why = (
            "OpenAI-only — no Anthropic endpoint"
            if spec.api_format == "openai"
            else f"authenticates via {spec.auth_env}"
        )
        error(
            f"--subscription-auth is not valid for {spec.key} agent models "
            f"({_why}). Omit the flag or pass --no-subscription-auth."
        )
        return
    no_subscription_auth = not bool(subscription_auth)
    if agent_needs_bridge(agent_model, no_subscription_auth):
        bridge_proxy.ensure_bridge_proxy_for_actor(
            agent_model, {"no_subscription_auth": no_subscription_auth}
        )


def _apply_subscription_auth_env(
    *,
    subscription_auth: bool,
    framework: str,
    agent_model: str,
    error,
) -> None:
    """Translate the local ``--subscription-auth`` choice into the
    ``BIRD_INTERACT_SUBSCRIPTION_AUTH`` signal env var that ``sdk_env`` reads
    (DEV-1602). Local claude_sdk agents run in-process, so this is the only
    wiring needed — ``sdk_env`` masks ``ANTHROPIC_API_KEY`` for the SDK
    subprocess while the parent keeps it for the litellm user-sim.

    ``subscription_auth`` is the tri-state BooleanOptionalAction value: ``True``
    (--subscription-auth), ``False`` (--no-subscription-auth), or ``None`` (the
    operator passed neither). Mirroring the cloud CLI, a claude_sdk* run on an
    Anthropic agent model MUST choose explicitly — a ``None`` there is an error
    (no silent default). ``error`` is a callable (``parser.error``) invoked with
    a message on misuse. When off, an ambient signal is actively CLEARED so a
    stray exported ``BIRD_INTERACT_SUBSCRIPTION_AUTH`` cannot hijack the run.
    The flag is Anthropic-only and claude_sdk-only.
    """
    is_claude_sdk = framework.startswith("claude_sdk")
    # DEV-1604: registry models NEVER use the Claude.ai OAuth path. For z.ai,
    # --subscription-auth is the ENDPOINT selector (coding-plan vs per-token
    # bridge), validated in `_maybe_start_bridge_proxy`; Moonshot/Doubleword
    # reject it there. So clear any ambient OAuth signal and return — do not run
    # the Anthropic-only OAuth machinery (which would reject z.ai here).
    if get_provider(agent_model) is not None:
        os.environ.pop("BIRD_INTERACT_SUBSCRIPTION_AUTH", None)
        return
    # The flag is Anthropic-ONLY: gate on the model being anthropic/*, NOT merely
    # "not a registry model" — otherwise a non-Anthropic non-registry model
    # (openai/*, gemini/*) would slip through onto the OAuth path (CodeRabbit).
    is_anthropic = agent_model.startswith("anthropic/")
    # Explicit-choice requirement (cloud parity): claude_sdk* + Anthropic model
    # must pass --subscription-auth or --no-subscription-auth.
    if is_claude_sdk and is_anthropic and subscription_auth is None:
        error(
            "an explicit --subscription-auth / --no-subscription-auth choice is "
            "required for claude_sdk* runs on an Anthropic agent model (no "
            "default, to prevent a silent fall-back to the API-key path)."
        )
        return
    if not subscription_auth:  # None (non-claude_sdk / non-Anthropic) or False → off
        os.environ.pop("BIRD_INTERACT_SUBSCRIPTION_AUTH", None)
        return
    if not is_claude_sdk:
        error(
            "--subscription-auth only applies to claude_sdk* frameworks; "
            f"got framework={framework!r}. Other frameworks authenticate via "
            "their own provider key env var."
        )
        return
    if not is_anthropic:
        error(
            f"--subscription-auth is Anthropic-only; got non-Anthropic agent "
            f"model {agent_model!r} (registry open-weight models use their "
            "provider key; omit the flag for them)."
        )
        return
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if not token:
        error(
            "--subscription-auth requires CLAUDE_CODE_OAUTH_TOKEN to be set "
            "in the env. Run `claude setup-token`, or omit the flag to use "
            "the ANTHROPIC_API_KEY path."
        )
        return
    if not token.startswith("sk-ant-oat01-"):
        error(
            "CLAUDE_CODE_OAUTH_TOKEN does not look like a Claude.ai OAuth token "
            "(expected sk-ant-oat01- prefix). Re-run `claude setup-token`."
        )
        return
    os.environ["BIRD_INTERACT_SUBSCRIPTION_AUTH"] = "1"


def _resolve_data_paths(args, *, error) -> None:
    """DEV-1638: derive ``--data``/``--db-path`` from the registry when BOTH are
    omitted; require both-or-neither. Explicit values pass through unchanged.
    Runs BEFORE the ``db_path`` ``.resolve()`` at the CLI boundary so a derived
    ``None`` can never reach ``Path(None)``."""
    if args.data is None and args.db_path is None:
        args.data = str(paths.benchmark_data_file(args.dataset))
        args.db_path = str(paths.benchmark_data_root(args.dataset))
    elif args.data is None or args.db_path is None:
        error(
            "--data and --db-path must be given together or both omitted "
            "(derived from the benchmark registry)."
        )


def _effective_instance_ids(args, filter_ids):
    """DEV-1638 (Codex #5): the concrete instance-id list the RUN will use, so
    provisioning + annotation sync scope to exactly the run (never the whole
    benchmark) and never trip on an unrelated missing dump/annotation.

    ``filter_ids`` (from ``--instance-id`` / ``--filter-ids``) wins; else
    ``--limit`` selects the first-N via the SAME loader ``run_evaluation`` uses
    (identical ordering); else ``None`` (whole benchmark).

    ``is not None`` (not truthiness) on BOTH ``filter_ids`` and ``--limit`` so an
    explicit empty selection propagates as ``[]`` (no scope) rather than
    collapsing to the whole-benchmark ``None`` (Codex PR #75 r2/r4). Only an
    omitted (``None``) filter AND ``None`` limit yield the whole benchmark."""
    if filter_ids is not None:
        return filter_ids
    if args.limit is not None:
        rows = load_benchmark_tasks(args.dataset, args.data, limit=args.limit)
        return [r["instance_id"] for r in rows]
    return None


def _maybe_bootstrap_local_postgres(args, effective_ids) -> None:
    """DEV-1638: for a postgres benchmark with no pre-set ``BIRD_PG_*``,
    provision a private local cluster + export ``BIRD_PG_*``. Gated on the
    CONNECTION signal (``BIRD_PG_HOST``), NOT on whether ``--data``/``--db-path``
    were derived (Codex #1): postgres connects via ``BIRD_PG_*``, and
    ``--db-path`` is only the data/KB root — so explicit paths must not suppress
    provisioning."""
    if get_benchmark(args.dataset).db_backend != "postgres":
        return
    if "BIRD_PG_HOST" in os.environ:
        return  # caller brought their own postgres connection
    # Distinguish an EMPTY id list (e.g. `--limit 0` → zero-task validation run)
    # from None (whole benchmark). Empty ⇒ nothing to provision; provisioning
    # the whole benchmark here would be the scope-expansion Codex PR #75 flagged.
    if effective_ids is not None and not effective_ids:
        return
    exports = provision_and_export(args.dataset, effective_ids, args.pg_port)
    os.environ.update(exports)


def _maybe_sync_annotations(args, effective_ids) -> None:
    """DEV-1638: best-effort pull of the authoritative task annotations from GCS
    into the local store (all backends), so the tolerant grader has the real
    ``gold_variants``/``evaluator_prompt`` instead of the implicit N1 fallback.
    Best-effort: a GCS failure warns, never aborts the run. Warns loudly on any
    id still missing after the sync (surfaces the silent N1 degradation)."""
    if args.skip_annotations:
        return
    # Empty id list (e.g. `--limit 0`) ⇒ nothing to sync; None ⇒ whole benchmark.
    if effective_ids is not None and not effective_ids:
        return
    try:
        result = sync_annotations(args.dataset, effective_ids)
    except Exception as exc:  # noqa: BLE001 — annotation sync is best-effort
        logger.warning(
            "annotation sync failed (%s); grading falls back to the implicit "
            "N1 annotation for any id lacking a local task.json.", exc,
        )
        return
    if result.get("missing_in_gcs"):
        logger.warning(
            "%d task annotation(s) missing in GCS after sync; those ids grade "
            "against the implicit N1 annotation only. Run a cloud `annotate` "
            "(or check the ids) to get authoritative gold_variants.",
            result["missing_in_gcs"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BIRD-Interact benchmark runner with pluggable agents"
    )
    parser.add_argument(
        "--framework",
        # DEV-1555 v0/v1: only the two aggregator tokens are user-facing.
        # The per-variant tokens (`claude_sdk_otf{,_v1}` / `*_raw{,_v1}` /
        # `*_ainteract{,_v1}`) remain accepted by `_make_runner` for
        # programmatic / test callers, but the CLI infers the variant
        # from (benchmark.one_shot × query_mode) — `claude_sdk` →
        # origin/main shape; `claude_sdk_v1` → this branch's shape.
        choices=[
            "claude_sdk",
            "claude_sdk_v1",
            # DEV-1609: the default OTF reference encoder.
            "claude_sdk_otf_encode",
            # non-SDK frameworks unchanged.
            "pydantic_ai",
            "pydantic_ai_recursive",
            "pydantic_ai_otf_encode",
            "mcp_agent",
            "agno",
            "smolagents",
        ],
        required=True,
        help="Agent framework to use",
    )
    parser.add_argument(
        "--mode",
        choices=["a-interact", "c-interact", "oracle", "one-shot"],
        required=True,
        help=(
            "REQUIRED (aligned with bird-interact-cloud: no default). "
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
            "REQUIRED — no default. Use canonical hyphenated names: "
            "mini-interact, livesqlbench-base-lite-sqlite, "
            "livesqlbench-base-lite (postgres), bird-interact-lite-exp (postgres)."
        ),
    )
    parser.add_argument(
        "--query-mode",
        choices=["slayer", "raw"],
        required=True,
        help=(
            "REQUIRED (aligned with bird-interact-cloud: no default). "
            "Query mode: slayer (semantic layer) or raw (direct SQL)"
        ),
    )
    parser.add_argument(
        "--data", default=None,
        help=(
            "Path to the benchmark's tasks JSONL. OPTIONAL (DEV-1638): derived "
            "from the benchmark registry when omitted. Pass BOTH --data and "
            "--db-path to override (non-standard local layout), or NEITHER to "
            "derive."
        ),
    )
    parser.add_argument(
        "--db-path", default=None,
        help=(
            "Path to the benchmark's data root (SQLite DBs, or the pg_dumps/ + "
            "KB root for postgres). OPTIONAL (DEV-1638): derived from the "
            "registry when omitted. Both-or-neither with --data. NOTE: for a "
            "postgres benchmark this is only the data/KB root — the DB "
            "connection comes from BIRD_PG_* (auto-provisioned; see --pg-port)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "results/<benchmark>/<YYYYMMDDtHHMM>_<framework>_<query_mode>/eval.json "
            "under the main checkout."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--patience", type=int, default=250,
        help=(
            "User patience budget (aligned with bird-interact-cloud's default "
            "of 250; the old local default of 3 was too low and skewed eval "
            "results)."
        ),
    )
    parser.add_argument(
        "--agent-model",
        required=True,
        help=(
            "REQUIRED (aligned with bird-interact-cloud: no default, to avoid "
            "a silent wrong-model run). LiteLLM-style PROVIDER/MODEL_ID for the "
            "system agent. "
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
        default="anthropic/claude-sonnet-4-6",
        help="LiteLLM model for user simulator (aligned with bird-interact-cloud default)",
    )
    parser.add_argument(
        "--slayer-storage-root",
        default="./slayer_storage",
        help="Root dir of per-DB SLayer model stores (only used in --query-mode slayer)",
    )
    parser.add_argument(
        "--subscription-auth",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="subscription_auth",
        help=(
            "DEV-1602: authenticate claude_sdk* agents via the Claude.ai "
            "subscription (CLAUDE_CODE_OAUTH_TOKEN, sk-ant-oat01- prefix) "
            "instead of ANTHROPIC_API_KEY. Aligned with bird-interact-cloud: an "
            "explicit --subscription-auth / --no-subscription-auth choice is "
            "REQUIRED for claude_sdk* runs on an Anthropic agent model (no "
            "silent default). When on for Anthropic, a valid "
            "CLAUDE_CODE_OAUTH_TOKEN must be in the env. DEV-1604: for z.ai the "
            "flag is recycled as the ENDPOINT selector (still ZAI_API_KEY, NOT "
            "OAuth): --subscription-auth = direct coding-plan; default / "
            "--no-subscription-auth = per-token OpenAI bridge. Doubleword "
            "(OpenAI-only) and Moonshot (provider-key-only) reject the flag."
        ),
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default True (aligned with bird-interact-cloud); pass "
            "--no-use-audited-gold-sql to opt out. "
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
    # DEV-1545: same flag as cloud/cli.py; flows through _make_runner and
    # is normalised to "v2" at the closure level before agent.run_task.
    parser.add_argument(
        "--user-sim-prompt-version",
        dest="user_sim_prompt_version",
        choices=["v2", "v3"],
        default=None,
        help=(
            "User-sim prompt variant. v2 = upstream default. v3 = DEV-1545 "
            "anti-fabrication variant. Unset = v2 at agent call time."
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
        "--pre-encoded-models",
        dest="pre_encoded_source",
        choices=["otf", "custom"],
        default=None,
        help=(
            "DEV-1586: run the SLayer agents against an ALREADY-encoded "
            "datasource (read-only; no model-mutation tools). "
            "'otf' = the encoding-agent output at "
            "slayer_models_otf/<benchmark>/<db>; 'custom' = the hand-curated "
            "slayer_models/<db>. When omitted (default), the SLayer agents "
            "encode KB items ON THE FLY. This flag REPLACES the retired "
            "--slayer-setup flag: the internal slayer_setup value is derived "
            "from it ('pre-encoded' when set, else 'on-the-fly')."
        ),
    )
    # DEV-1638: unify the local entrypoint — fold the postgres bootstrap the
    # old scripts/run_local_postgres.py did into bird-interact.
    parser.add_argument(
        "--env-file",
        default=os.environ.get("BIRD_ENV_FILE"),
        help=(
            "DEV-1638: dotenv file of auth vars (CLAUDE_CODE_OAUTH_TOKEN / "
            "ANTHROPIC_API_KEY / provider keys) to load into the environment "
            "BEFORE auth resolution. Best-effort: a missing file is skipped. "
            "Default: $BIRD_ENV_FILE if set, else no dotenv is loaded (rely on "
            "the ambient shell env). Applies to all backends."
        ),
    )
    parser.add_argument(
        "--pg-port", type=int,
        default=int(os.environ.get("BIRD_PG_PORT", DEFAULT_PORT)),
        help=(
            "DEV-1638: port for the auto-provisioned local postgres cluster "
            "(postgres benchmarks only, when BIRD_PG_HOST is not already set). "
            f"Default: $BIRD_PG_PORT if set, else {DEFAULT_PORT}."
        ),
    )
    parser.add_argument(
        "--skip-annotations", action="store_true", default=False,
        help=(
            "DEV-1638: skip the best-effort GCS task-annotation sync. By "
            "default bird-interact pulls any missing authoritative annotations "
            "so the tolerant grader has the real gold_variants/evaluator_prompt "
            "instead of degrading to the implicit N1 annotation."
        ),
    )
    args = parser.parse_args()
    # DEV-1638: load the auth dotenv FIRST, before any auth resolution reads the
    # env. Best-effort — a missing/None path is a no-op.
    if args.env_file:
        n_env = load_env_file(Path(args.env_file))
        if n_env:
            logger.info("loaded %d var(s) from --env-file %s", n_env, args.env_file)
        else:
            logger.info("no env file at --env-file %s (skipping)", args.env_file)
    # DEV-1602: translate --subscription-auth into the BIRD_INTERACT_SUBSCRIPTION_AUTH
    # signal env var (or clear an ambient one) before any agent is constructed.
    _apply_subscription_auth_env(
        subscription_auth=args.subscription_auth,
        framework=args.framework,
        agent_model=args.agent_model,
        error=parser.error,
    )
    # DEV-1586: derive the internal slayer_setup from the user-facing flag.
    from bird_interact_agents.agents._pre_encoded import derive_slayer_setup
    args.slayer_setup = derive_slayer_setup(args.pre_encoded_source)

    # DEV-1638: derive --data/--db-path from the registry when both omitted
    # (both-or-neither), BEFORE the .resolve() below so a derived None can never
    # reach Path(None).
    _resolve_data_paths(args, error=parser.error)

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
        _validate_framework_mode(
            framework=args.framework, dataset=args.dataset, mode=args.mode,
        )
        _validate_slayer_setup(
            slayer_setup=args.slayer_setup,
            framework=args.framework,
            query_mode=args.query_mode,
            mode=args.mode,
            pre_encoded_source=args.pre_encoded_source,
        )
    except ValueError as e:
        parser.error(str(e))

    # DEV-1604: start the Anthropic⇄OpenAI bridge proxy (Doubleword / z.ai
    # per-token) and point the base-url override at it BEFORE any runner is
    # built, but AFTER all the parser.error validation above — so an invalid
    # invocation fails fast without spawning a proxy or mutating the env.
    # Recycles --subscription-auth as the z.ai endpoint selector.
    _maybe_start_bridge_proxy(
        agent_model=args.agent_model,
        subscription_auth=args.subscription_auth,
        error=parser.error,
    )

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%dt%H%M")
        args.output = str(
            paths.results_root()
            / args.dataset
            / f"{ts}_{args.framework}_{args.query_mode}"
            / "eval.json"
        )

    if args.price_overrides:
        _apply_price_overrides(args.price_overrides)

    filter_ids: list[str] | None = None
    # `is not None` (not truthiness) so an EXPLICIT empty value (`--instance-id
    # ""` / `--filter-ids ""`) is REJECTED here rather than falling through to
    # whole-benchmark scope (Codex PR #75 r6). Only an OMITTED flag (None)
    # leaves filter_ids=None → whole benchmark.
    if args.instance_id is not None:
        # Accept either a single id or a comma-separated list. Whitespace
        # around items is trimmed; empty tokens are dropped.
        filter_ids = [s.strip() for s in args.instance_id.split(",") if s.strip()]
        # Reject input that parses to an empty list (e.g. "" / ",,, "). Without
        # this, filter_ids=[] later falls back to running the full
        # benchmark — a silent expansion of scope from a malformed flag.
        if not filter_ids:
            parser.error(
                "--instance-id must include at least one non-empty id",
            )
    elif args.filter_ids is not None:
        if not args.filter_ids:
            parser.error("--filter-ids requires a non-empty file path")
        with open(args.filter_ids) as f:
            filter_ids = [line.strip() for line in f if line.strip()]
        # Symmetric with the --instance-id empty check above (Codex PR #75): an
        # empty --filter-ids file must fail HERE, before provisioning + sync —
        # otherwise `filter_ids=[]` is falsy, `_effective_instance_ids` returns
        # None, and we would provision the whole benchmark + hit GCS for every
        # task before `run_evaluation` rejects the empty filter.
        if not filter_ids:
            parser.error(
                f"--filter-ids file {args.filter_ids!r} contained no instance_ids",
            )

    # DEV-1638: the concrete id list this run uses — so provisioning + sync
    # scope to exactly the run, never the whole benchmark.
    effective_ids = _effective_instance_ids(args, filter_ids)
    # For a postgres benchmark with no pre-set BIRD_PG_*, spin up + load a
    # private local cluster and export BIRD_PG_* BEFORE the run connects.
    _maybe_bootstrap_local_postgres(args, effective_ids)
    # Best-effort pull of the authoritative task annotations (all backends).
    _maybe_sync_annotations(args, effective_ids)

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
            reasoning_effort=args.reasoning_effort,
            user_sim_prompt_version=args.user_sim_prompt_version,
            pre_encoded_source=args.pre_encoded_source,
        )
    )


if __name__ == "__main__":
    main()
