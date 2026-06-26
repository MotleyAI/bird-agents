"""In-cluster Ray driver for the annotator agent (DEV-1518).

Invoked via `ray job submit -- python ray_app_annotator.py <args>` from the
laptop-side `bird-interact-cloud annotate` command.

Worker contract:
* Skip if both stable blobs exist (unless --override).
* Run one task → write 4 GCS paths on success (run-specific + stable for
  task_annotation and audited_gold_variants).
* Write attempt-N.json for every outcome (annotated / skipped / error) so
  list_attempts() / wait_until_done() work unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any

from bird_interact_agents import paths
from bird_interact_agents.agents.annotator.agent import AnnotatorResult
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.cloud import gcs as _gcs
from bird_interact_agents.cloud.ray_app import (
    HeartbeatWriter,
    _apply_actor_env_local,
    _load_secrets_file,
    _maybe_ensure_bridge,
    _run_with_actors,
    _with_actor_env,
    download_benchmark_data,
)
from bird_interact_agents.harness import load_benchmark_tasks


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS client default (overridable in tests via monkeypatch)
# ---------------------------------------------------------------------------

def _default_ray_job_id() -> str:
    """Read the Ray Jobs API submission id (`raysubmit_*`) from the runtime
    env Ray sets inside the job. Falls back to `"unknown"` for local runs."""
    return os.environ.get("RAY_JOB_SUBMISSION_ID", "unknown")


def default_gcs_client():
    return _gcs.default_gcs_client()


# ---------------------------------------------------------------------------
# Agent runner (replaceable in tests via monkeypatch)
# ---------------------------------------------------------------------------

def _run_agent(
    task_data: dict,
    cfg: dict,
    data_path_base: str,
) -> AnnotatorResult:
    """Run the annotator agent synchronously for one task."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    return asyncio.run(
        ann_agent.run_task(
            task_data=task_data,
            data_path_base=data_path_base,
            benchmark=cfg["benchmark"],
            model=cfg.get("model", "anthropic/claude-opus-4-7"),
            effort=cfg.get("effort", "medium"),
        )
    )


# ---------------------------------------------------------------------------
# Core per-task logic
# ---------------------------------------------------------------------------

def _run_one_task(
    *,
    task_data: dict,
    cfg: dict,
    run_id: str,
    data_path_base: str,
    gcs_client=None,
    attempt: int = 1,
) -> None:
    """Execute one annotator task and write all outputs to GCS."""
    client = gcs_client or default_gcs_client()
    instance_id = task_data["instance_id"]
    db = task_data["selected_database"]
    benchmark = cfg["benchmark"]
    override = cfg.get("override", False)
    t0 = time.monotonic()

    # Skip check: both stable blobs must be present to skip.
    if not override:
        ann_blob = _gcs.stable_task_annotation_blob(benchmark, db, instance_id)
        var_blob = _gcs.stable_audited_gold_variants_blob(benchmark, db, instance_id)
        try:
            both_exist = (
                _gcs.blob_exists(ann_blob, client=client)
                and _gcs.blob_exists(var_blob, client=client)
            )
        except Exception as exc:
            logger.warning(
                "[%s] skip-check failed (%s); proceeding with annotation",
                instance_id, exc,
            )
            both_exist = False
        if both_exist:
            logger.info(
                "[%s] skipping — both stable blobs exist; copying to run-scoped paths",
                instance_id,
            )
            try:
                bucket = client.bucket(_gcs.BUCKET_NAME)
                for src_name, dst_name in (
                    (ann_blob, _gcs.task_annotation_blob(run_id, instance_id)),
                    (var_blob, _gcs.audited_gold_variants_blob(run_id, instance_id)),
                ):
                    data = bucket.blob(src_name).download_as_bytes()
                    bucket.blob(dst_name).upload_from_string(data)
                attempt_row = {
                    "instance_id": instance_id,
                    "database": db,
                    "status": "skipped",
                    "duration_s": time.monotonic() - t0,
                }
                _write_attempt(run_id, instance_id, attempt_row, attempt=attempt, client=client)
            except Exception as exc:
                logger.warning(
                    "[%s] stable→run-scoped copy failed (%s); marking error so resubmit retries",
                    instance_id, exc,
                )
                attempt_row = {
                    "instance_id": instance_id,
                    "database": db,
                    "status": "error",
                    "error": f"stable→run-scoped copy failed: {exc}",
                    "duration_s": time.monotonic() - t0,
                }
                _write_attempt(run_id, instance_id, attempt_row, attempt=attempt, client=client)
            return

    # Run the agent.
    try:
        result = _run_agent(task_data=task_data, cfg=cfg, data_path_base=data_path_base)
    except Exception as exc:
        logger.error("[%s] agent raised: %s", instance_id, exc)
        attempt_row = {
            "instance_id": instance_id,
            "database": db,
            "status": "error",
            "error": str(exc),
            "duration_s": time.monotonic() - t0,
        }
        _write_attempt(run_id, instance_id, attempt_row, attempt=attempt, client=client)
        return

    if result.error:
        logger.warning("[%s] agent returned error: %s", instance_id, result.error)
        attempt_row = {
            "instance_id": instance_id,
            "database": db,
            "status": "error",
            "error": result.error,
            "duration_s": result.duration_s,
        }
        _write_attempt(run_id, instance_id, attempt_row, attempt=attempt, client=client)
        return

    # Success — write 4 GCS paths.
    ann = result.task_annotation
    variants = result.audited_gold_variants

    try:
        _gcs.write_task_annotation(run_id, instance_id, ann, client=client)
        _gcs.write_audited_gold_variants(
            run_id, instance_id, variants,
            benchmark=benchmark, selected_database=db, client=client,
        )
        _gcs.write_stable_task_annotation(benchmark, db, instance_id, ann, client=client)
        _gcs.write_stable_audited_gold_variants(benchmark, db, instance_id, variants, client=client)
    except Exception as exc:
        logger.error("[%s] GCS write failed after annotation: %s", instance_id, exc)
        attempt_row = {
            "instance_id": instance_id,
            "database": db,
            "status": "error",
            "error": f"GCS write failed: {exc}",
            "duration_s": result.duration_s,
        }
        _write_attempt(run_id, instance_id, attempt_row, attempt=attempt, client=client)
        return

    attempt_row = {
        "instance_id": instance_id,
        "database": db,
        "status": "annotated",
        "duration_s": result.duration_s,
    }
    _write_attempt(run_id, instance_id, attempt_row, attempt=attempt, client=client)


def _write_attempt(
    run_id: str,
    instance_id: str,
    row: dict,
    *,
    attempt: int = 1,
    client=None,
) -> None:
    blob_name = f"runs/{run_id}/rows/{instance_id}/attempt-{attempt}.json"
    client = client or default_gcs_client()
    blob = client.bucket(_gcs.BUCKET_NAME).blob(blob_name)
    blob.upload_from_string(
        json.dumps(row).encode(), content_type="application/json"
    )


# ---------------------------------------------------------------------------
# Task data loader (annotator only — no gold overlay needed)
# ---------------------------------------------------------------------------

def _load_annotator_task_data(
    instance_ids: list[str],
    *,
    benchmark: str,
) -> dict[str, dict]:
    """Load plain task data for the annotator (no audited-gold overlay).
    Gold is auto-discovered from gated_gold/<benchmark>/ via load_benchmark_tasks.
    """
    rows = load_benchmark_tasks(
        benchmark,
        str(paths.benchmark_data_file(benchmark)),
        filter_ids=instance_ids,
    )
    return {td["instance_id"]: td for td in rows}


# ---------------------------------------------------------------------------
# Ray actor class builder
# ---------------------------------------------------------------------------

def _build_annotator_actor_class():
    """Return a Ray-remote actor class whose `.run_one(task_data)` calls
    `_run_one_task`. Lazy import so test environments without Ray can still
    use the sequential path."""
    import ray  # type: ignore[import-not-found]

    @ray.remote
    class AnnotatorActor:
        def __init__(self, cfg: dict[str, Any], run_id: str, data_path_base: str,
                     attempt: int = 1):
            self.cfg = cfg
            self.run_id = run_id
            self.data_path_base = data_path_base
            self.attempt = attempt
            self.gcs_client = default_gcs_client()
            download_benchmark_data(cfg, client=self.gcs_client)
            # DEV-1604: bring up the Anthropic⇄OpenAI bridge if the annotator
            # runs a registry model (Doubleword / z.ai per-token) BEFORE the
            # first task builds the SDK session.
            _maybe_ensure_bridge(cfg)

        def run_one(self, task_data: dict) -> None:
            _run_one_task(
                task_data=task_data,
                cfg=self.cfg,
                run_id=self.run_id,
                data_path_base=self.data_path_base,
                gcs_client=self.gcs_client,
                attempt=self.attempt,
            )

    return AnnotatorActor


# ---------------------------------------------------------------------------
# Pool driver (reuses ray_app helpers)
# ---------------------------------------------------------------------------

def run_annotator_pool(
    *,
    run_id: str,
    instance_ids: list[str],
    task_data_by_id: dict[str, dict],
    cfg: dict[str, Any],
    data_path_base: str,
    num_actors: int,
    ray_job_id: str = "local",
    gcs_client=None,
    actor_env_vars: dict[str, str] | None = None,
    heartbeat_interval_s: float = 30.0,
    local_only: bool = False,
    attempt: int = 1,
) -> None:
    """Dispatch annotator tasks via a Ray actor pool (or sequentially)."""
    client = gcs_client or default_gcs_client()
    heartbeat = HeartbeatWriter(
        run_id=run_id, total=len(instance_ids), attempt=attempt,
        ray_job_id=ray_job_id, client=client,
        interval_s=heartbeat_interval_s,
    )

    try:
        import ray  # type: ignore[import-not-found]
        ray_available = True
    except ImportError:
        ray_available = False

    use_local = local_only or not ray_available

    heartbeat.start()
    try:
        if use_local:
            if actor_env_vars:
                _apply_actor_env_local(actor_env_vars)
            # DEV-1604: the sequential/local path has no AnnotatorActor, so start
            # the bridge here (the remote actor does it in __init__) — else a
            # doubleword/* local annotator run has no base-url override and fails
            # before the SDK session (Codex).
            _maybe_ensure_bridge(cfg)
            for iid in instance_ids:
                _run_one_task(
                    task_data=task_data_by_id[iid],
                    cfg=cfg,
                    run_id=run_id,
                    data_path_base=data_path_base,
                    gcs_client=client,
                    attempt=attempt,
                )
                heartbeat.tick_done()
        else:
            ActorCls = _with_actor_env(_build_annotator_actor_class(), actor_env_vars)
            actors = [
                ActorCls.remote(cfg, run_id, data_path_base, attempt)
                for _ in range(num_actors)
            ]
            _run_with_actors(
                actors=actors,
                instance_ids=instance_ids,
                task_data_by_id=task_data_by_id,
                run_id=run_id,
                attempt=attempt,
                gcs_client=client,
                heartbeat=heartbeat,
                actor_factory=lambda: ActorCls.remote(cfg, run_id, data_path_base, attempt),
                benchmark=cfg["benchmark"],
            )
        heartbeat.stop_and_flush(terminal_state="done")
    except Exception:
        heartbeat.stop_and_flush(terminal_state="error")
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Annotator Ray worker pool")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ray-job-id", default=_default_ray_job_id())
    p.add_argument("--benchmark", required=True)
    p.add_argument("--model", default="anthropic/claude-opus-4-7")
    p.add_argument("--effort", default="medium",
                   choices=("low", "medium", "high"))
    p.add_argument("--override", action="store_true")
    p.add_argument("--num-actors", type=int, default=4)
    p.add_argument("--benchmark-data-prefix", default=None)
    p.add_argument("--data-path-base", default=None)
    p.add_argument("--instance-ids", required=True, help="comma-separated list")
    p.add_argument(
        "--secrets-file", default=None,
        help="path to a JSON file of env vars (API keys) applied per-actor",
    )
    p.add_argument("--attempt", type=int, default=1,
                   help="attempt number (1-based); used in the GCS blob name")
    # DEV-1604: recycled --subscription-auth flag (z.ai endpoint selector;
    # default no-subscription = per-token bridge). Doubleword auto-bridges.
    p.add_argument("--subscription-auth", action=argparse.BooleanOptionalAction,
                   default=False, dest="subscription_auth")
    args = p.parse_args(argv)

    args.benchmark = get_benchmark(args.benchmark).name

    actor_env_vars = _load_secrets_file(args.secrets_file)
    instance_ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]

    download_benchmark_data(
        {"dataset": args.benchmark, "benchmark_data_prefix": args.benchmark_data_prefix},
    )

    task_data_by_id = _load_annotator_task_data(
        instance_ids, benchmark=args.benchmark,
    )

    _b = get_benchmark(args.benchmark)
    data_path_base = (
        args.data_path_base
        or (str(paths.benchmark_data_root(_b)) if os.environ.get("BIRD_BENCHMARKS_ROOT") else None)
        or _b.container_data_dir
    )

    cfg: dict[str, Any] = {
        "benchmark": args.benchmark,
        "dataset": args.benchmark,
        "benchmark_data_prefix": args.benchmark_data_prefix or "",
        "model": args.model,
        # DEV-1604: `_maybe_ensure_bridge` reads `agent_model` + the recycled
        # `no_subscription_auth` flag to decide the z.ai/Doubleword bridge.
        "agent_model": args.model,
        "framework": "annotator",
        "no_subscription_auth": not args.subscription_auth,
        "effort": args.effort,
        "override": args.override,
    }

    run_annotator_pool(
        run_id=args.run_id,
        instance_ids=instance_ids,
        task_data_by_id=task_data_by_id,
        cfg=cfg,
        data_path_base=data_path_base,
        num_actors=args.num_actors,
        ray_job_id=args.ray_job_id,
        actor_env_vars=actor_env_vars,
        attempt=args.attempt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
