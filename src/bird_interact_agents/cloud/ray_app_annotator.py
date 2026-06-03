"""In-cluster Ray driver for the annotator agent (DEV-1518).

Invoked via `ray job submit -- python ray_app_annotator.py <args>` from the
laptop-side `bird-interact-cloud annotate` command.

Worker contract:
* Skip if both stable blobs exist (unless --override).
* Run one task → write 4 GCS paths on success (run-specific + stable for
  task_annotation and audited_gold_variants).
* Write attempt-1.json for every outcome (annotated / skipped / error) so
  list_attempts() / wait_until_done() work unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from bird_interact_agents.cloud import gcs as _gcs
from bird_interact_agents.agents.annotator.agent import AnnotatorResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS client default (overridable in tests via monkeypatch)
# ---------------------------------------------------------------------------

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

    return asyncio.get_event_loop().run_until_complete(
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
        try:
            ann_blob = _gcs.stable_task_annotation_blob(benchmark, db, instance_id)
            var_blob = _gcs.stable_audited_gold_variants_blob(benchmark, db, instance_id)
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
            logger.info("[%s] skipping — both stable blobs exist", instance_id)
            attempt_row = {
                "instance_id": instance_id,
                "status": "skipped",
                "duration_s": time.monotonic() - t0,
            }
            _write_attempt(run_id, instance_id, attempt_row, client=client)
            return

    # Run the agent.
    try:
        result = _run_agent(task_data=task_data, cfg=cfg, data_path_base=data_path_base)
    except Exception as exc:
        logger.error("[%s] agent raised: %s", instance_id, exc)
        attempt_row = {
            "instance_id": instance_id,
            "status": "error",
            "error": str(exc),
            "duration_s": time.monotonic() - t0,
        }
        _write_attempt(run_id, instance_id, attempt_row, client=client)
        return

    if result.error:
        logger.warning("[%s] agent returned error: %s", instance_id, result.error)
        attempt_row = {
            "instance_id": instance_id,
            "status": "error",
            "error": result.error,
            "duration_s": result.duration_s,
        }
        _write_attempt(run_id, instance_id, attempt_row, client=client)
        return

    # Success — write 4 GCS paths.
    ann = result.task_annotation
    variants = result.audited_gold_variants

    try:
        _gcs.write_task_annotation(run_id, instance_id, ann, client=client)
        _gcs.write_audited_gold_variants(run_id, instance_id, variants, client=client)
        _gcs.write_stable_task_annotation(benchmark, db, instance_id, ann, client=client)
        _gcs.write_stable_audited_gold_variants(benchmark, db, instance_id, variants, client=client)
    except Exception as exc:
        logger.error("[%s] GCS write failed after annotation: %s", instance_id, exc)
        attempt_row = {
            "instance_id": instance_id,
            "status": "error",
            "error": f"GCS write failed: {exc}",
            "duration_s": result.duration_s,
        }
        _write_attempt(run_id, instance_id, attempt_row, client=client)
        return

    attempt_row = {
        "instance_id": instance_id,
        "status": "annotated",
        "duration_s": result.duration_s,
    }
    _write_attempt(run_id, instance_id, attempt_row, client=client)


def _write_attempt(
    run_id: str,
    instance_id: str,
    row: dict,
    *,
    client=None,
) -> None:
    blob_name = f"runs/{run_id}/rows/{instance_id}/attempt-1.json"
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
    """Load plain task data for the annotator (no audited-gold overlay)."""
    from bird_interact_agents import paths
    from bird_interact_agents.harness import load_benchmark_tasks

    rows = load_benchmark_tasks(
        benchmark,
        str(paths.benchmark_data_file(benchmark)),
        None,
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
        def __init__(self, cfg: dict[str, Any], run_id: str, data_path_base: str):
            self.cfg = cfg
            self.run_id = run_id
            self.data_path_base = data_path_base
            self.gcs_client = default_gcs_client()

        def run_one(self, task_data: dict) -> None:
            _run_one_task(
                task_data=task_data,
                cfg=self.cfg,
                run_id=self.run_id,
                data_path_base=self.data_path_base,
                gcs_client=self.gcs_client,
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
) -> None:
    """Dispatch annotator tasks via a Ray actor pool (or sequentially)."""
    from bird_interact_agents.cloud.ray_app import (
        HeartbeatWriter,
        _apply_actor_env_local,
        _run_with_actors,
        _with_actor_env,
    )

    client = gcs_client or default_gcs_client()
    heartbeat = HeartbeatWriter(
        run_id=run_id, total=len(instance_ids), attempt=1,
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
            for iid in instance_ids:
                _run_one_task(
                    task_data=task_data_by_id[iid],
                    cfg=cfg,
                    run_id=run_id,
                    data_path_base=data_path_base,
                    gcs_client=client,
                )
                heartbeat.tick_done()
        else:
            ActorCls = _with_actor_env(_build_annotator_actor_class(), actor_env_vars)
            actors = [
                ActorCls.remote(cfg, run_id, data_path_base)
                for _ in range(num_actors)
            ]
            _run_with_actors(
                actors=actors,
                instance_ids=instance_ids,
                task_data_by_id=task_data_by_id,
                run_id=run_id,
                attempt=1,
                gcs_client=client,
                heartbeat=heartbeat,
                actor_factory=lambda: ActorCls.remote(cfg, run_id, data_path_base),
            )
        heartbeat.stop_and_flush(terminal_state="done")
    except Exception:
        heartbeat.stop_and_flush(terminal_state="error")
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    from bird_interact_agents.benchmark import get_benchmark

    p = argparse.ArgumentParser(description="Annotator Ray worker pool")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ray-job-id", default="unknown")
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
    args = p.parse_args(argv)

    args.benchmark = get_benchmark(args.benchmark).name

    from bird_interact_agents.cloud.ray_app import (
        _load_secrets_file,
        download_benchmark_data,
    )

    actor_env_vars = _load_secrets_file(args.secrets_file)
    instance_ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]

    download_benchmark_data(
        {"dataset": args.benchmark, "benchmark_data_prefix": args.benchmark_data_prefix},
    )

    task_data_by_id = _load_annotator_task_data(instance_ids, benchmark=args.benchmark)

    _b = get_benchmark(args.benchmark)
    data_path_base = (
        args.data_path_base
        or os.environ.get(_b.data_root_env)
        or _b.container_data_dir
    )

    cfg: dict[str, Any] = {
        "benchmark": args.benchmark,
        "model": args.model,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
