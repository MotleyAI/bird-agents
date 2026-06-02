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
import time

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
        ann_blob = _gcs.stable_task_annotation_blob(benchmark, db, instance_id)
        var_blob = _gcs.stable_audited_gold_variants_blob(benchmark, db, instance_id)
        if _gcs.blob_exists(ann_blob, client=client) and _gcs.blob_exists(var_blob, client=client):
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

    _gcs.write_task_annotation(run_id, instance_id, ann, client=client)
    _gcs.write_audited_gold_variants(run_id, instance_id, variants, client=client)
    _gcs.write_stable_task_annotation(benchmark, db, instance_id, ann, client=client)
    _gcs.write_stable_audited_gold_variants(benchmark, db, instance_id, variants, client=client)

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
# CLI entry point (minimal — full arg-parsing is done by cli.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotator Ray worker")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", default="anthropic/claude-opus-4-7")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-path-base", default="/tmp/data")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--task-data-json", required=True)
    args = parser.parse_args()

    task_data = json.loads(args.task_data_json)
    cfg = {
        "benchmark": args.benchmark,
        "model": args.model,
        "effort": args.effort,
        "override": args.override,
    }
    _run_one_task(
        task_data=task_data,
        cfg=cfg,
        run_id=args.run_id,
        data_path_base=args.data_path_base,
    )
