"""DEV-1515 review-followup: ``driver.fetch`` must aggregate the per-row
submission_annotation.json files into the ``cascading_phase1`` block on
the downloaded eval.json. Without this the cloud's eval.json carries
neither the legacy dual-eval breakdown (removed in DEV-1515) nor the new
cascade block — published results lose their headline metric.

The actual aggregation logic is tested in
``test_cascading_report.py``; these tests pin the fetch-side wiring (no
op for legacy runs without per-row annotations; happy path rewrites
eval.json; missing annotations surfaced as a side-channel error rather
than swallowed).
"""
from __future__ import annotations

import json
from pathlib import Path


def _valid_submission_annotation_dict(instance_id: str = "alien_1") -> dict:
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": "alien",
        "task_annotation_ref": f"annotations/mini-interact/alien/{instance_id}.task.json",
        "annotated_by": "auto",
        "annotated_at": "2026-05-31",
        "submission": {
            "cloud_run_id": "r1",
            "trajectory_path": f"rows/{instance_id}/attempt-1.json",
        },
        "evaluation": {
            "phase1_against_original_gold": "pass",
            "phase1_against_audited_primary": "pass",
            "phase1_against_any_audited_variant": "pass",
            "verdict": "correct",
        },
        "failure_classification": {
            "primary": "no_fail",
            "agent_at_fault": False,
            "remediation_target": "other",
        },
    }


def _seed_run_dir(
    tmp_path: Path,
    *,
    instances: list[str],
    base_metrics: dict,
) -> Path:
    """Build a `<dest>/` shaped like what `fetch` finishes with: a base
    eval.json from collation + per-row submission_annotation.json files
    from the worker hook."""
    dest = tmp_path / "run_dest"
    dest.mkdir()
    (dest / "eval.json").write_text(json.dumps(base_metrics))
    for iid in instances:
        d = dest / "rows" / iid
        d.mkdir(parents=True)
        (d / "submission_annotation.json").write_text(
            json.dumps(_valid_submission_annotation_dict(iid)),
        )
    return dest


def test_fetch_emit_writes_cascading_phase1_into_eval_json(tmp_path: Path):
    """Happy path: per-row submission_annotation files exist and the
    downloaded eval.json gets a ``cascading_phase1`` block + back-compat
    ``phase1_count`` / ``phase1_rate`` aliases rewritten from N1."""
    from bird_interact_agents.cloud.driver import (
        _emit_cascading_phase1_on_fetch,
    )

    dest = _seed_run_dir(
        tmp_path,
        instances=["alien_1", "alien_2"],
        base_metrics={
            "framework": "claude_sdk",
            "phase1_count": 999,  # stale value the helper should rewrite
            "phase1_rate": 0.99,
        },
    )
    new_metrics = _emit_cascading_phase1_on_fetch(
        dest=dest, metrics={"framework": "claude_sdk"},
    )
    assert "cascading_phase1" in new_metrics, (
        "fetch must emit the cascading_phase1 block when per-row "
        "submission_annotation.json files exist"
    )
    on_disk = json.loads((dest / "eval.json").read_text())
    assert "cascading_phase1" in on_disk
    # phase1_count is the rewritten alias for N1; both rows have N1 pass.
    assert on_disk["cascading_phase1"]["counts"]["n1"] == 2
    assert on_disk["phase1_count"] == 2


def test_fetch_emit_noop_when_no_per_row_annotations(tmp_path: Path):
    """Older runs that pre-date the DEV-1515 worker code won't have
    `<rows>/<inst>/submission_annotation.json` files. The helper must
    skip cleanly rather than fail — those runs publish eval.json
    without the cascading block, which is correct since the per-row
    data isn't available."""
    from bird_interact_agents.cloud.driver import (
        _emit_cascading_phase1_on_fetch,
    )

    dest = tmp_path / "run_dest"
    dest.mkdir()
    (dest / "eval.json").write_text(json.dumps({"framework": "claude_sdk"}))
    # rows/ dir exists but contains an attempt-1.json (legacy worker)
    (dest / "rows" / "alien_1").mkdir(parents=True)
    (dest / "rows" / "alien_1" / "attempt-1.json").write_text("{}")

    new_metrics = _emit_cascading_phase1_on_fetch(
        dest=dest, metrics={"framework": "claude_sdk"},
    )
    assert "cascading_phase1" not in new_metrics
    on_disk = json.loads((dest / "eval.json").read_text())
    assert "cascading_phase1" not in on_disk


def test_fetch_emit_noop_when_eval_json_absent(tmp_path: Path):
    """Defensive: if eval.json was never written (collation failed?)
    we don't materialise one with only the cascading block — the
    headline metrics would be misleading. Skip and let the caller
    surface the collation failure."""
    from bird_interact_agents.cloud.driver import (
        _emit_cascading_phase1_on_fetch,
    )

    dest = tmp_path / "run_dest"
    dest.mkdir()
    d = dest / "rows" / "alien_1"
    d.mkdir(parents=True)
    (d / "submission_annotation.json").write_text(
        json.dumps(_valid_submission_annotation_dict("alien_1")),
    )

    new_metrics = _emit_cascading_phase1_on_fetch(
        dest=dest, metrics={"framework": "claude_sdk"},
    )
    assert "cascading_phase1" not in new_metrics
    assert not (dest / "eval.json").exists()
