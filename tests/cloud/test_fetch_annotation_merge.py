"""DEV-1515: `bird-interact-cloud fetch` merges per-row
submission_annotation.json files into `<main_checkout>/annotations/`.

Contract:
* Walks `<results>/cloud/<run-id>/rows/<instance>/submission_annotation.json`
  (downloaded from GCS) and merges to
  `<main_checkout>/annotations/<benchmark>/<db>/<instance>.submission.<run-id>.json`.
* No-overwrite-if-present (mirrors slayer_models_otf/ merge pattern).
* Schema-validates each candidate; rejects malformed files and reports
  them in the audit log.
* Audit log lands at `<results>/cloud/<run-id>/annotation_merge_report.json`.
* The new ``gcs.submission_annotation_blob`` / ``write_submission_annotation``
  / ``read_submission_annotation`` shape mirrors `row_blob` /
  `write_row` / `read_row`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# GCS blob path + helpers
# ---------------------------------------------------------------------------


def test_gcs_submission_annotation_blob_path():
    from bird_interact_agents.cloud import gcs

    blob = gcs.submission_annotation_blob("r-1", "alien_1")
    assert blob == "runs/r-1/rows/alien_1/submission_annotation.json"


def test_gcs_submission_annotation_roundtrip_uses_canonical_blob_name(
    fake_gcs_bucket,
):
    """Upload then download MUST use the exact `submission_annotation_blob`
    path — otherwise the worker write and fetch read can drift apart."""
    from bird_interact_agents.cloud import gcs

    client, store = fake_gcs_bucket
    payload = _valid_submission_annotation_dict("alien_1")
    gcs.write_submission_annotation(
        "r-1", "alien_1", payload, client=client,
    )
    assert "runs/r-1/rows/alien_1/submission_annotation.json" in store
    read = gcs.read_submission_annotation("r-1", "alien_1", client=client)
    assert read["instance_id"] == "alien_1"


# ---------------------------------------------------------------------------
# Merge — happy path + no-overwrite + schema validation
# ---------------------------------------------------------------------------


def _valid_submission_annotation_dict(instance_id: str = "alien_1") -> dict:
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": "alien",
        "task_annotation_ref": f"annotations/mini_interact/alien/{instance_id}.task.json",
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
            "primary": "other",
            "agent_at_fault": False,
            "remediation_target": "other",
        },
    }


def test_merge_writes_annotation_to_main_checkout(tmp_path, monkeypatch):
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    main_checkout.mkdir()
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows" / "alien_1"
    rows.mkdir(parents=True)
    (rows / "submission_annotation.json").write_text(
        json.dumps(_valid_submission_annotation_dict("alien_1"))
    )

    report = merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )

    dest = (
        main_checkout / "annotations" / "mini_interact" / "alien"
        / "alien_1.submission.r1.json"
    )
    assert dest.exists()
    assert report.merged == 1
    assert report.skipped_existing == 0
    assert report.rejected_invalid == 0


def test_merge_no_overwrite_if_present(tmp_path, monkeypatch):
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    dest_dir = main_checkout / "annotations" / "mini_interact" / "alien"
    dest_dir.mkdir(parents=True)
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    pre = _valid_submission_annotation_dict("alien_1")
    pre["annotated_by"] = "human-pre-existing"
    (dest_dir / "alien_1.submission.r1.json").write_text(json.dumps(pre))

    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows" / "alien_1"
    rows.mkdir(parents=True)
    fresh = _valid_submission_annotation_dict("alien_1")
    fresh["annotated_by"] = "auto-fresh"
    (rows / "submission_annotation.json").write_text(json.dumps(fresh))

    report = merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )
    assert report.merged == 0
    assert report.skipped_existing == 1

    surviving = json.loads(
        (dest_dir / "alien_1.submission.r1.json").read_text()
    )
    assert surviving["annotated_by"] == "human-pre-existing"


def test_merge_rejects_schema_invalid_file(tmp_path, monkeypatch):
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows" / "alien_1"
    rows.mkdir(parents=True)
    (rows / "submission_annotation.json").write_text(
        '{"this": "is", "not": "a valid SubmissionAnnotation"}'
    )

    report = merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )
    assert report.merged == 0
    assert report.rejected_invalid == 1
    # Destination must NOT have been created from invalid content.
    assert not (
        main_checkout / "annotations" / "mini_interact" / "alien"
        / "alien_1.submission.r1.json"
    ).exists()


def test_merge_writes_audit_report(tmp_path, monkeypatch):
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    downloaded = tmp_path / "downloaded"
    rows_dir = downloaded / "rows"
    for inst in ("a_1", "a_2"):
        d = rows_dir / inst
        d.mkdir(parents=True)
        (d / "submission_annotation.json").write_text(
            json.dumps(_valid_submission_annotation_dict(inst))
        )

    merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )
    audit = downloaded / "annotation_merge_report.json"
    assert audit.exists()
    body = json.loads(audit.read_text())
    assert body["merged"] == 2
    assert body["run_id"] == "r1"


# ---------------------------------------------------------------------------
# Codex r7: resubmit-aware overwrite. The canonical
# ``submission_annotation_path`` does NOT carry the per-task ``attempt``
# in the filename, so a partial fetch followed by a resubmit (which
# reuses the same ``run_id`` and bumps ``attempt``) would otherwise pin
# attempt-1's annotation forever — while ``eval.json`` / ``results.db``
# reflect attempt-2. The merge now parses ``submission.trajectory_path``
# (``rows/<iid>/attempt-N.json``) on both src and dest and overwrites
# ONLY when the new attempt is strictly newer.
# ---------------------------------------------------------------------------


def _make_dict_with_attempt(instance_id: str, attempt: int) -> dict:
    body = _valid_submission_annotation_dict(instance_id)
    body["submission"]["trajectory_path"] = (
        f"rows/{instance_id}/attempt-{attempt}.json"
    )
    body["annotated_by"] = f"attempt-{attempt}-grader"
    return body


def test_merge_overwrites_when_new_attempt_strictly_newer(tmp_path, monkeypatch):
    """Resubmit pushes attempt-2's annotation; the existing dest is
    attempt-1. The merge MUST overwrite and bump the
    ``overwritten_newer_attempt`` counter."""
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    dest_dir = main_checkout / "annotations" / "mini_interact" / "alien"
    dest_dir.mkdir(parents=True)
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    # Pre-existing dest from a prior partial fetch — attempt-1.
    (dest_dir / "alien_1.submission.r1.json").write_text(
        json.dumps(_make_dict_with_attempt("alien_1", 1)),
    )

    # Newly downloaded run dir carries attempt-2's annotation.
    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows" / "alien_1"
    rows.mkdir(parents=True)
    (rows / "submission_annotation.json").write_text(
        json.dumps(_make_dict_with_attempt("alien_1", 2)),
    )

    report = merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )

    assert report.overwritten_newer_attempt == 1, report
    assert report.skipped_existing == 0, report
    assert report.merged == 0, report
    surviving = json.loads(
        (dest_dir / "alien_1.submission.r1.json").read_text(),
    )
    assert surviving["annotated_by"] == "attempt-2-grader"
    assert surviving["submission"]["trajectory_path"] == (
        "rows/alien_1/attempt-2.json"
    )


def test_merge_does_not_overwrite_when_new_attempt_is_older_or_equal(tmp_path, monkeypatch):
    """Symmetric safety case: attempt-2 already on disk, attempt-1
    being merged — MUST keep attempt-2 (no regression to older row)."""
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    dest_dir = main_checkout / "annotations" / "mini_interact" / "alien"
    dest_dir.mkdir(parents=True)
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    (dest_dir / "alien_1.submission.r1.json").write_text(
        json.dumps(_make_dict_with_attempt("alien_1", 2)),
    )

    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows" / "alien_1"
    rows.mkdir(parents=True)
    (rows / "submission_annotation.json").write_text(
        json.dumps(_make_dict_with_attempt("alien_1", 1)),
    )

    report = merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )

    assert report.overwritten_newer_attempt == 0, report
    assert report.skipped_existing == 1, report
    surviving = json.loads(
        (dest_dir / "alien_1.submission.r1.json").read_text(),
    )
    assert surviving["annotated_by"] == "attempt-2-grader"


def test_merge_skips_when_attempts_equal(tmp_path, monkeypatch):
    """Equal attempts — preserve existing (no-op repeated fetch)."""
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import (
        merge_submission_annotations,
    )

    main_checkout = tmp_path / "checkout"
    dest_dir = main_checkout / "annotations" / "mini_interact" / "alien"
    dest_dir.mkdir(parents=True)
    fake_ann_root = main_checkout / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann_root)
    pre = _make_dict_with_attempt("alien_1", 1)
    pre["annotated_by"] = "human-pre-existing"
    (dest_dir / "alien_1.submission.r1.json").write_text(json.dumps(pre))

    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows" / "alien_1"
    rows.mkdir(parents=True)
    fresh = _make_dict_with_attempt("alien_1", 1)
    fresh["annotated_by"] = "auto-fresh"
    (rows / "submission_annotation.json").write_text(json.dumps(fresh))

    report = merge_submission_annotations(
        downloaded_run_dir=downloaded,
        run_id="r1",
        benchmark="mini-interact",
    )
    assert report.overwritten_newer_attempt == 0, report
    assert report.skipped_existing == 1, report
    surviving = json.loads(
        (dest_dir / "alien_1.submission.r1.json").read_text(),
    )
    assert surviving["annotated_by"] == "human-pre-existing"


def test_attempt_from_trajectory_path_parses_and_defaults():
    """Pin the parsing helper's contract — unparseable input returns
    None so the caller can fall back to no-overwrite."""
    from bird_interact_agents.cloud.post_run_merge import (
        _attempt_from_trajectory_path,
    )

    assert _attempt_from_trajectory_path("rows/alien_1/attempt-1.json") == 1
    assert _attempt_from_trajectory_path("rows/alien_1/attempt-42.json") == 42
    assert _attempt_from_trajectory_path(None) is None
    assert _attempt_from_trajectory_path("") is None
    assert _attempt_from_trajectory_path("rows/alien_1/something_else.json") is None
