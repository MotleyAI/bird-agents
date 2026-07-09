"""DEV-1649 regression (Codex #2): a saved edited-models archive under
runs/<bench>/<db>/<iid>/ must be INVISIBLE to the recursive `*.json`
annotation walkers.

iter_run_annotations uses ``root.rglob("*.json")``; the scratch we persist
contains ``_kb_rows.json``. Storing the store as an opaque ``.tar.gz``
(not a loose dir) ensures no ``*.json`` leaks — the walker returns only the
real ``<run_id>.json`` annotation, with no "unreadable" warnings.
"""

from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path


def _ann_dict(instance_id: str, db: str, run_id: str) -> dict:
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "task_annotation_ref": f"annotations/mini-interact/{db}/{instance_id}.task.json",
        "annotated_by": "auto",
        "annotated_at": "2026-07-07T00:00:00+00:00",
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{instance_id}/attempt-1.json",
        },
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "novel_reading_judgment": None,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "agent_miss",
            "matched_variant_id": None,
            "rationale": "",
        },
        "failure_classification": {
            "primary": "agent_miss",
            "secondary": [],
            "agent_at_fault": True,
            "remediation_target": "agent",
            "details": "",
        },
        "decision_point": None,
        "user_sim_interaction": {
            "n_asks": 0, "key_responses": [],
            "disclosed_resolutions": [], "undisclosed_resolutions": [],
        },
        "original_gold_annotated_correct": True,
    }


def _write_tarball_with_json(dest: Path) -> None:
    """A .tar.gz whose payload includes a _kb_rows.json (which would trip a
    naive walker if it were on the filesystem loose). Stage OUTSIDE the runs/
    tree so nothing loose lands where the walker looks."""
    import tempfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp())
    (staging / "alien").mkdir(parents=True)
    (staging / "alien" / "_kb_rows.json").write_text(json.dumps([{"id": 0}]))
    (staging / "alien" / "memories.yaml").write_text("[]\n")
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(staging / "alien", arcname="alien")


def test_edited_models_archive_invisible_to_walker(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(tmp_path / "runs"))
    from bird_interact_agents.eval.annotation_io import (
        iter_run_annotations,
        run_edited_models_archive,
    )

    iid_dir = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
    iid_dir.mkdir(parents=True)
    (iid_dir / "r1.json").write_text(json.dumps(_ann_dict("alien_1", "alien", "r1")))

    archive = run_edited_models_archive(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1",
    )
    _write_tarball_with_json(archive)

    with caplog.at_level(logging.WARNING):
        results = iter_run_annotations(benchmark="mini-interact")

    paths = {p.name for p, _ in results}
    assert paths == {"r1.json"}
    assert not any(
        "edited_models" in rec.getMessage() or "_kb_rows" in rec.getMessage()
        for rec in caplog.records
    )


def test_edited_models_archive_invisible_to_latest_run_walker(monkeypatch, tmp_path):
    """The sibling recursive walker (latest_run_per_instance) is also safe."""
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(tmp_path / "runs"))
    from bird_interact_agents.eval.annotation_io import (
        latest_run_per_instance,
        run_edited_models_archive,
    )

    iid_dir = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
    iid_dir.mkdir(parents=True)
    (iid_dir / "r1.json").write_text(json.dumps(_ann_dict("alien_1", "alien", "r1")))
    _write_tarball_with_json(
        run_edited_models_archive(
            benchmark="mini-interact", selected_database="alien",
            instance_id="alien_1",
        )
    )

    latest = latest_run_per_instance(benchmark="mini-interact")
    assert ("alien", "alien_1") in latest
