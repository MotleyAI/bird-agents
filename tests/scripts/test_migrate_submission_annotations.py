"""DEV-1591 regression: the migration script must not mask write/IO failures
as legacy-schema fallbacks.

The raw-copy fallback is scoped to ``SubmissionAnnotation.model_validate``
failures only. A failure inside ``write_run_annotation`` (provenance copy /
file I/O) must increment ``errors`` and make the script exit non-zero, rather
than being swallowed into a raw copy that prints "did not validate" and exits 0.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts"
    / "migrate_submission_annotations_to_runs.py"
)
_spec = importlib.util.spec_from_file_location(
    "migrate_submission_annotations_to_runs", SCRIPT,
)
migrate = importlib.util.module_from_spec(_spec)
sys.modules["migrate_submission_annotations_to_runs"] = migrate
_spec.loader.exec_module(migrate)


def _ann_dict(*, iid="alien_1", db="alien", run_id="r1") -> dict:
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": iid,
        "selected_database": db,
        "task_annotation_ref": f"annotations/mini-interact/{db}/{iid}.task.json",
        "annotated_by": "test",
        "annotated_at": "2026-06-01T10:00:00+00:00",
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{iid}/attempt-1.json",
            "predicted_row_count": 1,
            "duration_s": 1.0,
            "cost_usd_agent": 0.0,
            "cost_usd_user_sim": 0.0,
            "n_agent_turns": 1,
            "n_ask_user_calls": 0,
        },
        "evaluation": {
            "phase1_against_original_gold": "pass",
            "phase1_against_audited_primary": "pass",
            "phase1_against_any_audited_variant": "pass",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "novel_reading_judgment": None,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "correct",
            "matched_variant_id": "primary",
            "rationale": "",
        },
        "failure_classification": {
            "primary": "no_fail",
            "secondary": [],
            "agent_at_fault": False,
            "remediation_target": "other",
            "remediation_text": "",
            "details": "",
        },
        "decision_point": None,
        "user_sim_interaction": {
            "n_asks": 0, "key_responses": [],
            "disclosed_resolutions": [], "undisclosed_resolutions": [],
        },
        "original_gold_annotated_correct": True,
    }


@pytest.fixture
def roots(tmp_path, monkeypatch):
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setenv("BIRD_ANNOTATIONS_ROOT", str(ann))
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(results))
    return ann, runs, results


def _write_source(ann_root: Path, *, db="alien", iid="alien_1",
                  run_id="r1", content: dict) -> Path:
    src = ann_root / "mini-interact" / db / f"{iid}.submission.{run_id}.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(content))
    return src


def test_valid_record_migrates_and_exits_zero(roots):
    ann_root, runs_root, _results = roots
    _write_source(ann_root, content=_ann_dict())
    rc = migrate.main([])
    assert rc == 0
    dest = runs_root / "mini-interact" / "alien" / "alien_1" / "r1.json"
    assert dest.exists()


def test_legacy_unvalidatable_record_raw_copied_exits_zero(roots):
    """A genuine legacy record that fails model_validate is raw-copied and the
    run still succeeds (rc 0)."""
    ann_root, runs_root, _results = roots
    _write_source(ann_root, content={"kind": "submission_annotation",
                                      "totally": "legacy"})
    rc = migrate.main([])
    assert rc == 0
    dest = runs_root / "mini-interact" / "alien" / "alien_1" / "r1.json"
    assert json.loads(dest.read_text())["totally"] == "legacy"


def test_write_failure_is_not_masked(roots, monkeypatch):
    """A write/provenance failure (valid record) increments errors and makes
    the script exit non-zero — it is NOT swallowed as a legacy raw copy."""
    ann_root, runs_root, _results = roots
    _write_source(ann_root, content=_ann_dict())

    def _boom(*_a, **_k):
        raise OSError("disk full")

    # The script does `from ...annotation_io import write_run_annotation`
    # inside main() at call time, so patching the source module is what binds.
    import bird_interact_agents.eval.annotation_io as aio
    monkeypatch.setattr(aio, "write_run_annotation", _boom)

    rc = migrate.main([])
    assert rc == 1
    dest = runs_root / "mini-interact" / "alien" / "alien_1" / "r1.json"
    # The failed write must NOT have been masked by a raw copy.
    assert not dest.exists()


def test_raw_copy_write_failure_continues(roots, monkeypatch):
    """An IO failure during the LEGACY raw-copy path (record fails
    model_validate) increments errors and CONTINUES to the next record rather
    than aborting the whole loop — the error-continue contract applies to the
    raw copy too. A second, valid record must still be migrated."""
    ann_root, runs_root, _results = roots
    # alien_1: a genuine legacy record (fails model_validate) → raw-copy path,
    # whose write we make fail. alien_2: a valid record that must still migrate.
    _write_source(ann_root, content={"kind": "submission_annotation",
                                     "totally": "legacy"})
    _write_source(ann_root, iid="alien_2", run_id="r2",
                  content=_ann_dict(iid="alien_2", run_id="r2"))

    legacy_dest = runs_root / "mini-interact" / "alien" / "alien_1" / "r1.json"
    orig_write_text = Path.write_text

    def _selective(self, *a, **k):
        # Only the first record's raw-copy write fails; everything else (incl.
        # the second record's write) goes through normally.
        if self == legacy_dest:
            raise OSError("disk full")
        return orig_write_text(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _selective)

    rc = migrate.main([])
    assert rc == 1                       # the failed record kept exit non-zero
    assert not legacy_dest.exists()      # its raw copy did not land
    # ...but the loop continued and migrated the second, valid record.
    assert (runs_root / "mini-interact" / "alien" / "alien_2"
            / "r2.json").exists()
