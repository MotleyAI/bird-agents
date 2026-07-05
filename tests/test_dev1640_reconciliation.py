"""DEV-1640: parent-side reconciliation after the process pool drains.

Graded AND never-submitted tasks already populate the ``runs/`` golden
store from inside the worker (``grade_one_submission`` /
``write_failed_submission_annotation`` both call ``_write_to_runs``). The
ONE gap process isolation introduces is a HARD crash: a worker process
that dies (segfault / OOM-kill / ``os._exit``) before it can persist
anything leaves neither a row blob nor a ``runs/`` entry — so that task
would silently vanish from ``results.db`` AND the cascade denominator.

``reconcile_local_run`` closes exactly that gap: for any task missing its
row blob, it writes an error row (so ``collate`` counts it) and
writes/OVERWRITES a fail-everything ``runs/`` annotation (so the cascade
counts it, and any partial success verdict left by a crash-after-grade is
repaired). Tasks that DID persist a row are left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_interact_agents import local_pool
from bird_interact_agents import paths as paths_mod
from bird_interact_agents.eval.annotation_io import (
    read_submission_annotation,
    run_annotation_path,
    write_run_annotation,
)
from bird_interact_agents.eval.cascading_report import emit_cascading_eval_json
from bird_interact_agents.eval.grade_in_place import (
    write_failed_submission_annotation,
)

BENCHMARK = "mini-interact"
RUN_ID = "localrun-abc123"
ATTEMPT = 1


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(paths_mod, "runs_root", lambda: root)
    return root


def _seed_row_blob(run_dir: Path, iid: str, payload: dict) -> None:
    d = run_dir / "rows" / iid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"attempt-{ATTEMPT}.json").write_text(json.dumps(payload))


def _seed_runs_annotation(run_dir: Path, iid: str, db: str, sentinel: str) -> None:
    """Seed a schema-valid runs/ annotation carrying a sentinel detail so a
    test can prove reconcile did / did not overwrite it."""
    tmp_rows = run_dir / "_seed_rows"
    p = write_failed_submission_annotation(
        rows_dir=tmp_rows, instance_id=iid, selected_database=db,
        benchmark=BENCHMARK, run_id=RUN_ID,
        trajectory_path=f"rows/{iid}/attempt-{ATTEMPT}.json",
        failure_details=sentinel,
    )
    ann = read_submission_annotation(p)
    write_run_annotation(
        ann,
        run_annotation_path(benchmark=BENCHMARK, selected_database=db,
                            instance_id=iid, run_id=RUN_ID),
        benchmark=BENCHMARK, run_id=RUN_ID, allow_manifest_fallback=False,
    )


def test_reconcile_hard_crash_writes_error_row_and_runs_fail(tmp_path, runs_root):
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    td = {"instance_id": "crash1", "selected_database": "db_a"}

    local_pool.reconcile_local_run(
        run_dir=run_dir, tasks=[td], benchmark=BENCHMARK,
        run_id=RUN_ID, attempt=ATTEMPT,
    )
    row_path = run_dir / "rows" / "crash1" / f"attempt-{ATTEMPT}.json"
    assert row_path.exists()
    row = json.loads(row_path.read_text())
    assert row["phase1_passed"] is False
    assert row.get("error")

    dest = run_annotation_path(
        benchmark=BENCHMARK, selected_database="db_a", instance_id="crash1", run_id=RUN_ID,
    )
    assert dest.exists()
    ann = read_submission_annotation(dest)
    assert ann.evaluation.verdict == "eval_failed"
    assert ann.evaluation.phase1_against_original_gold == "fail"


def test_reconcile_hard_crash_overwrites_partial_success_annotation(tmp_path, runs_root):
    """Crash-after-grade: runs/ holds a stale (success) annotation but no
    row blob -> reconcile OVERWRITES runs/ to fail so row + cascade agree."""
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    td = {"instance_id": "partial1", "selected_database": "db_a"}
    _seed_runs_annotation(run_dir, "partial1", "db_a", sentinel="STALE_SUCCESS")

    local_pool.reconcile_local_run(
        run_dir=run_dir, tasks=[td], benchmark=BENCHMARK,
        run_id=RUN_ID, attempt=ATTEMPT,
    )
    dest = run_annotation_path(
        benchmark=BENCHMARK, selected_database="db_a", instance_id="partial1", run_id=RUN_ID,
    )
    ann = read_submission_annotation(dest)
    assert ann.evaluation.verdict == "eval_failed"
    assert ann.evaluation.phase1_against_original_gold == "fail"
    # Proof of OVERWRITE: the seeded sentinel is gone, replaced by the
    # crash-reconciliation detail.
    assert ann.failure_classification.details != "STALE_SUCCESS"
    assert "crashed" in ann.failure_classification.details.lower()


def test_reconcile_leaves_persisted_task_untouched(tmp_path, runs_root):
    """A task that persisted its row (graded or never-submitted, both write
    runs/ from inside the worker) must be left exactly as the worker wrote
    it — reconcile only fills the hard-crash gap."""
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    td = {"instance_id": "done1", "selected_database": "db_a"}
    _seed_row_blob(run_dir, "done1", {"instance_id": "done1", "database": "db_a"})
    _seed_runs_annotation(run_dir, "done1", "db_a", sentinel="WORKER_VERDICT")

    local_pool.reconcile_local_run(
        run_dir=run_dir, tasks=[td], benchmark=BENCHMARK,
        run_id=RUN_ID, attempt=ATTEMPT,
    )
    dest = run_annotation_path(
        benchmark=BENCHMARK, selected_database="db_a", instance_id="done1", run_id=RUN_ID,
    )
    ann = read_submission_annotation(dest)
    assert ann.failure_classification.details == "WORKER_VERDICT"


def test_reconcile_makes_cascade_denominator_equal_len_tasks(tmp_path, runs_root):
    run_dir = tmp_path / "out"
    run_dir.mkdir()
    tasks = [
        {"instance_id": "crash1", "selected_database": "db_a"},  # no row (crash)
        {"instance_id": "done1", "selected_database": "db_a"},   # row + runs/
    ]
    _seed_row_blob(run_dir, "done1", {"instance_id": "done1", "database": "db_a"})
    _seed_runs_annotation(run_dir, "done1", "db_a", sentinel="ok")

    local_pool.reconcile_local_run(
        run_dir=run_dir, tasks=tasks, benchmark=BENCHMARK,
        run_id=RUN_ID, attempt=ATTEMPT,
    )
    metrics = emit_cascading_eval_json(
        BENCHMARK, RUN_ID, tmp_path / "eval.json", base_metrics={},
    )
    assert metrics["cascading_phase1"]["n_dual_eval_tasks"] == 2
