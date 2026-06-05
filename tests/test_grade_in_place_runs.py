"""DEV-1533: grade_and_write dual write to rows_dir + runs/."""
from __future__ import annotations

import json
from pathlib import Path


def test_grade_and_write_also_writes_to_runs(monkeypatch, tmp_path):
    """grade_and_write must write to both rows_dir AND runs/."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_and_write
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation
    from bird_interact_agents.eval.annotation_io import run_annotation_path

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002
            return ([(1,)], ["a"])

    grade_and_write(
        rows_dir=rows_dir,
        instance_id="alien_1",
        benchmark="mini-interact",
        run_id="test-run",
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="x",
        ),
        audited_gold_rows=[],
        original_sol_sql=["SELECT 1"],
        submitted_sql="SELECT 1",
        db_path=Path("/dev/null"),
        conn=None,
        executor=FakeExec(),
        trajectory_path="rows/alien_1/attempt-1.json",
    )

    # rows_dir write (backward compat)
    assert (rows_dir / "alien_1" / "submission_annotation.json").exists()

    # runs/ write (new golden store)
    runs_dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="test-run",
    )
    assert runs_dest.exists()
    ann = json.loads(runs_dest.read_text())
    assert ann["instance_id"] == "alien_1"


def test_grade_and_write_overwrites_runs_on_local_rerun(monkeypatch, tmp_path):
    """Second call with same run_id (local rerun) must UPDATE runs/ annotation.

    The no-overwrite-on-same-attempt guard lives only in
    ``write_run_annotation_no_overwrite`` (used by the cloud fetch merge
    path). ``grade_and_write`` always overwrites so local reruns reusing
    the same output_dir / run_id pick up fresh grades.
    """
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_and_write
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation
    from bird_interact_agents.eval.annotation_io import run_annotation_path

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002
            return ([(1,)], ["a"])

    def _grade(submitted_sql: str = "SELECT 1"):
        grade_and_write(
            rows_dir=rows_dir,
            instance_id="alien_1",
            benchmark="mini-interact",
            run_id="test-run",
            task_annotation=implicit_task_annotation(
                instance_id="alien_1", selected_database="alien",
                benchmark="mini-interact", amb_user_query="x",
            ),
            audited_gold_rows=[],
            original_sol_sql=["SELECT 1"],
            submitted_sql=submitted_sql,
            db_path=Path("/dev/null"),
            conn=None,
            executor=FakeExec(),
            trajectory_path="rows/alien_1/attempt-1.json",
        )

    _grade("SELECT 1")
    runs_dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="test-run",
    )
    first_content = runs_dest.read_text()
    _grade("SELECT 999")
    # local rerun with different SQL — runs/ should be UPDATED (always overwrite)
    assert runs_dest.read_text() != first_content, (
        "grade_and_write must update runs/ on rerun even when run_id is the same"
    )


def test_write_failed_submission_annotation_also_writes_to_runs(monkeypatch, tmp_path):
    """write_failed_submission_annotation must also write to runs/."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import write_failed_submission_annotation
    from bird_interact_agents.eval.annotation_io import run_annotation_path

    rows_dir = tmp_path / "rows"

    write_failed_submission_annotation(
        rows_dir=rows_dir,
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        run_id="test-run",
        trajectory_path="rows/alien_1/attempt-1.json",
        failure_details="test failure",
    )

    runs_dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="test-run",
    )
    assert runs_dest.exists()
    ann = json.loads(runs_dest.read_text())
    assert ann["evaluation"]["verdict"] == "eval_failed"
    assert ann["failure_classification"]["primary"] == "other"


def test_trajectory_sidecar_written_alongside_run_annotation(monkeypatch, tmp_path):
    """grade_and_write should write a .trajectory.json sidecar if attempt file exists."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_and_write
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation
    from bird_interact_agents.eval.annotation_io import run_trajectory_path

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    # Pre-populate the attempt file that the sidecar copies from.
    attempt_dir = rows_dir / "alien_1"
    attempt_dir.mkdir()
    (attempt_dir / "attempt-1.json").write_text(
        json.dumps({"submitted_sql": "SELECT 1", "trajectory": []})
    )

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002
            return ([(1,)], ["a"])

    grade_and_write(
        rows_dir=rows_dir,
        instance_id="alien_1",
        benchmark="mini-interact",
        run_id="test-run",
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="x",
        ),
        audited_gold_rows=[],
        original_sol_sql=["SELECT 1"],
        submitted_sql="SELECT 1",
        db_path=Path("/dev/null"),
        conn=None,
        executor=FakeExec(),
        trajectory_path="rows/alien_1/attempt-1.json",
    )

    traj_dest = run_trajectory_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="test-run",
    )
    assert traj_dest.exists()
    traj = json.loads(traj_dest.read_text())
    assert traj["submitted_sql"] == "SELECT 1"
