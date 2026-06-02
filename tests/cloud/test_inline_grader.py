"""DEV-1515: cloud worker inline grading + per-row write to artefacts.

After ``execute_submit_action`` returns, ray_app.py must call
``grade_in_place.grade_and_write`` so each per-row artefact dir contains
a ``submission_annotation.json`` written by tolerant_grader.

This test isolates the wiring contract — the actual grader logic is
covered in ``tests/test_tolerant_grader_*.py``.
"""
from __future__ import annotations

from pathlib import Path


def test_ray_app_writes_submission_annotation_per_task(monkeypatch, tmp_path):
    """The worker MUST invoke grade_and_write for each task it runs."""
    from bird_interact_agents.cloud import ray_app
    from bird_interact_agents.eval import grade_in_place

    calls: list[dict] = []

    def fake_grade(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        rows_dir = kwargs["rows_dir"]
        instance_id = kwargs["instance_id"]
        d = rows_dir / instance_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "submission_annotation.json").write_text("{}")

    # After DEV-1515 round-4, the per-task grader helper lives in
    # ``grade_in_place`` (canonical location, shared with the local
    # runner); ``ray_app._grade_one_submission`` is now a thin alias.
    # Patch BOTH so the test passes regardless of which lookup path
    # the wiring code uses.
    monkeypatch.setattr(
        grade_in_place, "grade_and_write", fake_grade, raising=True,
    )
    monkeypatch.setattr(
        ray_app, "grade_and_write", fake_grade, raising=True,
    )

    # The simulated submit hook — adapter for ray_app's per-task path.
    # ray_app exposes a `_grade_one_submission(task_data, submitted_sql,
    # rows_dir, run_id, benchmark)` helper that is the integration seam.
    ray_app._grade_one_submission(
        task_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT gold"],
            "original_sol_sql": ["SELECT gold"],
        },
        submitted_sql="SELECT predicted",
        rows_dir=tmp_path,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        conn=None,
    )

    assert len(calls) == 1
    assert (tmp_path / "alien_1" / "submission_annotation.json").exists()


def test_ray_app_does_not_emit_legacy_phase1_passed_fields(monkeypatch, tmp_path):
    """The per-row result dict that ray_app uploads must NOT contain
    the legacy raw bool fields — those have been replaced by the
    submission_annotation path."""
    import inspect
    from bird_interact_agents.cloud import ray_app

    src = inspect.getsource(ray_app)
    assert "phase1_passed_audited" not in src
    assert "phase1_passed_original" not in src


def test_worker_uses_implicit_annotation_when_file_missing(monkeypatch, tmp_path):
    """If no <instance>.task.json exists in the baked annotations dir,
    the worker falls back to implicit_task_annotation IN MEMORY — no
    file gets written under annotations/."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.cloud import ray_app

    # Empty annotations dir.
    annotations_root = tmp_path / "annotations"
    annotations_root.mkdir()
    monkeypatch.setattr(
        paths_mod, "annotations_root", lambda: annotations_root,
    )

    captured: list[dict] = []

    def fake_grade(**kwargs):  # noqa: ANN003
        captured.append(kwargs)
        d = kwargs["rows_dir"] / kwargs["instance_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "submission_annotation.json").write_text("{}")

    from bird_interact_agents.eval import grade_in_place
    monkeypatch.setattr(
        grade_in_place, "grade_and_write", fake_grade, raising=True,
    )
    monkeypatch.setattr(ray_app, "grade_and_write", fake_grade, raising=True)

    ray_app._grade_one_submission(
        task_data={
            "instance_id": "alien_99",
            "selected_database": "alien",
            "sol_sql": ["SELECT gold"],
            "original_sol_sql": ["SELECT gold"],
            "amb_user_query": "x",
        },
        submitted_sql="SELECT predicted",
        rows_dir=tmp_path / "rows",
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        conn=None,
    )

    assert len(captured) == 1
    # No <instance>.task.json was written.
    assert not list(annotations_root.rglob("*.task.json"))
