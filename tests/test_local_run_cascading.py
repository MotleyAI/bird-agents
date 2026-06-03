"""DEV-1515: local-run end-to-end cascading-report wiring.

A local (non-cloud) run via ``run.py`` must produce the same
``cascading_phase1`` shape in its ``eval.json`` as a cloud run. The
shared inline grader path (``eval.grade_in_place``) is invoked per task
in both code paths.

Tests stub the executor so no real benchmark DB or LLM API is required.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_grade_in_place_writes_submission_annotation_per_task(tmp_path):
    """The shared helper writes one
    ``<rows_dir>/<instance>/submission_annotation.json`` per task."""
    from bird_interact_agents.eval.grade_in_place import grade_and_write
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    submitted = "SELECT gold"
    original_gold = "SELECT gold"

    class FakeExecutor:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002,ARG005  # noqa: ARG002
            return ([(1,)], ["a"]) if sql == submitted else ([(99,)], ["a"])

    grade_and_write(
        rows_dir=rows_dir,
        instance_id="alien_1",
        benchmark="mini-interact",
        run_id="local-test-run",
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="x",
        ),
        audited_gold_rows=[],
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=FakeExecutor(),
        trajectory_path="rows/alien_1/attempt-1.json",
        cost_usd_agent=0.0, cost_usd_user_sim=0.0,
        duration_s=0.5, n_agent_turns=1, n_ask_user_calls=0,
        predicted_row_count=1,
        llm_judge=None,
    )

    ann_path = rows_dir / "alien_1" / "submission_annotation.json"
    assert ann_path.exists()
    data = json.loads(ann_path.read_text())
    assert data["instance_id"] == "alien_1"
    # Cascade should pass at N1 (predicted == gold).
    assert data["evaluation"]["phase1_against_original_gold"] == "pass"


def test_local_run_eval_json_has_cascading_block(tmp_path):
    """End-to-end: simulate a 2-task local run; final eval.json carries
    cascading_phase1 derived from the per-row annotations."""
    from bird_interact_agents.eval.cascading_report import (
        emit_cascading_eval_json,
    )
    from bird_interact_agents.eval.grade_in_place import grade_and_write
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    class FakePass:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002,ARG005  # noqa: ARG002
            return ([(1,)], ["a"])
    class FakeFail:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002,ARG005  # noqa: ARG002
            if "predicted" in sql:
                return ([(99,)], ["a"])
            return ([(1,)], ["a"])

    for inst_id, exe in (
        ("t_pass", FakePass()),
        ("t_fail", FakeFail()),
    ):
        grade_and_write(
            rows_dir=rows_dir,
            instance_id=inst_id,
            benchmark="mini-interact",
            run_id="local-test-run",
            task_annotation=implicit_task_annotation(
                instance_id=inst_id, selected_database="alien",
                benchmark="mini-interact", amb_user_query="x",
            ),
            audited_gold_rows=[],
            original_sol_sql=["SELECT gold"],
            submitted_sql="SELECT predicted",
            db_path=Path("/dev/null"),
            conn=None,
            executor=exe,
            trajectory_path=f"rows/{inst_id}/attempt-1.json",
            cost_usd_agent=0.0, cost_usd_user_sim=0.0,
            duration_s=0.0, n_agent_turns=0, n_ask_user_calls=0,
            predicted_row_count=1,
            llm_judge=None,
        )

    out = tmp_path / "eval.json"
    emit_cascading_eval_json(
        rows_dir, out, base_metrics={"phase1_count": 1, "phase1_rate": 0.5},
    )
    metrics = json.loads(out.read_text())
    assert metrics["cascading_phase1"]["counts"]["n1"] == 1
    assert metrics["cascading_phase1"]["n_dual_eval_tasks"] == 2
