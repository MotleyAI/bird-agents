"""DEV-1533 / grader-consistency-fixes: harness short-circuit + verdict rename."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_harness_pass_short_circuits_sql_execution(monkeypatch, tmp_path):
    """grade_one_submission with harness_passed=True must skip SQL execution
    and write a correct annotation without touching the (nonexistent) DB."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_one_submission
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    task_data = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "sol_sql": ["SELECT gold"],
        "amb_user_query": "q1",
    }
    rows_dir = tmp_path / "rows"

    out = grade_one_submission(
        task_data=task_data,
        submitted_sql="SELECT agent",
        rows_dir=rows_dir,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/nonexistent/db.sqlite"),  # would raise if touched
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q1",
        ),
        harness_passed=True,
    )
    assert out.exists()
    ann = json.loads(out.read_text())
    assert ann["evaluation"]["verdict"] == "correct"
    assert ann["evaluation"]["phase1_against_original_gold"] == "pass"
    assert ann["evaluation"]["phase1_against_any_audited_variant"] == "pass"
    assert ann["failure_classification"]["primary"] == "no_fail"
    assert ann["evaluation"]["rationale"] == "harness_confirmed"


def test_harness_pass_none_runs_full_grader(monkeypatch, tmp_path):
    """harness_passed=None must run the full grader (mismatching SQL → agent_miss)."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_one_submission
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):
            return ([(1,)], ["a"])

    task_data = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "sol_sql": ["SELECT gold"],
        "amb_user_query": "q",
    }
    out = grade_one_submission(
        task_data=task_data,
        submitted_sql="SELECT 999",
        rows_dir=tmp_path / "rows",
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        conn=None,
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q",
        ),
        harness_passed=None,
    )
    ann = json.loads(out.read_text())
    # Mismatch → agent_miss (not harness_confirmed)
    assert ann["evaluation"]["rationale"] != "harness_confirmed"


def test_harness_pass_false_runs_full_grader(monkeypatch, tmp_path):
    """harness_passed=False must behave the same as harness_passed=None."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_one_submission
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):
            return ([(1,)], ["a"])

    task_data = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "sol_sql": ["SELECT gold"],
        "amb_user_query": "q",
    }
    out = grade_one_submission(
        task_data=task_data,
        submitted_sql="SELECT 999",
        rows_dir=tmp_path / "rows",
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        conn=None,
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q",
        ),
        harness_passed=False,
    )
    ann = json.loads(out.read_text())
    assert ann["evaluation"]["rationale"] != "harness_confirmed"


def test_verdict_catch_all_is_agent_miss():
    """verdict_label_from_cascade must return 'agent_miss' (not 'invalid')
    when all cascade tiers fail."""
    from bird_interact_agents.eval.grade_in_place import verdict_label_from_cascade
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    cascade = CascadeVerdict(
        n1_original_gold=False, n2_audited_primary=False,
        n3_any_audited_variant=False, n4_tie_order=False,
        n5_llm_judge=False, n6_numeric_epsilon=False,
        n7_trailing_whitespace=False, n8_column_order=False,
        n9_case_fold=False,
    )
    assert verdict_label_from_cascade(cascade) == "agent_miss"


def test_verdict_infra_failure_is_eval_failed(monkeypatch, tmp_path):
    """write_failed_submission_annotation must emit verdict='eval_failed'."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import write_failed_submission_annotation
    out = write_failed_submission_annotation(
        rows_dir=tmp_path / "rows",
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        run_id="r1",
        trajectory_path="rows/alien_1/attempt-1.json",
        failure_details="grader raised",
    )
    ann = json.loads(out.read_text())
    assert ann["evaluation"]["verdict"] == "eval_failed"
    assert ann["failure_classification"]["primary"] == "other"


def test_invalid_verdict_auto_migrates_to_agent_miss():
    """Parsing a legacy 'invalid' annotation auto-upgrades to 'agent_miss'."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation

    data = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": "alien_1",
        "selected_database": "alien",
        "task_annotation_ref": "x",
        "annotated_by": "auto",
        "annotated_at": "2026-05-31",
        "submission": {"cloud_run_id": "r1", "trajectory_path": "t"},
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "invalid",  # legacy
            "rationale": "",
        },
        "failure_classification": {
            "primary": "agent_miss",  # not "other" → migrates to agent_miss
            "agent_at_fault": True,
            "remediation_target": "agent",
        },
        "decision_point": None,
        "user_sim_interaction": {"n_asks": 0},
    }
    ann = SubmissionAnnotation.model_validate(data)
    assert ann.evaluation.verdict == "agent_miss"


def test_invalid_verdict_auto_migrates_to_eval_failed():
    """primary='other' + verdict='invalid' → 'eval_failed'."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation

    data = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": "alien_1",
        "selected_database": "alien",
        "task_annotation_ref": "x",
        "annotated_by": "auto",
        "annotated_at": "2026-05-31",
        "submission": {"cloud_run_id": "r1", "trajectory_path": "t"},
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "invalid",  # legacy
            "rationale": "",
        },
        "failure_classification": {
            "primary": "other",  # infra failure → eval_failed
            "agent_at_fault": False,
            "remediation_target": "other",
        },
        "decision_point": None,
        "user_sim_interaction": {"n_asks": 0},
    }
    ann = SubmissionAnnotation.model_validate(data)
    assert ann.evaluation.verdict == "eval_failed"


def test_agent_miss_verdict_validates():
    from bird_interact_agents.eval.annotation_schema import SubmissionEvaluation
    ev = SubmissionEvaluation(
        phase1_against_original_gold="fail",
        phase1_against_audited_primary="fail",
        phase1_against_any_audited_variant="fail",
        correct_up_to_tie_order=False,
        correct_under_numeric_epsilon=False,
        correct_under_trailing_whitespace=False,
        correct_under_column_order=False,
        correct_under_case_fold=False,
        numeric_epsilon=1e-6,
        verdict="agent_miss",
        rationale="",
    )
    assert ev.verdict == "agent_miss"


def test_eval_failed_verdict_validates():
    from bird_interact_agents.eval.annotation_schema import SubmissionEvaluation
    ev = SubmissionEvaluation(
        phase1_against_original_gold="fail",
        phase1_against_audited_primary="fail",
        phase1_against_any_audited_variant="fail",
        correct_up_to_tie_order=False,
        correct_under_numeric_epsilon=False,
        correct_under_trailing_whitespace=False,
        correct_under_column_order=False,
        correct_under_case_fold=False,
        numeric_epsilon=1e-6,
        verdict="eval_failed",
        rationale="",
    )
    assert ev.verdict == "eval_failed"


def test_harness_confirmed_counts_in_cascade_report(monkeypatch, tmp_path):
    """A harness-confirmed annotation must count as N1 pass in cascade."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_one_submission
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    rows_dir = tmp_path / "rows"

    grade_one_submission(
        task_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT 1"],
            "amb_user_query": "q",
        },
        submitted_sql="SELECT 1",
        rows_dir=rows_dir,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/nonexistent/db.sqlite"),
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q",
        ),
        harness_passed=True,
    )

    block = aggregate_cascading_phase1("mini-interact", "r1")
    assert block["counts"]["n1"] == 1
    assert block["rates"]["n1"] == pytest.approx(1.0)


def test_harness_pass_writes_runs_store_even_when_rows_dir_file_exists(
    monkeypatch, tmp_path,
):
    """DEV-1533: a rerun reusing the same rows_dir MUST still populate
    the runs/ golden store on the harness-shortcut path. Pre-fix, an
    existing ``rows_dir/<inst>/submission_annotation.json`` made
    ``_write_harness_confirmed_annotation`` early-return BEFORE writing
    to runs/, so cascade aggregation off the golden store saw a missing
    row even though rows_dir looked fine."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.annotation_io import run_annotation_path
    from bird_interact_agents.eval.grade_in_place import grade_one_submission
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    rows_dir = tmp_path / "rows"
    (rows_dir / "alien_1").mkdir(parents=True)
    # Pre-existing rows_dir submission_annotation — simulates a rerun.
    (rows_dir / "alien_1" / "submission_annotation.json").write_text(
        '{"stale": "from prior run"}'
    )

    grade_one_submission(
        task_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT gold"],
            "amb_user_query": "q",
        },
        submitted_sql="SELECT gold",
        rows_dir=rows_dir,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/nonexistent/db.sqlite"),
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q",
        ),
        harness_passed=True,
    )

    runs_path = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="r1",
    )
    assert runs_path.exists(), (
        "runs/ annotation MUST be written even when rows_dir's "
        "submission_annotation.json already exists; got missing"
    )
    body = json.loads(runs_path.read_text())
    assert body["evaluation"]["rationale"] == "harness_confirmed"


def test_harness_pass_preserves_dev1533_run_result_fields(monkeypatch, tmp_path):
    """DEV-1533: ``submitted_sql``, ``predicted_result`` and ``gold_result``
    must land on the harness-confirmed annotation. Without this, every
    passing row in a run with audited gold loses the run data the PR was
    created to preserve."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import grade_one_submission
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    task_data = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "sol_sql": ["SELECT gold"],
        "amb_user_query": "q1",
    }
    rows_dir = tmp_path / "rows"

    predicted_payload = [[1, "a"], [2, "b"]]
    gold_payload = [[1, "a"], [2, "b"]]

    out = grade_one_submission(
        task_data=task_data,
        submitted_sql="SELECT a, b FROM t",
        rows_dir=rows_dir,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/nonexistent/db.sqlite"),
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q1",
        ),
        harness_passed=True,
        predicted_result=predicted_payload,
        gold_result=gold_payload,
    )
    ann = json.loads(out.read_text())
    assert ann["submitted_sql"] == "SELECT a, b FROM t"
    assert ann["predicted_result"] == predicted_payload
    assert ann["gold_result"] == gold_payload


def test_decode_result_json_normalises_snapshot_dict():
    """DEV-1533 regression: ``predicted_result_json`` on the result row is
    the ``capture_result_snapshot`` shape — a dict with ``sample_rows`` —
    not a flat list. ``decode_result_json`` MUST extract ``sample_rows``
    so the ``Optional[List[Any]]`` schema slot accepts it. Pre-fix the
    decoded dict reached Pydantic and the inline grader raised
    ``ValidationError`` (7 of the cloud-run tasks failed this way)."""
    from bird_interact_agents.eval.grade_in_place import decode_result_json

    snapshot = {
        "columns": [{"name": "a", "type": "int"}, {"name": "b", "type": "str"}],
        "row_count": 2,
        "row_count_truncated": False,
        "sample_rows": [[1, "x"], [2, "y"]],
    }

    # JSON-string of the snapshot dict (the on-wire shape).
    assert decode_result_json(json.dumps(snapshot)) == [[1, "x"], [2, "y"]]
    # Already-decoded dict (defensive against pre-decoded callers).
    assert decode_result_json(snapshot) == [[1, "x"], [2, "y"]]
    # JSON-string of a bare list (legacy / hand-crafted callers).
    assert decode_result_json(json.dumps([[1, "x"]])) == [[1, "x"]]
    # Missing / unparseable.
    assert decode_result_json(None) is None
    assert decode_result_json("{not json") is None
    # Snapshot dict whose ``sample_rows`` is missing or wrong type → None.
    assert decode_result_json({"columns": [], "sample_rows": "oops"}) is None
    # ``sample_rows`` key entirely absent (e.g. caller passed a
    # ``row_count``-only summary dict) → None, not the dict.
    assert decode_result_json({"columns": [], "row_count": 0}) is None
    # ``capture_result_snapshot`` returns ``{"error": "..."}`` on SQL /
    # runtime failure (``agents/_submit.py:130, 173``). Pre-fix this fell
    # through as ``return payload`` and the inline grader raised
    # ValidationError on the dict.
    assert decode_result_json({"error": "QueryError: bad SQL"}) is None
    assert decode_result_json(json.dumps({"error": "boom"})) is None


def test_harness_pass_accepts_snapshot_dict_predicted_result(monkeypatch, tmp_path):
    """End-to-end DEV-1533 regression: passing the ``capture_result_snapshot``
    dict shape (the actual on-wire shape from cloud + local) through
    ``grade_one_submission(predicted_result=..., gold_result=...)`` MUST
    NOT raise. Pre-fix Pydantic rejected the dict at SubmissionAnnotation
    construction, the inline grader raised, and the task surfaced as
    ``verdict=eval_failed`` with no real cascade verdict."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.grade_in_place import (
        decode_result_json, grade_one_submission,
    )
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation

    snapshot = {
        "columns": [{"name": "x", "type": "int"}],
        "row_count": 3,
        "row_count_truncated": False,
        "sample_rows": [[1], [2], [3]],
    }

    rows_dir = tmp_path / "rows"
    out = grade_one_submission(
        task_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT gold"],
            "amb_user_query": "q",
        },
        submitted_sql="SELECT x FROM t",
        rows_dir=rows_dir,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/nonexistent/db.sqlite"),
        task_annotation=implicit_task_annotation(
            instance_id="alien_1", selected_database="alien",
            benchmark="mini-interact", amb_user_query="q",
        ),
        harness_passed=True,
        # Callers (run.py + ray_app.py) pipe the raw row through
        # ``decode_result_json`` — exercise the same code path here so a
        # future short-circuit can't quietly skip the normalisation.
        predicted_result=decode_result_json(snapshot),
        gold_result=decode_result_json(snapshot),
    )
    ann = json.loads(out.read_text())
    assert ann["predicted_result"] == [[1], [2], [3]]
    assert ann["gold_result"] == [[1], [2], [3]]
    assert ann["evaluation"]["rationale"] == "harness_confirmed"
