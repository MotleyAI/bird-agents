"""Normalizer maps SAR-Agent's verdict into the audited_gold superset schema."""

from __future__ import annotations

import pytest

from bird_interact_agents.sar_audit import normalizer
from bird_interact_agents.sar_audit.normalizer import SARVerdict, SampleRowResult


ORIG = "SELECT x FROM t ORDER BY x LIMIT 1"
REVISED = "SELECT x FROM t ORDER BY x ASC LIMIT 1"


def _norm(verdict: SARVerdict, *, task=None, sample_row_result=None, **kw):
    if task is None:
        task = {
            "instance_id": "credit_1",
            "selected_database": "credit",
            "sol_sql": [ORIG],
        }
    if sample_row_result is None:
        sample_row_result = SampleRowResult(row=[1], status="ok", error=None)
    return normalizer.to_normalized_row(
        task=task,
        verdict=verdict,
        sample_row_result=sample_row_result,
        audit_model_requested="claude-opus-4-7",
        audit_model_actual="claude-opus-4-7-20260121",
        step_count=5,
        cost_usd=0.01,
        skill_version="sar-agent/1.0",
        **kw,
    )


def test_status_clean(fixed_now):
    row = _norm(SARVerdict(correctness_flag=True, ambiguity_flag=False, reasoning="all good"))
    assert row["audit_status"] == "clean"
    assert row["audited_sol_sql"] == [ORIG]
    assert row["changes"] == []
    assert row["revised_question"] is None
    assert row["audited_at"] == fixed_now


def test_status_edited(fixed_now):
    row = _norm(
        SARVerdict(
            correctness_flag=False,
            ambiguity_flag=False,
            revised_sql=REVISED,
            reasoning="missing ASC",
        )
    )
    assert row["audit_status"] == "edited"
    assert row["audited_sol_sql"] == [REVISED]
    assert len(row["changes"]) == 1
    ch = row["changes"][0]
    assert ch["clause_kind"] == "sar_revision"
    assert ch["source"] == "sar_agent"
    assert ch["original"] == ORIG
    assert ch["replacement"] == REVISED
    assert ch["justified_by"] == []
    assert "missing ASC" in ch["why_unjustified"]


def test_status_unrecoverable(fixed_now):
    row = _norm(
        SARVerdict(
            correctness_flag=False,
            ambiguity_flag=False,
            revised_sql=None,
            reasoning="cannot determine intent",
        )
    )
    assert row["audit_status"] == "unrecoverable"
    # audited_sol_sql stays executable (copy of original) so sample_row works
    assert row["audited_sol_sql"] == [ORIG]
    assert len(row["changes"]) == 1
    ch = row["changes"][0]
    assert ch["clause_kind"] == "sar_unrecoverable"
    assert ch["source"] == "sar_agent"
    assert ch["original"] == ORIG
    assert ch["replacement"] == ""
    assert "cannot determine intent" in ch["why_unjustified"]
    assert ch["justified_by"] == []


def test_status_ambiguous_no_revision(fixed_now):
    row = _norm(
        SARVerdict(
            correctness_flag=True,
            ambiguity_flag=True,
            revised_sql=None,
            revised_question="Did you want strict ASC ordering?",
            reasoning="question is unclear",
        )
    )
    assert row["audit_status"] == "ambiguous"
    assert row["audited_sol_sql"] == [ORIG]
    assert len(row["changes"]) == 1
    ch = row["changes"][0]
    assert ch["clause_kind"] == "sar_ambiguous"
    assert ch["replacement"] == ""
    assert row["revised_question"] == "Did you want strict ASC ordering?"


def test_status_ambiguous_with_revision(fixed_now):
    row = _norm(
        SARVerdict(
            correctness_flag=False,
            ambiguity_flag=True,
            revised_sql=REVISED,
            revised_question="Did you want strict ASC ordering?",
            reasoning="ambiguous and wrong",
        )
    )
    # Ambiguity dominates audit_status even when SAR also revised the SQL.
    assert row["audit_status"] == "ambiguous"
    # But the revision is preserved.
    assert row["audited_sol_sql"] == [REVISED]
    assert len(row["changes"]) == 1
    ch = row["changes"][0]
    assert ch["clause_kind"] == "sar_ambiguous_revision"
    assert ch["original"] == ORIG
    assert ch["replacement"] == REVISED
    assert row["revised_question"] == "Did you want strict ASC ordering?"


def test_revised_question_only_when_ambiguity_flag_true(fixed_now):
    # ambiguity_flag=False → revised_question must be None even if SAR returns one
    row = _norm(
        SARVerdict(
            correctness_flag=False,
            ambiguity_flag=False,
            revised_sql=REVISED,
            revised_question="ignored phrasing",
            reasoning="wrong sql",
        )
    )
    assert row["revised_question"] is None


def test_all_changes_carry_source_sar_agent(fixed_now):
    for verdict in [
        SARVerdict(correctness_flag=False, ambiguity_flag=False, revised_sql=REVISED),
        SARVerdict(correctness_flag=False, ambiguity_flag=False, revised_sql=None),
        SARVerdict(correctness_flag=True, ambiguity_flag=True, revised_sql=None),
        SARVerdict(correctness_flag=False, ambiguity_flag=True, revised_sql=REVISED),
    ]:
        row = _norm(verdict)
        for ch in row["changes"]:
            assert ch["source"] == "sar_agent"
            assert ch["justified_by"] == []


def test_skill_version_and_audit_model_fields(fixed_now):
    row = _norm(SARVerdict(correctness_flag=True, ambiguity_flag=False))
    assert row["skill_version"] == "sar-agent/1.0"
    assert row["audit_model_requested"] == "claude-opus-4-7"
    assert row["audit_model_actual"] == "claude-opus-4-7-20260121"


def test_step_count_and_cost_passthrough(fixed_now):
    row = _norm(SARVerdict(correctness_flag=True, ambiguity_flag=False))
    assert row["step_count"] == 5
    assert row["cost_usd"] == 0.01


def test_sample_row_ok(fixed_now):
    row = _norm(
        SARVerdict(correctness_flag=True, ambiguity_flag=False),
        sample_row_result=SampleRowResult(row=[42, "foo"], status="ok", error=None),
    )
    assert row["audited_sample_row"] == [42, "foo"]
    assert row["audited_sample_row_status"] == "ok"
    assert row["audited_sample_row_error"] is None


def test_sample_row_empty(fixed_now):
    row = _norm(
        SARVerdict(correctness_flag=True, ambiguity_flag=False),
        sample_row_result=SampleRowResult(row=None, status="empty", error=None),
    )
    assert row["audited_sample_row"] is None
    assert row["audited_sample_row_status"] == "empty"
    assert row["audited_sample_row_error"] is None


def test_sample_row_error(fixed_now):
    row = _norm(
        SARVerdict(correctness_flag=True, ambiguity_flag=False),
        sample_row_result=SampleRowResult(row=None, status="error", error="no such table"),
    )
    assert row["audited_sample_row"] is None
    assert row["audited_sample_row_status"] == "error"
    assert row["audited_sample_row_error"] == "no such table"


def test_raw_trajectory_off_by_default(fixed_now):
    row = _norm(SARVerdict(correctness_flag=True, ambiguity_flag=False))
    assert row["raw_trajectory"] is None


def test_raw_trajectory_passthrough_when_present(fixed_now):
    row = _norm(
        SARVerdict(correctness_flag=True, ambiguity_flag=False),
        raw_trajectory=[{"step": 1, "action": "x"}],
    )
    assert row["raw_trajectory"] == [{"step": 1, "action": "x"}]


def test_instance_id_and_database_propagated(fixed_now):
    task = {
        "instance_id": "credit_99",
        "selected_database": "credit",
        "sol_sql": ["SELECT 1"],
    }
    row = _norm(SARVerdict(correctness_flag=True, ambiguity_flag=False), task=task)
    assert row["instance_id"] == "credit_99"
    assert row["selected_database"] == "credit"
    assert row["original_sol_sql"] == ["SELECT 1"]


@pytest.mark.parametrize(
    "correctness,ambiguity,revised,expected_status",
    [
        (True, False, None, "clean"),
        (False, False, REVISED, "edited"),
        (False, False, None, "unrecoverable"),
        (True, True, None, "ambiguous"),
        (False, True, None, "ambiguous"),
        (True, True, REVISED, "ambiguous"),
        (False, True, REVISED, "ambiguous"),
    ],
)
def test_mapping_table_exhaustive(fixed_now, correctness, ambiguity, revised, expected_status):
    row = _norm(
        SARVerdict(
            correctness_flag=correctness,
            ambiguity_flag=ambiguity,
            revised_sql=revised,
        )
    )
    assert row["audit_status"] == expected_status
