"""DEV-1533: monotone enforcement with original_gold_is_correct=False."""
from __future__ import annotations


def test_n1_does_not_force_n2_when_original_gold_is_wrong():
    """When original_gold_is_correct=False, N1=True must NOT force N2=True."""
    from bird_interact_agents.eval.tolerant_grader import enforce_monotone_cascade

    raw = {
        "n1_original_gold": True,   # agent matched original (wrong) gold
        "n2_audited_primary": False,  # agent did NOT match audited gold
        "n3_any_audited_variant": False,
        "n4_tie_order": False,
        "n5_llm_judge": False,
        "n6_numeric_epsilon": False,
        "n7_trailing_whitespace": False,
        "n8_column_order": False,
        "n9_case_fold": False,
    }
    result = enforce_monotone_cascade(raw, original_gold_is_correct=False)
    assert result["n1_original_gold"] is True, "N1 must still be recorded"
    assert result["n2_audited_primary"] is False, (
        "N2 must NOT be forced True when original gold is wrong"
    )
    assert result["n3_any_audited_variant"] is False
    assert result["n9_case_fold"] is False


def test_n1_forces_n2_when_original_gold_is_correct():
    """When original_gold_is_correct=True, N1=True still forces N2+ (default)."""
    from bird_interact_agents.eval.tolerant_grader import enforce_monotone_cascade

    raw = {
        "n1_original_gold": True,
        "n2_audited_primary": False,  # would be False independently
        "n3_any_audited_variant": False,
        "n4_tie_order": False,
        "n5_llm_judge": False,
        "n6_numeric_epsilon": False,
        "n7_trailing_whitespace": False,
        "n8_column_order": False,
        "n9_case_fold": False,
    }
    result = enforce_monotone_cascade(raw, original_gold_is_correct=True)
    assert result["n1_original_gold"] is True
    assert result["n2_audited_primary"] is True, "N2 must be forced True by N1"
    assert result["n9_case_fold"] is True


def test_n2_onwards_still_monotone_when_original_gold_is_wrong():
    """Even when original_gold_is_correct=False, N2→N3+ propagation holds."""
    from bird_interact_agents.eval.tolerant_grader import enforce_monotone_cascade

    raw = {
        "n1_original_gold": False,
        "n2_audited_primary": True,   # agent matched audited
        "n3_any_audited_variant": False,  # would violate monotone
        "n4_tie_order": False,
        "n5_llm_judge": False,
        "n6_numeric_epsilon": False,
        "n7_trailing_whitespace": False,
        "n8_column_order": False,
        "n9_case_fold": False,
    }
    result = enforce_monotone_cascade(raw, original_gold_is_correct=False)
    assert result["n2_audited_primary"] is True
    assert result["n3_any_audited_variant"] is True, "N3 forced by N2"
    assert result["n9_case_fold"] is True


def test_grade_submission_respects_original_gold_is_correct(tmp_path):
    """Integration: grade_submission reads original_gold_is_correct from
    task_annotation and applies the monotone fix."""
    from pathlib import Path
    from bird_interact_agents.eval.tolerant_grader import grade_submission
    from bird_interact_agents.eval import MetadataSufficiency, TaskAnnotation
    from bird_interact_agents.eval.annotation_schema import Provenance

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):
            if "wrong" in sql:
                return ([(1,)], ["a"])
            if "correct" in sql:
                return ([(99,)], ["a"])
            return ([(1,)], ["a"])  # agent matches wrong original

    ann = TaskAnnotation(
        instance_id="alien_1", selected_database="alien",
        annotated_by="test", annotated_at="2026-06-01",
        amb_user_query="x",
        original_gold_is_correct=False,  # original is wrong
        gold_variants=[],  # required when original_gold_is_correct=False and no audited
        metadata_sufficiency=MetadataSufficiency(verdict="sufficient", rationale="r"),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="alien_1",
        ),
    )
    audited_rows = [
        {
            "variant_id": "primary", "primary": True,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT correct"],
        }
    ]

    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=audited_rows,
        original_sol_sql=["SELECT wrong"],
        submitted_sql="SELECT agent",  # matches neither
        db_path=Path("/dev/null"),
        conn=None,
        executor=FakeExec(),
    )
    # Agent matched wrong original (N1), but original_gold_is_correct=False
    # so the grader should not inflate N2.
    assert verdict.n1_original_gold is True
    assert verdict.n2_audited_primary is False
