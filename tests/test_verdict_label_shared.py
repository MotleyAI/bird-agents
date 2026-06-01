"""DEV-1515 review-followup: cascade → SubmissionEvaluation.verdict
labelling MUST be consistent across the two persistence paths.

Two helpers map a ``CascadeVerdict`` to a ``SubmissionEvaluation``
fixture: ``grade_in_place._build_submission_annotation`` (the cloud
+ local inline grader) and ``annotate._eval_from_cascade`` (the
skeleton CLI + the regrade CLI). Before extracting
``verdict_label_from_cascade`` the annotate path returned
``verdict="invalid"`` for every cascade tier below N3, while the
inline grader returned ``"valid_interpretation"`` for N4/N5/N6/N7/N8.

Pin each cascade-tier → verdict mapping AND assert both helpers emit
the same value.
"""
from __future__ import annotations


def _cascade(**flags):
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    return CascadeVerdict(
        n1_original_gold=flags.get("n1", False),
        n2_audited_primary=flags.get("n2", False),
        n3_any_audited_variant=flags.get("n3", False),
        n4_tie_order=flags.get("n4", False),
        n5_llm_judge=flags.get("n5", False),
        n6_numeric_epsilon=flags.get("n6", False),
        n7_trailing_whitespace=flags.get("n7", False),
        n8_column_order=flags.get("n8", False),
        n9_case_fold=flags.get("n9", False),
        matched_variant_id=flags.get("matched_variant_id"),
        novel_reading_judgment=flags.get("novel"),
    )


def _both_verdicts(cascade) -> tuple[str, str]:
    """Return (inline_verdict, annotate_verdict) for the given cascade."""
    from bird_interact_agents.eval.grade_in_place import (
        verdict_label_from_cascade,
    )
    from bird_interact_agents.eval.annotate import _eval_from_cascade

    inline = verdict_label_from_cascade(cascade)
    ev = _eval_from_cascade(cascade)
    return inline, ev.verdict


def test_n3_strict_yields_correct_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n3=True))
    assert inline == "correct"
    assert annotate == "correct"


def test_n4_only_yields_valid_interpretation_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n4=True))
    assert inline == "valid_interpretation"
    assert annotate == "valid_interpretation", (
        "regression: annotate._eval_from_cascade used to return 'invalid' "
        "for N4 tie-order passes — the helper must reuse "
        "grade_in_place.verdict_label_from_cascade so the two persistence "
        "paths agree."
    )


def test_n5_only_yields_valid_interpretation_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n5=True, novel="pass"))
    assert inline == "valid_interpretation"
    assert annotate == "valid_interpretation"


def test_n6_only_yields_valid_interpretation_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n6=True))
    assert inline == "valid_interpretation"
    assert annotate == "valid_interpretation"


def test_n7_only_yields_valid_interpretation_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n7=True))
    assert inline == "valid_interpretation"
    assert annotate == "valid_interpretation"


def test_n8_only_yields_valid_interpretation_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n8=True))
    assert inline == "valid_interpretation"
    assert annotate == "valid_interpretation"


def test_n9_only_yields_valid_interpretation_on_both_paths():
    inline, annotate = _both_verdicts(_cascade(n9=True))
    assert inline == "valid_interpretation"
    assert annotate == "valid_interpretation"


def test_no_tier_yields_invalid_on_both_paths():
    inline, annotate = _both_verdicts(_cascade())
    assert inline == "invalid"
    assert annotate == "invalid"
