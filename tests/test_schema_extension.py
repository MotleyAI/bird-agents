"""DEV-1515: pin the SubmissionEvaluation + VariantMatch schema extensions.

The pre-DEV-1515 `SubmissionEvaluation` only carried N1-N5 fields. The
plan adds:
* `correct_under_numeric_epsilon: bool`
* `correct_under_trailing_whitespace: bool`
* `correct_under_column_order: bool`
* `numeric_epsilon: float` (records the threshold used)

The pre-DEV-1515 `VariantMatch` only carried `variant_id` + `match`.
The plan extends it with an optional `informational: VariantInformational`
sub-block carrying:
* rowset_relation, column_count_match,
  column_name_match_case_insensitive, column_order_match,
* first_divergent_row_index, first_divergent_cell_diff.

These tests prove the schema actually carries the new fields, with
``extra="forbid"`` still in effect.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


def _base_submission_evaluation_kwargs() -> dict:
    return dict(
        phase1_against_original_gold="fail",
        phase1_against_audited_primary="fail",
        phase1_against_any_audited_variant="pass",
        phase1_against_variants=[],
        correct_up_to_tie_order=True,
        novel_reading_judgment=None,
        verdict="valid_interpretation",
        matched_variant_id="primary",
        rationale="",
    )


def test_submission_evaluation_carries_n6_through_n8_fields():
    from bird_interact_agents.eval import SubmissionEvaluation

    ev = SubmissionEvaluation(
        **_base_submission_evaluation_kwargs(),
        correct_under_numeric_epsilon=True,
        correct_under_trailing_whitespace=True,
        correct_under_column_order=False,
        numeric_epsilon=1e-6,
    )
    assert ev.correct_under_numeric_epsilon is True
    assert ev.correct_under_trailing_whitespace is True
    assert ev.correct_under_column_order is False
    assert ev.numeric_epsilon == 1e-6


def test_submission_evaluation_roundtrips_new_fields():
    from bird_interact_agents.eval import SubmissionEvaluation

    ev = SubmissionEvaluation(
        **_base_submission_evaluation_kwargs(),
        correct_under_numeric_epsilon=True,
        correct_under_trailing_whitespace=False,
        correct_under_column_order=True,
        numeric_epsilon=5e-7,
    )
    revalidated = SubmissionEvaluation.model_validate_json(ev.model_dump_json())
    assert revalidated == ev


def test_submission_evaluation_extra_forbid_still_holds():
    from bird_interact_agents.eval import SubmissionEvaluation

    payload = SubmissionEvaluation(
        **_base_submission_evaluation_kwargs(),
        correct_under_numeric_epsilon=False,
        correct_under_trailing_whitespace=False,
        correct_under_column_order=False,
        numeric_epsilon=1e-6,
    ).model_dump()
    payload["a_field_not_in_schema"] = True
    with pytest.raises(ValidationError):
        SubmissionEvaluation.model_validate(payload)


def test_variant_match_has_optional_informational():
    from bird_interact_agents.eval import VariantMatch
    from bird_interact_agents.eval.annotation_schema import (
        VariantInformational,
    )

    vm_without = VariantMatch(variant_id="primary", match="equal_rowset")
    assert vm_without.informational is None

    info = VariantInformational(
        rowset_relation="strict_subset_of",
        column_count_match=True,
        column_name_match_case_insensitive=False,
        column_order_match=True,
        first_divergent_row_index=3,
        first_divergent_cell_diff="cell[3][1]: 'X' vs 'Y'",
    )
    vm_with = VariantMatch(
        variant_id="primary", match="strict_subset_of",
        informational=info,
    )
    assert vm_with.informational is not None
    assert vm_with.informational.first_divergent_row_index == 3


def test_variant_informational_extra_forbid():
    from bird_interact_agents.eval.annotation_schema import (
        VariantInformational,
    )

    payload = VariantInformational(
        rowset_relation="equal_rowset",
        column_count_match=True,
        column_name_match_case_insensitive=True,
        column_order_match=True,
        first_divergent_row_index=None,
        first_divergent_cell_diff=None,
    ).model_dump()
    payload["surprise"] = "not allowed"
    with pytest.raises(ValidationError):
        VariantInformational.model_validate(payload)
