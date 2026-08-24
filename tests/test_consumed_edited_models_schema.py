"""DEV-1778: `ConsumedEditedModels` type + the additive-optional
`consumed_edited_models` field on `SubmissionAnnotation` (no schema_version
bump; legacy annotations still validate)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bird_interact_agents.eval import (
    FailureClassification,
    SubmissionAnnotation,
    SubmissionEvaluation,
    SubmissionMetadata,
    read_submission_annotation,
    write_submission_annotation,
)
from bird_interact_agents.eval.annotation_schema import ConsumedEditedModels

_CONSUMED = {"db": "alien", "instance_id": "alien_1", "store_fp": "deadbeef" * 8}


def _ann(**overrides) -> SubmissionAnnotation:
    kwargs = dict(
        instance_id="alien_1",
        selected_database="alien",
        task_annotation_ref="annotations/mini-interact/alien/alien_1.task.json",
        annotated_by="test",
        annotated_at="2026-08-11",
        submission=SubmissionMetadata(
            cloud_run_id="r1",
            trajectory_path="rows/alien_1/attempt-1.json",
        ),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="pass",
            phase1_against_audited_primary="pass",
            phase1_against_any_audited_variant="pass",
            verdict="correct",
        ),
        failure_classification=FailureClassification(
            primary="no_fail",
            agent_at_fault=False,
            remediation_target="other",
        ),
    )
    kwargs.update(overrides)
    return SubmissionAnnotation(**kwargs)


def test_consumed_type_fields_and_forbid_extra():
    rec = ConsumedEditedModels(**_CONSUMED)
    assert (rec.db, rec.instance_id, rec.store_fp) == (
        "alien", "alien_1", _CONSUMED["store_fp"],
    )
    with pytest.raises(ValidationError):
        ConsumedEditedModels(db="alien", instance_id="alien_1", store_fp="x", extra=1)


def test_annotation_defaults_field_to_none():
    assert _ann().consumed_edited_models is None


def test_annotation_accepts_model_and_dict():
    assert _ann(consumed_edited_models=ConsumedEditedModels(**_CONSUMED)).consumed_edited_models.store_fp == _CONSUMED["store_fp"]
    # a plain dict (as carried on a result row) coerces on validation
    assert _ann(consumed_edited_models=_CONSUMED).consumed_edited_models.db == "alien"


def test_legacy_annotation_without_field_validates_and_schema_version_unchanged():
    payload = _ann().model_dump()
    payload.pop("consumed_edited_models", None)
    loaded = SubmissionAnnotation.model_validate(payload)
    assert loaded.consumed_edited_models is None
    assert loaded.schema_version == 1


def test_roundtrip_preserves_field(tmp_path):
    ann = _ann(consumed_edited_models=ConsumedEditedModels(**_CONSUMED))
    p = tmp_path / "alien_1.submission.r1.json"
    write_submission_annotation(ann, p)
    loaded = read_submission_annotation(p)
    assert loaded.consumed_edited_models == ann.consumed_edited_models


def test_annotation_still_forbids_unknown_top_level_field():
    payload = _ann().model_dump()
    payload["not_a_real_field"] = 1
    with pytest.raises(ValidationError):
        SubmissionAnnotation.model_validate(payload)
