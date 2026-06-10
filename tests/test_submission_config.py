"""DEV-1535 — `SubmissionConfig` round-trip in `SubmissionMetadata`.

The config is duplicated into every annotation per design choice (so
grep / one-off scripts don't need to JOIN against `run_metadata`).
"""

from __future__ import annotations

from bird_interact_agents.eval.annotation_schema import (
    FailureClassification,
    SubmissionAnnotation,
    SubmissionConfig,
    SubmissionEvaluation,
    SubmissionMetadata,
)


def _minimal_annotation(*, config: SubmissionConfig | None = None) -> SubmissionAnnotation:
    return SubmissionAnnotation(
        instance_id="alien_1", selected_database="alien",
        task_annotation_ref="annotations/x/y/z.task.json",
        annotated_by="t", annotated_at="2026-06-09",
        submission=SubmissionMetadata(
            cloud_run_id="r", trajectory_path="rows/alien_1/attempt-1.json",
            config=config,
        ),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="fail",
            phase1_against_audited_primary="fail",
            phase1_against_any_audited_variant="fail",
            phase1_against_variants=[],
            correct_up_to_tie_order=False,
            novel_reading_judgment=None,
            correct_under_numeric_epsilon=False,
            correct_under_trailing_whitespace=False,
            correct_under_column_order=False,
            correct_under_case_fold=False,
            numeric_epsilon=1e-6,
            verdict="agent_miss",
            matched_variant_id=None,
            rationale="",
            miss_diagnostics=None,
        ),
        failure_classification=FailureClassification(
            primary="other", agent_at_fault=False, remediation_target="other",
        ),
    )


def test_submission_config_round_trip():
    """Populated `SubmissionConfig` serializes + parses back unchanged.
    Validates the schema's JSON shape (every Optional field carries
    through; bool values stay bool, not int)."""
    cfg = SubmissionConfig(
        framework="claude_sdk", mode="a-interact",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        slayer_setup="on-the-fly", reasoning_effort="high",
        patience=250, max_depth=3, dataset="mini-interact",
        strict=False, use_audited_gold_sql=True, prompt_cache=True,
    )
    ann = _minimal_annotation(config=cfg)
    blob = ann.model_dump_json()
    restored = SubmissionAnnotation.model_validate_json(blob)
    assert restored.submission.config == cfg
    assert restored.submission.config.framework == "claude_sdk"
    assert restored.submission.config.strict is False
    assert restored.submission.config.use_audited_gold_sql is True


def test_submission_config_none_round_trip():
    """Back-compat: `config=None` round-trips. Older annotations on
    disk (pre-DEV-1535) have no `config` field at all; they parse
    cleanly with config=None."""
    ann = _minimal_annotation(config=None)
    blob = ann.model_dump_json()
    restored = SubmissionAnnotation.model_validate_json(blob)
    assert restored.submission.config is None


def test_submission_config_partial_round_trip():
    """Sparse config — some fields set, others None — preserves the
    distinction. SubmissionConfig is `extra="forbid"`, so a typo on
    any field would surface as a validation error rather than silently
    drop."""
    cfg = SubmissionConfig(
        framework="claude_sdk", patience=100,
        # other fields omitted → None
    )
    ann = _minimal_annotation(config=cfg)
    blob = ann.model_dump_json()
    restored = SubmissionAnnotation.model_validate_json(blob)
    assert restored.submission.config is not None
    assert restored.submission.config.framework == "claude_sdk"
    assert restored.submission.config.patience == 100
    assert restored.submission.config.reasoning_effort is None
    assert restored.submission.config.use_audited_gold_sql is None


def test_pre_dev1535_annotation_parses_without_config_key():
    """A SubmissionAnnotation JSON written before DEV-1535 has no
    `config` key on `submission`. Schema must accept it (config defaults
    to None) so historical annotations remain readable."""
    import json as _json
    ann_dict = {
        "instance_id": "alien_1", "selected_database": "alien",
        "task_annotation_ref": "annotations/x/y/z.task.json",
        "annotated_by": "t", "annotated_at": "2026-06-09",
        "submission": {
            "cloud_run_id": "r",
            "trajectory_path": "rows/alien_1/attempt-1.json",
            # No `config` key — pre-DEV-1535 shape.
        },
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "novel_reading_judgment": None,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "agent_miss",
            "matched_variant_id": None,
            "rationale": "",
            "miss_diagnostics": None,
        },
        "failure_classification": {
            "primary": "other", "agent_at_fault": False,
            "remediation_target": "other",
        },
    }
    restored = SubmissionAnnotation.model_validate_json(_json.dumps(ann_dict))
    assert restored.submission.config is None
