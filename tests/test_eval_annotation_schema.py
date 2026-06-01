"""Smoke tests for the DEV-1515 annotation schemas + I/O helpers.

Validates:
* Minimal-field construction succeeds for both annotation kinds.
* JSON round-trip preserves every field (write → read → equal).
* ``model_config = forbid`` rejects unknown top-level fields.
* Path helpers produce the agreed on-disk shape.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bird_interact_agents.eval import (
    AuditedGoldRef,
    FailureClassification,
    GoldVariantRef,
    MaskedTerm,
    MetadataSufficiency,
    SubmissionAnnotation,
    SubmissionEvaluation,
    SubmissionMetadata,
    TaskAnnotation,
    TrajectoryDecisionPoint,
    UserSimInteraction,
    VariantMatch,
    read_submission_annotation,
    read_task_annotation,
    submission_annotation_path,
    task_annotation_path,
    write_submission_annotation,
    write_task_annotation,
)
from bird_interact_agents.eval.annotation_schema import Provenance, UserSimResponseSummary


def _make_task_annotation() -> TaskAnnotation:
    return TaskAnnotation(
        instance_id="alien_42",
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-05-31",
        amb_user_query="Who is the alien?",
        external_knowledge=[1, 2, 3],
        masked_terms=[
            MaskedTerm(
                term="alien",
                type="knowledge_linking_ambiguity",
                metadata_evidence=["alien_kb.jsonl#1"],
            )
        ],
        metadata_sufficiency=MetadataSufficiency(
            verdict="ambiguous",
            rationale="KB hedges; sampled values show variants",
            evidence_sources_consulted=["alien_kb.jsonl#1"],
        ),
        gold_variants=[
            GoldVariantRef(
                variant_id="canonical_only",
                interpretation="KB literals only",
                primary=True,
                anchored_in=["alien_kb.jsonl#1"],
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id="alien_42",
                    variant_id="canonical_only",
                ),
                notes="primary variant",
            )
        ],
        evaluator_prompt=None,
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="alien_42",
            audited_gold_legacy_path=None,
        ),
    )


def _make_submission_annotation() -> SubmissionAnnotation:
    return SubmissionAnnotation(
        instance_id="alien_42",
        selected_database="alien",
        task_annotation_ref="annotations/mini-interact/alien/alien_42.task.json",
        annotated_by="test",
        annotated_at="2026-05-31",
        submission=SubmissionMetadata(
            cloud_run_id="20260531tXXXX",
            trajectory_path="results/cloud/.../alien_42/attempt-1.json",
            predicted_row_count=10,
            duration_s=42.0,
            cost_usd_agent=1.23,
            cost_usd_user_sim=0.04,
            n_agent_turns=37,
            n_ask_user_calls=3,
        ),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="fail",
            phase1_against_audited_primary="fail",
            phase1_against_any_audited_variant="pass",
            phase1_against_variants=[
                VariantMatch(variant_id="canonical_only", match="equal_rowset"),
            ],
            correct_up_to_tie_order=False,
            verdict="valid_interpretation",
            matched_variant_id="canonical_only",
            rationale="matched canonical_only",
        ),
        failure_classification=FailureClassification(
            primary="metadata_ambiguity",
            secondary=["gold_audit_quality"],
            agent_at_fault=False,
            remediation_target="audit",
            remediation_text="Re-audit per the audit-gold-sql contract.",
            details="see analyses/raw/alien_42.md",
        ),
        decision_point=TrajectoryDecisionPoint(
            trajectory_item_index=99,
            description="user-sim locked canonical-only reading",
        ),
        user_sim_interaction=UserSimInteraction(
            n_asks=3,
            key_responses=[UserSimResponseSummary(trajectory_idx=99, summary="…")],
            disclosed_resolutions=["alien"],
            undisclosed_resolutions=["alien synonyms"],
        ),
    )


def test_task_annotation_minimal_construct():
    ann = _make_task_annotation()
    assert ann.kind == "task_annotation"
    assert ann.schema_version == 1
    assert len(ann.gold_variants) == 1
    assert ann.gold_variants[0].primary is True


def test_submission_annotation_minimal_construct():
    ann = _make_submission_annotation()
    assert ann.kind == "submission_annotation"
    assert ann.evaluation.verdict == "valid_interpretation"
    assert ann.failure_classification.primary == "metadata_ambiguity"


def test_task_annotation_roundtrip(tmp_path):
    ann = _make_task_annotation()
    p = tmp_path / "alien_42.task.json"
    write_task_annotation(ann, p)
    loaded = read_task_annotation(p)
    assert loaded == ann


def test_submission_evaluation_default_includes_n9_case_fold():
    """N9 (case-fold) flag must default to False on new evaluations
    and round-trip through JSON write/read."""
    ann = _make_submission_annotation()
    assert ann.evaluation.correct_under_case_fold is False


def test_submission_evaluation_n9_case_fold_roundtrips(tmp_path):
    ann = _make_submission_annotation()
    # Flip the new tier to True to confirm it survives write→read.
    ann.evaluation.correct_under_case_fold = True
    p = tmp_path / "alien_42.submission.20260531t0001.json"
    write_submission_annotation(ann, p)
    loaded = read_submission_annotation(p)
    assert loaded.evaluation.correct_under_case_fold is True


def test_submission_annotation_roundtrip(tmp_path):
    ann = _make_submission_annotation()
    p = tmp_path / "alien_42.submission.20260531t0001.json"
    write_submission_annotation(ann, p)
    loaded = read_submission_annotation(p)
    assert loaded == ann


def test_task_annotation_forbid_extra():
    """Unknown top-level fields must fail validation — protects against
    silent schema drift when consumers add fields the harness will ignore."""
    payload = _make_task_annotation().model_dump()
    payload["a_field_that_should_not_exist"] = True
    with pytest.raises(ValidationError):
        TaskAnnotation.model_validate(payload)


def test_submission_annotation_forbid_extra():
    payload = _make_submission_annotation().model_dump()
    payload["another_unknown_field"] = 42
    with pytest.raises(ValidationError):
        SubmissionAnnotation.model_validate(payload)


def test_path_helpers(tmp_path):
    t = task_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_42",
        repo_root=tmp_path,
    )
    assert t == tmp_path / "annotations" / "mini-interact" / "alien" / "alien_42.task.json"
    s = submission_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_42",
        run_id="20260531t1008-claudes-slayer-890419",
        repo_root=tmp_path,
    )
    assert s == (
        tmp_path
        / "annotations"
        / "mini-interact"
        / "alien"
        / "alien_42.submission.20260531t1008-claudes-slayer-890419.json"
    )


def test_written_json_is_valid_utf8_and_human_readable(tmp_path):
    """A grep-friendly invariant: every annotation file is plain JSON
    with 2-space indent. Catches accidental `.write_text` swaps to
    binary / non-indented output."""
    ann = _make_task_annotation()
    p = tmp_path / "x.task.json"
    write_task_annotation(ann, p)
    text = p.read_text()
    assert text.endswith("\n")
    # JSON is parseable and the indent is 2.
    decoded = json.loads(text)
    assert decoded["instance_id"] == "alien_42"
    # Indent check: every continuation line of the body starts with `  ` (2 spaces).
    body_lines = text.splitlines()
    assert body_lines[0] == "{"
    indent_lines = [l for l in body_lines[1:-1] if l.strip()]
    assert all(l.startswith("  ") for l in indent_lines)
