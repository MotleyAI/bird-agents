"""DEV-1515: annotation-skeleton CLI tests for
``python -m bird_interact_agents.eval.annotate``.

Pins:
* Auto-fills mechanical fields (instance_id, masked_terms from
  critical_ambiguity, submission.* from trajectory, etc.).
* Leaves PENDING_HUMAN_REVIEW sentinels in human-judgment fields.
* ``--task-mode init`` skips existing files (default).
* ``--task-mode refresh`` preserves non-sentinel fields, overwrites
  mechanical fields only.
* ``--task-mode force-all`` overwrites the whole file.
* ``--submission-mode overwrite`` (default) always rewrites.
* ``--submission-mode init`` skips when file exists.
* ``--dry-run`` writes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


SAMPLE_TASK_ROW = {
    "instance_id": "alien_1",
    "selected_database": "alien",
    "amb_user_query": "Find some aliens.",
    "external_knowledge": [1, 2, 3],
    "sol_sql": ["SELECT gold FROM aliens"],
    "user_query_ambiguity": {
        "critical_ambiguity": [
            {"term": "some", "type": "intent_ambiguity"},
            {"term": "aliens", "type": "schema_linking_ambiguity"},
        ],
    },
}


def _write_attempt_json(rows_dir: Path, instance_id: str, *, submitted_sql: str):
    d = rows_dir / instance_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "attempt-1.json").write_text(json.dumps({
        "instance_id": instance_id,
        "submitted_sql": submitted_sql,
        "trajectory": [
            {"role": "agent", "content": "thinking"},
            {"role": "tool_call", "name": "ask_user", "args": {"q": "?"}},
            {"role": "user_sim", "content": "answer-1"},
        ],
        "duration_s": 12.3,
        "usage": {"cost_usd_agent": 0.42, "cost_usd_user_sim": 0.01,
                  "n_agent_turns": 3, "n_ask_user_calls": 1},
        "predicted_row_count": 5,
    }))


def test_task_annotation_skeleton_fills_mechanical_fields(tmp_path):
    from bird_interact_agents.eval.annotate import generate_task_annotation

    ann = generate_task_annotation(
        task_row=SAMPLE_TASK_ROW,
        benchmark="mini-interact",
    )
    assert ann.instance_id == "alien_1"
    assert ann.selected_database == "alien"
    assert ann.amb_user_query == "Find some aliens."
    assert ann.external_knowledge == [1, 2, 3]
    assert [m.term for m in ann.masked_terms] == ["some", "aliens"]
    assert ann.provenance.task_jsonl_instance_id == "alien_1"


def test_task_annotation_skeleton_leaves_sentinels(tmp_path):
    from bird_interact_agents.eval.annotate import (
        PENDING_HUMAN_REVIEW,
        generate_task_annotation,
    )

    ann = generate_task_annotation(
        task_row=SAMPLE_TASK_ROW, benchmark="mini-interact",
    )
    assert ann.metadata_sufficiency.rationale == PENDING_HUMAN_REVIEW
    assert ann.evaluator_prompt is None


def test_task_annotation_alias_benchmark_uses_canonical_jsonl_name():
    """``benchmark="mini-interact"`` (dash alias) must resolve to the
    canonical ``"mini_interact.jsonl"`` JSONL name, not the literal
    ``"mini-interact.jsonl"`` fallback."""
    from bird_interact_agents.eval.annotate import generate_task_annotation

    ann = generate_task_annotation(
        task_row=SAMPLE_TASK_ROW, benchmark="mini-interact",
    )
    assert ann.provenance.task_jsonl_path == "mini_interact.jsonl"


def test_submission_annotation_skeleton_fills_from_trajectory(tmp_path):
    from bird_interact_agents.eval.annotate import (
        generate_submission_annotation,
    )

    rows_dir = tmp_path / "rows"
    _write_attempt_json(rows_dir, "alien_1", submitted_sql="SELECT predicted")

    class StubGrader:
        def __call__(self, *, submitted_sql, **_kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary",
                novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    ann = generate_submission_annotation(
        rows_dir=rows_dir, instance_id="alien_1",
        selected_database="alien", benchmark="mini-interact",
        run_id="r1", task_row=SAMPLE_TASK_ROW,
        grader=StubGrader(),
    )
    assert ann.submission.duration_s == 12.3
    assert ann.submission.cost_usd_agent == 0.42
    assert ann.submission.n_ask_user_calls == 1
    assert ann.evaluation.phase1_against_original_gold == "pass"
    # user_sim_interaction walks the trajectory for ask_user calls.
    assert ann.user_sim_interaction.n_asks == 1


# ---------------------------------------------------------------------------
# Mode matrix
# ---------------------------------------------------------------------------


def _write_existing_task_annotation(
    tmp_path: Path,
    *,
    benchmark: str = "mini-interact",
    db: str = "alien",
    instance_id: str = "alien_1",
    rationale: str = "human-written",
    evaluator_prompt: str | None = "evaluator-rules",
) -> Path:
    from bird_interact_agents.eval import (
        AuditedGoldRef, GoldVariantRef, MaskedTerm,
        MetadataSufficiency, TaskAnnotation, task_annotation_path,
        write_task_annotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance

    ann = TaskAnnotation(
        instance_id=instance_id, selected_database=db,
        annotated_by="human", annotated_at="2026-05-30",
        amb_user_query="Find some aliens.",
        external_knowledge=[],
        masked_terms=[MaskedTerm(term="some", type="intent_ambiguity")],
        metadata_sufficiency=MetadataSufficiency(
            verdict="insufficient", rationale=rationale,
        ),
        gold_variants=[
            GoldVariantRef(
                variant_id="primary", interpretation="human-written-interp",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id=instance_id,
                ),
            ),
        ],
        evaluator_prompt=evaluator_prompt,
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=instance_id,
        ),
    )
    p = task_annotation_path(
        benchmark=benchmark, selected_database=db,
        instance_id=instance_id, repo_root=tmp_path,
    )
    write_task_annotation(ann, p)
    return p


def test_task_mode_init_skips_existing(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    p = _write_existing_task_annotation(tmp_path)
    pre_mtime = p.stat().st_mtime_ns

    from bird_interact_agents.eval.annotate import write_task_skeleton

    written = write_task_skeleton(
        task_row=SAMPLE_TASK_ROW, benchmark="mini-interact",
        mode="init", dry_run=False, repo_root=tmp_path,
    )
    assert written is None  # signal "skipped existing"
    assert p.stat().st_mtime_ns == pre_mtime  # untouched


def test_task_mode_refresh_overwrites_mechanical_only(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    p = _write_existing_task_annotation(
        tmp_path, rationale="human-written-RAT",
        evaluator_prompt="human-rules",
    )

    # Add a new ambiguity term to the task row — refresh should pull it in.
    new_row = dict(SAMPLE_TASK_ROW)
    new_row["user_query_ambiguity"] = {
        "critical_ambiguity": [
            {"term": "some", "type": "intent_ambiguity"},
            {"term": "aliens", "type": "schema_linking_ambiguity"},
            {"term": "Find", "type": "intent_ambiguity"},
        ],
    }

    from bird_interact_agents.eval.annotate import write_task_skeleton

    written_path = write_task_skeleton(
        task_row=new_row, benchmark="mini-interact",
        mode="refresh", dry_run=False, repo_root=tmp_path,
    )
    assert written_path == p

    from bird_interact_agents.eval import read_task_annotation
    refreshed = read_task_annotation(p)
    # Mechanical (masked_terms) updated.
    assert {m.term for m in refreshed.masked_terms} == {"some", "aliens", "Find"}
    # Human-judgment fields preserved.
    assert refreshed.metadata_sufficiency.rationale == "human-written-RAT"
    assert refreshed.evaluator_prompt == "human-rules"


def test_task_mode_force_all_overwrites_everything(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    _write_existing_task_annotation(
        tmp_path, rationale="human-written-RAT",
        evaluator_prompt="human-rules",
    )

    from bird_interact_agents.eval.annotate import (
        PENDING_HUMAN_REVIEW, write_task_skeleton,
    )

    p = write_task_skeleton(
        task_row=SAMPLE_TASK_ROW, benchmark="mini-interact",
        mode="force-all", dry_run=False, repo_root=tmp_path,
    )
    from bird_interact_agents.eval import read_task_annotation
    fresh = read_task_annotation(p)
    # Human edits BLOWN AWAY — sentinel restored.
    assert fresh.metadata_sufficiency.rationale == PENDING_HUMAN_REVIEW
    assert fresh.evaluator_prompt is None


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval.annotate import write_task_skeleton

    written = write_task_skeleton(
        task_row=SAMPLE_TASK_ROW, benchmark="mini-interact",
        mode="force-all", dry_run=True, repo_root=tmp_path,
    )
    # Dry-run reports the path it WOULD write to, but no file lands.
    assert written is not None
    assert not written.exists()


def test_submission_mode_overwrite_rewrites_existing(tmp_path, monkeypatch):
    """Default submission-mode (overwrite) replaces the existing file."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata, submission_annotation_path,
        write_submission_annotation,
    )
    existing = SubmissionAnnotation(
        instance_id="alien_1", selected_database="alien",
        task_annotation_ref="x", annotated_by="stale-author",
        annotated_at="2026-05-30",
        submission=SubmissionMetadata(cloud_run_id="r1", trajectory_path="t"),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="pass",
            phase1_against_audited_primary="pass",
            phase1_against_any_audited_variant="pass",
            verdict="correct",
        ),
        failure_classification=FailureClassification(
            primary="other", agent_at_fault=False, remediation_target="other",
        ),
    )
    p = submission_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="r1", repo_root=tmp_path,
    )
    write_submission_annotation(existing, p)

    rows_dir = tmp_path / "rows"
    _write_attempt_json(rows_dir, "alien_1", submitted_sql="SELECT x")

    class StubGrader:
        def __call__(self, **_kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary", novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    from bird_interact_agents.eval.annotate import write_submission_skeleton
    written = write_submission_skeleton(
        rows_dir=rows_dir, instance_id="alien_1",
        selected_database="alien", benchmark="mini-interact",
        run_id="r1", task_row=SAMPLE_TASK_ROW,
        grader=StubGrader(), mode="overwrite",
        dry_run=False, repo_root=tmp_path,
    )
    assert written == p
    from bird_interact_agents.eval import read_submission_annotation
    fresh = read_submission_annotation(p)
    assert fresh.annotated_by != "stale-author"


def test_submission_dry_run_writes_nothing(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    rows_dir = tmp_path / "rows"
    _write_attempt_json(rows_dir, "alien_1", submitted_sql="SELECT x")

    class StubGrader:
        def __call__(self, **_kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary", novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    from bird_interact_agents.eval.annotate import write_submission_skeleton
    written = write_submission_skeleton(
        rows_dir=rows_dir, instance_id="alien_1",
        selected_database="alien", benchmark="mini-interact",
        run_id="r1", task_row=SAMPLE_TASK_ROW,
        grader=StubGrader(), mode="overwrite",
        dry_run=True, repo_root=tmp_path,
    )
    assert written is not None
    assert not written.exists()


def test_submission_mode_init_skips_existing(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    from bird_interact_agents.eval import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata, submission_annotation_path,
        write_submission_annotation,
    )
    existing = SubmissionAnnotation(
        instance_id="alien_1", selected_database="alien",
        task_annotation_ref="x", annotated_by="human", annotated_at="2026-05-30",
        submission=SubmissionMetadata(
            cloud_run_id="r1", trajectory_path="t",
        ),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="pass",
            phase1_against_audited_primary="pass",
            phase1_against_any_audited_variant="pass",
            verdict="correct",
        ),
        failure_classification=FailureClassification(
            primary="other", agent_at_fault=False, remediation_target="other",
        ),
    )
    p = submission_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="r1", repo_root=tmp_path,
    )
    write_submission_annotation(existing, p)
    pre = p.read_bytes()

    rows_dir = tmp_path / "rows"
    _write_attempt_json(rows_dir, "alien_1", submitted_sql="SELECT x")

    class StubGrader:
        def __call__(self, **_kw):
            raise AssertionError("should not be invoked in init-skip path")

    from bird_interact_agents.eval.annotate import write_submission_skeleton
    written = write_submission_skeleton(
        rows_dir=rows_dir, instance_id="alien_1",
        selected_database="alien", benchmark="mini-interact",
        run_id="r1", task_row=SAMPLE_TASK_ROW,
        grader=StubGrader(), mode="init",
        dry_run=False, repo_root=tmp_path,
    )
    assert written is None
    assert p.read_bytes() == pre
