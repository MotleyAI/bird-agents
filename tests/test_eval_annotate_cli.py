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
    assert ann.provenance.task_jsonl_path.endswith("mini_interact.jsonl"), (
        f"task_jsonl_path should use underscore form; got {ann.provenance.task_jsonl_path!r}"
    )


def test_masked_terms_string_metadata_evidence_wrapped_in_list():
    """metadata_evidence as a plain string (real task data uses e.g. 'KB 3')
    must be wrapped in a list, not iterated char-by-char."""
    from bird_interact_agents.eval.annotate import generate_task_annotation

    row = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "amb_user_query": "q",
        "external_knowledge": [],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {"term": "x", "metadata_evidence": "KB 3"},
            ],
        },
    }
    ann = generate_task_annotation(task_row=row, benchmark="mini-interact")
    assert ann.masked_terms[0].metadata_evidence == ["KB 3"], (
        f"String metadata_evidence should be wrapped as list; got {ann.masked_terms[0].metadata_evidence!r}"
    )


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
    assert ann.task_annotation_ref.startswith("annotations/mini-interact/"), (
        f"task_annotation_ref should use hyphenated form; got {ann.task_annotation_ref!r}"
    )


def test_submission_annotation_reads_latest_attempt_not_hardcoded(tmp_path):
    """generate_submission_annotation picks attempt-2.json over attempt-1.json
    when both exist (regression: previously hardcoded attempt-1.json)."""
    from bird_interact_agents.eval.annotate import generate_submission_annotation

    rows_dir = tmp_path / "rows"
    d = rows_dir / "alien_1"
    d.mkdir(parents=True, exist_ok=True)
    base = {
        "instance_id": "alien_1",
        "trajectory": [],
        "duration_s": 1.0,
        "usage": {"cost_usd_agent": 0.01, "cost_usd_user_sim": 0.0,
                  "n_agent_turns": 1, "n_ask_user_calls": 0},
        "predicted_row_count": 0,
    }
    (d / "attempt-1.json").write_text(json.dumps({**base, "submitted_sql": "SELECT stale"}))
    (d / "attempt-2.json").write_text(json.dumps({**base, "submitted_sql": "SELECT latest"}))

    class StubGrader:
        def __call__(self, *, submitted_sql, **_kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            self.seen_sql = submitted_sql
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

    grader = StubGrader()
    generate_submission_annotation(
        rows_dir=rows_dir, instance_id="alien_1",
        selected_database="alien", benchmark="mini-interact",
        run_id="r1", task_row=SAMPLE_TASK_ROW,
        grader=grader,
    )
    assert grader.seen_sql == "SELECT latest", (
        f"Expected latest attempt SQL; got {grader.seen_sql!r}"
    )


def test_generate_submission_annotation_missing_dir_raises_file_not_found(tmp_path):
    """generate_submission_annotation raises FileNotFoundError (not iterdir crash)
    when the instance directory doesn't exist, after the _latest_attempt_file guard."""
    from bird_interact_agents.eval.annotate import generate_submission_annotation
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    # No sub-dir for alien_1 — it never ran.

    class StubGrader:
        def __call__(self, **_kw):
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary", novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    import pytest
    with pytest.raises(FileNotFoundError):
        generate_submission_annotation(
            rows_dir=rows_dir, instance_id="alien_1",
            selected_database="alien", benchmark="mini-interact",
            run_id="r1", task_row=SAMPLE_TASK_ROW,
            grader=StubGrader(),
        )


def test_resolve_db_sqlite_path_prefers_primary(tmp_path):
    """Primary {db}.sqlite is returned when it exists."""
    from bird_interact_agents.eval.annotate import _resolve_db_sqlite_path

    db_dir = tmp_path / "alien"
    db_dir.mkdir()
    primary = db_dir / "alien.sqlite"
    primary.touch()
    (db_dir / "alien_template.sqlite").touch()
    assert _resolve_db_sqlite_path(tmp_path, "alien") == primary


def test_resolve_db_sqlite_path_falls_back_to_template(tmp_path):
    """LiveSQLBench: _template.sqlite is used when {db}.sqlite is absent."""
    from bird_interact_agents.eval.annotate import _resolve_db_sqlite_path

    db_dir = tmp_path / "museum"
    db_dir.mkdir()
    tmpl = db_dir / "museum_template.sqlite"
    tmpl.touch()
    assert _resolve_db_sqlite_path(tmp_path, "museum") == tmpl


def test_resolve_db_sqlite_path_returns_primary_even_when_missing(tmp_path):
    """When neither file exists, primary path is returned (error surfaces at open)."""
    from bird_interact_agents.eval.annotate import _resolve_db_sqlite_path

    (tmp_path / "alien").mkdir()
    result = _resolve_db_sqlite_path(tmp_path, "alien")
    assert result.name == "alien.sqlite"


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
