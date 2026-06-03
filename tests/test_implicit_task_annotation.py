"""DEV-1515: implicit-default TaskAnnotation for instances without an
on-disk annotation file.

Contract:
* ``implicit_task_annotation(...)`` returns a schema-valid
  ``TaskAnnotation`` (the grader cannot use a "verdict='sufficient'"
  shortcut because TaskAnnotation has many required fields).
* Verdict is ``sufficient`` + ``original_gold_is_correct=True`` so the
  cascade collapses to N1 (audited primary == original gold).
* ``gold_variants`` is empty.
* ``evaluator_prompt`` is None — the LLM judge MUST NOT fire on these.
* The factory is a pure builder; it must not touch
  ``paths.annotations_root()`` or write any file.
* Re-running the grader against an instance without an annotation file
  produces zero new files under ``annotations/``.
"""
from __future__ import annotations


def test_implicit_task_annotation_is_schema_valid():
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    ann = implicit_task_annotation(
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="Some query.",
    )
    assert isinstance(ann, TaskAnnotation)
    assert ann.instance_id == "alien_1"
    assert ann.selected_database == "alien"
    assert ann.metadata_sufficiency.verdict == "sufficient"
    assert ann.original_gold_is_correct is True
    assert ann.gold_variants == []
    assert ann.evaluator_prompt is None


def test_implicit_task_annotation_provenance_is_set():
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    ann = implicit_task_annotation(
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="x",
    )
    assert ann.provenance.task_jsonl_instance_id == "alien_1"
    # The exact JSONL path is benchmark-specific; the key invariant is
    # that it's non-empty so downstream traceability works.
    assert ann.provenance.task_jsonl_path


def test_implicit_task_annotation_round_trips_through_pydantic():
    """An implicit annotation should serialize+revalidate, so the grader
    can hand it to any code path that expects a TaskAnnotation from disk."""
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    ann = implicit_task_annotation(
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="x",
    )
    revalidated = TaskAnnotation.model_validate_json(ann.model_dump_json())
    assert revalidated == ann


def test_implicit_task_annotation_does_not_write_to_disk(tmp_path, monkeypatch):
    """The factory is pure — it must not call any I/O helper. Monkeypatch
    write_task_annotation to raise; the factory should still return."""
    import bird_interact_agents.eval.annotation_io as io_mod

    def _no_write(*_a, **_kw):
        raise AssertionError(
            "implicit_task_annotation must not write to disk"
        )

    monkeypatch.setattr(io_mod, "write_task_annotation", _no_write)

    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    implicit_task_annotation(
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="x",
    )


def test_implicit_task_annotation_alias_benchmark_uses_canonical_jsonl_name():
    """Passing ``benchmark="mini-interact"`` (dash alias) must produce the
    canonical ``"mini_interact.jsonl"`` in provenance, not the literal
    ``"mini-interact.jsonl"`` fallback."""
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    ann = implicit_task_annotation(
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="x",
    )
    assert ann.provenance.task_jsonl_path == "mini_interact.jsonl"


def test_implicit_marker_distinguishes_from_human_authored():
    """Downstream consumers must be able to tell that an annotation
    was auto-synthesized vs. human-authored. ``annotated_by`` carries
    the marker."""
    from bird_interact_agents.eval.implicit_annotation import (
        IMPLICIT_ANNOTATED_BY,
        implicit_task_annotation,
    )

    ann = implicit_task_annotation(
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="x",
    )
    assert ann.annotated_by == IMPLICIT_ANNOTATED_BY
