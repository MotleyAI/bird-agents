"""DEV-1605: stamp the consumed OTF reference into the per-task annotation.

* ``ConsumedReference`` is a small Pydantic record {db, version,
  encoder_model, reference_fp}.
* ``SubmissionAnnotation`` gains an additive optional ``consumed_reference``
  field (default ``None``) — NO ``schema_version`` bump, so existing v1
  annotations on disk keep validating (the field is simply ``None``).
* ``grade_in_place`` constructors thread the value onto the annotation.
* ``dedupe_consumed_references`` builds the per-db manifest list.
"""

from __future__ import annotations

import json

from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
from bird_interact_agents.slayer_otf.encoder_types import ConsumedReference
from tests.test_eval_annotation_schema import _make_submission_annotation


def _cr(db="alien", version="opus-4-7", model="anthropic/claude-opus-4-7", fp="fp1"):
    return ConsumedReference(
        db=db, version=version, encoder_model=model, reference_fp=fp,
    )


def test_consumed_reference_fields():
    cr = _cr()
    assert cr.db == "alien"
    assert cr.version == "opus-4-7"
    assert cr.encoder_model == "anthropic/claude-opus-4-7"
    assert cr.reference_fp == "fp1"


def test_annotation_consumed_reference_defaults_none():
    ann = _make_submission_annotation()
    assert ann.consumed_reference is None


def test_annotation_consumed_reference_roundtrips(tmp_path):
    ann = _make_submission_annotation().model_copy(
        update={"consumed_reference": _cr()}
    )
    p = tmp_path / "ann.json"
    p.write_text(ann.model_dump_json(indent=2))
    back = SubmissionAnnotation.model_validate_json(p.read_text())
    assert back.consumed_reference == _cr()
    # schema_version is unchanged (still 1) — additive field, not a bump.
    assert back.schema_version == 1


def test_old_json_without_field_parses_to_none():
    """A pre-DEV-1605 annotation JSON (no consumed_reference key) must still
    validate, with the field defaulting to None."""
    ann = _make_submission_annotation()
    data = json.loads(ann.model_dump_json())
    data.pop("consumed_reference", None)  # simulate old on-disk shape
    back = SubmissionAnnotation.model_validate(data)
    assert back.consumed_reference is None


def test_write_failed_submission_annotation_threads_consumed_reference(tmp_path):
    """Codex HIGH#3: the failed-annotation constructor must accept + persist
    the consumed reference."""
    from bird_interact_agents.eval.grade_in_place import (
        write_failed_submission_annotation,
    )

    out = write_failed_submission_annotation(
        rows_dir=tmp_path / "rows",
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        run_id="20260625tABCD",
        trajectory_path="rows/alien_1/attempt-1.json",
        failure_details="no submitted_sql",
        consumed_reference=_cr(db="alien"),
    )
    ann = SubmissionAnnotation.model_validate_json(out.read_text())
    assert ann.consumed_reference == _cr(db="alien")


def test_dedupe_consumed_references_per_db():
    from bird_interact_agents.agents._pre_encoded import dedupe_consumed_references

    items = [
        _cr(db="alien", fp="fpA"),
        None,
        _cr(db="alien", fp="fpA"),       # duplicate of the same db
        _cr(db="households", fp="fpH"),
    ]
    out = dedupe_consumed_references(items)
    by_db = {c.db: c for c in out}
    assert set(by_db) == {"alien", "households"}
    assert by_db["alien"].reference_fp == "fpA"
    assert by_db["households"].reference_fp == "fpH"
    # stable: list, not dict (per the no-Dict-output preference / JSON manifest)
    assert isinstance(out, list)


def test_collect_consumed_references_from_run_dir(tmp_path):
    from bird_interact_agents.agents._pre_encoded import (
        collect_consumed_references_from_run_dir,
    )

    # two tasks for db 'alien' (same ref), one for 'households', one non-otf.
    def _write(iid, db, cr):
        d = tmp_path / "rows" / iid
        d.mkdir(parents=True)
        ann = _make_submission_annotation().model_copy(
            update={"consumed_reference": cr, "instance_id": iid,
                    "selected_database": db}
        )
        (d / "submission_annotation.json").write_text(ann.model_dump_json())

    _write("alien_1", "alien", _cr(db="alien", fp="fpA"))
    _write("alien_2", "alien", _cr(db="alien", fp="fpA"))
    _write("households_1", "households", _cr(db="households", fp="fpH"))
    _write("raw_1", "alien", None)  # non-otf: no consumed_reference

    out = collect_consumed_references_from_run_dir(tmp_path)
    by_db = {c.db: c for c in out}
    assert set(by_db) == {"alien", "households"}
    assert by_db["alien"].reference_fp == "fpA"
