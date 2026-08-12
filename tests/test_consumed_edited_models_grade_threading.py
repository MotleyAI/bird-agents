"""DEV-1778: `consumed_edited_models` threads through EVERY annotation-
construction branch in grade_in_place (Codex #4): harness-confirmed shortcut,
full cascade, and the fail-everything writer."""
from __future__ import annotations

import json
from pathlib import Path

from bird_interact_agents.eval.annotation_schema import ConsumedEditedModels

_CONSUMED = {"db": "alien", "instance_id": "alien_1", "store_fp": "cafe" * 16}


def _read(rows_dir: Path) -> dict:
    return json.loads(
        (rows_dir / "alien_1" / "submission_annotation.json").read_text()
    )


def _implicit():
    from bird_interact_agents.eval.implicit_annotation import implicit_task_annotation
    return implicit_task_annotation(
        instance_id="alien_1", selected_database="alien",
        benchmark="mini-interact", amb_user_query="x",
    )


def test_coerce_accepts_dict_model_and_none():
    from bird_interact_agents.eval.grade_in_place import (
        _coerce_consumed_edited_models,
    )

    assert _coerce_consumed_edited_models(None) is None
    assert _coerce_consumed_edited_models(_CONSUMED) == ConsumedEditedModels(**_CONSUMED)
    model = ConsumedEditedModels(**_CONSUMED)
    assert _coerce_consumed_edited_models(model) is model


def test_write_failed_stamps_consumed(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.grade_in_place import (
        write_failed_submission_annotation,
    )

    rows_dir = tmp_path / "rows"
    write_failed_submission_annotation(
        rows_dir=rows_dir,
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        run_id="r1",
        trajectory_path="rows/alien_1/attempt-1.json",
        failure_details="no submitted_sql",
        consumed_edited_models=_CONSUMED,
    )
    assert _read(rows_dir)["consumed_edited_models"] == _CONSUMED


def test_write_harness_confirmed_stamps_consumed(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.grade_in_place import (
        _write_harness_confirmed_annotation,
    )

    rows_dir = tmp_path / "rows"
    _write_harness_confirmed_annotation(
        rows_dir=rows_dir,
        instance_id="alien_1",
        selected_database="alien",
        benchmark="mini-interact",
        run_id="r1",
        attempt=1,
        task_annotation=_implicit(),
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        duration_s=None,
        n_agent_turns=None,
        n_ask_user_calls=None,
        predicted_row_count=None,
        user_sim_interaction=None,
        consumed_edited_models=_CONSUMED,
    )
    ann = _read(rows_dir)
    assert ann["evaluation"]["rationale"] == "harness_confirmed"
    assert ann["consumed_edited_models"] == _CONSUMED


def test_grade_and_write_stamps_consumed(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    class FakeExec:
        def __call__(self, sql, *, db_path, conn):  # noqa: ARG002
            return ([(1,)], ["a"])

    grade_and_write(
        rows_dir=rows_dir,
        instance_id="alien_1",
        benchmark="mini-interact",
        run_id="r1",
        task_annotation=_implicit(),
        audited_gold_rows=[],
        original_sol_sql=["SELECT 1"],
        submitted_sql="SELECT 1",
        db_path=Path("/dev/null"),
        conn=None,
        executor=FakeExec(),
        trajectory_path="rows/alien_1/attempt-1.json",
        consumed_edited_models=_CONSUMED,
    )
    assert _read(rows_dir)["consumed_edited_models"] == _CONSUMED


def test_grade_one_submission_harness_confirmed_stamps(monkeypatch, tmp_path):
    """Integration: the harness short-circuit path (harness_passed=True) still
    carries the consumed record through to disk."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.grade_in_place import grade_one_submission

    rows_dir = tmp_path / "rows"
    grade_one_submission(
        task_data={"instance_id": "alien_1", "selected_database": "alien"},
        submitted_sql="SELECT 1",
        rows_dir=rows_dir,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        task_annotation=_implicit(),
        harness_passed=True,
        consumed_edited_models=_CONSUMED,
    )
    ann = _read(rows_dir)
    assert ann["evaluation"]["rationale"] == "harness_confirmed"
    assert ann["consumed_edited_models"] == _CONSUMED


def test_grade_one_submission_full_cascade_forwards_consumed(monkeypatch, tmp_path):
    """The non-harness path (harness_passed=False) must forward the exact
    consumed kwarg into grade_and_write (Codex #4 handoff)."""
    import bird_interact_agents.eval.grade_in_place as gip

    captured: dict = {}

    def _fake_grade_and_write(**kw):
        captured["consumed"] = kw.get("consumed_edited_models")
        p = tmp_path / "rows" / "alien_1" / "submission_annotation.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return p

    monkeypatch.setattr(gip, "grade_and_write", _fake_grade_and_write)
    gip.grade_one_submission(
        task_data={"instance_id": "alien_1", "selected_database": "alien"},
        submitted_sql="SELECT 1",
        rows_dir=tmp_path / "rows",
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        task_annotation=_implicit(),
        harness_passed=False,
        consumed_edited_models=_CONSUMED,
    )
    assert captured["consumed"] == _CONSUMED
