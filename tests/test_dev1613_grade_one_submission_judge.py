"""DEV-1613: ``grade_one_submission`` builds + forwards the inline judge
from the run's ``agent_model``, and its harness short-circuit is exempted
for ``insufficient`` tasks so the cascade (incl. the N5 judge) is the
authoritative verdict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tests._dev1613_helpers import _make_task_annotation


# ---------------------------------------------------------------------------
# Judge construction + forwarding
# ---------------------------------------------------------------------------


def _capture_grade_and_write(monkeypatch):
    captured: dict = {}

    def _fake_grade_and_write(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return Path("/tmp/ignored.json")

    import bird_interact_agents.eval.grade_in_place as gip
    monkeypatch.setattr(gip, "grade_and_write", _fake_grade_and_write)
    return captured


def test_grade_one_submission_builds_judge_from_agent_model(monkeypatch):
    import bird_interact_agents.eval.grade_in_place as gip
    from bird_interact_agents.eval.tolerant_grader import LiteLLMJudge

    captured = _capture_grade_and_write(monkeypatch)
    gip.grade_one_submission(
        task_data={"instance_id": "alien_1", "selected_database": "alien",
                   "sol_sql": ["SELECT 1"]},
        submitted_sql="SELECT 2",
        rows_dir=Path("/tmp"),
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        task_annotation=_make_task_annotation(verdict="insufficient"),
        harness_passed=False,
        agent_model="anthropic/claude-opus-4-7",
    )
    judge = captured.get("llm_judge")
    assert isinstance(judge, LiteLLMJudge)
    assert judge.model_name == "anthropic/claude-opus-4-7"


def test_grade_one_submission_no_judge_without_agent_model(monkeypatch):
    import bird_interact_agents.eval.grade_in_place as gip

    captured = _capture_grade_and_write(monkeypatch)
    gip.grade_one_submission(
        task_data={"instance_id": "alien_1", "selected_database": "alien",
                   "sol_sql": ["SELECT 1"]},
        submitted_sql="SELECT 2",
        rows_dir=Path("/tmp"),
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        task_annotation=_make_task_annotation(verdict="insufficient"),
        harness_passed=False,
        agent_model=None,
    )
    assert captured.get("llm_judge") is None


def test_grade_one_submission_explicit_judge_takes_precedence(monkeypatch):
    import bird_interact_agents.eval.grade_in_place as gip

    captured = _capture_grade_and_write(monkeypatch)
    sentinel = object()
    gip.grade_one_submission(
        task_data={"instance_id": "alien_1", "selected_database": "alien",
                   "sol_sql": ["SELECT 1"]},
        submitted_sql="SELECT 2",
        rows_dir=Path("/tmp"),
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        task_annotation=_make_task_annotation(verdict="insufficient"),
        harness_passed=False,
        agent_model="anthropic/claude-opus-4-7",
        llm_judge=sentinel,
    )
    assert captured.get("llm_judge") is sentinel


# ---------------------------------------------------------------------------
# Harness short-circuit exemption for insufficient tasks
# ---------------------------------------------------------------------------


def _record_paths(monkeypatch):
    calls = {"grade_and_write": 0, "harness_confirmed": 0}

    import bird_interact_agents.eval.grade_in_place as gip

    def _fake_gw(**kwargs):  # noqa: ANN003
        calls["grade_and_write"] += 1
        return Path("/tmp/gw.json")

    def _fake_hc(**kwargs):  # noqa: ANN003
        calls["harness_confirmed"] += 1
        return Path("/tmp/hc.json")

    monkeypatch.setattr(gip, "grade_and_write", _fake_gw)
    monkeypatch.setattr(gip, "_write_harness_confirmed_annotation", _fake_hc)
    return calls


def _call(monkeypatch, ann, harness_passed: bool):
    import bird_interact_agents.eval.grade_in_place as gip
    gip.grade_one_submission(
        task_data={"instance_id": "alien_1", "selected_database": "alien",
                   "sol_sql": ["SELECT 1"]},
        submitted_sql="SELECT 1",
        rows_dir=Path("/tmp"),
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        task_annotation=ann,
        harness_passed=harness_passed,
        agent_model="anthropic/claude-opus-4-7",
    )


def test_short_circuit_exempts_insufficient_even_when_harness_passed(monkeypatch):
    """Insufficient task + harness_passed=True (e.g. the in-task judge
    flipped phase1_passed) MUST still run the full cascade, NOT the
    harness-confirmed all-pass shortcut — otherwise the N5 verdict is
    lost. ``original_gold_is_correct`` is left None (the risky case)."""
    calls = _record_paths(monkeypatch)
    ann = _make_task_annotation(verdict="insufficient")
    assert ann.original_gold_is_correct is None  # the case the guard must catch
    _call(monkeypatch, ann, harness_passed=True)
    assert calls["grade_and_write"] == 1
    assert calls["harness_confirmed"] == 0


def test_short_circuit_still_taken_for_sufficient_harness_pass(monkeypatch):
    """Sufficient task + harness_passed=True keeps the fast path —
    behaviour unchanged for the common case."""
    calls = _record_paths(monkeypatch)
    ann = _make_task_annotation(verdict="sufficient", evaluator_prompt=None)
    # sufficient annotations may carry original_gold_is_correct=True
    object.__setattr__(ann, "original_gold_is_correct", True)
    _call(monkeypatch, ann, harness_passed=True)
    assert calls["harness_confirmed"] == 1
    assert calls["grade_and_write"] == 0
