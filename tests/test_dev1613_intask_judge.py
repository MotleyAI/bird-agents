"""DEV-1613: the in-task submit-feedback grader runs the N5 judge for
``insufficient`` tasks (covering BOTH single-eval and dual-eval paths),
using the run's ``agent_model``. The agent-facing feedback is a single
boolean (``Result match: ...``), so a judge-accept flips that boolean to
True consistently (observation / reward / finished / phase1_passed).
"""
from __future__ import annotations

from contextlib import ExitStack
from typing import Any, List
from unittest.mock import patch

from tests._dev1613_helpers import _make_task_annotation


class _FakeStatus:
    def __init__(self, *, original_data: dict, current_phase: int = 1):
        self.original_data = original_data
        self.current_phase = current_phase
        self.remaining_budget = 100.0
        self.total_budget = 100.0
        self.force_submit = False
        self.last_reward = None
        self.successful_phase1_sql = ""
        self.phase1_completed = False
        self.phase2_completed = False
        self.idx = 0


class _FakeState:
    def __init__(self, *, original_data: dict, agent_model: Any = "MISSING"):
        self.status = _FakeStatus(original_data=original_data)
        self.data_path_base = "/dev/null"
        self.result = None
        if agent_model != "MISSING":
            self.agent_model = agent_model


class _CountingJudge:
    """Stand-in for LiteLLMJudge — records construction + accept value."""
    instances: List["_CountingJudge"] = []
    next_accept: Any = True

    def __init__(self, *, model, **kw):  # noqa: ANN003
        self.model = model
        self.accept = _CountingJudge.next_accept
        _CountingJudge.instances.append(self)

    def judge(self, **kwargs):  # noqa: ANN003
        return self.accept


def _patch_intask(stack: ExitStack, *, verdict: str, accept,
                  head_rows=None, recorder: list | None = None):
    """Patch the in-task judge collaborators inside ``_submit``."""
    import bird_interact_agents.agents._submit as sm

    ann = _make_task_annotation(verdict=verdict)
    stack.enter_context(patch.object(
        sm, "load_task_annotation_or_implicit", lambda **kw: ann))
    stack.enter_context(patch.object(
        sm, "load_audited_gold_rows_for", lambda **kw: []))
    stack.enter_context(patch.object(
        sm, "_fetch_head_rows", lambda *a, **kw: head_rows or []))

    _CountingJudge.instances = []
    _CountingJudge.next_accept = accept
    stack.enter_context(patch.object(sm, "LiteLLMJudge", _CountingJudge))

    if recorder is not None:
        real = sm.run_novel_reading_judge

        def _spy(**kwargs):  # noqa: ANN003
            recorder.append(kwargs)
            return real(**kwargs)
        stack.enter_context(patch.object(sm, "run_novel_reading_judge", _spy))


def _single_eval_state(agent_model="anthropic/claude-opus-4-7"):
    return _FakeState(
        original_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "dataset": "mini-interact",
            "sol_sql": ["GOLD SELECT *"],
            # NO original_sol_sql -> single-eval path
        },
        agent_model=agent_model,
    )


def _miss_eval():
    # (observation, reward, p1, p2, finished) -- deterministic miss.
    return lambda sql, status, dpb: ("Submitted. Result match: False", 0.0, False, False, False)  # noqa: E731


# ---------------------------------------------------------------------------
# Single-eval path
# ---------------------------------------------------------------------------


def test_intask_judge_accept_flips_single_eval(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state()
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            _miss_eval()))
        _patch_intask(stack, verdict="insufficient", accept=True)
        out = submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is True
    assert state.result["total_reward"] == 1.0
    assert state.result["finished"] is True
    # Agent-facing feedback is a single boolean -> must read True.
    assert "Result match: True" in out
    # The in-task judge MUST use the run's AGENT model.
    assert _CountingJudge.instances[0].model == "anthropic/claude-opus-4-7"


def test_intask_judge_reject_keeps_miss_single_eval(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state()
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            _miss_eval()))
        _patch_intask(stack, verdict="insufficient", accept=False)
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is False


def test_intask_judge_inconclusive_keeps_miss(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state()
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            _miss_eval()))
        _patch_intask(stack, verdict="insufficient", accept=None)
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is False


def test_intask_judge_not_constructed_for_sufficient(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state()
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            _miss_eval()))
        _patch_intask(stack, verdict="sufficient", accept=True)
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is False
    assert _CountingJudge.instances == []  # no judge built for sufficient


def test_intask_judge_skipped_when_no_agent_model(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state(agent_model="MISSING")  # attribute absent
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            _miss_eval()))
        _patch_intask(stack, verdict="insufficient", accept=True)
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is False
    assert _CountingJudge.instances == []


def test_intask_judge_receives_predicted_head_rows(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state()
    rows = [(1, "a"), (2, "b")]
    recorder: list = []
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            _miss_eval()))
        _patch_intask(stack, verdict="insufficient", accept=True,
                      head_rows=rows, recorder=recorder)
        submit_raw_sql(state, "PRED SELECT *")

    assert recorder, "run_novel_reading_judge was not invoked"
    assert recorder[0]["predicted_rows_head"] == rows


def test_intask_judge_does_not_fire_on_deterministic_pass(monkeypatch):
    """If the deterministic eval already passed, the judge must not run
    (and must not be constructed) — no wasted call."""
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _single_eval_state()
    pass_eval = lambda sql, status, dpb: ("Result match: True", 1.0, True, False, True)  # noqa: E731
    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.execute_submit_action",
            pass_eval))
        _patch_intask(stack, verdict="insufficient", accept=False)
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is True
    assert _CountingJudge.instances == []


# ---------------------------------------------------------------------------
# Dual-eval path (audited-gold overlay applied)
# ---------------------------------------------------------------------------


def test_fetch_head_rows_caps_at_twenty(tmp_path):
    """``_fetch_head_rows`` must return at most the first 20 predicted rows
    so the in-task judge prompt matches final grading's ``pred_rows[:20]``."""
    import sqlite3

    from bird_interact_agents.agents._submit import _fetch_head_rows

    db = tmp_path / "alien" / "alien.sqlite"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (n INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(25)])
    conn.commit()
    conn.close()

    rows = _fetch_head_rows(
        "SELECT n FROM t ORDER BY n",
        data_path_base=str(tmp_path),
        db_name="alien",
        db_file_path=str(db),
        benchmark=None,
    )
    assert 0 < len(rows) <= 20


def test_intask_judge_accept_flips_dual_eval(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _FakeState(
        original_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "dataset": "mini-interact",
            "sol_sql": ["AUD SELECT *"],
            "original_sol_sql": ["ORIG SELECT *"],
            # no audited_variants -> best-of block skipped
        },
        agent_model="anthropic/claude-opus-4-7",
    )

    def _fake_dual(*, pred_sql, audited_sol_sqls, original_sol_sqls,
                   status, data_path_base):
        return {
            "audited": {"observation": "Result match: False", "reward": 0.0,
                        "p1": False, "p2": False, "finished": False},
            "original": {"observation": "Result match: False", "reward": 0.0,
                         "p1": False, "p2": False, "finished": False},
        }

    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.evaluate_dual_gold",
            _fake_dual))
        _patch_intask(stack, verdict="insufficient", accept=True)
        out = submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is True
    assert state.result["total_reward"] == 1.0
    assert state.result["finished"] is True
    assert "Result match: True" in out
    assert _CountingJudge.instances[0].model == "anthropic/claude-opus-4-7"


def test_intask_judge_reject_keeps_miss_dual_eval(monkeypatch):
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _FakeState(
        original_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "dataset": "mini-interact",
            "sol_sql": ["AUD SELECT *"],
            "original_sol_sql": ["ORIG SELECT *"],
        },
        agent_model="anthropic/claude-opus-4-7",
    )

    def _fake_dual(*, pred_sql, audited_sol_sqls, original_sol_sqls,
                   status, data_path_base):
        return {
            "audited": {"observation": "Result match: False", "reward": 0.0,
                        "p1": False, "p2": False, "finished": False},
            "original": {"observation": "Result match: False", "reward": 0.0,
                         "p1": False, "p2": False, "finished": False},
        }

    with ExitStack() as stack:
        stack.enter_context(patch(
            "bird_interact_agents.agents._submit.evaluate_dual_gold",
            _fake_dual))
        _patch_intask(stack, verdict="insufficient", accept=False)
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["phase1_passed"] is False
