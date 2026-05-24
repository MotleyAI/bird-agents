"""submit_raw_sql / submit_slayer_query must populate the new diagnostic
fields on `state.result` for every submission outcome class.

Mocks `execute_submit_action` and `capture_result_snapshot` so the test
runs without touching real DBs or BIRD-Interact upstream wiring.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class _FakeStatus:
    def __init__(self, *, original_data: dict[str, Any], current_phase: int = 1):
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
    def __init__(self, **kw):
        self.status = _FakeStatus(**kw)
        self.data_path_base = "/dev/null"
        self.result = None


def _patch_submit_helpers(observation: str, p1: bool, p2: bool, *,
                          predicted: dict | None = None,
                          gold: dict | None = None):
    """Patch the helpers submit_raw_sql / submit_slayer_query call into.
    Returns the patch context manager."""
    from contextlib import ExitStack
    stack = ExitStack()

    fake_eval = lambda sql, status, dpb: (observation, 1.0 if p1 else 0.0, p1, p2, p1)  # noqa: E731

    captures = []

    def fake_capture(sql, db_name, dpb, **kw):
        captures.append(sql)
        if sql is None or not sql:
            return None
        if "GOLD" in (sql or ""):
            return gold
        return predicted

    stack.enter_context(patch(
        "bird_interact_agents.agents._submit.execute_submit_action",
        fake_eval,
    ))
    stack.enter_context(patch(
        "bird_interact_agents.agents._submit.capture_result_snapshot",
        fake_capture,
    ))
    return stack, captures


def test_submit_raw_sql_writes_passed_phase1():
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _FakeState(original_data={
        "selected_database": "households",
        "sol_sql": ["GOLD SELECT *"],
    })
    snapshot = {"columns": [{"name": "n", "type": "int"}],
                "row_count": 1, "row_count_truncated": False,
                "sample_rows": [[1]]}

    with _patch_submit_helpers("Phase 1 SQL Correct! ...", True, False,
                                predicted=snapshot, gold=snapshot)[0]:
        submit_raw_sql(state, "PRED SELECT *")

    r = state.result
    assert r["phase1_passed"] is True
    assert r["submission_status"] == "passed_phase1"
    assert r["phase1_observation"].startswith("Phase 1 SQL Correct")
    assert r["predicted_result_json"] is not None
    assert r["gold_result_json"] is not None
    # phase2_observation pre-existed as None on a fresh state.result.
    assert r["phase2_observation"] is None


def test_submit_raw_sql_classifies_wrong_result():
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _FakeState(original_data={
        "selected_database": "households",
        "sol_sql": ["GOLD SELECT *"],
    })
    obs = "Submitted SQL failed test case in Phase 1. Reason: Default test case failed: rows differ"
    snap = {"columns": [], "row_count": 0,
            "row_count_truncated": False, "sample_rows": []}

    with _patch_submit_helpers(obs, False, False,
                                predicted=snap, gold=snap)[0]:
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["submission_status"] == "wrong_result"
    assert state.result["phase1_observation"] == obs


def test_submit_raw_sql_classifies_sql_runtime_error():
    from bird_interact_agents.agents._submit import submit_raw_sql

    state = _FakeState(original_data={
        "selected_database": "households",
        "sol_sql": ["GOLD SELECT *"],
    })
    obs = "Submitted SQL failed test case in Phase 1. Reason: Error executing submitted SQL: no such column: foo"

    with _patch_submit_helpers(obs, False, False, predicted=None, gold=None)[0]:
        submit_raw_sql(state, "PRED SELECT *")

    assert state.result["submission_status"] == "sql_runtime_error"


def test_submit_slayer_query_classifies_json_error():
    from bird_interact_agents.agents._submit import submit_slayer_query

    state = _FakeState(original_data={
        "selected_database": "households",
        "sol_sql": ["GOLD SELECT *"],
    })

    with _patch_submit_helpers("ignored", False, False)[0]:
        submit_slayer_query(state, "{not json", lambda s: object())

    r = state.result
    assert r["submission_status"] == "json_error"
    assert r["submitted_sql"] is None
    assert r["submitted_query"] == "{not json"
    # No snapshots collected for json errors.
    assert r["predicted_result_json"] is None
    assert r["gold_result_json"] is None


def test_submit_slayer_query_classifies_translation_error():
    from bird_interact_agents.agents._submit import submit_slayer_query

    state = _FakeState(original_data={
        "selected_database": "households",
        "sol_sql": ["GOLD SELECT *"],
    })

    class _Client:
        def sql_sync(self, _q):
            raise RuntimeError("unknown measure x")

    with _patch_submit_helpers("ignored", False, False)[0]:
        submit_slayer_query(state, '{"source_model": "x"}', lambda s: _Client())

    r = state.result
    assert r["submission_status"] == "translation_error"
    assert r["submitted_sql"] is None
    assert "unknown measure" in r["phase1_observation"]


def test_submit_slayer_query_classifies_passed_after_translation():
    from bird_interact_agents.agents._submit import submit_slayer_query

    state = _FakeState(original_data={
        "selected_database": "households",
        "sol_sql": ["GOLD SELECT *"],
    })
    snap = {"columns": [], "row_count": 0,
            "row_count_truncated": False, "sample_rows": []}

    class _Client:
        def sql_sync(self, _q):
            return "SELECT * FROM rendered"

    with _patch_submit_helpers("Phase 1 SQL Correct! ...", True, False,
                                predicted=snap, gold=snap)[0]:
        submit_slayer_query(state, '{"source_model": "x"}', lambda s: _Client())

    r = state.result
    assert r["submission_status"] == "passed_phase1"
    assert r["submitted_sql"] == "SELECT * FROM rendered"
    assert r["submitted_query"] == '{"source_model": "x"}'
    assert r["predicted_result_json"] is not None
