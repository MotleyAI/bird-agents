"""Verify the Claude SDK tool functions work in isolation (no LLM)."""

from types import SimpleNamespace

import pytest

from bird_interact_agents.config import settings
from bird_interact_agents.harness import ACTION_COSTS


@pytest.mark.asyncio
async def test_get_schema_tool():
    """The get_schema tool returns schema text."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import load_db_data_if_needed, SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=20.0, total_budget=20.0,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })

    result = await agent_mod.get_schema.handler({})
    assert "content" in result
    assert "CREATE TABLE" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_execute_sql_tool():
    """The execute_sql tool runs a query and returns results."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import load_db_data_if_needed, SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=20.0, total_budget=20.0,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })

    result = await agent_mod.execute_sql.handler({"sql": "SELECT 1"})
    assert "content" in result
    assert "1" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_submit_correct_sql_tool():
    """Submitting ground-truth SQL via the tool marks phase1 passed."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import (
        load_db_data_if_needed,
        load_tasks,
        SampleStatus,
    )

    tasks = load_tasks(settings.data_path, limit=1)
    task = tasks[0]
    db_name = task["selected_database"]
    load_db_data_if_needed(db_name, settings.db_path)
    agent_mod._ctx_var.set({
        "status": SampleStatus(idx=0, original_data=task),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })

    sol_sql = task["sol_sql"][0] if isinstance(task["sol_sql"], list) else task["sol_sql"]
    await agent_mod.submit_sql.handler({"sql": sol_sql})

    result = agent_mod._ctx_var.get().get("result")
    assert result is not None
    assert result["phase1_passed"] is True


def _seed_ctx(monkeypatch, *, remaining_budget=20.0):
    """Pre-populate `_ctx_var` with the minimum dict the submit wrappers expect."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=remaining_budget, total_budget=remaining_budget,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })
    return agent_mod


@pytest.mark.asyncio
async def test_submit_sql_charges_budget_exactly_once(monkeypatch):
    from bird_interact_agents.agents import _submit

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, False, True),
    )
    status = agent_mod._ctx_var.get()["status"]
    start = status.remaining_budget

    result = await agent_mod.submit_sql.handler({"sql": "SELECT 1"})

    assert status.remaining_budget == start - ACTION_COSTS["submit_sql"]
    text = result["content"][0]["text"]
    assert text.count("Remaining budget:") == 1


@pytest.mark.asyncio
async def test_submit_query_charges_budget_exactly_once(monkeypatch):
    from bird_interact_agents.agents import _submit

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    fake_client = SimpleNamespace(sql_sync=lambda d: "SELECT 1")
    agent_mod._ctx_var.get()["_slayer_client"] = fake_client
    status = agent_mod._ctx_var.get()["status"]
    start = status.remaining_budget

    result = await agent_mod.submit_query.handler({"query_json": '{"models": ["m"]}'})

    assert status.remaining_budget == start - ACTION_COSTS["submit_query"]
    text = result["content"][0]["text"]
    assert text.count("Remaining budget:") == 1


@pytest.mark.asyncio
async def test_submit_query_bad_json_propagates_helper_message(monkeypatch):
    agent_mod = _seed_ctx(monkeypatch)
    status = agent_mod._ctx_var.get()["status"]
    start = status.remaining_budget

    result = await agent_mod.submit_query.handler({"query_json": "{not json"})

    text = result["content"][0]["text"]
    assert "Invalid JSON" in text
    assert text.count("Remaining budget:") == 1
    # Failure paths now record diagnostic state on `state.result` (the
    # JSON-error is needed in results.db for offline failure-mode
    # analysis). Lock the classifier verdict + that the task didn't pass.
    rec = agent_mod._ctx_var.get().get("result")
    assert rec is not None
    assert rec.get("submission_status") == "json_error"
    assert rec.get("phase1_passed") is False
    # Per DEV-1432: deterministic pre-eval failures (JSON parse) are
    # FREE — `execute_submit_action` never ran, so budget is untouched.
    assert status.remaining_budget == start


def test_state_view_exposes_ctx_result_for_multi_submit_preservation():
    """Regression for the multi-submit observation drop (DEV-1511 follow-up
    flagged by Codex): `_state_view().result` MUST reflect `_ctx["result"]`
    so the submit helpers' cross-phase observation-preservation logic
    (`prior = state.result or {}`, then
    `state.result["phaseN_observation"] = prior.get(...)`) sees the prior
    phase's diagnostic state.

    Was hard-coded to None — meaning `prior` was always empty `{}`, so the
    helper silently blanked the other-phase observation on every
    multi-submit run. Once the agent submitted in phase 2, the phase-1
    observation captured by the prior submit was overwritten with None
    before reaching the finalize row.
    """
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    agent_mod._ctx_var.set({"result": {"phase1_observation": "obs1"}})
    view = agent_mod._state_view()
    assert view.result == {"phase1_observation": "obs1"}

    # And: the empty case still returns None (no LookupError, no KeyError).
    agent_mod._ctx_var.set({})
    view = agent_mod._state_view()
    assert view.result is None
