"""Verify the smolagents agent's tools work in isolation (no LLM calls)."""

import pytest

from bird_interact_agents.config import settings


def _make_state():
    """Build a TaskState bound to the alien DB."""
    from bird_interact_agents.agents.smolagents.agent import TaskState
    from bird_interact_agents.harness import load_db_data_if_needed, SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    return TaskState(
        status=SampleStatus(idx=0, original_data=task_data),
        data_path_base=settings.db_path,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
        slayer_storage_dir="./slayer_storage/alien",
    )


def test_taskstate_constructible():
    state = _make_state()
    assert state.status is not None


def test_smolagents_imports():
    from bird_interact_agents.agents.smolagents.agent import (  # noqa: F401
        SmolagentsAgent,
        TaskState,
        _build_native_tools,
        _build_prompt,
    )

    assert SmolagentsAgent is not None


def test_native_tools_raw_mode():
    from bird_interact_agents.agents.smolagents.agent import _build_native_tools

    state = _make_state()
    tools = _build_native_tools(state, "raw")
    # smolagents @tool returns a Tool object — its name is on .name
    names = {t.name for t in tools}
    expected = {
        "execute_sql",
        "get_schema",
        "get_all_column_meanings",
        "get_column_meaning",
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
        "ask_user",
        "submit_sql",
    }
    assert names == expected


def test_native_tools_slayer_mode():
    from bird_interact_agents.agents.smolagents.agent import _build_native_tools

    state = _make_state()
    tools = _build_native_tools(state, "slayer")
    names = {t.name for t in tools}
    assert names == {"ask_user", "submit_query"}


def test_submit_sql_with_correct_sql():
    """submit_sql tool marks phase1_passed when given gold SQL.

    smolagents tools are synchronous, so this test is sync too.
    """
    from bird_interact_agents.agents.smolagents.agent import (
        _build_native_tools,
        TaskState,
    )
    from bird_interact_agents.harness import (
        SampleStatus,
        load_db_data_if_needed,
        load_tasks,
    )

    tasks = load_tasks(settings.data_path, limit=1)
    task = tasks[0]
    db_name = task["selected_database"]
    load_db_data_if_needed(db_name, settings.db_path)

    state = TaskState(
        status=SampleStatus(idx=0, original_data=task),
        data_path_base=settings.db_path,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    tools = _build_native_tools(state, "raw")
    submit_sql = next(t for t in tools if t.name == "submit_sql")

    sol_sql = task["sol_sql"][0] if isinstance(task["sol_sql"], list) else task["sol_sql"]
    # smolagents Tool is callable — its forward() method runs the underlying fn
    submit_sql(sol_sql)

    assert state.result is not None
    assert state.result["phase1_passed"] is True


@pytest.mark.asyncio
async def test_prompt_building_raw_a():
    from bird_interact_agents.agents.smolagents.agent import _build_prompt

    state = _make_state()
    task_data = {**state.status.original_data, "amb_user_query": "How many signals are there?"}
    prompt = await _build_prompt("raw", "a-interact", task_data, 12.0, state)
    assert "How many signals" in prompt
    assert "alien" in prompt
    assert "12" in prompt
