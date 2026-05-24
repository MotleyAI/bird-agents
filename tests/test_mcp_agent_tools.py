"""Verify the mcp-agent agent's tools work in isolation (no LLM calls)."""

import pytest

from bird_interact_agents.config import settings


def _make_state():
    """Build a TaskState bound to the alien DB."""
    from bird_interact_agents.agents.mcp_agent.agent import TaskState
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
    """TaskState can be built with the harness state."""
    state = _make_state()
    assert state.status is not None
    assert state.data_path_base


def test_mcp_agent_imports():
    """The mcp_agent agent module imports without errors."""
    from bird_interact_agents.agents.mcp_agent.agent import (  # noqa: F401
        McpAgentAgent,
        TaskState,
        _build_native_functions,
        _build_settings,
        _build_prompt,
    )

    assert McpAgentAgent is not None


def test_native_functions_raw_mode():
    """Raw-mode native functions list contains the full SQL toolset."""
    from bird_interact_agents.agents.mcp_agent.agent import _build_native_functions

    state = _make_state()
    fns = _build_native_functions(state, "raw")
    names = {f.__name__ for f in fns}
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


def test_native_functions_slayer_mode():
    """SLayer-mode native functions list is just ask_user + submit_query."""
    from bird_interact_agents.agents.mcp_agent.agent import _build_native_functions

    state = _make_state()
    fns = _build_native_functions(state, "slayer")
    names = {f.__name__ for f in fns}
    assert names == {"ask_user", "submit_query"}


def test_settings_includes_slayer_in_slayer_mode(tmp_path):
    """In slayer mode, _build_settings registers a slayer MCP server."""
    from bird_interact_agents.agents.mcp_agent.agent import _build_settings

    s = _build_settings(
        "slayer", str(tmp_path / "alien"), "anthropic/claude-sonnet-4-5",
    )
    assert "slayer" in s.mcp.servers
    server = s.mcp.servers["slayer"]
    assert server.transport == "stdio"
    assert server.command.endswith("slayer")
    assert server.args == ["mcp", "--ingest-on-startup"]


def test_settings_empty_in_raw_mode(tmp_path):
    """In raw mode, _build_settings has no MCP servers (no slayer)."""
    from bird_interact_agents.agents.mcp_agent.agent import _build_settings

    s = _build_settings(
        "raw", str(tmp_path / "alien"), "anthropic/claude-sonnet-4-5",
    )
    assert "slayer" not in s.mcp.servers


def test_settings_routes_non_anthropic_through_openai_compat(tmp_path, monkeypatch):
    """For a non-Anthropic model, _build_settings populates Settings.openai
    with the matching provider's base_url + key (here: Cerebras)."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    from bird_interact_agents.agents.mcp_agent.agent import _build_settings

    s = _build_settings("raw", str(tmp_path / "alien"), "cerebras/zai-glm-4.7")
    assert s.openai is not None
    assert s.openai.base_url == "https://api.cerebras.ai/v1"
    assert s.openai.api_key == "test-key"
    assert s.openai.default_model == "zai-glm-4.7"


@pytest.mark.asyncio
async def test_submit_sql_function_with_correct_sql():
    """The submit_sql closure marks phase1_passed when given gold SQL."""
    from bird_interact_agents.agents.mcp_agent.agent import _build_native_functions
    from bird_interact_agents.harness import load_db_data_if_needed, load_tasks, SampleStatus
    from bird_interact_agents.agents.mcp_agent.agent import TaskState

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
    fns = _build_native_functions(state, "raw")
    submit_sql = next(f for f in fns if f.__name__ == "submit_sql")

    sol_sql = task["sol_sql"][0] if isinstance(task["sol_sql"], list) else task["sol_sql"]
    await submit_sql(sol_sql)

    assert state.result is not None
    assert state.result["phase1_passed"] is True


@pytest.mark.asyncio
async def test_prompt_building_raw_a():
    """raw a-interact prompt embeds the user query and budget."""
    from bird_interact_agents.agents.mcp_agent.agent import _build_prompt

    state = _make_state()
    task_data = {**state.status.original_data, "amb_user_query": "How many signals are there?"}
    prompt = await _build_prompt("raw", "a-interact", task_data, 12.0, state)
    assert "How many signals" in prompt
    assert "alien" in prompt
    assert "12" in prompt
