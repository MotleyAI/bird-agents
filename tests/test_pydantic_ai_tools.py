"""Verify the PydanticAI agent's tools work in isolation (no LLM calls)."""

import pytest

from bird_interact_agents.config import settings


def _make_deps():
    """Build a TaskDeps for the alien DB, ready to be passed to a tool fn."""
    from bird_interact_agents.agents.pydantic_ai.agent import TaskDeps
    from bird_interact_agents.harness import load_db_data_if_needed, SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    return TaskDeps(
        status=SampleStatus(idx=0, original_data=task_data),
        data_path_base=settings.db_path,
        slayer_storage_dir=f"{settings.slayer_storage_root}/alien"
        if hasattr(settings, "slayer_storage_root")
        else "./slayer_storage/alien",
    )


def test_taskdeps_constructible():
    """TaskDeps can be built with the harness state."""
    deps = _make_deps()
    assert deps.status is not None
    assert deps.data_path_base


def test_pydantic_ai_agent_imports():
    """The PydanticAI agent module imports without errors."""
    from bird_interact_agents.agents.pydantic_ai.agent import (  # noqa: F401
        PydanticAIAgent,
        TaskDeps,
        _build_raw_a_agent,
        _build_raw_c_agent,
        _build_slayer_agent,
    )

    assert PydanticAIAgent is not None


def test_pydantic_ai_agent_factory_smoke():
    """Each (query_mode, eval_mode) combo builds a different agent."""
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    pa = PydanticAIAgent(
        slayer_storage_root="./slayer_storage", model="anthropic:claude-sonnet-4-5"
    )
    a1 = pa._select_agent("raw", "a-interact")
    a2 = pa._select_agent("raw", "c-interact")
    a3 = pa._select_agent("slayer", "a-interact", "./slayer_storage/alien")
    a4 = pa._select_agent("slayer", "c-interact", "./slayer_storage/alien")
    # Each is its own Agent instance
    assert len({id(a1), id(a2), id(a3), id(a4)}) == 4


@pytest.mark.asyncio
async def test_prompt_building_raw_a():
    """The raw a-interact prompt is built correctly."""
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    pa = PydanticAIAgent(slayer_storage_root="./slayer_storage")
    deps = _make_deps()
    task_data = deps.status.original_data
    task_data["amb_user_query"] = "How many signals are there?"
    prompt = await pa._build_prompt("raw", "a-interact", task_data, 12.0, deps)
    assert "How many signals" in prompt
    assert "alien" in prompt
    assert "12" in prompt  # budget


@pytest.mark.asyncio
async def test_prompt_building_raw_c_includes_schema():
    """The raw c-interact prompt embeds the full schema and knowledge."""
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    pa = PydanticAIAgent(slayer_storage_root="./slayer_storage")
    deps = _make_deps()
    task_data = deps.status.original_data
    task_data["amb_user_query"] = "?"
    prompt = await pa._build_prompt("raw", "c-interact", task_data, 12.0, deps)
    assert "CREATE TABLE" in prompt  # schema injected
