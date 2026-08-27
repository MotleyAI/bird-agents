"""DEV-1822: ClaudeSDKOtfCubeAgent structure — exact tool surface, budget
mapping, prompt, hermetic session, and boundary rejections."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from bird_interact_agents.agents.claude_sdk_otf_cube import agent as cube_mod
from bird_interact_agents.agents.claude_sdk_otf_cube import ClaudeSDKOtfCubeAgent


_EXPECTED_TOOLS = {
    "cube_meta", "cube_load", "cube_sql", "submit_cube_query",
    "get_schema", "get_column_meaning", "get_all_column_meanings",
    "get_all_external_knowledge_names", "get_knowledge_definition",
    "get_all_knowledge_definitions",
}


def test_tool_surface_exact():
    names = {t.name for t in cube_mod._CUBE_TOOLS}
    assert names == _EXPECTED_TOOLS


def test_no_sql_or_ask_user_tools():
    names = {t.name for t in cube_mod._CUBE_TOOLS}
    assert "execute_sql" not in names
    assert "ask_user" not in names
    assert "submit_sql" not in names
    assert "submit_query" not in names


def test_submit_tool_mapping_for_cube():
    from bird_interact_agents.agents._submit import SUBMIT_TOOL_BY_QUERY_MODE
    assert SUBMIT_TOOL_BY_QUERY_MODE["cube"] == "submit_cube_query"


def test_gate_message_names_cube_submit_tool():
    from bird_interact_agents.agents._submit import gate_or_none
    state = SimpleNamespace(status=SimpleNamespace(force_submit=True, remaining_budget=0.0))
    msg = gate_or_none(state, "cube_load", "cube")
    assert msg is not None and "submit_cube_query" in msg


def test_claude_sdk_gate_uses_cube_submit_tool_in_cube_mode():
    """Exercise the ACTUAL claude_sdk.agent._gate (not just the shared helper)
    so it can't retain a raw/slayer-only submit-tool branch."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import SampleStatus

    status = SampleStatus(idx=0, original_data={"selected_database": "alien"},
                          remaining_budget=0.0, total_budget=30.0)
    status.force_submit = True
    agent_mod._ctx_var.set({"status": status, "query_mode": "cube", "result": None})
    msg = agent_mod._gate("cube_load", status)
    assert msg is not None and "submit_cube_query" in msg


def test_action_costs_present():
    from bird_interact_agents.harness import ACTION_COSTS
    for t in ("cube_meta", "cube_load", "cube_sql", "submit_cube_query"):
        assert t in ACTION_COSTS


def test_prompt_formats():
    from bird_interact_agents.agents.claude_sdk_otf_cube.prompts import CUBE_ONE_SHOT
    text = CUBE_ONE_SHOT.format(budget=30.0, db_name="alien", user_query="how many?")
    assert "submit_cube_query" in text
    assert "cube" in text.lower()


def test_hermetic_session_used():
    src = inspect.getsource(cube_mod)
    assert "hermetic_claude_sdk_session" in src


def test_ctor_and_boundary_rejections():
    agent = ClaudeSDKOtfCubeAgent(model="anthropic/claude-haiku-4-5-20251001")
    assert agent.model == "anthropic/claude-haiku-4-5-20251001"
    # wrong query_mode rejected up front
    with pytest.raises(ValueError):
        asyncio.run(agent.run_task(
            {"selected_database": "alien", "instance_id": "alien_1",
             "dataset": "livesqlbench-base-lite", "amb_user_query": "q"},
            "/dev/null", 30.0, "raw", eval_mode="one-shot",
        ))
    # wrong eval_mode rejected up front
    with pytest.raises(ValueError):
        asyncio.run(agent.run_task(
            {"selected_database": "alien", "instance_id": "alien_1",
             "dataset": "livesqlbench-base-lite", "amb_user_query": "q"},
            "/dev/null", 30.0, "cube", eval_mode="a-interact",
        ))
