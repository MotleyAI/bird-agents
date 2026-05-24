"""Verify budget bookkeeping + force_submit gating across adapters.

Covers the CodeRabbit "budget / eval_mode enforcement" review group:
- harness.calculate_budget rejects unknown modes
- harness.update_budget trips force_submit at the boundary (<=)
- agno / mcp_agent / smolagents / pydantic_ai expose only [ask_user, submit_sql]
  in raw c-interact mode
- non-submit tools across all four adapters return the gate message and
  do NOT call execute_env_action when force_submit is set
"""

import asyncio
from typing import Any

import pytest

from bird_interact_agents.config import settings
from bird_interact_agents.harness import (
    ACTION_COSTS,
    SampleStatus,
    calculate_budget,
    update_budget,
)


# ---------------------------------------------------------------------------
# Harness-level fixes
# ---------------------------------------------------------------------------

def test_calculate_budget_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported budget mode"):
        calculate_budget({}, mode="bogus")


def test_calculate_budget_a_interact_explicit():
    # Explicit mode still works (no silent fall-through behaviour change).
    val = calculate_budget({}, patience=3, mode="a-interact")
    assert val == 6 + 0 + 6  # 6 + 2*amb(0) + 2*patience(3)


def test_calculate_budget_c_interact_explicit():
    val = calculate_budget({}, patience=3, mode="c-interact")
    assert val == ACTION_COSTS["ask_user"] * 3 + ACTION_COSTS["submit_sql"]


def test_update_budget_trips_force_submit_at_boundary():
    """`<=` boundary: when remaining_budget hits exactly submit_sql cost,
    force_submit must be set so the agent can't spend a smaller action and
    get stranded below the submit threshold.
    """
    submit_cost = ACTION_COSTS["submit_sql"]
    status = SampleStatus(
        idx=0,
        original_data={},
        remaining_budget=submit_cost + ACTION_COSTS["query"],
        total_budget=submit_cost + ACTION_COSTS["query"],
    )
    # Spend a `query` (cost=1). After this, remaining_budget == submit_cost.
    update_budget(status, "query")
    assert status.remaining_budget == submit_cost
    assert status.force_submit is True


# ---------------------------------------------------------------------------
# Adapter tool-list shape: raw c-interact returns only [ask_user, submit_sql]
# ---------------------------------------------------------------------------

def _make_state_for(adapter_module_path: str) -> Any:
    """Build the adapter's TaskState/Deps. The ones with TaskState share the
    same field names; pydantic_ai uses TaskDeps with similar shape.
    """
    from bird_interact_agents.harness import load_db_data_if_needed

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    status = SampleStatus(
        idx=0,
        original_data=task_data,
        remaining_budget=12.0,
        total_budget=12.0,
    )
    if adapter_module_path == "bird_interact_agents.agents.pydantic_ai.agent":
        from bird_interact_agents.agents.pydantic_ai.agent import TaskDeps

        return TaskDeps(
            status=status,
            data_path_base=settings.db_path,
            slayer_storage_dir="./slayer_storage/alien",
        )
    if adapter_module_path == "bird_interact_agents.agents.agno.agent":
        from bird_interact_agents.agents.agno.agent import TaskState
    elif adapter_module_path == "bird_interact_agents.agents.mcp_agent.agent":
        from bird_interact_agents.agents.mcp_agent.agent import TaskState
    elif adapter_module_path == "bird_interact_agents.agents.smolagents.agent":
        from bird_interact_agents.agents.smolagents.agent import TaskState
    else:
        raise ValueError(adapter_module_path)
    return TaskState(
        status=status,
        data_path_base=settings.db_path,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
        slayer_storage_dir="./slayer_storage/alien",
    )


@pytest.mark.parametrize("adapter", ["agno", "mcp_agent", "smolagents"])
def test_raw_c_interact_only_exposes_ask_and_submit(adapter):
    """In raw c-interact, agents must be limited to clarify-or-submit."""
    if adapter == "agno":
        from bird_interact_agents.agents.agno.agent import _build_native_tools as builder
        module = "bird_interact_agents.agents.agno.agent"
    elif adapter == "mcp_agent":
        from bird_interact_agents.agents.mcp_agent.agent import (
            _build_native_functions as builder,
        )
        module = "bird_interact_agents.agents.mcp_agent.agent"
    else:
        from bird_interact_agents.agents.smolagents.agent import _build_native_tools as builder
        module = "bird_interact_agents.agents.smolagents.agent"

    state = _make_state_for(module)
    tools = builder(state, "raw", "c-interact")
    # smolagents wraps with @tool — names live on the underlying object via .name
    names = {getattr(t, "name", None) or t.__name__ for t in tools}
    assert names == {"ask_user", "submit_sql"}


# ---------------------------------------------------------------------------
# Force_submit gating: a non-submit tool must NOT execute when force_submit
# is set. The agent is told to call submit_* instead.
# ---------------------------------------------------------------------------

def _set_force_submit(state_or_deps):
    state_or_deps.status.force_submit = True


def _patch_executor(monkeypatch) -> dict:
    """Patch the shared `_submit.execute_env_action` used by every adapter
    after the dedup pass. Returns a dict whose `"n"` key counts calls."""
    from bird_interact_agents.agents import _submit
    called = {"n": 0}
    monkeypatch.setattr(
        _submit, "execute_env_action",
        lambda *a, **kw: (called.__setitem__("n", called["n"] + 1), ("ran", 0))[1],
    )
    return called


@pytest.mark.asyncio
async def test_agno_execute_sql_blocked_when_force_submit(monkeypatch):
    from bird_interact_agents.agents.agno import agent as mod

    state = _make_state_for("bird_interact_agents.agents.agno.agent")
    _set_force_submit(state)
    called = _patch_executor(monkeypatch)

    tools = mod._build_native_tools(state, "raw", "a-interact")
    execute_sql = next(f for f in tools if f.__name__ == "execute_sql")
    out = await execute_sql("SELECT 1")
    assert "submit_sql" in out
    assert called["n"] == 0  # gate rejected before execute_env_action


@pytest.mark.asyncio
async def test_mcp_agent_execute_sql_blocked_when_force_submit(monkeypatch):
    from bird_interact_agents.agents.mcp_agent import agent as mod

    state = _make_state_for("bird_interact_agents.agents.mcp_agent.agent")
    _set_force_submit(state)
    called = _patch_executor(monkeypatch)

    fns = mod._build_native_functions(state, "raw", "a-interact")
    execute_sql = next(f for f in fns if f.__name__ == "execute_sql")
    out = await execute_sql("SELECT 1")
    assert "submit_sql" in out
    assert called["n"] == 0


def test_smolagents_execute_sql_blocked_when_force_submit(monkeypatch):
    from bird_interact_agents.agents.smolagents import agent as mod

    state = _make_state_for("bird_interact_agents.agents.smolagents.agent")
    _set_force_submit(state)
    called = _patch_executor(monkeypatch)

    tools = mod._build_native_tools(state, "raw", "a-interact")
    execute_sql = next(t for t in tools if t.name == "execute_sql")
    out = execute_sql("SELECT 1")
    assert "submit_sql" in out
    assert called["n"] == 0


def test_pydantic_ai_run_env_blocked_when_force_submit(monkeypatch):
    """pydantic_ai uses the shared `_submit.run_env_action`; verify the
    same gate runs from a TaskDeps-shaped state."""
    from bird_interact_agents.agents import _submit
    from bird_interact_agents.agents._tool_specs import BIRD_INTERACT_TOOLS

    deps = _make_state_for("bird_interact_agents.agents.pydantic_ai.agent")
    _set_force_submit(deps)
    called = _patch_executor(monkeypatch)

    spec = next(t for t in BIRD_INTERACT_TOOLS if t.name == "execute_sql")
    out = _submit.run_env_action(deps, spec, "raw", sql="SELECT 1")
    assert "submit_sql" in out
    assert called["n"] == 0
