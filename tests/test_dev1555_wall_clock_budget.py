"""DEV-1555 follow-up: wall-clock budget enforced AT THE AGENT LEVEL.

Today the per-task wall-clock cap (``BIRD_INTERACT_PER_TASK_TIMEOUT_S``)
is enforced from OUTSIDE the SDK session via ``asyncio.wait_for`` in
``run.py``. When it fires, the agent's in-memory ``trajectory`` is lost
— last seen on Kimi r7 which returned ``traj=0, cost=0, error="per-task
wall-clock cap of 1200s exceeded"`` with no usable diagnosis signal.

The new tracker uses the same shape as ``make_context_budget_hook``:
the agent stamps a monotonic start time on a shared ``state`` dict; a
PostToolUse hook emits one-shot warnings at 80% / 90% of budget; a
PreToolUse hook DENIES non-submit tools at 100% so the agent is forced
to call ``submit_query`` / ``submit_sql`` next. The outer
``asyncio.wait_for`` stays as a runaway safety net at budget + grace,
so a model that ignores the deny still terminates eventually.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_clock(monkeypatch):
    """Drop-in for ``time.monotonic`` inside ``context_budget``."""
    from bird_interact_agents.agents.claude_sdk import context_budget as cb

    state = {"now": 1_000.0}

    def _monotonic():
        return state["now"]

    monkeypatch.setattr(cb.time, "monotonic", _monotonic)
    return state


def _set_start(state, t=1_000.0):
    """Snapshot of update_wall_clock_start at a known monotonic value."""
    state["wall_clock_start"] = t


# ---------------------------------------------------------------------------
# update_wall_clock_start
# ---------------------------------------------------------------------------

def test_update_wall_clock_start_writes_monotonic(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_wall_clock_start,
    )

    state: dict = {}
    update_wall_clock_start(state)
    assert state["wall_clock_start"] == 1_000.0
    # Second call updates (last-write wins, per make_context_budget_hook
    # 'latest, not max' semantics).
    fake_clock["now"] = 2_500.0
    update_wall_clock_start(state)
    assert state["wall_clock_start"] == 2_500.0


# ---------------------------------------------------------------------------
# PostToolUse warnings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_warning_fires_once_at_80pct(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    post_hook, _pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    # Under 80%.
    fake_clock["now"] = 1_079.9
    assert await post_hook({"tool_name": "x"}, "tu", None) == {}
    # Cross 80%.
    fake_clock["now"] = 1_081.0
    out = await post_hook({"tool_name": "x"}, "tu", None)
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PostToolUse"
    assert "WALL-CLOCK BUDGET" in hso["additionalContext"]
    assert "submit_query" in hso["additionalContext"]
    # Same threshold again → silent (one-shot).
    fake_clock["now"] = 1_085.0
    assert await post_hook({"tool_name": "x"}, "tu", None) == {}


@pytest.mark.asyncio
async def test_post_final_warning_at_90pct_independent_of_80pct(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    post_hook, _pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    # 80% first.
    fake_clock["now"] = 1_081.0
    out = await post_hook({"tool_name": "x"}, "tu", None)
    assert "WALL-CLOCK BUDGET" in out["hookSpecificOutput"]["additionalContext"]
    # Cross 90%.
    fake_clock["now"] = 1_091.0
    out = await post_hook({"tool_name": "x"}, "tu", None)
    assert "FINAL WARNING" in out["hookSpecificOutput"]["additionalContext"]
    # Same threshold again → silent.
    fake_clock["now"] = 1_095.0
    assert await post_hook({"tool_name": "x"}, "tu", None) == {}


@pytest.mark.asyncio
async def test_post_skipping_straight_to_90_fires_final_once(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    post_hook, _pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    fake_clock["now"] = 1_092.0
    out = await post_hook({"tool_name": "x"}, "tu", None)
    assert "FINAL WARNING" in out["hookSpecificOutput"]["additionalContext"]
    fake_clock["now"] = 1_099.0
    assert await post_hook({"tool_name": "x"}, "tu", None) == {}


# ---------------------------------------------------------------------------
# PreToolUse deny at 100% — submit tools allowed, others denied
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_deny_after_budget_for_non_submit_tools(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    _post_hook, pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    # Within budget → allow.
    fake_clock["now"] = 1_099.9
    assert await pre_hook(
        {"tool_name": "mcp__slayer__query"}, "tu", None,
    ) == {}
    # Past budget → deny non-submit.
    fake_clock["now"] = 1_100.5
    out = await pre_hook(
        {"tool_name": "mcp__slayer__query"}, "tu", None,
    )
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert "Wall-clock budget exhausted" in hso["permissionDecisionReason"]
    assert "submit_query" in hso["permissionDecisionReason"]


@pytest.mark.asyncio
async def test_pre_deny_allows_submit_query(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    _post, pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    fake_clock["now"] = 1_500.0  # well past budget
    assert await pre_hook(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, "tu", None,
    ) == {}


@pytest.mark.asyncio
async def test_pre_deny_allows_submit_sql_for_raw_agents(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    _post, pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_sql",
    )
    fake_clock["now"] = 1_500.0
    assert await pre_hook(
        {"tool_name": "mcp__bird-interact-tools__submit_sql"}, "tu", None,
    ) == {}


# ---------------------------------------------------------------------------
# Disable semantics: budget=0 or None → no-op (preserves the existing
# "BIRD_INTERACT_PER_TASK_TIMEOUT_S=0 to disable" UX).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 0, 0.0])
async def test_no_op_when_budget_disabled(fake_clock, budget):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    post_hook, pre_hook = make_wall_clock_budget_hook(
        state, budget_s=budget, submit_tool="submit_query",
    )
    fake_clock["now"] = 1_000_000.0  # arbitrarily far past
    assert await post_hook({"tool_name": "x"}, "tu", None) == {}
    assert await pre_hook({"tool_name": "x"}, "tu", None) == {}


@pytest.mark.asyncio
async def test_no_op_when_start_time_missing():
    """If the agent forgets to call update_wall_clock_start, both hooks
    must be no-ops rather than crashing."""
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}  # no wall_clock_start
    post_hook, pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    assert await post_hook({"tool_name": "x"}, "tu", None) == {}
    assert await pre_hook({"tool_name": "x"}, "tu", None) == {}


# ---------------------------------------------------------------------------
# Hook identity pins (so the agent-options registration test in
# test_dev1555_subagent_options.py can match by __name__)
# ---------------------------------------------------------------------------

def test_hook_callable_names_are_pinned(fake_clock):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_wall_clock_budget_hook,
    )

    state: dict = {}
    _set_start(state)
    post_hook, pre_hook = make_wall_clock_budget_hook(
        state, budget_s=100.0, submit_tool="submit_query",
    )
    assert post_hook.__name__ == "wall_clock_budget_warning"
    assert pre_hook.__name__ == "wall_clock_budget_deny"
