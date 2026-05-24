"""Root clarifier must not receive the SLayer MCP toolset.

The root's only job is to decompose the user's question and dispatch
each block to a sub-clarifier via `spawn_subagent`. It does not need
`search`, `help`, `inspect_model`, or any other SLayer tool — and
having them available tempts the model into looking up tables and
naming them in the handoff, which starves the sub-clarifier's
table-family disambiguation step.

This module tests that architectural constraint at the factory level.
"""

from __future__ import annotations

from bird_interact_agents.agents.pydantic_ai_recursive import factories


def test_root_agent_has_no_user_toolset_even_when_server_supplied():
    """Even when production passes a non-None `shared_slayer_server`,
    `_build_root_clarifier` must not wire it into the Agent's toolsets.
    Pydantic-ai stores user-supplied toolsets in `agent._user_toolsets`;
    after the build, that list must be empty for the root."""
    from pydantic_ai.toolsets import FunctionToolset

    def _probe() -> str:
        return "probe"

    probe_toolset = FunctionToolset(tools=[_probe])
    agent = factories._build_root_clarifier(
        model="test",
        model_settings=None,
        shared_slayer_server=probe_toolset,
        max_depth=3,
    )
    assert list(agent._user_toolsets) == [], (
        "Root clarifier received a user-supplied toolset; the root's "
        "only tool must be `spawn_subagent`. Fix: remove the "
        "`kwargs['toolsets']` wire-up in `_build_root_clarifier`."
    )
