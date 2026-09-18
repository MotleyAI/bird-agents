"""DEV-1907: the SLayer MCP surface the claude_sdk agents advertise tracks
0.10.2 — 20 tools, no `query_nested`, no `help` — and every agent allow-list
derives against it without error.

Covers spec `slayer-agent-tool-surface`:
  * Requirement "No query_nested or help tool" / scenario "query_nested absent
    from the advertised surface".
"""

from __future__ import annotations

_PREFIX = "mcp__slayer__"

# The 0.10.2 MCP surface (probed from create_mcp_server in an isolated 0.10.2
# env). 0.9.6 carried a 21st tool, `query_nested`, folded into `query` in 0.10.2.
_EXPECTED_BARE = {
    "create_datasource", "create_model", "delete_datasource", "delete_model",
    "describe_datasource", "edit_datasource", "edit_model", "forget_memory",
    "get_datasource_priority", "ingest_datasource_models", "inspect",
    "inspect_model", "list_datasources", "models_summary", "query",
    "recommend_root_model", "save_memory", "search", "set_datasource_priority",
    "validate_models",
}
_EXPECTED_SURFACE = {f"{_PREFIX}{n}" for n in _EXPECTED_BARE}


def _surface():
    from bird_interact_agents.agents._slayer_tool_surface import all_slayer_mcp_tool_names

    all_slayer_mcp_tool_names.cache_clear()
    return all_slayer_mcp_tool_names()


def test_advertised_surface_is_the_0102_twenty_tool_set():
    assert _surface() == _EXPECTED_SURFACE


def test_surface_has_exactly_twenty_tools():
    assert len(_surface()) == 20


def test_query_nested_absent_from_surface():
    assert f"{_PREFIX}query_nested" not in _surface()


def test_help_absent_from_surface():
    assert f"{_PREFIX}help" not in _surface()


def test_agent_allow_lists_derive_without_query_nested_or_help():
    """Every claude_sdk allow-list derives against the pinned surface with no
    error, and neither dead tool appears in the allow-list or its complement.

    RED until the encoder allow-list drops `query_nested`: an allow-list naming
    a tool absent from the surface makes `derive_disallowed_slayer_tools` raise.
    """
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        SLAYER_MCP_TOOLS as otf_allow,
    )
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        SLAYER_MCP_TOOLS as encoder_allow,
    )

    for name, allow in [("otf", otf_allow), ("encoder", encoder_allow)]:
        assert "query_nested" not in allow, f"{name} allow-list still lists query_nested"
        assert "help" not in allow, f"{name} allow-list still lists help"
        disallowed = derive_disallowed_slayer_tools(allow)  # raises on an unknown tool
        assert f"{_PREFIX}query_nested" not in disallowed
        assert f"{_PREFIX}help" not in disallowed


def test_native_tool_registry_drops_query_nested_and_help():
    """The in-process native registry no longer registers a `query_nested`
    native nor allow-lists a SLayer `help` native."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    assert "query_nested" not in agent_mod._STATIC_NATIVE_TOOLS
    assert not hasattr(agent_mod, "query_nested"), "query_nested native still defined"
    assert "help" not in agent_mod._SLAYER_NATIVE_NAMES
    assert "query_nested" not in agent_mod._SLAYER_NATIVE_NAMES


def test_action_costs_table_names_no_dead_tools():
    """The harness action-cost table no longer prices the removed `help` /
    `query_nested` tools (spec: no dead tool in any action-cost table)."""
    from bird_interact_agents.harness import ACTION_COSTS

    assert "help" not in ACTION_COSTS
    assert "query_nested" not in ACTION_COSTS
