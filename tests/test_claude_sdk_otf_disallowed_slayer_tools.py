"""DEV-1644: the two-list SLayer tool design (a hand-written allow-list PLUS a
partial hand-written deny-list) drifted and leaked — every tool in neither
list kept its schema visible to the model, which then called it and ate a
"permission not granted" error. This collapses the design to a SINGLE
allow-list and derives ``disallowed = (all advertised slayer tools) − allowed``
(DEV-1644 Fix 1), so nothing can leak by construction.

Supersedes DEV-1548, which froze a partial 6-name deny-list. The
``SLAYER_MCP_DISALLOWED_TOOL_NAMES`` (one-shot) and ``DISALLOWED_TOOL_NAMES``
(encoder) constants are DELETED; the disallowed set is now computed by
``derive_disallowed_slayer_tools``.

The actual ``ClaudeAgentOptions(disallowed_tools=...)`` wiring is exercised by
the adapter tests in ``tests/test_claude_sdk_otf_v1_agent.py`` and
``tests/test_claude_sdk_otf_ainteract_v1_agent.py`` (which import the v0
modules). This file covers the constant-only / cross-module invariants.
"""

from __future__ import annotations


_LEAK_TOOLS = {
    "mcp__slayer__query",
    "mcp__slayer__query_nested",
    "mcp__slayer__describe_datasource",
    "mcp__slayer__edit_datasource",
    "mcp__slayer__delete_model",
}


def _surface():
    from bird_interact_agents.agents._slayer_tool_surface import (
        all_slayer_mcp_tool_names,
    )

    all_slayer_mcp_tool_names.cache_clear()
    return all_slayer_mcp_tool_names()


# ---------------------------------------------------------------------------
# Collapse pins: the hand-written deny-list constants must be GONE.
# ---------------------------------------------------------------------------


def test_one_shot_disallowed_constant_deleted():
    from bird_interact_agents.agents.claude_sdk_otf import agent as A

    assert not hasattr(A, "SLAYER_MCP_DISALLOWED_TOOL_NAMES"), (
        "DEV-1644 collapses to allow-list-only; the hand-written deny-list "
        "constant must be deleted and disallowed derived instead."
    )


def test_encoder_disallowed_constant_deleted():
    from bird_interact_agents.agents.claude_sdk_otf_encode import setup_encoder as S

    assert not hasattr(S, "DISALLOWED_TOOL_NAMES"), (
        "DEV-1644 collapses the encoder to allow-list-only; DISALLOWED_TOOL_NAMES "
        "must be deleted and disallowed derived instead."
    )


# ---------------------------------------------------------------------------
# Per-module no-leak: derived disallowed partitions the full surface.
# ---------------------------------------------------------------------------


def test_one_shot_allow_list_exact():
    """Codex round-2: pin the v0 agent allow-list exactly, so an accidental
    omission (which would then leak via the complement being over-broad) or
    an accidental grant is caught. Mirrors the encoder's exact-set pin."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS

    # DEV-1668: `help` removed (slayer 0.9.6 dropped the tool). list_datasources
    # / models_summary stay on the BASE allow-list (the legacy, lean=False
    # surface); lean drops them via effective_slayer_allow.
    assert set(SLAYER_MCP_TOOLS) == {
        "list_datasources", "models_summary", "inspect_model",
        "inspect", "search", "recommend_root_model", "create_model",
        "edit_model", "save_memory", "validate_models",
    }


def test_one_shot_no_leak():
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        SLAYER_MCP_TOOLS,
        _slayer_tool_names,
    )

    allow = set(_slayer_tool_names())
    disallowed = set(derive_disallowed_slayer_tools(SLAYER_MCP_TOOLS))
    # partition: disjoint + union == full surface.
    assert disallowed.isdisjoint(allow)
    assert disallowed | allow == _surface()
    # the historical leak tools are now hidden.
    assert _LEAK_TOOLS <= disallowed
    # save_memory stays a granted, VISIBLE affordance (encoder headroom).
    assert "mcp__slayer__save_memory" in allow
    assert "mcp__slayer__save_memory" not in disallowed


def test_encoder_no_leak_and_closes_list_datasources_gap():
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        SLAYER_MCP_TOOLS,
    )

    allow = {f"mcp__slayer__{t}" for t in SLAYER_MCP_TOOLS}
    disallowed = set(derive_disallowed_slayer_tools(SLAYER_MCP_TOOLS))
    assert disallowed.isdisjoint(allow)
    assert disallowed | allow == _surface()
    # The encoder self-tests DAGs via slayer query/query_nested — allowed,
    # never disallowed.
    for t in ("mcp__slayer__query", "mcp__slayer__query_nested"):
        assert t in allow
        assert t not in disallowed
    # Pre-existing leak the encoder's allow-list never granted and its old
    # deny-list never listed — now closed by the complement.
    assert "mcp__slayer__list_datasources" in disallowed
    # save_memory is REQUIRED here — allowed, never disallowed.
    assert "mcp__slayer__save_memory" in allow
    assert "mcp__slayer__save_memory" not in disallowed


def test_ainteract_shares_the_one_shot_allow_list():
    """The a-interact adapter must consume the one-shot allow-list (identity,
    not a drifted copy) so a single edit propagates to both adapters."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as one_shot
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as ainteract

    assert ainteract.SLAYER_MCP_TOOLS is one_shot.SLAYER_MCP_TOOLS
