"""DEV-1548: SLayer MCP tools listed in `disallowed_tools=` must (a) be
the exact six the trajectory audit identified as unused, (b) never
collide with the allow-list, and (c) be reachable from the a-interact
adapter via an explicit import — so a maintainer following stale
assertions cannot accidentally re-derive the wrong set.

The actual `ClaudeAgentOptions(disallowed_tools=...)` wiring is exercised
by the existing `_stub_env`-based adapter tests in
`tests/test_claude_sdk_otf_agent.py` and
`tests/test_claude_sdk_otf_ainteract_agent.py` (one new test per file).
This file only covers the cross-module / constant-only invariants.
"""

from __future__ import annotations


# The 6 SLayer MCP tools the agent never (or essentially never) calls in
# steady-state slayer-mode runs. `save_memory` is INTENTIONALLY OMITTED —
# kept on the allow-list (and OFF the disallow-list) to preserve encoder
# headroom even though current trajectory data shows zero calls. See
# DEV-1548 plan, Codex F4 — deliberate residual.
_EXPECTED_DISALLOWED = [
    "mcp__slayer__forget_memory",
    "mcp__slayer__get_datasource_priority",
    "mcp__slayer__set_datasource_priority",
    "mcp__slayer__create_datasource",
    "mcp__slayer__delete_datasource",
    "mcp__slayer__ingest_datasource_models",
]


def test_disallowed_slayer_tools_exact_membership():
    """Frozen contract on the 6 names. Each carries the `mcp__slayer__`
    prefix — that's the form the Claude Agent SDK matches against."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        SLAYER_MCP_DISALLOWED_TOOL_NAMES,
    )

    assert list(SLAYER_MCP_DISALLOWED_TOOL_NAMES) == _EXPECTED_DISALLOWED
    for name in SLAYER_MCP_DISALLOWED_TOOL_NAMES:
        assert name.startswith("mcp__slayer__"), name


def test_disallowed_tools_disjoint_from_allowed_tools():
    """A name cannot simultaneously appear in `SLAYER_MCP_TOOLS` (the
    allow list) and `SLAYER_MCP_DISALLOWED_TOOL_NAMES` (the disallow
    list) — `disallowed_tools=` would silently win, leaving the
    allow-list entry as a permission grant for a tool the model can
    never see. This is the contradictory-state failure mode Codex
    flagged on DEV-1548."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        SLAYER_MCP_DISALLOWED_TOOL_NAMES,
        _slayer_tool_names,
    )

    disallow = set(SLAYER_MCP_DISALLOWED_TOOL_NAMES)
    allow = set(_slayer_tool_names())
    overlap = disallow & allow
    assert not overlap, (
        f"SLAYER_MCP_DISALLOWED_TOOL_NAMES must be disjoint from "
        f"_slayer_tool_names(); overlap={sorted(overlap)}"
    )


def test_save_memory_explicitly_kept_off_disallow_list():
    """Pins the DEV-1548 interview decision: `save_memory` remains on the
    allow list AND must NOT appear in the disallow list. The encoder
    keeps the affordance even though current trajectory data shows zero
    calls."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        SLAYER_MCP_DISALLOWED_TOOL_NAMES,
        _slayer_tool_names,
    )

    assert "mcp__slayer__save_memory" not in set(SLAYER_MCP_DISALLOWED_TOOL_NAMES)
    assert "mcp__slayer__save_memory" in set(_slayer_tool_names())


def test_ainteract_module_imports_disallowed_tool_names_explicitly():
    """Codex F9 (post-plan): the a-interact adapter must consume
    `SLAYER_MCP_DISALLOWED_TOOL_NAMES` via an explicit import — NOT
    re-derive it locally — so the two adapters stay symmetric and a
    single edit to the source-of-truth list propagates everywhere.

    Reaching the constant as a top-level module attribute proves it was
    imported, and `is`-identity proves it's the same object (not a
    snapshot copy with drift potential)."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as one_shot
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import (
        agent as ainteract,
    )

    assert hasattr(ainteract, "SLAYER_MCP_DISALLOWED_TOOL_NAMES"), (
        "SLAYER_MCP_DISALLOWED_TOOL_NAMES must be imported into the "
        "a-interact adapter module so the same list is threaded into "
        "both adapters' ClaudeAgentOptions(disallowed_tools=...)"
    )
    assert (
        ainteract.SLAYER_MCP_DISALLOWED_TOOL_NAMES
        is one_shot.SLAYER_MCP_DISALLOWED_TOOL_NAMES
    )
