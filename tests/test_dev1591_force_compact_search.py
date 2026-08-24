"""DEV-1591: hardwire SLayer ``search`` to ``compact=True``.

Trajectory analysis of the cea364 opus run showed the model *explicitly*
passing ``compact=False`` on ~1/3 of broad ``question=`` discovery searches —
each one dragging a ~55K-char, 10-entity full render into the cached context
for every subsequent turn (~6x the cache-read of the raw-mode agent on the
same task). The prose-only prompt rule already forbade this and the model
ignored it, so the fix enforces it mechanically with a PreToolUse hook:

* Broad discovery ``search`` is description-only (``compact=True``), always.
* All targeted detail reads move to ``inspect`` (``compact=False``), which the
  hook never touches.

This pins the hook's behaviour (production code → required test). Per the
no-prompt-content-tests rule, the prompt rewrite itself is validated
behaviourally (cloud smoke), not here.
"""
from __future__ import annotations

import asyncio

import pytest

from bird_interact_agents.agents.claude_sdk_otf.agent import (
    _SLAYER_SEARCH_TOOL,
    _force_compact_search_hook,
)


def _run_hook(tool_name, tool_input):
    return asyncio.run(
        _force_compact_search_hook(
            {"tool_name": tool_name, "tool_input": tool_input},
            "tool-use-id",
            None,
        )
    )


def _updated_input(result):
    return result["hookSpecificOutput"]["updatedInput"]


def test_search_tool_name_is_the_full_mcp_name():
    assert _SLAYER_SEARCH_TOOL == "mcp__slayer__search"


def test_forces_compact_true_when_search_passes_compact_false():
    result = _run_hook(
        _SLAYER_SEARCH_TOOL,
        {"question": "economic status of a house", "max_results": 10, "compact": False},
    )
    out = _updated_input(result)
    assert out["compact"] is True
    # Other args are preserved verbatim.
    assert out["question"] == "economic status of a house"
    assert out["max_results"] == 10
    assert result["hookSpecificOutput"]["hookEventName"] == "PreToolUse"


def test_keeps_compact_true_when_already_true():
    result = _run_hook(
        _SLAYER_SEARCH_TOOL, {"question": "x", "compact": True}
    )
    assert _updated_input(result)["compact"] is True


def test_sets_compact_true_when_absent():
    result = _run_hook(_SLAYER_SEARCH_TOOL, {"question": "x"})
    assert _updated_input(result)["compact"] is True


def test_does_not_mutate_the_original_tool_input():
    original = {"question": "x", "compact": False}
    _run_hook(_SLAYER_SEARCH_TOOL, original)
    assert original["compact"] is False


def test_ignores_non_search_tools():
    # inspect is the targeted-detail tool — its compact=False must survive.
    assert _run_hook("mcp__slayer__inspect", {"reference": "db.m.c", "compact": False}) == {}
    assert _run_hook("mcp__slayer__create_model", {"name": "m"}) == {}
    assert _run_hook("mcp__bird-interact-tools__submit_query", {"query_json": "{}"}) == {}


def test_ignores_non_dict_tool_input():
    assert _run_hook(_SLAYER_SEARCH_TOOL, None) == {}
    assert _run_hook(_SLAYER_SEARCH_TOOL, "not-a-dict") == {}


def test_missing_tool_name_is_a_noop():
    result = asyncio.run(
        _force_compact_search_hook({"tool_input": {"question": "x"}}, "id", None)
    )
    assert result == {}


# ---------------------------------------------------------------------------
# v1 / single-agent path: the in-process SLayer native `search` wrapper.
#
# The two-stage v1 agents (and the non-claude_sdk frameworks' single-agent
# prompts) reach SLayer through `_make_slayer_native(...)`, an in-process
# `mcp__bird-interact-tools__*` wrapper — NOT the `mcp__slayer__*` stdio
# server the PreToolUse hook above matches. So enforcement for that path
# lives in the wrapper handler (mirroring its existing write-normalization),
# and is pinned here.
# ---------------------------------------------------------------------------


def _call_native_search(monkeypatch, args):
    """Build the in-process `search` native, stub out storage + the engine
    fn, and return the kwargs the SLayer fn actually received."""
    from bird_interact_agents.agents.claude_sdk import agent as sdk_agent

    monkeypatch.setattr(sdk_agent, "_ensure_slayer_storage_attached", lambda: None)

    received = {}

    def fake_get_fn(name):
        assert name == "search"

        def _fn(**kwargs):
            received.update(kwargs)
            return "ok"

        return _fn

    monkeypatch.setattr(sdk_agent._query_mod, "_get_slayer_tool_fn", fake_get_fn)

    native = sdk_agent._make_slayer_native("search")
    asyncio.run(native.handler(args))
    return received


def test_native_search_schema_hides_compact_from_the_agent():
    """v1 owns the advertised schema, so `compact` is DROPPED entirely — the
    agent never sees a param it isn't allowed to set (better than exposing it
    and silently overriding). The cached source schema must stay intact, and
    other params + other tools keep their `compact`."""
    from bird_interact_agents.agents.claude_sdk import agent as sdk_agent

    search = sdk_agent._make_slayer_native("search")
    props = search.input_schema.get("properties", {})
    assert "compact" not in props
    assert "question" in props  # the real discovery params survive

    # The lru_cached metadata is shared — stripping must not mutate it.
    _, raw = sdk_agent._slayer_tool_metadata("search")
    assert "compact" in raw.get("properties", {})

    # Only `search` is stripped; detail-read tools keep their compact knob.
    im = sdk_agent._make_slayer_native("inspect_model")
    assert "compact" in im.input_schema.get("properties", {})


def test_schema_without_param_drops_from_properties_and_required():
    from bird_interact_agents.agents.claude_sdk.agent import _schema_without_param

    src = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "compact": {"type": "boolean"}},
        "required": ["a", "compact"],
    }
    out = _schema_without_param(src, "compact")
    assert "compact" not in out["properties"]
    assert out["required"] == ["a"]
    # Input untouched.
    assert "compact" in src["properties"]
    assert src["required"] == ["a", "compact"]


def test_native_search_wrapper_forces_compact_true(monkeypatch):
    received = _call_native_search(
        monkeypatch, {"question": "economic status of a house", "compact": False}
    )
    assert received["compact"] is True
    assert received["question"] == "economic status of a house"


def test_native_search_wrapper_sets_compact_when_absent(monkeypatch):
    received = _call_native_search(monkeypatch, {"question": "x"})
    assert received["compact"] is True


def test_native_inspect_wrapper_does_not_force_compact(monkeypatch):
    """The detail-read native (`inspect_model`) must keep its caller's
    compact value — only `search` is hardwired."""
    from bird_interact_agents.agents.claude_sdk import agent as sdk_agent

    monkeypatch.setattr(sdk_agent, "_ensure_slayer_storage_attached", lambda: None)
    received = {}

    def fake_get_fn(name):
        def _fn(**kwargs):
            received.update(kwargs)
            return "ok"

        return _fn

    monkeypatch.setattr(sdk_agent._query_mod, "_get_slayer_tool_fn", fake_get_fn)
    native = sdk_agent._make_slayer_native("inspect_model")
    asyncio.run(native.handler({"model": "m", "compact": False}))
    assert received["compact"] is False


# ---------------------------------------------------------------------------
# `inspect` must be EXECUTABLE wherever the prompts now route detail reads to
# it. The claude_sdk agents gate tool execution by an explicit allow-list /
# native-name set (`allowed_tools` only auto-executes listed tools), so an
# unlisted `inspect` would be denied in headless runs. (The pydantic_ai*
# frameworks expose the whole SLayer MCP toolset, so they need no analogous
# pin.)
# ---------------------------------------------------------------------------


def test_inspect_is_on_v0_otf_slayer_allowlist():
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS

    assert "inspect" in SLAYER_MCP_TOOLS


def test_inspect_is_a_buildable_native():
    from bird_interact_agents.agents.claude_sdk.agent import _SLAYER_NATIVE_NAMES

    assert "inspect" in _SLAYER_NATIVE_NAMES


def test_inspect_is_on_v1_discovery_surface():
    from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
        DISCOVERY_NATIVE_TOOL_NAMES,
    )

    assert "mcp__bird-interact-tools__inspect" in DISCOVERY_NATIVE_TOOL_NAMES
