"""DEV-1555 Stage 1: discovery/main subagent split — options wiring.

For each of the four ``claude_sdk_otf*`` agents, pins the contract between
the module-level partition constants and the ``ClaudeAgentOptions`` actually
handed to the SDK:

* ``options.tools == ["Task"]`` — Task is the ONLY re-enabled built-in
  (Bash/Edit/WebFetch/ToolSearch stay suppressed; ToolSearch removal is
  load-bearing, see the in-module comment).
* ``options.agents`` declares exactly one subagent named ``discovery`` whose
  ``AgentDefinition.tools`` equals the module's ``DISCOVERY_TOOLS``, with
  ``model=None`` (inherit) and ``maxTurns == DISCOVERY_MAX_TURNS``.
* ``options.allowed_tools`` is the exact union of both partitions plus
  ``Task`` — discovery tools MUST stay permission-granted globally
  (``disallowed_tools`` would block them inside the subagent too); the
  main-loop block is the ``partition_deny`` PreToolUse hook.
* The ``partition_deny`` hook is registered on PreToolUse with a matcher
  equal to ``"|".join(sorted(discovery_only))``.
* The ``context_budget_warning`` hook is registered on PostToolUse.

Reuses the sibling test modules' ``_stub_env`` / ``_TASK`` fakes.
"""

from __future__ import annotations

import pytest

from tests import test_claude_sdk_otf_v1_agent as otf_t
from tests import test_claude_sdk_otf_ainteract_v1_agent as ainteract_t
from tests import test_claude_sdk_otf_ainteract_raw_v1_agent as ainteract_raw_t
from tests import test_claude_sdk_otf_raw_v1_agent as raw_t


_ASK = "mcp__bird-interact-tools__ask_user"
_KB_NATIVES = {
    "mcp__bird-interact-tools__get_all_external_knowledge_names",
    "mcp__bird-interact-tools__get_knowledge_definition",
    "mcp__bird-interact-tools__get_all_knowledge_definitions",
}
_SLAYER_DISCOVERY_MCP = {
    "mcp__slayer__search",
    "mcp__slayer__models_summary",
    "mcp__slayer__inspect_model",
    "mcp__slayer__list_datasources",
}
_SLAYER_MAIN_MCP = {
    "mcp__slayer__help",
    "mcp__slayer__create_model",
    "mcp__slayer__edit_model",
    "mcp__slayer__validate_models",
    "mcp__slayer__save_memory",
}
_SLAYER_MAIN_NATIVES = {
    # DEV-1555 CR r1 unification: `query_nested` is gone; the unified
    # `query` tool accepts a single SlayerQuery object OR a list of
    # stage objects.
    "mcp__bird-interact-tools__query",
    "mcp__bird-interact-tools__submit_query",
}
_RAW_DISCOVERY = {
    "mcp__bird-interact-tools__get_schema",
    "mcp__bird-interact-tools__get_all_column_meanings",
    "mcp__bird-interact-tools__get_column_meaning",
    "mcp__bird-interact-tools__get_all_external_knowledge_names",
    "mcp__bird-interact-tools__get_knowledge_definition",
    "mcp__bird-interact-tools__get_all_knowledge_definitions",
    "mcp__bird-interact-tools__execute_sql",
}
_RAW_MAIN = {
    "Task",
    "mcp__bird-interact-tools__execute_sql",
    "mcp__bird-interact-tools__submit_sql",
}


def _agent_module(name):
    import importlib

    return importlib.import_module(f"bird_interact_agents.agents.{name}.agent")


# ---------------------------------------------------------------------------
# Partition constants per module
# ---------------------------------------------------------------------------

def test_otf_partition_constants():
    m = _agent_module("claude_sdk_otf_v1")
    assert set(m.DISCOVERY_TOOLS) == _SLAYER_DISCOVERY_MCP | _KB_NATIVES
    assert set(m.MAIN_TOOLS) == (
        {"Task"} | _SLAYER_MAIN_MCP | _KB_NATIVES | _SLAYER_MAIN_NATIVES
    )


def test_ainteract_partition_constants():
    m = _agent_module("claude_sdk_otf_ainteract_v1")
    assert set(m.DISCOVERY_TOOLS) == _SLAYER_DISCOVERY_MCP | _KB_NATIVES | {_ASK}
    assert set(m.MAIN_TOOLS) == (
        {"Task", _ASK} | _SLAYER_MAIN_MCP | _KB_NATIVES | _SLAYER_MAIN_NATIVES
    )


def test_raw_partition_constants():
    m = _agent_module("claude_sdk_otf_raw_v1")
    assert set(m.DISCOVERY_TOOLS) == _RAW_DISCOVERY
    assert set(m.MAIN_TOOLS) == _RAW_MAIN


def test_ainteract_raw_partition_constants():
    m = _agent_module("claude_sdk_otf_ainteract_raw_v1")
    assert set(m.DISCOVERY_TOOLS) == _RAW_DISCOVERY | {_ASK}
    assert set(m.MAIN_TOOLS) == _RAW_MAIN | {_ASK}


@pytest.mark.parametrize(
    "name",
    [
        "claude_sdk_otf_v1",
        "claude_sdk_otf_ainteract_v1",
        "claude_sdk_otf_raw_v1",
        "claude_sdk_otf_ainteract_raw_v1",
    ],
)
def test_task_never_leaks_into_discovery(name):
    """Subagents cannot spawn subagents — Task must not be granted to
    the discovery agent."""
    m = _agent_module(name)
    assert "Task" not in m.DISCOVERY_TOOLS
    assert "Task" in m.MAIN_TOOLS


# ---------------------------------------------------------------------------
# Options wiring per agent (captured via the sibling fakes)
# ---------------------------------------------------------------------------

async def _run_and_capture(monkeypatch, tmp_path, *, sibling, module_name,
                           agent_cls_name, query_mode, eval_mode):
    m = _agent_module(module_name)
    captured = sibling._stub_env(monkeypatch, m, tmp_path / "store")
    agent = getattr(m, agent_cls_name)(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode,
        eval_mode=eval_mode,
    )
    return m, captured["options"]


_CASES = [
    pytest.param(
        otf_t, "claude_sdk_otf_v1", "ClaudeSDKOtfAgent", "slayer", "one-shot",
        id="otf",
    ),
    pytest.param(
        ainteract_t, "claude_sdk_otf_ainteract_v1", "ClaudeSDKOtfAInteractAgent",
        "slayer", "a-interact",
        id="ainteract",
    ),
    pytest.param(
        raw_t, "claude_sdk_otf_raw_v1", "ClaudeSDKOtfRawAgent", "raw", "one-shot",
        id="raw",
    ),
    pytest.param(
        ainteract_raw_t, "claude_sdk_otf_ainteract_raw_v1",
        "ClaudeSDKOtfAInteractRawAgent", "raw", "a-interact",
        id="ainteract_raw",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_options_tools_is_exactly_task(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    _, options = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode,
    )
    assert options.tools == ["Task"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_options_declare_discovery_agent_definition(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    from bird_interact_agents.agents.claude_sdk.partition import (
        DISCOVERY_AGENT_NAME,
        DISCOVERY_MAX_TURNS,
    )

    m, options = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode,
    )
    assert set(options.agents.keys()) == {DISCOVERY_AGENT_NAME}
    ad = options.agents[DISCOVERY_AGENT_NAME]
    assert set(ad.tools) == set(m.DISCOVERY_TOOLS)
    assert len(ad.tools) == len(set(ad.tools))  # no duplicates
    assert ad.model is None  # inherit the main session model
    assert ad.maxTurns == DISCOVERY_MAX_TURNS
    assert ad.prompt  # non-empty handoff prompt (content untested by design)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_allowed_tools_is_union_of_both_partitions(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    m, options = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode,
    )
    assert set(options.allowed_tools) == set(m.MAIN_TOOLS) | set(m.DISCOVERY_TOOLS)
    # The enforced partition must NOT come from a global disallow — that
    # would block discovery's own tools inside the subagent.
    assert not (set(options.disallowed_tools or []) & set(m.DISCOVERY_TOOLS))


def _hook_names(options, event):
    return [
        getattr(cb, "__name__", "?")
        for matcher in (options.hooks or {}).get(event, [])
        for cb in matcher.hooks
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_partition_deny_hook_registered_with_exact_matcher(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    m, options = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode,
    )
    discovery_only = set(m.DISCOVERY_TOOLS) - set(m.MAIN_TOOLS)
    assert discovery_only, "partition must have main-blocked tools"
    matchers = {
        matcher.matcher: list(matcher.hooks)
        for matcher in (options.hooks or {}).get("PreToolUse", [])
    }
    expected_matcher = "|".join(sorted(discovery_only))
    assert expected_matcher in matchers
    deny_cbs = [
        cb for cb in matchers[expected_matcher]
        if getattr(cb, "__name__", "?") == "partition_deny"
    ]
    assert deny_cbs
    # Behavioral check (Codex test-review #4): the REGISTERED callback must
    # close over this module's discovery-only set — deny a main-loop call,
    # allow the same tool inside the subagent.
    some_tool = sorted(discovery_only)[0]
    denied = await deny_cbs[0](
        {"tool_name": some_tool, "tool_input": {}}, "tu", None,
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    allowed = await deny_cbs[0](
        {"tool_name": some_tool, "tool_input": {}, "agent_id": "a1"}, "tu", None,
    )
    assert allowed == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_context_budget_hook_registered_on_post_tool_use(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    _, options = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode,
    )
    assert "context_budget_warning" in _hook_names(options, "PostToolUse")


class _StreamedAssistant:
    """Fake streamed message; type name must read AssistantMessage."""

    def __init__(self, context_tokens: int):
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 1,
            "cache_read_input_tokens": context_tokens,
            "cache_creation_input_tokens": 0,
        }


_StreamedAssistant.__name__ = "AssistantMessage"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_run_loop_feeds_streamed_usage_into_context_hook(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """Codex test-review #1: registering the hook is not enough — the run
    loop must update the shared state from each streamed AssistantMessage,
    and the registered hook must observe it. Shrink the window so the
    streamed message crosses 80%, then fire the captured hook."""
    m = _agent_module(module_name)
    monkeypatch.setattr(m, "context_window_for", lambda model: 1_000)

    captured = sibling._stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_StreamedAssistant(context_tokens=900)],
    )
    agent = getattr(m, agent_cls_name)(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode,
        eval_mode=eval_mode,
    )
    options = captured["options"]
    hooks = [
        cb
        for matcher in (options.hooks or {}).get("PostToolUse", [])
        for cb in matcher.hooks
        if getattr(cb, "__name__", "?") == "context_budget_warning"
    ]
    assert hooks
    out = await hooks[0]({"tool_name": "x"}, "tu", None)
    assert out["hookSpecificOutput"]["additionalContext"]
