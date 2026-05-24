"""Per-framework verification that --strict actually applies strict=True
to every outgoing tool definition.

For each framework we either:
- intercept the LLM-call boundary (litellm.completion / openai client) and
  inspect the captured `tools` payload,
- run the framework-native strict-rewrite hook standalone and assert the
  outputs uniformly carry strict=True,
- or assert SystemExit when the framework can't honour --strict.

No real network calls — every test stubs the model boundary.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.config import settings


# ---------------------------------------------------------------------------
# pydantic_ai — prepare_tools callback rewrites every ToolDefinition.strict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pydantic_ai_prepare_tools_forces_strict_true():
    """The prepare_tools callback returned by _make_prepare_tools(True)
    rewrites every ToolDefinition.strict, regardless of its prior value."""
    from pydantic_ai.tools import ToolDefinition

    from bird_interact_agents.agents.pydantic_ai.agent import _make_prepare_tools

    cb = _make_prepare_tools(True)
    defs = [
        ToolDefinition(name="a", parameters_json_schema={}, strict=None),
        ToolDefinition(name="b", parameters_json_schema={}, strict=False),
        ToolDefinition(name="c", parameters_json_schema={}, strict=True),
    ]
    out = await cb(None, defs)  # ctx unused
    assert all(t.strict is True for t in out)
    assert {t.name for t in out} == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_pydantic_ai_prepare_tools_forces_strict_false():
    """Default --no-strict path: every ToolDefinition.strict ends up False."""
    from pydantic_ai.tools import ToolDefinition

    from bird_interact_agents.agents.pydantic_ai.agent import _make_prepare_tools

    cb = _make_prepare_tools(False)
    defs = [ToolDefinition(name=n, parameters_json_schema={}, strict=True)
            for n in ("a", "b", "c")]
    out = await cb(None, defs)
    assert all(t.strict is False for t in out)


def test_pydantic_ai_each_factory_wires_prepare_tools(tmp_path):
    """All three pydantic_ai agent factories register the strict-forcing hook."""
    import os

    os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key")
    from bird_interact_agents.agents.pydantic_ai.agent import (
        _build_raw_a_agent, _build_raw_c_agent, _build_slayer_agent,
    )
    builds = (
        lambda: _build_raw_a_agent("anthropic:claude-sonnet-4-5", strict_value=True),
        lambda: _build_raw_c_agent("anthropic:claude-sonnet-4-5", strict_value=True),
        lambda: _build_slayer_agent(
            "anthropic:claude-sonnet-4-5",
            slayer_storage_dir=str(tmp_path),
            strict_value=True,
        ),
    )
    for build in builds:
        agent = build()
        # PydanticAI stores the hook on the Agent. The internal attribute
        # name varies between minor versions; tolerate both forms.
        hook = getattr(agent, "_prepare_tools", None) or getattr(
            agent, "prepare_tools", None
        )
        assert hook is not None, "prepare_tools hook missing"


# ---------------------------------------------------------------------------
# smolagents — _StrictLiteLLMModel mutates the tools dict before litellm call
# ---------------------------------------------------------------------------

def test_smolagents_strict_litellm_model_injects_strict_into_tools(monkeypatch):
    """The subclass overrides _prepare_completion_kwargs and adds
    strict=True to every entry's `function` dict."""
    import os

    os.environ.setdefault("CEREBRAS_API_KEY", "fake-key")
    from smolagents import LiteLLMModel

    from bird_interact_agents.agents.smolagents.agent import (
        _build_strict_litellm_model_class,
    )

    Strict = _build_strict_litellm_model_class()
    m = Strict(model_id="cerebras/zai-glm-4.7", _strict_value=True)

    fake_kwargs = {
        "tools": [
            {"type": "function", "function": {"name": "x", "parameters": {}}},
            {"type": "function", "function": {"name": "y", "parameters": {}}},
        ]
    }
    # Force the parent's _prepare_completion_kwargs to return our payload so
    # the override (which calls super()) actually runs against it.
    monkeypatch.setattr(
        LiteLLMModel, "_prepare_completion_kwargs",
        lambda self, *a, **kw: fake_kwargs,
    )
    out = m._prepare_completion_kwargs()
    assert all(t["function"]["strict"] is True for t in out["tools"])


# ---------------------------------------------------------------------------
# agno — _StrictLiteLLM.get_request_params sets strict on every tool
# ---------------------------------------------------------------------------

def test_agno_strict_litellm_subclass_injects_strict_into_request_params(monkeypatch):
    """When the agno path uses the production _StrictLiteLLM, every tool dict
    the request_params produces carries strict=True for the configured value."""
    import os

    os.environ.setdefault("CEREBRAS_API_KEY", "fake-key")
    from agno.models.litellm import LiteLLM

    from bird_interact_agents.agents.agno.agent import _build_strict_litellm_class

    fake_tools = [
        {"type": "function", "function": {"name": "x", "parameters": {}}},
        {"type": "function", "function": {"name": "y", "parameters": {}}},
    ]
    # Stub the parent so super().get_request_params returns our payload —
    # the production override should then mutate every tool with strict.
    monkeypatch.setattr(
        LiteLLM, "get_request_params",
        lambda self, tools=None: {"tools": list(fake_tools)},
    )
    StrictLiteLLM = _build_strict_litellm_class()
    m = StrictLiteLLM(id="cerebras/zai-glm-4.7", _strict_value=True)
    params = m.get_request_params(tools=fake_tools)
    assert all(t["function"]["strict"] is True for t in params["tools"])


# ---------------------------------------------------------------------------
# mcp_agent — must SystemExit clearly when --strict is requested
# ---------------------------------------------------------------------------

def test_mcp_agent_strict_true_exits_with_clear_error():
    """mcp_agent's OpenAIAugmentedLLM has no strict knob (upstream TODO).
    Constructing the agent with strict=True must SystemExit before any
    LLM call rather than silently produce a non-strict request."""
    from bird_interact_agents.agents.mcp_agent.agent import McpAgentAgent

    with pytest.raises(SystemExit) as ex:
        McpAgentAgent(model="cerebras/zai-glm-4.7", strict=True)
    msg = str(ex.value)
    assert "mcp_agent" in msg
    assert "--strict" in msg


def test_mcp_agent_strict_false_constructs_normally():
    """Default path (strict=False) does not raise — confirms our gate is
    only triggered by the True branch."""
    from bird_interact_agents.agents.mcp_agent.agent import McpAgentAgent

    agent = McpAgentAgent(model="cerebras/zai-glm-4.7", strict=False)
    assert agent.strict is False


# ---------------------------------------------------------------------------
# claude_sdk — strict flag is a documented no-op (Anthropic has no strict)
# ---------------------------------------------------------------------------

def test_claude_sdk_constructs_with_strict_flag_as_noop():
    """claude_sdk doesn't expose a strict CLI knob on the agent class — it's
    handled at the run.py boundary (logged warning, then ignored). Verify
    the agent constructs normally with our default Anthropic model so we
    don't accidentally regress that path."""
    from bird_interact_agents.agents.claude_sdk.agent import ClaudeSDKAgent

    agent = ClaudeSDKAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.model == "anthropic/claude-sonnet-4-5"
