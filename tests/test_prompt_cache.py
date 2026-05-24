"""Invariants for Anthropic prompt-cache plumbing on the pydantic-ai agent.

The actual cache hit/miss behavior is observable only end-to-end with a live
Anthropic API call (`cache_read_input_tokens` on the usage record). These
tests cover the plumbing: which models get cache settings attached, which
don't, and that the on/off flag controls it.
"""


def test_anthropic_cache_settings_has_instructions_and_tools():
    from bird_interact_agents.agents.pydantic_ai.agent import (
        _anthropic_cache_settings,
    )

    s = _anthropic_cache_settings()
    assert s is not None
    # AnthropicModelSettings is a TypedDict subclass; values are kept by key.
    assert s.get("anthropic_cache_instructions") is True
    assert s.get("anthropic_cache_tool_definitions") is True
    # We intentionally don't cache messages — agent's tool-call history
    # mutates per turn and isn't a high-value cache target.
    assert "anthropic_cache_messages" not in s


def test_pydantic_ai_agent_attaches_cache_settings_for_anthropic_default_on():
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    a = PydanticAIAgent(model="anthropic/claude-haiku-4-5-20251001")
    # prompt_cache defaults to True; model is Anthropic — settings attached.
    assert a._model_settings is not None
    assert a._model_settings.get("anthropic_cache_instructions") is True


def test_pydantic_ai_agent_no_cache_settings_when_disabled():
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    a = PydanticAIAgent(
        model="anthropic/claude-haiku-4-5-20251001",
        prompt_cache=False,
    )
    assert a._model_settings is None


def test_pydantic_ai_agent_no_cache_settings_for_non_anthropic(monkeypatch):
    """Non-Anthropic providers don't get AnthropicModelSettings even when
    prompt_cache=True — the keys would be no-ops and risk warnings from
    the underlying client."""
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    a = PydanticAIAgent(model="cerebras/zai-glm-4.7")
    assert a._model_settings is None

    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    a2 = PydanticAIAgent(model="doubleword/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    assert a2._model_settings is None


def test_pydantic_ai_agent_doubleword_model_built(monkeypatch):
    """Doubleword routing: model is an OpenAIChatModel, not a litellm string."""
    # Use monkeypatch so the env var is auto-removed at the end of this
    # test; the previous direct-mutation form leaked DOUBLEWORD_API_KEY
    # into the rest of the process and could shadow real config in
    # subsequent tests.
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key-for-unit-tests")
    from bird_interact_agents.agents.pydantic_ai.agent import PydanticAIAgent

    a = PydanticAIAgent(model="doubleword/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")
    # build_pydantic_ai_model returns an OpenAIChatModel instance for
    # OpenAI-compatible providers; native providers return a string.
    assert not isinstance(a.model, str)
    assert a.model_id == "doubleword/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
