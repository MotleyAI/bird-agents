"""Invariants for the LiteLLM <-> PydanticAI model-string helpers."""

from bird_interact_agents.model_string import (
    is_anthropic,
    native_model_id,
    to_pydantic_ai,
)


def test_is_anthropic_true_for_anthropic_prefix():
    assert is_anthropic("anthropic/claude-sonnet-4-5") is True
    assert is_anthropic("anthropic/claude-haiku-4-5-20251001") is True


def test_is_anthropic_false_for_other_providers():
    assert is_anthropic("cerebras/zai-glm-4.7") is False
    assert is_anthropic("openrouter/z-ai/glm-4.7-flash") is False
    assert is_anthropic("fireworks_ai/glm-4p7") is False
    # Bare ids (legacy, no prefix) — definitely not anthropic
    assert is_anthropic("claude-sonnet-4-5") is False


def test_to_pydantic_ai_simple_provider():
    assert to_pydantic_ai("cerebras/zai-glm-4.7") == "cerebras:zai-glm-4.7"
    assert to_pydantic_ai("anthropic/claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"


def test_to_pydantic_ai_only_first_slash_for_openrouter():
    """OpenRouter model ids contain a slash (`z-ai/glm-4.7-flash`); the
    swap must only touch the first slash so the model id stays intact."""
    assert (
        to_pydantic_ai("openrouter/z-ai/glm-4.7-flash")
        == "openrouter:z-ai/glm-4.7-flash"
    )


def test_to_pydantic_ai_passes_through_unprefixed():
    """Bare ids without a slash are returned unchanged so PydanticAI's own
    inference can pick a provider for them."""
    assert to_pydantic_ai("claude-sonnet-4-5") == "claude-sonnet-4-5"


def test_to_pydantic_ai_idempotent_for_colonized_openrouter():
    """Already-colonized OpenRouter ids must not be mangled into
    `openrouter:z-ai:glm-4.7-flash` — that broke PydanticAI lookups when
    callers pre-converted the string before handing it to the adapter."""
    assert (
        to_pydantic_ai("openrouter:z-ai/glm-4.7-flash")
        == "openrouter:z-ai/glm-4.7-flash"
    )


def test_to_pydantic_ai_idempotent_for_colonized_simple():
    """Colon form without a slash also survives a redundant conversion."""
    assert to_pydantic_ai("anthropic:claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"


def test_native_model_id_strips_prefix():
    assert native_model_id("anthropic/claude-sonnet-4-5") == "claude-sonnet-4-5"
    assert native_model_id("cerebras/zai-glm-4.7") == "zai-glm-4.7"
    # Multiple slashes — only the first is the provider boundary
    assert native_model_id("openrouter/z-ai/glm-4.7-flash") == "z-ai/glm-4.7-flash"
    # Bare id, no prefix -> unchanged
    assert native_model_id("claude-sonnet-4-5") == "claude-sonnet-4-5"


def test_build_pydantic_ai_model_returns_string_for_native_providers():
    """Cerebras / Anthropic / OpenRouter are pydantic_ai-native — a string
    is enough."""
    from bird_interact_agents.model_string import build_pydantic_ai_model

    assert build_pydantic_ai_model("cerebras/zai-glm-4.7") == "cerebras:zai-glm-4.7"
    assert (
        build_pydantic_ai_model("anthropic/claude-sonnet-4-5")
        == "anthropic:claude-sonnet-4-5"
    )
    assert (
        build_pydantic_ai_model("openrouter/z-ai/glm-4.7-flash")
        == "openrouter:z-ai/glm-4.7-flash"
    )


def test_build_pydantic_ai_model_returns_openai_compat_for_deepinfra(monkeypatch):
    """DeepInfra has no native pydantic_ai provider — wrap as
    OpenAIChatModel pointing at DeepInfra's OpenAI-compatible endpoint."""
    from bird_interact_agents.model_string import build_pydantic_ai_model

    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    m = build_pydantic_ai_model("deepinfra/moonshotai/Kimi-K2-Instruct")

    # The bare model name (no provider prefix) is what DeepInfra expects
    # on its OpenAI-compatible endpoint.
    assert m.model_name == "moonshotai/Kimi-K2-Instruct"
    base = str(m.client.base_url)
    assert "deepinfra.com" in base


def test_build_pydantic_ai_model_deepinfra_missing_key_errors(monkeypatch):
    from bird_interact_agents.model_string import build_pydantic_ai_model

    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    import pytest
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        build_pydantic_ai_model("deepinfra/moonshotai/Kimi-K2-Instruct")


def test_build_pydantic_ai_model_returns_openai_compat_for_doubleword(monkeypatch):
    """Doubleword.ai is OpenAI-compatible — wrap as OpenAIChatModel
    pointing at doubleword's endpoint."""
    from bird_interact_agents.model_string import build_pydantic_ai_model

    monkeypatch.setenv("DOUBLEWORD_API_KEY", "test-key")
    m = build_pydantic_ai_model("doubleword/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")

    assert m.model_name == "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
    base = str(m.client.base_url)
    assert "doubleword.ai" in base


def test_build_pydantic_ai_model_doubleword_missing_key_errors(monkeypatch):
    from bird_interact_agents.model_string import build_pydantic_ai_model

    monkeypatch.delenv("DOUBLEWORD_API_KEY", raising=False)
    import pytest
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        build_pydantic_ai_model("doubleword/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")


def test_build_pydantic_ai_model_idempotent_for_colonized_input():
    """Already-colonized inputs must pass straight through. Without the
    `:` guard, `openrouter:z-ai/glm-4.7-flash` would partition into
    provider=`openrouter:z-ai` and rest=`glm-4.7-flash` and re-emit as
    `openrouter:z-ai:glm-4.7-flash`, which PydanticAI can't resolve.
    """
    from bird_interact_agents.model_string import build_pydantic_ai_model

    assert (
        build_pydantic_ai_model("openrouter:z-ai/glm-4.7-flash")
        == "openrouter:z-ai/glm-4.7-flash"
    )
    assert (
        build_pydantic_ai_model("anthropic:claude-sonnet-4-5")
        == "anthropic:claude-sonnet-4-5"
    )
