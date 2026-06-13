"""DEV-1555 Stage 2: open-weight provider registry.

Single source of truth for open-weight backends usable by the
claude_sdk_otf* agents: endpoint(s), auth env var, api format, context
windows. First provider: Moonshot (Kimi K2.7 Code) via its
Anthropic-compatible endpoint.
"""

from __future__ import annotations

import pytest


_KIMI = "moonshot/kimi-k2.7-code"


def test_provider_spec_is_pydantic_model():
    from pydantic import BaseModel

    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert issubclass(pr.ProviderSpec, BaseModel)


def test_moonshot_entry_fields():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    spec = pr.get_provider(_KIMI)
    assert spec is not None
    assert spec.key == "moonshot"
    assert spec.api_format == "anthropic"
    assert spec.auth_env == "MOONSHOT_API_KEY"
    # Official pricing page: kimi-k2.7-code context window = 262,144 tokens.
    assert spec.default_context_window == 262_144
    assert spec.base_url == "https://api.moonshot.ai/anthropic"
    assert spec.openai_base_url == "https://api.moonshot.ai/v1"


def test_moonshot_litellm_pricing_registered():
    """Official prices (per 1M tokens): input $0.95 miss / $0.19 cache hit,
    output $4.00. Registered under BOTH the canonical `moonshot/...` string
    (agent-side cost rows) and the rewritten `openai/...` string (user-sim
    litellm route) so neither path falls back to the $0 unpriced warning."""
    import litellm

    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    pr.ensure_litellm_pricing()
    for key in ("moonshot/kimi-k2.7-code", "openai/kimi-k2.7-code"):
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=key, prompt_tokens=1_000_000, completion_tokens=1_000_000,
        )
        assert prompt_cost == pytest.approx(0.95, rel=1e-6)
        assert completion_cost == pytest.approx(4.00, rel=1e-6)


def test_ensure_litellm_pricing_idempotent():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    pr.ensure_litellm_pricing()
    pr.ensure_litellm_pricing()  # second call must not raise or duplicate


def test_get_provider_non_registry_models():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert pr.get_provider("anthropic/claude-sonnet-4-6") is None
    assert pr.get_provider("unknownprov/some-model") is None
    assert pr.get_provider("bare-model-id") is None


def test_is_supported_agent_model():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert pr.is_supported_agent_model("anthropic/claude-sonnet-4-6")
    assert pr.is_supported_agent_model(_KIMI)
    assert not pr.is_supported_agent_model("cerebras/zai-glm-4.7")
    assert not pr.is_supported_agent_model("openai/gpt-4o")
    assert not pr.is_supported_agent_model("bare-model-id")


def test_resolve_base_url_default_and_override(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    spec = pr.get_provider(_KIMI)
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    assert pr.resolve_base_url(spec) == "https://api.moonshot.ai/anthropic"
    monkeypatch.setenv(
        "BIRD_MOONSHOT_ANTHROPIC_BASE_URL", "https://other.example/anthropic",
    )
    assert pr.resolve_base_url(spec) == "https://other.example/anthropic"


def test_sdk_session_env_exact_keys(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    env = pr.sdk_session_env(_KIMI)
    assert env == {
        "ANTHROPIC_BASE_URL": "https://api.moonshot.ai/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "ms-key-1",
    }


def test_sdk_session_env_respects_base_url_override(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.setenv(
        "BIRD_MOONSHOT_ANTHROPIC_BASE_URL", "https://other.example/anthropic",
    )
    env = pr.sdk_session_env(_KIMI)
    assert env["ANTHROPIC_BASE_URL"] == "https://other.example/anthropic"


def test_sdk_session_env_missing_key_raises(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        pr.sdk_session_env(_KIMI)


def test_litellm_route_moonshot(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    litellm_model, kwargs = pr.litellm_route(_KIMI)
    assert litellm_model == "openai/kimi-k2.7-code"
    assert kwargs == {
        "api_base": "https://api.moonshot.ai/v1",
        "api_key": "ms-key-1",
    }


def test_required_env_for():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert pr.required_env_for(_KIMI) == ("MOONSHOT_API_KEY",)
    assert pr.required_env_for("anthropic/claude-sonnet-4-6") == ()
    assert pr.required_env_for("unknownprov/x") == ()


def test_requires_thinking():
    """Probed live (2026-06-12): kimi-k2.7-code rejects any /v1/messages
    request without thinking={"type":"enabled",...} — the flag drives the
    SDK session options and the autopsy request shape."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert pr.requires_thinking(_KIMI) is True
    assert pr.requires_thinking("moonshot/kimi-k2.6") is False
    assert pr.requires_thinking("anthropic/claude-sonnet-4-6") is False
    assert pr.requires_thinking("unknownprov/x") is False
