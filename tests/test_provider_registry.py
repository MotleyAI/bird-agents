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
    """Codex r2: the env dict neutralises ambient Anthropic API-key /
    OAuth-token vars by setting them to empty strings, then sets the
    provider's Bearer token via ANTHROPIC_AUTH_TOKEN."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    env = pr.sdk_session_env(_KIMI)
    assert env == {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "https://api.moonshot.ai/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "ms-key-1",
    }


def test_sdk_session_env_neutralises_ambient_anthropic_creds(monkeypatch):
    """Codex r2: an ambient ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN
    in the developer's shell would otherwise survive into the SDK
    subprocess (it inherits the parent env by default) and the SDK
    would silently authenticate against Anthropic instead of the
    provider's ANTHROPIC_BASE_URL endpoint. The session env emits
    empty-string overrides so neither variable carries through."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "leaked-bearer")
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")

    env = pr.sdk_session_env(_KIMI)
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    # The provider's Bearer wins for ANTHROPIC_AUTH_TOKEN even though an
    # ambient value was set — dict ordering puts our override last.
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ms-key-1"


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


def test_litellm_route_prefers_caller_api_key(monkeypatch):
    """CR r1: a caller-supplied ``api_key`` short-circuits the env-var
    lookup. Without this, threading ``api_key=`` through usage.py still
    hit ``provider_api_key(spec)`` (which raises on a missing env var)
    before ``kwargs.setdefault`` could preserve the caller's value."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    litellm_model, kwargs = pr.litellm_route(
        _KIMI, caller_api_key="explicit-key",
    )
    assert litellm_model == "openai/kimi-k2.7-code"
    assert kwargs["api_key"] == "explicit-key"


def test_litellm_route_falls_back_to_env_when_caller_unset(monkeypatch):
    """Belt: ``caller_api_key=None`` keeps the existing env-var path."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-env")
    _, kwargs = pr.litellm_route(_KIMI, caller_api_key=None)
    assert kwargs["api_key"] == "ms-key-env"


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


# ---------------------------------------------------------------------------
# DEV-1580: z.ai (Zhipu GLM) as the SECOND registry provider. Mirrors every
# Moonshot contract above. The GLM family is reachable via z.ai's
# Anthropic-compatible endpoint (https://api.z.ai/api/anthropic) with
# ZAI_API_KEY; unlike kimi-k2.7-code, NO GLM model requires `thinking`.
# ---------------------------------------------------------------------------

_GLM = "zai/glm-5.2"

# (model id, input $/M, cache-hit $/M or None, output $/M) per z.ai pricing
# docs (2026-06). GLM-5.x share a row; GLM-4.7/4.6 are the cheaper tier.
_ZAI_PRICED = [
    ("glm-5.2", 1.40, 0.26, 4.40),
    ("glm-5.1", 1.40, 0.26, 4.40),
    ("glm-4.7", 0.60, 0.11, 2.20),
    ("glm-4.6", 0.60, None, 2.20),
]


def test_zai_entry_fields():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    spec = pr.get_provider(_GLM)
    assert spec is not None
    assert spec.key == "zai"
    assert spec.api_format == "anthropic"
    assert spec.auth_env == "ZAI_API_KEY"
    assert spec.base_url_env == "BIRD_ZAI_ANTHROPIC_BASE_URL"
    assert spec.base_url == "https://api.z.ai/api/anthropic"
    assert spec.openai_base_url == "https://api.z.ai/api/paas/v4"
    # GLM models default to 200K; only glm-5.2 overrides up to its 1M window.
    assert spec.default_context_window == 200_000
    assert spec.model_context_windows == {"glm-5.2": 1_000_000}


def test_zai_litellm_pricing_registered():
    """Official z.ai prices (per 1M tokens). Registered under BOTH the
    canonical `zai/...` string (agent-side cost rows) and the rewritten
    `openai/...` string (user-sim litellm route) so neither path falls
    back to the $0 unpriced warning."""
    import litellm

    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    pr.ensure_litellm_pricing()
    for native_id, in_price, _cache, out_price in _ZAI_PRICED:
        for key in (f"zai/{native_id}", f"openai/{native_id}"):
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=key, prompt_tokens=1_000_000, completion_tokens=1_000_000,
            )
            assert prompt_cost == pytest.approx(in_price, rel=1e-6)
            assert completion_cost == pytest.approx(out_price, rel=1e-6)


def test_zai_pricing_uses_distinct_instances():
    """Codex r1: glm-5.1 / glm-5.2 carry identical prices but MUST be
    distinct ModelPricing objects — a shared instance would let a future
    edit to one silently mutate the other (pydantic models are mutable)."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    pricing = pr.get_provider(_GLM).model_pricing
    assert pricing["glm-5.1"] is not pricing["glm-5.2"]
    assert pricing["glm-5.1"] == pricing["glm-5.2"]  # same values, different obj


def test_zai_is_supported_agent_model():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert pr.is_supported_agent_model(_GLM)
    assert pr.is_supported_agent_model("zai/glm-4.7")


def test_zai_resolve_base_url_default_and_override(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    spec = pr.get_provider(_GLM)
    monkeypatch.delenv("BIRD_ZAI_ANTHROPIC_BASE_URL", raising=False)
    assert pr.resolve_base_url(spec) == "https://api.z.ai/api/anthropic"
    monkeypatch.setenv(
        "BIRD_ZAI_ANTHROPIC_BASE_URL", "https://other.example/anthropic",
    )
    assert pr.resolve_base_url(spec) == "https://other.example/anthropic"


def test_zai_sdk_session_env_exact_keys(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.delenv("BIRD_ZAI_ANTHROPIC_BASE_URL", raising=False)
    env = pr.sdk_session_env(_GLM)
    assert env == {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "zai-key-1",
    }


def test_zai_sdk_session_env_neutralises_ambient_anthropic_creds(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "leaked-bearer")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    env = pr.sdk_session_env(_GLM)
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    assert env["ANTHROPIC_AUTH_TOKEN"] == "zai-key-1"


def test_zai_sdk_session_env_respects_base_url_override(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.setenv(
        "BIRD_ZAI_ANTHROPIC_BASE_URL", "https://other.example/anthropic",
    )
    env = pr.sdk_session_env(_GLM)
    assert env["ANTHROPIC_BASE_URL"] == "https://other.example/anthropic"


def test_zai_sdk_session_env_missing_key_raises(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ZAI_API_KEY"):
        pr.sdk_session_env(_GLM)


def test_zai_litellm_route(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    litellm_model, kwargs = pr.litellm_route(_GLM)
    assert litellm_model == "openai/glm-5.2"
    assert kwargs == {
        "api_base": "https://api.z.ai/api/paas/v4",
        "api_key": "zai-key-1",
    }


def test_zai_litellm_route_prefers_caller_api_key(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    _, kwargs = pr.litellm_route(_GLM, caller_api_key="explicit-key")
    assert kwargs["api_key"] == "explicit-key"


def test_zai_required_env_for():
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    assert pr.required_env_for(_GLM) == ("ZAI_API_KEY",)
    assert pr.required_env_for("zai/glm-4.6") == ("ZAI_API_KEY",)


def test_zai_requires_thinking_is_false_for_all_glm():
    """Unlike kimi-k2.7-code, no GLM model requires thinking on the
    Anthropic /v1/messages endpoint — z.ai treats it as opt-in."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415
    for native_id, *_ in _ZAI_PRICED:
        assert pr.requires_thinking(f"zai/{native_id}") is False
