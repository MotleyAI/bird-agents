"""DEV-1639: Doubleword via its NATIVE Anthropic /v1/messages endpoint.

Doubleword now exposes an Anthropic-compatible endpoint with native prompt
caching, so `claude_sdk*` agents reach it directly (no DEV-1604 bridge) and it
is priced from its published realtime GLM-5.2-FP8 numbers. Scope is claude_sdk*
only — the pydantic_ai Doubleword path stays OpenAI-format (pinned here as a
scope guard).
"""

from __future__ import annotations

import pytest

from bird_interact_agents import provider_registry as pr
from bird_interact_agents import usage


_DW = "doubleword/zai-org/GLM-5.2-FP8"
_DW_NATIVE = "zai-org/GLM-5.2-FP8"


# ---------------------------------------------------------------------------
# Direct-endpoint routing (no bridge)
# ---------------------------------------------------------------------------


def test_doubleword_is_supported_and_never_bridges():
    assert pr.is_supported_agent_model(_DW)
    assert pr.agent_needs_bridge(_DW, True) is False
    assert pr.agent_needs_bridge(_DW, False) is False


def test_doubleword_resolve_base_url_direct(monkeypatch):
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    spec = pr.get_provider(_DW)
    assert spec.api_format == "anthropic"
    assert pr.resolve_base_url(spec) == "https://api.doubleword.ai"


def test_doubleword_sdk_session_env_direct_bearer(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    env = pr.sdk_session_env(_DW)
    assert env == {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "https://api.doubleword.ai",
        "ANTHROPIC_AUTH_TOKEN": "dw-key-1",
    }


def test_doubleword_override_still_wins(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    assert pr.sdk_session_env(_DW)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9"


# ---------------------------------------------------------------------------
# Pricing (realtime GLM-5.2-FP8) + litellm registration
# ---------------------------------------------------------------------------


def test_doubleword_pricing_values():
    p = pr.get_provider(_DW).model_pricing[_DW_NATIVE]
    assert p.input_cost_per_token == 1.40e-6
    assert p.output_cost_per_token == 4.40e-6
    assert p.cache_read_input_token_cost == 0.14e-6
    assert p.cache_write_5m_multiplier == 1.25
    assert p.cache_write_1h_multiplier == 2.0


def test_doubleword_context_window():
    assert pr.get_provider(_DW).default_context_window == 1_048_576


def test_ensure_litellm_pricing_registers_both_keys():
    import litellm

    pr.ensure_litellm_pricing()
    for key in ("doubleword/zai-org/GLM-5.2-FP8", "openai/zai-org/GLM-5.2-FP8"):
        assert key in litellm.model_cost
        assert litellm.model_cost[key]["input_cost_per_token"] == 1.40e-6
        assert litellm.model_cost[key]["output_cost_per_token"] == 4.40e-6


def test_doubleword_safe_cost_input_output_and_read(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "5m")
    # No cache-creation → base_input == prompt_tokens.
    p, c = usage._safe_cost(
        model=_DW, prompt_tokens=100, completion_tokens=10,
        cache_read_input_tokens=2000,
    )
    assert p == pytest.approx(100 * 1.40e-6 + 2000 * 0.14e-6)
    assert c == pytest.approx(10 * 4.40e-6)


def test_doubleword_input_tokens_includes_cache_creation_flag():
    assert pr.get_provider(_DW).input_tokens_includes_cache_creation is True


# ---------------------------------------------------------------------------
# per_token_openai_target retained (user-sim litellm route + pydantic_ai)
# ---------------------------------------------------------------------------


def test_per_token_openai_target_retained():
    base, native, auth_env = pr.per_token_openai_target(_DW)
    assert base == "https://api.doubleword.ai/v1"
    assert native == _DW_NATIVE
    assert auth_env == "DOUBLEWORD_API_KEY"


# ---------------------------------------------------------------------------
# Scope guard: pydantic_ai Doubleword path is UNCHANGED (still OpenAI-format)
# ---------------------------------------------------------------------------


def test_pydantic_ai_doubleword_still_openai_chat_model(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-for-unit-tests")
    from bird_interact_agents.model_string import build_pydantic_ai_model

    m = build_pydantic_ai_model(_DW)
    # OpenAI-compatible providers return an OpenAIChatModel instance, not a
    # litellm string — DEV-1639 did NOT touch the pydantic_ai path.
    assert not isinstance(m, str)
