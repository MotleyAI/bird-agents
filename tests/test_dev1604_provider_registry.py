"""DEV-1604 registry/bridge helpers, updated for DEV-1639.

DEV-1639: Doubleword now exposes a NATIVE Anthropic `/v1/messages` endpoint, so
`claude_sdk*` agents reach it DIRECTLY (anthropic-format, no bridge) and it is
priced (realtime GLM-5.2-FP8). The bridge proxy stays for z.ai per-token. This
module pins:

* the (updated) Doubleword registry entry (`base_url="https://api.doubleword.ai"`,
  `api_format="anthropic"`, FP8 native id `zai-org/GLM-5.2-FP8`, priced);
* `per_token_openai_target` (still valid — the openai_base_url the user-sim +
  pydantic_ai use);
* `agent_needs_bridge` (Doubleword NEVER now; z.ai only on the default
  `--no-subscription-auth` / per-token path);
* the `resolve_base_url` fail-fast guard for a hypothetical `None`-base_url
  provider (via a synthetic spec);
* that the anthropic-format providers (moonshot, z.ai coding-plan) are unaffected.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import provider_registry as pr

_DW = "doubleword/zai-org/GLM-5.2-FP8"
_DW_NATIVE = "zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"
_KIMI = "moonshot/kimi-k2.7-code"


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------


def test_doubleword_entry_fields():
    spec = pr.get_provider(_DW)
    assert spec is not None
    assert spec.key == "doubleword"
    # DEV-1639: Doubleword now exposes a NATIVE Anthropic endpoint, so it is
    # anthropic-format and reaches the SDK directly (no bridge).
    assert spec.api_format == "anthropic"
    assert spec.auth_env == "DOUBLEWORD_API_KEY"
    assert spec.base_url_env == "BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"
    assert spec.base_url == "https://api.doubleword.ai"
    # openai_base_url is retained for the user-sim litellm route + pydantic_ai.
    assert spec.openai_base_url == "https://api.doubleword.ai/v1"


def test_doubleword_native_id_has_embedded_slash():
    """The DW GLM-5.2 id is `zai-org/GLM-5.2-FP8` (FP8). It must survive the
    first-slash `_split`: provider=doubleword, native=zai-org/GLM-5.2-FP8."""
    assert pr.get_provider(_DW).key == "doubleword"
    _, native = pr._split(_DW)
    assert native == _DW_NATIVE


def test_doubleword_context_window():
    # DEV-1639: Doubleword's model page publishes GLM-5.2-FP8's max total tokens.
    assert pr.get_provider(_DW).default_context_window == 1_048_576


def test_doubleword_pricing_confirmed():
    """DEV-1639: GLM-5.2-FP8 realtime pricing is now published — input $1.40/M,
    output $4.40/M, cache-read 0.1× ($0.14/M), cache writes 1.25× (5m) / 2× (1h)."""
    spec = pr.get_provider(_DW)
    assert spec.pricing_confirmed is True
    pricing = spec.model_pricing[_DW_NATIVE]
    assert pricing.input_cost_per_token == 1.40e-6
    assert pricing.output_cost_per_token == 4.40e-6
    assert pricing.cache_read_input_token_cost == 0.14e-6
    assert pricing.cache_write_5m_multiplier == 1.25
    assert pricing.cache_write_1h_multiplier == 2.0
    # DW's native usage reports input_tokens INCLUSIVE of cache-creation.
    assert spec.input_tokens_includes_cache_creation is True


def test_existing_providers_pricing_confirmed_default_true():
    assert pr.get_provider(_KIMI).pricing_confirmed is True
    assert pr.get_provider(_GLM).pricing_confirmed is True


def test_doubleword_is_supported_agent_model():
    assert pr.is_supported_agent_model(_DW)


def test_doubleword_required_env_for():
    assert pr.required_env_for(_DW) == ("DOUBLEWORD_API_KEY",)


def test_doubleword_registers_litellm_pricing():
    """DEV-1639: with GLM-5.2-FP8 now priced, ensure_litellm_pricing registers
    the row under BOTH the canonical `doubleword/` and the rewritten `openai/`
    key (the user-sim litellm route uses the latter)."""
    import litellm

    pr.ensure_litellm_pricing()
    assert "doubleword/zai-org/GLM-5.2-FP8" in litellm.model_cost
    assert "openai/zai-org/GLM-5.2-FP8" in litellm.model_cost


def test_no_native_id_collision_between_zai_and_doubleword():
    """The issue feared a shared `openai/glm-5.2` row collision. It can't
    happen: DW's native id is `zai-org/GLM-5.2-FP8`, z.ai's is `glm-5.2`."""
    _, dw_native = pr._split(_DW)
    _, zai_native = pr._split(_GLM)
    assert dw_native != zai_native


# ---------------------------------------------------------------------------
# per_token_openai_target
# ---------------------------------------------------------------------------


def test_per_token_openai_target_doubleword():
    base, native, auth_env = pr.per_token_openai_target(_DW)
    assert base == "https://api.doubleword.ai/v1"
    assert native == _DW_NATIVE
    assert auth_env == "DOUBLEWORD_API_KEY"


def test_per_token_openai_target_zai():
    base, native, auth_env = pr.per_token_openai_target(_GLM)
    assert base == "https://api.z.ai/api/paas/v4"
    assert native == "glm-5.2"
    assert auth_env == "ZAI_API_KEY"


def test_per_token_openai_target_rejects_non_registry():
    with pytest.raises((ValueError, RuntimeError)):
        pr.per_token_openai_target("anthropic/claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# agent_needs_bridge — the truth table
# ---------------------------------------------------------------------------


# DEV-1604: the bridge keys on the recycled --subscription-auth flag, carried as
# no_subscription_auth (True = API-key/per-token path = bridge for z.ai).
@pytest.mark.parametrize(
    "model,no_subscription_auth,expected",
    [
        # DEV-1639: Doubleword now talks its NATIVE Anthropic endpoint directly
        # (anthropic-format) — it NEVER bridges, flag-agnostic.
        (_DW, True, False),
        (_DW, False, False),
        # z.ai bridges on the per-token path (no_subscription_auth=True, the
        # default); --subscription-auth (False) keeps the direct coding-plan.
        (_GLM, True, True),
        (_GLM, False, False),
        # Anthropic-format moonshot never bridges.
        (_KIMI, True, False),
        (_KIMI, False, False),
        # Anthropic-proper never bridges.
        ("anthropic/claude-sonnet-4-6", True, False),
        ("anthropic/claude-sonnet-4-6", False, False),
    ],
)
def test_agent_needs_bridge(model, no_subscription_auth, expected):
    assert pr.agent_needs_bridge(model, no_subscription_auth) is expected


# ---------------------------------------------------------------------------
# resolve_base_url guard (Codex #6): never leak None to ANY consumer
# ---------------------------------------------------------------------------


def test_resolve_base_url_none_base_url_no_override_raises(monkeypatch):
    """The DEV-1604 fail-fast guard still stands for a hypothetical OpenAI-only
    provider (base_url=None) with no override — it must raise rather than leak a
    falsy None. Exercised via a synthetic spec now that Doubleword (DEV-1639) has
    a real base_url."""
    spec = pr.ProviderSpec(
        key="synthetic", base_url=None, base_url_env="BIRD_SYNTHETIC_BASE_URL",
        api_format="openai", auth_env="SYNTHETIC_API_KEY",
        default_context_window=200_000,
    )
    monkeypatch.delenv("BIRD_SYNTHETIC_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="BIRD_SYNTHETIC_BASE_URL"):
        pr.resolve_base_url(spec)


def test_resolve_base_url_doubleword_default_direct(monkeypatch):
    """DEV-1639: with no override set, Doubleword resolves to its static NATIVE
    Anthropic endpoint (no bridge, no raise)."""
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    assert pr.resolve_base_url(pr.get_provider(_DW)) == "https://api.doubleword.ai"


def test_resolve_base_url_doubleword_override_wins(monkeypatch):
    monkeypatch.setenv(
        "BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "http://127.0.0.1:8788"
    )
    spec = pr.get_provider(_DW)
    assert pr.resolve_base_url(spec) == "http://127.0.0.1:8788"


def test_resolve_base_url_anthropic_providers_unaffected(monkeypatch):
    """Making base_url Optional must NOT change the anthropic-format providers:
    they still resolve their direct endpoint with no override set."""
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("BIRD_ZAI_ANTHROPIC_BASE_URL", raising=False)
    assert pr.resolve_base_url(pr.get_provider(_KIMI)) == (
        "https://api.moonshot.ai/anthropic"
    )
    assert pr.resolve_base_url(pr.get_provider(_GLM)) == (
        "https://api.z.ai/api/anthropic"
    )


# ---------------------------------------------------------------------------
# sdk_session_env fail-fast + proxy resolution (the 2nd guard)
# ---------------------------------------------------------------------------


def test_sdk_session_env_doubleword_default_direct(monkeypatch):
    """DEV-1639: with no override, the SDK session env points ANTHROPIC_BASE_URL
    at Doubleword's native endpoint and carries the Bearer provider key."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    env = pr.sdk_session_env(_DW)
    assert env == {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "https://api.doubleword.ai",
        "ANTHROPIC_AUTH_TOKEN": "dw-key-1",
    }


def test_sdk_session_env_doubleword_with_proxy_override(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv(
        "BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "http://127.0.0.1:8788"
    )
    env = pr.sdk_session_env(_DW)
    assert env == {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8788",
        "ANTHROPIC_AUTH_TOKEN": "dw-key-1",
    }


def test_sdk_session_env_doubleword_missing_key_raises(monkeypatch):
    monkeypatch.setenv(
        "BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "http://127.0.0.1:8788"
    )
    monkeypatch.delenv("DOUBLEWORD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DOUBLEWORD_API_KEY"):
        pr.sdk_session_env(_DW)


# ---------------------------------------------------------------------------
# DEV-1639: Doubleword is now priced (realtime GLM-5.2-FP8), so its cost rows are
# non-zero (the DEV-1604 $0-unpriced-fallback no longer applies to it).
# ---------------------------------------------------------------------------


def test_priced_doubleword_cost_is_nonzero():
    from bird_interact_agents import usage

    prompt_cost, completion_cost = usage._safe_cost(
        model=_DW, prompt_tokens=1000, completion_tokens=500,
    )
    assert prompt_cost == 1000 * 1.40e-6
    assert completion_cost == 500 * 4.40e-6


def test_priced_zai_cost_still_nonzero():
    from bird_interact_agents import usage

    prompt_cost, completion_cost = usage._safe_cost(
        model=_GLM, prompt_tokens=1_000_000, completion_tokens=1_000_000,
    )
    assert prompt_cost > 0
    assert completion_cost > 0
