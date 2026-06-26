"""DEV-1604: Doubleword registry entry + bridge-proxy resolution helpers.

Doubleword (`api.doubleword.ai/v1`) is OpenAI-Chat-Completions-only — it has
NO Anthropic `/v1/messages` endpoint, so `claude_sdk*` agents can only reach
it through the local Anthropic⇄OpenAI bridge proxy. This module pins:

* the registry entry (`base_url=None`, `api_format="openai"`, the FP8 native
  id `zai-org/GLM-5.2-FP8`, pricing deliberately UNSET pending the DW console);
* `per_token_openai_target` (the upstream the proxy POSTs to);
* `agent_needs_bridge` (Doubleword always; z.ai only on `--zai-billing
  per-token`);
* the `resolve_base_url` fail-fast guard so a `None` base_url never leaks to
  any consumer (`sdk_session_env` AND `eval/autopsy`);
* that the anthropic-format providers (moonshot, z.ai coding-plan) are wholly
  unaffected by making `base_url` Optional.
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
    assert spec.api_format == "openai"
    assert spec.auth_env == "DOUBLEWORD_API_KEY"
    assert spec.base_url_env == "BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"
    # OpenAI-only: NO Anthropic endpoint — the proxy supplies ANTHROPIC_BASE_URL.
    assert spec.base_url is None
    assert spec.openai_base_url == "https://api.doubleword.ai/v1"


def test_doubleword_native_id_has_embedded_slash():
    """The DW GLM-5.2 id is `zai-org/GLM-5.2-FP8` (FP8). It must survive the
    first-slash `_split`: provider=doubleword, native=zai-org/GLM-5.2-FP8."""
    assert pr.get_provider(_DW).key == "doubleword"
    _, native = pr._split(_DW)
    assert native == _DW_NATIVE


def test_doubleword_context_window_placeholder():
    # Placeholder: mirrors DW's published GLM-5.1 window (~198K, max 202,752)
    # because DW does NOT publish GLM-5.2-FP8's window — MUST-CONFIRM console.
    assert pr.get_provider(_DW).default_context_window == 200_000


def test_doubleword_pricing_unset_and_flagged_unconfirmed():
    """DW per-token price is not public; leaving it UNSET keeps cost rows at $0
    (absent), never WRONG. `pricing_confirmed=False` is the explicit flag so the
    placeholder can't silently ship as if real numbers were entered."""
    spec = pr.get_provider(_DW)
    assert spec.model_pricing == {}
    assert spec.pricing_confirmed is False


def test_existing_providers_pricing_confirmed_default_true():
    assert pr.get_provider(_KIMI).pricing_confirmed is True
    assert pr.get_provider(_GLM).pricing_confirmed is True


def test_doubleword_is_supported_agent_model():
    assert pr.is_supported_agent_model(_DW)


def test_doubleword_required_env_for():
    assert pr.required_env_for(_DW) == ("DOUBLEWORD_API_KEY",)


def test_doubleword_does_not_register_litellm_pricing():
    """With model_pricing empty, ensure_litellm_pricing registers NO doubleword
    row — so the $0-unpriced fallback applies (correct) rather than a wrong
    borrowed price."""
    import litellm

    pr.ensure_litellm_pricing()
    # No zai-org/GLM-5.2-FP8 row was registered under either prefix.
    assert "doubleword/zai-org/GLM-5.2-FP8" not in litellm.model_cost
    assert "openai/zai-org/GLM-5.2-FP8" not in litellm.model_cost


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
        # Doubleword (openai-format) ALWAYS bridges, flag-agnostic.
        (_DW, True, True),
        (_DW, False, True),
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


def test_resolve_base_url_openai_provider_no_override_raises(monkeypatch):
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    spec = pr.get_provider(_DW)
    with pytest.raises(RuntimeError, match="BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"):
        pr.resolve_base_url(spec)


def test_resolve_base_url_openai_provider_empty_override_raises(monkeypatch):
    """An empty-string override (a common env misconfig) is falsy — it must
    raise, not fall through to the `None` base_url."""
    monkeypatch.setenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "")
    spec = pr.get_provider(_DW)
    with pytest.raises(RuntimeError, match="BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"):
        pr.resolve_base_url(spec)


def test_resolve_base_url_openai_provider_with_override(monkeypatch):
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


def test_sdk_session_env_doubleword_no_override_fails_fast(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    # Never emit ANTHROPIC_BASE_URL=None/"" — must raise loudly instead.
    with pytest.raises(RuntimeError, match="BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"):
        pr.sdk_session_env(_DW)


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
# Unpriced-registry cost path (regression: a Doubleword user-sim must not crash
# the run). litellm doesn't know the `doubleword/` provider AND pricing is
# unset, so the cost calc must short-circuit to $0 instead of raising.
# ---------------------------------------------------------------------------


def test_unpriced_doubleword_cost_is_zero_not_crash():
    from bird_interact_agents import usage

    prompt_cost, completion_cost = usage._safe_cost(
        model=_DW, prompt_tokens=1000, completion_tokens=500,
    )
    assert prompt_cost == 0.0
    assert completion_cost == 0.0


def test_priced_zai_cost_still_nonzero():
    from bird_interact_agents import usage

    prompt_cost, completion_cost = usage._safe_cost(
        model=_GLM, prompt_tokens=1_000_000, completion_tokens=1_000_000,
    )
    assert prompt_cost > 0
    assert completion_cost > 0
