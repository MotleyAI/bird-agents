"""DEV-1555 Stage 2: provider keys through prereqs + driver secret shipping.

The registry is the single source of required env vars. For a claude_sdk
run with a registry agent model the driver ships the provider key (and the
renamed anthropic user-sim key when applicable) and NEVER raw
ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN — ambient Anthropic creds on
the submitter machine must not reach the workers of an open-weight run.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import driver, prereqs

_KIMI = "moonshot/kimi-k2.7-code"


def test_required_api_keys_knows_moonshot():
    assert prereqs._required_api_keys(_KIMI) == ("MOONSHOT_API_KEY",)
    # Existing mappings unchanged.
    assert prereqs._required_api_keys("anthropic/x") == ("ANTHROPIC_API_KEY",)
    assert prereqs._required_api_keys("cerebras/x") == ("CEREBRAS_API_KEY",)


def _clear_creds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anth")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")


def test_all_kimi_run_ships_only_provider_keys(monkeypatch):
    _clear_creds(monkeypatch)
    result = driver.read_api_keys_from_local_env(
        _KIMI, _KIMI,
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=True, dataset="mini-interact",
    )
    assert result["MOONSHOT_API_KEY"] == "ms-key-1"
    assert result["OPENAI_API_KEY"] == "openai-key"  # slayer embeddings
    assert "ANTHROPIC_API_KEY" not in result
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in result
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in result


def test_mixed_anthropic_user_sim_ships_renamed_key(monkeypatch):
    _clear_creds(monkeypatch)
    result = driver.read_api_keys_from_local_env(
        _KIMI, "anthropic/claude-haiku-4-5-20251001",
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=True, dataset="mini-interact",
    )
    assert result["MOONSHOT_API_KEY"] == "ms-key-1"
    # The worker must never see raw ANTHROPIC_API_KEY (the SDK would
    # auto-discover it and route the AGENT to Anthropic).
    assert "ANTHROPIC_API_KEY" not in result
    assert result["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] == "ambient-anth"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in result


def test_missing_provider_key_fails_fast(monkeypatch):
    _clear_creds(monkeypatch)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError, match="MOONSHOT_API_KEY"):
        driver.read_api_keys_from_local_env(
            _KIMI, _KIMI,
            query_mode="slayer", framework="claude_sdk",
            no_subscription_auth=True, dataset="mini-interact",
        )


def test_base_url_override_forwarded_when_set(monkeypatch):
    _clear_creds(monkeypatch)
    monkeypatch.setenv(
        "BIRD_MOONSHOT_ANTHROPIC_BASE_URL", "https://other.example/anthropic",
    )
    result = driver.read_api_keys_from_local_env(
        _KIMI, _KIMI,
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=True, dataset="mini-interact",
    )
    assert (
        result["BIRD_MOONSHOT_ANTHROPIC_BASE_URL"]
        == "https://other.example/anthropic"
    )


# ---------------------------------------------------------------------------
# DEV-1580: z.ai (GLM) shipping mirrors Moonshot. Critically, a zai run with
# ONLY ZAI_API_KEY set (MOONSHOT_API_KEY absent) must ship just ZAI_API_KEY —
# the driver ships the agent's own provider key, never every registry key.
# ---------------------------------------------------------------------------

_GLM = "zai/glm-5.2"


def test_required_api_keys_knows_zai():
    assert prereqs._required_api_keys(_GLM) == ("ZAI_API_KEY",)


def _clear_creds_zai(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anth")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    # The OTHER registry provider's key is deliberately ABSENT.
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)


def test_zai_run_ships_only_provider_keys(monkeypatch):
    _clear_creds_zai(monkeypatch)
    result = driver.read_api_keys_from_local_env(
        _GLM, _GLM,
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=True, dataset="mini-interact",
    )
    assert result["ZAI_API_KEY"] == "zai-key-1"
    assert result["OPENAI_API_KEY"] == "openai-key"  # slayer embeddings
    # A zai run must NOT demand or ship the unrelated Moonshot key.
    assert "MOONSHOT_API_KEY" not in result
    assert "ANTHROPIC_API_KEY" not in result
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in result
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in result


def test_zai_mixed_anthropic_user_sim_ships_renamed_key(monkeypatch):
    _clear_creds_zai(monkeypatch)
    result = driver.read_api_keys_from_local_env(
        _GLM, "anthropic/claude-haiku-4-5-20251001",
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=True, dataset="mini-interact",
    )
    assert result["ZAI_API_KEY"] == "zai-key-1"
    assert "ANTHROPIC_API_KEY" not in result
    assert result["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] == "ambient-anth"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in result


def test_zai_missing_provider_key_fails_fast(monkeypatch):
    _clear_creds_zai(monkeypatch)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError, match="ZAI_API_KEY"):
        driver.read_api_keys_from_local_env(
            _GLM, _GLM,
            query_mode="slayer", framework="claude_sdk",
            no_subscription_auth=True, dataset="mini-interact",
        )


def test_zai_base_url_override_forwarded_when_set(monkeypatch):
    _clear_creds_zai(monkeypatch)
    monkeypatch.setenv(
        "BIRD_ZAI_ANTHROPIC_BASE_URL", "https://other.example/anthropic",
    )
    result = driver.read_api_keys_from_local_env(
        _GLM, _GLM,
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=True, dataset="mini-interact",
    )
    assert (
        result["BIRD_ZAI_ANTHROPIC_BASE_URL"]
        == "https://other.example/anthropic"
    )


def test_oauth_branch_unchanged(monkeypatch):
    """Anthropic subscription runs keep the existing OAuth shipping."""
    _clear_creds(monkeypatch)
    result = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5-20251001",
        query_mode="slayer", framework="claude_sdk",
        no_subscription_auth=False, dataset="mini-interact",
    )
    assert result["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-ambient"
    assert "ANTHROPIC_API_KEY" not in result
    assert result["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] == "ambient-anth"
