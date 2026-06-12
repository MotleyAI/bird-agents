"""DEV-1555 Stage 2: actor-env credential hygiene for open-weight runs.

On an open-weight run (registry provider key shipped, no OAuth token),
ambient Anthropic credentials must be stripped from the actor process env
— the claude CLI auto-discovers ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN
and would silently route the agent to Anthropic instead of the configured
ANTHROPIC_BASE_URL backend. The OAuth invariant gains a sibling guard for
registry-agent runs.
"""

from __future__ import annotations

import os

import pytest

from bird_interact_agents.cloud import ray_app

_KIMI = "moonshot/kimi-k2.7-code"


def test_open_weight_actor_env_strips_anthropic_creds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anth")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ambient")

    ray_app._apply_actor_env_local(
        {"MOONSHOT_API_KEY": "ms-key-1", "OPENAI_API_KEY": "openai-key"}
    )
    assert os.environ.get("MOONSHOT_API_KEY") == "ms-key-1"
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ


def test_oauth_actor_env_behavior_unchanged(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anth")
    ray_app._apply_actor_env_local(
        {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x"}
    )
    assert os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") == "sk-ant-oat01-x"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_invariant_raises_when_oauth_survives_on_registry_run(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {"framework": "claude_sdk", "agent_model": _KIMI}
    with pytest.raises(RuntimeError):
        ray_app._assert_actor_oauth_invariant(cfg)


def test_invariant_quiet_on_clean_registry_run(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    cfg = {"framework": "claude_sdk", "agent_model": _KIMI}
    ray_app._assert_actor_oauth_invariant(cfg)  # must not raise


def test_invariant_unchanged_for_anthropic_oauth_run(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = {
        "framework": "claude_sdk",
        "agent_model": "anthropic/claude-sonnet-4-6",
    }
    ray_app._assert_actor_oauth_invariant(cfg)  # must not raise
