"""DEV-1555 Stage 2: context_window_for consults the provider registry.

Stage-1 pins (anthropic -> 1M behavior-preserving, unknown -> 200K
conservative) must survive; registry models get their real windows.
"""

from __future__ import annotations

from bird_interact_agents.agents.claude_sdk.context_budget import (
    context_window_for,
)


def test_stage1_pins_unchanged():
    assert context_window_for("anthropic/claude-opus-4-7") == 1_000_000
    assert context_window_for("anthropic/claude-sonnet-4-6") == 1_000_000
    assert context_window_for("unknownprov/some-model") == 200_000
    assert context_window_for("bare-model-id") == 200_000


def test_moonshot_window_from_registry():
    assert context_window_for("moonshot/kimi-k2.7-code") == 262_144


def test_per_model_override_beats_provider_default(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    spec = pr.get_provider("moonshot/kimi-k2.7-code")
    patched = spec.model_copy(
        update={"model_context_windows": {"kimi-special": 123_456}}
    )
    monkeypatch.setitem(pr.REGISTRY, "moonshot", patched)
    assert context_window_for("moonshot/kimi-special") == 123_456
    # Other models of the same provider keep the provider default.
    assert context_window_for("moonshot/kimi-k2.7-code") == 262_144
