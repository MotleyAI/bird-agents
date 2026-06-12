"""DEV-1555 Stage 2: user-sim litellm routing for registry providers.

`acompletion_tracked(model="moonshot/<id>")` must hit litellm with the
rewritten `openai/<id>` model + the provider's OpenAI-compatible api_base
and api_key — while the usage accumulator keeps the canonical
`moonshot/...` string. Anthropic routing stays byte-identical.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bird_interact_agents import usage as usage_mod
from bird_interact_agents.usage import TokenUsage, acompletion_tracked

_KIMI = "moonshot/kimi-k2.7-code"


def _install_recorder(monkeypatch, recorded: dict):
    async def _fake_acompletion(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=7, completion_tokens=3, reasoning_tokens=0,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            )
        )

    monkeypatch.setattr(usage_mod, "_acompletion", _fake_acompletion)
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))


@pytest.mark.asyncio
async def test_moonshot_user_sim_routed_via_openai_compat(monkeypatch):
    recorded: dict = {}
    _install_recorder(monkeypatch, recorded)
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")

    accum = TokenUsage()
    await acompletion_tracked(
        accum, scope="user_sim", model=_KIMI,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert recorded["model"] == "openai/kimi-k2.7-code"
    assert recorded["api_base"] == "https://api.moonshot.ai/v1"
    assert recorded["api_key"] == "ms-key-1"
    # The accumulator reports under the canonical string.
    rows = accum.model_dump()["breakdown"]
    assert rows and rows[0]["model"] == _KIMI


@pytest.mark.asyncio
async def test_caller_passed_api_key_not_clobbered(monkeypatch):
    recorded: dict = {}
    _install_recorder(monkeypatch, recorded)
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")

    await acompletion_tracked(
        TokenUsage(), scope="user_sim", model=_KIMI,
        messages=[], api_key="explicit-key",
    )
    assert recorded["api_key"] == "explicit-key"


@pytest.mark.asyncio
async def test_anthropic_user_sim_path_unchanged(monkeypatch):
    recorded: dict = {}
    _install_recorder(monkeypatch, recorded)
    monkeypatch.setenv(
        "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY", "renamed-key",
    )

    await acompletion_tracked(
        TokenUsage(), scope="user_sim",
        model="anthropic/claude-haiku-4-5-20251001", messages=[],
    )
    assert recorded["model"] == "anthropic/claude-haiku-4-5-20251001"
    assert recorded["api_key"] == "renamed-key"
    assert "api_base" not in recorded
