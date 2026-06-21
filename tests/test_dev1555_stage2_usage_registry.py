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


def test_agent_side_cost_rows_priced_for_registry_models():
    """Local-smoke regression (2026-06-12): the AGENT-side cost path
    (accumulate_assistant_usage -> add_call -> litellm.cost_per_token)
    crashed with litellm's bare 'This model isn't mapped yet' Exception
    because pricing registration only ran on the user-sim route. add_call
    must price registry models without any prior setup call."""
    accum = TokenUsage()
    accum.add_call(
        scope="agent", model=_KIMI, prompt=1_000_000, completion=1_000_000,
    )
    assert accum.cost_usd == pytest.approx(0.95 + 4.00, rel=1e-6)


def test_registry_cache_pricing_anthropic_convention():
    """Cache pricing must use Anthropic-convention inputs (prompt EXCLUDES
    cached tokens — that's what accumulate_assistant_usage passes) and
    Moonshot's hit/miss model: cache WRITES are cache misses billed at the
    full input rate ($0.95/M); reads at $0.19/M. litellm's openai-provider
    math gets both wrong (swallows non-cached input when cache_read is
    present; prices cache_creation at $0), so registry models are priced
    in-house."""
    accum = TokenUsage()
    accum.add_call(
        scope="agent", model=_KIMI,
        prompt=100_000,          # non-cached input -> 0.095
        completion=10_000,       # output -> 0.040
        cache_read=1_000_000,    # hits -> 0.19
        cache_write=500_000,     # misses (creation) -> 0.475
    )
    assert accum.cost_usd == pytest.approx(0.095 + 0.04 + 0.19 + 0.475, rel=1e-6)


# ---------------------------------------------------------------------------
# DEV-1580: z.ai (GLM) gets the same user-sim litellm route + in-house
# agent-side pricing as Moonshot.
# ---------------------------------------------------------------------------

_ZAI = "zai/glm-5.2"


@pytest.mark.asyncio
async def test_zai_user_sim_routed_via_openai_compat(monkeypatch):
    recorded: dict = {}
    _install_recorder(monkeypatch, recorded)
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")

    accum = TokenUsage()
    await acompletion_tracked(
        accum, scope="user_sim", model=_ZAI,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert recorded["model"] == "openai/glm-5.2"
    assert recorded["api_base"] == "https://api.z.ai/api/paas/v4"
    assert recorded["api_key"] == "zai-key-1"
    rows = accum.model_dump()["breakdown"]
    assert rows and rows[0]["model"] == _ZAI


def test_zai_agent_side_cost_rows_priced():
    """GLM-5.2 agent-side cost: input $1.40/M + output $4.40/M, priced
    in-house from the registry (no prior setup call needed)."""
    accum = TokenUsage()
    accum.add_call(
        scope="agent", model=_ZAI, prompt=1_000_000, completion=1_000_000,
    )
    assert accum.cost_usd == pytest.approx(1.40 + 4.40, rel=1e-6)


def test_zai_cache_pricing_honours_cache_hit_rate():
    """GLM-5.2 cache-hit reads bill at $0.26/M (not the $1.40/M input
    rate); cache writes are billed at the full input rate, same in-house
    convention as Moonshot."""
    accum = TokenUsage()
    accum.add_call(
        scope="agent", model=_ZAI,
        prompt=100_000,          # non-cached input -> 0.14
        completion=10_000,       # output -> 0.044
        cache_read=1_000_000,    # hits @ 0.26/M -> 0.26
        cache_write=500_000,     # writes @ 1.40/M input rate -> 0.70
    )
    assert accum.cost_usd == pytest.approx(
        0.14 + 0.044 + 0.26 + 0.70, rel=1e-6,
    )


def test_zai_glm46_cache_read_falls_back_to_input_rate():
    """Codex r2: glm-4.6 has NO cache_read_input_token_cost, so cache-read
    tokens bill at the full input rate ($0.60/M) — exercises the
    `cache_read_rate or input_rate` fallback branch in _safe_cost."""
    accum = TokenUsage()
    accum.add_call(
        scope="agent", model="zai/glm-4.6",
        prompt=0, completion=0,
        cache_read=1_000_000,    # no cache-hit price -> input rate 0.60/M
        cache_write=0,
    )
    assert accum.cost_usd == pytest.approx(0.60, rel=1e-6)


def test_zai_unknown_model_runs_but_reports_zero_cost(monkeypatch):
    """Documented contract: ANY zai/<id> is a supported agent model and
    routes through the OpenAI-compatible litellm path, but an id without a
    registry pricing entry reports $0 (litellm's NotFoundError -> warn-once,
    cost 0) rather than crashing."""
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    assert pr.is_supported_agent_model("zai/some-future-glm")
    assert pr.required_env_for("zai/some-future-glm") == ("ZAI_API_KEY",)
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    litellm_model, kwargs = pr.litellm_route("zai/some-future-glm")
    assert litellm_model == "openai/some-future-glm"
    assert kwargs["api_base"] == "https://api.z.ai/api/paas/v4"

    accum = TokenUsage()
    accum.add_call(
        scope="agent", model="zai/some-future-glm",
        prompt=1_000_000, completion=1_000_000,
    )
    assert accum.cost_usd == pytest.approx(0.0)


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
