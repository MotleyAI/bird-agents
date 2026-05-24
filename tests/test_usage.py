"""Tests for the TokenUsage accumulator and acompletion_tracked wrapper.

Covers:
- add_call() aggregates per (scope, model) into the breakdown list and
  rolls scope subtotals (agent_cost_usd / user_sim_cost_usd / cost_usd).
- merge() of two TokenUsage instances sums totals and merges breakdown
  rows by (scope, model).
- merge() is order-independent (associative + commutative on the fields
  we care about).
- Cost is computed via litellm.cost_per_token at add_call time; callers
  see numerical values, not raw token counts only.
- Unpriced models record cost_usd=0.0 without raising.
- acompletion_tracked() parses litellm.acompletion's response.usage and
  records exactly one matching call, with the same response object
  returned to the caller.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# TokenUsage / CallCost
# ---------------------------------------------------------------------------

def test_add_call_records_first_call():
    from bird_interact_agents.usage import TokenUsage

    u = TokenUsage()
    u.add_call(
        scope="agent",
        model="anthropic/claude-sonnet-4-5",
        prompt=1000,
        completion=200,
    )

    assert u.n_calls == 1
    assert u.prompt_tokens == 1000
    assert u.completion_tokens == 200
    assert len(u.breakdown) == 1
    row = u.breakdown[0]
    assert row.scope == "agent"
    assert row.model == "anthropic/claude-sonnet-4-5"
    assert row.prompt_tokens == 1000
    assert row.completion_tokens == 200
    assert row.n_calls == 1
    assert row.name == "agent::anthropic/claude-sonnet-4-5"


def test_add_call_aggregates_same_scope_and_model(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.usage import TokenUsage

    monkeypatch.setattr(
        usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0)
    )

    u = TokenUsage()
    u.add_call(scope="agent", model="m", prompt=100, completion=10)
    u.add_call(scope="agent", model="m", prompt=300, completion=20)

    assert u.n_calls == 2
    assert u.prompt_tokens == 400
    assert u.completion_tokens == 30
    assert len(u.breakdown) == 1
    assert u.breakdown[0].n_calls == 2
    assert u.breakdown[0].prompt_tokens == 400


def test_add_call_separates_scopes_and_models(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.usage import TokenUsage

    monkeypatch.setattr(
        usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0)
    )

    u = TokenUsage()
    u.add_call(scope="agent", model="m1", prompt=10, completion=1)
    u.add_call(scope="user_sim", model="m2", prompt=20, completion=2)
    u.add_call(scope="agent", model="m2", prompt=30, completion=3)

    assert len(u.breakdown) == 3
    names = {row.name for row in u.breakdown}
    assert names == {"agent::m1", "user_sim::m2", "agent::m2"}


def test_scope_rolled_costs(monkeypatch):
    """Cost is filled per-call via litellm.cost_per_token; agent/user_sim
    subtotals roll up correctly."""
    from bird_interact_agents import usage as usage_mod

    def fake_cost(*, model, prompt_tokens, completion_tokens, **_):
        # 1 USD per token for prompt and 2 USD per token for completion,
        # regardless of model — easy to verify.
        return prompt_tokens * 1.0, completion_tokens * 2.0

    monkeypatch.setattr(usage_mod, "_cost_per_token", fake_cost)

    u = usage_mod.TokenUsage()
    u.add_call(scope="agent", model="m", prompt=2, completion=3)       # 2*1 + 3*2 = 8
    u.add_call(scope="user_sim", model="m", prompt=1, completion=1)    # 1+2 = 3

    assert u.agent_cost_usd == pytest.approx(8.0)
    assert u.user_sim_cost_usd == pytest.approx(3.0)
    assert u.cost_usd == pytest.approx(11.0)


def test_unpriced_model_does_not_raise(monkeypatch):
    """`litellm.cost_per_token` raises `NotFoundError` when a model
    isn't in its pricing table — that's the expected, recoverable case
    (warn once, record $0)."""
    import litellm
    from bird_interact_agents import usage as usage_mod

    def fake_cost(*, model, **_):
        raise litellm.exceptions.NotFoundError(
            "model not in pricing table",
            model=model,
            llm_provider="test",
        )

    monkeypatch.setattr(usage_mod, "_cost_per_token", fake_cost)

    u = usage_mod.TokenUsage()
    u.add_call(scope="agent", model="exotic/model", prompt=10, completion=5)

    assert u.cost_usd == 0.0
    assert u.agent_cost_usd == 0.0
    assert u.breakdown[0].cost_usd == 0.0


def test_non_pricing_errors_are_not_swallowed(monkeypatch):
    """Anything that isn't a NotFoundError is an integration bug — let
    it surface instead of silently degrading cost to $0 across the run."""
    import pytest
    from bird_interact_agents import usage as usage_mod

    def fake_cost(*, model, **_):
        raise RuntimeError("transport blew up")

    monkeypatch.setattr(usage_mod, "_cost_per_token", fake_cost)

    u = usage_mod.TokenUsage()
    with pytest.raises(RuntimeError, match="transport blew up"):
        u.add_call(scope="agent", model="m", prompt=10, completion=5)


def test_merge_sums_totals_and_merges_breakdown(monkeypatch):
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(
        usage_mod, "_cost_per_token",
        lambda *, model, prompt_tokens, completion_tokens, **_: (
            prompt_tokens * 0.0, completion_tokens * 0.0,
        ),
    )

    a = usage_mod.TokenUsage()
    a.add_call(scope="agent", model="m1", prompt=10, completion=1)
    a.add_call(scope="user_sim", model="m2", prompt=20, completion=2)

    b = usage_mod.TokenUsage()
    b.add_call(scope="agent", model="m1", prompt=30, completion=3)
    b.add_call(scope="agent", model="m3", prompt=5, completion=0)

    a.merge(b)

    assert a.n_calls == 4
    assert a.prompt_tokens == 65
    assert a.completion_tokens == 6
    by_name = {row.name: row for row in a.breakdown}
    assert set(by_name) == {"agent::m1", "user_sim::m2", "agent::m3"}
    assert by_name["agent::m1"].n_calls == 2
    assert by_name["agent::m1"].prompt_tokens == 40


def test_merge_is_order_independent(monkeypatch):
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(
        usage_mod, "_cost_per_token",
        lambda *, model, prompt_tokens, completion_tokens, **_: (
            prompt_tokens * 1.0, completion_tokens * 1.0,
        ),
    )

    def make(scope, model, prompt, completion):
        u = usage_mod.TokenUsage()
        u.add_call(scope=scope, model=model, prompt=prompt, completion=completion)
        return u

    parts = [
        make("agent", "m1", 10, 1),
        make("agent", "m1", 30, 3),
        make("user_sim", "m2", 20, 2),
    ]

    forward = usage_mod.TokenUsage()
    for p in parts:
        forward.merge(p)

    reverse = usage_mod.TokenUsage()
    for p in reversed(parts):
        reverse.merge(p)

    assert forward.prompt_tokens == reverse.prompt_tokens
    assert forward.completion_tokens == reverse.completion_tokens
    assert forward.n_calls == reverse.n_calls
    assert forward.cost_usd == pytest.approx(reverse.cost_usd)
    assert forward.agent_cost_usd == pytest.approx(reverse.agent_cost_usd)
    assert forward.user_sim_cost_usd == pytest.approx(reverse.user_sim_cost_usd)
    assert (
        sorted(r.name for r in forward.breakdown)
        == sorted(r.name for r in reverse.breakdown)
    )


def test_model_dump_round_trip(monkeypatch):
    """TokenUsage.model_dump → model_validate round-trips cleanly so we can
    serialise it through eval.json."""
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(
        usage_mod, "_cost_per_token",
        lambda *, model, prompt_tokens, completion_tokens, **_: (0.5, 0.5),
    )

    u = usage_mod.TokenUsage()
    u.add_call(scope="agent", model="m", prompt=10, completion=5)

    blob = u.model_dump()
    revived = usage_mod.TokenUsage.model_validate(blob)
    assert revived.prompt_tokens == 10
    assert revived.completion_tokens == 5
    assert revived.cost_usd == pytest.approx(1.0)
    assert revived.breakdown[0].name == "agent::m"


# ---------------------------------------------------------------------------
# acompletion_tracked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acompletion_tracked_records_usage(monkeypatch):
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(
        usage_mod, "_cost_per_token",
        lambda **_: (0.0, 0.0),
    )

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=42,
            completion_tokens=7,
        ),
    )

    async def fake_acompletion(**kwargs):
        # capture keyword args so we can confirm they pass through
        fake_acompletion.last_kwargs = kwargs
        return fake_response

    monkeypatch.setattr(usage_mod, "_acompletion", fake_acompletion)

    accum = usage_mod.TokenUsage()
    out = await usage_mod.acompletion_tracked(
        accum,
        scope="user_sim",
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert out is fake_response
    assert accum.n_calls == 1
    assert accum.prompt_tokens == 42
    assert accum.completion_tokens == 7
    row = accum.breakdown[0]
    assert row.scope == "user_sim"
    assert row.model == "anthropic/claude-haiku-4-5-20251001"
    assert fake_acompletion.last_kwargs["model"] == row.model
    assert fake_acompletion.last_kwargs["messages"] == [
        {"role": "user", "content": "hi"}
    ]


@pytest.mark.asyncio
async def test_acompletion_tracked_handles_missing_usage(monkeypatch):
    """If the provider returns no `usage` block, we still return the
    response and leave the accumulator at zero."""
    from bird_interact_agents import usage as usage_mod

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
    )

    async def fake_acompletion(**_):
        return fake_response

    monkeypatch.setattr(usage_mod, "_acompletion", fake_acompletion)

    accum = usage_mod.TokenUsage()
    out = await usage_mod.acompletion_tracked(
        accum, scope="agent", model="m", messages=[],
    )

    assert out is fake_response
    assert accum.n_calls == 0
    assert accum.prompt_tokens == 0
    assert accum.breakdown == []
