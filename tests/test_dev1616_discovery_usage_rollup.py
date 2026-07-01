"""DEV-1616: roll the warm-discovery sub-agent's turns / tokens / cost into
the task total.

The two-stage ``claude_sdk_*_v1`` agents delegate schema/entity discovery to a
warm second ``ClaudeSDKClient`` reached through the in-process ``ask_discovery``
tool (``discovery_runtime.run_main_with_discovery`` +
``discovery_channel.DiscoveryChannel``). Pre-fix:

* discovery TOKENS/COST already flowed into the shared ``accum`` (each ask
  builds a fresh ``SdkUsageTracker(accum, model)`` with the default
  ``scope="agent"``), but
* ``n_agent_turns`` was backfilled from the MAIN client's trajectory only —
  discovery turns were off-book — AND that backfill counted *block-level*
  ``AssistantMessage`` events, not dedup'd turns.

This module pins the fixed contract:

* :class:`SdkUsageTracker` exposes ``committed_n_calls`` — the exact turn count
  it contributes to the ``agent::<model>`` breakdown row on ``finalize`` (so a
  degenerate ``ResultMessage``-only ask, committed as ``max(1, turns)``, is
  reflected consistently).
* :class:`DiscoveryChannel` accumulates ``turns`` across asks from that count.
* :func:`run_main_with_discovery` reports a :class:`DiscoveryRollup`
  (``n_discovery_turns`` / ``n_discovery_calls``) — even on the crash path — so
  the caller can surface ``usage["n_discovery_turns"]`` and the headline
  ``n_agent_turns`` (derived from the breakdown) includes discovery.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from bird_interact_agents.agents.claude_sdk.agent import SdkUsageTracker
from bird_interact_agents.usage import TokenUsage


_MODEL = "anthropic/claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Minimal SDK-message stand-ins (class NAME is what the tracker/channel match).
# ---------------------------------------------------------------------------
class _TextBlock:
    def __init__(self, text: str):
        self.text = text


_TextBlock.__name__ = "TextBlock"


class _AssistantMessage:
    def __init__(self, usage, text: str = ""):
        self.usage = usage
        self.content = [_TextBlock(text)]


_AssistantMessage.__name__ = "AssistantMessage"


class _ResultMessage:
    def __init__(self, usage):
        self.usage = usage
        self.content = []


_ResultMessage.__name__ = "ResultMessage"


def _usage(inp: int = 0, out: int = 0, cache_read: int = 0, cache_write: int = 0):
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }


# ---------------------------------------------------------------------------
# SdkUsageTracker.committed_n_calls
# ---------------------------------------------------------------------------
def test_committed_n_calls_zero_before_finalize():
    t = SdkUsageTracker(TokenUsage(), _MODEL)
    assert t.committed_n_calls == 0


def test_committed_n_calls_equals_turns_with_result():
    """Two dedup'd turns (4 shared-usage blocks) + a terminal ResultMessage →
    committed == 2, matching the agent breakdown row's n_calls."""
    accum = TokenUsage()
    t = SdkUsageTracker(accum, _MODEL)
    u1, u2 = _usage(1, 0), _usage(2, 3)
    t.observe(_AssistantMessage(u1))
    t.observe(_AssistantMessage(u1))  # same dict → same turn
    t.observe(_AssistantMessage(u2))
    t.observe(_AssistantMessage(u2))  # same dict → same turn
    t.observe(_ResultMessage(_usage(3, 3)))
    assert t.committed_n_calls == 2
    agent_rows = [r for r in accum.breakdown if r.scope == "agent"]
    assert len(agent_rows) == 1 and agent_rows[0].n_calls == 2


def test_committed_n_calls_result_only_is_one():
    """A ResultMessage with NO observed AssistantMessage commits max(1, 0) = 1;
    committed_n_calls must equal the same 1 so it stays consistent with the
    breakdown (else n_discovery_turns would undercount)."""
    accum = TokenUsage()
    t = SdkUsageTracker(accum, _MODEL)
    t.observe(_ResultMessage(_usage(5, 1)))
    assert t.committed_n_calls == 1
    assert [r for r in accum.breakdown if r.scope == "agent"][0].n_calls == 1


def test_committed_n_calls_crash_path_equals_turns():
    """No ResultMessage (crash): committed == the summed per-turn count."""
    accum = TokenUsage()
    t = SdkUsageTracker(accum, _MODEL)
    t.observe(_AssistantMessage(_usage(1, 1)))
    t.observe(_AssistantMessage(_usage(2, 2)))
    t.finalize()
    assert t.committed_n_calls == 2


def test_committed_n_calls_no_activity_is_zero():
    accum = TokenUsage()
    t = SdkUsageTracker(accum, _MODEL)
    t.finalize()
    assert t.committed_n_calls == 0


# ---------------------------------------------------------------------------
# run_main_with_discovery — the end-to-end rollup (real trackers, fake clients)
# ---------------------------------------------------------------------------
class _FakeDiscoveryClient:
    """Warm discovery client. ``scripts[n]`` is the message list its Nth
    ``receive_response`` yields."""

    def __init__(self, scripts):
        self._scripts = scripts
        self._n = 0
        self.queries: list[str] = []

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def receive_response(self):
        msgs = self._scripts[self._n] if self._n < len(self._scripts) else []
        self._n += 1
        for m in msgs:
            yield m

    async def aclose(self) -> None:
        return None


class _FakeMainClient:
    """Main loop. On ``receive_response`` it first drives discovery
    ``ask``s (simulating the agent calling ``ask_discovery`` mid-loop), then
    yields its own messages, optionally raising afterwards."""

    def __init__(self, *, ask_texts, msgs, raise_exc=None):
        self._ask_texts = ask_texts
        self._msgs = msgs
        self._raise_exc = raise_exc

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self):
        from bird_interact_agents.agents.claude_sdk.agent import _ctx

        for t in self._ask_texts:
            await _ctx.get("_discovery").ask(t)
        for m in self._msgs:
            yield m
        if self._raise_exc is not None:
            raise self._raise_exc

    async def aclose(self) -> None:
        return None


def _install_fake_session(monkeypatch, discovery_client, main_client):
    """Patch the discovery_runtime hermetic session so the 1st enter returns
    the discovery client, the 2nd the main client (the module's entry order)."""
    from bird_interact_agents.agents.claude_sdk import discovery_runtime as dr

    order = [discovery_client, main_client]

    @contextlib.asynccontextmanager
    async def _fake_session(model, *, mcp_servers, build_options, **kw):
        yield order.pop(0)

    monkeypatch.setattr(dr, "hermetic_claude_sdk_session", _fake_session)


def _linear_cost(**kw):
    """Price every token at 1e-6 so cost is exactly (prompt+completion)*1e-6 —
    lets a test assert discovery cost is folded into agent_cost_usd."""
    return (kw["prompt_tokens"] * 1e-6, kw["completion_tokens"] * 1e-6)


async def _drive(monkeypatch, *, discovery_scripts, main_ask_texts, main_msgs,
                 raise_exc=None, cost_per_token=lambda **_: (0.0, 0.0)):
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.agents.claude_sdk import discovery_runtime as dr
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(usage_mod, "_cost_per_token", cost_per_token)

    disc = _FakeDiscoveryClient(discovery_scripts)
    main = _FakeMainClient(
        ask_texts=main_ask_texts, msgs=main_msgs, raise_exc=raise_exc,
    )
    _install_fake_session(monkeypatch, disc, main)

    # The ask_discovery native reads the channel from the per-task contextvar,
    # and run_main_with_discovery writes it there — seed a live dict.
    agent_mod._ctx_var.set({"_discovery": None})

    accum = TokenUsage()
    usage_tracker = agent_mod.SdkUsageTracker(accum, _MODEL)
    trajectory: list[dict] = []
    rollup = dr.DiscoveryRollup()

    await dr.run_main_with_discovery(
        model=_MODEL,
        accum=accum,
        usage_tracker=usage_tracker,
        context_state={},
        main_mcp_servers={},
        discovery_mcp_servers={},
        build_main_options=lambda kw: None,
        build_discovery_options=lambda kw: None,
        initial_query="go",
        trajectory=trajectory,
        discovery_rollup=rollup,
    )
    usage_tracker.finalize()
    return accum, trajectory, rollup, disc


@pytest.mark.asyncio
async def test_rollup_includes_discovery_turns_tokens_and_calls(monkeypatch):
    """Two discovery asks (1 turn each) + one main turn: the rollup reports 2
    discovery turns/calls, the agent breakdown n_calls == main + discovery, and
    discovery TOKENS are summed (per-ask, NOT cumulative) into the total."""
    accum, trajectory, rollup, disc = await _drive(
        monkeypatch,
        discovery_scripts=[
            [_AssistantMessage(_usage(100, 5), "a"), _ResultMessage(_usage(100, 5))],
            [_AssistantMessage(_usage(200, 7), "b"), _ResultMessage(_usage(200, 7))],
        ],
        main_ask_texts=["q1", "q2"],
        main_msgs=[_AssistantMessage(_usage(10, 2), "m"), _ResultMessage(_usage(10, 2))],
        cost_per_token=_linear_cost,
    )

    # Discovery rolled up.
    assert rollup.n_discovery_turns == 2
    assert rollup.n_discovery_calls == 2
    assert disc.queries == ["q1", "q2"]

    # Headline turn count (breakdown agent-scope) == main(1) + discovery(2).
    agent_rows = [r for r in accum.breakdown if r.scope == "agent"]
    assert len(agent_rows) == 1
    assert agent_rows[0].n_calls == 3

    # Tokens: per-ask summed, not cumulative → main 10 + disc 100 + 200.
    assert agent_rows[0].prompt_tokens == 310
    assert accum.prompt_tokens == 310
    # n_calls semantics unchanged: exactly main(1) + discovery(2), no double count.
    assert accum.n_calls == 3

    # D3 — discovery COST is merged into the single agent scope: agent_cost_usd
    # equals the total (no user_sim here) and reflects main + discovery tokens
    # (324 = (10+100+200) prompt + (2+5+7) completion, each @ 1e-6).
    assert accum.agent_cost_usd == pytest.approx(324e-6)
    assert accum.agent_cost_usd == pytest.approx(accum.cost_usd)
    assert accum.agent_cost_usd > 0
    assert agent_rows[0].cost_usd == pytest.approx(324e-6)


@pytest.mark.asyncio
async def test_trajectory_holds_only_main_messages(monkeypatch):
    """Discovery messages must NOT leak into the trajectory (that's the
    main-loop record); only the rolled-up counts carry discovery."""
    _, trajectory, rollup, _ = await _drive(
        monkeypatch,
        discovery_scripts=[
            [_AssistantMessage(_usage(50, 1), "secret-discovery"),
             _ResultMessage(_usage(50, 1))],
        ],
        main_ask_texts=["q1"],
        main_msgs=[_AssistantMessage(_usage(10, 2), "main"), _ResultMessage(_usage(10, 2))],
    )
    assert rollup.n_discovery_turns == 1
    types = [e["type"] for e in trajectory]
    assert types == ["AssistantMessage", "ResultMessage"]
    assert all("secret-discovery" not in str(e.get("data")) for e in trajectory)


@pytest.mark.asyncio
async def test_crash_path_still_reports_discovery_turns(monkeypatch):
    """If the main loop raises AFTER a discovery ask, the rollup (updated in
    run_main_with_discovery's finally) must still report that ask's turns."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.agents.claude_sdk import discovery_runtime as dr
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    disc = _FakeDiscoveryClient(
        [[_AssistantMessage(_usage(80, 3), "x"), _ResultMessage(_usage(80, 3))]]
    )
    main = _FakeMainClient(
        ask_texts=["q1"], msgs=[], raise_exc=RuntimeError("boom"),
    )
    _install_fake_session(monkeypatch, disc, main)
    agent_mod._ctx_var.set({"_discovery": None})

    accum = TokenUsage()
    usage_tracker = agent_mod.SdkUsageTracker(accum, _MODEL)
    rollup = dr.DiscoveryRollup()

    with pytest.raises(RuntimeError, match="boom"):
        await dr.run_main_with_discovery(
            model=_MODEL,
            accum=accum,
            usage_tracker=usage_tracker,
            context_state={},
            main_mcp_servers={},
            discovery_mcp_servers={},
            build_main_options=lambda kw: None,
            build_discovery_options=lambda kw: None,
            initial_query="go",
            trajectory=[],
            discovery_rollup=rollup,
        )
    usage_tracker.finalize()
    # Discovery turn happened before the crash → rolled up + in the accum.
    assert rollup.n_discovery_turns == 1
    assert rollup.n_discovery_calls == 1
    assert [r for r in accum.breakdown if r.scope == "agent"][0].prompt_tokens == 80
