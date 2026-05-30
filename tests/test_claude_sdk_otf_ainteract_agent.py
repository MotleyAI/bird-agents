"""Unit tests for the `claude_sdk_otf_ainteract` agent (DEV-1507 split).

The agent runs the mini-interact / a-interact flavour of on-the-fly KB
encoding. It carries the same Claude-Agent-SDK plumbing as `claude_sdk_otf`
plus a native ``ask_user`` tool and three guards:

    1. A PreToolUse gate on ``submit_query`` that DENIES until ``ask_user``
       has been called at least once.
    2. A PostToolUse counter on ``ask_user`` that increments the count.
    3. A PostToolUse nag on every tool that fires every 10 calls when no
       ``ask_user`` has happened yet.

The factory MUST be invoked inside ``run_task`` (per-task), not on the
agent constructor, because a single agent instance is reused across
concurrent tasks via ``make_runner``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Construction + reasoning-effort
# ---------------------------------------------------------------------------

def test_init_rejects_non_on_the_fly():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfAInteractAgent(slayer_setup="pre-encoded")


def test_init_accepts_on_the_fly_default():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.slayer_setup == "on-the-fly"


def test_init_rejects_bad_reasoning_effort():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfAInteractAgent(reasoning_effort="turbo")


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def _tool_names(tools):
    return {t.name for t in tools}


def test_select_tools_a_interact_returns_five_native_tools():
    """4 knowledge tools + submit_query + ask_user = 5 native; 11 slayer = 16."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    names = _tool_names(m._select_tools("a-interact"))
    assert names == {
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
        "submit_query",
        "ask_user",
    }
    assert len(m._select_tools("a-interact")) == 5


def test_select_tools_rejects_unknown_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._select_tools(bad)


def test_slayer_tool_names_include_write_tools():
    """The ainteract OTF agent must also expose SLayer write tools."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    names = set(m._slayer_tool_names())
    for t in (
        "mcp__slayer__create_model",
        "mcp__slayer__edit_model",
        "mcp__slayer__save_memory",
        "mcp__slayer__query_nested",
        "mcp__slayer__validate_models",
    ):
        assert t in names, f"missing slayer write tool {t}"
    for t in (
        "mcp__slayer__query",
        "mcp__slayer__search",
        "mcp__slayer__inspect_model",
        "mcp__slayer__help",
        "mcp__slayer__list_datasources",
        "mcp__slayer__models_summary",
    ):
        assert t in names
    assert len(names) == 11


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

def test_build_prompt_is_ainteract_variant():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    td = {"amb_user_query": "how many widgets?", "selected_database": "shop"}
    prompt = m._build_prompt("a-interact", td, budget=20.0)
    assert "how many widgets?" in prompt
    assert "shop" in prompt
    assert "ask_user" in prompt.lower()
    assert "submit_query" in prompt.lower()


def test_build_prompt_rejects_unknown_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    td = {"amb_user_query": "?", "selected_database": "shop"}
    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._build_prompt(bad, td, budget=20.0)


def test_prompt_rule_zero_precedes_encoding():
    """Rule 0 (the ask_user-before-encoding instruction) must appear BEFORE the
    encoding workflow language so it's read first. Asserted on substring offset
    rather than an exact index — the contract is ordering, not position."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import prompts as p

    text = p.SLAYER_OTF_AINTERACT
    ask_offset = text.lower().find("ask_user")
    encode_marker = text.lower().find("create_model")
    if encode_marker == -1:
        encode_marker = text.lower().find("encode")
    assert ask_offset != -1, "prompt must mention ask_user"
    assert encode_marker != -1, "prompt must mention the encoding workflow"
    assert ask_offset < encode_marker, (
        "Rule 0 (ask_user) must precede the encoding workflow text"
    )


def test_prompt_has_submit_gate_warning():
    """The prompt must warn that the submit gate refuses until ask_user is
    called — sets reader expectations so the agent doesn't view the deny as a
    bug."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import prompts as p

    text = p.SLAYER_OTF_AINTERACT.lower()
    assert "submit" in text
    # Either "refuse" or "deny" or "block" wording is acceptable.
    assert any(w in text for w in ("refuse", "deny", "block")), text


def test_prompts_use_synthetic_examples_only():
    """Guards `feedback_prompts_synthetic_examples_only`: no real eval-set
    DB / table / column / value names may appear in the prompt."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import prompts as p

    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band",
        "taguatinga",
    ]
    low = p.SLAYER_OTF_AINTERACT.lower()
    for name in banned:
        assert name not in low, f"real eval-set name {name!r} leaked into prompt"


# ---------------------------------------------------------------------------
# Hook factory — pre-submit gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_submit_gate_denies_when_ask_count_zero():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    pre_gate, _counter, _nag = m._make_ask_user_guards()
    out = await pre_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    # Codex MED#1: pin the EXACT dict shape per the SDK's
    # PreToolUseHookSpecificOutput TypedDict.
    assert set(out) == {"hookSpecificOutput"}
    hso = out["hookSpecificOutput"]
    assert set(hso) == {
        "hookEventName", "permissionDecision", "permissionDecisionReason",
    }
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    reason = hso["permissionDecisionReason"]
    assert "ask_user" in reason
    assert "user-sim" in reason or "user simulator" in reason.lower()
    assert "operationalisation" in reason or "operationalization" in reason


@pytest.mark.asyncio
async def test_pre_submit_gate_allows_when_ask_count_positive():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    pre_gate, counter, _nag = m._make_ask_user_guards()
    # Simulate one ask_user PostToolUse.
    await counter(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    out = await pre_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Hook factory — post nag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_nag_quiet_in_first_nine_calls():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for _ in range(9):
        out = await nag({"tool_name": "mcp__slayer__query"}, None, None)
        assert out == {}


@pytest.mark.asyncio
async def test_post_nag_fires_at_ten_with_count():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for _ in range(9):
        await nag({"tool_name": "mcp__slayer__query"}, None, None)
    out = await nag({"tool_name": "mcp__slayer__query"}, None, None)
    # Codex MED#1: pin the EXACT dict shape per the SDK's
    # PostToolUseHookSpecificOutput TypedDict.
    assert set(out) == {"hookSpecificOutput"}
    hso = out["hookSpecificOutput"]
    assert set(hso) == {"hookEventName", "additionalContext"}
    assert hso["hookEventName"] == "PostToolUse"
    ctx = hso["additionalContext"]
    assert "10" in ctx
    assert "user-sim" in ctx or "user simulator" in ctx.lower()


@pytest.mark.asyncio
async def test_post_nag_fires_at_twenty():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for i in range(20):
        out = await nag({"tool_name": "mcp__slayer__query"}, None, None)
        if i == 9:
            assert "10" in out["hookSpecificOutput"]["additionalContext"]
        elif i == 19:
            assert "20" in out["hookSpecificOutput"]["additionalContext"]
        else:
            assert out == {}


@pytest.mark.asyncio
async def test_post_nag_silent_after_ask_user():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    _gate, counter, nag = m._make_ask_user_guards()
    # First tool call IS ask_user — increment counter, fire nag for the same
    # call (modeling SDK ordering); nag must NOT fire because tool is ask_user.
    await counter(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    out = await nag(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    assert out == {}
    # All subsequent calls — including the would-be 10th and 20th — must be
    # silent now that ask_count > 0.
    for _ in range(30):
        out = await nag({"tool_name": "mcp__slayer__query"}, None, None)
        assert out == {}


@pytest.mark.asyncio
async def test_post_nag_silent_when_tenth_call_is_ask_user():
    """Race regression: if the 10th tool call IS ask_user and the SDK fires
    nag BEFORE the counter increments, the nag would emit a false positive
    nag mentioning the ask the user just made. Skip nag when the current tool
    is ask_user."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for _ in range(9):
        await nag({"tool_name": "mcp__slayer__query"}, None, None)
    # 10th tool call is ask_user; nag must skip.
    out = await nag(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Per-task state isolation — the factory MUST be invoked inside run_task.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_isolation_across_factories():
    """Build two independent guard sets and interleave calls; counters stay
    independent. This regression protects against a future refactor that
    stores hook state on the agent instance instead of in a per-task closure."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    gate_a, counter_a, nag_a = m._make_ask_user_guards()
    gate_b, counter_b, nag_b = m._make_ask_user_guards()

    # Five tool calls on A; ask_user on B.
    for _ in range(5):
        await nag_a({"tool_name": "mcp__slayer__query"}, None, None)
    await counter_b(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )

    # B's submit gate now allows; A's still denies (ask_count_a == 0).
    out_a = await gate_a(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    out_b = await gate_b(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out_a["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out_b == {}


# ---------------------------------------------------------------------------
# run_task gating
# ---------------------------------------------------------------------------

_TASK = {
    "selected_database": "shop",
    "instance_id": "shop_1",
    "amb_user_query": "?",
    "knowledge_ambiguity": [],
}


@pytest.mark.asyncio
async def test_run_task_rejects_raw_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(
            dict(_TASK), "/tmp", 20.0, "raw", eval_mode="a-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_unsupported_eval_modes():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            await agent.run_task(
                dict(_TASK), "/tmp", 20.0, "slayer", eval_mode=bad,
            )


@pytest.mark.asyncio
async def test_run_task_accepts_mini_interact_alias():
    """Codex (PR #10 follow-up): the agent-level dataset gate must accept
    the documented ``mini-interact`` (hyphen) alias, matching the
    validator's canonicalization behavior. Without this, a programmatic
    `run_one_task` call that passes the alias would pass the validator
    but be silently turned into a failed result row at the agent layer."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="openai/gpt-4o")
    td = dict(_TASK, dataset="mini-interact")  # alias, not canonical
    # Non-anthropic model short-circuits to a skip-shaped row; key signal
    # is that we get *past* the dataset gate (no ValueError raised).
    row = await agent.run_task(
        td, "/tmp", 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


@pytest.mark.asyncio
async def test_run_task_rejects_livesqlbench_dataset():
    """ainteract is bound to mini_interact at the agent layer too —
    a programmatic caller (e.g. `make_runner`, which has no dataset arg)
    cannot bypass the CLI gate by passing task_data with the wrong dataset."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="livesqlbench")
    with pytest.raises(ValueError):
        await agent.run_task(td, "/tmp", 20.0, "slayer", eval_mode="a-interact")


@pytest.mark.asyncio
async def test_run_task_non_anthropic_model_skips():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="openai/gpt-4o")
    row = await agent.run_task(
        dict(_TASK), "/tmp", 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


# ---------------------------------------------------------------------------
# Storage path + ClaudeAgentOptions
# ---------------------------------------------------------------------------

class _FakeAssistant:
    def __init__(self, in_, out_, cache=0):
        self.usage = SimpleNamespace(
            input_tokens=in_, output_tokens=out_, cache_read_input_tokens=cache,
        )


_FakeAssistant.__name__ = "AssistantMessage"


def _make_fake_client(captured: dict, messages):
    class _FakeClient:
        def __init__(self, options):
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def query(self, *a, **kw):
            return None

        async def receive_response(self):
            for m in messages:
                yield m

    return _FakeClient


def _stub_env(monkeypatch, m, storage_dir, *, messages=(), captured=None, deleted=()):
    from bird_interact_agents import usage as usage_mod

    captured = captured if captured is not None else {}
    captured.setdefault("materialize_calls", 0)
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(m, "load_db_data_if_needed", lambda *a, **kw: None)

    def _fake_materialize(*a, **kw):
        captured["materialize_calls"] += 1
        return None

    monkeypatch.setattr(m, "materialize_task_db", _fake_materialize)
    monkeypatch.setattr(
        m, "slayer_mcp_stdio_config",
        lambda d: {"command": "slayer", "args": ["mcp"], "env": {}},
    )
    monkeypatch.setattr(m, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())

    async def fake_resolve(*, db_name, task_data, data_path_base, benchmark):
        captured["resolve_kwargs"] = {
            "db_name": db_name, "benchmark": benchmark,
            "data_path_base": data_path_base,
        }
        return str(storage_dir), list(deleted)

    monkeypatch.setattr(m, "resolve_otf_task_storage_dir", fake_resolve)
    monkeypatch.setattr(m, "ClaudeSDKClient", _make_fake_client(captured, messages))
    return captured


@pytest.mark.asyncio
async def test_run_task_uses_cache_resolver_with_mini_interact_benchmark(
    monkeypatch, tmp_path,
):
    """ainteract resolves per-task storage from the deterministic cache scoped
    to mini_interact, never the livesqlbench root."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    # The committed-models resolver must not even be imported.
    assert not hasattr(m, "resolve_task_storage_dir")

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["resolve_kwargs"]["benchmark"] == "mini_interact"


@pytest.mark.asyncio
async def test_run_task_attaches_slayer_write_tools(monkeypatch, tmp_path):
    """The ClaudeAgentOptions handed to the SDK must whitelist the slayer
    write tools so the agent can encode."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__slayer__create_model" in allowed
    assert "mcp__slayer__edit_model" in allowed
    assert "mcp__slayer__validate_models" in allowed


@pytest.mark.asyncio
async def test_run_task_whitelists_ask_user_and_submit_query(monkeypatch, tmp_path):
    """Both native tools that drive the ask-then-submit discipline must be on
    the allow-list."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__ask_user" in allowed
    assert "mcp__bird-interact-tools__submit_query" in allowed


@pytest.mark.asyncio
async def test_run_task_registers_three_guards_plus_turn_budget(
    monkeypatch, tmp_path,
):
    """Hook registration: PreToolUse has the submit-gate scoped to
    submit_query; PostToolUse carries the ask-counter, the nag, AND the
    existing turn-budget hook."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    hooks = captured["options"].hooks
    assert "PreToolUse" in hooks
    assert "PostToolUse" in hooks

    pre_matchers = hooks["PreToolUse"]
    # Exactly one PreToolUse matcher, scoped to submit_query.
    assert len(pre_matchers) == 1
    assert pre_matchers[0].matcher == "mcp__bird-interact-tools__submit_query"

    post_matchers = hooks["PostToolUse"]
    # Exactly three PostToolUse matchers: ask-counter (matcher == ask_user),
    # nag (matcher None), turn-budget (matcher None).
    assert len(post_matchers) == 3
    matchers = {pm.matcher for pm in post_matchers}
    assert "mcp__bird-interact-tools__ask_user" in matchers
    assert None in matchers


@pytest.mark.asyncio
async def test_run_task_registered_hooks_behave_correctly(monkeypatch, tmp_path):
    """Codex MED#2: not only count and matcher names — actually invoke the
    captured hook callables and prove (a) the submit-gate denies before
    ask_user has been called, (b) the ask-counter flips the gate to allow,
    (c) the nag fires on the 10th non-ask call when ask_count == 0."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    hooks = captured["options"].hooks

    pre_submit = hooks["PreToolUse"][0].hooks[0]
    post_by_matcher = {pm.matcher: pm.hooks[0] for pm in hooks["PostToolUse"]}
    counter = post_by_matcher["mcp__bird-interact-tools__ask_user"]
    # The two matcher=None hooks: one is the nag, one is the turn-budget. We
    # can't tell them apart by matcher alone — call BOTH and pick the one
    # that behaves as the nag. If neither fires for a slayer tool by call 10,
    # the nag wasn't registered.
    matcher_none_hooks = [
        pm.hooks[0] for pm in hooks["PostToolUse"] if pm.matcher is None
    ]
    assert len(matcher_none_hooks) == 2  # nag + turn-budget

    # (a) Pre-submit denies before any ask.
    out = await pre_submit(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    # (b) ask-counter increments; pre-submit now allows.
    await counter(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    out = await pre_submit(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out == {}

    # (c) Reset for an isolated nag test: a FRESH factory has ask_count==0.
    pre_gate2, _counter2, nag2 = m._make_ask_user_guards()
    for _ in range(9):
        out = await nag2({"tool_name": "mcp__slayer__query"}, None, None)
        assert out == {}
    out = await nag2({"tool_name": "mcp__slayer__query"}, None, None)
    assert "additionalContext" in out["hookSpecificOutput"]


@pytest.mark.asyncio
async def test_run_task_invokes_factory_per_call(monkeypatch, tmp_path):
    """Codex HIGH#1: the hook-state factory MUST be invoked inside run_task
    (per task), not on the agent constructor. Running the SAME agent
    instance through run_task twice must produce two independent guard
    states (regression for the per-task closure contract)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    call_count = [0]
    real_factory = m._make_ask_user_guards

    def _spy_factory():
        call_count[0] += 1
        return real_factory()

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    monkeypatch.setattr(m, "_make_ask_user_guards", _spy_factory)

    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert call_count[0] == 1
    # First-run hooks: simulate 5 prior tool calls and one ask_user, then
    # clear captured options so we can re-capture for the second run.
    first_post = captured["options"].hooks["PostToolUse"]
    first_counter = next(
        pm.hooks[0] for pm in first_post
        if pm.matcher == "mcp__bird-interact-tools__ask_user"
    )
    await first_counter(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )

    # Second invocation of the SAME agent — factory must be called again.
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert call_count[0] == 2, (
        "factory must be invoked per run_task call — state leaked across tasks"
    )
    # Second-run pre-submit must DENY because ask_count starts at 0, even
    # though we incremented the first run's counter.
    second_pre = captured["options"].hooks["PreToolUse"][0].hooks[0]
    out = await second_pre(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_run_task_accepts_explicit_mini_interact_dataset(
    monkeypatch, tmp_path,
):
    """Codex LOW#2: explicit positive case — task with an explicit
    ``dataset='mini_interact'`` marker is accepted at the agent layer
    (loader-stamped marker, not just defaulted)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="mini_interact")
    row = await agent.run_task(
        td, str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    # No exception; row has been finalised.
    assert row.get("instance_id") == "shop_1"


@pytest.mark.asyncio
async def test_run_task_restricts_tools_and_caps_turns(monkeypatch, tmp_path):
    """No Claude Code built-ins / ToolSearch; isolated settings; native
    max_turns at 2x the base."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m
    from bird_interact_agents.harness import MAX_MODEL_TURNS

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    opts = captured["options"]
    assert opts.tools == []
    assert opts.setting_sources == []
    assert opts.max_turns == 2 * MAX_MODEL_TURNS


@pytest.mark.asyncio
async def test_run_task_pins_requested_model(monkeypatch, tmp_path):
    """--agent-model must reach the SDK as the bare native model id."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-opus-4-7")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_run_task_passes_reasoning_effort(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(
        model="anthropic/claude-sonnet-4-5", reasoning_effort="high",
    )
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].effort == "high"


@pytest.mark.asyncio
async def test_run_task_default_effort_is_none(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].effort is None


@pytest.mark.asyncio
async def test_run_task_captures_usage(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m
    from bird_interact_agents import usage as usage_mod

    msgs = [_FakeAssistant(100, 20), _FakeAssistant(150, 30, cache=5)]
    _stub_env(monkeypatch, m, tmp_path / "store", messages=msgs)
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    rebuilt = usage_mod.TokenUsage.model_validate(row["usage"])
    assert rebuilt.prompt_tokens == 250
    assert rebuilt.completion_tokens == 50
    assert rebuilt.cache_read_tokens == 5


# ---------------------------------------------------------------------------
# Import isolation (mirror of the claude_sdk_otf test) — the new flavor must
# not drag in heavy pydantic_ai adapter packages either.
# ---------------------------------------------------------------------------

def test_import_does_not_pull_pydantic_ai_adapter_packages():
    """Run in a CLEAN interpreter so we don't mutate sys.modules in this
    process (deleting already-imported adapter modules would corrupt class
    identity for other tests in the same session)."""
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import importlib, sys
        importlib.import_module(
            "bird_interact_agents.agents.claude_sdk_otf_ainteract.agent"
        )
        leaked = [
            n for n in sys.modules
            if n.startswith("bird_interact_agents.agents.pydantic_ai")
        ]
        if leaked:
            print("LEAKED:" + ",".join(sorted(leaked)))
            sys.exit(1)
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "claude_sdk_otf_ainteract import leaked pydantic_ai ADAPTER packages: "
        f"{r.stdout.strip()}{r.stderr.strip()}"
    )
