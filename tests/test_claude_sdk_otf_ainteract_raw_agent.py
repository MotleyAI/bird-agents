"""Unit tests for the `claude_sdk_otf_ainteract_raw` agent (no SLayer).

This is the raw-SQL counterpart to `claude_sdk_otf_ainteract` (which uses
SLayer). It is mini-interact / a-interact only, uses `submit_sql` (not
`submit_query`) gated behind a mandatory `ask_user` call, and has no SLayer
MCP server. The ask-user guards are a parallel factory to the slayer variant
but reference `submit_sql` as the gated tool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_init_accepts_default():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    agent = ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.model == "anthropic/claude-sonnet-4-5"


def test_init_rejects_bad_reasoning_effort():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfAInteractRawAgent(reasoning_effort="turbo")


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def _tool_names(tools):
    return {t.name for t in tools}


def test_select_tools_a_interact_returns_nine_native_tools():
    """7 BIRD_INTERACT_TOOLS + submit_sql + ask_user = 9 native tools.
    No SLayer tools."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    names = _tool_names(m._select_tools("a-interact"))
    assert names == {
        "execute_sql",
        "get_schema",
        "get_all_column_meanings",
        "get_column_meaning",
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
        "submit_sql",
        "ask_user",
    }
    assert "submit_query" not in names
    assert len(names) == 9


def test_select_tools_rejects_unknown_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._select_tools(bad)


def test_no_slayer_tool_names_function():
    """Raw agent must NOT have a _slayer_tool_names function."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    assert not hasattr(m, "_slayer_tool_names")


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

def test_build_prompt_is_ainteract_variant():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    td = {"amb_user_query": "how many widgets?", "selected_database": "shop"}
    prompt = m._build_prompt("a-interact", td, budget=20.0)
    assert "how many widgets?" in prompt
    assert "shop" in prompt
    assert "ask_user" in prompt.lower()
    # submit_sql, not submit_query
    assert "submit_sql" in prompt
    assert "submit_query" not in prompt


def test_build_prompt_rejects_unknown_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    td = {"amb_user_query": "?", "selected_database": "shop"}
    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._build_prompt(bad, td, budget=20.0)


def test_prompt_rule_zero_precedes_ask_user_before_sql():
    """Rule 0 (ask_user before SQL) must appear BEFORE the SQL-writing workflow."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import prompts as p

    text = p.RAW_OTF_AINTERACT
    ask_offset = text.lower().find("ask_user")
    sql_offset = text.lower().find("execute_sql")
    if sql_offset == -1:
        sql_offset = text.lower().find("select ")
    assert ask_offset != -1, "prompt must mention ask_user"
    assert sql_offset != -1, "prompt must mention SQL-writing guidance"
    assert ask_offset < sql_offset, (
        "Rule 0 (ask_user) must precede the SQL-writing guidance"
    )


def test_prompt_has_submit_gate_warning():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import prompts as p

    text = p.RAW_OTF_AINTERACT.lower()
    assert "submit" in text
    assert any(w in text for w in ("refuse", "deny", "block")), text


def test_prompt_absent_slayer_vocab():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import prompts as p

    text = p.RAW_OTF_AINTERACT
    for term in ("submit_query", "create_model", "edit_model", "[kb=", "mcp__slayer__"):
        assert term not in text, (
            f"SLayer term {term!r} leaked into raw ainteract prompt"
        )


def test_prompts_use_synthetic_examples_only():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import prompts as p

    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band",
        "taguatinga",
    ]
    low = p.RAW_OTF_AINTERACT.lower()
    for name in banned:
        assert name not in low, f"real eval-set name {name!r} leaked into raw prompt"


# ---------------------------------------------------------------------------
# Hook factory — pre-submit gate (gated on submit_sql, not submit_query)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_submit_gate_denies_when_ask_count_zero():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    pre_gate, _counter, _nag = m._make_ask_user_guards()
    out = await pre_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_sql"}, None, None,
    )
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
async def test_pre_submit_gate_does_not_deny_submit_query():
    """The gate is scoped to submit_sql only — submit_query (if ever called)
    must not be denied (it's not the raw submission tool)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    pre_gate, _counter, _nag = m._make_ask_user_guards()
    out = await pre_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    # submit_query is not the guarded tool — gate returns {} (allow / no-op)
    assert out == {}


@pytest.mark.asyncio
async def test_pre_submit_gate_allows_when_ask_count_positive():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    pre_gate, counter, _nag = m._make_ask_user_guards()
    await counter(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    out = await pre_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_sql"}, None, None,
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Hook factory — post nag (same semantics as slayer variant)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_nag_quiet_in_first_nine_calls():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for _ in range(9):
        out = await nag({"tool_name": "execute_sql"}, None, None)
        assert out == {}


@pytest.mark.asyncio
async def test_post_nag_fires_at_ten():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for _ in range(9):
        await nag({"tool_name": "execute_sql"}, None, None)
    out = await nag({"tool_name": "execute_sql"}, None, None)
    assert set(out) == {"hookSpecificOutput"}
    hso = out["hookSpecificOutput"]
    assert set(hso) == {"hookEventName", "additionalContext"}
    assert hso["hookEventName"] == "PostToolUse"
    ctx = hso["additionalContext"]
    assert "10" in ctx
    assert "user-sim" in ctx or "user simulator" in ctx.lower()


@pytest.mark.asyncio
async def test_post_nag_silent_after_ask_user():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _gate, counter, nag = m._make_ask_user_guards()
    await counter({"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None)
    out = await nag({"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None)
    assert out == {}
    for _ in range(30):
        out = await nag({"tool_name": "execute_sql"}, None, None)
        assert out == {}


@pytest.mark.asyncio
async def test_state_isolation_across_factories():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    gate_a, counter_a, nag_a = m._make_ask_user_guards()
    gate_b, counter_b, nag_b = m._make_ask_user_guards()

    for _ in range(5):
        await nag_a({"tool_name": "execute_sql"}, None, None)
    await counter_b({"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None)

    out_a = await gate_a(
        {"tool_name": "mcp__bird-interact-tools__submit_sql"}, None, None,
    )
    out_b = await gate_b(
        {"tool_name": "mcp__bird-interact-tools__submit_sql"}, None, None,
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
    "dataset": "mini-interact",
}


@pytest.mark.asyncio
async def test_run_task_rejects_slayer_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    agent = ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(
            dict(_TASK), "/tmp", 20.0, "slayer", eval_mode="a-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_unsupported_eval_modes():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    agent = ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            await agent.run_task(
                dict(_TASK), "/tmp", 20.0, "raw", eval_mode=bad,
            )


@pytest.mark.asyncio
async def test_run_task_rejects_livesqlbench_dataset():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    agent = ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="livesqlbench-base-lite-sqlite")
    with pytest.raises(ValueError):
        await agent.run_task(td, "/tmp", 20.0, "raw", eval_mode="a-interact")


@pytest.mark.asyncio
async def test_run_task_accepts_mini_interact_alias():
    """Agent-level dataset gate accepts the mini-interact alias."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    agent = ClaudeSDKOtfAInteractRawAgent(model="openai/gpt-4o")
    td = dict(_TASK, dataset="mini-interact")
    row = await agent.run_task(td, "/tmp", 20.0, "raw", eval_mode="a-interact")
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


@pytest.mark.asyncio
async def test_run_task_non_anthropic_model_skips():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.agent import (
        ClaudeSDKOtfAInteractRawAgent,
    )

    agent = ClaudeSDKOtfAInteractRawAgent(model="openai/gpt-4o")
    row = await agent.run_task(
        dict(_TASK), "/tmp", 20.0, "raw", eval_mode="a-interact",
    )
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


# ---------------------------------------------------------------------------
# FakeAssistant / _stub_env helpers
# ---------------------------------------------------------------------------

class _FakeAssistant:
    def __init__(self, in_, out_, cache=0):
        self.usage = SimpleNamespace(
            input_tokens=in_, output_tokens=out_, cache_read_input_tokens=cache,
        )


_FakeAssistant.__name__ = "AssistantMessage"


def _make_fake_client(
    captured: dict, messages,
    *, m_module=None,
    prefill_result=None, prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
    prefill_asks: int = 0,
):
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
            if prefill_result is not None and prefill_timing == "before":
                m_module._ctx_var.get()["result"] = dict(prefill_result)
            for msg in messages:
                yield msg
            if prefill_asks:
                m_module._ctx_var.get()["asks_used"] = prefill_asks
            if prefill_result is not None and prefill_timing == "after":
                m_module._ctx_var.get()["result"] = dict(prefill_result)
            if raise_after_prefill is not None:
                raise raise_after_prefill

    return _FakeClient


def _stub_env(
    monkeypatch, m, storage_dir,
    *,
    messages=(), captured=None,
    prefill_result=None, prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
    prefill_asks: int = 0,
):
    from bird_interact_agents import usage as usage_mod

    captured = captured if captured is not None else {}
    captured.setdefault("materialize_calls", 0)
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(m, "load_db_data_if_needed", lambda *a, **kw: None)

    def _fake_materialize(*a, **kw):
        captured["materialize_calls"] += 1

    monkeypatch.setattr(m, "materialize_task_db", _fake_materialize)
    monkeypatch.setattr(m, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(
        m, "ClaudeSDKClient",
        _make_fake_client(
            captured, messages,
            m_module=m,
            prefill_result=prefill_result,
            prefill_timing=prefill_timing,
            raise_after_prefill=raise_after_prefill,
            prefill_asks=prefill_asks,
        ),
    )
    return captured


# ---------------------------------------------------------------------------
# Storage path + ClaudeAgentOptions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_does_not_call_slayer_mcp(monkeypatch, tmp_path):
    """Raw ainteract agent has no SLayer MCP server."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    assert not hasattr(m, "slayer_mcp_stdio_config"), (
        "raw ainteract agent must not import slayer_mcp_stdio_config"
    )
    assert not hasattr(m, "resolve_otf_task_storage_dir"), (
        "raw ainteract agent must not import resolve_otf_task_storage_dir"
    )


@pytest.mark.asyncio
async def test_run_task_does_not_whitelist_slayer_tools(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert not any(t.startswith("mcp__slayer__") for t in allowed)


@pytest.mark.asyncio
async def test_run_task_whitelists_ask_user_and_submit_sql(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__ask_user" in allowed
    assert "mcp__bird-interact-tools__submit_sql" in allowed
    assert "mcp__bird-interact-tools__submit_query" not in allowed


@pytest.mark.asyncio
async def test_run_task_registers_three_guards_plus_turn_budget(monkeypatch, tmp_path):
    """PreToolUse gate scoped to submit_sql; PostToolUse: ask-counter,
    nag, turn-budget."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    hooks = captured["options"].hooks
    assert "PreToolUse" in hooks
    assert "PostToolUse" in hooks

    pre_matchers = hooks["PreToolUse"]
    # [0] submit_sql gate, [1] discovery-only partition deny (DEV-1555).
    assert len(pre_matchers) == 2
    # Gate must be scoped to submit_sql, not submit_query.
    assert pre_matchers[0].matcher == "mcp__bird-interact-tools__submit_sql"

    post_matchers = hooks["PostToolUse"]
    # ask-counter, nag, turn-budget, context-budget (DEV-1555).
    assert len(post_matchers) == 4
    matchers = {pm.matcher for pm in post_matchers}
    assert "mcp__bird-interact-tools__ask_user" in matchers
    assert None in matchers


@pytest.mark.asyncio
async def test_run_task_restricts_tools_and_caps_turns(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m
    from bird_interact_agents.harness import MAX_MODEL_TURNS

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    opts = captured["options"]
    assert opts.tools == ["Task"]  # DEV-1555: only built-in re-enabled
    assert opts.setting_sources == []
    assert opts.max_turns == 2 * MAX_MODEL_TURNS


@pytest.mark.asyncio
async def test_run_task_invokes_factory_per_call(monkeypatch, tmp_path):
    """Hook-state factory must be invoked inside run_task (per task),
    not on the agent constructor."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    call_count = [0]
    real_factory = m._make_ask_user_guards

    def _spy_factory():
        call_count[0] += 1
        return real_factory()

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    monkeypatch.setattr(m, "_make_ask_user_guards", _spy_factory)

    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert call_count[0] == 1

    # Second invocation — factory must be called again, state fresh.
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert call_count[0] == 2, (
        "factory must be invoked per run_task call — state leaked across tasks"
    )
    second_pre = captured["options"].hooks["PreToolUse"][0].hooks[0]
    out = await second_pre(
        {"tool_name": "mcp__bird-interact-tools__submit_sql"}, None, None,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_run_task_pins_requested_model(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-opus-4-7")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert captured["options"].model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_run_task_passes_reasoning_effort(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(
        model="anthropic/claude-sonnet-4-5", reasoning_effort="high",
    )
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert captured["options"].effort == "high"


@pytest.mark.asyncio
async def test_run_task_default_effort_is_none(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert captured["options"].effort is None


@pytest.mark.asyncio
async def test_run_task_captures_usage(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m
    from bird_interact_agents import usage as usage_mod

    msgs = [_FakeAssistant(100, 20), _FakeAssistant(150, 30, cache=5)]
    _stub_env(monkeypatch, m, tmp_path / "store", messages=msgs)
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    rebuilt = usage_mod.TokenUsage.model_validate(row["usage"])
    assert rebuilt.prompt_tokens == 250
    assert rebuilt.completion_tokens == 50
    assert rebuilt.cache_read_tokens == 5


# ---------------------------------------------------------------------------
# n_ask_user_calls reporting (DEV-1519 parity for raw ainteract)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_writes_n_ask_user_calls_zero(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_asks=0,
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert row["usage"]["n_ask_user_calls"] == 0


@pytest.mark.asyncio
async def test_run_task_writes_n_ask_user_calls_nonzero(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_asks=3,
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert row["usage"]["n_ask_user_calls"] == 3


@pytest.mark.asyncio
async def test_run_task_exception_path_writes_n_ask_user_calls(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(50, 10)],
        prefill_asks=2,
        raise_after_prefill=RuntimeError("boom"),
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert "boom" in (row.get("error") or "")
    assert row["usage"]["n_ask_user_calls"] == 2


# ---------------------------------------------------------------------------
# DEV-1511: diagnostic-field propagation
# ---------------------------------------------------------------------------

def _full_prefill(**overrides):
    base = {
        "submission_status": "submitted_ok",
        "predicted_result_json": "[{\"a\": 1}]",
        "gold_result_json": "[{\"a\": 1}]",
        "phase1_observation": "PASS",
        "phase1_passed": True,
        "phase2_passed": False,
        "total_reward": 1.0,
        "submitted_sql": "SELECT 1",
        "submitted_query": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_run_task_propagates_diagnostic_fields_on_happy_path(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=_full_prefill(),
        prefill_timing="after",
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    assert "phase2_observation" in row
    assert row["phase2_observation"] is None
    assert row["phase1_passed"] is True
    assert row["submitted_sql"] == "SELECT 1"
    assert row["error"] is None


@pytest.mark.asyncio
async def test_run_task_propagation_defaults_to_none_when_never_submitted(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert row["submission_status"] is None
    assert row["predicted_result_json"] is None
    assert row["phase1_observation"] is None
    assert row["phase2_observation"] is None


@pytest.mark.asyncio
async def test_run_task_exception_path_propagates_partial_result(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    prefill = _full_prefill(
        phase2_passed=True, total_reward=0.75, phase2_observation="p2 ok",
    )
    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=prefill,
        prefill_timing="after",
        raise_after_prefill=RuntimeError("boom"),
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert row["error"] == "boom"
    assert row["submission_status"] == "submitted_ok"
    assert row["phase1_passed"] is True
    assert row["phase2_observation"] == "p2 ok"
    assert row["submitted_sql"] == "SELECT 1"


@pytest.mark.asyncio
async def test_run_task_exception_before_ctx_set_yields_empty_diagnostics(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")

    def _boom(*a, **kw):
        raise RuntimeError("early-setup boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert "early-setup boom" in (row.get("error") or "")
    assert row["submission_status"] is None
    assert row["phase1_passed"] is False


@pytest.mark.asyncio
async def test_run_task_exception_path_isolated_from_stale_context(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")
    m._ctx_var.set({
        "result": {
            "submission_status": "STALE_SHOULD_NOT_LEAK",
            "phase1_passed": True,
            "predicted_result_json": "STALE",
        },
    })

    def _boom(*a, **kw):
        raise RuntimeError("early boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    assert "early boom" in (row.get("error") or "")
    assert row["submission_status"] != "STALE_SHOULD_NOT_LEAK"
    assert row["submission_status"] is None
    assert row["phase1_passed"] is False
