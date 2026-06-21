"""Unit tests for the `claude_sdk_otf_raw` agent (no LLM, no SLayer).

This is the raw-SQL counterpart to `claude_sdk_otf` (which uses SLayer).
It is livesqlbench / one-shot only, uses `submit_sql` (not `submit_query`),
and has no SLayer MCP server. The prompts share constants from
`_shared_otf_prompts` with the slayer OTF variant for a fair comparison.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_init_accepts_default():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import (
        ClaudeSDKOtfRawAgent,
    )

    agent = ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.model == "anthropic/claude-sonnet-4-5"


def test_init_rejects_bad_reasoning_effort():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import (
        ClaudeSDKOtfRawAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfRawAgent(reasoning_effort="turbo")


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def _tool_names(tools):
    return {t.name for t in tools}


def test_select_tools_one_shot_returns_eight_native_tools():
    """All 7 raw BIRD_INTERACT_TOOLS + submit_sql = 8 native tools.
    No ask_user, no SLayer tools."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    names = _tool_names(m._select_tools("one-shot"))
    assert names == {
        "execute_sql",
        "get_schema",
        "get_all_column_meanings",
        "get_column_meaning",
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
        "submit_sql",
    }
    assert "ask_user" not in names
    assert "submit_query" not in names
    assert len(names) == 8


def test_select_tools_rejects_non_one_shot():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._select_tools(bad)


def test_no_slayer_tool_names_function():
    """Raw agent must NOT have a _slayer_tool_names function
    (there is no SLayer MCP server)."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    assert not hasattr(m, "_slayer_tool_names")


# ---------------------------------------------------------------------------
# Prompt selection + hygiene
# ---------------------------------------------------------------------------

def test_build_prompt_uses_raw_one_shot_template():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import prompts as p

    td = {"amb_user_query": "how many widgets?", "selected_database": "shop"}
    out = m._build_prompt("one-shot", td, budget=20.0)
    assert "how many widgets?" in out
    assert "shop" in out
    # No ask_user in one-shot.
    assert "ask_user" not in out.lower()
    # submit_sql, not submit_query.
    assert "submit_sql" in out
    assert "submit_query" not in out
    assert p.RAW_OTF_ONE_SHOT


def test_build_prompt_rejects_non_one_shot():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    td = {"amb_user_query": "?", "selected_database": "shop"}
    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._build_prompt(bad, td, budget=20.0)


def test_prompt_absent_slayer_vocab():
    """Raw prompt must not mention SLayer-specific concepts."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import prompts as p

    text = p.RAW_OTF_ONE_SHOT
    for term in ("submit_query", "create_model", "edit_model", "[kb=", "mcp__slayer__"):
        assert term not in text, f"SLayer term {term!r} leaked into raw one-shot prompt"


def test_prompt_has_execute_sql_and_get_schema():
    """Raw prompt must mention the DB-exploration tools."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import prompts as p

    low = p.RAW_OTF_ONE_SHOT.lower()
    for tool_name in ("execute_sql", "get_schema"):
        assert tool_name in low or "schema" in low, (
            f"raw one-shot prompt should guide agent to use {tool_name}"
        )


def test_prompts_use_synthetic_examples_only():
    """Guards feedback_prompts_synthetic_examples_only."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import prompts as p

    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band",
        "taguatinga",
    ]
    low = p.RAW_OTF_ONE_SHOT.lower()
    for name in banned:
        assert name not in low, f"real eval-set name {name!r} leaked into raw prompt"


# ---------------------------------------------------------------------------
# run_task gating
# ---------------------------------------------------------------------------

_TASK = {
    "selected_database": "shop",
    "instance_id": "shop_1",
    "amb_user_query": "?",
    "knowledge_ambiguity": [],
    "dataset": "livesqlbench-base-lite-sqlite",
}


@pytest.mark.asyncio
async def test_run_task_rejects_slayer_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import ClaudeSDKOtfRawAgent

    agent = ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(dict(_TASK), "/tmp", 20.0, "slayer", eval_mode="one-shot")


@pytest.mark.asyncio
async def test_run_task_rejects_unsupported_eval_modes():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import ClaudeSDKOtfRawAgent

    agent = ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            await agent.run_task(dict(_TASK), "/tmp", 20.0, "raw", eval_mode=bad)


@pytest.mark.asyncio
async def test_run_task_rejects_mini_interact_dataset():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import ClaudeSDKOtfRawAgent

    agent = ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="mini-interact")
    with pytest.raises(ValueError):
        await agent.run_task(td, "/tmp", 20.0, "raw", eval_mode="one-shot")


@pytest.mark.asyncio
async def test_run_task_non_anthropic_model_skips():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import ClaudeSDKOtfRawAgent

    agent = ClaudeSDKOtfRawAgent(model="openai/gpt-4o")
    row = await agent.run_task(
        dict(_TASK), "/tmp", 20.0, "raw", eval_mode="one-shot",
    )
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


@pytest.mark.asyncio
async def test_run_task_accepts_livesqlbench_alias():
    """Agent-level dataset gate must accept the canonical token."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import ClaudeSDKOtfRawAgent

    agent = ClaudeSDKOtfRawAgent(model="openai/gpt-4o")
    td = dict(_TASK, dataset="livesqlbench-base-lite-sqlite")
    row = await agent.run_task(td, "/tmp", 20.0, "raw", eval_mode="one-shot")
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
    captured: dict,
    messages,
    *,
    m_module,
    prefill_result=None,
    prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
):
    class _FakeClient:
        def __init__(self, options):
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get_mcp_status(self):
            names = list((captured["options"].mcp_servers or {}).keys())
            return {"mcpServers": [
                {"name": n, "status": "connected"} for n in names
            ]}

        async def query(self, *a, **kw):
            return None

        async def receive_response(self):
            if prefill_result is not None and prefill_timing == "before":
                m_module._ctx_var.get()["result"] = dict(prefill_result)
            for msg in messages:
                yield msg
            if prefill_result is not None and prefill_timing == "after":
                m_module._ctx_var.get()["result"] = dict(prefill_result)
            if raise_after_prefill is not None:
                raise raise_after_prefill

    return _FakeClient


def _stub_env(
    monkeypatch, m, storage_dir,
    *,
    messages=(), captured=None,
    prefill_result=None,
    prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
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
    from bird_interact_agents.agents.claude_sdk import sdk_env as _sdk_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stub")
    monkeypatch.setattr(
        _sdk_env, "ClaudeSDKClient",
        _make_fake_client(
            captured, messages,
            m_module=m,
            prefill_result=prefill_result,
            prefill_timing=prefill_timing,
            raise_after_prefill=raise_after_prefill,
        ),
    )
    return captured


# ---------------------------------------------------------------------------
# Storage path + allowed tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_does_not_call_slayer_mcp(monkeypatch, tmp_path):
    """Raw agent has no SLayer MCP server — `slayer_mcp_stdio_config` must
    not be imported or called."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    assert not hasattr(m, "slayer_mcp_stdio_config"), (
        "raw agent must not import slayer_mcp_stdio_config"
    )
    assert not hasattr(m, "resolve_otf_task_storage_dir"), (
        "raw agent must not import resolve_otf_task_storage_dir"
    )


@pytest.mark.asyncio
async def test_run_task_does_not_whitelist_slayer_tools(monkeypatch, tmp_path):
    """The ClaudeAgentOptions must not whitelist any mcp__slayer__* tool."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    allowed = set(captured["options"].allowed_tools)
    assert not any(t.startswith("mcp__slayer__") for t in allowed), (
        "raw agent must not whitelist any mcp__slayer__* tool"
    )


@pytest.mark.asyncio
async def test_run_task_does_not_disallow_slayer_tools(monkeypatch, tmp_path):
    """DEV-1548 plan: the raw adapter is OUT OF SCOPE — it exposes no
    SLayer tools, so threading `disallowed_tools=` here would be a
    confusing no-op carrying false-positive maintenance signal.
    Asserts the raw adapter's `ClaudeAgentOptions` is left empty on the
    `disallowed_tools` field (SDK default = empty list)."""
    from bird_interact_agents.agents.claude_sdk_otf_raw import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert captured["options"].disallowed_tools == [], (
        "raw adapter exposes no SLayer tools; the DEV-1548 plan explicitly "
        "leaves disallowed_tools= empty (SDK default). A maintainer who "
        "needs to disallow built-ins should add a SEPARATE constant rather "
        "than reusing SLAYER_MCP_DISALLOWED_TOOL_NAMES from the slayer-aware "
        "adapters."
    )


@pytest.mark.asyncio
async def test_run_task_whitelists_submit_sql_not_submit_query(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__submit_sql" in allowed
    assert "mcp__bird-interact-tools__submit_query" not in allowed
    assert "mcp__bird-interact-tools__ask_user" not in allowed


@pytest.mark.asyncio
async def test_run_task_whitelists_all_raw_tools(monkeypatch, tmp_path):
    """All 7 BIRD_INTERACT_TOOLS + submit_sql must appear on the allow-list."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    allowed = set(captured["options"].allowed_tools)
    for tool_name in (
        "execute_sql", "get_schema", "get_all_column_meanings", "get_column_meaning",
        "get_all_external_knowledge_names", "get_knowledge_definition",
        "get_all_knowledge_definitions", "submit_sql",
    ):
        assert f"mcp__bird-interact-tools__{tool_name}" in allowed, (
            f"raw agent must whitelist {tool_name}"
        )


@pytest.mark.asyncio
async def test_run_task_restricts_tools_and_caps_turns(monkeypatch, tmp_path):
    """No Claude Code built-ins; isolated settings; max_turns = 2× MAX_MODEL_TURNS."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m
    from bird_interact_agents.harness import MAX_MODEL_TURNS

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    opts = captured["options"]
    assert opts.tools == ["Task"]  # DEV-1555: only built-in re-enabled
    assert opts.setting_sources == []
    assert opts.max_turns == 2 * MAX_MODEL_TURNS
    assert "PostToolUse" in (opts.hooks or {})


@pytest.mark.asyncio
async def test_run_task_one_shot_livesqlbench_calls_materialize(monkeypatch, tmp_path):
    """LiveSQLBench one-shot: per-task DB isolation via materialize_task_db."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert captured["materialize_calls"] == 1


@pytest.mark.asyncio
async def test_run_task_pins_requested_model(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-opus-4-7")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert captured["options"].model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_run_task_passes_reasoning_effort(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(
        model="anthropic/claude-sonnet-4-5", reasoning_effort="high",
    )
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert captured["options"].effort == "high"


@pytest.mark.asyncio
async def test_run_task_default_effort_is_none(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert captured["options"].effort is None


# ---------------------------------------------------------------------------
# Turn budget hook — must say "submit_sql" not "submit_query"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_budget_hook_warns_near_cap_with_submit_sql():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    hook = m._make_turn_budget_hook(max_turns=5, warn_within=3)
    assert await hook({}, None, None) == {}        # call 1 -> 4 left
    out = await hook({}, None, None)               # call 2 -> 3 left -> warn
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "submit_sql" in ctx, "turn budget hook must say 'submit_sql' not 'submit_query'"
    assert "submit_query" not in ctx
    assert "3" in ctx


# ---------------------------------------------------------------------------
# Usage accumulation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_captures_usage(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m
    from bird_interact_agents import usage as usage_mod

    msgs = [_FakeAssistant(100, 20), _FakeAssistant(150, 30, cache=5)]
    _stub_env(monkeypatch, m, tmp_path / "store", messages=msgs)
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    rebuilt = usage_mod.TokenUsage.model_validate(row["usage"])
    assert rebuilt.prompt_tokens == 250
    assert rebuilt.completion_tokens == 50
    assert rebuilt.cache_read_tokens == 5


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
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=_full_prefill(),
        prefill_timing="after",
    )
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
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
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
    )
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert row["submission_status"] is None
    assert row["predicted_result_json"] is None
    assert row["gold_result_json"] is None
    assert row["phase1_observation"] is None
    assert row["phase2_observation"] is None


@pytest.mark.asyncio
async def test_run_task_exception_path_propagates_partial_result(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    prefill = _full_prefill(
        submission_status="submitted_ok", phase1_passed=True,
        phase2_passed=True, total_reward=0.75,
        phase2_observation="p2 ok",
    )
    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=prefill,
        prefill_timing="after",
        raise_after_prefill=RuntimeError("boom"),
    )
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
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
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")

    def _boom(*a, **kw):
        raise RuntimeError("early-setup boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert "early-setup boom" in (row.get("error") or "")
    assert row["submission_status"] is None
    assert row["phase1_passed"] is False


@pytest.mark.asyncio
async def test_run_task_exception_path_isolated_from_stale_context(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1 import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")
    m._ctx_var.set({
        "result": {
            "submission_status": "STALE_SHOULD_NOT_LEAK",
            "phase1_passed": True,
            "predicted_result_json": "STALE",
            "gold_result_json": "STALE",
            "phase1_observation": "STALE",
        },
    })

    def _boom(*a, **kw):
        raise RuntimeError("early boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)
    agent = m.ClaudeSDKOtfRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "raw", eval_mode="one-shot",
    )
    assert "early boom" in (row.get("error") or "")
    assert row["submission_status"] != "STALE_SHOULD_NOT_LEAK"
    assert row["submission_status"] is None
    assert row["phase1_passed"] is False
