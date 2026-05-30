"""Unit tests for the `claude_sdk_otf` agent (no LLM, no real cache build).

After DEV-1507 the framework is **livesqlbench / one-shot only**. The
mini-interact / a-interact behavior lives in the sibling
`claude_sdk_otf_ainteract` flavor (see
`tests/test_claude_sdk_otf_ainteract_agent.py`).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Construction + mode gating
# ---------------------------------------------------------------------------

def test_init_rejects_non_on_the_fly():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    with pytest.raises(ValueError):
        ClaudeSDKOtfAgent(slayer_setup="pre-encoded")


def test_init_accepts_on_the_fly_default():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.slayer_setup == "on-the-fly"


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def _tool_names(tools):
    return {t.name for t in tools}


def test_select_tools_one_shot_returns_four_native_tools():
    """3 knowledge tools + submit_query = 4 native; no ask_user.
    Total tool count (with 11 slayer) is 15."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    names = _tool_names(m._select_tools("one-shot"))
    assert names == {
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
        "submit_query",
    }
    assert "ask_user" not in names


def test_select_tools_rejects_a_interact_and_others():
    """After DEV-1507 the narrowed flavor is one-shot only."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._select_tools(bad)


def test_slayer_tool_names_include_write_tools():
    """The OTF agent must be able to WRITE models, unlike the read-only
    claude_sdk slayer mode. 11 slayer tools total."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    names = set(m._slayer_tool_names())
    for t in (
        "mcp__slayer__create_model",
        "mcp__slayer__edit_model",
        "mcp__slayer__save_memory",
        "mcp__slayer__query_nested",
        "mcp__slayer__validate_models",
    ):
        assert t in names, f"missing write tool {t}"
    # read tools still present
    for t in ("mcp__slayer__query", "mcp__slayer__search", "mcp__slayer__inspect_model"):
        assert t in names
    assert len(names) == 11


# ---------------------------------------------------------------------------
# Prompt selection + hygiene
# ---------------------------------------------------------------------------

def test_build_prompt_uses_one_shot_template():
    """After DEV-1507 the narrowed agent has only the one-shot template;
    a-interact lives in the ainteract flavor's prompts module."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m
    from bird_interact_agents.agents.claude_sdk_otf import prompts as p

    td = {"amb_user_query": "how many widgets?", "selected_database": "shop"}
    out = m._build_prompt("one-shot", td, budget=20.0)
    assert "how many widgets?" in out
    assert "shop" in out
    # one-shot does NOT instruct ask_user.
    assert "ask_user" not in out.lower()
    assert p.SLAYER_OTF_ONE_SHOT
    assert not hasattr(p, "SLAYER_OTF_A_INTERACT"), (
        "SLAYER_OTF_A_INTERACT must be removed from the narrowed module"
    )


def test_build_prompt_rejects_non_one_shot():
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    td = {"amb_user_query": "?", "selected_database": "shop"}
    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._build_prompt(bad, td, budget=20.0)


def test_prompts_encode_in_sequence_and_no_inlining():
    from bird_interact_agents.agents.claude_sdk_otf import prompts as p

    text = p.SLAYER_OTF_ONE_SHOT
    low = text.lower()
    # KB self-annotation contract
    assert "kb_id" in low or "[kb=" in low
    # the core no-inlining instruction
    assert "inlin" in low
    # encourage referencing created entities in the final query
    assert "final query" in low or "reference" in low
    # find relevant KB via search
    assert "search" in low
    # encode in dependency order through declared joins (not invented)
    assert "join" in low
    assert "order" in low or "depend" in low or "sequence" in low


def test_prompts_use_synthetic_examples_only():
    """Guards `feedback_prompts_synthetic_examples_only`: no real eval-set
    DB / table / column / value names may appear in the prompt."""
    from bird_interact_agents.agents.claude_sdk_otf import prompts as p

    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band",
        "taguatinga",
    ]
    low = p.SLAYER_OTF_ONE_SHOT.lower()
    for name in banned:
        assert name not in low, f"real eval-set name {name!r} leaked into prompt"


# ---------------------------------------------------------------------------
# run_task gating
# ---------------------------------------------------------------------------

_TASK = {
    "selected_database": "shop",
    "instance_id": "shop_1",
    "amb_user_query": "?",
    "knowledge_ambiguity": [],
    "dataset": "livesqlbench",
}


@pytest.mark.asyncio
async def test_run_task_rejects_raw_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(dict(_TASK), "/tmp", 20.0, "raw", eval_mode="one-shot")


@pytest.mark.asyncio
async def test_run_task_rejects_unsupported_eval_modes():
    """After DEV-1507 the narrowed agent rejects every non-one-shot mode at
    the agent boundary (defense in depth on top of CLI gates)."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            await agent.run_task(
                dict(_TASK), "/tmp", 20.0, "slayer", eval_mode=bad,
            )


@pytest.mark.asyncio
async def test_run_task_rejects_mini_interact_dataset():
    """claude_sdk_otf is bound to livesqlbench at the agent layer — a
    programmatic caller (`make_runner` has no dataset arg) cannot bypass
    the CLI gate by passing task_data with the wrong dataset."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="mini_interact")
    with pytest.raises(ValueError):
        await agent.run_task(td, "/tmp", 20.0, "slayer", eval_mode="one-shot")


@pytest.mark.asyncio
async def test_run_task_non_anthropic_model_skips():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="openai/gpt-4o")
    row = await agent.run_task(
        dict(_TASK), "/tmp", 20.0, "slayer", eval_mode="one-shot",
    )
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


# ---------------------------------------------------------------------------
# Storage path + allowed tools (the heart of the feature)
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
async def test_run_task_uses_cache_resolver_not_committed(monkeypatch, tmp_path):
    """OTF agent must resolve per-task storage from the deterministic cache
    (`resolve_otf_task_storage_dir`), NOT the committed-models path."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    # The committed-models resolver must not even be imported into the module.
    assert not hasattr(m, "resolve_task_storage_dir")

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert "resolve_kwargs" in captured
    # Narrowed flavor is livesqlbench-only — the cache root is scoped accordingly.
    assert captured["resolve_kwargs"]["benchmark"] == "livesqlbench"


@pytest.mark.asyncio
async def test_run_task_attaches_slayer_write_tools(monkeypatch, tmp_path):
    """The ClaudeAgentOptions handed to the SDK must whitelist the slayer
    write tools so the agent can encode."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__slayer__create_model" in allowed
    assert "mcp__slayer__edit_model" in allowed


@pytest.mark.asyncio
async def test_run_task_does_not_whitelist_ask_user(monkeypatch, tmp_path):
    """Narrowed flavor: ask_user must NOT be on the allow-list (livesqlbench
    has no user simulator)."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__submit_query" in allowed
    assert "mcp__bird-interact-tools__ask_user" not in allowed


@pytest.mark.asyncio
async def test_run_task_restricts_tools_and_caps_turns(monkeypatch, tmp_path):
    """No Claude Code built-ins / ToolSearch (so MCP tools aren't deferred);
    isolated settings; native max_turns at 2x the base; turn-budget hook."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m
    from bird_interact_agents.harness import MAX_MODEL_TURNS

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    opts = captured["options"]
    assert opts.tools == []
    assert opts.setting_sources == []
    assert opts.max_turns == 2 * MAX_MODEL_TURNS == m._MAX_TURNS
    assert "PostToolUse" in (opts.hooks or {})


def test_accumulate_assistant_usage_dict_shaped_and_skips_result():
    """Regression: the live SDK delivers `msg.usage` as a DICT. The shared
    helper must read it (not via getattr→0) AND skip the cumulative
    ResultMessage so agent tokens/cost aren't zero or double-counted."""
    from bird_interact_agents.agents.claude_sdk.agent import (
        accumulate_assistant_usage,
    )
    from bird_interact_agents import usage as usage_mod

    class _AM:
        def __init__(self, usage):
            self.usage = usage

    _AM.__name__ = "AssistantMessage"

    class _RM:
        def __init__(self, usage):
            self.usage = usage

    _RM.__name__ = "ResultMessage"

    accum = usage_mod.TokenUsage()
    accumulate_assistant_usage(
        accum,
        _AM({
            "input_tokens": 100, "output_tokens": 20,
            "cache_read_input_tokens": 5, "cache_creation_input_tokens": 7,
        }),
        "anthropic/claude-opus-4-7",
    )
    # cumulative ResultMessage usage must be ignored (no double count)
    accumulate_assistant_usage(
        accum, _RM({"input_tokens": 9999, "output_tokens": 9999}),
        "anthropic/claude-opus-4-7",
    )
    assert accum.n_calls == 1
    assert accum.prompt_tokens == 100
    assert accum.completion_tokens == 20
    assert accum.cache_read_tokens == 5
    assert accum.cache_write_tokens == 7
    assert accum.agent_cost_usd > 0  # opus priced via litellm, not zero


@pytest.mark.asyncio
async def test_turn_budget_hook_warns_near_cap():
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    hook = m._make_turn_budget_hook(max_turns=5, warn_within=3)
    assert await hook({}, None, None) == {}        # call 1 -> 4 left
    out = await hook({}, None, None)               # call 2 -> 3 left -> warn
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "submit_query" in ctx
    assert "3" in ctx


@pytest.mark.asyncio
async def test_run_task_pins_requested_model(monkeypatch, tmp_path):
    """--agent-model must reach the SDK as the bare native model id, not be
    silently replaced by the claude CLI default (Codex finding)."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-opus-4-7")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert captured["options"].model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_run_task_passes_reasoning_effort(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(
        model="anthropic/claude-sonnet-4-5", reasoning_effort="high",
    )
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert captured["options"].effort == "high"


@pytest.mark.asyncio
async def test_run_task_default_effort_is_none(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert captured["options"].effort is None


def test_init_rejects_bad_reasoning_effort():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    with pytest.raises(ValueError):
        ClaudeSDKOtfAgent(reasoning_effort="turbo")


@pytest.mark.asyncio
async def test_run_task_one_shot_livesqlbench(monkeypatch, tmp_path):
    """LiveSQLBench one-shot: storage resolved with benchmark='livesqlbench',
    materialize_task_db called (per-task DB copy)."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )

    assert captured["resolve_kwargs"]["benchmark"] == "livesqlbench"
    assert captured["materialize_calls"] == 1


@pytest.mark.asyncio
async def test_run_task_captures_usage(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf import agent as m
    from bird_interact_agents import usage as usage_mod

    msgs = [_FakeAssistant(100, 20), _FakeAssistant(150, 30, cache=5)]
    _stub_env(monkeypatch, m, tmp_path / "store", messages=msgs)
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    rebuilt = usage_mod.TokenUsage.model_validate(row["usage"])
    assert rebuilt.prompt_tokens == 250
    assert rebuilt.completion_tokens == 50
    assert rebuilt.cache_read_tokens == 5


# ---------------------------------------------------------------------------
# Import isolation (Codex): no pydantic_ai ADAPTER package pulled in
# ---------------------------------------------------------------------------

def test_import_does_not_pull_pydantic_ai_adapter_packages():
    """claude_sdk_otf must not drag in the heavy pydantic_ai *adapter*
    packages (`agents.pydantic_ai{,_recursive,_otf_encode}`) — that was the
    Codex concern about importing the shared OTF resolver from the recursive
    adapter. The pydantic_ai *core* lib is a pre-existing transitive dep of
    `slayer_otf` (via reference_build -> agents._session_log) and is NOT in
    scope here, so it is allowed.

    Run in a CLEAN interpreter so we don't mutate this process's sys.modules
    (deleting already-imported adapter modules would corrupt class identity
    for other tests in the same session).
    """
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import importlib, sys
        importlib.import_module("bird_interact_agents.agents.claude_sdk_otf.agent")
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
        "claude_sdk_otf import leaked pydantic_ai ADAPTER packages: "
        f"{r.stdout.strip()}{r.stderr.strip()}"
    )
