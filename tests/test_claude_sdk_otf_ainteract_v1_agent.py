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

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Construction + reasoning-effort
# ---------------------------------------------------------------------------

def test_init_rejects_non_on_the_fly():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfAInteractAgent(slayer_setup="pre-encoded")


def test_init_accepts_on_the_fly_default():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    assert agent.slayer_setup == "on-the-fly"


def test_init_rejects_bad_reasoning_effort():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfAInteractAgent(reasoning_effort="turbo")


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def _tool_names(tools):
    return {t.name for t in tools}


def test_ainteract_partition_has_ask_user_on_both_clients():
    """DEV-1581 R2: the a-interact partition is the one-shot partition + the
    in-process ``ask_user`` native on BOTH clients (main asks on submit-feedback
    ambiguities; discovery does the bulk of clarification)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    ask = "mcp__bird-interact-tools__ask_user"
    main = set(m.MAIN_NATIVE_TOOL_NAMES)
    disc = set(m.DISCOVERY_NATIVE_TOOL_NAMES)
    assert ask in main and ask in disc
    assert "mcp__bird-interact-tools__submit_query" in main
    # DEV-1629: search / inspect_model live on BOTH clients now (main introspects
    # directly; discovery keeps them for ask_discovery). models_summary stays
    # discovery-only.
    for t in ("mcp__bird-interact-tools__search",
              "mcp__bird-interact-tools__inspect_model"):
        assert t in disc and t in main
    assert "mcp__bird-interact-tools__models_summary" in disc
    assert "mcp__bird-interact-tools__models_summary" not in main


def test_main_native_tool_names_include_write_tools():
    """The ainteract OTF main client must also expose SLayer write tools."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    names = set(m.MAIN_NATIVE_TOOL_NAMES)
    for t in (
        "mcp__bird-interact-tools__create_model",
        "mcp__bird-interact-tools__edit_model",
        "mcp__bird-interact-tools__validate_models",
    ):
        assert t in names, f"missing write tool {t}"


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

def test_build_prompt_is_ainteract_variant():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    td = {"amb_user_query": "how many widgets?", "selected_database": "shop"}
    prompt = m._build_prompt("a-interact", td, budget=20.0)
    assert "how many widgets?" in prompt
    assert "shop" in prompt
    assert "ask_user" in prompt.lower()
    assert "submit_query" in prompt.lower()


def test_build_prompt_rejects_unknown_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    td = {"amb_user_query": "?", "selected_database": "shop"}
    for bad in ("one-shot", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._build_prompt(bad, td, budget=20.0)


def test_prompt_rule_zero_precedes_encoding():
    """Rule 0 (the ask_user-before-encoding instruction) must appear BEFORE the
    encoding workflow language so it's read first. Asserted on substring offset
    rather than an exact index — the contract is ordering, not position."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import prompts as p

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import prompts as p

    text = p.SLAYER_OTF_AINTERACT.lower()
    assert "submit" in text
    # Either "refuse" or "deny" or "block" wording is acceptable.
    assert any(w in text for w in ("refuse", "deny", "block")), text


def test_prompts_use_synthetic_examples_only():
    """Guards `feedback_prompts_synthetic_examples_only`: no real eval-set
    DB / table / column / value names may appear in the prompt."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import prompts as p

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _gate, _counter, nag = m._make_ask_user_guards()
    for _ in range(9):
        out = await nag({"tool_name": "mcp__slayer__query"}, None, None)
        assert out == {}


@pytest.mark.asyncio
async def test_post_nag_fires_at_ten_with_count():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    "dataset": "mini-interact",
}


@pytest.mark.asyncio
async def test_run_task_rejects_raw_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(
            dict(_TASK), "/tmp", 20.0, "raw", eval_mode="a-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_unsupported_eval_modes():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="livesqlbench-base-lite-sqlite")
    with pytest.raises(ValueError):
        await agent.run_task(td, "/tmp", 20.0, "slayer", eval_mode="a-interact")


@pytest.mark.asyncio
async def test_run_task_non_anthropic_model_skips():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
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


def _make_fake_client(
    captured: dict, messages,
    *, m_module=None, prefill_result=None, prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
    prefill_asks: int = 0,
):
    """Build a fake `ClaudeSDKClient`. See the equivalent helper in
    test_claude_sdk_otf_v1_agent.py for the prefill semantics.

    ``prefill_asks`` simulates the ``ask_user`` tool having been called
    ``N`` times during the message loop — pokes ``asks_used`` into the
    per-task context dict, which the real tool's handler does at
    runtime. Used by the n_ask_user_calls reporting tests so they
    don't have to wire real MCP tools.
    """
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
    messages=(), captured=None, deleted=(),
    prefill_result=None, prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
    prefill_asks: int = 0,
):
    from pathlib import Path
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.eval.annotation_schema import (
        MetadataSufficiency, Provenance, TaskAnnotation,
    )
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    captured = captured if captured is not None else {}
    captured.setdefault("materialize_calls", 0)
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(m, "load_db_data_if_needed", lambda *a, **kw: None)

    def _fake_materialize(*a, **kw):
        captured["materialize_calls"] += 1
        return None

    monkeypatch.setattr(m, "materialize_task_db", _fake_materialize)

    def _fake_slayer_mcp(storage_dir, **kw):
        # DEV-1508: capture the ingest_on_startup kwarg so the regression
        # test (`test_run_task_passes_ingest_on_startup_false_to_slayer_mcp`)
        # can assert the OTF adapter opted out of startup re-ingest.
        captured["slayer_mcp_kw"] = dict(kw)
        return {"command": "slayer", "args": ["mcp"], "env": {}}

    # DEV-1581 R2: v1 replaced slayer stdio + create_sdk_mcp_server with the
    # in-process build_bird_interact_server (no slayer process). Patch
    # whichever the module exposes (raising=False → harmless no-op if absent).
    monkeypatch.setattr(
        m, "slayer_mcp_stdio_config", _fake_slayer_mcp, raising=False,
    )
    monkeypatch.setattr(
        m, "create_sdk_mcp_server", lambda **kw: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        m, "build_bird_interact_server", lambda *a, **kw: SimpleNamespace(),
        raising=False,
    )

    async def fake_resolve(*, db_name, task_data, data_path_base, benchmark,
                           apply_edited_models=False):
        captured["resolve_kwargs"] = {
            "db_name": db_name, "benchmark": benchmark,
            "data_path_base": data_path_base,
        }
        return str(storage_dir), list(deleted)

    monkeypatch.setattr(m, "resolve_otf_task_storage_dir", fake_resolve)
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
            prefill_asks=prefill_asks,
        ),
    )

    # Autopsy precondition + trigger mocks: create a real annotation file so
    # _ann_path.exists() returns True, then return a passing cascade so the
    # autopsy never actually fires an LLM call in unit tests.
    ann_dir = Path(storage_dir).parent
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / "fake_ann.json"
    ann_file.write_text("{}")
    _fake_ann = TaskAnnotation(
        instance_id=_TASK["instance_id"],
        selected_database=_TASK["selected_database"],
        annotated_by="test",
        annotated_at="2024-01-01",
        amb_user_query=_TASK["amb_user_query"],
        metadata_sufficiency=MetadataSufficiency(verdict="sufficient", rationale="test"),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=_TASK["instance_id"],
        ),
    )
    _passing_cascade = CascadeVerdict(
        n1_original_gold=True, n2_audited_primary=True, n3_any_audited_variant=True,
        n4_tie_order=True, n5_llm_judge=True, n6_numeric_epsilon=True,
        n7_trailing_whitespace=True, n8_column_order=True, n9_case_fold=True,
    )
    monkeypatch.setattr(m, "task_annotation_path", lambda **kw: ann_file)
    monkeypatch.setattr(m, "load_task_annotation_or_implicit", lambda **kw: _fake_ann)
    monkeypatch.setattr(m, "grade_submission", lambda **kw: _passing_cascade)
    monkeypatch.setattr(m, "load_audited_gold_rows_for", lambda **kw: [])

    return captured


@pytest.mark.asyncio
async def test_run_task_uses_cache_resolver_with_mini_interact_benchmark(
    monkeypatch, tmp_path,
):
    """ainteract resolves per-task storage from the deterministic cache scoped
    to mini_interact, never the livesqlbench root."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    # The committed-models resolver must not even be imported.
    assert not hasattr(m, "resolve_task_storage_dir")

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["resolve_kwargs"]["benchmark"] == "mini-interact"


@pytest.mark.asyncio
async def test_run_task_attaches_slayer_write_tools(monkeypatch, tmp_path):
    """The MAIN ClaudeAgentOptions handed to the SDK must whitelist the slayer
    write tools (in-process bird-interact-tools natives) so the agent can
    encode. ``captured['options']`` is the LAST client created, i.e. main."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__create_model" in allowed
    assert "mcp__bird-interact-tools__edit_model" in allowed
    assert "mcp__bird-interact-tools__validate_models" in allowed


@pytest.mark.asyncio
async def test_run_task_whitelists_ask_user_and_submit_query(monkeypatch, tmp_path):
    """Both native tools that drive the ask-then-submit discipline must be on
    the allow-list."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__ask_user" in allowed
    assert "mcp__bird-interact-tools__submit_query" in allowed


@pytest.mark.asyncio
async def test_run_task_passes_disallowed_slayer_tools_to_sdk(
    monkeypatch, tmp_path,
):
    """DEV-1644: the a-interact OTF adapter must thread the DERIVED disallowed
    set (`= all advertised slayer tools − allow-list`) into the
    `ClaudeAgentOptions.disallowed_tools=` field — same contract as the
    sibling one-shot adapter (both derive from the shared allow-list so the
    two adapters stay symmetric). The unit contract is that the live
    `ClaudeAgentOptions` instance reaching the SDK carries the complement so
    nothing leaks; the cloud-smoke acceptance criterion asserts the SDK
    actually applies it.
    """
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    expected = derive_disallowed_slayer_tools(m.SLAYER_MCP_TOOLS)
    assert captured["options"].disallowed_tools == expected
    allow = set(m._slayer_tool_names())
    got = set(captured["options"].disallowed_tools)
    assert got.isdisjoint(allow)
    for leak in (
        "mcp__slayer__query", "mcp__slayer__query_nested",
        "mcp__slayer__describe_datasource", "mcp__slayer__edit_datasource",
        "mcp__slayer__delete_model",
    ):
        assert leak in got


@pytest.mark.asyncio
async def test_run_task_pre_encoded_derives_disallowed_from_stripped_allow_list(
    monkeypatch, tmp_path,
):
    """DEV-1644 + DEV-1586: pre-encoded (read-only) a-interact must derive the
    disallowed set from the WRITE-STRIPPED allow-list, hiding the write-tool
    schemas."""
    from bird_interact_agents.agents._pre_encoded import (
        WRITE_SLAYER_TOOL_NAMES,
        strip_write_slayer_tools,
    )
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")

    async def _fake_pre_encoded(*, db_name, task_data, data_path_base, benchmark, source):
        return str(tmp_path / "store"), []

    monkeypatch.setattr(m, "resolve_pre_encoded_storage_dir", _fake_pre_encoded)

    agent = m.ClaudeSDKOtfAInteractAgent(
        model="anthropic/claude-sonnet-4-5",
        slayer_setup="pre-encoded",
        pre_encoded_source="otf",
    )
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    expected = derive_disallowed_slayer_tools(
        strip_write_slayer_tools(m.SLAYER_MCP_TOOLS)
    )
    got = set(captured["options"].disallowed_tools)
    assert captured["options"].disallowed_tools == expected
    assert WRITE_SLAYER_TOOL_NAMES <= got
    assert got.isdisjoint(set(captured["options"].allowed_tools))


@pytest.mark.asyncio
async def test_run_task_registers_three_guards_plus_turn_budget(
    monkeypatch, tmp_path,
):
    """Hook registration (DEV-1581 R2 MAIN client): PreToolUse has the
    submit_query matcher (ask-user + query-before-submit gates) plus the
    wall-clock deny; PostToolUse carries the ask-counter, the nag, the
    turn-budget hook, context-budget, wall-clock-warning, AND the all-tools
    tracker. No partition-deny / normalize-write hooks (R2 removed them)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    hooks = captured["options"].hooks
    assert "PreToolUse" in hooks
    assert "PostToolUse" in hooks

    pre_matchers = hooks["PreToolUse"]
    # Two PreToolUse matchers: [0] submit_query (ask + query gates),
    # [1] wall-clock budget deny (matcher None — fires on every tool, only
    # denies past-budget non-submits).
    assert len(pre_matchers) == 2
    submit_pm = next(
        pm for pm in pre_matchers
        if pm.matcher == "mcp__bird-interact-tools__submit_query"
    )
    assert len(submit_pm.hooks) == 2  # ask-user gate, then query-before-submit

    post_matchers = hooks["PostToolUse"]
    # Exactly six PostToolUse matchers: ask-counter (matcher == ask_user),
    # nag, turn-budget, context-budget (DEV-1555), wall-clock-warning
    # (DEV-1555 follow-up), tracker (all matcher None except ask-counter).
    assert len(post_matchers) == 6
    matchers = {pm.matcher for pm in post_matchers}
    assert "mcp__bird-interact-tools__ask_user" in matchers
    assert None in matchers


@pytest.mark.asyncio
async def test_run_task_registered_hooks_behave_correctly(monkeypatch, tmp_path):
    """Codex MED#2: not only count and matcher names — actually invoke the
    captured hook callables and prove (a) the ask-user gate denies before
    ask_user has been called, (b) the ask-counter flips it to allow,
    (c) the nag fires on the 10th non-ask call when ask_count == 0,
    (d) the query-before-submit gate denies when last tool was not query,
    (e) the tracker + gate together allow after a query call."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    hooks = captured["options"].hooks

    ask_user_gate = hooks["PreToolUse"][0].hooks[0]
    pre_query_gate = hooks["PreToolUse"][0].hooks[1]
    post_by_matcher = {pm.matcher: pm.hooks[0] for pm in hooks["PostToolUse"]}
    counter = post_by_matcher["mcp__bird-interact-tools__ask_user"]
    # The five matcher=None PostToolUse hooks: nag, turn-budget,
    # context-budget (DEV-1555), wall-clock-warning (DEV-1555 follow-up),
    # tracker.
    matcher_none_hooks = [
        pm.hooks[0] for pm in hooks["PostToolUse"] if pm.matcher is None
    ]
    assert len(matcher_none_hooks) == 5  # nag + turn-budget + context + wall-clock + tracker
    # Tracker is the last registered None-matcher hook.
    tracker = matcher_none_hooks[-1]

    # (a) ask-user gate denies before any ask.
    out = await ask_user_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    # (b) ask-counter increments; ask-user gate now allows.
    await counter(
        {"tool_name": "mcp__bird-interact-tools__ask_user"}, None, None,
    )
    out = await ask_user_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out == {}

    # (c) Reset for an isolated nag test: a FRESH factory has ask_count==0.
    pre_gate2, _counter2, nag2 = m._make_ask_user_guards()
    for _ in range(9):
        out = await nag2(
            {"tool_name": "mcp__bird-interact-tools__query"}, None, None,
        )
        assert out == {}
    out = await nag2(
        {"tool_name": "mcp__bird-interact-tools__query"}, None, None,
    )
    assert "additionalContext" in out["hookSpecificOutput"]

    # (d) Query gate denies when last_tool has not been set to a query tool.
    out = await pre_query_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "`query`" in out["hookSpecificOutput"]["permissionDecisionReason"]

    # (e) Tracker records a query call; gate now allows. DEV-1534 Fix C:
    # the SLAYER_QUERY_TOOLS allowlist points at the bird-interact-tools
    # wrappers (not the SLayer subprocess tools), so the satisfying name
    # is `mcp__bird-interact-tools__query`.
    await tracker(
        {"tool_name": "mcp__bird-interact-tools__query"}, None, None,
    )
    out = await pre_query_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out == {}

    # (f) Tracker records a non-query call after the query; gate denies again.
    await tracker({"tool_name": "mcp__slayer__edit_model"}, None, None)
    out = await pre_query_gate(
        {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_run_task_invokes_factory_per_call(monkeypatch, tmp_path):
    """Codex HIGH#1: the hook-state factory MUST be invoked inside run_task
    (per task), not on the agent constructor. Running the SAME agent
    instance through run_task twice must produce two independent guard
    states (regression for the per-task closure contract)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="mini-interact")
    row = await agent.run_task(
        td, str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    # No exception; row has been finalised.
    assert row.get("instance_id") == "shop_1"


@pytest.mark.asyncio
async def test_run_task_restricts_tools_and_caps_turns(monkeypatch, tmp_path):
    """No Claude Code built-ins / ToolSearch; isolated settings; native
    max_turns at 2x the base."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m
    from bird_interact_agents.harness import MAX_MODEL_TURNS

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    opts = captured["options"]
    assert opts.tools == []  # DEV-1581 R2: no built-ins (ask_discovery native)
    assert opts.setting_sources == []
    assert opts.max_turns == 2 * MAX_MODEL_TURNS


@pytest.mark.asyncio
async def test_run_task_pins_requested_model(monkeypatch, tmp_path):
    """--agent-model must reach the SDK as the bare native model id."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-opus-4-7")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_run_task_passes_reasoning_effort(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].effort is None


@pytest.mark.asyncio
async def test_run_task_captures_usage(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m
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


def _shared_turn_msgs():
    """4 AssistantMessage events spanning 2 dedup'd TURNS (each turn = 2
    content blocks sharing ONE usage object, mirroring the live SDK)."""
    from types import SimpleNamespace

    class _AM:
        def __init__(self, usage):
            self.usage = usage

    _AM.__name__ = "AssistantMessage"
    u1 = SimpleNamespace(
        input_tokens=10, output_tokens=2,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    u2 = SimpleNamespace(
        input_tokens=20, output_tokens=3,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    return [_AM(u1), _AM(u1), _AM(u2), _AM(u2)]


@pytest.mark.asyncio
async def test_run_task_n_agent_turns_counts_turns_not_blocks(monkeypatch, tmp_path):
    """DEV-1616: the finalized row's n_agent_turns is the dedup'd TURN count
    (2), not the block count (4 AssistantMessage events), and it carries the
    n_discovery_turns field (0 with no discovery call in this stub)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store", messages=_shared_turn_msgs())
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["n_agent_turns"] == 2
    assert row["usage"]["n_agent_turns"] == 2
    assert row["usage"]["n_discovery_turns"] == 0


# ---------------------------------------------------------------------------
# Import isolation (mirror of the claude_sdk_otf test) — the new flavor must
# not drag in heavy pydantic_ai adapter packages either.
# ---------------------------------------------------------------------------

def test_import_does_not_pull_pydantic_ai_adapter_packages(
    import_isolation_results,
):
    """Drives the consolidated ``import_isolation_results`` subprocess
    fixture (DEV-1508 perf — see ``tests/conftest.py``); the check still
    runs in a fresh sub-interpreter with sys.modules cleared, shared
    across the three boundary tests so we don't pay Python startup three
    times."""
    r = import_isolation_results[
        "claude_sdk_otf_ainteract_no_pydantic_ai_adapter"
    ]
    assert r["ok"], (
        "claude_sdk_otf_ainteract import leaked pydantic_ai ADAPTER packages: "
        f"leaked={r.get('leaked')!r} error={r.get('error')!r}"
    )


# ---------------------------------------------------------------------------
# DEV-1511: diagnostic-field propagation from `_ctx["result"]` to the
# finalized row, mirrored for the post-DEV-1507 ainteract flavor. Same
# shape as `test_claude_sdk_otf_v1_agent.py`'s DEV-1511 block.
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
        "submitted_query": "{\"models\": [\"m\"]}",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_run_task_propagates_diagnostic_fields_on_happy_path(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=_full_prefill(),
        prefill_timing="after",
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    assert "phase2_observation" in row
    assert row["phase2_observation"] is None
    assert row["phase1_passed"] is True
    assert row["submitted_query"] == "{\"models\": [\"m\"]}"
    assert row["error"] is None


@pytest.mark.asyncio
async def test_run_task_propagates_phase2_observation(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result={
            "submission_status": "wrong_result",
            "phase1_passed": True,
            "phase2_passed": False,
            "phase2_observation": "p2 fail observation",
        },
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["phase2_observation"] == "p2 fail observation"
    assert "phase1_observation" in row
    assert row["phase1_observation"] is None


@pytest.mark.asyncio
async def test_run_task_propagation_defaults_to_none_when_never_submitted(
    monkeypatch, tmp_path,
):
    """Adapter contract: row carries None (not the misleading
    "never_submitted" sentinel) when no submit happened. The sentinel
    lives only in run.py's downstream setdefault, not in this row."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    prefill = _full_prefill(
        phase2_passed=True, total_reward=0.75,
        phase2_observation="p2 ok",
        phase1_observation_audited="audited-obs",
        phase1_observation_original="original-obs",
    )
    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=prefill,
        prefill_timing="after",
        raise_after_prefill=RuntimeError("boom"),
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["error"] == "boom"
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    assert row["phase2_observation"] == "p2 ok"
    assert row["phase1_passed"] is True
    assert row["phase2_passed"] is True
    assert row["total_reward"] == 0.75
    assert row["submitted_query"] == "{\"models\": [\"m\"]}"
    assert row["submitted_sql"] == "SELECT 1"
    assert row["phase1_observation_audited"] == "audited-obs"
    assert row["phase1_observation_original"] == "original-obs"


@pytest.mark.asyncio
async def test_run_task_exception_before_ctx_set_yields_empty_diagnostics(
    monkeypatch, tmp_path,
):
    """Early-setup exception (`load_db_data_if_needed` raises before
    `_ctx_var.set(...)`) must not crash with LookupError, and must return
    None for the diagnostic fields — not a stale dict from a prior task."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")

    def _boom(*a, **kw):
        raise RuntimeError("early-setup boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)

    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert "early-setup boom" in (row.get("error") or "")
    assert row["submission_status"] is None
    assert row["predicted_result_json"] is None
    assert row["gold_result_json"] is None
    assert row["phase1_observation"] is None
    assert row["phase2_observation"] is None
    assert row["phase1_passed"] is False


@pytest.mark.asyncio
async def test_run_task_exception_path_isolated_from_stale_context(
    monkeypatch, tmp_path,
):
    """Stale-context isolation (Codex blocker): a prior task's ContextVar
    must not leak into this row when an early-setup failure hits the
    exception path before `_ctx_var.set(...)` runs."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

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

    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert "early boom" in (row.get("error") or "")
    assert row["submission_status"] != "STALE_SHOULD_NOT_LEAK"
    assert row["submission_status"] is None
    assert row["predicted_result_json"] is None
    assert row["gold_result_json"] is None
    assert row["phase1_observation"] is None
    assert row["phase1_passed"] is False


# ---------------------------------------------------------------------------
# DEV-1519: n_ask_user_calls reporting. ``asks_used`` is incremented in the
# per-task ctx dict by the ``ask_user`` tool; without it on the result
# row's ``usage`` dict the grader sees 0 and falsely flags
# ``never_asked_user`` on every interactive miss.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_writes_n_ask_user_calls_zero(monkeypatch, tmp_path):
    """Happy path with no ``ask_user`` calls — usage carries
    ``n_ask_user_calls == 0`` (NOT missing)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_asks=0,
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["usage"]["n_ask_user_calls"] == 0


@pytest.mark.asyncio
async def test_run_task_writes_n_ask_user_calls_nonzero(monkeypatch, tmp_path):
    """Happy path with 3 simulated ``ask_user`` calls — usage carries
    ``n_ask_user_calls == 3``."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_asks=3,
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["usage"]["n_ask_user_calls"] == 3


@pytest.mark.asyncio
async def test_run_task_exception_path_writes_n_ask_user_calls(
    monkeypatch, tmp_path,
):
    """Error path also propagates the ``asks_used`` count — the agent
    asked twice before the failure, so usage carries
    ``n_ask_user_calls == 2`` on the resulting row."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(50, 10)],
        prefill_asks=2,
        raise_after_prefill=RuntimeError("boom"),
    )
    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert "boom" in (row.get("error") or "")
    assert row["usage"]["n_ask_user_calls"] == 2
