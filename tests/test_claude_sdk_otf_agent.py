"""Unit tests for the `claude_sdk_otf` agent (no LLM, no real cache build).

After DEV-1507 the framework is **livesqlbench / one-shot only**. The
mini-interact / a-interact behavior lives in the sibling
`claude_sdk_otf_ainteract` flavor (see
`tests/test_claude_sdk_otf_ainteract_agent.py`).
"""

from __future__ import annotations

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


def test_select_tools_one_shot_returns_six_native_tools():
    """3 knowledge tools + the DEV-1534 Fix C query/query_nested wrappers
    + submit_query = 6 native; no ask_user. Total tool count (with 9
    slayer subprocess tools after Fix C moves query/query_nested out
    of the subprocess allowlist) is 15."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    names = _tool_names(m._select_tools("one-shot"))
    assert names == {
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
        "query",
        "query_nested",
        "submit_query",
    }
    assert "ask_user" not in names


def test_select_tools_rejects_a_interact_and_others():
    """After DEV-1507 the narrowed flavor is one-shot only."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    for bad in ("a-interact", "c-interact", "oracle"):
        with pytest.raises(ValueError):
            m._select_tools(bad)


# ---------------------------------------------------------------------------
# DEV-1534 Codex post-merge: create_model / edit_model PreToolUse hook
# normalizes backing-query text-equality filters before SLayer persists
# them on the model.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_write_filters_hook_rewrites_create_model_query():
    """A `create_model` call whose backing `query` carries a string-equality
    filter must have that filter wrapped in `lower(trim(col)) = '<lower>'`
    before SLayer sees it — otherwise the persisted model has
    case-sensitive backing filters that no later query-time
    normalization can repair."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        _normalize_write_tool_filters_hook,
    )

    input_data = {
        "tool_name": "mcp__slayer__create_model",
        "tool_input": {
            "name": "premium_orders",
            "query": {
                "source_model": "orders",
                "filters": ["category == 'Gadgets'"],
            },
        },
    }
    out = await _normalize_write_tool_filters_hook(input_data, None, None)
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["query"]["filters"] == ["lower(trim(category)) == 'gadgets'"]
    # Input dict must not be mutated (deep-copy contract).
    assert input_data["tool_input"]["query"]["filters"] == [
        "category == 'Gadgets'"
    ]


@pytest.mark.asyncio
async def test_normalize_write_filters_hook_rewrites_edit_model_source_queries():
    """`edit_model.source_queries` is a list of stages each with its own
    `filters` — every stage's filters must be normalized."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        _normalize_write_tool_filters_hook,
    )

    input_data = {
        "tool_name": "mcp__slayer__edit_model",
        "tool_input": {
            "model_name": "orders",
            "source_queries": [
                {"source_model": "orders", "filters": ["status == 'OPEN'"]},
                {"source_model": "products", "filters": ["category == 'Premium'"]},
            ],
        },
    }
    out = await _normalize_write_tool_filters_hook(input_data, None, None)
    updated = out["hookSpecificOutput"]["updatedInput"]
    assert updated["source_queries"][0]["filters"] == [
        "lower(trim(status)) == 'open'"
    ]
    assert updated["source_queries"][1]["filters"] == [
        "lower(trim(category)) == 'premium'"
    ]


@pytest.mark.asyncio
async def test_normalize_write_filters_hook_skips_non_write_tools():
    """Tools other than create_model / edit_model must pass through
    unchanged — the hook returns `{}` so the SDK falls through to the
    next hook (or the default allow-and-forward)."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        _normalize_write_tool_filters_hook,
    )

    for unrelated in (
        "mcp__slayer__inspect_model",
        "mcp__bird-interact-tools__query",
        "mcp__slayer__save_memory",
        "",
    ):
        out = await _normalize_write_tool_filters_hook(
            {"tool_name": unrelated, "tool_input": {"x": 1}}, None, None,
        )
        assert out == {}


def test_slayer_tool_names_include_write_tools():
    """The OTF agent must be able to WRITE models, unlike the read-only
    claude_sdk slayer mode. After DEV-1534 Fix C, ``query`` and
    ``query_nested`` are served by our bird-interact-tools wrappers
    (NOT the SLayer subprocess), leaving 9 SLayer subprocess tools."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    names = set(m._slayer_tool_names())
    for t in (
        "mcp__slayer__create_model",
        "mcp__slayer__edit_model",
        "mcp__slayer__save_memory",
        "mcp__slayer__validate_models",
    ):
        assert t in names, f"missing write tool {t}"
    # Discovery / read tools still come from the SLayer subprocess MCP.
    for t in ("mcp__slayer__search", "mcp__slayer__inspect_model"):
        assert t in names
    # DEV-1534 Fix C: query / query_nested moved off the SLayer
    # subprocess allowlist onto bird-interact-tools wrappers.
    for t in ("mcp__slayer__query", "mcp__slayer__query_nested"):
        assert t not in names, (
            f"{t} should be served by the bird-interact-tools wrapper "
            "after DEV-1534 Fix C, not the SLayer subprocess MCP server."
        )
    assert len(names) == 9


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
    "dataset": "livesqlbench-base-lite-sqlite",
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
async def test_run_task_accepts_livesqlbench_alias():
    """Codex (PR #10 follow-up): the agent-level dataset gate must accept
    the canonical token regardless of any alias normalization upstream.
    The narrowed flavor is livesqlbench-only; canonicalization happens
    via `get_benchmark(dataset).name` so any future alias would be honored
    consistently with the validator."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="openai/gpt-4o")
    td = dict(_TASK, dataset="livesqlbench-base-lite-sqlite")  # canonical
    row = await agent.run_task(
        td, "/tmp", 20.0, "slayer", eval_mode="one-shot",
    )
    # Got past dataset gate; non-anthropic short-circuit produced a skip row.
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


@pytest.mark.asyncio
async def test_run_task_rejects_mini_interact_dataset():
    """claude_sdk_otf is bound to livesqlbench at the agent layer — a
    programmatic caller (`make_runner` has no dataset arg) cannot bypass
    the CLI gate by passing task_data with the wrong dataset."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="mini-interact")
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


def _make_fake_client(
    captured: dict,
    messages,
    *,
    m_module,
    prefill_result=None,
    prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
):
    """Build a fake `ClaudeSDKClient`.

    The optional `prefill_result` simulates a `submit_query` tool call
    inside the agent loop by mutating the per-task `_ctx["result"]` at the
    requested timing:

    * `"before"` — set BEFORE yielding any message (mirrors a submit on
      turn 1).
    * `"after"`  — set AFTER yielding all messages (mirrors a submit
      that landed on the final turn, just before finalize).

    `raise_after_prefill` lets a test exercise the exception path with a
    partial `_ctx["result"]` already in place (agent submitted then the
    SDK loop crashed).
    """
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
            if prefill_result is not None and prefill_timing == "after":
                m_module._ctx_var.get()["result"] = dict(prefill_result)
            if raise_after_prefill is not None:
                raise raise_after_prefill

    return _FakeClient


def _stub_env(
    monkeypatch, m, storage_dir,
    *,
    messages=(), captured=None, deleted=(),
    prefill_result=None,
    prefill_timing: str = "after",
    raise_after_prefill: Exception | None = None,
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
        captured["slayer_mcp_kw"] = dict(kw)
        return {"command": "slayer", "args": ["mcp"], "env": {}}

    monkeypatch.setattr(m, "slayer_mcp_stdio_config", _fake_slayer_mcp)
    monkeypatch.setattr(m, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())

    async def fake_resolve(*, db_name, task_data, data_path_base, benchmark):
        captured["resolve_kwargs"] = {
            "db_name": db_name, "benchmark": benchmark,
            "data_path_base": data_path_base,
        }
        return str(storage_dir), list(deleted)

    monkeypatch.setattr(m, "resolve_otf_task_storage_dir", fake_resolve)
    monkeypatch.setattr(
        m, "ClaudeSDKClient",
        _make_fake_client(
            captured, messages,
            m_module=m,
            prefill_result=prefill_result,
            prefill_timing=prefill_timing,
            raise_after_prefill=raise_after_prefill,
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
            task_jsonl_path="livesqlbench.jsonl",
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
    assert captured["resolve_kwargs"]["benchmark"] == "livesqlbench-base-lite-sqlite"


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
async def test_run_task_passes_ingest_on_startup_false_to_slayer_mcp(
    monkeypatch, tmp_path,
):
    """DEV-1508: claude_sdk_otf must boot the slayer MCP WITHOUT
    --ingest-on-startup. The OTF cache is post-ingestion by construction;
    the Claude Agent SDK has no startup-timeout knob, so a slow re-ingest
    leaves slayer status='pending' for the whole session and the agent
    silently loses every mcp__slayer__* tool."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert captured["slayer_mcp_kw"].get("ingest_on_startup") is False


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


@pytest.mark.asyncio
async def test_run_task_passes_disallowed_slayer_tools_to_sdk(
    monkeypatch, tmp_path,
):
    """DEV-1548: the one-shot OTF adapter must thread
    `SLAYER_MCP_DISALLOWED_TOOL_NAMES` verbatim into the
    `ClaudeAgentOptions.disallowed_tools=` field — `allowed_tools=` only
    gates auto-execute permission; `disallowed_tools=` is what removes
    the JSON Schema from the model's per-turn cacheable prefix. The unit
    contract is that the live `ClaudeAgentOptions` instance reaching the
    SDK carries the canonical list; the cloud-smoke acceptance criterion
    asserts the SDK actually applies it (no disallowed names in
    `SystemMessage.data.tools`).
    """
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert (
        captured["options"].disallowed_tools
        == m.SLAYER_MCP_DISALLOWED_TOOL_NAMES
    )


def test_accumulate_assistant_usage_dict_shaped_and_skips_result(monkeypatch):
    """Regression: the live SDK delivers `msg.usage` as a DICT. The shared
    helper must read it (not via getattr→0) AND skip the cumulative
    ResultMessage so agent tokens/cost aren't zero or double-counted."""
    from bird_interact_agents.agents.claude_sdk.agent import (
        accumulate_assistant_usage,
    )
    from bird_interact_agents import usage as usage_mod

    # Pin pricing so the >0 assertion below doesn't depend on litellm's
    # cost-map being loaded for the given model (CodeRabbit on PR #9).
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (1e-6, 1e-6))

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

    assert captured["resolve_kwargs"]["benchmark"] == "livesqlbench-base-lite-sqlite"
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

def test_import_does_not_pull_pydantic_ai_adapter_packages(
    import_isolation_results,
):
    """claude_sdk_otf must not drag in the heavy pydantic_ai *adapter*
    packages (`agents.pydantic_ai{,_recursive,_otf_encode}`) — that was the
    Codex concern about importing the shared OTF resolver from the recursive
    adapter. The pydantic_ai *core* lib is a pre-existing transitive dep of
    `slayer_otf` (via reference_build -> agents._session_log) and is NOT in
    scope here, so it is allowed.

    Drives the consolidated ``import_isolation_results`` subprocess fixture
    (DEV-1508 perf — see ``tests/conftest.py``); the check still runs in a
    fresh sub-interpreter with sys.modules cleared, just shared across the
    three boundary tests."""
    r = import_isolation_results["claude_sdk_otf_no_pydantic_ai_adapter"]
    assert r["ok"], (
        "claude_sdk_otf import leaked pydantic_ai ADAPTER packages: "
        f"leaked={r.get('leaked')!r} error={r.get('error')!r}"
    )


# ---------------------------------------------------------------------------
# DEV-1511: diagnostic-field propagation from _ctx["result"] to the
# finalized row. The submit helpers (`agents/_submit.py::_diagnostic_payload`)
# populate `submission_status`, `predicted_result_json`, `gold_result_json`,
# `phase1_observation` (and `phase2_observation` on phase 2) onto
# `state.result`. The bug: `run_task` did not propagate them, so every
# Claude-SDK row was mis-labeled `submission_status="never_submitted"` by
# `run.py`'s setdefault — including rows that PASSED.
# ---------------------------------------------------------------------------


def _full_prefill(**overrides):
    """A `_ctx['result']` snapshot shaped like what `_diagnostic_payload`
    produces on a successful submit. Tests override individual keys."""
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
    """The 5 fields (`submission_status`, `predicted_result_json`,
    `gold_result_json`, `phase1_observation`, `phase2_observation`) must
    appear on the returned row when the submit helper populated them."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    prefill = _full_prefill()
    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=prefill,
        prefill_timing="after",
    )
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    # phase2_observation absent from prefill => key MUST still be present
    # on the row (with value None), so downstream classifiers can read it
    # unconditionally instead of using `.get(...)`.
    assert "phase2_observation" in row
    assert row["phase2_observation"] is None
    # Pre-existing pass-throughs must still work (regression guard).
    assert row["phase1_passed"] is True
    assert row["submitted_query"] == "{\"models\": [\"m\"]}"
    assert row["error"] is None


@pytest.mark.asyncio
async def test_run_task_propagates_phase2_observation(monkeypatch, tmp_path):
    """`phase2_observation` is set by `_diagnostic_payload` when the submit
    runs in phase 2 (symmetric with `phase1_observation`). Same root-cause
    drop; must propagate too."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    prefill = {
        "submission_status": "wrong_result",
        "phase1_passed": True,
        "phase2_passed": False,
        "phase2_observation": "p2 fail observation",
        # phase1_observation deliberately omitted
    }
    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        prefill_result=prefill,
    )
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert row["phase2_observation"] == "p2 fail observation"
    # phase1_observation absent from prefill => key present with value None
    assert "phase1_observation" in row
    assert row["phase1_observation"] is None


@pytest.mark.asyncio
async def test_run_task_propagation_defaults_to_none_when_never_submitted(
    monkeypatch, tmp_path,
):
    """Adapter-contract coverage: when the agent never called submit, the
    adapter's returned row must carry `None` (not a misleading sentinel) for
    the 5 diagnostic fields. The `"never_submitted"` sentinel lives only in
    `run.py`'s downstream `setdefault`, not in this adapter's row."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    _stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_FakeAssistant(100, 20)],
        # no prefill => _ctx["result"] stays None
    )
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
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
    """If the SDK loop crashes AFTER a successful submit, the exception
    finalize block must rescue the diagnostic state from `_ctx["result"]`
    rather than dropping it on the floor with `phase1_passed=False` and
    `submission_status=None`. Mirrors the happy-path field set INCLUDING
    pre-existing pass-throughs (`phase2_passed`, `total_reward`, the dual-
    eval columns) — a rewrite must not silently drop any of them."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    prefill = _full_prefill(
        submission_status="submitted_ok", phase1_passed=True,
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
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert row["error"] == "boom"
    # Rescued diagnostic state (the 5 new fields)
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    assert row["phase2_observation"] == "p2 ok"
    # Rescued pre-existing fields — must still pass through on exception.
    assert row["phase1_passed"] is True
    assert row["phase2_passed"] is True
    assert row["total_reward"] == 0.75
    assert row["submitted_query"] == "{\"models\": [\"m\"]}"
    assert row["submitted_sql"] == "SELECT 1"
    # Dual-eval columns: still pass-through.
    assert row["phase1_observation_audited"] == "audited-obs"
    assert row["phase1_observation_original"] == "original-obs"


@pytest.mark.asyncio
async def test_run_task_exception_before_ctx_set_yields_empty_diagnostics(
    monkeypatch, tmp_path,
):
    """If an early-setup call (`load_db_data_if_needed`,
    `materialize_task_db`, `resolve_otf_task_storage_dir`) raises BEFORE
    `_ctx_var.set(...)` runs, the exception finalize block must not crash
    with LookupError — and must return None for the diagnostic fields."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")
    # Override load_db_data_if_needed to raise early.
    def _boom(*a, **kw):
        raise RuntimeError("early-setup boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)

    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
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
    """Codex blocker: if a prior task in the same async context left
    `_ctx_var` pointing at a stale dict, an early-setup failure in the
    current task must NOT propagate the stale `result` into the new row.

    The fix is to read the exception-path `result` from a LOCAL variable
    populated alongside `_ctx_var.set(...)`, not from `_ctx_var.get()`.
    This test verifies that contract by pre-setting `_ctx_var` to a stale
    fake, then forcing an early-setup failure, and asserting the row
    diagnostics are NOT the stale ones."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    _stub_env(monkeypatch, m, tmp_path / "store")
    # Simulate a stale per-context contextvar from a prior task.
    m._ctx_var.set({
        "result": {
            "submission_status": "STALE_SHOULD_NOT_LEAK",
            "phase1_passed": True,
            "predicted_result_json": "STALE",
            "gold_result_json": "STALE",
            "phase1_observation": "STALE",
        },
    })
    # Force the early-setup phase to raise (before this run's _ctx_var.set).
    def _boom(*a, **kw):
        raise RuntimeError("early boom")

    monkeypatch.setattr(m, "load_db_data_if_needed", _boom)

    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="one-shot",
    )
    assert "early boom" in (row.get("error") or "")
    # Crucially: NOT the stale values.
    assert row["submission_status"] != "STALE_SHOULD_NOT_LEAK"
    assert row["submission_status"] is None
    assert row["predicted_result_json"] is None
    assert row["gold_result_json"] is None
    assert row["phase1_observation"] is None
    assert row["phase1_passed"] is False
