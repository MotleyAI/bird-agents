"""Unit tests for the `claude_sdk_otf` agent (no LLM, no real cache build).

The framework is a single Claude-Agent-SDK agent that encodes the DB's KB
items into the per-task SLayer store on the fly (off the deterministic OTF
cache) and then queries. It is slayer-query-mode only and supports the
`a-interact` and `one-shot` eval modes.
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


def test_select_tools_a_interact_has_ask_user_and_submit_query():
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    names = _tool_names(m._select_tools("a-interact"))
    assert "ask_user" in names
    assert "submit_query" in names


def test_select_tools_one_shot_omits_ask_user_keeps_submit_query():
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    names = _tool_names(m._select_tools("one-shot"))
    assert "submit_query" in names
    assert "ask_user" not in names


def test_select_tools_rejects_unknown_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    with pytest.raises(ValueError):
        m._select_tools("c-interact")


def test_slayer_tool_names_include_write_tools():
    """The OTF agent must be able to WRITE models, unlike the read-only
    claude_sdk slayer mode."""
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


# ---------------------------------------------------------------------------
# Prompt selection + hygiene
# ---------------------------------------------------------------------------

def test_build_prompt_variant_matches_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf import agent as m
    from bird_interact_agents.agents.claude_sdk_otf import prompts as p

    td = {"amb_user_query": "how many widgets?", "selected_database": "shop"}
    a = m._build_prompt("a-interact", td, budget=20.0)
    o = m._build_prompt("one-shot", td, budget=20.0)
    assert a != o
    # the formatted prompts must carry the question + budget
    assert "how many widgets?" in a and "how many widgets?" in o
    # a-interact MUST instruct the agent to ask the user; one-shot must NOT
    assert "ask_user" in a.lower()
    assert "ask_user" not in o.lower()
    # both descend from the declared constants
    assert p.SLAYER_OTF_A_INTERACT and p.SLAYER_OTF_ONE_SHOT


def test_prompts_encode_in_sequence_and_no_inlining():
    from bird_interact_agents.agents.claude_sdk_otf import prompts as p

    for text in (p.SLAYER_OTF_A_INTERACT, p.SLAYER_OTF_ONE_SHOT):
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
    DB / table / column / value names may appear in the prompts."""
    from bird_interact_agents.agents.claude_sdk_otf import prompts as p

    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band",
        "taguatinga",
    ]
    for text in (p.SLAYER_OTF_A_INTERACT, p.SLAYER_OTF_ONE_SHOT):
        low = text.lower()
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
}


@pytest.mark.asyncio
async def test_run_task_rejects_raw_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(dict(_TASK), "/tmp", 20.0, "raw", eval_mode="a-interact")


@pytest.mark.asyncio
async def test_run_task_rejects_unsupported_eval_mode():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(
            dict(_TASK), "/tmp", 20.0, "slayer", eval_mode="c-interact",
        )


@pytest.mark.asyncio
async def test_run_task_non_anthropic_model_skips():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="openai/gpt-4o")
    row = await agent.run_task(
        dict(_TASK), "/tmp", 20.0, "slayer", eval_mode="a-interact",
    )
    assert row["phase1_passed"] is False
    assert "anthropic" in (row.get("error") or "").lower()


@pytest.mark.asyncio
async def test_run_task_one_shot_requires_one_shot_benchmark():
    """mini-interact tasks (benchmark.one_shot == False) cannot run one-shot."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK)  # no "dataset" => mini_interact
    with pytest.raises(ValueError):
        await agent.run_task(td, "/tmp", 20.0, "slayer", eval_mode="one-shot")


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
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert "resolve_kwargs" in captured
    assert captured["resolve_kwargs"]["benchmark"] == "mini_interact"


@pytest.mark.asyncio
async def test_run_task_attaches_slayer_write_tools(monkeypatch, tmp_path):
    """The ClaudeAgentOptions handed to the SDK must whitelist the slayer
    write tools so the agent can encode."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__slayer__create_model" in allowed
    assert "mcp__slayer__edit_model" in allowed


@pytest.mark.asyncio
async def test_run_task_pins_requested_model(monkeypatch, tmp_path):
    """--agent-model must reach the SDK as the bare native model id, not be
    silently replaced by the claude CLI default (Codex finding)."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-opus-4-7")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
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
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].effort == "high"


@pytest.mark.asyncio
async def test_run_task_default_effort_is_none(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
    )
    assert captured["options"].effort is None


def test_init_rejects_bad_reasoning_effort():
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    with pytest.raises(ValueError):
        ClaudeSDKOtfAgent(reasoning_effort="turbo")


@pytest.mark.asyncio
async def test_run_task_one_shot_livesqlbench(monkeypatch, tmp_path):
    """One-shot LiveSQLBench: storage resolved with benchmark='livesqlbench',
    materialize_task_db called, and the native ask_user tool is NOT whitelisted."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    captured = _stub_env(monkeypatch, m, tmp_path / "store")
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    td = dict(_TASK, dataset="livesqlbench")
    await agent.run_task(td, str(tmp_path), 20.0, "slayer", eval_mode="one-shot")

    assert captured["resolve_kwargs"]["benchmark"] == "livesqlbench"
    assert captured["materialize_calls"] == 1
    allowed = set(captured["options"].allowed_tools)
    assert "mcp__bird-interact-tools__submit_query" in allowed
    assert "mcp__bird-interact-tools__ask_user" not in allowed


@pytest.mark.asyncio
async def test_run_task_captures_usage(monkeypatch, tmp_path):
    from bird_interact_agents.agents.claude_sdk_otf import agent as m
    from bird_interact_agents import usage as usage_mod

    msgs = [_FakeAssistant(100, 20), _FakeAssistant(150, 30, cache=5)]
    _stub_env(monkeypatch, m, tmp_path / "store", messages=msgs)
    agent = m.ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(_TASK), str(tmp_path), 20.0, "slayer", eval_mode="a-interact",
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
