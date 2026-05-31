"""Verify the claude_sdk adapter records token usage.

Two pathways:

1. The user-simulator's two `litellm.acompletion` calls go through
   `acompletion_tracked`, which writes to the contextvar-stored
   `TokenUsage` accumulator.
2. Each `AssistantMessage` / `ResultMessage` from the SDK's
   `receive_response()` loop carries a `usage` block with
   `input_tokens` / `output_tokens` / `cache_read_input_tokens`.

Both tests stub the underlying APIs so no real network call happens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_user_sim_records_tracked_usage(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.claude_sdk import agent as cs_agent
    from bird_interact_agents.harness import SampleStatus, _schema_cache

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="<s>resp</s>"))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
    )

    async def fake_acompletion(**_):
        return fake_resp

    import litellm
    monkeypatch.setattr(usage_mod, "_acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    from bird_interact_agents.agents import _submit
    monkeypatch.setattr(_submit, "build_user_encoder_prompt", lambda *a, **kw: "enc")
    monkeypatch.setattr(_submit, "build_user_decoder_prompt", lambda *a, **kw: "dec")
    monkeypatch.setattr(
        _submit, "parse_encoder_response",
        lambda raw: {"action_type": "answer", "encoded_data": "x"},
    )

    _schema_cache["fake_db"] = "CREATE TABLE foo (x INT);"
    status = SampleStatus(
        idx=0,
        original_data={"selected_database": "fake_db", "instance_id": "fake_1"},
    )
    accum = usage_mod.TokenUsage()
    cs_agent._ctx_var.set({
        "status": status,
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "user_sim_prompt_version": "v2",
        "usage": accum,
    })

    out = await cs_agent._ask_user_impl("any question?")

    assert out == "resp"
    assert accum.n_calls == 2
    assert accum.prompt_tokens == 22
    assert accum.completion_tokens == 6
    assert all(row.scope == "user_sim" for row in accum.breakdown)


@pytest.mark.asyncio
async def test_run_task_captures_assistant_usage(monkeypatch):
    """`run_task` must read each AssistantMessage/ResultMessage's `usage`
    block and merge it into the returned dict's `usage` field."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.claude_sdk import agent as cs_agent

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(cs_agent, "load_db_data_if_needed", lambda *a, **kw: None)

    # Stub _build_prompt and _select_tools to skip slayer setup.
    async def fake_build_prompt(*a, **kw):
        return "instructions"

    monkeypatch.setattr(cs_agent, "_build_prompt", fake_build_prompt)
    monkeypatch.setattr(cs_agent, "_select_tools", lambda *a, **kw: [])

    # Stub create_sdk_mcp_server (it tries to wire MCP servers)
    monkeypatch.setattr(
        cs_agent, "create_sdk_mcp_server", lambda **kw: SimpleNamespace(),
    )

    # Build fake messages: two AssistantMessage-shaped objects with usage.
    class _FakeAssistant:
        def __init__(self, in_, out_, cache=0):
            self.usage = SimpleNamespace(
                input_tokens=in_, output_tokens=out_,
                cache_read_input_tokens=cache,
            )

    _FakeAssistant.__name__ = "AssistantMessage"

    fake_messages = [_FakeAssistant(100, 20), _FakeAssistant(150, 30, cache=5)]

    class _FakeClient:
        def __init__(self, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def query(self, *a, **kw):
            return None

        async def receive_response(self):
            for m in fake_messages:
                yield m

    monkeypatch.setattr(cs_agent, "ClaudeSDKClient", _FakeClient)

    inst = cs_agent.ClaudeSDKAgent(model="anthropic/claude-sonnet-4-5")
    task_data = {
        "selected_database": "fake_db",
        "instance_id": "fake_1",
        "amb_user_query": "?",
        "ambiguity": [],
    }
    result = await inst.run_task(
        task_data, data_path_base="/tmp/ignored",
        budget=18, query_mode="raw", eval_mode="a-interact",
    )

    assert "usage" in result
    rebuilt = usage_mod.TokenUsage.model_validate(result["usage"])
    assert rebuilt.prompt_tokens == 250
    assert rebuilt.completion_tokens == 50
    assert rebuilt.cache_read_tokens == 5
    assert any(row.scope == "agent" for row in rebuilt.breakdown)


# ---------------------------------------------------------------------------
# DEV-1511: diagnostic-field propagation from _ctx["result"] to the
# finalized row for the non-OTF `claude_sdk` adapter (mirrors the OTF
# tests in test_claude_sdk_otf_agent.py). The submit helpers populate
# `submission_status` / `predicted_result_json` / `gold_result_json` /
# `phase1_observation` (+ phase2 variant) on `state.result`; the bug:
# `run_task` did not propagate them.
# ---------------------------------------------------------------------------


class _FakeAssistant:
    def __init__(self, in_, out_, cache=0):
        self.usage = SimpleNamespace(
            input_tokens=in_, output_tokens=out_, cache_read_input_tokens=cache,
        )


_FakeAssistant.__name__ = "AssistantMessage"


def _make_propagation_client(
    captured: dict, messages, *,
    cs_agent_mod, prefill_result=None, prefill_timing="after",
    raise_after_prefill=None,
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
                cs_agent_mod._ctx_var.get()["result"] = dict(prefill_result)
            for msg in messages:
                yield msg
            if prefill_result is not None and prefill_timing == "after":
                cs_agent_mod._ctx_var.get()["result"] = dict(prefill_result)
            if raise_after_prefill is not None:
                raise raise_after_prefill

    return _FakeClient


def _stub_cs_env(
    monkeypatch, *,
    messages=(), prefill_result=None, prefill_timing="after",
    raise_after_prefill=None,
):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.claude_sdk import agent as cs_agent

    captured: dict = {}
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(cs_agent, "load_db_data_if_needed", lambda *a, **kw: None)

    async def fake_build_prompt(*a, **kw):
        return "instructions"

    monkeypatch.setattr(cs_agent, "_build_prompt", fake_build_prompt)
    monkeypatch.setattr(cs_agent, "_select_tools", lambda *a, **kw: [])
    monkeypatch.setattr(
        cs_agent, "create_sdk_mcp_server", lambda **kw: SimpleNamespace(),
    )

    async def fake_resolve(*, slayer_storage_root, db_name, task_data, query_mode):
        return "", []

    monkeypatch.setattr(cs_agent, "resolve_task_storage_dir", fake_resolve)
    monkeypatch.setattr(
        cs_agent, "ClaudeSDKClient",
        _make_propagation_client(
            captured, messages,
            cs_agent_mod=cs_agent,
            prefill_result=prefill_result,
            prefill_timing=prefill_timing,
            raise_after_prefill=raise_after_prefill,
        ),
    )
    return cs_agent, captured


def _cs_task():
    return {
        "selected_database": "fake_db",
        "instance_id": "fake_1",
        "amb_user_query": "?",
        "ambiguity": [],
    }


def _full_cs_prefill(**overrides):
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
async def test_run_task_propagates_diagnostic_fields_on_happy_path(monkeypatch):
    cs_agent, _ = _stub_cs_env(
        monkeypatch,
        messages=[_FakeAssistant(100, 20)],
        prefill_result=_full_cs_prefill(),
        prefill_timing="after",
    )
    inst = cs_agent.ClaudeSDKAgent(model="anthropic/claude-sonnet-4-5")
    row = await inst.run_task(
        _cs_task(), data_path_base="/tmp/ignored",
        budget=18, query_mode="raw", eval_mode="a-interact",
    )
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    # phase2_observation absent from prefill => key present with value None
    assert "phase2_observation" in row
    assert row["phase2_observation"] is None
    assert row["phase1_passed"] is True
    assert row["submitted_query"] == "{\"models\": [\"m\"]}"
    assert row["error"] is None


@pytest.mark.asyncio
async def test_run_task_propagates_phase2_observation(monkeypatch):
    cs_agent, _ = _stub_cs_env(
        monkeypatch,
        messages=[_FakeAssistant(100, 20)],
        prefill_result={
            "submission_status": "wrong_result",
            "phase1_passed": True,
            "phase2_passed": False,
            "phase2_observation": "p2 fail observation",
        },
    )
    inst = cs_agent.ClaudeSDKAgent(model="anthropic/claude-sonnet-4-5")
    row = await inst.run_task(
        _cs_task(), data_path_base="/tmp/ignored",
        budget=18, query_mode="raw", eval_mode="a-interact",
    )
    assert row["phase2_observation"] == "p2 fail observation"
    assert "phase1_observation" in row
    assert row["phase1_observation"] is None


@pytest.mark.asyncio
async def test_run_task_propagation_defaults_to_none_when_never_submitted(
    monkeypatch,
):
    """Adapter contract: row carries None (not the misleading
    `"never_submitted"` sentinel) when no submit happened. The sentinel
    lives only in `run.py`'s downstream setdefault, not in this row."""
    cs_agent, _ = _stub_cs_env(
        monkeypatch, messages=[_FakeAssistant(100, 20)],
    )
    inst = cs_agent.ClaudeSDKAgent(model="anthropic/claude-sonnet-4-5")
    row = await inst.run_task(
        _cs_task(), data_path_base="/tmp/ignored",
        budget=18, query_mode="raw", eval_mode="a-interact",
    )
    assert row["submission_status"] is None
    assert row["predicted_result_json"] is None
    assert row["gold_result_json"] is None
    assert row["phase1_observation"] is None
    assert row["phase2_observation"] is None


@pytest.mark.asyncio
async def test_run_task_exception_path_propagates_partial_result(monkeypatch):
    """SDK loop crashes AFTER a successful submit; finalize must rescue
    `_ctx["result"]` diagnostics rather than dropping them. Asserts the
    full mirror of the happy-path field set — including pre-existing
    fields (`phase2_passed`, `total_reward`, dual-eval columns,
    `phase2_observation`) — so a rewrite cannot silently drop any."""
    prefill = _full_cs_prefill(
        phase2_passed=True, total_reward=0.75,
        phase2_observation="p2 ok",
        phase1_observation_audited="audited-obs",
        phase1_observation_original="original-obs",
    )
    cs_agent, _ = _stub_cs_env(
        monkeypatch,
        messages=[_FakeAssistant(100, 20)],
        prefill_result=prefill,
        prefill_timing="after",
        raise_after_prefill=RuntimeError("boom"),
    )
    inst = cs_agent.ClaudeSDKAgent(model="anthropic/claude-sonnet-4-5")
    row = await inst.run_task(
        _cs_task(), data_path_base="/tmp/ignored",
        budget=18, query_mode="raw", eval_mode="a-interact",
    )
    assert row["error"] == "boom"
    # 5 new diagnostic fields
    assert row["submission_status"] == "submitted_ok"
    assert row["predicted_result_json"] == "[{\"a\": 1}]"
    assert row["gold_result_json"] == "[{\"a\": 1}]"
    assert row["phase1_observation"] == "PASS"
    assert row["phase2_observation"] == "p2 ok"
    # Pre-existing rescue
    assert row["phase1_passed"] is True
    assert row["phase2_passed"] is True
    assert row["total_reward"] == 0.75
    assert row["submitted_query"] == "{\"models\": [\"m\"]}"
    assert row["submitted_sql"] == "SELECT 1"
    # Dual-eval columns: still pass-through.
    assert row["phase1_observation_audited"] == "audited-obs"
    assert row["phase1_observation_original"] == "original-obs"
