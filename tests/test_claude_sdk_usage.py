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
