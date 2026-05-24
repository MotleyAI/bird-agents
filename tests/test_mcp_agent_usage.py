"""Verify the mcp_agent adapter records user-sim token usage and flags
the agent-side as partial (mcp_agent's SDK does not expose token counts)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_user_sim_records_tracked_usage(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.mcp_agent import agent as mc_agent
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
    state = mc_agent.TaskState(
        status=SampleStatus(
            idx=0,
            original_data={"selected_database": "fake_db", "instance_id": "fake_1"},
        ),
        data_path_base="/tmp/ignored",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    out = await _submit.ask_user_impl(state, "?")

    assert out == "resp"
    assert state.usage.n_calls == 2
    assert state.usage.prompt_tokens == 22
    assert state.usage.completion_tokens == 6
    assert all(row.scope == "user_sim" for row in state.usage.breakdown)


@pytest.mark.asyncio
async def test_run_task_marks_partial_usage(monkeypatch):
    """mcp_agent's SDK does not expose per-call token counts, so the agent
    leg is recorded as `partial=True` rather than fabricating numbers."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.mcp_agent import agent as mc_agent

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(mc_agent, "load_db_data_if_needed", lambda *a, **kw: None)

    async def fake_build_prompt(*a, **kw):
        return "instructions"

    monkeypatch.setattr(mc_agent, "_build_prompt", fake_build_prompt)
    monkeypatch.setattr(mc_agent, "_build_native_functions", lambda *a, **kw: [])
    monkeypatch.setattr(
        mc_agent, "_build_settings", lambda *a, **kw: SimpleNamespace(),
    )

    # Stub the mcp_agent App entirely.
    class _FakeRun:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

    class _FakeApp:
        def __init__(self, **kw):
            pass

        def run(self):
            return _FakeRun()

    class _FakeAgent:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def attach_llm(self, _LLM):
            class _LLMObj:
                async def generate_str(self, **_):
                    return "final answer"
            return _LLMObj()

    import mcp_agent.app as _mcp_app_mod
    import mcp_agent.agents.agent as _mcp_agent_mod
    monkeypatch.setattr(_mcp_app_mod, "MCPApp", _FakeApp)
    monkeypatch.setattr(_mcp_agent_mod, "Agent", _FakeAgent)

    inst = mc_agent.McpAgentAgent(model="anthropic/claude-sonnet-4-5")
    task_data = {
        "selected_database": "fake_db",
        "instance_id": "fake_1",
        "amb_user_query": "?",
    }
    result = await inst.run_task(
        task_data, data_path_base="/tmp/ignored",
        budget=18, query_mode="raw", eval_mode="a-interact",
    )

    assert "usage" in result
    rebuilt = usage_mod.TokenUsage.model_validate(result["usage"])
    assert rebuilt.partial is True
    assert not any(row.scope == "agent" for row in rebuilt.breakdown)
