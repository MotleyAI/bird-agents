"""Verify the agno adapter records token usage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_user_sim_records_tracked_usage(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.agno import agent as ag_agent
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
    state = ag_agent.TaskState(
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
async def test_run_task_captures_agent_metrics(monkeypatch):
    """Agno's `RunResponse.metrics` records input/output token counts;
    `run_task` must read those and put them on the result dict."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.agno import agent as ag_agent

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(ag_agent, "load_db_data_if_needed", lambda *a, **kw: None)

    async def fake_build_prompt(*a, **kw):
        return "instructions"

    monkeypatch.setattr(ag_agent, "_build_prompt", fake_build_prompt)
    monkeypatch.setattr(ag_agent, "_build_native_tools", lambda *a, **kw: [])

    fake_response = SimpleNamespace(
        content="final answer",
        metrics=SimpleNamespace(input_tokens=300, output_tokens=60),
    )

    class _FakeAgent:
        def __init__(self, **kw):
            self.kw = kw

        async def arun(self, *a, **kw):
            return fake_response

    import agno.agent as _agno_agent_mod
    monkeypatch.setattr(_agno_agent_mod, "Agent", _FakeAgent)
    # Stub the model class so it doesn't require an API key.
    import agno.models.anthropic as _agno_anthropic_mod
    monkeypatch.setattr(
        _agno_anthropic_mod, "Claude",
        lambda **kw: SimpleNamespace(id=kw.get("id")),
    )

    inst = ag_agent.AgnoAgent(model_id="anthropic/claude-sonnet-4-5")
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
    assert rebuilt.prompt_tokens == 300
    assert rebuilt.completion_tokens == 60
    assert any(row.scope == "agent" for row in rebuilt.breakdown)
