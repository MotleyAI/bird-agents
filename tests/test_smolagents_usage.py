"""Verify the smolagents adapter records token usage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_user_sim_records_tracked_usage(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.smolagents import agent as sa_agent
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
    state = sa_agent.TaskState(
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
async def test_run_task_captures_agent_usage(monkeypatch):
    """`_run_smolagents_sync` must read `agent.monitor.total_input_token_count`
    / `total_output_token_count` and add them to state.usage."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.smolagents import agent as sa_agent

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(sa_agent, "load_db_data_if_needed", lambda *a, **kw: None)

    async def fake_build_prompt(*a, **kw):
        return "instructions"

    monkeypatch.setattr(sa_agent, "_build_prompt", fake_build_prompt)
    monkeypatch.setattr(sa_agent, "_build_native_tools", lambda *a, **kw: [])

    def fake_run_sync(prompt, user_query, native_tools, query_mode,
                     slayer_storage_dir, model_id, strict, state):
        # Mimic the post-`.run()` capture path.
        state.usage.add_call(
            scope="agent",
            model=model_id,
            prompt=400,
            completion=80,
        )
        return "final answer"

    monkeypatch.setattr(sa_agent, "_run_smolagents_sync", fake_run_sync)

    inst = sa_agent.SmolagentsAgent(model_id="cerebras/zai-glm-4.7")
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
    assert rebuilt.prompt_tokens == 400
    assert rebuilt.completion_tokens == 80
    assert any(row.scope == "agent" for row in rebuilt.breakdown)
