"""Verify the pydantic_ai adapter records token usage.

Two pathways are wired:

1. The user-simulator's two `litellm.acompletion` calls go through
   `acompletion_tracked`, which writes to `TaskDeps.usage`.
2. The main agent's run returns a `RunUsage`, which `run_task` reads via
   `agent_run.usage()` and folds into the same accumulator.

Both tests stub the underlying APIs so no real network call happens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_user_sim_records_tracked_usage(monkeypatch):
    """The pydantic_ai adapter's user-sim path goes through the shared
    `_submit.ask_user_impl` helper. End-to-end coverage of the user-sim
    accumulator lives in test_submit_helpers.py — this test is just a
    smoke check that the pydantic_ai adapter still threads a TaskDeps
    through to it."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents import _submit
    from bird_interact_agents.agents.pydantic_ai import agent as pa_agent
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

    monkeypatch.setattr(_submit, "build_user_encoder_prompt", lambda *a, **kw: "enc")
    monkeypatch.setattr(_submit, "build_user_decoder_prompt", lambda *a, **kw: "dec")
    monkeypatch.setattr(
        _submit, "parse_encoder_response",
        lambda raw: {"action_type": "answer", "encoded_data": "x"},
    )

    monkeypatch.setitem(_schema_cache, "fake_db", "CREATE TABLE foo (x INT);")
    status = SampleStatus(
        idx=0,
        original_data={"selected_database": "fake_db", "instance_id": "fake_1"},
    )
    deps = pa_agent.TaskDeps(
        status=status,
        data_path_base="/tmp/ignored",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
    )

    out = await _submit.ask_user_impl(deps, "any question?")

    assert out == "resp"
    assert deps.usage.n_calls == 2
    assert deps.usage.prompt_tokens == 22
    assert deps.usage.completion_tokens == 6
    assert all(row.scope == "user_sim" for row in deps.usage.breakdown)


@pytest.mark.asyncio
async def test_run_task_captures_agent_usage(monkeypatch):
    """`run_task` must read `agent_run.usage()` and merge it into the
    returned dict's `usage` field."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai import agent as pa_agent

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    fake_run_usage = SimpleNamespace(
        input_tokens=999,
        output_tokens=42,
        cache_read_tokens=0,
    )
    fake_run_result = SimpleNamespace(
        output="final answer",
        usage=lambda: fake_run_usage,
    )

    class _FakeAgent:
        async def run(self, **_):
            return fake_run_result

    monkeypatch.setattr(
        pa_agent.PydanticAIAgent, "_select_agent",
        lambda self, *a, **kw: _FakeAgent(),
    )

    async def fake_build_prompt(self, *a, **kw):
        return "instructions"

    monkeypatch.setattr(
        pa_agent.PydanticAIAgent, "_build_prompt", fake_build_prompt,
    )
    monkeypatch.setattr(pa_agent, "load_db_data_if_needed", lambda *a, **kw: None)

    inst = pa_agent.PydanticAIAgent(model="anthropic/claude-sonnet-4-5")
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
    blob = result["usage"]
    rebuilt = usage_mod.TokenUsage.model_validate(blob)
    assert rebuilt.prompt_tokens == 999
    assert rebuilt.completion_tokens == 42
    assert any(row.scope == "agent" for row in rebuilt.breakdown)
