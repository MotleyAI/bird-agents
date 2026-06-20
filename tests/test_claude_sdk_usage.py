"""Verify the shared user-simulator call records tracked token usage.

DEV-1534 deleted the pre-OTF ``ClaudeSDKAgent`` orchestrator. The OTF
agents own their own ``run_task`` loops; their AssistantMessage-usage
capture + ``_ctx['result']`` propagation contract is regression-tested
in ``test_claude_sdk_otf_v1_agent.py`` / ``test_claude_sdk_otf_ainteract_v1_agent.py``
(see the DEV-1511 propagation blocks there). This file now keeps only
the shared user-simulator usage test, which exercises the ``_ask_user_impl``
shim in ``claude_sdk/agent.py`` (still alive — every OTF agent reuses
it via the imported ``ask_user`` @tool).
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
