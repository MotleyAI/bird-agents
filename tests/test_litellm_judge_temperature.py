"""LiteLLMJudge must not send an explicit ``temperature`` by default.

Newer Anthropic models (e.g. claude-opus-4-7) reject an explicit temperature
with a 400 ("temperature is deprecated for this model"). The judge previously
sent ``temperature=0.0`` unconditionally, so the call 400'd and ``judge()``
returned ``None`` — silently collapsing an ``insufficient`` task to
``agent_miss`` (no novel-reading verdict). The judge now omits temperature
unless a caller explicitly opts in.
"""
from __future__ import annotations

from typing import Any, Dict, List

from bird_interact_agents.eval.tolerant_grader import LiteLLMJudge


def _ok_response(decision: str) -> Dict[str, Any]:
    return {"choices": [{"message": {"content": f"reasoning...\n{decision}"}}]}


def test_judge_omits_temperature_by_default(monkeypatch):
    import litellm

    calls: List[Dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return _ok_response("ACCEPT")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    judge = LiteLLMJudge(model="anthropic/claude-opus-4-7")
    verdict = judge.judge(instance_id="households_15", evaluator_prompt="judge me")

    assert verdict is True
    assert len(calls) == 1
    assert "temperature" not in calls[0], "temperature must NOT be sent by default"
    assert calls[0]["model"] == "anthropic/claude-opus-4-7"


def test_judge_sends_temperature_only_when_opted_in(monkeypatch):
    import litellm

    calls: List[Dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return _ok_response("REJECT")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    judge = LiteLLMJudge(model="anthropic/claude-sonnet-4-6", temperature=0.0)
    verdict = judge.judge(instance_id="x", evaluator_prompt="p")

    assert verdict is False
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0


def test_judge_returns_none_on_error(monkeypatch):
    import litellm

    calls: List[Dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        raise Exception("overloaded_error: server is busy")

    monkeypatch.setattr(litellm, "completion", fake_completion)

    judge = LiteLLMJudge(model="anthropic/claude-opus-4-7")
    verdict = judge.judge(instance_id="x", evaluator_prompt="p")

    assert verdict is None
    assert len(calls) == 1, "a single attempt; no retry machinery"
