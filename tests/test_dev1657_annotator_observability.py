"""DEV-1657: annotator observability (trajectory + usage) and the forced-submit
nudge.

Motivation: `doubleword/zai-org/GLM-5.2-FP8` ran the full 60-turn cap twice as
the annotator and never called `submit_annotation` ("Agent did not submit an
annotation after 60 turns"), but `run_task` persisted NO trajectory and NO
usage — so the failure could not be diagnosed at all. These tests pin:

1. `run_task` populates `AnnotatorResult.trajectory` (serialized SDK messages)
   on BOTH the success and the never-submitted error paths.
2. `run_task` populates `AnnotatorResult.usage` from the SDK stream (it was
   always `{}` before, hence the all-zero eval.json).
3. A forced-submit nudge: when the exploration budget is spent without a
   submission, `run_task` issues one explicit "submit now" prompt and lets the
   model recover, instead of silently failing.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Task data + a valid annotation payload
# ---------------------------------------------------------------------------

def _task_mini(instance_id: str = "shop_1") -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": "shop",
        "amb_user_query": "How many premium orders?",
        "sol_sql": ["SELECT COUNT(*) FROM orders WHERE tier='Premium';"],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
        "knowledge_ambiguity": [],
        "external_knowledge": [3],
    }


def _valid_ta_json(instance_id: str = "shop_1", db: str = "shop") -> str:
    return json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-07-08",
        "amb_user_query": "How many premium orders?",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 3 pins the tier.",
            "evidence_sources_consulted": ["kb:3"],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": instance_id,
        },
    })


# ---------------------------------------------------------------------------
# Configurable fake SDK client
# ---------------------------------------------------------------------------

class AssistantMessage:
    """Stand-in for the SDK AssistantMessage — `type(msg).__name__` must be
    exactly 'AssistantMessage' for the run loop's turn counter and usage
    observer to fire."""

    def __init__(self, usage: dict | None = None) -> None:
        self.usage = usage


class ResultMessage:
    """Stand-in for the SDK terminal ResultMessage; observing one commits the
    per-cycle usage tracker. Used to prove the nudge round's usage is NOT
    dropped by a tracker that already finalized on the exploration cycle."""

    def __init__(self, usage: dict | None = None) -> None:
        self.usage = usage


def _install_fake_sdk(monkeypatch, ann_agent, *, rounds):
    """Install a fake ClaudeSDKClient driven by `rounds`.

    Each call to `receive_response()` consumes the next entry of `rounds` (a
    list of "actions"). An action is either:
      * ("turn", usage_dict|None)          → yield one AssistantMessage
      * ("submit", ta_json, variants_json) → call submit_annotation, then yield
    Rounds beyond the list length yield nothing (empty stream). Returns the
    list that records every `query()` prompt (initial + any nudge).
    """
    rounds_iter = iter(rounds)
    queries: list[str] = []

    class FakeClient:
        def __init__(self, options):
            self._options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def get_mcp_status(self):
            names = list((self._options.mcp_servers or {}).keys())
            return {"mcpServers": [{"name": n, "status": "connected"} for n in names]}

        async def query(self, prompt, *a, **kw):
            queries.append(prompt)

        async def receive_response(self):
            try:
                actions = next(rounds_iter)
            except StopIteration:
                actions = []
            for action in actions:
                if action[0] == "submit":
                    await ann_agent.submit_annotation({
                        "task_annotation_json": action[1],
                        "audited_gold_variants_json": action[2],
                    })
                    yield AssistantMessage()
                elif action[0] == "result":
                    yield ResultMessage(usage=action[1])
                else:  # ("turn", usage)
                    yield AssistantMessage(usage=action[1])
                if ann_agent._ctx.get("_submission_done"):
                    break

    from bird_interact_agents.agents.claude_sdk import sdk_env as _sdk_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-stub")
    # Pin the auth path to the API-key branch. Another test (run.py
    # --subscription-auth) sets BIRD_INTERACT_SUBSCRIPTION_AUTH process-wide and
    # can leak into later tests; without clearing it, hermetic_claude_sdk_session
    # takes the subscription path and hard-fails (no CLAUDE_CODE_OAUTH_TOKEN).
    monkeypatch.delenv("BIRD_INTERACT_SUBSCRIPTION_AUTH", raising=False)
    monkeypatch.setattr(_sdk_env, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(ann_agent, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(ann_agent, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(ann_agent, "materialize_task_db", lambda *a, **kw: None)
    return queries


async def _run(ann_agent, *, max_turns=None):
    return await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="anthropic/claude-opus-4-7",
        effort="medium",
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# 1. Trajectory capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trajectory_captured_on_success(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _install_fake_sdk(monkeypatch, ann_agent, rounds=[[("submit", _valid_ta_json(), "[]")]])
    result = await _run(ann_agent)

    assert result.error is None
    assert isinstance(result.trajectory, list)
    assert len(result.trajectory) >= 1
    # Each entry is a serialized message dict (never raw objects).
    for entry in result.trajectory:
        assert isinstance(entry, dict)
    # Round-trips through JSON (cloud persists it as a JSON blob).
    json.dumps(result.trajectory, default=str)


@pytest.mark.asyncio
async def test_trajectory_captured_on_never_submitted(monkeypatch):
    """The whole point of DEV-1657: a never-submitted run must still leave a
    trajectory so the failure can be diagnosed."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    # cap=2 → no reserve/nudge; two turns, no submit → error.
    _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[[("turn", None), ("turn", None)]],
    )
    result = await _run(ann_agent, max_turns=2)

    assert result.task_annotation is None
    assert result.error is not None
    assert "did not submit" in result.error
    assert len(result.trajectory) >= 1


# ---------------------------------------------------------------------------
# 2. Usage capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_populated_from_stream(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[[
            ("turn", {"input_tokens": 100, "output_tokens": 20}),
            ("submit", _valid_ta_json(), "[]"),
        ]],
    )
    result = await _run(ann_agent)

    assert result.error is None
    assert isinstance(result.usage, dict)
    assert result.usage.get("prompt_tokens", 0) == 100
    assert result.usage.get("completion_tokens", 0) == 20
    assert result.usage.get("n_calls", 0) >= 1


@pytest.mark.asyncio
async def test_usage_populated_on_never_submitted(monkeypatch):
    """Usage must be captured even when the task fails — otherwise cost/turn
    accounting for a doomed run reads as zero (the original symptom)."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[[("turn", {"input_tokens": 50, "output_tokens": 10})]],
    )
    result = await _run(ann_agent, max_turns=1)

    assert result.error is not None
    assert result.usage.get("prompt_tokens", 0) == 50


# ---------------------------------------------------------------------------
# 3. Forced-submit nudge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_nudge_recovers_when_exploration_did_not_submit(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    # cap=4 → reserve=2, explore_cap=2. Round 1 spends the 2 exploration turns
    # without submitting; the nudge round then submits.
    queries = _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[
            [("turn", None), ("turn", None)],       # exploration: no submit
            [("submit", _valid_ta_json(), "[]")],   # nudge round: submit
        ],
    )
    result = await _run(ann_agent, max_turns=4)

    assert result.error is None, result.error
    assert result.task_annotation is not None
    # Exactly one nudge was issued: initial query + one forced-submit prompt.
    assert len(queries) == 2
    assert queries[1] == ann_agent._FORCE_SUBMIT_NUDGE


@pytest.mark.asyncio
async def test_usage_counted_across_nudge_round(monkeypatch):
    """Usage from the forced-submit nudge cycle must be added to the total. A
    single tracker reused across cycles finalizes on the exploration cycle's
    ResultMessage and would drop the nudge round — this pins the per-cycle
    tracker fix."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    # cap=10 → reserve=5, explore_cap=5. Exploration ends with a ResultMessage
    # (usage=60) after 2 turns; the nudge round contributes another 40.
    _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[
            [
                ("turn", {"input_tokens": 30, "output_tokens": 5}),
                ("turn", {"input_tokens": 30, "output_tokens": 5}),
                ("result", {"input_tokens": 60, "output_tokens": 10}),
            ],
            [
                ("turn", {"input_tokens": 40, "output_tokens": 8}),
                ("submit", _valid_ta_json(), "[]"),
            ],
        ],
    )
    result = await _run(ann_agent, max_turns=10)

    assert result.error is None, result.error
    # 60 (exploration ResultMessage) + 40 (nudge round) — the nudge is NOT lost.
    assert result.usage.get("prompt_tokens", 0) == 100


@pytest.mark.asyncio
async def test_no_nudge_when_submitted_during_exploration(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    queries = _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[[("submit", _valid_ta_json(), "[]")]],
    )
    result = await _run(ann_agent, max_turns=4)

    assert result.error is None
    # Only the initial query — no nudge when the model submits on its own.
    assert len(queries) == 1


@pytest.mark.asyncio
async def test_nudge_fires_at_most_once(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    # cap=6 → reserve=3, explore_cap=3. Neither round submits: one nudge, then
    # the hard cap ends it. Never more than one nudge (no infinite loop).
    queries = _install_fake_sdk(
        monkeypatch, ann_agent,
        rounds=[
            [("turn", None), ("turn", None), ("turn", None)],  # exploration
            [("turn", None), ("turn", None), ("turn", None)],  # nudge round
        ],
    )
    result = await _run(ann_agent, max_turns=6)

    assert result.error is not None
    assert "did not submit" in result.error
    assert len(queries) == 2  # initial + exactly one nudge
