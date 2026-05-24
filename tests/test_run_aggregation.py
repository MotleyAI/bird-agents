"""Verify `run_evaluation` sums per-task usage into a top-level
`total_usage` block on `eval.json`."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_run_evaluation_aggregates_usage(tmp_path, monkeypatch):
    from bird_interact_agents import usage as usage_mod
    import bird_interact_agents.run as run_mod

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    # Two stub tasks; oracle mode has its own runner that doesn't need an
    # LLM, but we replace the runner-selector so we don't need a real DB.
    fake_tasks = [
        {"instance_id": "t1", "selected_database": "fake", "amb_user_query": "q1"},
        {"instance_id": "t2", "selected_database": "fake", "amb_user_query": "q2"},
    ]
    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: fake_tasks)
    monkeypatch.setattr(
        run_mod, "calculate_budget", lambda *a, **kw: 18,
    )

    async def fake_oracle(td, dpb):
        # Each task contributes a usage block.
        u = usage_mod.TokenUsage()
        u.add_call(scope="agent", model="m", prompt=100, completion=20)
        u.add_call(scope="user_sim", model="u", prompt=200, completion=30)
        return {
            "task_id": td["instance_id"],
            "instance_id": td["instance_id"],
            "database": "fake",
            "phase1_passed": False,
            "phase2_passed": False,
            "total_reward": 0.0,
            "trajectory": [],
            "error": None,
            "usage": u.model_dump(),
        }

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    output_path = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        data_path="ignored",
        data_dir="ignored",
        output_path=str(output_path),
        mode="oracle",
        query_mode="raw",
        framework="pydantic_ai",  # ignored in oracle mode
        concurrency=1,
    )

    assert "total_usage" in metrics
    total = usage_mod.TokenUsage.model_validate(metrics["total_usage"])
    assert total.prompt_tokens == (100 + 200) * 2
    assert total.completion_tokens == (20 + 30) * 2
    # Two scopes, one model each ⇒ two breakdown rows.
    assert {row.scope for row in total.breakdown} == {"agent", "user_sim"}

    # File on disk has the same data.
    on_disk = json.loads(output_path.read_text())
    assert "total_usage" in on_disk
    assert on_disk["total_usage"]["prompt_tokens"] == (100 + 200) * 2
