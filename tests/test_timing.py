"""Verify `run_evaluation` records per-task wall-clock duration and
surfaces leg-level timing aggregates on `eval.json`."""

from __future__ import annotations

import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_run_evaluation_records_per_task_duration(tmp_path, monkeypatch):
    import bird_interact_agents.run as run_mod

    fake_tasks = [
        {"instance_id": f"t{i}", "selected_database": "fake",
         "amb_user_query": f"q{i}"}
        for i in range(2)
    ]
    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: fake_tasks)
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        await asyncio.sleep(0.05)
        return {
            "task_id": td["instance_id"],
            "instance_id": td["instance_id"],
            "database": "fake",
            "phase1_passed": False,
            "phase2_passed": False,
            "total_reward": 0.0,
            "trajectory": [],
            "error": None,
        }

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    output_path = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored",
        output_path=str(output_path),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
    )

    # Per-task durations land on each result.
    for r in metrics["results"]:
        assert r["duration_s"] >= 0.05

    # Aggregates land on the metrics block.
    assert "total_duration_s" in metrics
    assert "avg_duration_s" in metrics
    assert "p50_duration_s" in metrics
    assert "max_duration_s" in metrics
    assert metrics["total_duration_s"] >= 0.10
    assert metrics["avg_duration_s"] >= 0.05
    assert metrics["max_duration_s"] >= 0.05
    # File on disk has the same data.
    on_disk = json.loads(output_path.read_text())
    assert on_disk["total_duration_s"] >= 0.10
    assert on_disk["results"][0]["duration_s"] >= 0.05


@pytest.mark.asyncio
async def test_failing_task_still_gets_duration(tmp_path, monkeypatch):
    """Tasks that raise still get a `duration_s` so per-task latency is
    always present."""
    import bird_interact_agents.run as run_mod

    fake_tasks = [
        {"instance_id": "boom", "selected_database": "fake",
         "amb_user_query": "q"},
    ]
    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: fake_tasks)
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        await asyncio.sleep(0.02)
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    output_path = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored",
        output_path=str(output_path),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
    )

    assert len(metrics["results"]) == 1
    r = metrics["results"][0]
    assert r["error"] == "kaboom"
    assert r["duration_s"] >= 0.02
