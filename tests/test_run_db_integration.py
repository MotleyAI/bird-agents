"""run_evaluation must write a row to results.db immediately after each
task finishes — not at the end of the run."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_db_row_appears_after_each_task(tmp_path, monkeypatch):
    """Verify the per-task DB write isn't deferred to end-of-run."""
    import bird_interact_agents.run as run_mod

    fake_tasks = [
        {"instance_id": "t1", "selected_database": "fake",
         "amb_user_query": "q1", "sol_sql": ["SELECT 1"]},
        {"instance_id": "t2", "selected_database": "fake",
         "amb_user_query": "q2", "sol_sql": ["SELECT 2"]},
    ]
    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: fake_tasks)
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    db_path = tmp_path / "results.db"
    seen_rows: list[int] = []

    async def fake_oracle(td, dpb):
        # Take a beat so the test can observe the gap between task 1
        # and task 2.
        await asyncio.sleep(0.01)
        # Mid-task probe: count rows already on disk.
        if td["instance_id"] == "t2":
            with sqlite3.connect(db_path) as c:
                seen_rows.append(c.execute(
                    "SELECT COUNT(*) FROM task_results"
                ).fetchone()[0])
        return {
            "task_id": td["instance_id"],
            "instance_id": td["instance_id"],
            "database": "fake",
            "phase1_passed": False,
            "phase2_passed": False,
            "total_reward": 0.0,
            "trajectory": [],
            "error": None,
            "submitted_sql": f"SQL for {td['instance_id']}",
            "ground_truth_sql": td["sol_sql"][0],
        }

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    output_path = tmp_path / "eval.json"
    await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored",
        output_path=str(output_path),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,  # serial so the probe is deterministic
        agent_model="anthropic/claude-haiku-4-5-20251001",
    )

    # While task 2 was running, task 1's row was already on disk.
    assert seen_rows == [1]

    # Final state: both rows present, with submitted_sql + ground_truth_sql.
    with sqlite3.connect(db_path) as c:
        rows = list(c.execute(
            "SELECT instance_id, submitted_sql, ground_truth_sql "
            "FROM task_results ORDER BY instance_id"
        ))
    assert rows == [
        ("t1", "SQL for t1", "SELECT 1"),
        ("t2", "SQL for t2", "SELECT 2"),
    ]


@pytest.mark.asyncio
async def test_failed_task_still_lands_in_db(tmp_path, monkeypatch):
    """Tasks that raise must still produce a row (with `error` populated)
    so a crashing run doesn't lose evidence of which tasks were
    attempted."""
    import bird_interact_agents.run as run_mod

    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: [
        {"instance_id": "boom", "selected_database": "fake",
         "amb_user_query": "q"},
    ])
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        raise RuntimeError("intentional explosion")

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    db_path = tmp_path / "results.db"
    await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored",
        output_path=str(tmp_path / "eval.json"),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
        agent_model="anthropic/claude-haiku-4-5-20251001",
    )

    with sqlite3.connect(db_path) as c:
        rows = list(c.execute(
            "SELECT instance_id, error FROM task_results"
        ))
    assert rows == [("boom", "intentional explosion")]


@pytest.mark.asyncio
async def test_diagnostic_columns_persist(tmp_path, monkeypatch):
    """Diagnostic fields produced by submit_* must reach the DB row.

    The harness now classifies every submission and stores the
    classifier verdict + observation + result snapshots. Verify the
    `_persist` plumbing forwards all of those through to results.db.
    """
    import bird_interact_agents.run as run_mod

    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: [
        {"instance_id": "t1", "selected_database": "fake",
         "amb_user_query": "what is the count?", "sol_sql": ["SELECT 1"]},
    ])
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        return {
            "task_id": td["instance_id"], "instance_id": td["instance_id"],
            "database": "fake", "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "trajectory": [], "error": None,
            "submission_status": "wrong_result",
            "phase1_observation": "Default test case failed: rows differ",
            "phase2_observation": None,
            "predicted_result_json": '{"row_count": 3}',
            "gold_result_json": '{"row_count": 5}',
            "n_agent_turns": 4,
        }
    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    db_path = tmp_path / "results.db"
    await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored",
        output_path=str(tmp_path / "eval.json"),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
        agent_model="anthropic/claude-haiku-4-5-20251001",
    )

    with sqlite3.connect(db_path) as c:
        rows = list(c.execute(
            "SELECT instance_id, user_query, submission_status, "
            "phase1_observation, predicted_result_json, gold_result_json, "
            "n_agent_turns FROM task_results"
        ))
    assert rows == [(
        "t1", "what is the count?", "wrong_result",
        "Default test case failed: rows differ",
        '{"row_count": 3}', '{"row_count": 5}', 4,
    )]


@pytest.mark.asyncio
async def test_run_metadata_recorded(tmp_path, monkeypatch):
    import bird_interact_agents.run as run_mod

    monkeypatch.setattr(run_mod, "load_tasks", lambda *a, **kw: [
        {"instance_id": "t1", "selected_database": "fake",
         "amb_user_query": "q"},
    ])
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        return {
            "task_id": td["instance_id"],
            "instance_id": td["instance_id"],
            "database": "fake",
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "trajectory": [], "error": None,
        }

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    db_path = tmp_path / "results.db"
    await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored",
        output_path=str(tmp_path / "eval.json"),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
    )

    with sqlite3.connect(db_path) as c:
        rows = list(c.execute(
            "SELECT agent_model, framework, mode FROM run_metadata"
        ))
    assert rows == [(
        "anthropic/claude-haiku-4-5-20251001", "pydantic_ai", "oracle",
    )]
