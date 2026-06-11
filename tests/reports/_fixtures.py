"""Synthetic trajectory + run-layout builders for the reports test suite.

Kept separate from ``conftest.py`` so they can be imported directly from
test modules (``from tests.reports._fixtures import …``) without depending
on pytest fixture injection.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Trajectory step builders
# ---------------------------------------------------------------------------


def assistant_msg(
    *,
    model: str = "claude-opus-4-7",
    thinking: str = "",
    text: str = "",
    tool_use: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One AssistantMessage trajectory entry."""
    content: list[dict[str, Any]] = []
    if thinking:
        content.append({"thinking": thinking, "signature": "sig"})
    if text:
        content.append({"text": text})
    if tool_use is not None:
        content.append(tool_use)
    return {
        "type": "AssistantMessage",
        "data": {
            "content": content,
            "model": model,
            "parent_tool_use_id": None,
            "error": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "message_id": "msg_synthetic",
            "stop_reason": "tool_use" if tool_use is not None else "end_turn",
            "session_id": "sess_synthetic",
        },
    }


def tool_use_block(
    *, tool_use_id: str, name: str, inp: dict[str, Any]
) -> dict[str, Any]:
    return {"id": tool_use_id, "name": name, "input": inp, "type": "tool_use"}


def tool_result_msg(*, tool_use_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "UserMessage",
        "data": {
            "content": [
                {
                    "tool_use_id": tool_use_id,
                    "type": "tool_result",
                    "content": content,
                }
            ],
            "uuid": "uuid_synthetic",
            "parent_tool_use_id": None,
            "tool_use_result": {"text": content},
        },
    }


def user_text_msg(*, text: str) -> dict[str, Any]:
    return {
        "type": "UserMessage",
        "data": {
            "content": [{"type": "text", "text": text}],
            "uuid": "uuid_user_text",
            "parent_tool_use_id": None,
            "tool_use_result": None,
        },
    }


def system_msg(*, text: str = "system-init") -> dict[str, Any]:
    return {
        "type": "SystemMessage",
        "data": {"subtype": "init", "data": {"text": text}},
    }


# ---------------------------------------------------------------------------
# Whole-trajectory builders
# ---------------------------------------------------------------------------


def build_trajectory(
    *,
    instance_id: str = "alien_1",
    database: str = "alien",
    task_id: str | None = None,
    phase1_passed: bool = True,
    phase2_passed: bool = False,
    submitted_sql: str = "SELECT 1",
    ground_truth_sql: str = "SELECT 1",
    submission_status: str = "passed_phase1",
    phase1_observation: str | None = "Phase 1 SQL Correct! (Reward: 1 points). No Phase 2. Task finished.",
    phase2_observation: str | None = None,
    trajectory_steps: list[dict[str, Any]] | None = None,
    error: str | None = None,
    duration_s: float = 12.3,
    n_agent_turns: int = 3,
) -> dict[str, Any]:
    return {
        "task_id": task_id or instance_id,
        "instance_id": instance_id,
        "database": database,
        "phase1_passed": phase1_passed,
        "phase2_passed": phase2_passed,
        "total_reward": float(int(phase1_passed) + int(phase2_passed)),
        "submitted_sql": submitted_sql,
        "submitted_query": submitted_sql,
        "submission_status": submission_status,
        "predicted_result_json": json.dumps({"row_count": 0, "sample_rows": []}),
        "gold_result_json": json.dumps({"row_count": 0, "sample_rows": []}),
        "phase1_observation": phase1_observation,
        "phase2_observation": phase2_observation,
        "trajectory": trajectory_steps or [],
        "error": error,
        "usage": {"cost_usd": 0.0, "breakdown": []},
        "ground_truth_sql": ground_truth_sql,
        "n_agent_turns": n_agent_turns,
        "duration_s": duration_s,
    }


def trajectory_one_phase_pass(
    *, sql: str = "SELECT 1", instance_id: str = "alien_1"
) -> dict[str, Any]:
    steps = [
        system_msg(),
        user_text_msg(text="Find rows where x = 1."),
        assistant_msg(
            text="Submitting.",
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": sql},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_1",
            content="Phase 1 SQL Correct! (Reward: 1 points). No Phase 2. Task finished.",
        ),
    ]
    return build_trajectory(
        instance_id=instance_id,
        submitted_sql=sql,
        submission_status="passed_phase1",
        trajectory_steps=steps,
    )


def trajectory_two_phase_pass(
    *,
    phase1_sql: str = "SELECT 1",
    phase2_sql: str = "SELECT 2",
    instance_id: str = "alien_2",
) -> dict[str, Any]:
    steps = [
        system_msg(),
        user_text_msg(text="Find rows."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_p1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": phase1_sql},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_p1",
            content="Phase 1 SQL Correct! (Reward: 1 points). Moving to Phase 2.",
        ),
        user_text_msg(text="Now also include the follow-up."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_p2",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": phase2_sql},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_p2",
            content="Phase 2 SQL Correct! (Reward: 1 points). Task finished.",
        ),
    ]
    return build_trajectory(
        instance_id=instance_id,
        phase1_passed=True,
        phase2_passed=True,
        submitted_sql=phase2_sql,
        submission_status="passed_phase2",
        phase1_observation="Phase 1 SQL Correct! (Reward: 1 points). Moving to Phase 2.",
        phase2_observation="Phase 2 SQL Correct! (Reward: 1 points). Task finished.",
        trajectory_steps=steps,
    )


def trajectory_phase1_retry_then_phase2(
    *,
    phase1_wrong_sql: str = "SELECT 999",
    phase1_right_sql: str = "SELECT 1",
    phase2_sql: str = "SELECT 2",
    instance_id: str = "alien_3",
) -> dict[str, Any]:
    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_a",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": phase1_wrong_sql},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_a",
            content="Submitted SQL failed test case in Phase 1. Reason: row mismatch. Please try again.",
        ),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_b",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": phase1_right_sql},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_b",
            content="Phase 1 SQL Correct! (Reward: 1 points). Moving to Phase 2.",
        ),
        user_text_msg(text="Follow-up."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_c",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": phase2_sql},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_c",
            content="Phase 2 SQL Correct! (Reward: 1 points). Task finished.",
        ),
    ]
    return build_trajectory(
        instance_id=instance_id,
        phase1_passed=True,
        phase2_passed=True,
        submitted_sql=phase2_sql,
        submission_status="passed_phase2",
        trajectory_steps=steps,
    )


def trajectory_no_submits(*, instance_id: str = "alien_4") -> dict[str, Any]:
    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_x",
                name="mcp__bird-interact-tools__get_schema",
                inp={},
            ),
        ),
        tool_result_msg(tool_use_id="tu_x", content="schema text"),
    ]
    return build_trajectory(
        instance_id=instance_id,
        phase1_passed=False,
        phase2_passed=False,
        submitted_sql="",
        submission_status="error",
        phase1_observation=None,
        phase2_observation=None,
        trajectory_steps=steps,
        error="budget_exhausted",
    )


# ---------------------------------------------------------------------------
# Stage a runs/ + results/ layout on disk
# ---------------------------------------------------------------------------


def stage_run(
    tmp_root: Path,
    *,
    benchmark: str,
    run_id: str,
    framework: str = "claude_sdk",
    mode: str = "a-interact",
    query_mode: str = "slayer",
    agent_model: str = "anthropic/claude-opus-4-7",
    user_sim_model: str = "anthropic/claude-sonnet-4-6",
    instances: list[tuple[str, str, dict[str, Any]]] | None = None,
    started_at: str = "2026-06-10T10:00:00+00:00",
) -> tuple[Path, Path]:
    """Lay out a run on disk that mirrors what the cloud merge produces.

    Each ``instances`` entry is ``(database, instance_id, trajectory_obj)``.
    Returns ``(runs_root, results_root)`` so the caller can monkeypatch
    ``paths.runs_root`` / ``paths.results_root`` to point at them.
    """
    instances = instances or []
    runs_root = tmp_root / "runs"
    results_root = tmp_root / "results"
    runs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    for db, inst_id, traj in instances:
        inst_dir = runs_root / benchmark / db / inst_id
        inst_dir.mkdir(parents=True, exist_ok=True)
        (inst_dir / f"{run_id}.trajectory.json").write_text(json.dumps(traj))
        (inst_dir / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "submission_annotation",
                    "instance_id": inst_id,
                    "selected_database": db,
                    "submission": {
                        "cloud_run_id": run_id,
                        "duration_s": traj.get("duration_s", 12.3),
                    },
                }
            )
        )

    db_dir = results_root / benchmark / "cloud" / run_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "results.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE task_results (
            run_id TEXT, framework TEXT, mode TEXT, query_mode TEXT,
            instance_id TEXT, database TEXT, started_at TEXT,
            duration_s REAL, phase1_passed INTEGER, phase2_passed INTEGER,
            total_reward REAL, submitted_sql TEXT, submitted_query TEXT,
            ground_truth_sql TEXT, error TEXT, usage_json TEXT, user_query TEXT,
            submission_status TEXT,
            phase1_observation TEXT, phase2_observation TEXT,
            predicted_result_json TEXT, gold_result_json TEXT,
            n_agent_turns INTEGER, tool_call_stats_json TEXT,
            phase1_observation_audited TEXT, phase1_observation_original TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE run_metadata (
            run_id TEXT, framework TEXT, mode TEXT,
            agent_model TEXT, user_sim_model TEXT, started_at TEXT
        )
        """
    )
    con.execute(
        "INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, framework, mode, agent_model, user_sim_model, started_at),
    )
    for db, inst_id, traj in instances:
        con.execute(
            """INSERT INTO task_results
               (run_id, framework, mode, query_mode,
                instance_id, database, started_at, duration_s,
                phase1_passed, phase2_passed, total_reward,
                submitted_sql, submitted_query, ground_truth_sql,
                error, usage_json, submission_status,
                phase1_observation, phase2_observation,
                predicted_result_json, gold_result_json,
                n_agent_turns)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                framework,
                mode,
                query_mode,
                inst_id,
                db,
                started_at,
                traj.get("duration_s", 12.3),
                int(traj["phase1_passed"]),
                int(traj["phase2_passed"]),
                traj.get("total_reward", 0.0),
                traj["submitted_sql"],
                traj.get("submitted_query", traj["submitted_sql"]),
                traj.get("ground_truth_sql", ""),
                traj.get("error"),
                json.dumps(traj.get("usage", {})),
                traj.get("submission_status", "error"),
                traj.get("phase1_observation"),
                traj.get("phase2_observation"),
                traj.get("predicted_result_json"),
                traj.get("gold_result_json"),
                traj.get("n_agent_turns", 0),
            ),
        )
    con.commit()
    con.close()
    return runs_root, results_root
