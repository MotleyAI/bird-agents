"""Per-task SQLite sink for benchmark results.

`run_evaluation` opens one DB per output dir (`<output_dir>/results.db`)
and inserts a row for each completed task — both successes and
failures — *immediately* after that task returns. This survives mid-run
crashes that the old end-of-run `eval.json` dump did not: every
completed task's data lands on disk before the next one starts.

Schema is intentionally narrow and stable: the columns are the
analysis-relevant fields (pass/fail, costs, SQLs, errors). Per-task
JSON blobs (token usage, full trajectory) live in TEXT columns so we
don't have to migrate the schema every time we add a derived metric.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel


_TASK_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS task_results (
    run_id          TEXT NOT NULL,
    framework       TEXT NOT NULL,
    mode            TEXT NOT NULL,
    query_mode      TEXT NOT NULL,
    instance_id     TEXT NOT NULL,
    database        TEXT NOT NULL,
    started_at      REAL NOT NULL,
    duration_s      REAL NOT NULL,
    phase1_passed   INTEGER NOT NULL,
    phase2_passed   INTEGER NOT NULL,
    total_reward    REAL NOT NULL,
    submitted_sql   TEXT,
    submitted_query TEXT,
    ground_truth_sql TEXT,
    error           TEXT,
    usage_json      TEXT NOT NULL DEFAULT '{}',
    user_query      TEXT,
    submission_status TEXT NOT NULL DEFAULT 'never_submitted',
    phase1_observation TEXT,
    phase2_observation TEXT,
    predicted_result_json TEXT,
    gold_result_json TEXT,
    n_agent_turns  INTEGER,
    tool_call_stats_json TEXT,
    PRIMARY KEY (run_id, framework, mode, query_mode, instance_id)
)
"""

# Columns introduced after the original DDL. `open_db` will ALTER an existing
# table to add any of these that are missing, so result DBs from prior runs
# remain readable and writable by current code.
_DIAGNOSTIC_COLUMNS: list[tuple[str, str]] = [
    ("user_query", "TEXT"),
    ("submission_status", "TEXT NOT NULL DEFAULT 'never_submitted'"),
    ("phase1_observation", "TEXT"),
    ("phase2_observation", "TEXT"),
    ("predicted_result_json", "TEXT"),
    ("gold_result_json", "TEXT"),
    ("n_agent_turns", "INTEGER"),
    # Per-task tool-call statistics extracted from the agent's message
    # history: per-tool call/error counts plus a bounded sample of
    # validation-error / missing-tool messages. Shape:
    #   {"per_tool": [{"tool": str, "n_calls": int, "n_errors": int}, ...],
    #    "total_calls": int, "total_errors": int,
    #    "error_samples": [{"tool": str, "error": str}, ...]}
    ("tool_call_stats_json", "TEXT"),
]

_RUN_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS run_metadata (
    run_id          TEXT NOT NULL,
    framework       TEXT NOT NULL,
    mode            TEXT NOT NULL,
    agent_model     TEXT NOT NULL,
    user_sim_model  TEXT NOT NULL,
    started_at      REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, framework, mode)
)
"""


class TaskResultRow(BaseModel):
    """One row in `task_results`. Field order matches the DDL."""

    run_id: str
    framework: str
    mode: str
    query_mode: str
    instance_id: str
    database: str
    started_at: float
    duration_s: float
    phase1_passed: bool
    phase2_passed: bool
    total_reward: float
    submitted_sql: str | None = None
    submitted_query: str | None = None
    ground_truth_sql: str | None = None
    error: str | None = None
    usage_json: str = "{}"
    # Diagnostic columns — populated by submit_* helpers; default to safe
    # values so call sites that pre-date the columns don't have to know
    # about them.
    user_query: str | None = None
    submission_status: str = "never_submitted"
    phase1_observation: str | None = None
    phase2_observation: str | None = None
    predicted_result_json: str | None = None
    gold_result_json: str | None = None
    n_agent_turns: int | None = None
    tool_call_stats_json: str | None = None


def open_db(path: Path | str) -> sqlite3.Connection:
    """Open (or create) the results DB at `path` and ensure the schema
    exists. Caller is responsible for closing the connection.

    Idempotently adds any diagnostic columns missing from a pre-existing
    table — older result DBs gain the new columns as NULL rather than
    being abandoned.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(_TASK_RESULTS_DDL)
    conn.execute(_RUN_METADATA_DDL)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(task_results)")}
    for name, sql_type in _DIAGNOSTIC_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE task_results ADD COLUMN {name} {sql_type}")
    conn.commit()
    return conn


def insert_task_result(conn: sqlite3.Connection, row: TaskResultRow) -> None:
    """Upsert a task result. Re-inserting the same primary key replaces
    the prior row, supporting reruns/retries within an output dir."""
    conn.execute(
        """
        INSERT OR REPLACE INTO task_results
        (run_id, framework, mode, query_mode, instance_id, database,
         started_at, duration_s, phase1_passed, phase2_passed,
         total_reward, submitted_sql, submitted_query, ground_truth_sql,
         error, usage_json,
         user_query, submission_status, phase1_observation,
         phase2_observation, predicted_result_json, gold_result_json,
         n_agent_turns, tool_call_stats_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            row.run_id, row.framework, row.mode, row.query_mode,
            row.instance_id, row.database, row.started_at, row.duration_s,
            int(row.phase1_passed), int(row.phase2_passed),
            row.total_reward, row.submitted_sql, row.submitted_query,
            row.ground_truth_sql, row.error, row.usage_json,
            row.user_query, row.submission_status, row.phase1_observation,
            row.phase2_observation, row.predicted_result_json,
            row.gold_result_json, row.n_agent_turns,
            row.tool_call_stats_json,
        ),
    )
    conn.commit()


def insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    agent_model: str,
    user_sim_model: str,
    framework: str,
    mode: str,
    started_at: float = 0.0,
) -> None:
    """Record the per-run header so downstream tools (compare_results,
    failure-mode analysis) can correlate task rows with the model that
    produced them."""
    conn.execute(
        """
        INSERT OR REPLACE INTO run_metadata
        (run_id, framework, mode, agent_model, user_sim_model, started_at)
        VALUES (?,?,?,?,?,?)
        """,
        (run_id, framework, mode, agent_model, user_sim_model, started_at),
    )
    conn.commit()
