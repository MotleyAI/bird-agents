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
    phase1_observation_audited TEXT,
    phase1_observation_original TEXT,
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
    # DEV-1515 round 9 (Codex r8): dual-eval observation strings
    # produced by every agent flavor's submit helpers +
    # ``agents/_submit.py``. ``run.py`` was passing these into
    # ``TaskResultRow`` but they had been silently dropped from the
    # model + DDL; Pydantic's default ``extra="ignore"`` ate them
    # without warning, so ``results.db`` lost the audited/original
    # observation diagnostic — re-add them as nullable TEXT columns.
    ("phase1_observation_audited", "TEXT"),
    ("phase1_observation_original", "TEXT"),
    # DEV-1649: provenance for the persisted edited-models feature. Both NULL
    # unless --save-edited-models / --apply-edited-models were in effect.
    ("edited_models_saved_path", "TEXT"),
    ("edited_models_applied_from", "TEXT"),
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

# DEV-1535: extended config snapshot per run. Same additive-ALTER pattern
# as `_DIAGNOSTIC_COLUMNS` so pre-existing result DBs from earlier code
# revisions gain the new columns without breaking. All nullable for
# back-compat — older `insert_run_metadata` callers (e.g. legacy tests
# that only pass the original 5 kwargs) leave them as NULL.
_RUN_METADATA_DIAGNOSTIC_COLUMNS: list[tuple[str, str]] = [
    ("query_mode", "TEXT"),
    ("slayer_setup", "TEXT"),
    # DEV-1586: which pre-encoded reference fed a pre-encoded run
    # (otf=encoding-agent output, custom=hand-curated; NULL on-the-fly).
    ("pre_encoded_source", "TEXT"),
    ("patience", "INTEGER"),
    ("max_depth", "INTEGER"),
    ("reasoning_effort", "TEXT"),
    ("dataset", "TEXT"),
    ("strict", "INTEGER"),
    ("use_audited_gold_sql", "INTEGER"),
    ("prompt_cache", "INTEGER"),
]


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
    # DEV-1515 round 9 (Codex r8): the agent flavors + agents/_submit.py
    # have always emitted these observation strings; they were
    # accidentally dropped from this model along with the dual-eval
    # bool flags that DEV-1515 retired. Restoring as nullable TEXT so
    # ``run.py::_persist`` actually writes them through to ``results.db``.
    phase1_observation_audited: str | None = None
    phase1_observation_original: str | None = None
    # DEV-1649: edited-models persistence provenance (both nullable).
    edited_models_saved_path: str | None = None
    edited_models_applied_from: str | None = None


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
    # DEV-1535: additive ALTER for run_metadata's extended config snapshot.
    existing_rm = {
        row[1] for row in conn.execute("PRAGMA table_info(run_metadata)")
    }
    for name, sql_type in _RUN_METADATA_DIAGNOSTIC_COLUMNS:
        if name not in existing_rm:
            conn.execute(f"ALTER TABLE run_metadata ADD COLUMN {name} {sql_type}")
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
         n_agent_turns, tool_call_stats_json,
         phase1_observation_audited, phase1_observation_original,
         edited_models_saved_path, edited_models_applied_from)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            row.phase1_observation_audited,
            row.phase1_observation_original,
            row.edited_models_saved_path,
            row.edited_models_applied_from,
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
    # DEV-1535: extended config snapshot. All nullable + keyword-only with
    # None defaults so back-compat callers (legacy tests, older invocations)
    # work unchanged. New callers populate the full block.
    query_mode: str | None = None,
    slayer_setup: str | None = None,
    pre_encoded_source: str | None = None,
    patience: int | None = None,
    max_depth: int | None = None,
    reasoning_effort: str | None = None,
    dataset: str | None = None,
    strict: bool | None = None,
    use_audited_gold_sql: bool | None = None,
    prompt_cache: bool | None = None,
) -> None:
    """Record the per-run header so downstream tools (compare_results,
    failure-mode analysis) can correlate task rows with the model that
    produced them.

    DEV-1535: the extended config snapshot replaces the previous practice
    of parsing the cloud `run-id` substring (`-slayer-` vs `-raw-`) to
    guess the mode. Joins on `(run_id, framework, mode)` against
    `task_results` now expose the full configuration block.
    """
    # SQLite has no native bool; coerce via int() so the column is
    # readable as `WHERE strict = 1`.
    def _b(x: bool | None) -> int | None:
        return None if x is None else int(bool(x))
    conn.execute(
        """
        INSERT OR REPLACE INTO run_metadata
        (run_id, framework, mode, agent_model, user_sim_model, started_at,
         query_mode, slayer_setup, pre_encoded_source, patience, max_depth,
         reasoning_effort, dataset, strict, use_audited_gold_sql, prompt_cache)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, framework, mode, agent_model, user_sim_model, started_at,
            query_mode, slayer_setup, pre_encoded_source, patience, max_depth,
            reasoning_effort, dataset, _b(strict), _b(use_audited_gold_sql),
            _b(prompt_cache),
        ),
    )
    conn.commit()
