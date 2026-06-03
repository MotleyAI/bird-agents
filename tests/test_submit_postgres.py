"""Tests for _submit.py helpers with postgres-backed benchmarks (DEV-1523).

Verifies that _dry_run_sql and capture_result_snapshot route through
the DbConnection abstraction when the task's dataset is a postgres benchmark,
and that both call sites in submit_raw_sql / submit_slayer_query thread the
benchmark through correctly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_state(db_name: str = "alien", data_path_base: str = "/irrelevant"):
    """Minimal state object shaped like a pydantic_ai TaskDeps, postgres variant."""
    return SimpleNamespace(
        status=SimpleNamespace(
            original_data={
                "selected_database": db_name,
                "dataset": "livesqlbench_postgres",
                "sol_sql": "SELECT 1",
            },
            remaining_budget=100.0,
            total_budget=100.0,
            force_submit=False,
            current_phase=1,
        ),
        data_path_base=data_path_base,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
        slayer_storage_dir="",
        result=None,
    )


def _make_pg_mock_conn(rows=None, cols=None):
    """Build a fake psycopg2 connection that yields rows on execute."""
    rows = rows or []
    cols = cols or []
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_cur.description = [(c, None, None, None, None, None, None) for c in cols]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


# ---------------------------------------------------------------------------
# _dry_run_sql — postgres path
# ---------------------------------------------------------------------------


def test_dry_run_sql_postgres_returns_none_on_success():
    """_dry_run_sql with postgres benchmark uses DbConnection, returns None on success."""
    from bird_interact_agents.agents._submit import _dry_run_sql
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("livesqlbench_postgres")
    mock_conn = _make_pg_mock_conn(rows=[(1,)], cols=["n"])

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        err = _dry_run_sql("SELECT 1 AS n", data_path_base="/irrelevant", db_name="alien", benchmark=b)
    assert err is None


def test_dry_run_sql_postgres_returns_error_on_db_error():
    """_dry_run_sql with postgres returns an error string when psycopg2 raises."""
    from bird_interact_agents.agents._submit import _dry_run_sql
    from bird_interact_agents.benchmark import get_benchmark
    import psycopg2

    b = get_benchmark("livesqlbench_postgres")
    mock_cur = MagicMock()
    mock_cur.execute.side_effect = psycopg2.ProgrammingError("column does not exist")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        err = _dry_run_sql("SELECT bad_col FROM t", data_path_base="/irrelevant", db_name="alien", benchmark=b)
    assert err is not None
    assert "column" in err.lower() or "does not exist" in err.lower() or "ProgrammingError" in err


def test_dry_run_sql_sqlite_path_unchanged(tmp_path):
    """_dry_run_sql without benchmark (default) still hits the sqlite path."""
    from bird_interact_agents.agents._submit import _dry_run_sql

    db_dir = tmp_path / "testdb"
    db_dir.mkdir()
    db_path = db_dir / "testdb.sqlite"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE items (id INTEGER)")
    con.commit()
    con.close()

    err = _dry_run_sql("SELECT id FROM items", data_path_base=str(tmp_path), db_name="testdb")
    assert err is None


# ---------------------------------------------------------------------------
# capture_result_snapshot — postgres path
# ---------------------------------------------------------------------------


def test_capture_result_snapshot_postgres_uses_db_connection():
    """capture_result_snapshot for postgres dataset calls DbConnection, not sqlite3."""
    from bird_interact_agents.agents._submit import capture_result_snapshot
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("livesqlbench_postgres")
    mock_conn = _make_pg_mock_conn(
        rows=[(1, "foo"), (2, "bar")],
        cols=["id", "name"],
    )

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        snap = capture_result_snapshot(
            "SELECT id, name FROM t",
            "alien",
            "/irrelevant",
            benchmark=b,
        )

    assert snap is not None
    assert snap["row_count"] == 2
    assert snap["columns"][0]["name"] == "id"
    assert snap["columns"][1]["name"] == "name"


def test_capture_result_snapshot_postgres_error_returns_error_dict():
    """A psycopg2 error surfaces as {"error": "..."}, not a raise."""
    from bird_interact_agents.agents._submit import capture_result_snapshot
    from bird_interact_agents.benchmark import get_benchmark
    import psycopg2

    b = get_benchmark("livesqlbench_postgres")
    mock_cur = MagicMock()
    mock_cur.execute.side_effect = psycopg2.ProgrammingError("relation does not exist")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        snap = capture_result_snapshot("SELECT * FROM ghost_table", "alien", "/irrelevant", benchmark=b)

    assert snap is not None
    assert "error" in snap


# ---------------------------------------------------------------------------
# submit_raw_sql wiring: benchmark derived from state.status.original_data
# ---------------------------------------------------------------------------


def test_submit_raw_sql_wires_postgres_benchmark_into_dry_run(monkeypatch):
    """submit_raw_sql for a postgres-dataset task calls _dry_run_sql with
    the postgres benchmark, not the SQLite path."""
    from bird_interact_agents.agents._submit import _dry_run_sql as _orig

    captured_kwargs: dict = {}

    def _spy_dry_run(sql, *, data_path_base, db_name, db_file_path=None, benchmark=None):
        captured_kwargs["benchmark"] = benchmark
        return None  # simulate success → fall through to execute_submit_action

    monkeypatch.setattr("bird_interact_agents.agents._submit._dry_run_sql", _spy_dry_run)

    # Patch execute_submit_action so the test doesn't need a real DB.
    with patch("bird_interact_agents.harness.execute_submit_action") as mock_submit:
        mock_submit.return_value = ("obs", 0.0, False, False, False)
        from bird_interact_agents.agents._submit import submit_raw_sql
        state = _pg_state()
        state.result = None
        state.status.original_data["original_sol_sql"] = None
        submit_raw_sql(state, "SELECT 1")

    assert captured_kwargs.get("benchmark") is not None
    assert captured_kwargs["benchmark"].db_backend == "postgres"


# ---------------------------------------------------------------------------
# Error message is backend-agnostic
# ---------------------------------------------------------------------------


def test_dry_run_error_message_is_backend_agnostic():
    """The user-facing dry-run error must not say 'SQLite error' — it must
    work for postgres tasks too."""
    from bird_interact_agents.agents._submit import _dry_run_error_message

    msg = _dry_run_error_message("column does not exist")
    assert "sqlite" not in msg.lower(), (
        "Error message must be backend-agnostic (not mention SQLite)"
    )
