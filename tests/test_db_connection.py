"""Tests for the DbConnection abstraction (DEV-1523).

SqliteDbConnection and PostgresDbConnection implement the same Protocol.
make_db_connection dispatches based on benchmark.db_backend.
Postgres tests use a mocked psycopg2 connection.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest

from bird_interact_agents.benchmark import get_benchmark, Benchmark


# ---------------------------------------------------------------------------
# Helpers: tiny real SQLite DB for SqliteDbConnection tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_sqlite(tmp_path: Path):
    db_dir = tmp_path / "mydb"
    db_dir.mkdir()
    db_path = db_dir / "mydb.sqlite"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("CREATE TABLE t (x INTEGER, y TEXT)")
        con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        con.commit()
    finally:
        con.close()
    return {"db_path": db_path, "db_name": "mydb", "data_path_base": tmp_path}


# ---------------------------------------------------------------------------
# SqliteDbConnection
# ---------------------------------------------------------------------------


def test_sqlite_connection_execute_returns_rows_and_cols(tiny_sqlite):
    from bird_interact_agents.db_connection import SqliteDbConnection

    conn = SqliteDbConnection(tiny_sqlite["db_path"], read_only=True)
    rows, cols = conn.execute("SELECT x, y FROM t ORDER BY x")
    assert cols == ["x", "y"]
    assert rows == [(1, "a"), (2, "b")]
    conn.close()


def test_sqlite_connection_context_manager(tiny_sqlite):
    from bird_interact_agents.db_connection import SqliteDbConnection

    with SqliteDbConnection(tiny_sqlite["db_path"], read_only=True) as conn:
        rows, cols = conn.execute("SELECT x FROM t WHERE x = 1")
    assert rows == [(1,)]
    assert cols == ["x"]


def test_sqlite_connection_read_only_rejects_write(tiny_sqlite):
    from bird_interact_agents.db_connection import SqliteDbConnection

    with SqliteDbConnection(tiny_sqlite["db_path"], read_only=True) as conn:
        with pytest.raises(Exception):
            conn.execute("INSERT INTO t VALUES (3, 'c')")


def test_sqlite_connection_rw_allows_write(tiny_sqlite):
    from bird_interact_agents.db_connection import SqliteDbConnection

    db_path = tiny_sqlite["db_path"]
    with SqliteDbConnection(db_path, read_only=False) as conn:
        conn.execute("INSERT INTO t VALUES (3, 'c')")
    # verify persisted
    con = sqlite3.connect(str(db_path))
    count = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    con.close()
    assert count == 3


# ---------------------------------------------------------------------------
# PostgresDbConnection (mocked psycopg2)
# ---------------------------------------------------------------------------


def _mock_psycopg2_connect(rows, col_names):
    """Build a fake psycopg2 connection that returns rows + col_names on cursor.execute."""
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_cur.description = [(c, None, None, None, None, None, None) for c in col_names]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cur


def test_postgres_connection_execute_returns_rows_and_cols():
    from bird_interact_agents.db_connection import PostgresDbConnection

    mock_conn, mock_cur = _mock_psycopg2_connect([(1, "a"), (2, "b")], ["x", "y"])
    pg = PostgresDbConnection(mock_conn)
    rows, cols = pg.execute("SELECT x, y FROM t ORDER BY x")
    assert rows == [(1, "a"), (2, "b")]
    assert cols == ["x", "y"]


def test_postgres_connection_context_manager():
    from bird_interact_agents.db_connection import PostgresDbConnection

    mock_conn, mock_cur = _mock_psycopg2_connect([(42,)], ["n"])
    with PostgresDbConnection(mock_conn) as pg:
        rows, cols = pg.execute("SELECT 42 AS n")
    assert rows == [(42,)]
    assert cols == ["n"]
    mock_conn.close.assert_called_once()


def test_postgres_connection_read_only_uses_transaction():
    """read_only=True must wrap execution in BEGIN + ROLLBACK so writes are never committed."""
    from bird_interact_agents.db_connection import PostgresDbConnection

    mock_conn, mock_cur = _mock_psycopg2_connect([], [])
    pg = PostgresDbConnection(mock_conn, read_only=True)
    pg.execute("SELECT 1")
    # Check that BEGIN and ROLLBACK were issued via the cursor
    executed = [str(c.args[0]).strip().upper() for c in mock_cur.execute.call_args_list]
    assert "BEGIN" in executed or any("BEGIN" in s for s in executed)
    assert "ROLLBACK" in executed or any("ROLLBACK" in s for s in executed)
    pg.close()


def test_postgres_connection_execute_raises_on_db_error():
    from bird_interact_agents.db_connection import PostgresDbConnection
    import psycopg2

    mock_cur = MagicMock()
    mock_cur.execute.side_effect = psycopg2.ProgrammingError("column does not exist")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    pg = PostgresDbConnection(mock_conn)
    with pytest.raises(Exception):
        pg.execute("SELECT nonexistent_col FROM t")


# ---------------------------------------------------------------------------
# make_db_connection factory
# ---------------------------------------------------------------------------


def test_factory_dispatches_sqlite_for_sqlite_backend(tiny_sqlite, monkeypatch):
    from bird_interact_agents.db_connection import make_db_connection, SqliteDbConnection
    from bird_interact_agents.benchmark import MINI_INTERACT

    conn = make_db_connection(
        "mydb",
        data_path_base=tiny_sqlite["data_path_base"],
        benchmark=MINI_INTERACT,
    )
    assert isinstance(conn, SqliteDbConnection)
    conn.close()


def test_factory_dispatches_postgres_for_postgres_backend(monkeypatch):
    from bird_interact_agents.db_connection import make_db_connection, PostgresDbConnection
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("livesqlbench_postgres")
    mock_conn, _ = _mock_psycopg2_connect([], [])

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        conn = make_db_connection("alien", data_path_base="/irrelevant", benchmark=b)
    assert isinstance(conn, PostgresDbConnection)
    conn.close()


def test_factory_reads_pg_env_vars(monkeypatch):
    """make_db_connection for postgres reads BIRD_PG_HOST/PORT/USER/PASSWORD."""
    from bird_interact_agents.db_connection import make_db_connection
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("mini_interact_postgres")
    monkeypatch.setenv("BIRD_PG_HOST", "pg.example.com")
    monkeypatch.setenv("BIRD_PG_PORT", "5433")
    monkeypatch.setenv("BIRD_PG_USER", "testuser")
    monkeypatch.setenv("BIRD_PG_PASSWORD", "secret")

    captured: dict = {}

    def fake_open(db_name, host, port, user, password):
        captured.update(dict(db_name=db_name, host=host, port=port, user=user, password=password))
        mock_conn, _ = _mock_psycopg2_connect([], [])
        return mock_conn

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", side_effect=fake_open):
        conn = make_db_connection("alien", data_path_base="/irrelevant", benchmark=b)

    assert captured["host"] == "pg.example.com"
    assert captured["port"] == 5433
    assert captured["user"] == "testuser"
    assert captured["password"] == "secret"
    assert captured["db_name"] == "alien"
    conn.close()
