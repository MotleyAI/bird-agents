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
    """read_only=True must wrap execution in BEGIN READ ONLY and connection-level ROLLBACK."""
    from bird_interact_agents.db_connection import PostgresDbConnection

    mock_conn, mock_cur = _mock_psycopg2_connect([], [])
    pg = PostgresDbConnection(mock_conn, read_only=True)
    pg.execute("SELECT 1")
    # BEGIN is issued via the cursor; ROLLBACK goes through connection.rollback()
    executed = [str(c.args[0]).strip().upper() for c in mock_cur.execute.call_args_list]
    assert any("BEGIN" in s for s in executed)
    mock_conn.rollback.assert_called_once()
    pg.close()


def test_postgres_connection_read_only_error_uses_conn_rollback():
    """read_only=True error path must use connection.rollback(), not cur.execute('ROLLBACK'),
    so the rollback succeeds even when the connection is in aborted-transaction state."""
    import psycopg2

    from bird_interact_agents.db_connection import PostgresDbConnection

    mock_cur = MagicMock()
    mock_cur.execute.side_effect = [
        None,  # BEGIN READ ONLY succeeds
        psycopg2.ProgrammingError("syntax error"),  # actual SQL fails
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    pg = PostgresDbConnection(mock_conn, read_only=True)
    with pytest.raises(psycopg2.ProgrammingError):
        pg.execute("BAD SQL")

    # Must use connection-level rollback, not cur.execute("ROLLBACK")
    mock_conn.rollback.assert_called_once()
    rollback_via_cursor = any(
        "ROLLBACK" in str(c.args[0]).upper()
        for c in mock_cur.execute.call_args_list
        if c.args
    )
    assert not rollback_via_cursor, "ROLLBACK must not be issued via cursor.execute"
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

    def fake_open(db_name, host, port, user, password, statement_timeout_ms=30000):
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


# ---------------------------------------------------------------------------
# PostgresDbConnection.execute_sequence
# ---------------------------------------------------------------------------


def _mock_conn_sequence(result_rows_per_call: list[tuple[list, list]]):
    """Build a fake psycopg2 connection whose cursor.execute cycles through
    result_rows_per_call for each non-transaction-control call."""
    call_idx = {"n": -1}  # mutable cell for the closure

    class _FakeCur:
        def __init__(self):
            self.description = None
            self._rows = []

        def execute(self, sql):
            s = sql.strip().upper()
            if s in ("BEGIN", "BEGIN READ ONLY", "ROLLBACK", "COMMIT"):
                self.description = None
                self._rows = []
                return
            call_idx["n"] += 1
            idx = min(call_idx["n"], len(result_rows_per_call) - 1)
            rows, cols = result_rows_per_call[idx]
            self._rows = rows
            self.description = (
                [(c, None, None, None, None, None, None) for c in cols]
                if cols else None
            )

        def fetchall(self):
            return self._rows

    mock_cur = _FakeCur()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


def test_postgres_execute_sequence_returns_last_result():
    """execute_sequence must return the result of the LAST statement."""
    from bird_interact_agents.db_connection import PostgresDbConnection

    mock_conn, _ = _mock_conn_sequence([
        ([], []),            # setup stmt — no rows
        ([(7,)], ["val"]),   # final SELECT
    ])
    pg = PostgresDbConnection(mock_conn, read_only=True)
    rows, cols = pg.execute_sequence(["CREATE TEMP TABLE t (v INT)", "SELECT 7 AS val"])
    assert rows == [(7,)]
    assert cols == ["val"]


def test_postgres_execute_sequence_issues_one_begin_conn_rollback():
    """execute_sequence must issue exactly ONE BEGIN READ ONLY via cursor and ONE
    connection.rollback() regardless of the number of statements."""
    from bird_interact_agents.db_connection import PostgresDbConnection

    issued: list[str] = []

    class _SpyCur:
        def __init__(self):
            self.description = None

        def execute(self, sql):
            issued.append(sql.strip().upper())
            self.description = None

        def fetchall(self):
            return []

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = _SpyCur()

    pg = PostgresDbConnection(mock_conn, read_only=True)
    pg.execute_sequence(["stmt1", "stmt2", "stmt3"])

    begin_count = sum(1 for s in issued if s in ("BEGIN", "BEGIN READ ONLY"))
    rollback_via_cursor = sum(1 for s in issued if s == "ROLLBACK")
    assert begin_count == 1, f"expected 1 BEGIN, got {begin_count}; issued={issued}"
    assert rollback_via_cursor == 0, (
        f"ROLLBACK must not go through cursor.execute; issued={issued}"
    )
    mock_conn.rollback.assert_called_once()


def test_make_db_connection_passes_statement_timeout_to_psycopg2(monkeypatch):
    """make_db_connection must pass BIRD_PG_STATEMENT_TIMEOUT to psycopg2.connect
    via the options keyword so slow agent queries cannot hang workers."""
    import os
    from unittest.mock import patch, MagicMock
    from types import SimpleNamespace
    from bird_interact_agents.db_connection import make_db_connection

    captured: dict = {}

    def _fake_open(db_name, host, port, user, password, statement_timeout_ms=30000):
        captured["statement_timeout_ms"] = statement_timeout_ms
        return MagicMock()

    benchmark = SimpleNamespace(db_backend="postgres")

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", _fake_open):
        monkeypatch.setenv("BIRD_PG_STATEMENT_TIMEOUT", "15000")
        make_db_connection("mydb", benchmark=benchmark, read_only=True)

    assert captured["statement_timeout_ms"] == 15000


def test_make_db_connection_default_statement_timeout(monkeypatch):
    """Default BIRD_PG_STATEMENT_TIMEOUT is 30000 ms."""
    from unittest.mock import patch, MagicMock
    from types import SimpleNamespace
    from bird_interact_agents.db_connection import make_db_connection

    captured: dict = {}

    def _fake_open(db_name, host, port, user, password, statement_timeout_ms=30000):
        captured["statement_timeout_ms"] = statement_timeout_ms
        return MagicMock()

    benchmark = SimpleNamespace(db_backend="postgres")

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", _fake_open):
        monkeypatch.delenv("BIRD_PG_STATEMENT_TIMEOUT", raising=False)
        make_db_connection("mydb", benchmark=benchmark, read_only=True)

    assert captured["statement_timeout_ms"] == 30000


def test_open_psycopg2_connection_sets_options():
    """_open_psycopg2_connection forwards statement_timeout_ms as -c options."""
    import sys
    from unittest.mock import MagicMock, patch
    from bird_interact_agents.db_connection import _open_psycopg2_connection

    fake_psycopg2 = MagicMock()
    fake_psycopg2.connect.return_value = MagicMock()

    with patch.dict(sys.modules, {"psycopg2": fake_psycopg2}):
        _open_psycopg2_connection("db", "host", 5432, "u", "p", 20000)

    assert fake_psycopg2.connect.called
    _, kwargs = fake_psycopg2.connect.call_args
    assert "statement_timeout=20000" in kwargs.get("options", "")
