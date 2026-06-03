"""DbConnection abstraction for SQLite and Postgres backends (DEV-1523).

All SQL execution in the harness routes through this interface so callers
are decoupled from the storage backend. The backend is selected by the
``Benchmark.db_backend`` field; ``make_db_connection`` is the single
factory callers use.

Postgres connection parameters are read from env vars:

* ``BIRD_PG_HOST``              (default: ``localhost``)
* ``BIRD_PG_PORT``              (default: ``5432``)
* ``BIRD_PG_USER``              (default: ``bird_interact``)
* ``BIRD_PG_PASSWORD``          (default: ``bird_interact``)
* ``BIRD_PG_STATEMENT_TIMEOUT`` (default: ``30000`` ms; 0 = no limit)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


ExecutorResult = tuple[list[tuple], list[str]]


@runtime_checkable
class DbConnection(Protocol):
    """Minimal SQL execution interface — both backends satisfy this."""

    def execute(self, sql: str) -> ExecutorResult:
        """Execute ``sql`` and return ``(rows, column_names)``."""
        ...

    def execute_sequence(self, sqls: list[str]) -> ExecutorResult:
        """Execute all ``sqls`` in one transaction, returning the last result.

        Only meaningful for Postgres (SQLiteDbConnection may raise
        NotImplementedError). Callers that need this must ensure they are
        working with a postgres backend.
        """
        ...

    def close(self) -> None:
        """Release the underlying connection."""
        ...

    def __enter__(self) -> "DbConnection":
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


class SqliteDbConnection:
    """Wraps sqlite3. ``read_only=True`` opens with ``?mode=ro`` URI mode."""

    def __init__(self, db_path: Path | str, *, read_only: bool = True) -> None:
        db_path = Path(db_path).resolve()
        self._read_only = read_only
        if read_only:
            uri = f"file:{db_path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=30)
        else:
            self._conn = sqlite3.connect(str(db_path), timeout=30)
        self._conn.execute("PRAGMA busy_timeout = 30000")

    def execute(self, sql: str) -> ExecutorResult:
        cur = self._conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
        if not self._read_only:
            self._conn.commit()
        return rows, cols

    def execute_sequence(self, sqls: list[str]) -> ExecutorResult:
        raise NotImplementedError("execute_sequence is postgres-only")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteDbConnection":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Postgres implementation
# ---------------------------------------------------------------------------


def _open_psycopg2_connection(
    db_name: str,
    host: str,
    port: int,
    user: str,
    password: str,
    statement_timeout_ms: int = 30000,
) -> Any:
    """Open a psycopg2 connection. Extracted so tests can monkeypatch it."""
    import psycopg2  # noqa: PLC0415 — deferred to avoid hard dep without postgres extra

    options = f"-c statement_timeout={statement_timeout_ms}" if statement_timeout_ms else ""
    return psycopg2.connect(
        dbname=db_name,
        host=host,
        port=port,
        user=user,
        password=password,
        options=options,
    )


class PostgresDbConnection:
    """Wraps a psycopg2 connection.

    ``read_only=True`` wraps each ``execute`` call in ``BEGIN READ ONLY``/
    ``ROLLBACK`` so writes to permanent tables are rejected server-side —
    this is the dry-run / grading path.

    ``read_only=False`` skips the ``BEGIN``/``ROLLBACK`` wrapper, which lets
    session state (TEMP tables, SET vars) persist across multiple ``execute``
    calls on the same connection — used by ``tolerant_grader._multi_sql_execute``
    for multi-statement gold SQL.  It does **not** commit writes: psycopg2's
    default autocommit=False means any DML lands in an implicit open transaction
    that is rolled back when the connection is closed.
    """

    def __init__(self, conn: Any, *, read_only: bool = False) -> None:
        self._conn = conn
        self._read_only = read_only

    def execute(self, sql: str) -> ExecutorResult:
        import psycopg2  # noqa: PLC0415

        cur = self._conn.cursor()
        if self._read_only:
            # BEGIN READ ONLY rejects DML against permanent tables server-side.
            # It does NOT prevent a multi-statement string with an embedded
            # COMMIT from terminating the transaction early — psycopg2's simple
            # protocol executes the whole string in one round-trip, so an
            # embedded COMMIT commits whatever preceded it before the ROLLBACK
            # in the finally block fires.  The assumption is that benchmark SQL
            # never contains embedded transaction control; enforcement against
            # permanent writes comes solely from the READ ONLY mode.
            # Temp-table writes (CREATE TEMP TABLE, INSERT INTO temp …) are
            # still permitted by Postgres even in a READ ONLY transaction.
            cur.execute("BEGIN READ ONLY")
        try:
            cur.execute(sql)
            # Non-SELECT statements (CREATE TEMP TABLE, SET, INSERT without
            # RETURNING) leave cur.description as None; fetchall() on those
            # raises ProgrammingError: no results to fetch.
            rows = cur.fetchall() if cur.description is not None else []
            cols = [d[0] for d in (cur.description or [])]
        except psycopg2.Error:
            if not self._read_only:
                try:
                    self._conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            if self._read_only:
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
        return rows, cols

    def execute_sequence(self, sqls: list[str]) -> ExecutorResult:
        """Execute all sqls in a SINGLE transaction, returning the last result.

        Unlike ``execute``, which wraps each call in its own ``BEGIN/ROLLBACK``,
        this method issues one ``BEGIN READ ONLY`` before the first statement and
        one ``ROLLBACK`` after the last, so intermediate DDL (temp tables, CTEs
        materialised into session state) is visible to subsequent statements.
        """
        import psycopg2  # noqa: PLC0415

        cur = self._conn.cursor()
        if self._read_only:
            cur.execute("BEGIN READ ONLY")
        try:
            rows: list = []
            cols: list = []
            for sql in sqls:
                cur.execute(sql)
                rows = cur.fetchall() if cur.description is not None else []
                cols = [d[0] for d in (cur.description or [])]
        except psycopg2.Error:
            if not self._read_only:
                try:
                    self._conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            if self._read_only:
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
        return rows, cols

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PostgresDbConnection":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_db_connection(
    db_name: str,
    *,
    data_path_base: str | Path = "",
    benchmark: Any,  # Benchmark — avoid circular import; duck-typed on db_backend
    db_file_path: str | None = None,
    read_only: bool = True,
) -> DbConnection:
    """Return a ``SqliteDbConnection`` or ``PostgresDbConnection`` based on
    ``benchmark.db_backend``. The only place in the codebase that branches
    on the backend — all callers above use ``DbConnection`` uniformly.

    For SQLite:  honours ``db_file_path`` when set on the task (LiveSQLBench
                 per-task isolation), else resolves
                 ``<data_path_base>/<db_name>/<db_name>.sqlite``.
    For Postgres: reads connection params from env vars (defaults match
                  upstream BIRD-Interact).
    """
    if getattr(benchmark, "db_backend", "sqlite") == "postgres":
        host = os.environ.get("BIRD_PG_HOST", "localhost")
        port = int(os.environ.get("BIRD_PG_PORT", "5432"))
        user = os.environ.get("BIRD_PG_USER", "bird_interact")
        password = os.environ.get("BIRD_PG_PASSWORD", "bird_interact")
        stmt_timeout = int(os.environ.get("BIRD_PG_STATEMENT_TIMEOUT", "30000"))
        conn = _open_psycopg2_connection(db_name, host, port, user, password, stmt_timeout)
        return PostgresDbConnection(conn, read_only=read_only)

    # SQLite path
    if db_file_path:
        db_path = Path(db_file_path)
    else:
        db_path = Path(data_path_base) / db_name / f"{db_name}.sqlite"
    return SqliteDbConnection(db_path, read_only=read_only)
