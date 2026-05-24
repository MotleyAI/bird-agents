"""Driver opens sqlite DB with mode=ro & immutable=1; INSERTs are rejected."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bird_interact_agents.sar_audit import driver


def test_read_only_connection_rejects_writes(fake_db):
    """`driver.open_readonly_sqlite` opens a connection that refuses writes."""
    con = driver.open_readonly_sqlite(fake_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("INSERT INTO t VALUES (99)")
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE u (y INTEGER)")
    finally:
        con.close()


def test_uri_contains_mode_ro_and_immutable(fake_db, monkeypatch):
    """The driver constructs its URI with both `mode=ro` and `immutable=1`."""
    captured = {}

    original = sqlite3.connect

    def spy_connect(target, *args, **kwargs):
        captured["target"] = target
        captured["kwargs"] = kwargs
        return original(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    con = driver.open_readonly_sqlite(fake_db)
    con.close()

    assert isinstance(captured["target"], str)
    assert "mode=ro" in captured["target"]
    assert "immutable=1" in captured["target"]
    assert captured["kwargs"].get("uri") is True


def test_read_only_can_still_query(fake_db):
    con = driver.open_readonly_sqlite(fake_db)
    try:
        rows = list(con.execute("SELECT x FROM t ORDER BY x"))
    finally:
        con.close()
    assert rows == [(1,), (2,), (3,)]
