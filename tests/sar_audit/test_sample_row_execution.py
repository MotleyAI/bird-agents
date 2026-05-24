"""Driver executes audited_sol_sql against sqlite and stores first row + status."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bird_interact_agents.sar_audit import driver


def test_sample_row_ok(fake_db):
    result = driver.execute_sample_row(fake_db, "SELECT x FROM t ORDER BY x LIMIT 1")
    assert result.status == "ok"
    assert result.row == [1]
    assert result.error is None


def test_sample_row_empty(fake_empty_db):
    result = driver.execute_sample_row(fake_empty_db, "SELECT x FROM t WHERE x > 100")
    assert result.status == "empty"
    assert result.row is None
    assert result.error is None


def test_sample_row_error_on_missing_table(fake_db):
    result = driver.execute_sample_row(fake_db, "SELECT * FROM nonexistent_table")
    assert result.status == "error"
    assert result.row is None
    assert result.error is not None
    assert "no such table" in result.error.lower()


def test_sample_row_error_on_syntax_error(fake_db):
    result = driver.execute_sample_row(fake_db, "SELEXT bad sql")
    assert result.status == "error"
    assert result.row is None
    assert result.error is not None


def test_sample_row_returns_list_not_tuple(fake_db):
    """Tuples don't JSON-serialise round-trip; the driver normalises to list."""
    result = driver.execute_sample_row(fake_db, "SELECT x, x*2 FROM t ORDER BY x LIMIT 1")
    assert result.status == "ok"
    assert isinstance(result.row, list)
    assert result.row == [1, 2]
