"""DEV-1515 round-5 follow-up: ``normalize_sol_sql`` pins the shape of
``sol_sql`` / ``original_sol_sql`` before the grader sees it.

Pre-fix the local + regrade plumbing did ``list(value or [])`` which
silently turned a bare string ``"SELECT 1"`` into
``["S", "E", "L", "E", "C", "T", " ", "1"]``. The grader then ran each
character through sqlite as a one-character SQL statement, raised
``sqlite3.OperationalError`` per character, and the row's N1 dropped
to False under the broad-except catch — silently. This test pins the
five shapes the helper has to handle correctly so the regression
can't reappear.
"""
from __future__ import annotations

from bird_interact_agents.eval.grade_in_place import normalize_sol_sql


def test_normalize_sol_sql_none_returns_empty_list():
    assert normalize_sol_sql(None) == []


def test_normalize_sol_sql_empty_string_returns_empty_list():
    assert normalize_sol_sql("") == []


def test_normalize_sol_sql_empty_list_returns_empty_list():
    assert normalize_sol_sql([]) == []


def test_normalize_sol_sql_string_wraps_as_single_item_list():
    """The load-bearing case — without this, ``list("SELECT 1")`` would
    return ``["S", "E", "L", "E", "C", "T", " ", "1"]`` and the grader
    would execute each character separately."""
    assert normalize_sol_sql("SELECT 1") == ["SELECT 1"]


def test_normalize_sol_sql_list_passes_through():
    assert normalize_sol_sql(["SELECT a", "SELECT b"]) == [
        "SELECT a", "SELECT b",
    ]


def test_normalize_sol_sql_tuple_coerces_to_list():
    """Tuples are accepted by the grader caller surface; coerce them so
    the downstream contract (``list[str]``) is honored."""
    assert normalize_sol_sql(("SELECT a",)) == ["SELECT a"]


def test_normalize_sol_sql_does_not_split_into_chars_for_long_string():
    """Belt-and-braces: the multi-character SQL case must not regress to
    the per-character split semantics. Asserts the return is a 1-element
    list whose element is the original string verbatim."""
    sql = (
        "WITH cte AS (SELECT * FROM tbl) "
        "SELECT col1, col2 FROM cte WHERE x = 1"
    )
    result = normalize_sol_sql(sql)
    assert len(result) == 1
    assert result[0] == sql
