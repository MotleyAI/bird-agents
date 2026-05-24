"""Unit tests for the submission-status classifier and snapshot helper.

The harness writes one of these statuses per task; offline failure-mode
analysis uses them as the primary axis. Each class boundary needs an
explicit test or it'll silently mis-bucket a real-world failure.
"""

from __future__ import annotations

import sqlite3

from bird_interact_agents.agents._submit import (
    capture_result_snapshot,
    classify_submission,
)


def test_classify_passed_phase1():
    assert classify_submission(p1=True, p2=False, observation="ok") == "passed_phase1"


def test_classify_passed_phase2():
    assert classify_submission(p1=True, p2=True, observation="ok") == "passed_phase2"


def test_classify_json_error_short_circuits():
    """json_error must win over later flags — same call, multiple sources."""
    assert classify_submission(
        p1=False, p2=False, observation=None, json_failed=True,
    ) == "json_error"


def test_classify_translation_error():
    assert classify_submission(
        p1=False, p2=False, observation=None, translation_failed=True,
    ) == "translation_error"


def test_classify_infrastructure_error():
    assert classify_submission(
        p1=False, p2=False, observation="boom", infrastructure_failed=True,
    ) == "infrastructure_error"


def test_classify_sql_runtime_error_from_observation():
    """The canonical evaluator returns 'Error executing submitted SQL: ...'
    when the submitted SQL itself crashes — distinguishable from a clean
    'rows differ' wrong-result outcome."""
    obs = "Error executing submitted SQL: no such column: foo"
    assert classify_submission(p1=False, p2=False, observation=obs) == "sql_runtime_error"


def test_classify_sql_timeout_observation():
    obs = "Submitted SQL execution timed out"
    assert classify_submission(p1=False, p2=False, observation=obs) == "sql_runtime_error"


def test_classify_wrong_result_default():
    obs = "Submitted SQL failed test case in Phase 1. Reason: Default test case failed: rows differ"
    assert classify_submission(p1=False, p2=False, observation=obs) == "wrong_result"


def test_classify_wrong_result_when_no_observation():
    """Belt-and-braces: even with no observation, a non-passing,
    non-error path falls through to wrong_result."""
    assert classify_submission(p1=False, p2=False, observation=None) == "wrong_result"


# ---------------------------------------------------------------------------
# capture_result_snapshot
# ---------------------------------------------------------------------------


def _build_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    conn.executemany("INSERT INTO t (a, b) VALUES (?, ?)", [
        (1, "x"), (2, "y"), (3, None),
    ])
    conn.commit()
    conn.close()


def test_snapshot_returns_columns_types_count_and_sample(tmp_path):
    db_dir = tmp_path / "fixture"
    db_dir.mkdir()
    _build_db(db_dir / "fixture.sqlite")

    snap = capture_result_snapshot("SELECT a, b FROM t ORDER BY a", "fixture", str(tmp_path))
    assert snap is not None
    assert "error" not in snap
    assert [c["name"] for c in snap["columns"]] == ["a", "b"]
    # First non-null value drives type inference.
    assert snap["columns"][0]["type"] == "int"
    assert snap["columns"][1]["type"] == "str"
    assert snap["row_count"] == 3
    assert snap["sample_rows"][:2] == [[1, "x"], [2, "y"]]


def test_snapshot_returns_none_for_empty_or_missing_db(tmp_path):
    assert capture_result_snapshot("", "fixture", str(tmp_path)) is None
    assert capture_result_snapshot("SELECT 1", "no_such", str(tmp_path)) is None


def test_snapshot_returns_error_dict_on_runtime_failure(tmp_path):
    db_dir = tmp_path / "fixture"
    db_dir.mkdir()
    _build_db(db_dir / "fixture.sqlite")

    snap = capture_result_snapshot(
        "SELECT no_such_col FROM t", "fixture", str(tmp_path),
    )
    assert isinstance(snap, dict)
    assert "error" in snap
    assert "OperationalError" in snap["error"]
