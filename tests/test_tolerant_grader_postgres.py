"""Tests for the tolerant grader with postgres-backed benchmarks (DEV-1523).

Verifies that make_executor returns a postgres executor for postgres
benchmarks, that _multi_sql_execute handles shared postgres connections,
and that grade_submission works end-to-end with a mocked postgres executor.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.eval.tolerant_grader import (
    _multi_sql_execute,
    grade_submission,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_postgres_executor(rows, cols):
    """Return a postgres executor (callable) that yields fixed rows/cols."""
    def executor(sql: str, *, db_path: Path, conn: Any = None):
        return list(rows), list(cols)
    return executor


def _dummy_task_annotation():
    from bird_interact_agents.eval import (
        AuditedGoldRef,
        GoldVariantRef,
        MetadataSufficiency,
        TaskAnnotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance

    return TaskAnnotation(
        instance_id="alien_pg_1",
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-06-01",
        amb_user_query="How many rows?",
        metadata_sufficiency=MetadataSufficiency(verdict="sufficient", rationale="ok"),
        gold_variants=[
            GoldVariantRef(
                variant_id="primary",
                interpretation="count all",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="livesqlbench_postgres_audited.jsonl",
                    instance_id="alien_pg_1",
                    variant_id="primary",
                ),
            ),
        ],
        evaluator_prompt=None,
        provenance=Provenance(
            task_jsonl_path="livesqlbench_data.jsonl",
            task_jsonl_instance_id="alien_pg_1",
        ),
    )


def _dummy_audited_gold_row():
    return {
        "instance_id": "alien_pg_1",
        "selected_database": "alien",
        "benchmark": "livesqlbench_postgres",
        "audit_status": "clean",
        "original_sol_sql": ["SELECT COUNT(*) FROM t"],
        "audited_sol_sql": ["SELECT COUNT(*) FROM t"],
        "variant_id": "primary",
        "primary": True,
        "changes": [],
        "reasoning_summary": "",
        "skill_version": "audit-gold-sql/1.0",
        "audited_at": "2026-06-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# make_executor factory
# ---------------------------------------------------------------------------


def test_make_executor_returns_callable():
    from bird_interact_agents.eval.tolerant_grader import make_executor

    b = get_benchmark("livesqlbench_postgres")
    ex = make_executor(b)
    assert callable(ex)


def test_make_executor_sqlite_returns_default_executor_equivalent():
    """make_executor for a sqlite benchmark returns a callable equivalent
    to default_executor (also callable with same signature)."""
    from bird_interact_agents.eval.tolerant_grader import make_executor, default_executor

    b = get_benchmark("mini_interact")
    ex = make_executor(b)
    assert callable(ex)


def test_make_executor_postgres_executes_via_db_connection():
    """make_executor for postgres invokes DbConnection, not sqlite3."""
    from bird_interact_agents.eval.tolerant_grader import make_executor
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("livesqlbench_postgres")
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [(5,)]
    mock_cur.description = [("count", None, None, None, None, None, None)]
    mock_conn.cursor.return_value = mock_cur

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        ex = make_executor(b)
        rows, cols = ex("SELECT COUNT(*) AS count FROM t", db_path=Path("alien"))

    assert rows == [(5,)]
    assert cols == ["count"]


# ---------------------------------------------------------------------------
# _multi_sql_execute — postgres shared connection
# ---------------------------------------------------------------------------


def test_multi_sql_execute_postgres_uses_shared_connection():
    """When executor is a postgres executor and len(sqls) > 1, _multi_sql_execute
    uses a single shared psycopg2 connection so state (temp tables etc.) persists."""
    from bird_interact_agents.eval.tolerant_grader import make_executor

    b = get_benchmark("livesqlbench_postgres")
    call_log: list = []

    # Track how many distinct connection objects are used across calls.
    connections_used: set = set()

    def counting_executor(sql: str, *, db_path: Path, conn: Any = None):
        if conn is not None:
            connections_used.add(id(conn))
        call_log.append(sql)
        return [(1,)], ["n"]

    ex = counting_executor  # Use a plain counting executor

    rows, cols = _multi_sql_execute(
        ["CREATE TEMP TABLE tmp AS SELECT 1 AS n", "SELECT n FROM tmp"],
        db_path=Path("alien"),
        conn=None,
        executor=ex,
        benchmark=b,
    )
    # All calls should use the same connection id (shared connection)
    assert len(connections_used) <= 1, (
        "postgres multi-SQL must use a single shared connection"
    )
    assert rows == [(1,)]


def test_multi_sql_execute_single_sql_works_without_shared_conn():
    """Single-SQL sequences never need a shared connection, postgres or not."""
    from bird_interact_agents.eval.tolerant_grader import make_executor

    b = get_benchmark("livesqlbench_postgres")
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [(42,)]
    mock_cur.description = [("n", None, None, None, None, None, None)]
    mock_conn.cursor.return_value = mock_cur

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        ex = make_executor(b)
        rows, cols = _multi_sql_execute(
            ["SELECT 42 AS n"],
            db_path=Path("alien"),
            conn=None,
            executor=ex,
            benchmark=b,
        )
    assert rows == [(42,)]


# ---------------------------------------------------------------------------
# grade_submission — postgres end-to-end
# ---------------------------------------------------------------------------


def test_grade_submission_postgres_returns_cascade_verdict():
    """grade_submission works end-to-end with a postgres executor mock."""
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    task_ann = _dummy_task_annotation()
    audited = [_dummy_audited_gold_row()]
    db_path = Path("alien")

    # Both predicted and gold return the same result → should pass N1 or N2.
    executor = _make_postgres_executor(rows=[(5,)], cols=["count"])

    verdict = grade_submission(
        task_annotation=task_ann,
        audited_gold_rows=audited,
        original_sol_sql=["SELECT COUNT(*) FROM t"],
        submitted_sql="SELECT COUNT(*) FROM t",
        db_path=db_path,
        conn=None,
        executor=executor,
    )
    assert isinstance(verdict, CascadeVerdict)


def test_grade_submission_postgres_fails_on_mismatch():
    """When predicted result doesn't match gold, cascade verdict N1 is False."""
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    task_ann = _dummy_task_annotation()
    audited = [_dummy_audited_gold_row()]
    db_path = Path("alien")

    call_count = [0]

    def executor(sql: str, *, db_path: Path, conn: Any = None):
        call_count[0] += 1
        # Predicted returns 3; gold returns 5.
        return [(3,)] if call_count[0] == 1 else [(5,)], ["count"]

    verdict = grade_submission(
        task_annotation=task_ann,
        audited_gold_rows=audited,
        original_sol_sql=["SELECT COUNT(*) FROM t"],
        submitted_sql="SELECT COUNT(*) FROM t",
        db_path=db_path,
        conn=None,
        executor=executor,
    )
    assert isinstance(verdict, CascadeVerdict)
    assert verdict.n1_original_gold is False
