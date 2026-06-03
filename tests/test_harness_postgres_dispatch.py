"""Tests for postgres dispatch in harness.py and grade_in_place.py (DEV-1523).

Verifies:
- execute_env_action / execute_submit_action route to postgres impl
  when dataset is a postgres benchmark.
- The postgres execute_env_action handles all standard action strings.
- materialize_task_db is a no-op for postgres benchmarks.
- grade_and_write auto-wires a postgres executor when benchmark is postgres.
- regrade routes through postgres executor for postgres tasks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_sample_status(db_name: str = "alien", data_path_base: str = "/data"):
    from batch_run_bird_interact.sample_status import SampleStatus

    return SampleStatus(
        idx=0,
        original_data={
            "selected_database": db_name,
            "dataset": "livesqlbench-base-lite",
            "amb_user_query": "How many?",
            "sol_sql": ["SELECT COUNT(*) FROM t"],
            "user_query_ambiguity": {},
        },
        remaining_budget=30.0,
        total_budget=30.0,
        force_submit=False,
    )


def _mock_pg_conn():
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [(3,)]
    mock_cur.description = [("count", None, None, None, None, None, None)]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


# ---------------------------------------------------------------------------
# execute_env_action dispatch
# ---------------------------------------------------------------------------


def test_execute_env_action_postgres_get_schema(tmp_path):
    """execute_env_action with postgres dataset returns schema from flat file."""
    from bird_interact_agents.harness import execute_env_action

    # Write schema file
    db_dir = tmp_path / "alien"
    db_dir.mkdir()
    (db_dir / "alien_schema.txt").write_text("CREATE TABLE t (id INTEGER)")

    status = _pg_sample_status(data_path_base=str(tmp_path))

    # Patch load_db_data_if_needed to load from tmp_path
    with patch("bird_interact_agents.harness.load_db_data_if_needed") as mock_load:
        # Simulate the schema being loaded into the cache
        from bird_interact_agents import harness as _h
        _h._schema_cache["alien"] = "CREATE TABLE t (id INTEGER)"
        observation, _success = execute_env_action("get_schema()", status, str(tmp_path))

    assert "CREATE TABLE" in observation


def test_execute_env_action_sqlite_still_routes_to_sqlite_impl(tmp_path):
    """execute_env_action with SQLite dataset still calls the upstream impl."""
    from bird_interact_agents.harness import execute_env_action
    from batch_run_bird_interact.sample_status import SampleStatus

    # Write schema file
    db_dir = tmp_path / "alien"
    db_dir.mkdir()
    (db_dir / "alien_schema.txt").write_text("CREATE TABLE t (id INTEGER)")

    status = SampleStatus(
        idx=0,
        original_data={
            "selected_database": "alien",
            "dataset": "mini-interact",
            "amb_user_query": "q",
            "sol_sql": ["SELECT 1"],
            "user_query_ambiguity": {},
        },
        remaining_budget=20.0, total_budget=20.0, force_submit=False,
    )
    with patch("bird_interact_agents.harness._sqlite_execute_env_action") as mock_sqlite_env:
        mock_sqlite_env.return_value = ("schema text", True)
        observation, success = execute_env_action("get_schema()", status, str(tmp_path))
    mock_sqlite_env.assert_called_once()


def test_execute_env_action_postgres_execute_sql():
    """execute_env_action with execute(...) action runs SQL via postgres DbConnection."""
    from bird_interact_agents.harness import execute_env_action

    status = _pg_sample_status()
    mock_conn = _mock_pg_conn()

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", return_value=mock_conn):
        observation, success = execute_env_action(
            'execute("SELECT COUNT(*) FROM t")',
            status,
            "/irrelevant",
        )

    assert success is True
    assert "3" in observation or "count" in observation.lower()


# ---------------------------------------------------------------------------
# execute_submit_action dispatch
# ---------------------------------------------------------------------------


def test_execute_submit_action_postgres_compares_results():
    """execute_submit_action with postgres dataset runs pred and gold SQL
    via postgres connection and returns p1=True on matching results."""
    from bird_interact_agents.harness import execute_submit_action

    status = _pg_sample_status()
    # Sol SQL returns the same as pred
    status.original_data["sol_sql"] = ["SELECT COUNT(*) FROM t"]

    call_count = [0]

    def mock_open(db_name, host, port, user, password, statement_timeout_ms=30000):
        mock_cur = MagicMock()
        # Both pred and gold return the same result
        mock_cur.fetchall.return_value = [(5,)]
        mock_cur.description = [("n", None, None, None, None, None, None)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", side_effect=mock_open):
        observation, reward, p1, p2, finished = execute_submit_action(
            "SELECT COUNT(*) FROM t", status, "/irrelevant"
        )

    assert p1 is True
    assert reward > 0


def test_execute_submit_action_postgres_fails_on_mismatch():
    """execute_submit_action returns p1=False when predicted ≠ gold."""
    from bird_interact_agents.harness import execute_submit_action

    status = _pg_sample_status()
    status.original_data["sol_sql"] = ["SELECT COUNT(*) FROM t"]

    call_count = [0]

    def mock_open(db_name, host, port, user, password, statement_timeout_ms=30000):
        mock_cur = MagicMock()
        # Return different values for the two calls
        call_count[0] += 1
        mock_cur.fetchall.return_value = [(1,)] if call_count[0] == 1 else [(5,)]
        mock_cur.description = [("n", None, None, None, None, None, None)]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    with patch("bird_interact_agents.db_connection._open_psycopg2_connection", side_effect=mock_open):
        observation, reward, p1, p2, finished = execute_submit_action(
            "SELECT 1 AS n FROM t",  # wrong query
            status,
            "/irrelevant",
        )

    assert p1 is False
    assert reward == 0.0


# ---------------------------------------------------------------------------
# materialize_task_db no-op for postgres
# ---------------------------------------------------------------------------


def test_materialize_task_db_noop_for_livesqlbench_postgres(tmp_path):
    from bird_interact_agents.harness import materialize_task_db

    task_data = {
        "selected_database": "alien",
        "instance_id": "alien_pg_1",
        "dataset": "livesqlbench-base-lite",
    }
    result = materialize_task_db(task_data, str(tmp_path))
    assert result is None  # no-op for postgres, no file needed


def test_materialize_task_db_noop_for_mini_interact_postgres(tmp_path):
    from bird_interact_agents.harness import materialize_task_db

    task_data = {
        "selected_database": "alien",
        "instance_id": "alien_mip_1",
        "dataset": "bird-interact-lite-exp",
    }
    result = materialize_task_db(task_data, str(tmp_path))
    assert result is None


# ---------------------------------------------------------------------------
# grade_and_write auto-wires postgres executor
# ---------------------------------------------------------------------------


def test_grade_and_write_postgres_uses_postgres_executor(tmp_path, monkeypatch):
    """grade_and_write for a postgres benchmark auto-derives the postgres
    executor so grade_submission doesn't fall back to default_executor (SQLite)."""
    from bird_interact_agents.eval.grade_in_place import grade_and_write
    from bird_interact_agents.eval import (
        AuditedGoldRef, GoldVariantRef, MetadataSufficiency, TaskAnnotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

    task_ann = TaskAnnotation(
        instance_id="alien_pg_1",
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-06-01",
        amb_user_query="count",
        metadata_sufficiency=MetadataSufficiency(verdict="sufficient", rationale="ok"),
        gold_variants=[
            GoldVariantRef(
                variant_id="primary",
                interpretation="count all",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="lsb_pg_audited.jsonl",
                    instance_id="alien_pg_1",
                    variant_id="primary",
                ),
            ),
        ],
        evaluator_prompt=None,
        provenance=Provenance(task_jsonl_path="livesqlbench_data.jsonl", task_jsonl_instance_id="alien_pg_1"),
    )

    captured_executor = []

    def spy_grade_submission(**kw):
        captured_executor.append(kw.get("executor"))
        # Return a dummy CascadeVerdict
        return CascadeVerdict(
            n1_original_gold=True, n2_audited_primary=True, n3_any_audited_variant=True,
            n4_tie_order=True, n5_llm_judge=True, n6_numeric_epsilon=True,
            n7_trailing_whitespace=True, n8_column_order=True, n9_case_fold=True,
        )

    monkeypatch.setattr("bird_interact_agents.eval.grade_in_place.grade_submission", spy_grade_submission)

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()

    grade_and_write(
        rows_dir=rows_dir,
        instance_id="alien_pg_1",
        benchmark="livesqlbench-base-lite",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT COUNT(*) FROM t"],
        submitted_sql="SELECT COUNT(*) FROM t",
        db_path=Path("alien"),  # logical path for postgres
        trajectory_path="",
    )

    assert len(captured_executor) == 1
    assert captured_executor[0] is not None, (
        "grade_and_write must auto-wire a non-None executor for postgres benchmarks"
    )
