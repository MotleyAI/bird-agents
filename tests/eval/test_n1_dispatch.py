"""Tests for the cascade N1 dispatch in `tolerant_grader.grade_submission`.

When the benchmark is in the upstream-ex_base supported set
(mini-interact + livesqlbench-*), N1 is computed via
`compare_pred_vs_gold_ex_base`. For unsupported benchmarks (bird-interact-*)
or when the shim raises `ExBaseUnavailableError`, N1 falls back to the
legacy `_set_equal(pred_rows, orig_rows)` path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_db_with_value(path: Path, table: str, col: str, val) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(f"CREATE TABLE {table} ({col})")
    conn.execute(f"INSERT INTO {table} ({col}) VALUES (?)", (val,))
    conn.commit()
    conn.close()


def _make_task_annotation():
    """Minimal TaskAnnotation that grade_submission's signature requires."""
    from bird_interact_agents.eval.annotation_schema import (
        MetadataSufficiency, Provenance, TaskAnnotation,
    )
    return TaskAnnotation(
        instance_id="iid_x",
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-01-01",
        amb_user_query="How many rows?",
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="test",
        ),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="iid_x",
        ),
    )


def test_n1_dispatch_calls_ex_base_for_mini_interact(tmp_path: Path):
    """Mini-interact benchmark: N1 is computed via
    `compare_pred_vs_gold_ex_base`, not legacy `_set_equal`."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.eval import tolerant_grader as tg

    db_path = tmp_path / "x.sqlite"
    _make_db_with_value(db_path, "t", "v", 1)
    task = _make_task_annotation()

    with patch.object(
        tg, "compare_pred_vs_gold_ex_base", return_value=True,
    ) as ex_base_call:
        verdict = tg.grade_submission(
            task_annotation=task,
            audited_gold_rows=[],
            original_sol_sql=["SELECT v FROM t"],
            submitted_sql="SELECT v FROM t",
            db_path=db_path,
            benchmark=get_benchmark("mini-interact"),
        )
    assert ex_base_call.called
    assert verdict.n1_original_gold is True


def test_n1_dispatch_uses_legacy_for_bird_interact_lite_exp(tmp_path: Path):
    """`bird-interact-lite-exp` is NOT scoped this PR. N1 keeps the
    legacy `_set_equal` path; the ex_base shim is never called."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.eval import tolerant_grader as tg

    db_path = tmp_path / "x.sqlite"
    _make_db_with_value(db_path, "t", "v", 1)
    task = _make_task_annotation()

    with patch.object(tg, "compare_pred_vs_gold_ex_base") as ex_base_call:
        tg.grade_submission(
            task_annotation=task,
            audited_gold_rows=[],
            original_sol_sql=["SELECT v FROM t"],
            submitted_sql="SELECT v FROM t",
            db_path=db_path,
            benchmark=get_benchmark("bird-interact-lite-exp"),
        )
    ex_base_call.assert_not_called()


def test_n1_dispatch_uses_legacy_for_no_benchmark(tmp_path: Path):
    """`benchmark=None` (legacy callers, tests) keeps the legacy path."""
    from bird_interact_agents.eval import tolerant_grader as tg

    db_path = tmp_path / "x.sqlite"
    _make_db_with_value(db_path, "t", "v", 1)
    task = _make_task_annotation()

    with patch.object(tg, "compare_pred_vs_gold_ex_base") as ex_base_call:
        tg.grade_submission(
            task_annotation=task,
            audited_gold_rows=[],
            original_sol_sql=["SELECT v FROM t"],
            submitted_sql="SELECT v FROM t",
            db_path=db_path,
            benchmark=None,
        )
    ex_base_call.assert_not_called()


def test_n1_fallback_when_ex_base_unavailable_returns_legacy_verdict(
    tmp_path: Path,
):
    """When the shim raises `ExBaseUnavailableError`, N1 falls back to
    `_set_equal(pred_rows, orig_rows)` so a missing upstream tree never
    crashes grading. The fallback verdict is the legacy verdict, not
    auto-False."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.eval import tolerant_grader as tg
    from bird_interact_agents.eval.upstream_ex_base import ExBaseUnavailableError

    db_path = tmp_path / "x.sqlite"
    _make_db_with_value(db_path, "t", "v", 1)
    task = _make_task_annotation()

    with patch.object(
        tg, "compare_pred_vs_gold_ex_base",
        side_effect=ExBaseUnavailableError("upstream not installed"),
    ):
        verdict = tg.grade_submission(
            task_annotation=task,
            audited_gold_rows=[],
            original_sol_sql=["SELECT v FROM t"],
            submitted_sql="SELECT v FROM t",
            db_path=db_path,
            benchmark=get_benchmark("mini-interact"),
        )
    # Both pred and gold execute to [(1,)] — legacy _set_equal returns True.
    assert verdict.n1_original_gold is True
