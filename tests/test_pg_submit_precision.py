"""DEV-1606 Defect 3 (postgres) — in-task pg grader must round to 2dp.

``_pg_execute_submit_action`` compared raw psycopg2 rows via ``Counter``
(with ``_pg_hashable_row`` only to make dict/list cells hashable) — no 2dp
normalization, so it was stricter than the SQLite ``ex_base`` in-task
grader. The pure helper ``_compare_pg_rows_2dp`` applies the same upstream
``preprocess_results`` (2dp + dict/list canonicalization) before the
ordered/unordered comparison, and falls back to ``_pg_hashable_row`` (NOT
identity — identity would re-raise ``unhashable dict``) when the upstream
normalizer is unavailable.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from bird_interact_agents import harness
from bird_interact_agents.eval import upstream_ex_base as ueb


def _upstream_available() -> bool:
    try:
        ueb._load_mini_interact_module()
        return True
    except Exception:  # noqa: BLE001
        return False


requires_upstream = pytest.mark.skipif(
    not _upstream_available(),
    reason="upstream grader tree not resolvable in this env",
)

_MINI = SimpleNamespace(name="mini-interact")
_BOGUS = SimpleNamespace(name="nonexistent-benchmark")


@requires_upstream
def test_pg_2dp_full_precision_matches_rounded_unordered():
    assert harness._compare_pg_rows_2dp(
        [(94.15248,)], [(94.15,)], benchmark=_MINI, ordered=False,
    ) is True


@requires_upstream
def test_pg_2dp_distinct_values_do_not_match():
    assert harness._compare_pg_rows_2dp(
        [(94.15248,)], [(94.99,)], benchmark=_MINI, ordered=False,
    ) is False


@requires_upstream
def test_pg_2dp_ordered_respects_row_order():
    pred = [(1.0,), (2.0,)]
    gold = [(2.0,), (1.0,)]
    assert harness._compare_pg_rows_2dp(
        pred, gold, benchmark=_MINI, ordered=True,
    ) is False
    assert harness._compare_pg_rows_2dp(
        pred, gold, benchmark=_MINI, ordered=False,
    ) is True


@requires_upstream
def test_pg_2dp_handles_dict_cells():
    """JSONB cells (dict) are canonicalized by preprocess_results."""
    assert harness._compare_pg_rows_2dp(
        [({"a": 1, "b": 2},)], [({"b": 2, "a": 1},)],
        benchmark=_MINI, ordered=False,
    ) is True


def test_pg_2dp_fallback_handles_dict_cells_without_upstream():
    """When the upstream normalizer is unavailable the helper must still
    not raise ``unhashable type: dict`` — it falls back to
    ``_pg_hashable_row``."""
    assert harness._compare_pg_rows_2dp(
        [({"a": 1},)], [({"a": 1},)],
        benchmark=_BOGUS, ordered=False,
    ) is True


def test_pg_2dp_fallback_is_precision_strict_without_upstream():
    """The fallback path does not 2dp-normalize (identity precision), so
    full-precision vs rounded does NOT match — confirming the fallback is
    the hashable-safe legacy compare, not silent normalization."""
    assert harness._compare_pg_rows_2dp(
        [(94.15248,)], [(94.15,)],
        benchmark=_BOGUS, ordered=False,
    ) is False


# ---------------------------------------------------------------------------
# _pg_execute_submit_action end-to-end: strips ROUND from the gold SQL and
# routes the comparison through the 2dp helper. A fake DB connection returns
# canned rows so the SQL string (not the DB) is the assertion surface.
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, recorder, pred_rows, gold_rows):
        self._rec = recorder
        self._pred = pred_rows
        self._gold = gold_rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql):
        self._rec["pred_sql"].append(sql)
        return self._pred, []

    def execute_sequence(self, sqls):
        self._rec["gold_sqls"].extend(sqls)
        return self._gold, []


@requires_upstream
def test_pg_execute_submit_action_strips_round_and_uses_2dp(monkeypatch):
    from types import SimpleNamespace

    rec = {"pred_sql": [], "gold_sqls": []}

    def fake_make_db_connection(db_name, **kwargs):
        # Agent full-precision; gold 2dp → match only after 2dp normalize.
        return _FakeConn(rec, [(94.15248,)], [(94.15,)])

    monkeypatch.setattr(harness, "make_db_connection", fake_make_db_connection)

    status = SimpleNamespace(original_data={
        "selected_database": "alien",
        "dataset": "livesqlbench-base-lite",  # postgres backend
        "sol_sql": ["SELECT ROUND(x, 2) AS m FROM t"],
        "conditions": {},
    })
    obs, reward, p1, p2, finished = harness._pg_execute_submit_action(
        "SELECT x AS m FROM t", status, "/tmp/ignored",
    )
    # remove_round stripped ROUND from the executed gold SQL.
    assert all("ROUND" not in s.upper() for s in rec["gold_sqls"])
    # 2dp normalization made the full-precision agent value match.
    assert p1 is True
    assert reward == 1.0
