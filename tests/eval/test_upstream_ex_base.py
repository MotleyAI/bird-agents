"""Tests for the upstream-ex_base shim that powers N1.

The shim wraps upstream's `test_case_default` pipeline (remove_comments
+ remove_distinct + remove_round + ex_base) so our cascade tier N1
matches the upstream harness's grading semantics:

- 2-dp Decimal/float rounding via `preprocess_results`
- date / datetime normalisation to "YYYY-MM-DD"
- set-dedup (not multiset) equality
- ROUND() / DISTINCT / comment cleanup applied to BOTH SQLs

One deliberate deviation from upstream: when BOTH preprocessed result
lists are empty, the shim returns True (matches our legacy "both empty
= pass" behavior), while upstream returns 0. This is intentional and
pinned by `test_compare_pred_vs_gold_ex_base_both_empty_returns_true`.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — synthetic SQLite DB used by the comparison tests
# ---------------------------------------------------------------------------


def _make_sqlite_db(path: Path, *, rows: list[tuple]) -> sqlite3.Connection:
    """Create a tiny SQLite DB with one table `t(val)` populated with `rows`."""
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE t (val)")
    cur.executemany("INSERT INTO t (val) VALUES (?)", [(v,) for v in rows])
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# is_mutation_sql
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", [
    "INSERT INTO t VALUES (1)",
    "insert into t values (1)",
    "  UPDATE t SET x=1 WHERE id=2",
    "DELETE FROM t WHERE id=1",
    "CREATE TABLE u (x INT)",
    "DROP TABLE t",
    "ALTER TABLE t ADD COLUMN y INT",
    "TRUNCATE TABLE t",
    "REPLACE INTO t (id, x) VALUES (1, 2)",
])
def test_is_mutation_sql_positive(sql: str):
    from bird_interact_agents.eval.upstream_ex_base import is_mutation_sql

    assert is_mutation_sql(sql) is True


@pytest.mark.parametrize("sql", [
    # Round 5 (Codex): CTE-prefixed mutations. SQLite + Postgres both
    # accept these — the mutation verb sits AFTER a `WITH ... AS (...)`
    # block, so the statement-start regex alone misses them and the
    # dispatcher would have routed straight to upstream's writeable
    # exec path, committing the mutation against the per-task DB.
    "WITH x AS (SELECT id FROM t WHERE v > 0) DELETE FROM t WHERE id IN (SELECT id FROM x)",
    "with cte as (select 1 as v) insert into t (v) select v from cte",
    "WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) UPDATE t SET val = (SELECT x FROM a) WHERE val = (SELECT y FROM b)",
    "WITH temp_data AS (SELECT * FROM src) CREATE TABLE dest AS SELECT * FROM temp_data",
])
def test_is_mutation_sql_detects_cte_prefixed_mutations(sql: str):
    from bird_interact_agents.eval.upstream_ex_base import is_mutation_sql

    assert is_mutation_sql(sql) is True


@pytest.mark.parametrize("sql", [
    # Negative: a SELECT with a `WITH` clause but a SELECT body. The
    # verb-target regex must NOT match this.
    "WITH x AS (SELECT 1 AS v) SELECT * FROM x",
    "WITH a AS (SELECT id, val FROM t) SELECT id, val FROM a WHERE val > 0",
])
def test_is_mutation_sql_negative_for_cte_select(sql: str):
    from bird_interact_agents.eval.upstream_ex_base import is_mutation_sql

    assert is_mutation_sql(sql) is False


@pytest.mark.parametrize("sql", [
    # Round 3 (Codex): commented mutations must still be detected. Upstream's
    # remove_comments runs before exec, so a commented mutation would
    # otherwise sneak past the dispatcher and commit through ex_base.
    "-- explanation\nINSERT INTO t VALUES (1)",
    "/* multi\nline */ UPDATE t SET x=1",
    "  -- leading\n   DELETE FROM t WHERE id=1",
    "/* a */ /* b */ CREATE TABLE u (x INT)",
])
def test_is_mutation_sql_strips_comments_before_match(sql: str):
    from bird_interact_agents.eval.upstream_ex_base import is_mutation_sql

    assert is_mutation_sql(sql) is True


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "SELECT * FROM t WHERE val > 0",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "  select count(*) from t  ",
    # "REPLACE" inside a string literal must not trip the regex
    "SELECT REPLACE(name, 'a', 'b') FROM t",
    # Substring of a column name must not trip
    "SELECT updated_at FROM t",
    "SELECT inserted_at FROM t",
    "SELECT created_at FROM t",
    # Word-boundary adversarial cases (Codex finding #8)
    "SELECT dropoff FROM t",
    "SELECT * FROM truncate_value",
    "SELECT replaceable FROM t",
    # Substring inside identifiers
    "SELECT * FROM inserts_log",  # 'inserts' shouldn't trip 'INSERT'
])
def test_is_mutation_sql_negative(sql: str):
    from bird_interact_agents.eval.upstream_ex_base import is_mutation_sql

    assert is_mutation_sql(sql) is False


# ---------------------------------------------------------------------------
# compare_pred_vs_gold_ex_base — semantic correctness
# ---------------------------------------------------------------------------


def test_compare_pred_vs_gold_ex_base_rounds_to_2dp(tmp_path: Path):
    """A 4th-decimal float divergence rounds away at 2 dp and the shim
    reports True. The legacy `_set_equal` would call this a mismatch."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    pred_db = tmp_path / "p.sqlite"
    gold_db = tmp_path / "g.sqlite"
    pred_conn = _make_sqlite_db(pred_db, rows=[1.234567])
    gold_conn = _make_sqlite_db(gold_db, rows=[1.23])
    pred_conn.close()

    # Single conn for `ex_base` execution; both SQLs run against `gold_db`
    # with their literal rowsets via UNION ALL — keep it simple by
    # binding values inline.
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val)")
    conn.execute("INSERT INTO t (val) VALUES (1.234567)")
    pred_sqls = ["SELECT val FROM t"]
    sol_sqls = ["SELECT 1.23 AS val"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    gold_conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_set_semantics(tmp_path: Path):
    """Predicted has duplicate rows, gold has one. Upstream set-dedup =>
    True. The legacy `_set_equal` multiset would say False."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [(1,), (1,), (1,), (2,)])
    pred_sqls = ["SELECT val FROM t"]
    sol_sqls = ["SELECT 1 AS val UNION ALL SELECT 2 AS val"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_normalizes_dates(tmp_path: Path):
    """A `date` column compared against the gold's `YYYY-MM-DD` string
    matches after `preprocess_results`'s strftime normalisation."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("CREATE TABLE t (d DATE)")
    conn.execute("INSERT INTO t (d) VALUES (?)", (_dt.date(2026, 6, 11),))
    pred_sqls = ["SELECT d FROM t"]
    sol_sqls = ["SELECT '2026-06-11' AS d"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_strips_distinct_and_round():
    """Codex finding #2: `ex_base` itself does NOT strip ROUND/DISTINCT —
    `test_case_default` does. The shim MUST apply the cleanup. A pred SQL
    that wraps the gold expression in `ROUND(..., 4)` must still match a
    gold computing the same value, because both ROUND calls get stripped."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [(1.23,), (1.23,), (4.56,)])
    # Pred uses DISTINCT + ROUND; gold doesn't. After upstream's cleanup
    # they're identical SELECTs.
    pred_sqls = ["SELECT DISTINCT ROUND(val, 4) FROM t"]
    sol_sqls = ["SELECT val FROM t"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_both_empty_returns_true():
    """Legacy deviation: both predicted and gold returning zero rows is
    a PASS in our shim (matches legacy `_set_equal([], [])`), even though
    upstream `ex_base` would return 0. Documented in the spec."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    # No inserts — both queries return empty.
    pred_sqls = ["SELECT val FROM t WHERE val > 1000"]
    sol_sqls = ["SELECT val FROM t WHERE val > 1000"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_one_side_empty_returns_false():
    """Asymmetric empty: pred returns rows, gold returns nothing (or
    vice-versa) — still a real mismatch. Upstream behaviour preserved."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.execute("INSERT INTO t (val) VALUES (1)")
    pred_sqls = ["SELECT val FROM t"]
    sol_sqls = ["SELECT val FROM t WHERE val > 1000"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is False


def test_compare_pred_vs_gold_ex_base_fails_on_real_mismatch():
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [(1,), (2,), (3,)])
    pred_sqls = ["SELECT val FROM t"]
    sol_sqls = ["SELECT 1 AS val UNION ALL SELECT 2 AS val"]  # missing 3

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is False


# ---------------------------------------------------------------------------
# Dispatch correctness
# ---------------------------------------------------------------------------


def test_compare_pred_vs_gold_ex_base_dispatches_to_mini_interact(monkeypatch):
    """Mini-interact benchmarks invoke the mini-interact upstream module's
    `ex_base`, NOT the livesqlbench one."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    mini_called: list[tuple] = []
    lsb_called: list[tuple] = []

    fake_mini = MagicMock()
    fake_mini.ex_base = lambda p, s, db, conn, conditions: (
        mini_called.append((p, s, db, conditions)) or 1
    )
    fake_mini.remove_comments = lambda x: x
    fake_mini.remove_distinct = lambda x: x
    fake_mini.remove_round = lambda x: x

    fake_lsb = MagicMock()
    fake_lsb.ex_base = lambda p, s, db, conn, conditions: (
        lsb_called.append((p, s, db, conditions)) or 1
    )
    fake_lsb.remove_comments = lambda x: x
    fake_lsb.remove_distinct = lambda x: x
    fake_lsb.remove_round = lambda x: x

    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: fake_mini)
    monkeypatch.setattr(mod, "_load_livesqlbench_module", lambda: fake_lsb)

    mod.compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
        db_name="x.sqlite", conn=MagicMock(),
        conditions=None,
    )
    assert len(mini_called) == 1
    assert len(lsb_called) == 0


def test_compare_pred_vs_gold_ex_base_dispatches_to_livesqlbench(monkeypatch):
    from bird_interact_agents.eval import upstream_ex_base as mod

    mini_called: list = []
    lsb_called: list = []
    fake_mini = MagicMock()
    fake_mini.ex_base = lambda *a, **kw: mini_called.append(a) or 1
    fake_mini.remove_comments = lambda x: x
    fake_mini.remove_distinct = lambda x: x
    fake_mini.remove_round = lambda x: x
    fake_lsb = MagicMock()
    fake_lsb.ex_base = lambda *a, **kw: lsb_called.append(a) or 1
    fake_lsb.remove_comments = lambda x: x
    fake_lsb.remove_distinct = lambda x: x
    fake_lsb.remove_round = lambda x: x

    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: fake_mini)
    monkeypatch.setattr(mod, "_load_livesqlbench_module", lambda: fake_lsb)

    mod.compare_pred_vs_gold_ex_base(
        benchmark="livesqlbench-base-lite",
        pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
        db_name="alien", conn=MagicMock(),
        conditions=None,
    )
    assert len(lsb_called) == 1
    assert len(mini_called) == 0


def test_compare_pred_vs_gold_ex_base_unknown_benchmark_raises(monkeypatch):
    """Benchmark not in the supported set => ExBaseUnavailableError so
    the caller (N1 dispatch) can fall back to legacy `_set_equal`."""
    from bird_interact_agents.eval.upstream_ex_base import (
        ExBaseUnavailableError,
        compare_pred_vs_gold_ex_base,
    )

    with pytest.raises(ExBaseUnavailableError):
        compare_pred_vs_gold_ex_base(
            benchmark="bird-interact-lite-exp",
            pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
            db_name="x", conn=MagicMock(),
            conditions=None,
        )


def test_compare_pred_vs_gold_ex_base_pg_rolls_back_on_return(monkeypatch):
    """LiveSQLBench Postgres path MUST rollback the conn after grading so
    pred-side mutations cannot leak into the next grade on the same conn.
    Verified by spying on `conn.rollback`."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    fake_lsb = MagicMock()
    fake_lsb.ex_base = lambda *a, **kw: 1
    fake_lsb.remove_comments = lambda x: x
    fake_lsb.remove_distinct = lambda x: x
    fake_lsb.remove_round = lambda x: x
    monkeypatch.setattr(mod, "_load_livesqlbench_module", lambda: fake_lsb)
    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: MagicMock())

    conn = MagicMock()
    mod.compare_pred_vs_gold_ex_base(
        benchmark="livesqlbench-base-lite",
        pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
        db_name="alien", conn=conn,
        conditions=None,
    )
    conn.rollback.assert_called_once()


def test_compare_pred_vs_gold_ex_base_closes_owned_sqlite_conn_on_success(
    tmp_path: Path, monkeypatch,
):
    """Codex round 4 #1: when the caller passes ``conn=None`` the shim
    opens a SQLite conn itself and MUST close it on the success path,
    otherwise every N1 comparison leaks a file descriptor."""
    import sqlite3
    from bird_interact_agents.eval import upstream_ex_base as mod

    # Build a real on-disk SQLite DB so the upstream PRAGMAs (which
    # need a real file) don't fail.
    db_path = tmp_path / "alien.sqlite"
    seed = sqlite3.connect(str(db_path))
    seed.execute("CREATE TABLE t (val INT)")
    seed.execute("INSERT INTO t (val) VALUES (1)")
    seed.commit()
    seed.close()

    # Spy on sqlite3.connect to count the owned-conn open/close.
    real_connect = sqlite3.connect
    opened: list = []

    class _SpyConn:
        def __init__(self, inner):
            self._inner = inner
            self.closed = False
            opened.append(self)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            self.closed = True
            self._inner.close()

    def spy_connect(*args, **kwargs):
        return _SpyConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    mod.compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=["SELECT val FROM t"], sol_sqls=["SELECT val FROM t"],
        db_name=str(db_path), conn=None,
        conditions=None,
    )
    assert opened, "shim did not open a SQLite conn even though conn=None"
    # Every owned conn we opened was closed.
    assert all(c.closed for c in opened), (
        "shim leaked a SQLite conn after the comparison"
    )


def test_compare_pred_vs_gold_ex_base_closes_owned_sqlite_conn_on_exception(
    tmp_path: Path, monkeypatch,
):
    """Even when upstream `ex_base` raises, the owned conn we opened
    must close so a flaky upstream call can't exhaust FDs."""
    import sqlite3
    from bird_interact_agents.eval import upstream_ex_base as mod

    db_path = tmp_path / "alien.sqlite"
    seed = sqlite3.connect(str(db_path))
    seed.execute("CREATE TABLE t (val INT)")
    seed.commit()
    seed.close()

    real_connect = sqlite3.connect
    opened: list = []

    class _SpyConn:
        def __init__(self, inner):
            self._inner = inner
            self.closed = False
            opened.append(self)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            self.closed = True
            self._inner.close()

    def spy_connect(*args, **kwargs):
        return _SpyConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    # Force upstream `ex_base` to raise.
    fake_mini = MagicMock()
    fake_mini.ex_base = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("simulated upstream blow-up")
    )
    fake_mini.remove_comments = lambda x: x
    fake_mini.remove_distinct = lambda x: x
    fake_mini.remove_round = lambda x: x
    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: fake_mini)

    with pytest.raises(RuntimeError):
        mod.compare_pred_vs_gold_ex_base(
            benchmark="mini-interact",
            pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
            db_name=str(db_path), conn=None,
            conditions=None,
        )
    assert opened
    assert all(c.closed for c in opened), (
        "shim leaked a SQLite conn after upstream raised"
    )


def test_compare_pred_vs_gold_ex_base_does_not_close_caller_supplied_conn(
    monkeypatch,
):
    """When the caller PROVIDES a conn (the cloud SQLite inline path),
    the shim must NOT close it — that's the caller's responsibility."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    fake_mini = MagicMock()
    fake_mini.ex_base = lambda *a, **kw: 1
    fake_mini.remove_comments = lambda x: x
    fake_mini.remove_distinct = lambda x: x
    fake_mini.remove_round = lambda x: x
    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: fake_mini)

    caller_conn = MagicMock()
    mod.compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
        db_name="x.sqlite", conn=caller_conn,
        conditions=None,
    )
    caller_conn.close.assert_not_called()


def test_compare_pred_vs_gold_ex_base_pg_rolls_back_on_exception(monkeypatch):
    """The rollback fires even when ex_base raises so a bad conn doesn't
    poison the next caller."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    fake_lsb = MagicMock()

    def _boom(*a, **kw):
        raise RuntimeError("ex_base went boom")

    fake_lsb.ex_base = _boom
    fake_lsb.remove_comments = lambda x: x
    fake_lsb.remove_distinct = lambda x: x
    fake_lsb.remove_round = lambda x: x
    monkeypatch.setattr(mod, "_load_livesqlbench_module", lambda: fake_lsb)
    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: MagicMock())

    conn = MagicMock()
    with pytest.raises(RuntimeError):
        mod.compare_pred_vs_gold_ex_base(
            benchmark="livesqlbench-base-lite",
            pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
            db_name="alien", conn=conn,
            conditions=None,
        )
    # Codex round-2 finding #7: assert "at least once", not "exactly once",
    # so a defensively-double-rolling impl doesn't fail this test.
    assert conn.rollback.call_count >= 1


# ---------------------------------------------------------------------------
# Codex round-2 additions
# ---------------------------------------------------------------------------


def test_compare_pred_vs_gold_ex_base_ordered_comparison_pass():
    """conditions={'order': True} compares result lists positionally (not as
    sets). Same rows in matching ORDER must pass."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [(3,), (1,), (2,)])
    pred_sqls = ["SELECT val FROM t ORDER BY val"]
    sol_sqls = ["SELECT 1 AS val UNION ALL SELECT 2 AS val UNION ALL SELECT 3 AS val"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions={"order": True},
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_ordered_comparison_fails_when_order_differs():
    """conditions={'order': True}: same rows in different order must FAIL.
    Without `conditions['order']=True`, the set-dedup path would have
    accepted them; the conditions arg must reach upstream `ex_base`."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [(1,), (2,), (3,)])
    pred_sqls = ["SELECT val FROM t"]  # natural order 1,2,3
    sol_sqls = ["SELECT 3 AS val UNION ALL SELECT 2 AS val UNION ALL SELECT 1 AS val"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions={"order": True},
    )
    conn.close()
    assert result is False


def test_compare_pred_vs_gold_ex_base_strips_distinct_only_on_pred_side():
    """Codex round-2 finding #2: prove the cleanup is applied to BOTH
    sides, not silently only one side. DISTINCT on pred alone must not
    cause a real-mismatch failure."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.executemany("INSERT INTO t (val) VALUES (?)", [(1,), (1,), (2,)])
    pred_sqls = ["SELECT DISTINCT val FROM t"]
    sol_sqls = ["SELECT val FROM t"]  # no DISTINCT

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_strips_round_only_on_gold_side():
    """Symmetric: ROUND() on gold alone must not cause a real-mismatch
    failure."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val REAL)")
    conn.execute("INSERT INTO t (val) VALUES (1.5)")
    pred_sqls = ["SELECT val FROM t"]
    sol_sqls = ["SELECT ROUND(val, 4) FROM t"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_strips_comments():
    """Codex round-2 finding #3: upstream's `remove_comments` is part of
    the cleanup pipeline. SQL containing comments that would change
    parsing behaviour must still execute and grade correctly."""
    from bird_interact_agents.eval.upstream_ex_base import (
        compare_pred_vs_gold_ex_base,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (val INT)")
    conn.execute("INSERT INTO t (val) VALUES (1)")
    pred_sqls = ["SELECT val /* trailing comment */ FROM t -- line comment"]
    sol_sqls = ["SELECT val FROM t"]

    result = compare_pred_vs_gold_ex_base(
        benchmark="mini-interact",
        pred_sqls=pred_sqls, sol_sqls=sol_sqls,
        db_name=":memory:", conn=conn,
        conditions=None,
    )
    conn.close()
    assert result is True


def test_compare_pred_vs_gold_ex_base_loader_import_failure_raises_unavailable(
    monkeypatch,
):
    """Codex round-2 finding #4: if the lazy loader raises `ImportError`
    (upstream tree missing), the public surface raises
    `ExBaseUnavailableError`, not the raw ImportError."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    def _raise_import(*a, **kw):
        raise ImportError("upstream tree not installed")

    monkeypatch.setattr(mod, "_load_mini_interact_module", _raise_import)

    with pytest.raises(mod.ExBaseUnavailableError):
        mod.compare_pred_vs_gold_ex_base(
            benchmark="mini-interact",
            pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
            db_name=":memory:", conn=MagicMock(),
            conditions=None,
        )


def test_compare_pred_vs_gold_ex_base_loader_filenotfound_raises_unavailable(
    monkeypatch,
):
    """Same shape, FileNotFoundError (upstream root env var points
    nowhere)."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    def _raise_fnf(*a, **kw):
        raise FileNotFoundError("test_utils.py not found at configured root")

    monkeypatch.setattr(mod, "_load_livesqlbench_module", _raise_fnf)

    with pytest.raises(mod.ExBaseUnavailableError):
        mod.compare_pred_vs_gold_ex_base(
            benchmark="livesqlbench-base-lite",
            pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
            db_name="alien", conn=MagicMock(),
            conditions=None,
        )


def test_load_module_from_file_isolates_db_utils_between_upstream_trees(
    tmp_path: Path,
):
    """Both upstream trees define their own ``db_utils.py``. Without
    cache isolation in ``_load_module_from_file``, Python reuses the
    first tree's ``sys.modules['db_utils']`` for the second tree's
    bare ``from db_utils import ...`` — leaking mini-interact's sqlite3
    helpers into livesqlbench's psycopg2 module (or vice versa).
    Regression test (CodeRabbit round 2): build two minimal trees with
    distinct ``db_utils`` marker constants, load each, and assert
    the bound helpers are tree-specific."""
    import sys
    from bird_interact_agents.eval import upstream_ex_base as mod

    tree_a = tmp_path / "tree_a"
    tree_a.mkdir()
    (tree_a / "db_utils.py").write_text("ORIGIN = 'tree_a'\n")
    (tree_a / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = 'a:' + ORIGIN\n"
    )

    tree_b = tmp_path / "tree_b"
    tree_b.mkdir()
    (tree_b / "db_utils.py").write_text("ORIGIN = 'tree_b'\n")
    (tree_b / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = 'b:' + ORIGIN\n"
    )

    # Pre-pollute the cache to simulate a prior load.
    prior = sys.modules.pop("db_utils", None)
    try:
        a = mod._load_module_from_file(
            "test_utils_a", tree_a / "test_utils.py",
            sys_path_addition=tree_a,
        )
        b = mod._load_module_from_file(
            "test_utils_b", tree_b / "test_utils.py",
            sys_path_addition=tree_b,
        )
        # Each module bound the correct sibling at exec time.
        assert a.WHICH == "a:tree_a"
        assert b.WHICH == "b:tree_b"
    finally:
        if prior is not None:
            sys.modules["db_utils"] = prior
        else:
            sys.modules.pop("db_utils", None)
        sys.modules.pop("test_utils_a", None)
        sys.modules.pop("test_utils_b", None)


def test_load_module_from_file_reloading_first_tree_still_finds_own_db_utils(
    tmp_path: Path,
):
    """Codex round 2: ``sys.path.insert(0, ...)`` was conditional on
    absence — so after loading tree_a then tree_b, ``sys.path`` looked
    like ``[tree_b_dir, tree_a_dir, ...]``. A subsequent reload of
    tree_a would re-execute its ``test_utils.py`` but the bare
    ``from db_utils import ...`` would walk ``sys.path`` in order and
    pick tree_b's sibling first. Regression: each load must re-front
    its own dir."""
    import sys
    from bird_interact_agents.eval import upstream_ex_base as mod

    tree_a = tmp_path / "tree_a"
    tree_a.mkdir()
    (tree_a / "db_utils.py").write_text("ORIGIN = 'tree_a'\n")
    (tree_a / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = ORIGIN\n"
    )

    tree_b = tmp_path / "tree_b"
    tree_b.mkdir()
    (tree_b / "db_utils.py").write_text("ORIGIN = 'tree_b'\n")
    (tree_b / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = ORIGIN\n"
    )

    prior_modules = {
        k: sys.modules.get(k) for k in ("db_utils",)
    }
    prior_path = list(sys.path)
    try:
        mod._load_module_from_file(
            "tu_a_first", tree_a / "test_utils.py", sys_path_addition=tree_a,
        )
        mod._load_module_from_file(
            "tu_b_after", tree_b / "test_utils.py", sys_path_addition=tree_b,
        )
        # Now reload tree_a; without the re-front, this picks tree_b's
        # db_utils because tree_b's dir is at position 0 of sys.path.
        a_again = mod._load_module_from_file(
            "tu_a_reload", tree_a / "test_utils.py",
            sys_path_addition=tree_a,
        )
        assert a_again.WHICH == "tree_a"
    finally:
        for k, v in prior_modules.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
        for k in ("tu_a_first", "tu_b_after", "tu_a_reload"):
            sys.modules.pop(k, None)
        sys.path[:] = prior_path


def test_load_module_from_file_restores_prior_db_utils_after_load(tmp_path: Path):
    """The cache-isolation snapshot restores whatever ``db_utils`` was
    in ``sys.modules`` BEFORE the load, so a third-party caller with
    its own ``db_utils`` import doesn't get clobbered."""
    import sys
    import types
    from bird_interact_agents.eval import upstream_ex_base as mod

    sentinel = types.ModuleType("db_utils")
    sentinel.ORIGIN = "caller_sentinel"  # type: ignore[attr-defined]
    sys.modules["db_utils"] = sentinel

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "db_utils.py").write_text("ORIGIN = 'tree_internal'\n")
    (tree / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = ORIGIN\n"
    )
    try:
        m = mod._load_module_from_file(
            "test_utils_iso", tree / "test_utils.py",
            sys_path_addition=tree,
        )
        assert m.WHICH == "tree_internal"
        # Post-load: the caller's sentinel is restored.
        assert sys.modules["db_utils"] is sentinel
    finally:
        sys.modules.pop("db_utils", None)
        sys.modules.pop("test_utils_iso", None)


@pytest.mark.parametrize("benchmark", [
    "livesqlbench-base-lite-sqlite",
    "livesqlbench-base-lite",
    "livesqlbench-base-full",
    "livesqlbench-large",
])
def test_compare_pred_vs_gold_ex_base_dispatches_for_all_livesqlbench_variants(
    monkeypatch, benchmark: str,
):
    """Codex round-2 finding #5 + #6: every LSB-shape benchmark in the
    supported set must dispatch — including the SQLite variant."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    seen: list[str] = []
    fake_mini = MagicMock()
    fake_mini.ex_base = lambda *a, **kw: (seen.append("mini") or 1)
    fake_mini.remove_comments = lambda x: x
    fake_mini.remove_distinct = lambda x: x
    fake_mini.remove_round = lambda x: x
    fake_lsb = MagicMock()
    fake_lsb.ex_base = lambda *a, **kw: (seen.append("lsb") or 1)
    fake_lsb.remove_comments = lambda x: x
    fake_lsb.remove_distinct = lambda x: x
    fake_lsb.remove_round = lambda x: x
    monkeypatch.setattr(mod, "_load_mini_interact_module", lambda: fake_mini)
    monkeypatch.setattr(mod, "_load_livesqlbench_module", lambda: fake_lsb)

    mod.compare_pred_vs_gold_ex_base(
        benchmark=benchmark,
        pred_sqls=["SELECT 1"], sol_sqls=["SELECT 1"],
        db_name="alien", conn=MagicMock(),
        conditions=None,
    )
    # The SQLite LSB variant uses upstream livesqlbench's grader (same
    # algorithm, sqlite3 driver) — dispatched to LSB module.
    assert seen == ["lsb"]


# ---------------------------------------------------------------------------
# DEV-1550: upstream-tree root resolution. The cloud actor MUST be able to
# load the upstream grader modules; the prior author-private absolute paths
# (`/home/james/...`) meant the loaders silently `FileNotFoundError`d on every
# other machine, and the N1 dispatch downgraded to legacy `_set_equal` without
# anyone noticing.
# ---------------------------------------------------------------------------


def test_cloud_grader_root_constants_are_under_in_image_bake_dir():
    """The in-image bake paths must point under ``/app/upstream_graders/``
    (matched 1:1 by ``Dockerfile.cloud``'s ``COPY --from=...`` lines) —
    otherwise the dispatch silently falls back to legacy ``_set_equal`` in
    the cloud actor."""
    from bird_interact_agents.eval.upstream_ex_base import (
        _CLOUD_BIRD_INTERACT_ROOT, _CLOUD_LIVESQLBENCH_ROOT,
    )

    assert str(_CLOUD_BIRD_INTERACT_ROOT).startswith("/app/upstream_graders/")
    assert str(_CLOUD_LIVESQLBENCH_ROOT).startswith("/app/upstream_graders/")
    # No author-private hardcoded paths anywhere in the defaults.
    assert not str(_CLOUD_BIRD_INTERACT_ROOT).startswith("/home/")
    assert not str(_CLOUD_LIVESQLBENCH_ROOT).startswith("/home/")


_MARKER_REL_BIRD_INTERACT = (
    "mini_interact/knowledge_based/mini_interact_conv/evaluation/test_utils.py"
)
_MARKER_REL_LIVESQLBENCH = "evaluation/src/test_utils.py"


def _populate_grader_markers(root: Path, marker_rel: str) -> None:
    """Drop a `test_utils.py` + `db_utils.py` pair under the eval dir
    derived from `marker_rel`, so the round-8 resolver accepts `root`."""
    from bird_interact_agents.eval.upstream_ex_base import (
        REQUIRED_UPSTREAM_GRADER_MARKERS,
    )

    eval_dir = (root / marker_rel).parent
    eval_dir.mkdir(parents=True, exist_ok=True)
    for marker in REQUIRED_UPSTREAM_GRADER_MARKERS:
        (eval_dir / marker).write_text("")


def test_resolve_upstream_root_prefers_env_var(monkeypatch, tmp_path: Path):
    """Env override wins over both the in-image bake path and the
    sibling-of-main-checkout discovery — local devs must be able to point
    the loader at an arbitrary checkout. Round 8: override is also
    validated, so it must contain the full marker set to be accepted."""
    from bird_interact_agents.eval.upstream_ex_base import (
        _CLOUD_BIRD_INTERACT_ROOT, _resolve_upstream_root,
    )

    override = tmp_path / "my-bird-interact-fork"
    _populate_grader_markers(override, _MARKER_REL_BIRD_INTERACT)
    monkeypatch.setenv("BIRD_BIRD_INTERACT_ROOT", str(override))
    resolved = _resolve_upstream_root(
        "BIRD_BIRD_INTERACT_ROOT", _CLOUD_BIRD_INTERACT_ROOT, "BIRD-Interact",
        marker_rel=_MARKER_REL_BIRD_INTERACT,
    )
    assert resolved == override


def test_resolve_upstream_root_falls_back_to_sibling_of_main_checkout(
    monkeypatch, tmp_path: Path,
):
    """With no env override and no in-image bake dir, the resolver returns
    the sibling-of-main-checkout path produced by
    ``paths.bird_interact_upstream_root`` (the common local-dev layout)."""
    from bird_interact_agents.eval.upstream_ex_base import _resolve_upstream_root

    monkeypatch.delenv("BIRD_BIRD_INTERACT_ROOT", raising=False)
    sentinel = tmp_path / "sibling-bird-interact"
    _populate_grader_markers(sentinel, _MARKER_REL_BIRD_INTERACT)

    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "bird_interact_upstream_root", lambda: sentinel)

    # Point the cloud path at a directory that does NOT exist, so the
    # resolver falls through to the sibling branch.
    nonexistent_cloud = tmp_path / "no-such-cloud-bake"
    resolved = _resolve_upstream_root(
        "BIRD_BIRD_INTERACT_ROOT", nonexistent_cloud, "BIRD-Interact",
        marker_rel=_MARKER_REL_BIRD_INTERACT,
    )
    assert resolved == sentinel


def test_resolve_upstream_root_prefers_cloud_bake_when_marker_present(
    monkeypatch, tmp_path: Path,
):
    """Inside the cloud actor (where ``/app/upstream_graders/...`` is baked
    by ``Dockerfile.cloud``), the resolver picks the in-image path over the
    sibling discovery — but only when the deeper ``test_utils.py`` marker
    is actually present (Codex round 7: a partial bake that leaves the
    cloud root dir alone but drops the inner file must fall through to
    the sibling, not silently downgrade)."""
    from bird_interact_agents.eval.upstream_ex_base import _resolve_upstream_root

    monkeypatch.delenv("BIRD_LIVESQLBENCH_ROOT", raising=False)
    cloud_bake = tmp_path / "app-upstream-graders-livesqlbench"
    _populate_grader_markers(cloud_bake, _MARKER_REL_LIVESQLBENCH)

    # If the resolver ignored the present cloud bake and consulted
    # `paths.livesqlbench_upstream_root` instead, this raise-on-call
    # would trip.
    import bird_interact_agents.paths as paths_mod

    def _explode():
        raise AssertionError("sibling discovery used while cloud bake present")

    monkeypatch.setattr(paths_mod, "livesqlbench_upstream_root", _explode)
    resolved = _resolve_upstream_root(
        "BIRD_LIVESQLBENCH_ROOT", cloud_bake, "livesqlbench",
        marker_rel=_MARKER_REL_LIVESQLBENCH,
    )
    assert resolved == cloud_bake


def test_resolve_upstream_root_falls_through_on_partial_cloud_bake(
    monkeypatch, tmp_path: Path,
):
    """Codex round 7: a partial bake that leaves the cloud root directory
    present BUT drops the deeper ``test_utils.py`` marker must NOT
    short-circuit the sibling discovery — otherwise the loader silently
    raises FileNotFoundError downstream and N1 falls back to legacy
    ``_set_equal`` without any operator-visible signal."""
    from bird_interact_agents.eval.upstream_ex_base import _resolve_upstream_root

    monkeypatch.delenv("BIRD_BIRD_INTERACT_ROOT", raising=False)

    cloud_root_present = tmp_path / "app-upstream-graders-bird-interact"
    cloud_root_present.mkdir()  # dir exists but marker is absent

    sibling = tmp_path / "sibling-bird-interact-full-bake"
    _populate_grader_markers(sibling, _MARKER_REL_BIRD_INTERACT)

    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(
        paths_mod, "bird_interact_upstream_root", lambda: sibling,
    )

    resolved = _resolve_upstream_root(
        "BIRD_BIRD_INTERACT_ROOT", cloud_root_present, "BIRD-Interact",
        marker_rel=_MARKER_REL_BIRD_INTERACT,
    )
    assert resolved == sibling, (
        "Partial cloud bake (dir present, marker missing) short-circuited "
        "the sibling fallback — that's the silent-degrade case the round-7 "
        "tightening exists to prevent."
    )


# ---------------------------------------------------------------------------
# DEV-1550 round 8: every candidate branch (env override, cloud bake,
# sibling discovery) must validate against the COMPLETE marker set.
# Previously the env-override branch returned unconditionally and the
# sibling branch was unvalidated — both routes silently downgraded N1
# when the named tree was incomplete.
# ---------------------------------------------------------------------------


def test_resolve_upstream_root_falls_through_on_partial_env_override(
    monkeypatch, tmp_path: Path,
):
    """Env override is honoured if it's COMPLETE; an incomplete override
    must fall through to the cloud / sibling rather than silently win.
    Otherwise a stale `$BIRD_BIRD_INTERACT_ROOT` pointing at a partial
    fork masks a fully-baked cloud tree and N1 silently degrades."""
    from bird_interact_agents.eval.upstream_ex_base import _resolve_upstream_root

    partial_override = tmp_path / "stale-bird-interact-fork"
    # Only test_utils.py — no db_utils.py. Round-7 build-time guard
    # rejected this; round-8 resolver must too.
    eval_dir = partial_override / Path(_MARKER_REL_BIRD_INTERACT).parent
    eval_dir.mkdir(parents=True)
    (eval_dir / "test_utils.py").write_text("")
    monkeypatch.setenv("BIRD_BIRD_INTERACT_ROOT", str(partial_override))

    sibling = tmp_path / "sibling-complete"
    _populate_grader_markers(sibling, _MARKER_REL_BIRD_INTERACT)
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(
        paths_mod, "bird_interact_upstream_root", lambda: sibling,
    )

    nonexistent_cloud = tmp_path / "no-such-cloud-bake"
    resolved = _resolve_upstream_root(
        "BIRD_BIRD_INTERACT_ROOT", nonexistent_cloud, "BIRD-Interact",
        marker_rel=_MARKER_REL_BIRD_INTERACT,
    )
    assert resolved == sibling, (
        "Incomplete env override masked the complete sibling tree — "
        "round-8 resolver must validate every branch's marker set."
    )


def test_resolve_upstream_root_falls_through_on_partial_sibling(
    monkeypatch, tmp_path: Path,
):
    """An incomplete sibling-of-checkout tree must not silently win
    either — code paths that don't go through ``build_and_push`` (local
    regrade, dev shell, etc.) would otherwise import the upstream
    successfully via test_utils.py but crash on db_utils.py and N1
    silently downgrades."""
    from bird_interact_agents.eval.upstream_ex_base import _resolve_upstream_root

    monkeypatch.delenv("BIRD_LIVESQLBENCH_ROOT", raising=False)
    nonexistent_cloud = tmp_path / "no-such-cloud-bake"

    partial_sibling = tmp_path / "sibling-partial"
    eval_dir = partial_sibling / Path(_MARKER_REL_LIVESQLBENCH).parent
    eval_dir.mkdir(parents=True)
    (eval_dir / "test_utils.py").write_text("")  # MISSING db_utils.py

    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(
        paths_mod, "livesqlbench_upstream_root", lambda: partial_sibling,
    )

    with pytest.raises(
        __import__(
            "bird_interact_agents.eval.upstream_ex_base", fromlist=["x"],
        ).ExBaseUnavailableError,
    ) as excinfo:
        _resolve_upstream_root(
            "BIRD_LIVESQLBENCH_ROOT", nonexistent_cloud, "livesqlbench",
            marker_rel=_MARKER_REL_LIVESQLBENCH,
        )
    msg = str(excinfo.value)
    assert "db_utils.py" in msg
    assert "livesqlbench" in msg
    assert "BIRD_LIVESQLBENCH_ROOT" in msg, (
        "Failure message must name the env-var remediation."
    )


def test_load_module_from_file_rolls_back_state_on_exec_module_failure(
    tmp_path: Path,
):
    """Codex round 12: an exception during ``exec_module`` MUST leave
    ``sys.path`` and ``sys.modules[name]`` exactly as they were before
    the load — otherwise a partially initialised upstream module
    lingers under that name, and the prepended sys.path entry leaks
    into unrelated callers."""
    import sys
    from bird_interact_agents.eval import upstream_ex_base as mod

    # Tree that fails mid-exec_module. `raise RuntimeError(...)` at top
    # level fires when `exec_module` runs the module body. db_utils is
    # present so the import chain reaches the failing line.
    tree = tmp_path / "tree_failing"
    tree.mkdir()
    (tree / "db_utils.py").write_text("ORIGIN = 'fail-tree'\n")
    (tree / "test_utils.py").write_text(
        "from db_utils import ORIGIN\n"
        "raise RuntimeError('synthetic exec failure')\n"
    )

    name = "test_utils_failure_rollback"
    # Snapshot the world before the (failing) load.
    sys_path_before = list(sys.path)
    name_before = sys.modules.get(name)

    with pytest.raises(RuntimeError, match="synthetic exec failure"):
        mod._load_module_from_file(
            name, tree / "test_utils.py", sys_path_addition=tree,
        )

    assert sys.path == sys_path_before, (
        "sys.path was not restored after exec_module failure — the "
        "prepended sys_path_addition leaked into the process."
    )
    assert sys.modules.get(name) == name_before, (
        "sys.modules[name] still points at the partially initialised "
        "upstream module after exec_module failure. An unrelated "
        "importer using this name would see the broken stub."
    )


def test_load_module_from_file_is_thread_safe_under_concurrent_loads(
    tmp_path: Path,
):
    """Codex round 11: ``_load_module_from_file`` mutates process-global
    ``sys.path`` + ``sys.modules`` and runs ``exec_module``; two
    threads loading DIFFERENT upstream trees can interleave the
    re-front / evict / exec steps, and one tree's grader binds the
    other tree's ``db_utils``. Without the module-load lock this test
    flaps (or wedges) — with the lock, each thread sees its own
    consistent sibling."""
    import sys
    import threading
    from bird_interact_agents.eval import upstream_ex_base as mod

    n_threads = 8
    iterations_per_thread = 6

    tree_a = tmp_path / "tree_a"
    tree_a.mkdir()
    (tree_a / "db_utils.py").write_text("ORIGIN = 'tree_a'\n")
    (tree_a / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = ORIGIN\n"
    )

    tree_b = tmp_path / "tree_b"
    tree_b.mkdir()
    (tree_b / "db_utils.py").write_text("ORIGIN = 'tree_b'\n")
    (tree_b / "test_utils.py").write_text(
        "from db_utils import ORIGIN\nWHICH = ORIGIN\n"
    )

    barrier = threading.Barrier(n_threads)
    mismatches: list[tuple[str, str]] = []
    mismatches_lock = threading.Lock()

    def worker(which_tree: str):
        tree = tree_a if which_tree == "a" else tree_b
        expected = "tree_a" if which_tree == "a" else "tree_b"
        unique_id = threading.get_ident()
        barrier.wait()
        for i in range(iterations_per_thread):
            name = f"test_utils_{which_tree}_{unique_id}_{i}"
            loaded = mod._load_module_from_file(
                name, tree / "test_utils.py", sys_path_addition=tree,
            )
            if loaded.WHICH != expected:
                with mismatches_lock:
                    mismatches.append((expected, loaded.WHICH))
            sys.modules.pop(name, None)

    threads = [
        threading.Thread(target=worker, args=(("a" if i % 2 == 0 else "b"),))
        for i in range(n_threads)
    ]
    prior = sys.modules.pop("db_utils", None)
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        if prior is not None:
            sys.modules["db_utils"] = prior
        else:
            sys.modules.pop("db_utils", None)

    assert not mismatches, (
        f"Cross-tree binding leaked under concurrent loads: {mismatches[:5]} "
        f"(showing first 5 of {len(mismatches)})."
    )


def test_resolve_upstream_root_raises_actionable_message_when_no_candidate_valid(
    monkeypatch, tmp_path: Path,
):
    """If env, cloud, and sibling all fail validation, the resolver
    raises ExBaseUnavailableError naming every candidate's failure mode
    so the operator can fix the right tree."""
    from bird_interact_agents.eval import upstream_ex_base as mod

    monkeypatch.delenv("BIRD_BIRD_INTERACT_ROOT", raising=False)
    nonexistent_cloud = tmp_path / "no-cloud"
    nonexistent_sibling = tmp_path / "no-sibling"

    monkeypatch.setattr(
        __import__("bird_interact_agents.paths", fromlist=["x"]),
        "bird_interact_upstream_root", lambda: nonexistent_sibling,
    )

    with pytest.raises(mod.ExBaseUnavailableError) as excinfo:
        mod._resolve_upstream_root(
            "BIRD_BIRD_INTERACT_ROOT", nonexistent_cloud, "BIRD-Interact",
            marker_rel=_MARKER_REL_BIRD_INTERACT,
        )
    msg = str(excinfo.value)
    assert "in-image bake" in msg
    assert "sibling-of-checkout" in msg
    assert "BIRD_BIRD_INTERACT_ROOT" in msg
    assert "BIRD-Interact" in msg
