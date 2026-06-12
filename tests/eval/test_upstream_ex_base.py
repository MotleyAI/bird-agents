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
