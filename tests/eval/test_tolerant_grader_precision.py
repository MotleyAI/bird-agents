"""DEV-1606 Defect 3 — cascade precision must match the ``ex_base`` grader.

N1 / the in-task grader run upstream ``remove_round`` (strip ROUND from
SQL) + ``preprocess_results`` (round floats/Decimals to 2dp ROUND_HALF_UP,
dates→'YYYY-MM-DD', dict/list→sorted-key JSON). The cascade's N2/N3 and the
cell tiers compared RAW python rows with ``epsilon=1e-6`` — far stricter.
A full-precision agent value (94.15248…) vs a 2dp gold (94.15) PASSES
ex_base but FAILED every cascade tier.

Contract under test:

* ``upstream_ex_base.preprocess_rows_like_ex_base(benchmark, rows)`` —
  2dp normalization via the upstream module; identity on
  unsupported/unavailable.
* ``upstream_ex_base.clean_sqls_like_ex_base(benchmark, sqls)`` —
  ``remove_round`` via the upstream module; identity on
  unsupported/unavailable.
* ``grade_submission`` normalizes pred/orig/variant rows (and strips ROUND
  from the executed SQL) so N2/N3 accept what ex_base accepts.

Tests that need the upstream grader tree skip cleanly when it is absent.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bird_interact_agents.eval import tolerant_grader as tg
from bird_interact_agents.eval import upstream_ex_base as ueb


def _upstream_available() -> bool:
    try:
        ueb._load_mini_interact_module()
        return True
    except Exception:  # noqa: BLE001 — ExBaseUnavailableError or load error
        return False


requires_upstream = pytest.mark.skipif(
    not _upstream_available(),
    reason="upstream mini-interact grader tree not resolvable in this env",
)


# ---------------------------------------------------------------------------
# Adapter helpers — identity fallback (no upstream needed)
# ---------------------------------------------------------------------------


def test_preprocess_rows_identity_on_unsupported_benchmark():
    rows = [(1.11111, "x")]
    out = ueb.preprocess_rows_like_ex_base("nonexistent-benchmark", rows)
    assert list(out) == [(1.11111, "x")]


def test_clean_sqls_identity_on_unsupported_benchmark():
    sqls = ["SELECT ROUND(x, 2) FROM t"]
    out = ueb.clean_sqls_like_ex_base("nonexistent-benchmark", sqls)
    assert list(out) == ["SELECT ROUND(x, 2) FROM t"]


# ---------------------------------------------------------------------------
# Adapter helpers — real 2dp normalization (needs upstream)
# ---------------------------------------------------------------------------


@requires_upstream
def test_preprocess_rows_rounds_to_2dp():
    out = ueb.preprocess_rows_like_ex_base("mini-interact", [(94.15248,)])
    assert [tuple(r) for r in out] == [(94.15,)]


@requires_upstream
def test_clean_sqls_strips_round():
    out = ueb.clean_sqls_like_ex_base("mini-interact", ["SELECT ROUND(x, 1) FROM t"])
    assert "ROUND" not in out[0].upper()


# ---------------------------------------------------------------------------
# grade_submission — N2/N3 accept full-precision-vs-2dp (stub executor)
# ---------------------------------------------------------------------------


class _KeyedExec:
    def __init__(self, table):
        self._table = table

    def __call__(self, sql, *, db_path, conn):
        for key, payload in self._table.items():
            if key in sql:
                return payload
        return ([], [])


def _annotation():
    from bird_interact_agents.eval import MetadataSufficiency, TaskAnnotation
    from bird_interact_agents.eval.annotation_schema import Provenance

    return TaskAnnotation(
        instance_id="reverse_logistics_18", selected_database="reverse_logistics",
        annotated_by="test", annotated_at="2026-06-01",
        amb_user_query="x",
        original_gold_is_correct=True,
        gold_variants=[],
        metadata_sufficiency=MetadataSufficiency(verdict="sufficient", rationale="r"),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="reverse_logistics_18",
        ),
    )


@requires_upstream
def test_grade_full_precision_vs_2dp_passes_n2_n3():
    """Agent returns full precision; the primary audited variant is 2dp.
    After 2dp normalization both collapse to 94.15 → N2/N3 pass."""
    table = {
        "PRED": ([(94.15248,)], ["metric"]),
        "GOLD": ([(94.15,)], ["metric"]),
        "ORIG": ([(99.0,)], ["metric"]),  # distinct → N1 must fail
    }
    v = tg.grade_submission(
        task_annotation=_annotation(),
        audited_gold_rows=[{
            "variant_id": "primary", "primary": True,
            "audit_status": "edited", "audited_sol_sql": ["SELECT GOLD"],
        }],
        original_sol_sql=["SELECT ORIG"],
        submitted_sql="SELECT PRED",
        db_path=Path("/dev/null"),
        conn=None,
        executor=_KeyedExec(table),
        benchmark="mini-interact",
    )
    assert v.n1_original_gold is False
    assert v.n2_audited_primary is True
    assert v.n3_any_audited_variant is True
    # Persisted result sets are the NORMALIZED (2dp) rows the tiers
    # compared — the single-normalized-vars contract (Codex #6).
    assert v.predicted_rows == [[94.15]]
    assert v.gold_rows == [[94.15]]


# ---------------------------------------------------------------------------
# grade_submission — remove_round on a real temp SQLite DB
# ---------------------------------------------------------------------------


@requires_upstream
def test_grade_remove_round_real_sqlite(tmp_path):
    """gold = ``ROUND(x, 1)`` (94.2), pred = ``x`` (94.156). With
    remove_round the gold strips to ``x`` → 94.156 → 2dp 94.16; pred →
    94.16. The original gold (``x * 2``) differs so N1 fails, isolating
    the N2 fix."""
    db_path = tmp_path / "reverse_logistics.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (x REAL)")
    conn.execute("INSERT INTO t (x) VALUES (94.156)")
    conn.commit()
    conn.close()

    v = tg.grade_submission(
        task_annotation=_annotation(),
        audited_gold_rows=[{
            "variant_id": "primary", "primary": True,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT ROUND(x, 1) AS metric FROM t"],
        }],
        original_sol_sql=["SELECT x * 2 AS metric FROM t"],
        submitted_sql="SELECT x AS metric FROM t",
        db_path=db_path,
        conn=None,
        executor=None,  # default SQLite executor
        benchmark="mini-interact",
    )
    assert v.n1_original_gold is False
    assert v.n2_audited_primary is True
