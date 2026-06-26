"""DEV-1606 Defect 2 — cascade relaxation tiers must COMPOSE.

The pre-fix cascade evaluates N6 (numeric-epsilon), N7 (trailing
whitespace), N8 (column-order) and N9 (case-fold) INDEPENDENTLY, so an
answer that is correct only under a COMBINATION of tolerances (e.g.
column-reorder AND 2dp-epsilon) passes no single tier and is mislabeled
``agent_miss``.

Contract under test:

* ``compare_relaxed`` — multiset row equality under a composable cell
  predicate (epsilon + optional rstrip + optional case-fold) using a
  PROPER bipartite matching (not greedy first-match).
* ``compare_column_order_relaxed`` — column-reorder alignment THEN
  ``compare_relaxed``.
* Wiring in ``grade_submission``:
  - N8 = strict column-order OR (reorder ∘ epsilon)         [N8 ∘ N6]
  - N9 = case-fold OR (no-reorder ∘ epsilon ∘ strip ∘ case)
                     OR (reorder ∘ epsilon ∘ strip ∘ case)  [terminal]
* Monotonicity is preserved (``enforce_monotone_cascade`` unchanged).

Mechanical contracts only — no prompt-content / behavioural assertions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents.eval import tolerant_grader as tg


# ---------------------------------------------------------------------------
# compare_relaxed — bipartite multiset match under a composable cell predicate
# ---------------------------------------------------------------------------


def test_compare_relaxed_epsilon_exact():
    assert tg.compare_relaxed(
        [(1.0,), (2.0,)], [(2.0,), (1.0,)],
        epsilon=1e-6, strip=False, casefold=False,
    ) is True


def test_compare_relaxed_epsilon_within_tolerance():
    assert tg.compare_relaxed(
        [(1.0000005,)], [(1.0,)],
        epsilon=1e-6, strip=False, casefold=False,
    ) is True


def test_compare_relaxed_epsilon_outside_tolerance():
    assert tg.compare_relaxed(
        [(1.5,)], [(1.0,)],
        epsilon=1e-6, strip=False, casefold=False,
    ) is False


def test_compare_relaxed_casefold():
    assert tg.compare_relaxed(
        [("ABC",)], [("abc",)],
        epsilon=1e-6, strip=False, casefold=True,
    ) is True
    # Without casefold the same pair must NOT match.
    assert tg.compare_relaxed(
        [("ABC",)], [("abc",)],
        epsilon=1e-6, strip=False, casefold=False,
    ) is False


def test_compare_relaxed_casefold_is_unicode_aware():
    """casefold (not lower) lifts Unicode-equivalent case pairs like
    'ß' == 'SS' (CodeRabbit DEV-1606)."""
    assert tg.compare_relaxed(
        [("straße",)], [("STRASSE",)],
        epsilon=1e-6, strip=False, casefold=True,
    ) is True


def test_compare_relaxed_trailing_whitespace():
    assert tg.compare_relaxed(
        [("a   ",)], [("a",)],
        epsilon=1e-6, strip=True, casefold=False,
    ) is True


def test_compare_relaxed_rowcount_mismatch():
    assert tg.compare_relaxed(
        [(1.0,)], [(1.0,), (1.0,)],
        epsilon=1e-6, strip=False, casefold=False,
    ) is False


def test_compare_relaxed_uses_bipartite_not_greedy():
    """Codex #4: a greedy first-match consumer can false-fail when the
    epsilon intervals overlap. With epsilon=0.1, pred [(0.1,), (0.2,)]
    vs gold [(0.2,), (0.0,)] has a valid perfect matching
    (0.1→0.0, 0.2→0.2) but a greedy matcher that grabs 0.1→0.2 first
    strands 0.2. A correct bipartite matcher returns True."""
    assert tg.compare_relaxed(
        [(0.1,), (0.2,)], [(0.2,), (0.0,)],
        epsilon=0.1, strip=False, casefold=False,
    ) is True


def test_compare_numeric_epsilon_bipartite_fix():
    """The pre-existing greedy bug in ``compare_numeric_epsilon`` is fixed
    by routing through the same bipartite matcher."""
    assert tg.compare_numeric_epsilon(
        [(0.1,), (0.2,)], [(0.2,), (0.0,)], epsilon=0.1,
    ) is True


# ---------------------------------------------------------------------------
# compare_column_order_relaxed — reorder alignment THEN relaxed compare
# ---------------------------------------------------------------------------


def test_compare_column_order_relaxed_reorder_plus_epsilon():
    # pred columns are [cost, carbon]; gold columns are [carbon, cost].
    assert tg.compare_column_order_relaxed(
        [(2.0000005, 1.0000005)], [(1.0, 2.0)],
        pred_cols=["cost", "carbon"], gold_cols=["carbon", "cost"],
        epsilon=1e-6, strip=False, casefold=False,
    ) is True


def test_compare_column_order_relaxed_reorder_plus_case():
    assert tg.compare_column_order_relaxed(
        [(1.0, "abc")], [("ABC", 1.0)],
        pred_cols=["x", "label"], gold_cols=["label", "x"],
        epsilon=1e-6, strip=False, casefold=True,
    ) is True
    # casefold off → the reordered string still differs by case → no match.
    assert tg.compare_column_order_relaxed(
        [(1.0, "abc")], [("ABC", 1.0)],
        pred_cols=["x", "label"], gold_cols=["label", "x"],
        epsilon=1e-6, strip=False, casefold=False,
    ) is False


def test_compare_column_order_relaxed_rejects_duplicate_names():
    # Duplicate column names make reorder alignment ill-defined → reject
    # (mirrors compare_column_order's duplicate-name guard).
    assert tg.compare_column_order_relaxed(
        [(1.0, 2.0)], [(1.0, 2.0)],
        pred_cols=["a", "a"], gold_cols=["a", "a"],
        epsilon=1e-6, strip=False, casefold=False,
    ) is False


def test_compare_column_order_relaxed_rejects_name_mismatch():
    # Column NAME sets differ → no alignment possible → False (the
    # non-reorder relaxed path in N9 is what rescues this case).
    assert tg.compare_column_order_relaxed(
        [(1.0, 2.0)], [(1.0, 2.0)],
        pred_cols=["a", "b"], gold_cols=["c", "d"],
        epsilon=1e-6, strip=False, casefold=False,
    ) is False


# ---------------------------------------------------------------------------
# grade_submission tier wiring (benchmark=None → no 2dp normalization, so
# the exact epsilon offsets below are the differentiator).
# ---------------------------------------------------------------------------


class _KeyedExec:
    """Executor returning canned ``(rows, cols)`` keyed by a substring of
    the SQL. Mirrors the FakeExec pattern used across the grader tests."""

    def __init__(self, table: dict[str, tuple[list, list]]):
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
        instance_id="alien_1", selected_database="alien",
        annotated_by="test", annotated_at="2026-06-01",
        amb_user_query="x",
        original_gold_is_correct=True,
        gold_variants=[],
        metadata_sufficiency=MetadataSufficiency(verdict="sufficient", rationale="r"),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="alien_1",
        ),
    )


def _grade(*, pred_payload, gold_payload, original_payload):
    """Run grade_submission with a single (primary) audited variant whose
    rows are ``gold_payload`` and an original gold returning
    ``original_payload`` (so N1 fails when it differs from pred)."""
    table = {
        "PRED": pred_payload,
        "GOLD": gold_payload,
        "ORIG": original_payload,
    }
    return tg.grade_submission(
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
        benchmark=None,
    )


def test_grade_reorder_plus_epsilon_passes_at_n8():
    """(column-reorder + epsilon) — fails N6 (positional epsilon) but
    passes N8 (reorder ∘ epsilon). Pre-fix this was ``agent_miss``."""
    v = _grade(
        pred_payload=([(2.0000005, 1.0000005)], ["cost", "carbon"]),
        gold_payload=([(1.0, 2.0)], ["carbon", "cost"]),
        original_payload=([(9.0, 9.0)], ["carbon", "cost"]),
    )
    assert v.n6_numeric_epsilon is False
    assert v.n8_column_order is True
    assert v.n9_case_fold is True  # monotone


def test_grade_reorder_plus_case_passes_at_n9_not_n8():
    v = _grade(
        pred_payload=([(1.0, "abc")], ["x", "label"]),
        gold_payload=([("ABC", 1.0)], ["label", "x"]),
        original_payload=([("ZZZ", 9.0)], ["label", "x"]),
    )
    assert v.n8_column_order is False
    assert v.n9_case_fold is True


def test_grade_same_order_epsilon_plus_case_mismatched_names_passes_at_n9():
    """Codex #3: same column ORDER but DIFFERENT names, correct only under
    epsilon+case. compare_case_fold (case only) and the name-aligned
    reorder path both fail; only the non-reorder ``compare_relaxed`` path
    in N9 rescues it."""
    v = _grade(
        pred_payload=([("abc", 1.0000005)], ["a", "b"]),
        gold_payload=([("ABC", 1.0)], ["c", "d"]),
        original_payload=([("ZZZ", 9.0)], ["c", "d"]),
    )
    assert v.n8_column_order is False
    assert v.n9_case_fold is True


def test_grade_epsilon_only_still_passes_at_n6():
    """Pure positional epsilon (no reorder) must still pass at N6 — the
    composition must not regress the independent tiers."""
    v = _grade(
        pred_payload=([(1.0000005, 2.0000005)], ["carbon", "cost"]),
        gold_payload=([(1.0, 2.0)], ["carbon", "cost"]),
        original_payload=([(9.0, 9.0)], ["carbon", "cost"]),
    )
    assert v.n6_numeric_epsilon is True


def test_grade_genuinely_wrong_is_agent_miss():
    """A disjoint answer passes no tier — composition must not soften a
    real miss."""
    v = _grade(
        pred_payload=([(100.0, 200.0)], ["carbon", "cost"]),
        gold_payload=([(1.0, 2.0)], ["carbon", "cost"]),
        original_payload=([(9.0, 9.0)], ["carbon", "cost"]),
    )
    assert v.n6_numeric_epsilon is False
    assert v.n8_column_order is False
    assert v.n9_case_fold is False
