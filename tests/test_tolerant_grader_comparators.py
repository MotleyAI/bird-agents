"""DEV-1515: pure-comparator unit tests for tolerant_grader.

These exercise N4 (tie-order), N6 (numeric-epsilon), N7 (trailing-
whitespace), and N8 (column-order) on hand-canned row tuples. No SQLite
execution — purely the comparator predicates.

The end-to-end cascade is tested separately in
``test_tolerant_grader_orchestration.py`` with a fake executor.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# N4 — tie-order tolerance (bucket by ORDER BY columns; set-equal each bucket)
# ---------------------------------------------------------------------------


def test_n4_equal_rows_same_order_passes():
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    pred = [("A", 1), ("B", 2), ("B", 3)]
    gold = [("A", 1), ("B", 2), ("B", 3)]
    assert compare_tie_order(pred, gold, orderby_indices=[0]) is True


def test_n4_reordered_within_bucket_passes():
    """Bucket by column 0 (ORDER BY first col). Rows with the same
    bucket-key should set-equal within the bucket regardless of in-bucket
    order."""
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    pred = [("A", 1), ("B", 3), ("B", 2)]
    gold = [("A", 1), ("B", 2), ("B", 3)]
    assert compare_tie_order(pred, gold, orderby_indices=[0]) is True


def test_n4_reordered_across_buckets_fails():
    """Different bucket order is a real failure — ORDER BY column is
    the user-visible ordering."""
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    pred = [("B", 2), ("B", 3), ("A", 1)]
    gold = [("A", 1), ("B", 2), ("B", 3)]
    assert compare_tie_order(pred, gold, orderby_indices=[0]) is False


def test_n4_no_orderby_keys_collapses_to_strict():
    """An empty ``orderby_indices`` means "no ORDER BY in gold" — N4
    should fall back to set-equality (which is what N3 does)."""
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    pred = [("X", 1), ("Y", 2)]
    gold = [("Y", 2), ("X", 1)]
    assert compare_tie_order(pred, gold, orderby_indices=[]) is True


def test_n4_two_orderby_columns_bucket_by_tuple():
    """ORDER BY first_col, second_col → bucket key is the (col0, col1) tuple."""
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    pred = [("A", 1, "Z"), ("A", 1, "Y"), ("B", 2, "X")]
    gold = [("A", 1, "Y"), ("A", 1, "Z"), ("B", 2, "X")]
    assert compare_tie_order(pred, gold, orderby_indices=[0, 1]) is True


# ---------------------------------------------------------------------------
# N6 — numeric-epsilon tolerance
# ---------------------------------------------------------------------------


def test_n6_default_epsilon_matches_close_floats():
    from bird_interact_agents.eval.tolerant_grader import (
        compare_numeric_epsilon,
    )

    pred = [(1.0000001,)]
    gold = [(1.0,)]
    assert compare_numeric_epsilon(pred, gold, epsilon=1e-6) is True


def test_n6_outside_epsilon_fails():
    from bird_interact_agents.eval.tolerant_grader import (
        compare_numeric_epsilon,
    )

    pred = [(1.01,)]
    gold = [(1.0,)]
    assert compare_numeric_epsilon(pred, gold, epsilon=1e-6) is False


def test_n6_non_numeric_cells_compared_strictly():
    """String cells must still strict-equal under N6 — epsilon is per-cell
    type-aware."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_numeric_epsilon,
    )

    pred = [("hi", 1.0)]
    gold = [("HI", 1.0)]
    assert compare_numeric_epsilon(pred, gold, epsilon=1e-6) is False


def test_n6_int_and_float_with_same_value_match():
    """`1` (int) and `1.0` (float) should compare equal under numeric-eps."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_numeric_epsilon,
    )

    pred = [(1,)]
    gold = [(1.0,)]
    assert compare_numeric_epsilon(pred, gold, epsilon=1e-6) is True


def test_n6_null_cells_compared_strictly():
    """`None` vs `None` matches; `None` vs a value does not."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_numeric_epsilon,
    )

    assert compare_numeric_epsilon(
        [(None,)], [(None,)], epsilon=1e-6,
    ) is True
    assert compare_numeric_epsilon(
        [(None,)], [(0.0,)], epsilon=1e-6,
    ) is False


def test_n6_row_count_mismatch_fails():
    from bird_interact_agents.eval.tolerant_grader import (
        compare_numeric_epsilon,
    )

    pred = [(1.0,), (2.0,)]
    gold = [(1.0,)]
    assert compare_numeric_epsilon(pred, gold, epsilon=1e-6) is False


# ---------------------------------------------------------------------------
# N7 — trailing-whitespace tolerance
# ---------------------------------------------------------------------------


def test_n7_trailing_whitespace_strip_matches():
    from bird_interact_agents.eval.tolerant_grader import (
        compare_trailing_whitespace,
    )

    pred = [("High Income ",), ("Low Income\t",)]
    gold = [("High Income",), ("Low Income",)]
    assert compare_trailing_whitespace(pred, gold) is True


def test_n7_internal_whitespace_preserved():
    """Stripping trailing only — internal spaces matter."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_trailing_whitespace,
    )

    pred = [("High  Income",)]  # two internal spaces
    gold = [("High Income",)]
    assert compare_trailing_whitespace(pred, gold) is False


def test_n7_non_string_cells_unchanged():
    """Numeric cells unaffected — they're not strings to strip."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_trailing_whitespace,
    )

    pred = [("A ", 1.0)]
    gold = [("A", 1.0)]
    assert compare_trailing_whitespace(pred, gold) is True


# ---------------------------------------------------------------------------
# N8 — column-order tolerance via case-insensitive name align
# ---------------------------------------------------------------------------


def test_n8_reordered_columns_match_by_name():
    from bird_interact_agents.eval.tolerant_grader import compare_column_order

    pred_rows = [("foo", 1), ("bar", 2)]
    pred_cols = ["name", "id"]
    gold_rows = [(1, "foo"), (2, "bar")]
    gold_cols = ["id", "name"]
    assert compare_column_order(
        pred_rows, gold_rows, pred_cols=pred_cols, gold_cols=gold_cols,
    ) is True


def test_n8_column_name_case_insensitive():
    from bird_interact_agents.eval.tolerant_grader import compare_column_order

    pred_rows = [(1, "foo")]
    pred_cols = ["ID", "NAME"]
    gold_rows = [(1, "foo")]
    gold_cols = ["id", "name"]
    assert compare_column_order(
        pred_rows, gold_rows, pred_cols=pred_cols, gold_cols=gold_cols,
    ) is True


def test_n8_missing_column_name_fails():
    """Predicted has an extra column or a different name — N8 cannot align."""
    from bird_interact_agents.eval.tolerant_grader import compare_column_order

    pred_rows = [(1, "foo")]
    pred_cols = ["id", "label"]
    gold_rows = [(1, "foo")]
    gold_cols = ["id", "name"]
    assert compare_column_order(
        pred_rows, gold_rows, pred_cols=pred_cols, gold_cols=gold_cols,
    ) is False


def test_n8_column_count_mismatch_fails():
    from bird_interact_agents.eval.tolerant_grader import compare_column_order

    pred_rows = [(1, "foo", "extra")]
    pred_cols = ["id", "name", "extra"]
    gold_rows = [(1, "foo")]
    gold_cols = ["id", "name"]
    assert compare_column_order(
        pred_rows, gold_rows, pred_cols=pred_cols, gold_cols=gold_cols,
    ) is False


# ---------------------------------------------------------------------------
# N9 — case-fold tolerance on string cells
# ---------------------------------------------------------------------------


def test_n9_case_fold_lifts_case_only_mismatch():
    from bird_interact_agents.eval.tolerant_grader import compare_case_fold

    pred = [("HIGH",), ("Low",)]
    gold = [("high",), ("low",)]
    assert compare_case_fold(pred, gold) is True


def test_n9_non_string_cells_unchanged():
    from bird_interact_agents.eval.tolerant_grader import compare_case_fold

    pred = [("A", 1.0)]
    gold = [("a", 1.0)]
    assert compare_case_fold(pred, gold) is True


def test_n9_row_count_mismatch_fails():
    from bird_interact_agents.eval.tolerant_grader import compare_case_fold

    pred = [("A",), ("B",)]
    gold = [("a",)]
    assert compare_case_fold(pred, gold) is False


def test_n9_content_difference_beyond_case_fails():
    """Case-fold must not paper over genuine content differences."""
    from bird_interact_agents.eval.tolerant_grader import compare_case_fold

    pred = [("Apple",)]
    gold = [("orange",)]
    assert compare_case_fold(pred, gold) is False


# ---------------------------------------------------------------------------
# ORDER BY parser — N4 input
# ---------------------------------------------------------------------------


def test_parse_orderby_no_clause_returns_empty():
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    assert parse_orderby_keys("SELECT a, b FROM t") == []


def test_parse_orderby_named_column_resolves_to_select_index():
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    keys = parse_orderby_keys("SELECT a, b FROM t ORDER BY b")
    indices = [k.column_index for k in keys]
    assert indices == [1]


def test_parse_orderby_bare_integer_uses_as_select_index():
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    keys = parse_orderby_keys("SELECT a, b FROM t ORDER BY 2 DESC")
    indices = [k.column_index for k in keys]
    assert indices == [1]


def test_parse_orderby_alias_resolves_to_select_index():
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    keys = parse_orderby_keys(
        "SELECT a, b AS total FROM t ORDER BY total"
    )
    indices = [k.column_index for k in keys]
    assert indices == [1]


def test_parse_orderby_multiple_keys_preserved_in_order():
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    keys = parse_orderby_keys(
        "SELECT a, b, c FROM t ORDER BY a, c DESC, b"
    )
    indices = [k.column_index for k in keys]
    assert indices == [0, 2, 1]


def test_parse_orderby_expression_not_in_select_returns_none_marker():
    """`ORDER BY a + b` when `a + b` is not a select-list expression
    cannot be mapped to a column index. The parser flags this so the
    caller can fall back to N3 strict-equality for this variant."""
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    keys = parse_orderby_keys("SELECT a, b FROM t ORDER BY a + b")
    # The key carries no resolvable index → caller collapses N4 to N3.
    assert any(k.column_index is None for k in keys)


def test_parse_orderby_nulls_first_last_does_not_error():
    """SQLite's NULLS FIRST/LAST syntax must parse without crashing — the
    actual NULL ordering is a downstream concern; for bucketing we only
    care about the column key."""
    from bird_interact_agents.eval.tolerant_grader import parse_orderby_keys

    keys = parse_orderby_keys(
        "SELECT a, b FROM t ORDER BY a NULLS LAST, b NULLS FIRST"
    )
    assert [k.column_index for k in keys] == [0, 1]


# ---------------------------------------------------------------------------
# Codex r12: comparator-boundary cases that pre-fix would either crash
# the grader (compare_tie_order index out-of-range) or falsely pass
# (compare_column_order with duplicate names). Both surface from valid
# agent misses, so they MUST land as ``False`` cleanly, not as
# AssertionError / silent-pass.
# ---------------------------------------------------------------------------


def test_compare_tie_order_returns_false_when_pred_too_narrow():
    """A wrong-projection agent submission (fewer columns than the
    gold ORDER BY references) must NOT crash the grader. Pre-fix
    ``row[i]`` raised ``IndexError`` and the cloud/local fallback
    wrote a generic fail-everything annotation instead of a structured
    cascade-miss."""
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    # ORDER BY column index 5 — pred only has 2 columns per row.
    pred = [("A", 1), ("B", 2)]
    gold = [("A", 1, 0, 0, 0, "X"), ("B", 2, 0, 0, 0, "Y")]
    assert (
        compare_tie_order(pred, gold, orderby_indices=[5]) is False
    ), "must return False, not raise IndexError"


def test_compare_tie_order_returns_false_when_gold_too_narrow():
    """Symmetric: gold too narrow for an index also returns False
    cleanly."""
    from bird_interact_agents.eval.tolerant_grader import compare_tie_order

    pred = [("A", 1, 0, 0, 0, "X"), ("B", 2, 0, 0, 0, "Y")]
    gold = [("A", 1), ("B", 2)]
    assert (
        compare_tie_order(pred, gold, orderby_indices=[5]) is False
    ), "must return False, not raise IndexError"


def test_compare_column_order_rejects_duplicate_column_names_pred():
    """N8 tolerance: pred has a duplicate-named projection. The pre-fix
    ``pred_l.index(c)`` mapped every duplicate occurrence in gold to
    pred's FIRST matching position, silently ignoring later
    duplicate columns' values. With duplicates the column-order
    concept is ill-defined — return False rather than falsely pass."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_column_order,
    )

    # Pred: 3 columns named (a, b, a). Values for the duplicate "a"
    # columns DIFFER, so a real comparison must NOT pass.
    pred = [("X", "Y", "Z"), ("X2", "Y2", "Z2")]
    gold = [("X", "X", "Y"), ("X2", "X2", "Y2")]  # gold reads a,a,b
    assert compare_column_order(
        pred, gold,
        pred_cols=["a", "b", "a"],
        gold_cols=["a", "a", "b"],
    ) is False, (
        "must NOT falsely pass — pred's second 'a' column has Z/Z2 "
        "but gold expects X/X2; duplicate names make column-order "
        "ill-defined"
    )


def test_compare_column_order_rejects_duplicate_column_names_gold():
    """Symmetric: duplicates on the GOLD side also return False."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_column_order,
    )

    pred = [("X", "Y", "Z")]
    gold = [("X", "X", "Y")]
    assert compare_column_order(
        pred, gold,
        pred_cols=["a", "b", "c"],
        gold_cols=["a", "a", "b"],
    ) is False


def test_compare_column_order_distinct_names_still_pass():
    """Belt-and-braces: the duplicate guard MUST NOT regress the
    canonical "same names, different order" pass."""
    from bird_interact_agents.eval.tolerant_grader import (
        compare_column_order,
    )

    pred = [("X", "Y"), ("X2", "Y2")]
    gold = [("Y", "X"), ("Y2", "X2")]
    assert compare_column_order(
        pred, gold,
        pred_cols=["a", "b"],
        gold_cols=["b", "a"],
    ) is True
