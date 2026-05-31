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
