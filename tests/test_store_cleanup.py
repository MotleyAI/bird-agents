"""DEV-1671: unit tests for the identical-result comparison helpers."""

from __future__ import annotations

from bird_interact_agents.slayer_otf.store_cleanup import (
    extract_result_rows,
    results_identical,
    values_equal,
)


def test_extract_rows_from_tool_json_with_trailing_prose():
    out = ('[{"return_sal.avg_sal": "-256.47"}]\n\nMeasure attributes:\n'
           '  return_sal.avg_sal: format=(type=float)')
    assert extract_result_rows(out) == [["-256.47"]]


def test_extract_rows_from_list():
    assert extract_result_rows([{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == [[1, 2], [3, 4]]


def test_extract_rows_non_json_returns_empty():
    assert extract_result_rows("Error: model not found") == []


def test_extract_rows_bracket_inside_string_value():
    # Codex #8: a ']' inside a JSON string value must not truncate the parse
    out = '[{"note": "a]b", "v": "5"}]\n\nMeasure attributes: ...'
    assert extract_result_rows(out) == [["a]b", "5"]]


def test_identical_ignores_column_headers():
    a = '[{"return_sal.avg_sal": "-256.47"}]'
    b = '[{"return_sal.round(sal_avg,_2)": "-256.47"}]'
    assert results_identical(a, b) is True


def test_identical_numeric_decimal_equality():
    # trailing-zero / formatting differences are equal as Decimals
    assert results_identical('[{"x": "10.50"}]', '[{"x": "10.5"}]') is True


def test_not_identical_on_value_difference():
    assert results_identical('[{"x": "1"}]', '[{"x": "2"}]') is False


def test_not_identical_on_row_count():
    assert results_identical('[{"x": 1}]', '[{"x": 1}, {"x": 1}]') is False


def test_round_ndigits_tolerance():
    # both round to 1.23 at 2 dp
    assert results_identical('[{"x": "1.234"}]', '[{"x": "1.2338"}]', round_ndigits=2) is True
    # 1.23 vs 1.24 at 2 dp
    assert results_identical('[{"x": "1.234"}]', '[{"x": "1.244"}]', round_ndigits=2) is False


def test_null_handling():
    assert values_equal(None, None) is True
    assert values_equal(None, 0) is False
    assert results_identical('[{"x": null}]', '[{"x": null}]') is True


def test_order_sensitive_vs_multiset():
    a = '[{"x": 1}, {"x": 2}]'
    b = '[{"x": 2}, {"x": 1}]'
    assert results_identical(a, b) is False
    assert results_identical(a, b, order_sensitive=False) is True


def test_multi_column_positional():
    a = '[{"m.acct": "A", "m.srs": "0.90"}]'
    b = '[{"s.acct": "A", "s.srs": "0.9"}]'
    assert results_identical(a, b) is True
    # positional: a swap of columns is NOT identical
    c = '[{"m.srs": "0.90", "m.acct": "A"}]'
    assert results_identical(a, c) is False
