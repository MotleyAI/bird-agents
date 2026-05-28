"""Tests for deterministic Mode-B filter normalization (DEV-1478 follow-up).

`normalize_mode_b_filter` wraps the column side of in-scope text comparisons in
`lower(trim(...))` and lowercases the literal; `normalize_tool_filters` /
`normalize_query_payload` apply it to the filter-bearing args of each query
surface. See plan: case/whitespace mismatch on text filters caused
`wrong_result` losses; the NL questions carry no casing info, so normalized
matching is the semantically correct reading.
"""

from __future__ import annotations

from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_mode_b_filter,
    normalize_query_payload,
    normalize_tool_filters,
)


# ---------------------------------------------------------------------------
# normalize_mode_b_filter — single-expression rewrite
# ---------------------------------------------------------------------------


def test_eq_python_style():
    assert (
        normalize_mode_b_filter("category == 'Gadgets'")
        == "lower(trim(category)) == 'gadgets'"
    )


def test_eq_sql_style_single_equals():
    """SQL-style single `=` must canonicalize (proves the
    `_preprocess_sql_operators` reuse — bare ast.parse would choke)."""
    assert (
        normalize_mode_b_filter("category = 'Gadgets'")
        == "lower(trim(category)) == 'gadgets'"
    )


def test_neq_angle_brackets():
    assert (
        normalize_mode_b_filter("category <> 'X'")
        == "lower(trim(category)) != 'x'"
    )


def test_neq_python_style():
    assert (
        normalize_mode_b_filter("category != 'X'")
        == "lower(trim(category)) != 'x'"
    )


def test_in_uppercase_keyword_and_tuple():
    out = normalize_mode_b_filter("category IN ('A', 'B')")
    assert out == "lower(trim(category)) in ('a', 'b')"


def test_not_in():
    out = normalize_mode_b_filter("category not in ('A', 'B')")
    assert out == "lower(trim(category)) not in ('a', 'b')"


def test_already_wrapped_but_uppercase_literal_still_lowercased():
    """Codex HIGH: a pre-wrapped LHS with an un-lowered literal must still get
    the literal lowercased — otherwise lower(trim(col))='Apartment' never
    matches at runtime."""
    assert (
        normalize_mode_b_filter("lower(trim(category)) == 'Gadgets'")
        == "lower(trim(category)) == 'gadgets'"
    )


def test_canonical_form_is_fixed_point():
    canonical = "lower(trim(category)) == 'gadgets'"
    assert normalize_mode_b_filter(canonical) == canonical


def test_partial_wrap_canonicalized():
    """`lower(col)` (no trim) → canonical `lower(trim(col))`."""
    assert (
        normalize_mode_b_filter("lower(category) == 'X'")
        == "lower(trim(category)) == 'x'"
    )


def test_dotted_column_reference():
    assert (
        normalize_mode_b_filter("model.col == 'X'")
        == "lower(trim(model.col)) == 'x'"
    )


def test_variable_placeholder_left_untouched():
    """Codex MEDIUM: SLayer substitutes `{Var}` before parsing; lowercasing
    `'{Status}'` → `'{status}'` would break substitution."""
    assert (
        normalize_mode_b_filter("status == '{Status}'")
        == "status == '{Status}'"
    )


def test_numeric_rhs_untouched():
    assert normalize_mode_b_filter("age == 25") == "age == 25"
    assert normalize_mode_b_filter("score > 0.5") == "score > 0.5"


def test_ordering_operator_untouched():
    # `>` is out of scope even with a string RHS (rare, but don't touch).
    assert normalize_mode_b_filter("created > '2024-01-01'") == (
        "created > '2024-01-01'"
    )


def test_compound_only_string_eq_side_normalized():
    out = normalize_mode_b_filter("category == 'X' and qty > 5")
    assert out == "lower(trim(category)) == 'x' and qty > 5"


def test_like_left_untouched():
    assert normalize_mode_b_filter("name like 'x%'") == "name like 'x%'"


def test_colon_aggregation_untouched():
    assert normalize_mode_b_filter("amount:count > 5") == "amount:count > 5"


def test_unparseable_is_noop():
    weird = "this is not <<>> valid mode-b @@"
    assert normalize_mode_b_filter(weird) == weird


def test_empty_and_non_string_inputs():
    assert normalize_mode_b_filter("") == ""
    assert normalize_mode_b_filter("   ") == "   "


def test_column_vs_column_untouched():
    # RHS is a column, not a string literal → out of scope.
    assert normalize_mode_b_filter("col_a == col_b") == "col_a == col_b"


def test_combined_text_and_aggregation_in_one_string_is_noop():
    """v1 boundary (Codex LOW): a text predicate COMBINED with an
    aggregation in ONE filter string no-ops (the colon-agg isn't handled
    by _preprocess_sql_operators, so the whole string fails ast.parse and
    is returned unchanged). The idiomatic form is separate list entries."""
    combined = "category == 'X' and amount:sum > 5"
    assert normalize_mode_b_filter(combined) == combined
    # …whereas split into separate entries, the text side normalizes:
    assert normalize_mode_b_filter("category == 'X'") == (
        "lower(trim(category)) == 'x'"
    )


# ---------------------------------------------------------------------------
# normalize_query_payload — submit path (single dict / list of stages)
# ---------------------------------------------------------------------------


def test_payload_single_dict():
    q = {"source_model": "widgets", "dimensions": ["category"],
         "filters": ["category == 'Gadgets'"]}
    out = normalize_query_payload(q)
    assert out["filters"] == ["lower(trim(category)) == 'gadgets'"]
    # dimensions / source_model untouched
    assert out["dimensions"] == ["category"]
    assert out["source_model"] == "widgets"


def test_payload_does_not_mutate_input():
    q = {"filters": ["category == 'Gadgets'"]}
    _ = normalize_query_payload(q)
    assert q["filters"] == ["category == 'Gadgets'"], "input must not be mutated"


def test_payload_list_of_stages():
    stages = [
        {"name": "s1", "source_model": "widgets",
         "filters": ["category == 'A'"]},
        {"source_model": "s1", "filters": ["region == 'B'"]},
    ]
    out = normalize_query_payload(stages)
    assert out[0]["filters"] == ["lower(trim(category)) == 'a'"]
    assert out[1]["filters"] == ["lower(trim(region)) == 'b'"]


def test_payload_no_filters_key():
    q = {"source_model": "widgets", "dimensions": ["category"]}
    assert normalize_query_payload(q) == q


# ---------------------------------------------------------------------------
# normalize_tool_filters — per-tool arg dispatch
# ---------------------------------------------------------------------------


def test_tool_query_top_level_filters():
    args = {"source_model": "widgets", "filters": ["category == 'Gadgets'"]}
    out = normalize_tool_filters("query", args)
    assert out["filters"] == ["lower(trim(category)) == 'gadgets'"]


def test_tool_query_nested_queries():
    args = {"queries": [
        {"source_model": "widgets", "filters": ["category == 'A'"]},
        {"source_model": "stage1", "filters": ["region == 'B'"]},
    ]}
    out = normalize_tool_filters("query_nested", args)
    assert out["queries"][0]["filters"] == ["lower(trim(category)) == 'a'"]
    assert out["queries"][1]["filters"] == ["lower(trim(region)) == 'b'"]


def test_tool_create_model_single_query_dict():
    args = {"name": "m", "query": {"source_model": "w",
                                   "filters": ["category == 'A'"]}}
    out = normalize_tool_filters("create_model", args)
    assert out["query"]["filters"] == ["lower(trim(category)) == 'a'"]


def test_tool_create_model_query_stage_list():
    args = {"name": "m", "query": [
        {"source_model": "w", "filters": ["category == 'A'"]},
        {"source_model": "s1", "filters": ["region == 'B'"]},
    ]}
    out = normalize_tool_filters("create_model", args)
    assert out["query"][0]["filters"] == ["lower(trim(category)) == 'a'"]
    assert out["query"][1]["filters"] == ["lower(trim(region)) == 'b'"]


def test_tool_edit_model_source_queries():
    args = {"name": "m", "source_queries": [
        {"source_model": "w", "filters": ["category == 'A'"]},
    ]}
    out = normalize_tool_filters("edit_model", args)
    assert out["source_queries"][0]["filters"] == ["lower(trim(category)) == 'a'"]


def test_tool_edit_model_add_filters_is_mode_a_untouched():
    """`add_filters`/`remove_filters` are Mode-A SQL — must NOT be touched."""
    args = {"name": "m", "add_filters": ["deleted_at IS NULL"],
            "remove_filters": ["status = 'X'"]}
    out = normalize_tool_filters("edit_model", args)
    assert out["add_filters"] == ["deleted_at IS NULL"]
    assert out["remove_filters"] == ["status = 'X'"]


def test_tool_filters_does_not_mutate_input():
    args = {"filters": ["category == 'Gadgets'"]}
    _ = normalize_tool_filters("query", args)
    assert args["filters"] == ["category == 'Gadgets'"]


def test_tool_unknown_tool_passthrough():
    args = {"filters": ["category == 'Gadgets'"]}
    # A non-filter-bearing tool name → returned unchanged (deep copy, same data)
    assert normalize_tool_filters("inspect_model", args) == args


def test_tool_non_dict_args_passthrough():
    assert normalize_tool_filters("query", None) is None
    assert normalize_tool_filters("query", "x") == "x"
