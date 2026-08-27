"""DEV-1822 (Codex C1): materialize Cube's parameterized `/v1/sql` into a
standalone, regrade-safe Postgres statement.

Cube returns `sql.sql = [text_with_$N, params]`; the graded submission must be
one executable SQL string with the params substituted as safe literals, while
`$N` tokens inside string literals / quoted identifiers are left untouched.
"""

from __future__ import annotations

import datetime as dt

import pytest

from bird_interact_agents.cube_local.sql_render import materialize_sql, render_literal


# --- literal rendering ------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "NULL"),
        (True, "TRUE"),
        (False, "FALSE"),
        (3, "3"),
        (1.5, "1.5"),
        ("plain", "'plain'"),
        ("O'Brien", "'O''Brien'"),          # single-quote doubling
        ("US", "'US'"),
        (dt.date(2021, 5, 1), "'2021-05-01'"),
    ],
)
def test_render_literal(value, expected):
    assert render_literal(value) == expected


def test_render_literal_array():
    assert render_literal([1, 2, 3]) == "ARRAY[1, 2, 3]"
    assert render_literal(["a", "b"]) == "ARRAY['a', 'b']"


def test_render_literal_rejects_unknown_type():
    with pytest.raises((TypeError, ValueError)):
        render_literal({"a": 1})


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), float("-inf")])
def test_render_literal_rejects_non_finite(bad):
    with pytest.raises((ValueError, TypeError)):
        render_literal(bad)


# --- placeholder substitution ----------------------------------------------

def test_basic_substitution():
    out = materialize_sql("SELECT * FROM t WHERE a = $1 AND b = $2", ["x", 3])
    assert out == "SELECT * FROM t WHERE a = 'x' AND b = 3"


def test_injection_shaped_value_is_escaped_not_interpolated():
    out = materialize_sql("SELECT * FROM t WHERE name = $1", ["x'; DROP TABLE t; --"])
    assert out == "SELECT * FROM t WHERE name = 'x''; DROP TABLE t; --'"


def test_placeholder_inside_string_literal_is_untouched():
    # The literal text '$1' must survive; only the real placeholder is filled.
    out = materialize_sql("SELECT '$1' AS lit, x FROM t WHERE y = $1", ["v"])
    assert out == "SELECT '$1' AS lit, x FROM t WHERE y = 'v'"


def test_placeholder_inside_quoted_identifier_is_untouched():
    out = materialize_sql('SELECT "col$1", x FROM t WHERE y = $1', ["v"])
    assert out == 'SELECT "col$1", x FROM t WHERE y = \'v\''


def test_repeated_placeholder_uses_same_param():
    out = materialize_sql("SELECT $1 WHERE a = $1", ["v"])
    assert out == "SELECT 'v' WHERE a = 'v'"


def test_multi_digit_placeholders():
    params = [f"v{i}" for i in range(1, 13)]
    out = materialize_sql("x=$11 y=$1 z=$12", params)
    assert out == "x='v11' y='v1' z='v12'"


def test_no_params_no_placeholders():
    assert materialize_sql("SELECT 1", []) == "SELECT 1"


def test_extra_unused_params_tolerated():
    # lenient contract: Cube always supplies exactly-used params; surplus is harmless
    assert materialize_sql("SELECT 1", ["unused"]) == "SELECT 1"
    assert materialize_sql("WHERE a = $1", ["x", "extra"]) == "WHERE a = 'x'"


def test_placeholder_without_param_raises():
    with pytest.raises((ValueError, IndexError)):
        materialize_sql("SELECT $1, $2", ["only-one"])
