"""DEV-1648: pure NULL-safe Postgres cast emission (`casts.py`).

Pins the canonical SQL strings emitted for refined-over-TEXT columns so
the drift check sees a *derived* (non-bare) expression and aggregation
input is cast. All string-level, no DB — the erroring-value behaviour is
covered by the ``@pytest.mark.integration`` live-PG tests.
"""

from __future__ import annotations

import pytest

from slayer.core.models import DataType

from bird_interact_agents.slayer_pipeline.casts import (
    INTEGER_REGEX,
    NUMERIC_REGEX,
    pg_nullsafe_cast,
    quote_ident,
    strptime_to_pg_format,
    strptime_to_regex,
)


# ---------------------------------------------------------------------------
# quote_ident
# ---------------------------------------------------------------------------


def test_quote_ident_basic() -> None:
    assert quote_ident("score_val") == '"score_val"'


def test_quote_ident_escapes_embedded_quote() -> None:
    assert quote_ident('we"ird') == '"we""ird"'


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------


def test_numeric_regex_constant() -> None:
    assert NUMERIC_REGEX == r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$"


def test_integer_regex_is_unbounded_integer_shape() -> None:
    # Unbounded: the cast target is ::numeric (no overflow), so any-length
    # integer text is accepted with no valid-value loss.
    assert INTEGER_REGEX == r"^\s*[+-]?\d+\s*$"


# ---------------------------------------------------------------------------
# strptime -> Postgres to_char/to_timestamp format
# ---------------------------------------------------------------------------


def test_pg_format_iso_date() -> None:
    assert strptime_to_pg_format("%Y-%m-%d") == "YYYY-MM-DD"


def test_pg_format_dotted_datetime() -> None:
    assert strptime_to_pg_format("%Y.%m.%d %H:%M:%S") == "YYYY.MM.DD HH24:MI:SS"


def test_pg_format_iso_t_literal_and_microseconds() -> None:
    # The 'T' literal must be double-quoted for Postgres; %f -> US.
    assert (
        strptime_to_pg_format("%Y-%m-%dT%H:%M:%S.%f")
        == 'YYYY-MM-DD"T"HH24:MI:SS.US'
    )


def test_pg_format_slash_dmy() -> None:
    assert strptime_to_pg_format("%d/%m/%Y") == "DD/MM/YYYY"


def test_pg_format_unsupported_tokens_return_none() -> None:
    # Month-name / 12h / AM-PM / day-of-year are unsupported in v1.
    assert strptime_to_pg_format("%B %d, %Y") is None
    assert strptime_to_pg_format("%I:%M %p") is None
    assert strptime_to_pg_format("%j") is None


# ---------------------------------------------------------------------------
# strptime -> validity regex (shape guard)
# ---------------------------------------------------------------------------


def test_regex_iso_date() -> None:
    assert strptime_to_regex("%Y-%m-%d") == r"^\s*\d{4}-\d{2}-\d{2}\s*$"


def test_regex_dotted_date_escapes_dots() -> None:
    assert strptime_to_regex("%Y.%m.%d") == r"^\s*\d{4}\.\d{2}\.\d{2}\s*$"


def test_regex_fraction_is_optional() -> None:
    # A ".%f" tail becomes an OPTIONAL group so a column mixing
    # '…:22.330375' and '…:22' both validate.
    assert (
        strptime_to_regex("%Y-%m-%d %H:%M:%S.%f")
        == r"^\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d{1,6})?\s*$"
    )


def test_regex_t_literal_not_escaped() -> None:
    assert (
        strptime_to_regex("%Y-%m-%dT%H:%M:%S")
        == r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*$"
    )


# ---------------------------------------------------------------------------
# pg_nullsafe_cast — numeric
# ---------------------------------------------------------------------------


def test_cast_double_bare_column() -> None:
    assert pg_nullsafe_cast("created", DataType.DOUBLE) == (
        r"CASE WHEN (created) ~ '^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$' "
        r"THEN (created)::double precision END"
    )


def test_cast_int_uses_numeric() -> None:
    assert pg_nullsafe_cast("dur_sec", DataType.INT) == (
        r"CASE WHEN (dur_sec) ~ '^\s*[+-]?\d+\s*$' "
        r"THEN (dur_sec)::numeric END"
    )


# ---------------------------------------------------------------------------
# pg_nullsafe_cast — temporal
# ---------------------------------------------------------------------------


def test_cast_date() -> None:
    assert pg_nullsafe_cast("d", DataType.DATE, "%Y-%m-%d") == (
        r"CASE WHEN (d) ~ '^\s*\d{4}-\d{2}-\d{2}\s*$' "
        r"THEN to_date((d), 'YYYY-MM-DD') END"
    )


def test_cast_timestamp_dotted_forces_timestamp_type() -> None:
    # match_ts case: dotted, no fraction. to_timestamp returns timestamptz
    # so it is forced back to timestamp-without-tz.
    assert pg_nullsafe_cast("match_ts", DataType.TIMESTAMP, "%Y.%m.%d %H:%M:%S") == (
        r"CASE WHEN (match_ts) ~ '^\s*\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}\s*$' "
        r"THEN (to_timestamp((match_ts), 'YYYY.MM.DD HH24:MI:SS'))::timestamp END"
    )


def test_cast_timestamp_iso_microseconds_optional() -> None:
    # created_ts case: ISO with optional microseconds; default frac -> US.
    assert pg_nullsafe_cast("created_ts", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f") == (
        r"CASE WHEN (created_ts) ~ '^\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d{1,6})?\s*$' "
        r"THEN (to_timestamp((created_ts), 'YYYY-MM-DD HH24:MI:SS.US'))::timestamp END"
    )


def test_cast_timestamp_millisecond_width() -> None:
    # 3-digit fraction columns must map %f -> MS (not US) so the value is
    # scaled correctly (330 ms, not 330 us).
    assert pg_nullsafe_cast(
        "c", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f", frac_pg="MS"
    ) == (
        r"CASE WHEN (c) ~ '^\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d{1,6})?\s*$' "
        r"THEN (to_timestamp((c), 'YYYY-MM-DD HH24:MI:SS.MS'))::timestamp END"
    )


def test_cast_date_dotted_uses_to_date() -> None:
    # Non-ISO date still goes through to_date (uniform, non-aborting).
    assert pg_nullsafe_cast("d", DataType.DATE, "%Y.%m.%d") == (
        r"CASE WHEN (d) ~ '^\s*\d{4}\.\d{2}\.\d{2}\s*$' "
        r"THEN to_date((d), 'YYYY.MM.DD') END"
    )


# ---------------------------------------------------------------------------
# pg_nullsafe_cast — None / unsupported paths
# ---------------------------------------------------------------------------


def test_cast_temporal_without_format_returns_none() -> None:
    assert pg_nullsafe_cast("d", DataType.DATE) is None
    assert pg_nullsafe_cast("d", DataType.TIMESTAMP, None) is None


def test_cast_unsupported_format_returns_none() -> None:
    # %B is unsupported -> caller leaves the column TEXT + warns.
    assert pg_nullsafe_cast("d", DataType.DATE, "%B %d, %Y") is None


def test_cast_non_castable_types_return_none() -> None:
    assert pg_nullsafe_cast("c", DataType.TEXT) is None
    assert pg_nullsafe_cast("c", DataType.BOOLEAN) is None


# ---------------------------------------------------------------------------
# Inner-expression genericity (JSON leaf reuse) + SQL-literal safety
# ---------------------------------------------------------------------------


def test_cast_wraps_arbitrary_inner_expression() -> None:
    inner = "jsonb_extract_path_text(\"socioeconomic\", 'Bath_Count')"
    out = pg_nullsafe_cast(inner, DataType.DOUBLE)
    assert out == (
        f"CASE WHEN ({inner}) ~ '{NUMERIC_REGEX}' "
        f"THEN ({inner})::double precision END"
    )


def test_cast_preserves_regex_backslashes() -> None:
    # Postgres standard_conforming_strings=on keeps backslashes literal,
    # so the emitted SQL literal must still contain '\d' / '\s'.
    out = pg_nullsafe_cast("c", DataType.DOUBLE)
    assert r"\d" in out and r"\s" in out
