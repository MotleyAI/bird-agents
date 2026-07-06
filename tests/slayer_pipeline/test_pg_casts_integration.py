"""DEV-1648: execute the emitted NULL-safe casts against a REAL Postgres.

String-level tests pin the SQL; these prove the acceptance criterion that
the emitted expression NEVER aborts on an un-castable / overflowing /
invalid value (it yields NULL) and returns the right value on good input.

``@pytest.mark.integration`` — excluded from the default suite (needs a
live PG). Run with ``... pytest -m integration`` while the local
livesqlbench Postgres is up (BIRD_PG_* set), or it self-skips.
"""

from __future__ import annotations

import os

import pytest

from slayer.core.models import DataType

from bird_interact_agents.slayer_pipeline.casts import pg_nullsafe_cast

pytestmark = pytest.mark.integration


def _engine():
    sa = pytest.importorskip("sqlalchemy")
    from urllib.parse import quote

    host = os.environ.get("BIRD_PG_HOST", "localhost")
    port = os.environ.get("BIRD_PG_PORT", "5432")
    user = quote(os.environ.get("BIRD_PG_USER", "bird_interact"), safe="")
    password = quote(os.environ.get("BIRD_PG_PASSWORD", "bird_interact"), safe="")
    db = os.environ.get("BIRD_PG_DB", "postgres")
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    try:
        eng = sa.create_engine(url)
        with eng.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return eng, sa
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no reachable Postgres for integration test: {exc}")


def _eval_cast(expr: str, value):
    eng, sa = _engine()
    # Wrap the cast (written over column "c") around a text literal.
    sql = f"SELECT {expr} AS out FROM (SELECT :v ::text AS c) s"
    with eng.connect() as conn:
        return conn.execute(sa.text(sql), {"v": value}).scalar()


# ---------------------------------------------------------------------------
# Numeric — no abort, correct value / NULL
# ---------------------------------------------------------------------------


def test_double_good_value() -> None:
    expr = pg_nullsafe_cast("c", DataType.DOUBLE)
    assert _eval_cast(expr, "3.14") == pytest.approx(3.14)


def test_double_bad_value_is_null_not_error() -> None:
    expr = pg_nullsafe_cast("c", DataType.DOUBLE)
    assert _eval_cast(expr, "N/A") is None


def test_bigint_overflow_is_null_not_error() -> None:
    # 30-digit value: must NULL (bounded regex), never abort on ::bigint.
    expr = pg_nullsafe_cast("c", DataType.INT)
    assert _eval_cast(expr, "1" * 30) is None


def test_bigint_good_value() -> None:
    expr = pg_nullsafe_cast("c", DataType.INT)
    assert _eval_cast(expr, "128") == 128


# ---------------------------------------------------------------------------
# Temporal — no abort; fractional-second variants
# ---------------------------------------------------------------------------


def test_timestamp_iso_with_microseconds() -> None:
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f")
    out = _eval_cast(expr, "2025-02-19 08:31:22.330375")
    assert out is not None and str(out).startswith("2025-02-19 08:31:22")


def test_timestamp_iso_without_fraction_still_parses() -> None:
    # Optional-fraction format must also parse a no-fraction value.
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f")
    out = _eval_cast(expr, "2025-02-19 08:31:22")
    assert out is not None and str(out).startswith("2025-02-19 08:31:22")


def test_timestamp_dotted_value() -> None:
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y.%m.%d %H:%M:%S")
    out = _eval_cast(expr, "2025.02.19 08:31:22")
    assert out is not None and str(out).startswith("2025-02-19 08:31:22")


def test_timestamp_garbage_is_null_not_error() -> None:
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f")
    assert _eval_cast(expr, "not a timestamp") is None


def test_date_iso_t_literal() -> None:
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y-%m-%dT%H:%M:%S")
    out = _eval_cast(expr, "2025-02-19T08:31:22")
    assert out is not None and str(out).startswith("2025-02-19 08:31:22")


def test_timestamp_millisecond_width_scales_correctly() -> None:
    # 3-digit fraction under MS -> 330 milliseconds (not 330 microseconds).
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f", frac_pg="MS")
    out = _eval_cast(expr, "2025-02-19 08:31:22.330")
    assert out is not None and str(out).startswith("2025-02-19 08:31:22.33")


def test_timestamp_invalid_calendar_is_null_not_abort() -> None:
    # Shape-valid but impossible date (Feb 30). to_timestamp must NOT abort
    # the query — the whole point of avoiding ::timestamp.
    expr = pg_nullsafe_cast("c", DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f")
    # Must return SOMETHING (rolled-over date) or NULL, but never raise.
    out = _eval_cast(expr, "2025-02-30 08:31:22")
    assert out is None or str(out).startswith("2025-03")


def test_date_invalid_calendar_no_abort() -> None:
    expr = pg_nullsafe_cast("c", DataType.DATE, "%Y-%m-%d")
    out = _eval_cast(expr, "2025-13-01")  # month 13
    # to_date is lenient; must not raise.
    assert out is None or str(out).startswith("20")
