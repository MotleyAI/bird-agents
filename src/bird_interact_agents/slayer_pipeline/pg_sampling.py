"""DEV-1648: deterministic date-format detection from sampled PG values.

Phase 2 refines ``DATE.``/``TIMESTAMP.``-token columns that carry NO
explicit strftime format. We discover the format by sampling live
Postgres values and picking the first ordered candidate that round-trips
ALL samples (no LLM). The fractional-second width (3 vs 6 digits) selects
the Postgres ``MS``/``US`` token so ``to_timestamp`` scales correctly.

Sampling is a thin wrapper over SQLAlchemy; the query builders are pure
and unit-tested, and any DB error degrades to ``[]`` (the caller then
leaves the column TEXT + warns — never silently NULLs a column).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional

from .casts import quote_ident as _quote_ident

SAMPLE_LIMIT = 20

# Ordered candidate BASE formats (no fractional seconds). ISO / dash-YMD
# first, then dotted, slashed, and day/month-first. Order resolves genuine
# ambiguity (e.g. 05/06/2025) deterministically — day-first preferred.
_DATE_BASES = [
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
]
_DATETIME_BASES = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y.%m.%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]

_FRAC_RE = re.compile(r"\.(\d{1,6})\s*$")


def _strip_fraction(value: str) -> str:
    return _FRAC_RE.sub("", value)


def _all_match(samples: list[str], base_fmt: str) -> bool:
    """True iff every sample parses under *base_fmt* (a trailing ``.<digits>``
    fraction is stripped first, so the base format matches both fractional
    and non-fractional rows)."""
    for value in samples:
        candidate = value
        try:
            datetime.strptime(candidate, base_fmt)
            continue
        except ValueError:
            pass
        stripped = _strip_fraction(candidate)
        if stripped == candidate:
            return False
        try:
            datetime.strptime(stripped, base_fmt)
        except ValueError:
            return False
    return True


def detect_date_format(samples: list[str], *, with_time: bool) -> Optional[str]:
    """First ordered candidate strptime format that round-trips ALL samples.

    Appends ``.%f`` iff at least one sample carries a fractional-seconds
    tail (the regex renders it optional, so a mixed column still matches).
    ``with_time`` restricts candidates to date-only vs datetime by the
    column's ``DATE.``/``TIMESTAMP.`` token. Returns ``None`` if nothing
    round-trips every sample.
    """
    cleaned = [s.strip() for s in samples if s and s.strip()]
    if not cleaned:
        return None
    has_fraction = any(_FRAC_RE.search(s) for s in cleaned)
    bases = _DATETIME_BASES if with_time else _DATE_BASES
    for base in bases:
        if _all_match(cleaned, base):
            if with_time and has_fraction and base.endswith("%S"):
                return base + ".%f"
            return base
    return None


def detect_fraction_pg_token(samples: list[str]) -> str:
    """Return the Postgres fractional token for the sampled values.

    ``MS`` iff every fractional sample is exactly 3 digits, else ``US``
    (6-digit, mixed-width, or no fraction — the safe default).
    """
    widths = {
        len(m.group(1))
        for s in samples
        if s and (m := _FRAC_RE.search(s.strip()))
    }
    if widths == {3}:
        return "MS"
    return "US"


# ---------------------------------------------------------------------------
# Live sampling (thin SQLAlchemy wrapper; query builders are pure)
# ---------------------------------------------------------------------------


def _sample_query(table: str, col: str, *, limit: int = SAMPLE_LIMIT) -> str:
    q = _quote_ident(col)
    return (
        f"SELECT DISTINCT {q} FROM {_quote_ident(table)} "
        f"WHERE {q} IS NOT NULL AND {q} <> '' "
        f"ORDER BY {q} LIMIT {limit}"
    )


def _extract_sample_query(table: str, extract_sql: str, *, limit: int = SAMPLE_LIMIT) -> str:
    # ORDER BY the extracted value (mirrors _sample_query) so repeated calls
    # on the same DB state return the same sample set -> deterministic format
    # detection / re-encode.
    return (
        f"SELECT DISTINCT {extract_sql} FROM {_quote_ident(table)} "
        f"WHERE {extract_sql} IS NOT NULL AND {extract_sql} <> '' "
        f"ORDER BY {extract_sql} LIMIT {limit}"
    )


def _make_engine(db_url: str):
    import sqlalchemy as sa

    return sa.create_engine(db_url)


def _run_query(db_url: str, query: str) -> list[str]:
    try:
        import sqlalchemy as sa

        engine = _make_engine(db_url)
        with engine.connect() as conn:
            rows = conn.execute(sa.text(query))
            return [str(r[0]) for r in rows if r[0] is not None]
    except Exception:  # noqa: BLE001 — best-effort; caller degrades to TEXT + warn
        return []


def make_pg_sampler(db_url: str) -> Callable[[str, str], list[str]]:
    """Return ``sampler(table, col) -> list[str]`` over the live PG DB."""

    def sampler(table: str, col: str) -> list[str]:
        return _run_query(db_url, _sample_query(table, col))

    return sampler


def make_pg_extract_sampler(db_url: str) -> Callable[[str, str], list[str]]:
    """Return ``sampler(table, extract_sql) -> list[str]`` for JSON leaves."""

    def sampler(table: str, extract_sql: str) -> list[str]:
        return _run_query(db_url, _extract_sample_query(table, extract_sql))

    return sampler
