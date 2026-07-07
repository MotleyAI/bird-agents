"""DEV-1648: NULL-safe Postgres cast emission for JSON-leaf columns.

Used by :mod:`.jsonb` to type a JSONB leaf whose extracted value is text:
the guarded CAST is baked into the leaf's derived ``Column.sql`` so the
leaf is typed (DOUBLE / INT / DATE / TIMESTAMP) with the cast applied
everywhere. (Top-level physically-TEXT columns are NOT cast on Postgres —
a self-referencing cast in a base column's sql cycles SLayer's column
expansion; they stay TEXT and carry the semantic type in their
description. See :func:`.overlay._apply_meaning_to_column`.)

Every emitted cast is NULL-safe for the primary failure mode — an
**un-castable value** (non-numeric / wrong-shape text) falls through to
NULL instead of aborting the whole query (Postgres has no ``TRY_CAST``).
The numeric regex gates the erroring ``::`` cast; temporal casts use the
lenient ``to_date``/``to_timestamp`` (never ``::timestamp``, which aborts
on an invalid calendar date such as ``2025-02-30``).

Known, accepted limitations (out of scope — the data does not exercise
them, and they are bounded by SLayer, not this helper):

* **Out-of-range numeric.** A shape-valid but out-of-range number
  (``1e309``, an integer > 2.1e9) can still abort — not on our inner cast
  but on the OUTER ``CAST(… AS DOUBLE PRECISION)`` / ``CAST(… AS INT)``
  that SLayer's compiler wraps around every typed column (``DataType.INT``
  compiles to int4). We do not control that outer cast; switching the
  inner cast to ``::numeric`` only moves the abort there.
* **Invalid calendar dates** roll over rather than NULL (``to_date`` is
  lenient). The bar is no-abort, not calendar validation.

Pure module: no DB, no I/O. Callers own identifier quoting — the
``inner_sql`` passed here is wrapped verbatim so a JSON-leaf extract
expression reuses the same helper.
"""

from __future__ import annotations

from typing import Optional

from slayer.core.models import DataType

# Numeric shape guards. INTEGER accepts any-length integer text because the
# cast target is ``::numeric`` (arbitrary precision) — it never overflows, so
# an unbounded guard loses no valid value (a bounded ``::bigint`` would NULL
# valid 19-digit bigints or abort on overflow).
NUMERIC_REGEX = r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$"
INTEGER_REGEX = r"^\s*[+-]?\d+\s*$"

# strptime %token -> Postgres to_char/to_timestamp template token.
# v1 supports NUMERIC-only tokens; %f is resolved per-column to US or MS
# (microseconds vs milliseconds) by the caller-supplied ``frac``. Any
# other %token (month name %b/%B, 12h %I, AM/PM %p, day-of-year %j, ...)
# is unsupported -> ``strptime_to_pg_format`` returns None and the caller
# leaves the column TEXT + warns.
_PG_FORMAT_TOKENS = {
    "%Y": "YYYY", "%y": "YY", "%m": "MM", "%d": "DD",
    "%H": "HH24", "%M": "MI", "%S": "SS",
}
_REGEX_TOKENS = {
    "%Y": r"\d{4}", "%y": r"\d{2}", "%m": r"\d{2}", "%d": r"\d{2}",
    "%H": r"\d{2}", "%M": r"\d{2}", "%S": r"\d{2}",
}
_REGEX_META = set(r".^$*+?()[]{}|\/")


def quote_ident(name: str) -> str:
    """Double-quote a Postgres identifier (embedded ``"`` doubled)."""
    return '"' + name.replace('"', '""') + '"'


def _sql_literal_body(text: str) -> str:
    """Escape a value destined for a single-quoted SQL string literal.

    Our regexes/format strings carry no single quotes; this is defensive.
    Backslashes are intentionally left literal — Postgres
    ``standard_conforming_strings=on`` (default) keeps ``\\d``/``\\s`` intact
    for the regex engine.
    """
    return text.replace("'", "''")


def strptime_to_pg_format(fmt: str, frac: str = "US") -> Optional[str]:
    """Translate a Python strptime format to a Postgres date/time template.

    ``frac`` is the Postgres token substituted for ``%f`` (``US`` = 6-digit
    microseconds, ``MS`` = 3-digit milliseconds). Returns ``None`` if *any*
    ``%token`` is unsupported. Literal ASCII-letter runs (e.g. the ``T`` in
    an ISO 8601 stamp) are double-quoted so Postgres treats them literally.
    """
    out: list[str] = []
    lit: list[str] = []

    def flush() -> None:
        if not lit:
            return
        run = "".join(lit)
        lit.clear()
        rendered = ""
        in_quote = False
        for ch in run:
            if ch.isalpha():
                if not in_quote:
                    rendered += '"'
                    in_quote = True
                rendered += ch
            else:
                if in_quote:
                    rendered += '"'
                    in_quote = False
                rendered += ch
        if in_quote:
            rendered += '"'
        out.append(rendered)

    i = 0
    while i < len(fmt):
        if fmt[i] == "%" and i + 1 < len(fmt):
            token = fmt[i : i + 2]
            if token == "%f":
                flush()
                out.append(frac)
                i += 2
                continue
            if token in _PG_FORMAT_TOKENS:
                flush()
                out.append(_PG_FORMAT_TOKENS[token])
                i += 2
                continue
            return None  # unsupported %token
        lit.append(fmt[i])
        i += 1
    flush()
    return "".join(out)


def strptime_to_regex(fmt: str) -> str:
    """Anchored shape-validation regex for *fmt*.

    A trailing ``.%f`` fractional group is rendered OPTIONAL so a column
    mixing ``…:22.330375`` and ``…:22`` both validate. Numeric tokens
    become fixed-width ``\\d{n}`` classes; literals are regex-escaped.
    """
    body: list[str] = []
    i = 0
    while i < len(fmt):
        if fmt[i : i + 3] == ".%f":
            body.append(r"(\.\d{1,6})?")
            i += 3
            continue
        if fmt[i] == "%" and i + 1 < len(fmt):
            token = fmt[i : i + 2]
            if token == "%f":
                body.append(r"\d{1,6}")
                i += 2
                continue
            if token in _REGEX_TOKENS:
                body.append(_REGEX_TOKENS[token])
                i += 2
                continue
            body.append(r"\S+")  # unsupported token (unreachable for real casts)
            i += 2
            continue
        ch = fmt[i]
        body.append("\\" + ch if ch in _REGEX_META else ch)
        i += 1
    return r"^\s*" + "".join(body) + r"\s*$"


def pg_nullsafe_cast(
    inner_sql: str,
    target: DataType,
    source_format: Optional[str] = None,
    frac_pg: str = "US",
) -> Optional[str]:
    """Return a NULL-safe Postgres cast of *inner_sql* to *target*.

    ``inner_sql`` is wrapped verbatim (caller owns identifier quoting), so a
    bare ``"col"`` or a ``jsonb_extract_path_text(...)`` extract both work.
    Returns ``None`` when no cast applies (non-castable target, or a
    temporal target with a missing/unsupported ``source_format``).
    """
    if target == DataType.DOUBLE:
        return (
            f"CASE WHEN ({inner_sql}) ~ '{_sql_literal_body(NUMERIC_REGEX)}' "
            f"THEN ({inner_sql})::double precision END"
        )
    if target == DataType.INT:
        # ::numeric (not ::bigint): arbitrary precision, so no overflow abort
        # and no valid-value loss regardless of magnitude.
        return (
            f"CASE WHEN ({inner_sql}) ~ '{_sql_literal_body(INTEGER_REGEX)}' "
            f"THEN ({inner_sql})::numeric END"
        )
    if target in (DataType.DATE, DataType.TIMESTAMP):
        if not source_format:
            return None
        pgfmt = strptime_to_pg_format(source_format, frac=frac_pg)
        if pgfmt is None:
            return None
        regex = _sql_literal_body(strptime_to_regex(source_format))
        pgfmt_lit = _sql_literal_body(pgfmt)
        if target == DataType.DATE:
            expr = f"to_date(({inner_sql}), '{pgfmt_lit}')"
        else:
            expr = f"(to_timestamp(({inner_sql}), '{pgfmt_lit}'))::timestamp"
        return f"CASE WHEN ({inner_sql}) ~ '{regex}' THEN {expr} END"
    return None
