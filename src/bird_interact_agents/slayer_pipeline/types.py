"""Parse the leading SQL-type token of a column-meaning description and
map it to SLayer's `DataType` enum.

Convention (from BIRD-Interact `<db>_column_meaning_base.json` leaf
strings and `fields_meaning` entries): a description that opens with
the source SQL type — `REAL.`, `INTEGER.`, `CHAR(10).`,
`DECIMAL(12,3).`, `DOUBLE PRECISION.`, `JSONB.`, or a custom enum like
`RefundMethod_enum.` — signals the column's type. The regex matches
the *first* such token; everything after the first period is the human
description.

Returns:
- `(DataType, meta_patch)` on a match. `meta_patch` carries auxiliary
  info to stamp into `Column.meta` (e.g. `{"enum_name": "X_enum"}` or
  `{"jsonb": True}` for the leaf-expansion pass to find).
- `None` when no recognised leading token is present — caller should
  default to `DataType.TEXT` and log a warning.
"""

from __future__ import annotations

import re
from typing import Optional

from slayer.core.models import DataType

LEADING_TYPE_RE = re.compile(
    r"^\s*(?:A\s+)?([A-Za-z][A-Za-z_]*(?:\s+PRECISION)?)(?:\([\d,\s]+\))?(?=[.\s])",
)

_NUMERIC_TOKENS = {
    "REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT",
    "DECIMAL", "NUMERIC", "MONEY",
}
_INTEGER_TOKENS = {"INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT", "SERIAL"}
_TEXT_TOKENS = {"TEXT", "CHAR", "VARCHAR", "NCHAR", "NVARCHAR", "STRING"}
_BOOL_TOKENS = {"BOOLEAN", "BOOL"}
_DATE_TOKENS = {"DATE"}
_TIMESTAMP_TOKENS = {"TIMESTAMP", "DATETIME"}


def parse_leading_type_token(
    description: Optional[str],
) -> Optional[tuple[DataType, dict]]:
    """Return ``(DataType, meta_patch)`` if the description starts with a
    recognised SQL-type token; otherwise ``None``."""
    if not description:
        return None
    match = LEADING_TYPE_RE.match(description)
    if not match:
        return None
    captured = match.group(1).strip()
    # Case discipline: only accept all-uppercase tokens (REAL, JSONB,
    # DATE, ...) or explicit `_enum` suffixes. Rejects prose-leading
    # capitalised words like "Date stored as TEXT in '%Y-%m-%d'..." that
    # otherwise spuriously infer DATE.
    if not captured.isupper() and not captured.endswith("_enum"):
        return None
    raw = captured.upper()

    if raw.endswith("_ENUM"):
        return DataType.TEXT, {"enum_name": captured}
    if raw == "JSONB":
        return DataType.TEXT, {"jsonb": True}
    if raw in _NUMERIC_TOKENS:
        return DataType.DOUBLE, {}
    if raw in _INTEGER_TOKENS:
        return DataType.INT, {}
    if raw in _TEXT_TOKENS:
        return DataType.TEXT, {}
    if raw in _BOOL_TOKENS:
        return DataType.BOOLEAN, {}
    if raw in _DATE_TOKENS:
        return DataType.DATE, {}
    if raw in _TIMESTAMP_TOKENS:
        return DataType.TIMESTAMP, {}
    return None
