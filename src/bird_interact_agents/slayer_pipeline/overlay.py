"""Apply `<db>_column_meaning_base.json` to a live SLayer model.

For every existing schema-ingested top-level column on a model, set
``description`` from the matching ``column_meaning`` string, then set
``Column.type`` via the leading-type-token parser, then handle the
DEV-1381 date-format annotation (``"Date stored as TEXT in
'<strftime>'. Cast at encode time to TIMESTAMP."``) by retyping to
``TIMESTAMP`` and rewriting ``Column.sql`` to a SQLite-native parse
expression when the source format is non-ISO.

Phase 3 (JSONB-leaf expansion) appends new derived columns *after*
this pass runs; phase 4 (LLM date detection) acts on any column whose
type is still ``TEXT`` after phase 2.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Optional

from slayer.core.models import Column, DataType, SlayerModel

from .types import parse_leading_type_token

# DEV-1381 annotation grammar:
#   "Date stored as TEXT in '<strftime>'. Cast at encode time to TIMESTAMP."
DATE_ANNOTATION_RE = re.compile(
    r"Date(?:time)?\s+stored\s+as\s+TEXT\s+in\s+'([^']+)'\.\s*"
    r"Cast\s+at\s+encode\s+time\s+to\s+TIMESTAMP",
    re.IGNORECASE,
)

# Formats that SQLite's date functions parse natively — Column.sql can
# stay as the bare column name (passthrough) and `Column.type=TIMESTAMP`
# is safe. Anything outside this set requires a reformat from
# ``_sqlite_reformat_sql`` to be usable; if neither matches, leave the
# column as TEXT.
ISO_TEXT_DATE_FORMATS = frozenset({
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
})


def load_meanings(meanings_path: Path) -> dict[str, dict[str, dict | str]]:
    """Return ``{table_lower: {col_lower: meaning_entry}}``.

    *meaning_entry* is either a plain string (non-JSONB columns; the
    value is the column description) or a dict with at least
    ``column_meaning`` (string) and optionally ``fields_meaning`` (dict)
    for JSONB columns.
    """
    raw = json.loads(meanings_path.read_text(encoding="utf-8"))
    by_table: dict[str, dict[str, dict | str]] = {}
    for key, value in raw.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, col = parts
        by_table.setdefault(table.lower(), {})[col.lower()] = value
    return by_table


def _column_meaning_text(entry: dict | str) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        text = entry.get("column_meaning")
        if isinstance(text, str):
            return text
    return None


def _sqlite_reformat_sql(col_name: str, fmt: str) -> Optional[str]:
    """Return SQLite SQL that reformats a `fmt`-string TEXT column into
    an ISO ``YYYY-MM-DD[ HH:MM:SS]`` form parseable by SQLite's date
    functions and by SLayer's TIMESTAMP-typed column path.

    Returns ``None`` for two distinct cases — callers must disambiguate
    via ``ISO_TEXT_DATE_FORMATS``:

    - ISO-already format: ``Column.sql`` stays as the bare column name
      (passthrough). Caller still promotes ``Column.type=TIMESTAMP``.
    - Unsupported format: caller must leave the column as ``TEXT`` and
      log a warning; promoting the type without a rewrite produces a
      column SLayer can't actually parse.
    """
    if fmt in ISO_TEXT_DATE_FORMATS:
        return None

    # Simple substring-based reformatters for the common non-ISO patterns
    # observed in BIRD-Interact gold SQL.
    if fmt == "%d/%m/%Y":
        return (
            f"SUBSTR({col_name}, 7, 4) || '-' || "
            f"SUBSTR({col_name}, 4, 2) || '-' || "
            f"SUBSTR({col_name}, 1, 2)"
        )
    if fmt == "%m/%d/%Y":
        return (
            f"SUBSTR({col_name}, 7, 4) || '-' || "
            f"SUBSTR({col_name}, 1, 2) || '-' || "
            f"SUBSTR({col_name}, 4, 2)"
        )
    if fmt == "%Y/%m/%d":
        return f"REPLACE({col_name}, '/', '-')"
    if fmt == "%d-%m-%Y":
        return (
            f"SUBSTR({col_name}, 7, 4) || '-' || "
            f"SUBSTR({col_name}, 4, 2) || '-' || "
            f"SUBSTR({col_name}, 1, 2)"
        )
    # Unsupported format: caller will record a warning and leave the
    # column TEXT.
    return None


def _apply_meaning_to_column(col: Column, meaning_text: str) -> None:
    """Mutate *col* in place: description, type from leading token,
    DEV-1381 date-format annotation."""
    if meaning_text and not col.description:
        col.description = meaning_text

    annot = DATE_ANNOTATION_RE.search(meaning_text or "")
    if annot is not None:
        fmt = annot.group(1)
        new_sql = _sqlite_reformat_sql(col.name, fmt)
        if new_sql is None and fmt not in ISO_TEXT_DATE_FORMATS:
            # No SQLite rewrite path and not an ISO passthrough — leave
            # the column TEXT so phase 4 can still try, and surface the
            # unsupported format via the apply_overlay warning path.
            return
        col.type = DataType.TIMESTAMP
        if new_sql is not None:
            col.sql = new_sql
        # else: ISO-already, sql stays as the bare column.
        meta = col.meta or {}
        meta["date_source_format"] = fmt
        meta["detected_by"] = "column_meaning_annotation"
        col.meta = meta
        return

    parsed = parse_leading_type_token(meaning_text)
    if parsed is None:
        return
    data_type, meta_patch = parsed
    col.type = data_type
    if meta_patch:
        meta = col.meta or {}
        meta.update(meta_patch)
        col.meta = meta


def apply_overlay(
    model: SlayerModel, by_table: dict[str, dict[str, dict | str]]
) -> tuple[int, list[str]]:
    """Apply meanings to *model* in place.

    Returns ``(num_columns_touched, warnings)``. Warnings list any
    described date columns whose source format we don't know how to
    reformat (still set to TIMESTAMP, but Column.sql stays raw).
    """
    table = (model.sql_table or model.name).lower()
    col_to_meaning = by_table.get(table, {})
    if not col_to_meaning:
        return 0, []
    touched = 0
    warnings: list[str] = []
    for column in model.columns:
        meaning_entry = col_to_meaning.get(column.name.lower())
        if meaning_entry is None:
            continue
        text = _column_meaning_text(meaning_entry)
        if text is None:
            continue
        before_description = column.description
        before_type = column.type
        before_sql = column.sql
        before_meta = copy.deepcopy(column.meta)
        _apply_meaning_to_column(column, text)
        if (
            column.description != before_description
            or column.type != before_type
            or column.sql != before_sql
            or column.meta != before_meta
        ):
            touched += 1
        annot = DATE_ANNOTATION_RE.search(text)
        if (
            annot is not None
            and _sqlite_reformat_sql(column.name, annot.group(1)) is None
            and annot.group(1) not in ISO_TEXT_DATE_FORMATS
        ):
            warnings.append(
                f"{table}.{column.name}: DEV-1381 annotation has "
                f"unsupported source_format '{annot.group(1)}'; column "
                f"left as TEXT."
            )
    return touched, warnings
