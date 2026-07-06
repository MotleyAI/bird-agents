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
from typing import Callable, Optional

from slayer.core.models import Column, DataType, SlayerModel

from .casts import pg_nullsafe_cast, quote_ident
from .pg_sampling import detect_date_format, detect_fraction_pg_token
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


def _resolve_join_side(
    side: str, default_table: str, name_to_table: dict[str, str]
) -> tuple[str, str]:
    """Resolve one side of a ``join_pairs`` entry to ``(table, col)`` lower.

    A dotted ``"model.col"`` names its own model (resolved to that model's
    physical table via *name_to_table*); a bare ``"col"`` lives on
    *default_table* (already the source/target model's physical table). The
    physical table is what ``apply_overlay`` keys the guard on, so the join's
    ``target_model`` (a model NAME) must be resolved to its ``sql_table``
    before this — otherwise a model whose name differs from its table would
    escape the guard.
    """
    if "." in side:
        model_or_table, col = side.split(".", 1)
        key = model_or_table.lower()
        return name_to_table.get(key, key), col.lower()
    return default_table.lower(), side.lower()


def collect_key_columns(models: list[SlayerModel]) -> frozenset[tuple[str, str]]:
    """DB-wide set of ``(table_lower, col_lower)`` for every PK and every
    join-key participant (both local and referenced sides).

    TABLE-SCOPED: a common name like ``id`` guards only the table where it
    is actually a key. Guarded columns keep their description but are never
    retyped / rewritten, so they stay base/TEXT and match live
    introspection (no drift, no lossy cast on an identifier).
    """
    # A join's ``target_model`` (and a dotted side's prefix) is a model NAME;
    # the guard is keyed on the physical table. Map name -> table so an alias
    # that differs from its sql_table still guards the right table.
    name_to_table = {
        model.name.lower(): (model.sql_table or model.name).lower()
        for model in models
    }
    keys: set[tuple[str, str]] = set()
    for model in models:
        table = (model.sql_table or model.name).lower()
        for column in model.columns:
            if column.primary_key:
                keys.add((table, column.name.lower()))
        for join in model.joins or []:
            target_table = name_to_table.get(
                (join.target_model or "").lower(), (join.target_model or "").lower()
            )
            for pair in join.join_pairs or []:
                if len(pair) < 2:
                    continue
                keys.add(_resolve_join_side(pair[0], table, name_to_table))
                keys.add(_resolve_join_side(pair[1], target_table, name_to_table))
    return frozenset(keys)


def _apply_meaning_to_column(
    col: Column,
    meaning_text: str,
    *,
    backend: str = "sqlite",
    is_key: bool = False,
    pg_sampler: Optional[Callable[[str, str], list[str]]] = None,
    table: str = "",
) -> Optional[str]:
    """Mutate *col* in place: description, type from leading token, DEV-1381
    date annotation. Returns a warning string when a postgres refinement is
    declined (column left TEXT), else ``None``.

    On postgres, a refined column becomes *derived* — ``col.sql`` is a
    NULL-safe guarded CAST — but ONLY when the physical (pre-overlay,
    ingested) type is TEXT; an already-typed live column is left bare.
    Guarded PK/join-key columns keep their description and nothing else.
    """
    if meaning_text and not col.description:
        col.description = meaning_text

    if is_key:
        return None  # guarded: description only, never retyped/rewritten

    physical_type = col.type  # pre-overlay == the live physical type

    annot = DATE_ANNOTATION_RE.search(meaning_text or "")
    if annot is not None:
        fmt = annot.group(1)
        if backend == "postgres":
            if physical_type != DataType.TEXT:
                return None  # already typed live -> no cast, no drift
            cast_sql = pg_nullsafe_cast(quote_ident(col.name), DataType.TIMESTAMP, fmt)
            if cast_sql is None:
                return (
                    f"{table}.{col.name}: DEV-1381 annotation source_format "
                    f"'{fmt}' unsupported on postgres; column left TEXT."
                )
            col.type = DataType.TIMESTAMP
            col.sql = cast_sql
            meta = col.meta or {}
            meta["date_source_format"] = fmt
            meta["detected_by"] = "column_meaning_annotation"
            col.meta = meta
            return None
        # SQLite (unchanged).
        new_sql = _sqlite_reformat_sql(col.name, fmt)
        if new_sql is None and fmt not in ISO_TEXT_DATE_FORMATS:
            return None
        col.type = DataType.TIMESTAMP
        if new_sql is not None:
            col.sql = new_sql
        meta = col.meta or {}
        meta["date_source_format"] = fmt
        meta["detected_by"] = "column_meaning_annotation"
        col.meta = meta
        return None

    parsed = parse_leading_type_token(meaning_text)
    if parsed is None:
        return None
    data_type, meta_patch = parsed

    def _stamp_meta() -> None:
        if meta_patch:
            meta = col.meta or {}
            meta.update(meta_patch)
            col.meta = meta

    if backend == "postgres":
        if physical_type != DataType.TEXT:
            return None  # already typed live -> leave bare, no drift
        if data_type in (DataType.DOUBLE, DataType.INT):
            col.sql = pg_nullsafe_cast(quote_ident(col.name), data_type)
            col.type = data_type
            _stamp_meta()
            return None
        if data_type in (DataType.DATE, DataType.TIMESTAMP):
            samples = pg_sampler(table, col.name) if pg_sampler else []
            fmt = detect_date_format(
                samples, with_time=(data_type == DataType.TIMESTAMP)
            )
            if not fmt:
                return (
                    f"{table}.{col.name}: {data_type.name} token but no date "
                    f"format detected from {len(samples)} sampled value(s); "
                    f"column left TEXT."
                )
            frac = detect_fraction_pg_token(samples)
            cast_sql = pg_nullsafe_cast(
                quote_ident(col.name), data_type, fmt, frac_pg=frac
            )
            if cast_sql is None:
                return (
                    f"{table}.{col.name}: detected format '{fmt}' unsupported "
                    f"on postgres; column left TEXT."
                )
            col.type = data_type
            col.sql = cast_sql
            meta = col.meta or {}
            meta["date_source_format"] = fmt
            meta["detected_by"] = "pg_sample"
            col.meta = meta
            return None
        if data_type == DataType.BOOLEAN:
            # A BOOLEAN token over a physically-TEXT column would drift the
            # same way (persisted BOOLEAN vs live TEXT) and no NULL-safe
            # boolean cast is in scope (value spellings vary). Leave it TEXT
            # to match live introspection.
            return (
                f"{table}.{col.name}: BOOLEAN token over a physically-TEXT "
                f"column is not cast on postgres; column left TEXT."
            )
        # TEXT / enum / jsonb tokens (all map to DataType.TEXT): meta only.
        col.type = data_type
        _stamp_meta()
        return None

    # SQLite (unchanged): set type from token, sql stays bare.
    col.type = data_type
    _stamp_meta()
    return None


def apply_overlay(
    model: SlayerModel,
    by_table: dict[str, dict[str, dict | str]],
    *,
    backend: str = "sqlite",
    key_columns: frozenset[tuple[str, str]] = frozenset(),
    pg_sampler: Optional[Callable[[str, str], list[str]]] = None,
) -> tuple[int, list[str]]:
    """Apply meanings to *model* in place.

    Returns ``(num_columns_touched, warnings)``. On ``backend='postgres'``,
    refined-over-TEXT non-key columns are rewritten to NULL-safe guarded
    casts (DEV-1648); ``key_columns`` (DB-wide PK/join keys) are guarded;
    the SQLite path is byte-unchanged.
    """
    table_actual = model.sql_table or model.name
    table_lc = table_actual.lower()
    col_to_meaning = by_table.get(table_lc, {})
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
        is_key = (table_lc, column.name.lower()) in key_columns
        warn = _apply_meaning_to_column(
            column, text, backend=backend, is_key=is_key,
            pg_sampler=pg_sampler, table=table_actual,
        )
        if warn:
            warnings.append(warn)
        if (
            column.description != before_description
            or column.type != before_type
            or column.sql != before_sql
            or column.meta != before_meta
        ):
            touched += 1
        # SQLite-only legacy warning for an unsupported DEV-1381 format.
        if backend != "postgres":
            annot = DATE_ANNOTATION_RE.search(text)
            if (
                annot is not None
                and _sqlite_reformat_sql(column.name, annot.group(1)) is None
                and annot.group(1) not in ISO_TEXT_DATE_FORMATS
            ):
                warnings.append(
                    f"{table_lc}.{column.name}: DEV-1381 annotation has "
                    f"unsupported source_format '{annot.group(1)}'; column "
                    f"left as TEXT."
                )
    return touched, warnings
