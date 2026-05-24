"""JSONB-leaf auto-expansion (DEV-1379).

Walks `<db>_column_meaning_base.json[<db>|<table>|<col>]["fields_meaning"]`
recursively and emits one SLayer `Column` per terminal leaf. Re-runs
are idempotent via `meta.derived_from`.

Drift detection (warnings only): when given a SQLite connection, compares
each JSONB column's actual key set (sampled rows) to the documented
`fields_meaning` keys and reports both ways. We do **not** auto-generate
columns for undocumented keys in this pass — that's deferred.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel

from slayer.core.models import Column, DataType

from .types import parse_leading_type_token

DRIFT_SAMPLE_ROWS = 100


class LeafInfo(BaseModel):
    """One emitted leaf: a column-shaped dict plus a typing-warning flag."""

    column_name: str
    path: list[str]
    description: str
    sql: str
    type: DataType
    enum_name: Optional[str] = None
    label: str
    no_type_token: bool = False


def walk_fields_meaning(
    json_col: str, fields_meaning: dict
) -> Iterable[LeafInfo]:
    """Yield one ``LeafInfo`` per terminal-string entry in *fields_meaning*."""
    yield from _walk(json_col, [], fields_meaning)


def _json_path(path: list[str]) -> str:
    """Build a JSON-path expression for `JSON_EXTRACT`.

    Numeric segments become array indexes (`[N]`); everything else is
    dotted. `["foo", "3"]` → `$.foo[3]` (SQLite's array indexing),
    not `$.foo.3` (which targets the object key "3" instead).
    """
    out = "$"
    for segment in path:
        out += f"[{segment}]" if segment.isdigit() else f".{segment}"
    return out


def _walk(
    json_col: str, path: list[str], node: object
) -> Iterable[LeafInfo]:
    if isinstance(node, str):
        leaf_key = path[-1]
        full_path = [*path]
        column_name = json_col + "__" + "__".join(full_path)
        json_path = _json_path(full_path)
        parsed = parse_leading_type_token(node)
        if parsed is None:
            data_type = DataType.TEXT
            enum_name = None
            no_token = True
        else:
            data_type, meta_patch = parsed
            enum_name = meta_patch.get("enum_name")
            no_token = False
        yield LeafInfo(
            column_name=column_name,
            path=full_path,
            description=node,
            sql=f"JSON_EXTRACT({json_col}, '{json_path}')",
            type=data_type,
            enum_name=enum_name,
            label=_humanise(leaf_key),
            no_type_token=no_token,
        )
        return
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _walk(json_col, [*path, key], child)


def _humanise(key: str) -> str:
    """`Bath_Count` -> `Bath Count`; `restocking_fee` -> `Restocking Fee`."""
    return re.sub(r"[_\s]+", " ", key).strip().title()


def leaf_to_column(leaf: LeafInfo, json_col: str) -> Column:
    """Build a Column from a LeafInfo, stamping ``meta.derived_from``."""
    meta: dict = {"derived_from": {"json_col": json_col, "path": list(leaf.path)}}
    if leaf.enum_name is not None:
        meta["enum_name"] = leaf.enum_name
    return Column(
        name=leaf.column_name,
        sql=leaf.sql,
        type=leaf.type,
        primary_key=False,
        description=leaf.description,
        label=leaf.label,
        hidden=False,
        meta=meta,
    )


def expand_one_column(
    json_col: str, meaning_entry: dict
) -> tuple[list[Column], list[str]]:
    """Expand one ``fields_meaning`` dict into Columns + typing-warning lines.

    Returns ``(columns, warnings)``. *warnings* lists leaves whose
    description had no recognised leading type token (defaulted to
    TEXT) — caller appends them to ``column_typing_warnings.txt``.
    """
    fields = meaning_entry.get("fields_meaning") or {}
    columns: list[Column] = []
    warnings: list[str] = []
    for leaf in walk_fields_meaning(json_col, fields):
        columns.append(leaf_to_column(leaf, json_col))
        if leaf.no_type_token:
            warnings.append(
                f"{json_col} :: {'.'.join(leaf.path)}: no leading type "
                f"token in fields_meaning; defaulted to TEXT"
            )
    return columns, warnings


def jsonb_meaning_entries(meanings: dict) -> Iterable[tuple[str, str, dict]]:
    """Yield ``(table, json_col, entry)`` for every JSONB column in the
    BIRD `<db>_column_meaning_base.json` blob.

    A "JSONB column" is detected as a dict-valued entry with a
    ``fields_meaning`` sub-key. Some entries are plain strings — those
    aren't JSONB and are skipped.
    """
    for key, value in meanings.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, col = parts
        if isinstance(value, dict) and isinstance(
            value.get("fields_meaning"), dict
        ):
            yield table.lower(), col.lower(), value


def detect_drift(
    sqlite_path: Path, table: str, json_col: str, documented_keys: set[str]
) -> tuple[set[str], set[str]]:
    """Sample rows from *table* and compare top-level JSON keys to the
    documented set.

    Returns ``(undocumented_in_data, ghost_in_meanings)``:
    - *undocumented_in_data*: actual keys present in JSON but absent
      from ``fields_meaning``.
    - *ghost_in_meanings*: documented keys absent from every sampled
      row.

    Only top-level keys are inspected. Deep-key drift is out of scope
    for the first pass.
    """
    if not sqlite_path.exists():
        return set(), set()
    actual: set[str] = set()
    rows_seen = 0
    # immutable=1 lets the open succeed without writing a journal — the
    # mini-interact data dir isn't a sandbox-writable path.
    conn = sqlite3.connect(f"file:{sqlite_path}?immutable=1", uri=True)
    try:
        cur = conn.execute(
            f'SELECT "{json_col}" FROM "{table}" '
            f'WHERE "{json_col}" IS NOT NULL LIMIT {DRIFT_SAMPLE_ROWS}'
        )
        for (raw,) in cur:
            rows_seen += 1
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                actual.update(parsed.keys())
    except sqlite3.Error:
        return set(), set()
    finally:
        conn.close()
    if rows_seen == 0:
        return set(), set()
    undocumented = actual - documented_keys
    ghost = documented_keys - actual
    return undocumented, ghost
