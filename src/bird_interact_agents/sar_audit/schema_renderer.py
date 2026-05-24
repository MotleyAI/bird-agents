"""Render a schema string for SAR-Agent's `schema` slot."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def render_schema(*, db_path: Path, column_meanings: dict) -> str:
    """Return DDL for every table in `db_path`, annotated with column
    meanings from `column_meanings` (full `<db>_column_meaning_base.json`).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        rows = list(
            con.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
    finally:
        con.close()

    out: list[str] = []
    for name, sql in rows:
        out.append(f"-- TABLE: {name}")
        if sql:
            out.append(sql.strip() + ";")
        annotations = _annotations_for_table(name, column_meanings)
        if annotations:
            out.append(f"-- column meanings for {name}:")
            for ann in annotations:
                out.append(f"--   {ann}")
        out.append("")
    return "\n".join(out)


def _annotations_for_table(table: str, column_meanings: dict) -> list[str]:
    prefix = f"{table}|"
    out: list[str] = []
    for key, val in column_meanings.items():
        if not key.startswith(prefix):
            continue
        if isinstance(val, str):
            out.append(f"{key} — {val}")
        elif isinstance(val, dict):
            top_meaning = val.get("column_meaning")
            if top_meaning:
                out.append(f"{key} — {top_meaning}")
            fields = val.get("fields_meaning") or {}
            for sub_key, sub_val in _flatten_fields(key, fields):
                out.append(f"{sub_key} — {sub_val}")
    return out


def _flatten_fields(prefix: str, blob) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(blob, dict):
        return out
    for k, v in blob.items():
        composite = f"{prefix}|{k}"
        if isinstance(v, str):
            out.append((composite, v))
        elif isinstance(v, dict):
            top = v.get("column_meaning")
            if isinstance(top, str) and top:
                out.append((composite, top))
            nested = v.get("fields_meaning")
            if isinstance(nested, dict):
                out.extend(_flatten_fields(composite, nested))
    return out
