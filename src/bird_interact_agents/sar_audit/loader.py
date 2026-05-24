"""Disk loaders for mini-interact artefacts used by the SAR-audit driver."""

from __future__ import annotations

import json
from pathlib import Path


def load_task_list(*, db: str, mini_interact_path: Path) -> list[dict]:
    """Return all mini-interact tasks for the given database.

    Filters `mini_interact.jsonl` by `selected_database == db`.
    """
    out: list[dict] = []
    with mini_interact_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("selected_database") == db:
                out.append(row)
    return out


def load_kb(*, db: str, mini_interact_root: Path) -> list[dict]:
    """Return the full `<db>_kb.jsonl` knowledge base."""
    path = mini_interact_root / db / f"{db}_kb.jsonl"
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_column_meanings(*, db: str, mini_interact_root: Path) -> dict:
    """Return the full `<db>_column_meaning_base.json` blob."""
    path = mini_interact_root / db / f"{db}_column_meaning_base.json"
    return json.loads(path.read_text())


def locate_db_sqlite(*, db: str, mini_interact_root: Path) -> Path:
    """Return the path to `<db>.sqlite` (does not check existence — caller's job)."""
    return mini_interact_root / db / f"{db}.sqlite"
