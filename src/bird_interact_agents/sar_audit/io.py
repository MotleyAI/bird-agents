"""JSONL read/append helpers for the SAR-audit driver."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = {
    "instance_id",
    "selected_database",
    "audit_status",
    "audit_model_requested",
    "skill_version",
}


def read_existing_rows(path: Path) -> list[dict]:
    """Return rows from `path` (empty list if missing). Invalid rows
    are silently dropped — the driver treats them as absent so the task
    is redone."""
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if not REQUIRED_KEYS.issubset(row.keys()):
                continue
            out.append(row)
    return out


def append_row(path: Path, row: dict) -> None:
    """Atomic-ish append: open in append mode, write one line, fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def append_failure(path: Path, failure: dict) -> None:
    """Same shape as `append_row`, but for the failures sidecar."""
    append_row(path, failure)


def index_by_instance_id(rows: list[dict]) -> dict[str, dict]:
    return {r["instance_id"]: r for r in rows}
