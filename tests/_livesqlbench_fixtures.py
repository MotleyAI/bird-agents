"""Synthetic LiveSQLBench dataset + gold-sidecar builders for the test suite.

Underscore-prefixed module name keeps pytest from collecting it as a test.
The helpers build a tmp on-disk layout that mirrors the real dataset:

```
<root>/
  livesqlbench_data_sqlite.jsonl   # public task jsonl
  <db>/
    <db>_template.sqlite           # real (tiny) sqlite, NOT an LFS pointer
    <db>_kb.jsonl                  # KB rows (one per line)
    <db>_column_meaning_base.json
    <db>_schema.txt
```

…and a sibling `<gold>.jsonl` with the gated `sol_sql`/`test_cases`/
`external_knowledge` rows keyed by `instance_id`. Tests pass `data_path`
+ `gold_file` (or `data_path_base`) into the same code paths the
production CLI uses, so there is no separate test-only ingest.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


def make_tiny_sqlite(path: Path, *, n_rows: int = 3) -> None:
    """Create a minimal real sqlite file: one table `widgets(id INTEGER,
    name TEXT, qty INTEGER)` seeded with `n_rows` deterministic rows.

    Used for both the per-DB template (`<db>_template.sqlite`) AND, after
    `prepare_livesqlbench.py`, the stable `<db>.sqlite` OTF ingests. Tests
    that need the latter materialise it explicitly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        con.execute(
            "CREATE TABLE widgets (id INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, qty INTEGER NOT NULL)"
        )
        con.executemany(
            "INSERT INTO widgets(id, name, qty) VALUES (?, ?, ?)",
            [(i, f"widget_{i}", i * 10) for i in range(1, n_rows + 1)],
        )
        con.commit()
    finally:
        con.close()


def make_lfs_pointer(path: Path, size: int = 2371584) -> None:
    """Write a git-LFS pointer file at `path` (the 132-byte form the real
    livesqlbench-base-lite-sqlite ships before `git lfs pull` runs).
    `prepare_livesqlbench.py` must refuse to ingest these.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:e160643c9abb43ba69917532f2654637e677460df25678f9daecdefb416046ed\n"
        f"size {size}\n"
    )


def make_lsb_dataset(
    root: Path,
    *,
    dbs: Sequence[str],
    tasks: Sequence[dict],
    template_as_lfs_pointer: bool = False,
    materialize_stable_sqlite: bool = False,
) -> Path:
    """Build a synthetic LiveSQLBench dataset under `root`.

    `tasks` is a list of public task dicts (NO sol_sql/test_cases/
    external_knowledge — those come from the gold sidecar). Each task
    MUST have `instance_id`, `selected_database`, `query`, `category`,
    `conditions` at minimum; everything else is defaulted.

    `template_as_lfs_pointer=True` writes LFS-pointer files for every
    `<db>_template.sqlite` instead of real sqlite — used to test the
    prepare script's refusal path. Mutually exclusive with
    `materialize_stable_sqlite=True` (which mirrors the post-prepare state
    where `<db>.sqlite` is already present).

    Returns `root / "livesqlbench_data_sqlite.jsonl"`.
    """
    if template_as_lfs_pointer and materialize_stable_sqlite:
        raise ValueError(
            "template_as_lfs_pointer and materialize_stable_sqlite are "
            "mutually exclusive in fixture builds",
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Per-DB files.
    for db in dbs:
        db_dir = root / db
        db_dir.mkdir(parents=True, exist_ok=True)
        template = db_dir / f"{db}_template.sqlite"
        if template_as_lfs_pointer:
            make_lfs_pointer(template)
        else:
            make_tiny_sqlite(template)
        if materialize_stable_sqlite and not template_as_lfs_pointer:
            make_tiny_sqlite(db_dir / f"{db}.sqlite")
        # KB / column-meaning / schema — small but real.
        (db_dir / f"{db}_kb.jsonl").write_text("")
        (db_dir / f"{db}_column_meaning_base.json").write_text(
            json.dumps({"widgets.id": "row id",
                        "widgets.name": "widget name",
                        "widgets.qty": "quantity"}, indent=2),
        )
        (db_dir / f"{db}_schema.txt").write_text(
            "CREATE TABLE widgets (id INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, qty INTEGER NOT NULL);\n",
        )

    # Task jsonl.
    data_file = root / "livesqlbench_data_sqlite.jsonl"
    with data_file.open("w") as f:
        for task in tasks:
            f.write(json.dumps(_default_public_task_fields(task)) + "\n")
    return data_file


def _default_public_task_fields(task: dict) -> dict:
    """Fill in the public-record fields the real dataset always carries
    so test inputs can be terse without skipping load-bearing keys."""
    out = {
        "preprocess_sql": [],
        "clean_up_sqls": [],
        "category": "Query",
        "high_level": True,
        "conditions": {"decimal": [], "distinct": False, "order": False},
        "difficulty_tier": "Moderate",
        # Public rows always have empty sol_sql / test_cases / external_knowledge.
        "sol_sql": [],
        "external_knowledge": [],
        "test_cases": [],
    }
    out.update(task)
    return out


def make_lsb_gold(
    path: Path,
    *,
    rows: Iterable[dict],
) -> Path:
    """Write a gated-gold sidecar at `path`. Each row MUST carry
    `instance_id`. `sol_sql` defaults to a list with one trivial SELECT,
    so the loader's empty-`sol_sql` fail-fast doesn't trip on every
    test by accident.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            entry = {
                "sol_sql": ["SELECT id FROM widgets"],
                "external_knowledge": [],
                "test_cases": [],
                **row,
            }
            f.write(json.dumps(entry) + "\n")
    return path


def public_task(
    instance_id: str,
    selected_database: str,
    *,
    query: str | None = None,
    category: str = "Query",
    conditions: dict | None = None,
) -> dict:
    """One synthetic public-task dict. `query` defaults to a sentinel so
    the loader's `query→amb_user_query` shim is exercised even without
    bespoke per-test wording.
    """
    return {
        "instance_id": instance_id,
        "selected_database": selected_database,
        "query": query or f"unambiguous request for {instance_id}",
        "category": category,
        "conditions": conditions or {
            "decimal": [], "distinct": False, "order": False,
        },
    }
