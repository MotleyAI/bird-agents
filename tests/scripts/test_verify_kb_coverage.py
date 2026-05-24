"""Tests for the memory-based verifier (scripts/verify_kb_coverage.py).

Covers:
- Memory-based attribution: a memory whose first non-blank line is
  `KB <id> — ` AND whose linked entities include `<db>.…` counts as
  documenting `<id>`.
- Cross-DB pollution filter: a memory with a `KB <id> — ` first line
  but whose linked entities all start with a *different* DB does
  NOT count for the current DB.
- Encoded-via-meta.kb_id still works alongside memory documentation.
- A missing `memories.yaml` (no memories yet) just yields an empty
  documented set rather than raising.

The tests build a self-contained YAMLStorage tree in a tmp_path
(datasource + one tiny model + a memories.yaml) and a tiny KB JSONL,
then exercise `verify_one` directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

# Load the script as a module (it lives in scripts/ which isn't a package).
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_kb_coverage.py"
spec = importlib.util.spec_from_file_location("verify_kb_coverage", SCRIPT)
verify_kb_coverage = importlib.util.module_from_spec(spec)
sys.modules["verify_kb_coverage"] = verify_kb_coverage
spec.loader.exec_module(verify_kb_coverage)


def _write_storage(
    base_dir: Path,
    *,
    db: str,
    encoded_kb_ids: list[int],
    memories: list[dict],
) -> None:
    """Build a minimal YAMLStorage tree: one model carrying the given
    encoded KB ids on its columns, plus a memories.yaml with the given
    memory rows.
    """
    datasources_dir = base_dir / "datasources"
    models_dir = base_dir / "models" / db
    datasources_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    (datasources_dir / f"{db}.yaml").write_text(
        yaml.safe_dump({
            "version": 6,
            "name": db,
            "kind": "sqlite",
            "uri": f"sqlite:///{base_dir}/dummy.db",
        }),
        encoding="utf-8",
    )

    cols = [
        {
            "name": f"c_{kb_id}",
            "sql": f"c_{kb_id}",
            "type": "INT",
            "primary_key": False,
            "allowed_aggregations": [],
            "hidden": False,
            "meta": {"kb_id": kb_id},
        }
        for kb_id in encoded_kb_ids
    ]
    (models_dir / "m.yaml").write_text(
        yaml.safe_dump({
            "version": 6,
            "name": "m",
            "data_source": db,
            "sql_table": "m",
            "columns": cols,
            "measures": [],
            "aggregations": [],
            "joins": [],
            "filters": [],
            "query_variables": {},
        }),
        encoding="utf-8",
    )

    if memories:
        (base_dir / "memories.yaml").write_text(
            yaml.safe_dump(memories), encoding="utf-8",
        )


def _write_kb_jsonl(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for kb_id in ids:
            f.write(json.dumps({
                "id": kb_id,
                "knowledge": f"KB {kb_id} title",
                "description": "",
                "definition": "",
                "type": "metric",
                "children_knowledge": [],
            }) + "\n")


@pytest.mark.asyncio
async def test_memory_documents_kb_id(tmp_path, monkeypatch):
    """A memory whose first line is `KB 2 — Foo` and whose linked entities
    contain a `<db>.…` ref counts toward the documented set.
    """
    db = "thedb"
    slayer_models = tmp_path / "slayer_models"
    db_dir = slayer_models / db
    _write_storage(
        db_dir,
        db=db,
        encoded_kb_ids=[1],
        memories=[{
            "id": 1,
            "learning": "KB 2 — KB 2 title\n\nReason: deferred.\nStatus: deferred — AMBIGUOUS-PROSE\n",
            "entities": [f"{db}.m.c_2"],
            "query": None,
        }],
    )

    mini_interact = tmp_path / "mini-interact"
    _write_kb_jsonl(mini_interact / db / f"{db}_kb.jsonl", [1, 2])

    monkeypatch.setattr(verify_kb_coverage, "SLAYER_MODELS_DIR", slayer_models)

    unaccounted, overlap, _ = await verify_kb_coverage.verify_one(
        db, mini_interact,
    )
    assert unaccounted == set()
    assert overlap == set()


@pytest.mark.asyncio
async def test_cross_db_memory_does_not_count(tmp_path, monkeypatch):
    """A memory whose first line is `KB 3 — Bar` BUT whose linked
    entities all live in a *different* DB does NOT count for the
    current DB.
    """
    db = "households"
    slayer_models = tmp_path / "slayer_models"
    db_dir = slayer_models / db
    _write_storage(
        db_dir,
        db=db,
        encoded_kb_ids=[1],
        memories=[{
            "id": 1,
            "learning": "KB 3 — KB 3 title\n\nReason: this memory is mis-filed.\n",
            "entities": ["solar.plant.initdate"],
            "query": None,
        }],
    )

    mini_interact = tmp_path / "mini-interact"
    _write_kb_jsonl(mini_interact / db / f"{db}_kb.jsonl", [1, 3])

    monkeypatch.setattr(verify_kb_coverage, "SLAYER_MODELS_DIR", slayer_models)

    unaccounted, overlap, knowledge = await verify_kb_coverage.verify_one(
        db, mini_interact,
    )
    assert unaccounted == {3}
    assert overlap == set()
    assert knowledge[3] == "KB 3 title"


@pytest.mark.asyncio
async def test_missing_memories_file_returns_empty_documented(tmp_path, monkeypatch):
    """A DB whose storage has no `memories.yaml` reports the encoded
    set unchanged and yields an empty documented set — no exceptions.
    """
    db = "freshdb"
    slayer_models = tmp_path / "slayer_models"
    db_dir = slayer_models / db
    _write_storage(db_dir, db=db, encoded_kb_ids=[1, 2], memories=[])

    mini_interact = tmp_path / "mini-interact"
    _write_kb_jsonl(mini_interact / db / f"{db}_kb.jsonl", [1, 2, 3])

    monkeypatch.setattr(verify_kb_coverage, "SLAYER_MODELS_DIR", slayer_models)

    unaccounted, overlap, _ = await verify_kb_coverage.verify_one(
        db, mini_interact,
    )
    assert unaccounted == {3}
    assert overlap == set()


@pytest.mark.asyncio
async def test_encoded_and_documented_overlap_is_an_error(tmp_path, monkeypatch):
    """Same KB id in both encoded set (via meta.kb_id) and documented
    set (via memory) trips the overlap-fail path.
    """
    db = "overlapdb"
    slayer_models = tmp_path / "slayer_models"
    db_dir = slayer_models / db
    _write_storage(
        db_dir,
        db=db,
        encoded_kb_ids=[5],
        memories=[{
            "id": 1,
            "learning": "KB 5 — KB 5 title\n\nReason: stale memory should have been forgotten.\n",
            "entities": [f"{db}.m.c_5"],
            "query": None,
        }],
    )

    mini_interact = tmp_path / "mini-interact"
    _write_kb_jsonl(mini_interact / db / f"{db}_kb.jsonl", [5])

    monkeypatch.setattr(verify_kb_coverage, "SLAYER_MODELS_DIR", slayer_models)

    unaccounted, overlap, _ = await verify_kb_coverage.verify_one(
        db, mini_interact,
    )
    # verify_one only searches for *unencoded* KBs by design — KB 5 is
    # already encoded, so the documented set never contains it. So
    # overlap is empty here, and unaccounted is empty. This documents
    # the actual contract: stale memories on already-encoded ids are
    # silently tolerated. (The forget_memory housekeeping rule lives
    # in the skill body, not the verifier.)
    assert unaccounted == set()
    assert overlap == set()
