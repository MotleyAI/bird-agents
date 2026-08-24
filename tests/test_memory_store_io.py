"""DEV-1668: unit tests for the shared SLayer memory-store IO helper.

The helper is the single choke point for reading / writing / copying OTF KB
memories through slayer 0.9.6's per-id ``memories/<id>.md`` storage (DEV-1658),
preserving user-supplied ids and the encoder's fixed EPOCH ``created_at``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from slayer.memories.models import Memory
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.memory_store_io import (
    persist_memories,
    read_memories,
    write_memories_files,
)

EPOCH = "1970-01-01T00:00:00Z"


def _dict(mid: str, learning: str = "KB body", entities=None) -> dict:
    return {
        "version": 1, "id": mid, "learning": learning,
        "entities": entities or [], "query": None, "created_at": EPOCH,
    }


async def test_persist_then_read_round_trips_id_and_created_at(tmp_path: Path):
    await persist_memories(tmp_path, [_dict("db_kb_1", "KB 1 — body")])
    mems = read_memories(tmp_path)
    assert [m.id for m in mems] == ["db_kb_1"]
    assert mems[0].learning == "KB 1 — body"
    # EPOCH preserved verbatim (not the wall clock).
    assert mems[0].created_at.year == 1970


async def test_persist_writes_per_id_md_files(tmp_path: Path):
    await persist_memories(tmp_path, [_dict("a"), _dict("b")])
    names = sorted(p.name for p in (tmp_path / "memories").glob("*.md"))
    assert names == ["a.md", "b.md"]
    # No legacy flat file is produced.
    assert not (tmp_path / "memories.yaml").exists()


async def test_persist_replace_drops_removed_ids(tmp_path: Path):
    await persist_memories(tmp_path, [_dict("k1"), _dict("k2"), _dict("k3")])
    # Re-persist a subset with replace=True — k2 must be deleted.
    await persist_memories(tmp_path, [_dict("k1"), _dict("k3")], replace=True)
    assert {m.id for m in read_memories(tmp_path)} == {"k1", "k3"}


async def test_persist_without_replace_upserts_only(tmp_path: Path):
    await persist_memories(tmp_path, [_dict("k1"), _dict("k2")])
    await persist_memories(tmp_path, [_dict("k1", "updated")])
    by_id = {m.id: m for m in read_memories(tmp_path)}
    assert set(by_id) == {"k1", "k2"}  # k2 survives (no replace)
    assert by_id["k1"].learning == "updated"


async def test_persist_accepts_memory_objects(tmp_path: Path):
    mem = Memory(id="obj", learning="L", entities=[])
    await persist_memories(tmp_path, [mem])
    assert [m.id for m in read_memories(tmp_path)] == ["obj"]


def test_write_memories_files_sync_round_trips(tmp_path: Path):
    write_memories_files(tmp_path, [_dict("s1", "KB s1"), _dict("s2", "KB s2")])
    assert {m.id for m in read_memories(tmp_path)} == {"s1", "s2"}


def test_read_memories_empty_when_absent(tmp_path: Path):
    assert read_memories(tmp_path) == []


def test_read_memories_orders_by_numeric_id(tmp_path: Path):
    # Lexical order would put "10" before "2"; read_memories must sort by the
    # trailing integer so KB paragraphs join in KB order (CodeRabbit).
    write_memories_files(tmp_path, [
        _dict("db_kb_2"), _dict("db_kb_10"), _dict("db_kb_1"),
    ])
    assert [m.id for m in read_memories(tmp_path)] == [
        "db_kb_1", "db_kb_2", "db_kb_10",
    ]


def test_read_memories_numeric_ids_sort_numerically(tmp_path: Path):
    write_memories_files(tmp_path, [_dict("2"), _dict("10"), _dict("1")])
    assert [m.id for m in read_memories(tmp_path)] == ["1", "2", "10"]


def test_read_memories_migrates_legacy_flat_file(tmp_path: Path):
    # A legacy flat ``memories.yaml`` must be migrated to per-id on read.
    (tmp_path / "memories.yaml").write_text(
        yaml.safe_dump([_dict("legacy_1", "KB legacy")]), encoding="utf-8",
    )
    mems = read_memories(tmp_path)
    assert [m.id for m in mems] == ["legacy_1"]
    # Migration converted it to the per-id layout and removed the flat file.
    assert (tmp_path / "memories" / "legacy_1.md").exists()
    assert not (tmp_path / "memories.yaml").exists()


async def test_persisted_store_is_readable_via_slayer_storage(tmp_path: Path):
    """A store written by the helper is a valid slayer store — the runtime opens
    it via ``YAMLStorage`` and lists the same memories."""
    await persist_memories(tmp_path, [_dict("db_kb_7", "KB 7 — x")])
    storage = YAMLStorage(base_dir=str(tmp_path))
    listed = await storage.list_memories()
    assert [m.id for m in listed] == ["db_kb_7"]
