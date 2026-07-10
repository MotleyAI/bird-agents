"""DEV-1668: persist / read SLayer memories through slayer's own storage API.

slayer 0.9.6 (DEV-1658) stores memories as per-id ``memories/<id>.md`` files and
migrates away any legacy flat ``memories.yaml`` on the next ``YAMLStorage`` open.
bird-agents used to hand-write the flat file to preserve user-supplied ids
(``<db>_kb_<n>``) and a fixed EPOCH ``created_at`` for reproducible builds. Since
the flat file is now a transient migration input (and is deleted after the first
open), we persist through the storage layer instead — slayer owns the on-disk
format, and we keep the id + EPOCH determinism by writing fully-populated
``Memory`` rows via the row-writer (``save_memory`` allocates ids but does not
accept ``created_at``).

This is the single choke point for reading / writing / copying OTF KB memories;
every writer (runtime scratch, cache build, reference build), reader (autopsy),
and copier (hard8 variant) routes through it so the storage format stays a
slayer concern (see [[feedback_reuse_slayer_not_reinvent]]).
"""

from __future__ import annotations

from pathlib import Path

from slayer.memories.models import Memory
from slayer.storage.yaml_storage import (
    YAMLStorage,
    _md_to_memory,
    _memory_to_md,
    migrate_memories_layout,
)


def _as_memory(m: dict | Memory) -> Memory:
    return m if isinstance(m, Memory) else Memory.model_validate(m)


async def persist_memories(
    store_dir: Path | str, mems: list[dict | Memory], *, replace: bool = False,
) -> None:
    """Persist ``mems`` (encoder dicts or ``Memory`` objects) into the slayer
    store at ``store_dir`` as per-id ``memories/<id>.md`` files.

    Uses the row-writer so both the user-supplied ``id`` AND the encoder's fixed
    EPOCH ``created_at`` are preserved verbatim (the public ``save_memory``
    allocates ids and stamps ``created_at`` with the wall clock).

    When ``replace`` is true, any memory already on disk whose id is NOT in
    ``mems`` is deleted first — the whole store becomes exactly ``mems`` (the
    semantics of the old flat ``memories.yaml`` overwrite, needed by the
    KB-deletion masking path)."""
    storage = YAMLStorage(base_dir=str(store_dir))
    keep = {_as_memory(m).id for m in mems}
    if replace:
        for existing in await storage.list_memories():
            if existing.id not in keep:
                await storage.delete_memory(existing.id)
    for m in mems:
        await storage._save_memory_row(_as_memory(m))


def write_memories_files(store_dir: Path | str, mems: list[dict | Memory]) -> None:
    """Synchronous, event-loop-free writer of per-id ``memories/<id>.md`` files.

    For sync callers / test fixtures that cannot ``await`` and may run inside an
    event loop (so ``asyncio.run`` is unavailable). Reuses slayer's own
    ``_memory_to_md`` serializer so the on-disk bytes match what ``save_memory``
    would produce; ids + ``created_at`` are preserved verbatim."""
    mem_dir = Path(store_dir) / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    for m in mems:
        mem = _as_memory(m)
        (mem_dir / f"{mem.id}.md").write_text(
            _memory_to_md(mem), encoding="utf-8",
        )


def read_memories(store_dir: Path | str) -> list[Memory]:
    """Read every memory from ``store_dir`` (per-id ``memories/<id>.md``).

    Synchronous and event-loop-free: it runs slayer's flat→per-id migration
    (idempotent, a no-op once migrated) and then parses each ``.md`` with
    slayer's own ``_md_to_memory``. Tolerates a store that still carries a
    legacy flat ``memories.yaml`` (migrated in place first)."""
    base = Path(store_dir)
    migrate_memories_layout(str(base))
    mem_dir = base / "memories"
    if not mem_dir.is_dir():
        return []
    return [
        _md_to_memory(fp.stem, fp.read_text(encoding="utf-8"))
        for fp in sorted(mem_dir.glob("*.md"))
    ]
