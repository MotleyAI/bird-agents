"""Per-task runtime glue for the on-the-fly setup path.

Takes a built :class:`~bird_interact_agents.slayer_otf.cache.CacheEntry`
and materialises a per-task SLayer storage scratch dir:

1. ``shutil.copytree`` the cache to ``<work_dir>/<db>/``.
2. Resolve the datasource ``connection_string`` from its portable
   relative form to an absolute path (same trick as
   ``hard8_preprocessor.build_task_variant_storage``).
3. Encode KB rows into ``memories.yaml`` via
   :func:`encode_kb_as_memories`, with the task's ``deleted_kb_ids``
   filtered out and cross-refs scrubbed in lockstep.

Returns the scratch base_dir, suitable for passing into
``slayer_mcp_stdio_config`` (i.e., it's the dir that becomes
``SLAYER_STORAGE``).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

import yaml

from slayer.memories.models import MEMORY_CANONICAL_PREFIX
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.cache import CacheEntry
from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)
from bird_interact_agents.slayer_pipeline.portable_connection import (
    resolve_committed_connection_string,
)

logger = logging.getLogger(__name__)


async def prepare_task_storage(
    *,
    db: str,
    deleted_kb_ids: set[int],
    cache_entry: CacheEntry,
    work_dir: Path,
    mini_interact_root: Path,
) -> Path:
    """Materialise the per-task SLayer storage at ``<work_dir>/<db>/``.

    Args:
        db: Datasource id (== the cache's db folder name).
        deleted_kb_ids: KB ids the task wants masked.
        cache_entry: Result of :func:`ensure_db_cache`.
        work_dir: Per-task scratch dir. The function creates the
            ``<db>/`` subdir inside it.
        mini_interact_root: Current source-of-truth mini-interact path
            (== the harness's ``--db-path``). Used to re-anchor the
            datasource ``connection_string`` so a cache built against
            a different root still points at the right sqlite file.

    Returns:
        The path that should be set as ``SLAYER_STORAGE`` for the
        per-task SLayer MCP server.
    """
    scratch = work_dir / db
    # ``copytree`` requires the destination to NOT pre-exist. The
    # harness's _task_variant_workdir creates the parent eagerly, but
    # this <db>/ subdir is ours to own — wipe it if a previous run
    # left it behind.
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(cache_entry.cache_dir, scratch)

    await _rewrite_datasource_connection_string(
        db=db, scratch=scratch, mini_interact_root=mini_interact_root,
    )
    if deleted_kb_ids:
        # Re-encode + drop the matching embedding rows in lockstep so
        # SLayer's search corpus stays free of dangling memory:<id>
        # tokens (entities) and stale embedding rows.
        _write_memories_yaml(
            db=db, scratch=scratch,
            kb_rows=cache_entry.kb_rows, deleted_kb_ids=deleted_kb_ids,
        )
        _prune_deleted_memory_embeddings(
            db=db, scratch=scratch, deleted_kb_ids=deleted_kb_ids,
        )
    # else: the cache's pre-built memories.yaml + embeddings.db are
    # already correct for the no-deletion case.
    return scratch


def _expected_connection_string(db: str, mini_interact_root: Path) -> str:
    """Build the sqlite URL the datasource SHOULD carry for the current
    mini-interact root. Matches the 4-slash absolute form SQLAlchemy
    expects (and that ``slayer datasources create`` would itself emit
    when run against this root)."""
    abs_sqlite = (mini_interact_root / db / f"{db}.sqlite").resolve()
    return f"sqlite:////{abs_sqlite.as_posix().lstrip('/')}"


async def _rewrite_datasource_connection_string(
    *,
    db: str,
    scratch: Path,
    mini_interact_root: Path,
) -> None:
    """Re-anchor the copied datasource's ``connection_string`` to the
    current mini-interact root.

    The cache was built by ``slayer datasources create`` against
    whatever root the FIRST call used. If a subsequent call's
    ``--db-path`` points elsewhere (e.g. a different worktree, or a
    fresh clone), the cache's absolute path is stale. The fingerprint
    fix prevents *most* of these (different root → different cache),
    but this rewrite is the belt-and-suspenders: it normalises the
    connection_string to the current root unconditionally, so even an
    in-tree cache rebuild that landed against an unexpected absolute
    prefix gets corrected before SLayer opens the sqlite file.

    Also handles the legacy relative form (``sqlite:///<db>/...``) via
    :func:`resolve_committed_connection_string`.
    """
    storage = YAMLStorage(base_dir=str(scratch))
    ds = await storage.get_datasource(db)
    if ds is None:
        raise RuntimeError(
            f"slayer_otf: cached scratch is missing datasource {db!r}; "
            f"cache_dir layout may have changed."
        )
    if ds.connection_string is None:
        return

    # Step 1: convert relative form → absolute (no-op for absolute).
    resolved = resolve_committed_connection_string(
        ds.connection_string, mini_interact_root,
    )
    # Step 2: force the absolute path to point at the current root.
    # Cheap to compute and write — strings are short — so we always
    # overwrite rather than diffing slash counts.
    expected = _expected_connection_string(db, mini_interact_root)
    if expected != resolved:
        resolved = expected
    if resolved != ds.connection_string:
        ds = ds.model_copy(update={"connection_string": resolved})
        await storage.save_datasource(ds)


def _write_memories_yaml(
    *,
    db: str,
    scratch: Path,
    kb_rows: list[dict],
    deleted_kb_ids: set[int],
) -> None:
    """Encode KB rows to memories and write them to ``<scratch>/memories.yaml``.

    We bypass ``YAMLStorage.save_memory`` and write the file directly so
    user-supplied memory ids (``<db>_kb_<n>``) are preserved verbatim —
    the same trick ``hard8_preprocessor._copy_memories_and_embeddings``
    uses for its variant copy."""
    mems = encode_kb_as_memories(
        db, kb_rows, deleted_kb_ids=set(deleted_kb_ids),
    )
    (scratch / "memories.yaml").write_text(
        yaml.safe_dump(mems, sort_keys=False)
    )


def _prune_deleted_memory_embeddings(
    *,
    db: str,
    scratch: Path,
    deleted_kb_ids: set[int],
) -> None:
    """Drop embedding rows for memories the deletion filter removed.

    The cache pre-populates embeddings for ALL KB memories; the per-task
    scratch must mirror the per-task ``memories.yaml`` exactly or
    SearchService's channel 3 will return embedding hits for memory
    ids that no longer exist in the YAML. Mirrors
    ``hard8_preprocessor._prune_dropped_memory_embeddings``'s strategy
    (DELETE by canonical_id) but keyed on our string ids.
    """
    if not deleted_kb_ids:
        return
    db_path = scratch / "embeddings.db"
    if not db_path.is_file():
        return  # embeddings disabled in cache — nothing to prune.
    to_delete = [
        f"{MEMORY_CANONICAL_PREFIX}{db}_kb_{kb_id}"
        for kb_id in deleted_kb_ids
    ]
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "DELETE FROM embeddings WHERE canonical_id = ?",
            [(cid,) for cid in to_delete],
        )
        con.commit()
    finally:
        con.close()
