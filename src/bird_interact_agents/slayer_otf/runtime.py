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
import tempfile
import uuid
from pathlib import Path

from slayer.memories.models import MEMORY_CANONICAL_PREFIX

from bird_interact_agents import paths as _paths
from bird_interact_agents.benchmark import get_benchmark as _get_benchmark
from bird_interact_agents.eval.annotation_io import run_edited_models_archive
from bird_interact_agents.hard8_preprocessor import extract_deleted_kb_ids
from bird_interact_agents.slayer_otf import edited_models as _edited_models
from bird_interact_agents.slayer_otf.cache import CacheEntry, ensure_db_cache
from bird_interact_agents.slayer_otf.datasource_reanchor import (
    rewrite_datasource_connection_string as _rewrite_datasource_connection_string,
)
from bird_interact_agents.memory_store_io import persist_memories
from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)
from bird_interact_agents.slayer_otf.timing import log_otf_event, otf_timer

logger = logging.getLogger(__name__)


def _otf_work_dir(instance_id: str) -> Path:
    """Per-INVOCATION scratch dir for the shared on-the-fly storage path.

    A fresh uuid suffix keeps every invocation's dir unique. Without it,
    two concurrent runs of the same task — or a recursive-adapter run that
    shares the ``bird_interact_slayer_otf`` prefix — could ``rmtree`` each
    other's live per-task SLayer store mid-run, since
    ``prepare_task_storage`` deletes ``<work_dir>/<db>`` before copying the
    cache (CodeRabbit).
    """
    p = (
        Path(tempfile.gettempdir())
        / "bird_interact_slayer_otf"
        / f"{instance_id}-{uuid.uuid4().hex[:8]}"
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


async def resolve_otf_task_storage_dir(
    *,
    db_name: str,
    task_data: dict,
    data_path_base: str,
    benchmark: str,
    apply_edited_models: bool = False,
) -> tuple[str, list[int]]:
    """Cache-only per-task SLayer storage for the on-the-fly path.

    Materialises the per-DB deterministic OTF cache (idempotent — phases
    1-3 ingest + KB-as-memories, NO LLM) and copies it into a per-task
    scratch dir with this task's ``deleted_knowledge`` KB memories masked
    (HARD-8). The agent then encodes KB items into THIS scratch at task
    time and queries off them; nothing is persisted back to the cache.

    Shared by ``claude_sdk_otf`` and structurally identical to the
    recursive adapter's private ``_resolve_otf_task_storage_dir`` — kept
    here (not imported from that adapter) so a Claude-SDK-only framework
    does not drag in ``pydantic_ai``.

    ``benchmark`` selects the per-benchmark scoped cache root (DEV-1462);
    ``"mini_interact"`` keeps the legacy root. ``db_root`` is threaded as
    the resolved ``--db-path`` so it overrides ``$BIRD_DB_PATH`` when the
    per-task datasource connection string is re-anchored.
    """
    deleted = sorted(extract_deleted_kb_ids(task_data))
    instance_id = task_data["instance_id"]
    # ``.resolve()`` is load-bearing: ``_phase1_ingest`` formats the sqlite
    # path into a 4-slash absolute URL, so a relative ``--db-path`` would
    # otherwise root at ``/<rel>/...`` and ingest the wrong file.
    mini_interact_root = Path(data_path_base).resolve()
    with otf_timer(
        "resolve_otf_task_storage_dir",
        instance_id=instance_id, db=db_name,
        deleted_kb_ids=len(deleted),
    ):
        with otf_timer(
            "ensure_db_cache", instance_id=instance_id, db=db_name,
        ):
            cache_entry = await ensure_db_cache(
                db_name,
                cache_root=_paths.slayer_otf_cache_root(benchmark=benchmark),
                mini_interact_root=mini_interact_root,
                benchmark=_get_benchmark(benchmark) if benchmark else None,
            )
        work_dir = _otf_work_dir(instance_id)
        log_otf_event(
            "otf_work_dir.resolved",
            instance_id=instance_id, work_dir=str(work_dir),
        )
        # DEV-1649: stash the source cache fingerprint so the agent's save hook
        # (maybe_save_edited_models) can stamp it into the saved store's meta.
        current_cache_fp = _edited_models.read_cache_fp(cache_entry)
        task_data["_edited_models_cache_fp"] = current_cache_fp
        # Reuse the task's saved edited store when --apply-edited-models is set
        # and a valid snapshot exists. apply_or_none owns the whole decision
        # (archive presence + meta validation + materialise), so there is exactly
        # ONE gate and it falls back to the cache path on any miss.
        if apply_edited_models:
            applied = await _edited_models.apply_or_none(
                benchmark=benchmark, db=db_name, instance_id=instance_id,
                work_dir=work_dir, task_deleted_kb_ids=set(deleted),
                current_cache_fp=current_cache_fp,
                mini_interact_root=mini_interact_root, db_root=mini_interact_root,
            )
            if applied is not None:
                log_otf_event(
                    "otf.edited_models.applied",
                    instance_id=instance_id, db=db_name, src=str(applied),
                )
                task_data["_edited_models_applied_from"] = str(
                    run_edited_models_archive(
                        benchmark=benchmark, selected_database=db_name,
                        instance_id=instance_id,
                    )
                )
                return str(applied), deleted
        with otf_timer(
            "prepare_task_storage", instance_id=instance_id, db=db_name,
        ):
            scratch = await prepare_task_storage(
                db=db_name,
                deleted_kb_ids=set(deleted),
                cache_entry=cache_entry,
                work_dir=work_dir,
                mini_interact_root=mini_interact_root,
                db_root=mini_interact_root,
            )
        # DEV-1649: capture the pre-agent baseline (post re-anchor + post mask)
        # so --save-edited-models can tell whether the agent actually edited.
        _edited_models.write_baseline_manifest(work_dir, scratch)
    return str(scratch), deleted


async def prepare_task_storage(
    *,
    db: str,
    deleted_kb_ids: set[int],
    cache_entry: CacheEntry,
    work_dir: Path,
    mini_interact_root: Path,
    db_root: Path | None = None,
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
        db_root: Authoritative DB root that overrides ``$BIRD_DB_PATH``
            when re-anchoring (DEV-1462). A LiveSQLBench run threads its
            ``--db-path`` here so conftest's / a dev shell's
            ``$BIRD_DB_PATH=<mini-interact>`` can't silently re-anchor an
            overlapping DB name (e.g. ``alien``) to the wrong sqlite.
            Mirrors the otf_encode adapter's ``ensure_db_reference`` /
            ``build_task_variant_storage`` ``db_root`` semantics.

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
        with otf_timer("prepare_task_storage.rmtree_existing", db=db):
            shutil.rmtree(scratch)
    with otf_timer("prepare_task_storage.copytree", db=db):
        shutil.copytree(cache_entry.cache_dir, scratch)

    with otf_timer("prepare_task_storage.rewrite_conn_string", db=db):
        await _rewrite_datasource_connection_string(
            db=db, scratch=scratch, mini_interact_root=mini_interact_root,
            db_root=db_root,
        )
    if deleted_kb_ids:
        # Re-encode + drop the matching embedding rows in lockstep so
        # SLayer's search corpus stays free of dangling memory:<id>
        # tokens (entities) and stale embedding rows.
        with otf_timer(
            "prepare_task_storage.kb_mask",
            db=db, deleted_kb_ids=len(deleted_kb_ids),
        ):
            await _write_memories(
                db=db, scratch=scratch,
                kb_rows=cache_entry.kb_rows, deleted_kb_ids=deleted_kb_ids,
            )
            _prune_deleted_memory_embeddings(
                db=db, scratch=scratch, deleted_kb_ids=deleted_kb_ids,
            )
    # else: the cache's pre-built memories.yaml + embeddings.db are
    # already correct for the no-deletion case.
    return scratch


async def _write_memories(
    *,
    db: str,
    scratch: Path,
    kb_rows: list[dict],
    deleted_kb_ids: set[int],
) -> None:
    """Encode KB rows to memories and persist them into ``<scratch>``.

    DEV-1668: persist through the slayer storage layer
    (:func:`memory_store_io.persist_memories`) rather than hand-writing a flat
    ``memories.yaml`` — slayer 0.9.6 stores per-id ``memories/<id>.md`` and the
    helper preserves the user-supplied ids (``<db>_kb_<n>``) + EPOCH
    ``created_at`` via the row-writer."""
    mems = encode_kb_as_memories(
        db, kb_rows, deleted_kb_ids=set(deleted_kb_ids),
    )
    # replace=True: the scratch was copied from the cache's FULL memory store;
    # re-encoding with deletions must drop the masked memories, not just upsert
    # the survivors (mirrors the old whole-file overwrite).
    await persist_memories(scratch, mems, replace=True)


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
