"""HARD-8 preprocessor: build per-task SLayer model variants by dropping
entities whose ``meta.kb_id`` (or any element of ``meta.kb_ids``) appears
in the task's ``knowledge_ambiguity[*].deleted_knowledge`` list.

Used by the slayer-mode harness to mask KB-derived entities that the
benchmark intentionally hides from the agent. The canonical per-DB
YAML at ``slayer_models/<db>/`` is never modified — variants are
written to a task-scoped scratch directory and discarded by the runner.

The mini-interact dataset uses a single int per ambiguity entry:
``{"deleted_knowledge": 52, ...}``. We accept either int or list-of-int
to be robust.
"""

import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from slayer.core.models import SlayerModel
from slayer.storage.yaml_storage import YAMLStorage


# Memories saved by the kb-to-slayer-models skill start their `learning`
# body with `KB <int> — ` (em-dash, U+2014). The verify_kb_coverage gate
# enforces this convention; this regex matches it.
_KB_PREFIX_RE = re.compile(r"^\s*KB\s+(\d+)\s+—")


def _memory_kb_id(learning: str) -> Optional[int]:
    """Parse the leading ``KB <id> — `` marker out of a memory's learning
    body. Returns ``None`` when the body doesn't start with that marker
    (e.g. ad-hoc memories saved by hand)."""
    m = _KB_PREFIX_RE.match(learning or "")
    return int(m.group(1)) if m else None


def extract_deleted_kb_ids(task_data: dict) -> set[int]:
    """Flatten ``knowledge_ambiguity[*].deleted_knowledge`` into an int set.

    Each ambiguity entry's ``deleted_knowledge`` is normally a single int;
    a list of ints is also accepted. Empty / missing returns an empty set.
    """
    out: set[int] = set()
    for item in task_data.get("knowledge_ambiguity") or []:
        dk = item.get("deleted_knowledge")
        if dk is None:
            continue
        if isinstance(dk, list):
            for x in dk:
                out.add(int(x))
        else:
            out.add(int(dk))
    return out


def _entity_kb_ids(meta: Optional[Dict[str, Any]]) -> set[int]:
    """Return the KB ids referenced by an entity via its ``meta`` dict.

    Accepts either ``meta.kb_id`` (single int) or ``meta.kb_ids`` (list
    of ints), per the translate-mini-interact-kb skill contract.
    """
    if not meta:
        return set()
    ids: set[int] = set()
    if meta.get("kb_id") is not None:
        ids.add(int(meta["kb_id"]))
    kb_ids = meta.get("kb_ids")
    if kb_ids:
        for x in kb_ids:
            ids.add(int(x))
    return ids


def _apply_deletions(model: SlayerModel, deleted: set[int]) -> Optional[SlayerModel]:
    """Return a new ``SlayerModel`` with deletion-matching entities dropped,
    or ``None`` if the model itself should be dropped.
    """
    if _entity_kb_ids(model.meta) & deleted:
        return None
    surviving_columns = [
        c for c in model.columns if not (_entity_kb_ids(c.meta) & deleted)
    ]
    surviving_measures = [
        m for m in model.measures if not (_entity_kb_ids(m.meta) & deleted)
    ]
    surviving_aggregations = [
        a for a in model.aggregations if not (_entity_kb_ids(a.meta) & deleted)
    ]
    if (
        len(surviving_columns) == len(model.columns)
        and len(surviving_measures) == len(model.measures)
        and len(surviving_aggregations) == len(model.aggregations)
    ):
        return model
    # model_copy keeps untouched fields (joins, filters, source_queries, etc.)
    return model.model_copy(
        update={
            "columns": surviving_columns,
            "measures": surviving_measures,
            "aggregations": surviving_aggregations,
        }
    )


async def build_task_variant_storage(
    *,
    canonical_storage_root: Path,
    db_name: str,
    deleted_kb_ids: set[int],
    work_dir: Path,
    mini_interact_root: Path | None = None,
) -> Path:
    """Build a per-task SLayer YAMLStorage, optionally with HARD-8
    deletions applied.

    Always materialises a fresh per-task copy of the canonical models
    so the SLayer MCP server can safely write back type-refinement
    metadata, and so any agent ``create_model`` / ``edit_model`` /
    ``delete_model`` calls land in the temp dir rather than mutating
    the committed canonical reference.

    Parameters
    ----------
    canonical_storage_root
        The ``slayer_models/`` root containing per-DB folders.
    db_name
        The DB name (folder name under ``canonical_storage_root``).
    deleted_kb_ids
        KB ids to mask. Empty set means "copy everything verbatim";
        non-empty means "copy minus the matching entities".
    work_dir
        Task-scoped scratch directory. The variant is written to
        ``<work_dir>/<db_name>/``.
    mini_interact_root
        Root the portable (relative) datasource connection string is
        re-anchored at when ``$BIRD_DB_PATH`` is unset. Defaults to the sibling
        ``mini-interact/`` next to ``canonical_storage_root``'s parent. Pass the
        run's ``--db-path`` so a non-default dataset is honoured at task time
        (the OTF flow builds its reference against that same root). ``$BIRD_DB_PATH``
        still wins when set.

    Returns
    -------
    Path
        The base_dir to hand to ``YAMLStorage`` / ``SLAYER_STORAGE`` for
        this task.
    """
    canonical = canonical_storage_root / db_name
    src = YAMLStorage(base_dir=str(canonical))
    # Materialise into a UNIQUE per-build scratch dir under work_dir. A fresh
    # dir each call means (a) no model from a prior run reusing the same
    # work_dir survives (the OTF flow's work_dir is the deterministic
    # ``/tmp/bird_interact_slayer_otf/<instance_id>``), and (b) concurrent
    # builds sharing a work_dir (a duplicate instance_id, or two runs of the
    # same task) never delete each other's storage — unlike clearing a shared
    # ``work_dir/<db>`` path (Codex finding). Orphan dirs are scratch (/tmp,
    # cleared on reboot). The caller uses the returned path.
    work_dir.mkdir(parents=True, exist_ok=True)
    variant_root = Path(tempfile.mkdtemp(prefix=f"{db_name}-", dir=str(work_dir)))
    dst = YAMLStorage(base_dir=str(variant_root))

    ds = await src.get_datasource(db_name)
    if ds is not None:
        # The committed YAML carries a relative connection_string
        # (`sqlite:///<db>/<db>.sqlite`) so it stays portable across
        # machines. Resolve it to an absolute path here, anchored at
        # ``$BIRD_DB_PATH`` or a sibling ``mini-interact/`` next to
        # ``slayer_models/``, before the SLayer MCP server reads it.
        from bird_interact_agents.slayer_pipeline.portable_connection import (
            resolve_committed_connection_string,
        )

        root = (
            mini_interact_root
            if mini_interact_root is not None
            else canonical_storage_root.parent.parent / "mini-interact"
        )
        resolved = resolve_committed_connection_string(
            ds.connection_string or "", root
        )
        if resolved != ds.connection_string:
            ds = ds.model_copy(update={"connection_string": resolved})
        await dst.save_datasource(ds)

    for name in await src.list_models():
        model = await src.get_model(name)
        if model is None:
            continue
        if deleted_kb_ids:
            kept = _apply_deletions(model, deleted_kb_ids)
            if kept is None:
                continue
            await dst.save_model(kept)
        else:
            await dst.save_model(model)

    # Copy memories + embeddings. YAMLStorage.save_memory allocates new
    # ids (so embedding rows keyed on `memory:<old_id>` would point to
    # nothing post-copy); we therefore bypass the storage API and copy
    # the underlying files directly, preserving ids and filtering
    # deleted-KB rows in lockstep so memory rows and their embedding
    # rows stay in sync.
    _copy_memories_and_embeddings(
        canonical=canonical, variant_root=variant_root,
        deleted_kb_ids=deleted_kb_ids,
    )
    return variant_root


def _copy_memories_and_embeddings(
    *,
    canonical: Path,
    variant_root: Path,
    deleted_kb_ids: set[int],
) -> None:
    """Copy ``memories.yaml`` and ``embeddings.db`` from canonical → variant,
    filtering out any memory whose ``KB <n>`` header lands in
    ``deleted_kb_ids`` and the corresponding ``memory:<n>`` embedding row.

    Memory ids are preserved across the copy so embedding ``canonical_id``
    references (``memory:<id>``) keep resolving inside the variant.
    """
    src_mem = canonical / "memories.yaml"
    dst_mem = variant_root / "memories.yaml"
    surviving_ids: set[int] = set()
    if src_mem.exists():
        rows = _load_yaml_list(src_mem)
        kept: list[dict[str, Any]] = []
        for row in rows:
            kb_id = _memory_kb_id(row.get("learning") or "")
            if kb_id is not None and kb_id in deleted_kb_ids:
                continue
            kept.append(row)
            if isinstance(row.get("id"), int):
                surviving_ids.add(int(row["id"]))
        _dump_yaml_list(dst_mem, kept)

    src_emb = canonical / "embeddings.db"
    dst_emb = variant_root / "embeddings.db"
    if src_emb.exists():
        shutil.copyfile(src_emb, dst_emb)
        if deleted_kb_ids:
            _prune_dropped_memory_embeddings(dst_emb, surviving_ids)


def _load_yaml_list(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    data = yaml.safe_load(text) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list at top level")
    return data


def _dump_yaml_list(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump(list(rows), sort_keys=False))


def _prune_dropped_memory_embeddings(
    db_path: Path, surviving_memory_ids: set[int],
) -> None:
    """Delete embedding rows whose ``canonical_id`` is ``memory:<n>`` for
    any ``n`` not in ``surviving_memory_ids``. Non-memory canonical_ids
    are left alone — they belong to the model/column/measure tree, which
    ``_apply_deletions`` has already pruned at the YAML level (unused
    embedding rows are inert because the search corpus is rebuilt from
    the YAMLs each call).
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT canonical_id FROM embeddings WHERE canonical_id LIKE 'memory:%'"
        ).fetchall()
        to_delete: list[str] = []
        for (cid,) in rows:
            try:
                mid = int(cid.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            if mid not in surviving_memory_ids:
                to_delete.append(cid)
        if to_delete:
            con.executemany(
                "DELETE FROM embeddings WHERE canonical_id = ?",
                [(cid,) for cid in to_delete],
            )
            con.commit()
    finally:
        con.close()
