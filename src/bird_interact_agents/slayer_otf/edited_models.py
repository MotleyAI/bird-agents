"""DEV-1649: persist an on-the-fly slayer agent's edited per-task SLayer store
and reuse it on the next run of the same task.

A slayer-mode (on-the-fly) agent mutates its per-task scratch store (encoding KB
items via ``create_model`` / ``edit_model`` / ``save_memory``). Today the scratch
is discarded after the run. With ``--save-edited-models`` we snapshot the whole
scratch — on task success, and only if the agent actually changed it — as a
single ``edited_models.tar.gz`` under the ``runs/`` golden store, keyed by
``(benchmark, db, instance_id)`` (latest-wins overwrite). With
``--apply-edited-models`` the next run of that task starts from the saved store
instead of the bare deterministic cache.

Design notes:

* **Archive, not a loose dir** — the ``runs/`` annotation walkers use
  ``rglob("*.json")`` and the scratch contains ``_kb_rows.json``; an opaque
  ``.tar.gz`` leaks no ``*.json`` so every current/future walker is safe.
* **No-edit detection** compares the post-run scratch against a baseline
  manifest captured right after materialisation (post re-anchor + post HARD-8
  mask), so those preparation mutations never count as agent edits.
* **Apply skips KB re-encode** — the saved store already reflects both the
  task's deleted-KB masking AND the agent's edits; re-encoding from the cache's
  ``kb_rows`` would clobber the agent-authored ``memories.yaml``.
* **Meta validation** — the archive carries ``_edited_models_meta.json`` with
  the task's ``deleted_kb_ids`` and the source cache fingerprint; apply falls
  back to the fresh cache if either diverges from the current task/cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import uuid
from pathlib import Path

from bird_interact_agents.eval.annotation_io import run_edited_models_archive
from bird_interact_agents.slayer_otf.datasource_reanchor import (
    rewrite_datasource_connection_string,
)
from bird_interact_agents.slayer_otf.timing import log_otf_event

logger = logging.getLogger(__name__)

ARCHIVE_NAME = "edited_models.tar.gz"
_BASELINE_MANIFEST = "_otf_edit_baseline.json"
_STORE_META = "_edited_models_meta.json"
# Sidecar files that must never be packed into the archive nor counted as
# content when detecting agent edits.
_TRANSIENT_SUFFIXES = (".db-wal", ".db-shm")
_EXCLUDED_NAMES = frozenset({_BASELINE_MANIFEST, _STORE_META})


# --------------------------------------------------------------------------
# manifest / change detection
# --------------------------------------------------------------------------


def _is_transient(name: str) -> bool:
    return name in _EXCLUDED_NAMES or name.endswith(_TRANSIENT_SUFFIXES)


def content_manifest(root: Path) -> dict[str, str]:
    """Map ``relpath -> sha256`` for every file under ``root``, EXCLUDING
    transient sqlite sidecars, the baseline manifest, and the store-meta file.
    Deterministic (sorted keys)."""
    root = Path(root)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_transient(path.name):
            continue
        rel = path.relative_to(root).as_posix()
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        out[rel] = h
    return out


def write_baseline_manifest(work_dir: Path, scratch: Path) -> None:
    """Write ``content_manifest(scratch)`` to ``work_dir/_BASELINE_MANIFEST``.

    Called at the END of scratch materialisation (after re-anchor + mask) so it
    captures the exact pre-agent state. The baseline lives in ``work_dir`` (the
    parent of ``<db>/``) so it is never packed into the archive nor removed by
    ``prepare_task_storage``'s ``<work_dir>/<db>`` wipe."""
    manifest = content_manifest(scratch)
    (Path(work_dir) / _BASELINE_MANIFEST).write_text(json.dumps(manifest, sort_keys=True))


def scratch_changed(work_dir: Path, scratch: Path) -> bool:
    """True iff the scratch differs from the stored baseline (missing baseline
    → conservatively True)."""
    baseline_fp = Path(work_dir) / _BASELINE_MANIFEST
    if not baseline_fp.is_file():
        return True
    try:
        baseline = json.loads(baseline_fp.read_text())
    except Exception:  # noqa: BLE001
        return True
    return content_manifest(scratch) != baseline


# --------------------------------------------------------------------------
# store meta
# --------------------------------------------------------------------------


def read_cache_fp(cache_entry) -> str:
    """The source cache fingerprint used to validate a saved store on apply and
    to stamp a freshly-saved store's meta — the ``_cache_fp.txt`` content,
    falling back to the in-memory ``fingerprint``."""
    try:
        return (Path(cache_entry.cache_dir) / "_cache_fp.txt").read_text()
    except OSError:
        return cache_entry.fingerprint


def store_meta(
    *,
    benchmark: str,
    db: str,
    instance_id: str,
    deleted_kb_ids,
    cache_fp: str,
) -> dict:
    """The ``_STORE_META`` payload validated on apply (Codex #6)."""
    return {
        "benchmark": benchmark,
        "db": db,
        "instance_id": instance_id,
        "deleted_kb_ids": sorted(int(x) for x in deleted_kb_ids),
        "cache_fp": cache_fp,
    }


# --------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------


def _checkpoint_wal(scratch: Path) -> None:
    """Fold any WAL into the main sqlite files so the archived ``embeddings.db``
    is self-contained (the ``*.db-wal``/``*.db-shm`` sidecars are not packed)."""
    for db_file in scratch.rglob("*.db"):
        try:
            con = sqlite3.connect(db_file)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.commit()
            finally:
                con.close()
        except sqlite3.Error as exc:  # pragma: no cover - locked/non-sqlite .db
            # Best-effort, but NOT silent: a failed checkpoint can leave WAL
            # data outside the archived .db (the WAL sidecar is excluded), so
            # surface it — the "self-contained archive" invariant is at risk.
            logger.warning(
                "edited_models: WAL checkpoint failed for %s: %s: %s; "
                "archived embeddings may be missing recently-committed rows",
                db_file, type(exc).__name__, exc,
            )
            continue


def save_edited_store(
    *,
    benchmark: str,
    db: str,
    instance_id: str,
    work_dir: Path,
    scratch: Path,
    deleted_kb_ids,
    cache_fp: str,
) -> Path | None:
    """Snapshot ``scratch`` to the per-task archive. Returns the archive path if
    written, else ``None`` (no agent edits vs the baseline — D7)."""
    scratch = Path(scratch)
    work_dir = Path(work_dir)
    if not scratch_changed(work_dir, scratch):
        return None

    _checkpoint_wal(scratch)
    meta = store_meta(
        benchmark=benchmark, db=db, instance_id=instance_id,
        deleted_kb_ids=deleted_kb_ids, cache_fp=cache_fp,
    )

    dest = run_edited_models_archive(
        benchmark=benchmark, selected_database=db, instance_id=instance_id,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"

    def _filter(info: tarfile.TarInfo):
        base = info.name.rsplit("/", 1)[-1]
        if _is_transient(base):
            return None
        return info

    try:
        with tarfile.open(tmp, "w:gz") as tar:
            tar.add(scratch, arcname=db, filter=_filter)
            meta_bytes = json.dumps(meta, sort_keys=True).encode()
            info = tarfile.TarInfo(name=f"{db}/{_STORE_META}")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()

    log_otf_event(
        "otf.edited_models.saved", instance_id=instance_id, db=db, dest=str(dest),
    )
    return dest


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


async def materialize_from_saved_store(
    *,
    db: str,
    archive: Path,
    work_dir: Path,
    task_deleted_kb_ids,
    current_cache_fp: str,
    mini_interact_root: Path,
    db_root: Path | None = None,
) -> Path | None:
    """Untar ``archive`` into ``work_dir/<db>``, validate ``_STORE_META``,
    re-anchor the datasource connection_string, and write a fresh baseline.

    Returns the scratch Path on success, or ``None`` to signal the caller to
    FALL BACK to the fresh cache (missing/invalid archive, or a meta mismatch —
    Codex #6). Deliberately does NOT re-run KB masking (would clobber the
    agent-authored ``memories.yaml``)."""
    archive = Path(archive)
    work_dir = Path(work_dir)
    if not archive.is_file():
        return None

    scratch = work_dir / db
    if scratch.exists():
        shutil.rmtree(scratch)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive, "r:gz") as tar:
            # PEP 706 ``filter="data"`` only exists on 3.11.4+/3.12+; the
            # project's ``requires-python=">=3.11"`` admits 3.11.0-3.11.3, where
            # the kwarg raises. Gate on the backport marker (``tarfile.data_filter``).
            if hasattr(tarfile, "data_filter"):
                tar.extractall(work_dir, filter="data")
            else:  # pragma: no cover - only on pre-3.11.4 interpreters
                tar.extractall(work_dir)
    except (tarfile.TarError, OSError):
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        return None

    meta_fp = scratch / _STORE_META
    if not meta_fp.is_file():
        shutil.rmtree(scratch, ignore_errors=True)
        return None
    try:
        meta = json.loads(meta_fp.read_text())
    except Exception:  # noqa: BLE001
        shutil.rmtree(scratch, ignore_errors=True)
        return None

    saved_deleted = sorted(int(x) for x in meta.get("deleted_kb_ids", []))
    want_deleted = sorted(int(x) for x in task_deleted_kb_ids)
    if saved_deleted != want_deleted or meta.get("cache_fp") != current_cache_fp:
        shutil.rmtree(scratch, ignore_errors=True)
        return None

    # The meta file is an apply-time artefact; drop it from the live scratch so
    # it neither pollutes SLayer's view nor a subsequent baseline/manifest.
    meta_fp.unlink()

    await rewrite_datasource_connection_string(
        db=db, scratch=scratch, mini_interact_root=mini_interact_root,
        db_root=db_root,
    )
    write_baseline_manifest(work_dir, scratch)
    return scratch


async def apply_or_none(
    *,
    benchmark: str,
    db: str,
    instance_id: str,
    work_dir: Path,
    task_deleted_kb_ids,
    current_cache_fp: str,
    mini_interact_root: Path,
    db_root: Path | None = None,
) -> Path | None:
    """Resolve + validate the saved store for a task. Returns the materialised
    scratch Path, or ``None`` (no archive, or meta mismatch) to fall back."""
    archive = run_edited_models_archive(
        benchmark=benchmark, selected_database=db, instance_id=instance_id,
    )
    if not archive.is_file():
        return None
    return await materialize_from_saved_store(
        db=db, archive=archive, work_dir=Path(work_dir),
        task_deleted_kb_ids=task_deleted_kb_ids, current_cache_fp=current_cache_fp,
        mini_interact_root=mini_interact_root, db_root=db_root,
    )


# --------------------------------------------------------------------------
# agent-facing save gate
# --------------------------------------------------------------------------


def _task_succeeded(row: dict) -> bool:
    """Success = in-task audited ``phase1_passed`` (best-of audited variants
    under ``--use-audited-gold-sql``, DEV-1606)."""
    return bool(row.get("phase1_passed"))


def maybe_save_edited_models(
    row: dict,
    *,
    benchmark: str,
    save_edited_models: bool,
    work_dir,
    slayer_storage_dir,
    deleted_kb_ids,
    cache_fp: str,
) -> Path | None:
    """Save the edited store when enabled + successful, stamping the row with
    ``edited_models_saved_path``. Best-effort: never raises into the run."""
    if not (save_edited_models and slayer_storage_dir and work_dir):
        return None
    if not _task_succeeded(row):
        return None
    try:
        dest = save_edited_store(
            benchmark=benchmark, db=row["database"], instance_id=row["instance_id"],
            work_dir=Path(work_dir), scratch=Path(slayer_storage_dir),
            deleted_kb_ids=deleted_kb_ids, cache_fp=cache_fp,
        )
    except Exception as exc:  # noqa: BLE001
        log_otf_event(
            "otf.edited_models.save_failed",
            instance_id=row.get("instance_id"), error=repr(exc),
        )
        return None
    if dest is not None:
        row["edited_models_saved_path"] = str(dest)
    return dest
