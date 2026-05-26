"""DEV-1470: laptop-side merger that promotes per-DB cloud-encoded OTF
references into the global warm cache.

Called from :func:`bird_interact_agents.cloud.driver.fetch` after
:func:`bird_interact_agents.cloud.collation.collate`. Walks
``<run_dir>/post_run/slayer_models_otf/<shard>/<db>/`` across every shard, and
for each db merges the union of files into
``paths.slayer_models_otf_root()/<db>/`` using a per-file newest-source-mtime-wins
rule.

Single-fingerprint-per-run invariant (L1, Codex round-1)
--------------------------------------------------------
The per-DB reference fingerprint is
:func:`bird_interact_agents.slayer_otf.cache.fingerprint_of` of the input
dataset files (sqlite, KB jsonl, column meanings). These do NOT change during
a run, so EVERY shard in a single run produces the SAME ``_reference_fp.txt``
content. Cross-shard per-file picks therefore always assemble a coherent
db — they never mix files from two *different* fingerprints within one run.
Cross-run mixing never happens because each ``fetch(run_id)`` lands its
shards in a per-run directory and the merger is scoped to that dir.

Marker invariant (H1, Codex round-1)
------------------------------------
Other code paths (notably :func:`bird_interact_agents.slayer_otf.reference_build.ensure_db_reference`)
treat ``_reference_fp.txt`` as the on-disk completeness gate: marker present
⇒ content complete ⇒ safe to reuse. The merger preserves that invariant by:

1. taking a CROSS-PROCESS ``fcntl.flock(LOCK_EX)`` on
   ``<reference_root>/<db>.merge.lock`` (shared with
   :func:`bird_interact_agents.slayer_otf.reference_build._build_reference`'s
   build lock — H4), so two concurrent fetchers AND any in-flight encoder
   serialize per-db;
2. inside the lock, unlinking the local ``_reference_fp.txt`` FIRST (so any
   concurrent reader observes "marker absent ⇒ rebuild" instead of "marker
   present + partially-updated content");
3. then atomically replacing each picked file via ``tmp + os.replace`` in the
   parent dir;
4. then writing the new ``_reference_fp.txt`` LAST — also via atomic
   ``os.replace`` (M3).

Shard completeness (M1, Codex round-1)
--------------------------------------
A shard upload is atomic from the merger's POV only when its
``_upload_complete`` marker exists. Shards missing either ``_upload_complete``
or the ``_source_mtimes.json`` sidecar are recorded in
``ignored_shards`` and skipped.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_MARKER = "_reference_fp.txt"
_SIDECAR = "_source_mtimes.json"
_UPLOAD_COMPLETE = "_upload_complete"


# ---------------------------------------------------------------------------
# Per-DB cross-process lock
# ---------------------------------------------------------------------------


@contextmanager
def _per_db_merge_lock(reference_root: Path, db: str) -> Iterator[None]:
    """Acquire ``fcntl.flock(LOCK_EX)`` on
    ``<reference_root>/<db>.merge.lock`` — shared with
    :func:`bird_interact_agents.slayer_otf.reference_build._build_reference`'s
    build lock so an in-flight encoder cannot interleave with the merge."""
    reference_root.mkdir(parents=True, exist_ok=True)
    lock_path = reference_root / f"{db}.build.lock"
    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------


def _post_run_root(run_dir: Path) -> Path:
    return run_dir / "post_run" / "slayer_models_otf"


def _shard_dirs(run_dir: Path) -> list[Path]:
    root = _post_run_root(run_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _is_shard_complete(shard_dir: Path, db: str) -> bool:
    """A shard's db slice is mergeable only when BOTH the ``_upload_complete``
    marker and the ``_source_mtimes.json`` sidecar are present."""
    db_dir = shard_dir / db
    return (
        (db_dir / _UPLOAD_COMPLETE).is_file()
        and (db_dir / _SIDECAR).is_file()
    )


def _read_sidecar(shard_dir: Path, db: str) -> dict[str, float]:
    try:
        return json.loads((shard_dir / db / _SIDECAR).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Per-file atomic replace
# ---------------------------------------------------------------------------


def _atomic_replace(src: Path, dst: Path, source_mtime: float) -> None:
    """Copy ``src``'s bytes into a tmp sibling of ``dst``, stamp the tmp with
    ``source_mtime``, then ``os.replace`` onto ``dst``. Per-file atomicity: a
    crash mid-copy leaves ``dst`` either untouched (its prior content) or
    absent — never partial.

    Codex r2: preserving ``source_mtime`` is load-bearing for the
    newest-mtime-wins rule across runs. Without it, ``dst`` inherits the
    laptop's fetch time, which is necessarily newer than the cloud's source
    mtime — and a later fetch of a *genuinely newer* cloud reference would
    be wrongly skipped because the comparison reads ``dst.stat().st_mtime``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{dst.name}.merge-", dir=str(dst.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(src.read_bytes())
        os.utime(tmp, (source_mtime, source_mtime))  # stamp BEFORE replace
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _write_marker_atomic(content: bytes, dst: Path, source_mtime: float) -> None:
    """Write ``_reference_fp.txt`` via tmp + ``os.replace`` so a reader can
    never observe a half-written marker (M3). Symmetric to ``_atomic_replace``
    but takes raw bytes (no source file). Source mtime is preserved for the
    same reason ``_atomic_replace`` preserves it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{dst.name}.merge-", dir=str(dst.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(content)
        os.utime(tmp, (source_mtime, source_mtime))  # stamp BEFORE replace
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# Per-DB merge
# ---------------------------------------------------------------------------


def _merge_one_db(
    *, db: str, shard_dirs: list[Path], reference_root: Path,
) -> tuple[int, int]:
    """Merge a single db across all *complete* shards. Returns
    ``(files_updated, files_skipped)``.

    Strategy:
      - Build ``picks: {rel_path: (max_source_mtime, shard_dir)}`` across all
        complete shards.
      - Under the per-db cross-process lock:
          * unlink local ``_reference_fp.txt`` (if any) FIRST,
          * per-file atomic-replace any rel_path whose source_mtime is newer
            than the local target's mtime (treat missing local = 0),
          * write the new ``_reference_fp.txt`` LAST (atomic replace).
    """
    # Build the per-file pick map across complete shards only.
    picks: dict[str, tuple[float, Path]] = {}
    for sdir in shard_dirs:
        if not _is_shard_complete(sdir, db):
            continue
        sidecar = _read_sidecar(sdir, db)
        for rel, mtime in sidecar.items():
            try:
                m = float(mtime)
            except (TypeError, ValueError):
                continue
            cur = picks.get(rel)
            if cur is None or m > cur[0]:
                picks[rel] = (m, sdir)
    # _upload_complete + _source_mtimes.json are scaffolding, not content.
    picks.pop(_UPLOAD_COMPLETE, None)
    picks.pop(_SIDECAR, None)

    if not picks:
        return 0, 0

    local_db = reference_root / db
    local_marker = local_db / _MARKER
    marker_pick: tuple[float, Path] | None = picks.pop(_MARKER, None)

    with _per_db_merge_lock(reference_root, db):
        # Codex r2: capture local marker mtime + bytes BEFORE deciding
        # anything. Without this, the previous implementation unlinked the
        # local marker before comparing marker mtimes — so an older cloud
        # marker (or even a missing one) would always "win" the marker
        # comparison and either downgrade the local marker or leave the
        # tree marker-less (violating "marker present ⇒ content complete").
        if local_marker.is_file():
            local_marker_mtime = local_marker.stat().st_mtime
            local_marker_bytes: bytes | None = local_marker.read_bytes()
        else:
            local_marker_mtime = 0.0
            local_marker_bytes = None

        # Pre-compute per-file decisions (no writes yet).
        content_ops: list[tuple[Path, Path, float, bool]] = []
        for rel, (source_mtime, sdir) in sorted(picks.items()):
            src = sdir / db / rel
            dst = local_db / rel
            try:
                local_mtime = dst.stat().st_mtime
            except FileNotFoundError:
                local_mtime = 0.0
            content_ops.append((src, dst, source_mtime, source_mtime > local_mtime))

        marker_will_replace = False
        marker_source_mtime = 0.0
        marker_src: Path | None = None
        if marker_pick is not None:
            marker_source_mtime, marker_sdir = marker_pick
            marker_src = marker_sdir / db / _MARKER
            marker_will_replace = marker_source_mtime > local_marker_mtime

        any_content_change = any(will for *_x, will in content_ops)

        # Skip the whole touch when nothing will actually change — preserves
        # the marker-present-⇒-content-complete invariant against concurrent
        # readers, since we never unlink the marker we leave in place.
        if not any_content_change and not marker_will_replace:
            skipped = len(content_ops) + (1 if marker_pick is not None else 0)
            return 0, skipped

        # SOMETHING will change. Unlink the local marker so any concurrent
        # reader observes "marker absent ⇒ incomplete, rebuild" during the
        # partial-content interval.
        if local_marker_bytes is not None:
            try:
                os.unlink(local_marker)
            except FileNotFoundError:
                pass

        files_updated = 0
        files_skipped = 0
        for src, dst, source_mtime, will_replace in content_ops:
            if will_replace:
                _atomic_replace(src, dst, source_mtime)
                files_updated += 1
            else:
                files_skipped += 1

        # Marker LAST. Either write the cloud marker (it won the comparison)
        # OR restore the local marker (cloud lost, but we'd unlinked it to
        # protect partial-content readers — we MUST put it back so the
        # invariant holds at the end of the merge).
        if marker_will_replace and marker_src is not None:
            _write_marker_atomic(
                marker_src.read_bytes(), local_marker, marker_source_mtime,
            )
            files_updated += 1
        else:
            if marker_pick is not None:
                files_skipped += 1
            if local_marker_bytes is not None:
                # Restore the local marker we unlinked.
                _write_marker_atomic(
                    local_marker_bytes, local_marker, local_marker_mtime,
                )

    return files_updated, files_skipped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_post_run_into_warm_cache(
    *, run_dir: Path, reference_root: Path,
) -> dict:
    """Walk ``<run_dir>/post_run/slayer_models_otf/<shard>/<db>/`` across every
    shard and merge into ``<reference_root>/<db>/`` using newest-source-mtime-wins.

    See the module docstring for the marker / sharding / fingerprint
    invariants the merge respects.

    Returns a report dict::

        {
          "merged_dbs": [{"db": ..., "files_updated": int, "files_skipped": int}, ...],
          "ignored_shards": [<shard_name>, ...],
        }
    """
    run_dir = Path(run_dir)
    reference_root = Path(reference_root)
    shard_dirs = _shard_dirs(run_dir)

    # Per-shard completeness: a shard is "fully ignored" iff NO db under it is
    # complete (missing _upload_complete and/or _source_mtimes.json for every
    # db). We still record it for the report. The per-db merger re-checks
    # completeness, so a half-complete shard still contributes its complete
    # db-slices.
    ignored_shards: list[str] = []
    dbs_seen: set[str] = set()
    for sdir in shard_dirs:
        any_complete = False
        for db_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            db = db_dir.name
            dbs_seen.add(db)
            if _is_shard_complete(sdir, db):
                any_complete = True
        if not any_complete:
            ignored_shards.append(sdir.name)

    merged: list[dict] = []
    for db in sorted(dbs_seen):
        # Only attempt the merge if at least one shard has a complete slice
        # for this db; otherwise leave the local dir untouched.
        if not any(_is_shard_complete(sdir, db) for sdir in shard_dirs):
            continue
        files_updated, files_skipped = _merge_one_db(
            db=db, shard_dirs=shard_dirs, reference_root=reference_root,
        )
        merged.append({
            "db": db,
            "files_updated": files_updated,
            "files_skipped": files_skipped,
        })

    return {"merged_dbs": merged, "ignored_shards": ignored_shards}
