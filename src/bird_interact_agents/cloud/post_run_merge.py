"""DEV-1470: laptop-side merger that promotes per-DB cloud-encoded OTF
references into the global warm cache.

Called from :func:`bird_interact_agents.cloud.driver.fetch` after
:func:`bird_interact_agents.cloud.collation.collate`. Walks
``<run_dir>/post_run/slayer_models_otf/<shard>/<db>/`` across every shard, and
for each db decides cloud-vs-local at the WHOLE-DB level using the
``_reference_fp.txt`` marker as the binding tie-breaker (Codex round-5).

Two-level merge rule
--------------------
The merger has TWO distinct decisions and uses different rules for each:

1. **Cross-shard picks (within the cloud side)** — when multiple cloud
   shards exist for the same db, pick the file with the max source_mtime
   per relative path. Safe because of the single-fingerprint-per-run
   invariant (L1) — see below.

2. **Cloud-vs-local merge** — WHOLE-DB decision. If the cloud's chosen
   ``_reference_fp.txt`` source_mtime is newer than the local marker's
   mtime, the ENTIRE cloud reference wins as a unit: every picked file
   (content + marker) is atomic-replaced even if some local files have
   newer individual mtimes. Otherwise NOTHING is touched.

   Why whole-db: the marker BINDS the content. Mixing files from
   different fingerprints under one marker violates the
   ``marker present ⇒ content matches fingerprint`` invariant that
   downstream readers (:func:`bird_interact_agents.slayer_otf.reference_build._reuse_reference`,
   :func:`bird_interact_agents.slayer_otf.cache.ensure_db_cache`) trust.
   The previous per-file mtime-wins rule could let a newer cloud marker
   win while one or more cloud payload files lost to newer local files
   (or vice versa), producing a mixed-fingerprint tree marked complete
   for the wrong fp.

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
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, ValidationError

from bird_interact_agents.eval.annotation_io import (
    run_annotation_path,
    run_trajectory_path,
    write_run_annotation_no_overwrite,
)
from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation, TaskAnnotation

logger = logging.getLogger(__name__)

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
        # Codex r5: cloud-vs-local must be a WHOLE-DB decision keyed off the
        # marker, NOT a per-file mtime comparison. The previous per-file
        # logic could let a newer cloud marker win while one or more cloud
        # payload files lost to newer local files (or vice versa), leaving
        # a mixed-fingerprint tree marked as complete for the wrong fp —
        # violating the "marker present ⇒ content matches fingerprint"
        # invariant that downstream readers (`_reuse_reference`,
        # `cache.ensure_db_cache`) depend on.
        #
        # New rule: if the cloud's best `_reference_fp.txt` source_mtime is
        # newer than the local marker's, the WHOLE cloud reference wins —
        # every picked file (content + marker) is atomic-replaced. Else,
        # nothing is touched. Per-file picks ACROSS shards are still
        # mtime-wins (safe because all shards in a single run carry the
        # same fingerprint per the L1 invariant in the module docstring).

        if local_marker.is_file():
            local_marker_mtime = local_marker.stat().st_mtime
        else:
            local_marker_mtime = 0.0

        # No cloud marker for this db → cloud content cannot be trusted
        # (no fingerprint to bind it to). Skip everything.
        if marker_pick is None:
            return 0, len(picks)

        marker_source_mtime, marker_sdir = marker_pick
        marker_src = marker_sdir / db / _MARKER

        if marker_source_mtime <= local_marker_mtime:
            # Local marker is at-least-as-new as any cloud marker → local
            # reference is fresher (or same fingerprint). Don't touch
            # anything. files_skipped accounts for the cloud picks we ignored.
            return 0, len(picks) + 1

        # Cloud wins the whole db. Unlink the local marker FIRST so any
        # concurrent reader observes "marker absent ⇒ incomplete, rebuild"
        # during the per-file replace window — preserves the invariant for
        # readers that don't take the per-db merge flock (e.g. `_reuse_reference`
        # is reads-only under the build flock, not the merge flock).
        if local_marker.is_file():
            try:
                os.unlink(local_marker)
            except FileNotFoundError:
                pass

        # Codex r6: delete LOCAL stale files that aren't in the cloud's
        # picks before writing the new content + marker. Without this,
        # `<db>/` ends up containing the cloud's picked files PLUS any
        # old local files the cloud reference doesn't include — same
        # class of mixed-fingerprint bug as r5 caught for content vs
        # marker, just on the file-presence axis. The new marker (cloud's
        # fp) would sit on top of leftover files from the prior fp,
        # silently breaking the marker-binds-content invariant.
        picked_relpaths = set(picks.keys()) | {_MARKER}
        if local_db.is_dir():
            for p in list(local_db.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(local_db).as_posix()
                if rel in picked_relpaths:
                    continue
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            # Sweep empty subdirs left behind by stale-file removal,
            # bottom-up. The local_db itself stays — it's about to be
            # re-populated by the atomic replaces below.
            for p in sorted(local_db.rglob("*"), reverse=True):
                if p.is_dir() and p != local_db:
                    try:
                        p.rmdir()
                    except OSError:
                        pass  # not empty (e.g. still contains a picked file)

        files_updated = 0
        for rel, (source_mtime, sdir) in sorted(picks.items()):
            src = sdir / db / rel
            dst = local_db / rel
            _atomic_replace(src, dst, source_mtime)
            files_updated += 1

        # Marker LAST.
        _write_marker_atomic(
            marker_src.read_bytes(), local_marker, marker_source_mtime,
        )
        files_updated += 1

    return files_updated, 0


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
    skipped_dbs: list[dict] = []
    for db in sorted(dbs_seen):
        if not any(_is_shard_complete(sdir, db) for sdir in shard_dirs):
            # Codex r7: this db appeared under some shard but has no
            # complete slice anywhere. Without recording it explicitly,
            # a partial-shard failure (e.g. actor uploaded db_a fully but
            # died mid-db_b → db_b's slice is incomplete, db_a's isn't)
            # would silently lose db_b's warm-cache artifact: db_a lands
            # in `merged_dbs`, the shard isn't in `ignored_shards`
            # (db_a's slice IS complete), and the user has no signal
            # that db_b existed and was supposed to land.
            skipped_dbs.append({
                "db": db,
                "reason": "no complete shard slice (incomplete upload)",
            })
            continue
        files_updated, files_skipped = _merge_one_db(
            db=db, shard_dirs=shard_dirs, reference_root=reference_root,
        )
        merged.append({
            "db": db,
            "files_updated": files_updated,
            "files_skipped": files_skipped,
        })

    report = {
        "merged_dbs": merged,
        "skipped_dbs": skipped_dbs,
        "ignored_shards": ignored_shards,
    }

    # Codex r6: persist the report to disk under `run_dir` so it survives
    # past the `fetch()` return value (which the CLI currently discards).
    # Without this, ignored shards and merge counts are visible only to
    # direct Python callers — post-run merge failures become un-auditable.
    # Best-effort: a logging-side write failure must NOT poison the
    # already-completed merge.
    try:
        (run_dir / "merge_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    except OSError:
        pass

    return report


# ---------------------------------------------------------------------------
# DEV-1515: fetch-side merge of per-row submission_annotation.json files
# into the main checkout's annotations/ tree. Same posture as the OTF
# reference merge above — no-overwrite, schema-validated, auditable.
# ---------------------------------------------------------------------------




class AnnotationMergeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    benchmark: str
    merged: int = 0
    skipped_existing: int = 0
    overwritten_newer_attempt: int = 0
    rejected_invalid: int = 0
    merged_paths: list[str] = []
    skipped_paths: list[str] = []
    overwritten_paths: list[str] = []
    rejected_paths: list[str] = []


_ATTEMPT_RE = re.compile(r"attempt-(\d+)\.json")


def _attempt_from_trajectory_path(traj: str | None) -> int | None:
    """Parse ``rows/<iid>/attempt-N.json`` → ``N``. Returns None when
    the path is missing or unparseable so callers can fall back to a
    safe default (no-overwrite)."""
    if not traj:
        return None
    m = _ATTEMPT_RE.search(traj)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def merge_submission_annotations(
    *,
    downloaded_run_dir: Path,
    run_id: str,
    benchmark: str,
) -> AnnotationMergeReport:
    """Walk ``<downloaded_run_dir>/rows/<inst>/submission_annotation.json``
    and merge each into ``<annotations_root>/<benchmark>/<db>/<inst>.submission.<run_id>.json``
    (resolved via ``paths.annotations_root()`` so ``BIRD_ANNOTATIONS_ROOT`` is honoured).

    Contract:
    * No-overwrite-if-present. A pre-existing destination is preserved
      (logged as ``skipped_existing``).
    * Each candidate is schema-validated via
      :class:`bird_interact_agents.eval.annotation_schema.SubmissionAnnotation`.
      Invalid files are recorded as ``rejected_invalid``; they do NOT
      create a destination file.
    * Writes an audit report at
      ``<downloaded_run_dir>/annotation_merge_report.json``.

    ``annotations_root`` takes priority over ``main_checkout_root`` so
    callers can honour ``BIRD_ANNOTATIONS_ROOT`` by passing
    ``paths.annotations_root()`` directly.  When neither is given,
    ``submission_annotation_path`` falls back to ``paths.annotations_root()``.
    """
    rows_dir = downloaded_run_dir / "rows"
    report = AnnotationMergeReport(run_id=run_id, benchmark=benchmark)
    # DEV-1591: the actor already wrote version/agent_model onto each record
    # at the source, so the merge just preserves them; this manifest is only a
    # fallback to COPY version/agent_model onto any record that lacks them
    # (legacy runs). Loaded once (best-effort) from the fetched run dir.
    _manifest: "dict | None" = None
    # DEV-1591: distinguish "manifest absent" from "present-but-unreadable". A
    # corrupt manifest must NOT re-enable the downstream self-load fallback
    # (which would draw version/agent_model from a same-named run under the
    # default results root); pass allow_manifest_fallback=False in that case.
    _manifest_unreadable = False
    _manifest_path = downloaded_run_dir / "manifest.json"
    if _manifest_path.exists():
        try:
            _manifest = json.loads(_manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            _manifest = None
            _manifest_unreadable = True
    if rows_dir.exists():
        for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
            src = sub / "submission_annotation.json"
            if not src.exists():
                continue
            try:
                ann = SubmissionAnnotation.model_validate_json(
                    src.read_text(),
                )
            except (ValidationError, ValueError) as e:
                report.rejected_invalid += 1
                report.rejected_paths.append(
                    f"{src}: {type(e).__name__}: {e}"
                )
                continue
            # DEV-1533: write to runs/<benchmark>/<db>/<inst>/<run_id>.json
            dest = run_annotation_path(
                benchmark=benchmark,
                selected_database=ann.selected_database,
                instance_id=ann.instance_id,
                run_id=run_id,
                repo_root=None,
            )
            dest_existed = dest.exists()
            written = write_run_annotation_no_overwrite(
                ann, dest, benchmark=benchmark, run_id=run_id,
                manifest=_manifest,
                allow_manifest_fallback=not _manifest_unreadable,
            )
            if written:
                if dest_existed:
                    report.overwritten_newer_attempt += 1
                    report.overwritten_paths.append(str(dest))
                else:
                    report.merged += 1
                    report.merged_paths.append(str(dest))
            else:
                report.skipped_existing += 1
                report.skipped_paths.append(str(dest))

            # Trajectory sidecar — best-effort copy of attempt-N.json.
            # Refresh whenever the annotation itself was (re)written so an
            # accepted newer attempt never leaves the prior attempt's
            # trajectory sitting next to the new annotation. When the
            # annotation was preserved (``written`` False), leave the
            # existing sidecar alone.
            traj_dest = run_trajectory_path(
                benchmark=benchmark,
                selected_database=ann.selected_database,
                instance_id=ann.instance_id,
                run_id=run_id,
                repo_root=None,
            )
            if written:
                traj_rel = ann.submission.trajectory_path or ""
                m = re.search(r"rows/(.+)$", traj_rel)
                if m:
                    traj_src = rows_dir / ann.instance_id / Path(m.group(1)).name
                    if not traj_src.exists():
                        traj_src = rows_dir / m.group(1)
                    if traj_src.exists():
                        traj_dest.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(traj_src, traj_dest)
                        except Exception:  # noqa: BLE001
                            pass

    # Audit log lives alongside the downloaded run dir (next to the
    # OTF merge_report.json above, for symmetry).
    audit_path = downloaded_run_dir / "annotation_merge_report.json"
    try:
        audit_path.write_text(report.model_dump_json(indent=2) + "\n")
    except Exception:  # noqa: BLE001
        logger.warning("[post_run_merge] failed to write audit log %s", audit_path, exc_info=True)
    return report


# ---------------------------------------------------------------------------
# DEV-1518: task annotation + audited gold variant merge
# ---------------------------------------------------------------------------

class TaskAnnotationMergeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merged: int = 0
    errors: int = 0
    error_details: list[str] = []


class AuditedGoldVariantsMergeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    added: int = 0
    skipped_duplicate: int = 0
    errors: int = 0
    error_details: list[str] = []




def _normalise_benchmark(benchmark: str) -> str:
    return benchmark


def merge_task_annotations(
    *,
    downloaded_run_dir: Path,
    benchmark: str,
    annotations_root: Path,
) -> TaskAnnotationMergeReport:
    """Merge per-task TaskAnnotation files from a downloaded run into local storage.

    Reads from ``downloaded_run_dir/rows/<instance_id>/task_annotation.json``.
    Writes to ``annotations_root/<benchmark>/<db>/<instance_id>.task.json``.
    Always overwrites existing files (skip logic was upstream at the worker).
    """
    bm = _normalise_benchmark(benchmark)
    report = TaskAnnotationMergeReport()
    rows_dir = downloaded_run_dir / "rows"
    if not rows_dir.exists():
        return report

    for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
        src = sub / "task_annotation.json"
        if not src.exists():
            continue
        try:
            data = json.loads(src.read_text())
            TaskAnnotation.model_validate(data)
            db = data["selected_database"]
            instance_id = data["instance_id"]
        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            report.errors += 1
            report.error_details.append(f"{src}: {e}")
            continue

        dest = annotations_root / bm / db / f"{instance_id}.task.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2) + "\n")
        report.merged += 1

    return report


def merge_audited_gold_variants(
    *,
    downloaded_run_dir: Path,
    benchmark: str,
    audited_gold_root: Path,
    override: bool = False,
) -> AuditedGoldVariantsMergeReport:
    """Merge per-task audited gold variant JSONL files into the consolidated JSONL.

    Reads from ``downloaded_run_dir/rows/<instance_id>/audited_gold_variants.jsonl``.
    Writes to ``audited_gold_root/<benchmark>_audited.jsonl`` as one grouped
    ``AuditedGoldRow`` JSON line per ``instance_id``.

    Deduplicates by ``instance_id`` (clean-slate per task — all variants for a
    re-annotated task replace the old row). An empty source file signals
    ``original_gold_is_correct=True``; those instances are purged from the
    consolidated JSONL (no row = original is the gold).

    When ``override=True``, incoming rows replace existing rows with the same
    ``instance_id`` so that a re-annotation run updates the local consolidated
    file rather than leaving it stale.
    """
    from bird_interact_agents.eval.annotation_schema import AuditedGoldRow

    bm = _normalise_benchmark(benchmark)
    report = AuditedGoldVariantsMergeReport()
    rows_dir = downloaded_run_dir / "rows"
    if not rows_dir.exists():
        return report

    consolidated = audited_gold_root / bm / f"{bm}_audited.jsonl"

    def _read_consolidated() -> dict[str, str]:
        """Return existing rows as {instance_id: grouped_row_json}.

        Handles both the current grouped format (``variants`` key present,
        one object per instance_id) and the legacy flat format (one object
        per variant, multiple rows per instance_id).  Legacy rows are
        grouped via ``AuditedGoldRow.from_flat_rows`` so that all variants
        survive a partial override rerun."""
        out: dict[str, str] = {}
        flat_by_iid: dict[str, list[dict]] = {}
        flat_order: list[str] = []
        if not consolidated.exists():
            return out
        for raw in consolidated.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                if not isinstance(d, dict):
                    continue
                if "variants" in d:
                    row = AuditedGoldRow.model_validate(d)
                    out[row.instance_id] = row.model_dump_json()
                    continue
                iid = d.get("instance_id", "")
            except Exception:
                continue
            if iid:
                if iid not in flat_by_iid:
                    flat_order.append(iid)
                flat_by_iid.setdefault(iid, []).append(d)
        for iid in flat_order:
            if iid in out:
                continue
            try:
                out[iid] = AuditedGoldRow.from_flat_rows(flat_by_iid[iid]).model_dump_json()
            except Exception as e:  # noqa: BLE001
                # Surface the drop in the merge report so a partial-override
                # rerun against a legacy file with unrecoverable / malformed
                # rows doesn't silently lose those tasks (override=True
                # rewrites the consolidated file from this map).
                report.errors += 1
                report.error_details.append(
                    f"{consolidated}: legacy flat row(s) for {iid} dropped "
                    f"during migration to grouped format: {e}"
                )
                continue
        return out

    def _read_incoming_rows(src: Path) -> "list[AuditedGoldRow]":
        """Parse all valid AuditedGoldRow entries from a per-task JSONL file.

        New grouped-format lines (``variants`` key present) are returned as-is.
        Legacy flat rows are grouped by ``instance_id`` first so all variants
        for the same task are consolidated into one ``AuditedGoldRow``.
        """
        rows: list[AuditedGoldRow] = []
        flat_by_iid: dict[str, list[dict]] = {}
        flat_order: list[str] = []
        for raw_line in src.read_text().splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError as e:
                report.errors += 1
                report.error_details.append(f"{src}: JSON decode error: {e}")
                continue
            if not isinstance(d, dict):
                report.errors += 1
                report.error_details.append(
                    f"{src}: expected JSON object, got {type(d).__name__}"
                )
                continue
            if "variants" in d:
                try:
                    rows.append(AuditedGoldRow.model_validate(d))
                except Exception as e:  # noqa: BLE001
                    report.errors += 1
                    report.error_details.append(
                        f"{src}: AuditedGoldRow validation error: {e}"
                    )
            else:
                iid = d.get("instance_id", "")
                if not iid:
                    report.errors += 1
                    report.error_details.append(
                        f"{src}: flat-row missing required field 'instance_id'"
                    )
                    continue
                if iid not in flat_by_iid:
                    flat_order.append(iid)
                flat_by_iid.setdefault(iid, []).append(d)
        for iid in flat_order:
            try:
                rows.append(AuditedGoldRow.from_flat_rows(flat_by_iid[iid]))
            except (ValueError, Exception) as e:  # noqa: BLE001
                report.errors += 1
                report.error_details.append(
                    f"{src}: from_flat_rows error for {iid}: {e}"
                )
        return rows

    if override:
        # Upsert mode: replace all rows for re-annotated instance_ids.
        existing_ordered = _read_consolidated()

        # Collect all valid incoming rows before touching existing_ordered so
        # that a validation failure on new data never silently drops old rows.
        incoming: dict[str, str] = {}  # instance_id → serialised AuditedGoldRow
        rerun_iids: set[str] = set()
        for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
            src = sub / "audited_gold_variants.jsonl"
            if not src.exists():
                continue
            rerun_iids.add(sub.name)
            for row in _read_incoming_rows(src):
                incoming[row.instance_id] = row.model_dump_json()

        # Purge only after all new rows are validated and ready.
        for iid in list(existing_ordered.keys()):
            if iid in rerun_iids:
                del existing_ordered[iid]

        for iid, serialised in incoming.items():
            existing_ordered[iid] = serialised
            report.added += 1

        if existing_ordered:
            consolidated.parent.mkdir(parents=True, exist_ok=True)
            consolidated.write_text(
                "\n".join(existing_ordered.values()) + "\n"
            )
        elif consolidated.exists():
            consolidated.write_text("")
        return report

    # Append-only mode (default): skip rows whose instance_id already exists.
    existing_ids: set[str] = set(_read_consolidated().keys())

    new_lines: list[str] = []
    for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
        src = sub / "audited_gold_variants.jsonl"
        if not src.exists():
            continue
        for row in _read_incoming_rows(src):
            if row.instance_id in existing_ids:
                report.skipped_duplicate += 1
                continue
            existing_ids.add(row.instance_id)
            new_lines.append(row.model_dump_json())
            report.added += 1

    if new_lines:
        consolidated.parent.mkdir(parents=True, exist_ok=True)
        if consolidated.exists() and consolidated.stat().st_size > 0:
            if not consolidated.read_bytes().endswith(b"\n"):
                with consolidated.open("a") as f:
                    f.write("\n")
        with consolidated.open("a") as f:
            for line in new_lines:
                f.write(line + "\n")

    return report
