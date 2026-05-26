"""DEV-1470: laptop-side merger that promotes per-DB cloud-encoded OTF
references from `<run_dir>/post_run/slayer_models_otf/<shard>/<db>/` into the
global warm cache at `paths.slayer_models_otf_root()/<db>/`.

Contract:

* For each `<db>` present under any shard, build a per-file pick map by taking
  the entry with the max `source_mtime` across shards (read from each shard's
  `_source_mtimes.json` sidecar). Shards missing `_upload_complete` or
  `_source_mtimes.json` are IGNORED.
* For each `(rel_path, source_mtime)`: if `source_mtime > target.stat().st_mtime`
  (0 if absent), atomically replace the local file via `tmp + os.replace` in
  the parent dir.
* Inside a per-DB `fcntl.flock` lock on `<reference_root>/<db>.merge.lock`:
  unlink local `_reference_fp.txt` FIRST (so any concurrent reader sees
  "marker absent ⇒ incomplete, rebuild"), then per-file merge, then write
  `_reference_fp.txt` LAST. Preserves the on-disk invariant "marker present ⇒
  content complete" at every moment a reader could observe.
* Returns `{"merged_dbs": [{"db": ..., "files_updated": N,
  "files_skipped": M}], "ignored_shards": [...]}`.

Coherence note: the merger may take files from different shards in a single
DB. This is safe because `cache.fingerprint_of(db, mini_interact_root)`
depends only on the input dataset (static during a run), so all shards in one
run produce the same fingerprint. Cross-run merging never happens because each
`fetch` lands its shards in its own `run_dir`.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from bird_interact_agents.cloud import post_run_merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shard(
    run_dir: Path, shard: str, db: str, files: dict[str, tuple[bytes, float]],
    *, write_complete: bool = True, write_sidecar: bool = True,
) -> Path:
    """Lay down `run_dir/post_run/slayer_models_otf/<shard>/<db>/` with the
    listed files (rel_path -> (content_bytes, source_mtime)), the sidecar,
    and (optionally) the `_upload_complete` marker."""
    base = run_dir / "post_run" / "slayer_models_otf" / shard / db
    base.mkdir(parents=True, exist_ok=True)
    mtimes: dict[str, float] = {}
    for rel, (content, mtime) in files.items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        os.utime(target, (mtime, mtime))
        mtimes[rel] = mtime
    if write_sidecar:
        (base / "_source_mtimes.json").write_text(json.dumps(mtimes))
    if write_complete:
        (base / "_upload_complete").write_text("ok")
    return base


def _make_local(
    reference_root: Path, db: str, files: dict[str, tuple[bytes, float]],
) -> Path:
    """Lay down `reference_root/<db>/` with the listed files."""
    base = reference_root / db
    base.mkdir(parents=True, exist_ok=True)
    for rel, (content, mtime) in files.items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        os.utime(target, (mtime, mtime))
    return base


# ---------------------------------------------------------------------------
# Newest-source-mtime-wins per file
# ---------------------------------------------------------------------------


def test_newer_cloud_overwrites_older_local(tmp_path: Path):
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_local(ref_root, "db_a", {
        "models/x.yaml": (b"name: local-old\n", 1000.0),
        "_reference_fp.txt": (b"fp-local-old", 1000.0),
    })
    _make_shard(run_dir, "host-1", "db_a", {
        "models/x.yaml": (b"name: cloud-new\n", 2000.0),
        "_reference_fp.txt": (b"fp-cloud-new", 2000.0),
    })

    report = post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )

    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"name: cloud-new\n"
    assert (ref_root / "db_a" / "_reference_fp.txt").read_bytes() == b"fp-cloud-new"
    # M4 — report shape: top-level keys + per-db {db, files_updated, files_skipped}.
    assert set(report.keys()) >= {"merged_dbs", "ignored_shards"}
    [merged] = report["merged_dbs"]
    assert set(merged.keys()) == {"db", "files_updated", "files_skipped"}, (
        f"per-db report shape must be {{'db', 'files_updated', 'files_skipped'}}; "
        f"got {sorted(merged.keys())}"
    )
    assert merged["db"] == "db_a"
    assert merged["files_updated"] == 2, (
        f"both x.yaml and _reference_fp.txt should be updated, got {merged}"
    )
    assert merged["files_skipped"] == 0


def test_older_cloud_does_NOT_overwrite_newer_local(tmp_path: Path):
    """If the local file has been modified more recently (e.g. the user
    re-encoded locally after the cloud finished), the merger must leave it
    alone. M4 — `files_skipped` counts these picks."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_local(ref_root, "db_a", {
        "models/x.yaml": (b"name: local-NEWER\n", 5000.0),
    })
    _make_shard(run_dir, "host-1", "db_a", {
        "models/x.yaml": (b"name: cloud-older\n", 2000.0),
    })

    report = post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )
    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"name: local-NEWER\n"
    [merged] = report["merged_dbs"]
    assert merged["files_updated"] == 0
    assert merged["files_skipped"] >= 1


def test_db_absent_in_shards_is_left_untouched(tmp_path: Path):
    """A db present locally but with NO shard upload at all must not be
    touched (no marker unlink, no file writes)."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_local(ref_root, "db_untouched", {
        "models/keep.yaml": (b"name: keep\n", 1000.0),
        "_reference_fp.txt": (b"fp-local", 1000.0),
    })
    # Shard for a DIFFERENT db, so post_run/ exists but db_untouched isn't there.
    _make_shard(run_dir, "host-1", "other_db", {
        "models/x.yaml": (b"x\n", 2000.0),
    })

    post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )
    assert (ref_root / "db_untouched" / "_reference_fp.txt").read_bytes() == b"fp-local"
    assert (ref_root / "db_untouched" / "models" / "keep.yaml").read_bytes() == b"name: keep\n"


def test_multishard_picks_newest_per_file(tmp_path: Path):
    """When two shards have different files (or the same file with different
    mtimes), the per-file pick is the max source_mtime across shards."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"

    _make_shard(run_dir, "host-A", "db_a", {
        "models/x.yaml": (b"from-A\n", 2000.0),
        "models/y.yaml": (b"y-A-old\n", 1500.0),
        "_reference_fp.txt": (b"fp", 2000.0),
    })
    _make_shard(run_dir, "host-B", "db_a", {
        "models/y.yaml": (b"y-B-newer\n", 2500.0),
        "_reference_fp.txt": (b"fp", 2200.0),
    })

    post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )

    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"from-A\n"
    assert (ref_root / "db_a" / "models" / "y.yaml").read_bytes() == b"y-B-newer\n"


# ---------------------------------------------------------------------------
# Marker invariant (H1): marker is written LAST + deleted FIRST when local present
# ---------------------------------------------------------------------------


def test_existing_marker_unlinked_before_content_merge(tmp_path: Path):
    """H1 — if local `<db>/` already has `_reference_fp.txt`, the merger must
    unlink it BEFORE touching any other file, so a concurrent reader cannot
    observe "marker present + partially-updated content". The marker is then
    written LAST after all content is in place."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_local(ref_root, "db_a", {
        "models/x.yaml": (b"local-old\n", 1000.0),
        "_reference_fp.txt": (b"fp-local-old", 1000.0),
    })
    _make_shard(run_dir, "host-1", "db_a", {
        "models/x.yaml": (b"cloud-new\n", 2000.0),
        "_reference_fp.txt": (b"fp-cloud-new", 2000.0),
    })

    # Spy on os.replace to capture ordering.
    real_replace = os.replace
    order: list[tuple[str, str]] = []

    def spy_replace(src, dst):
        order.append(("replace", str(dst)))
        real_replace(src, dst)

    real_unlink = os.unlink

    def spy_unlink(p, *a, **kw):
        order.append(("unlink", str(p)))
        real_unlink(p, *a, **kw)

    import builtins  # ensure module exists; no-op
    _ = builtins  # silence linter
    monkeypatch_replace = pytest.MonkeyPatch()
    monkeypatch_replace.setattr(os, "replace", spy_replace)
    monkeypatch_replace.setattr(os, "unlink", spy_unlink)
    try:
        post_run_merge.merge_post_run_into_warm_cache(
            run_dir=run_dir, reference_root=ref_root,
        )
    finally:
        monkeypatch_replace.undo()

    marker_str = str(ref_root / "db_a" / "_reference_fp.txt")
    content_str = str(ref_root / "db_a" / "models" / "x.yaml")

    # Find the first unlink/replace of the local marker, and the writes of
    # content + marker.
    unlink_marker_idx = next(
        (i for i, (kind, p) in enumerate(order)
         if kind == "unlink" and p == marker_str),
        None,
    )
    assert unlink_marker_idx is not None, (
        f"local marker must be unlinked before content merge; saw {order}"
    )

    content_write_idx = next(
        i for i, (kind, p) in enumerate(order)
        if kind == "replace" and p == content_str
    )
    # M3 — the marker must also be written via atomic `os.replace`
    # (tmp + rename), not `open(target, 'w').write(...)` in place. The
    # invariant "marker present ⇒ content complete" is much weaker if a
    # reader can observe a half-written marker file.
    marker_write_idx = next(
        (i for i, (kind, p) in enumerate(order)
         if kind == "replace" and p == marker_str),
        None,
    )
    assert marker_write_idx is not None, (
        f"`_reference_fp.txt` must be written via atomic `os.replace`, not "
        f"open()/write(); saw {order}"
    )
    assert unlink_marker_idx < content_write_idx < marker_write_idx, (
        f"order must be: unlink-marker → write-content → write-marker; "
        f"saw {order}"
    )


def test_no_local_marker_means_no_unlink(tmp_path: Path):
    """If local `<db>/` doesn't exist or has no marker, the merger doesn't
    have a marker to unlink — it just writes everything in place + marker
    last."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"  # absent
    _make_shard(run_dir, "host-1", "db_a", {
        "models/x.yaml": (b"new\n", 2000.0),
        "_reference_fp.txt": (b"fp-cloud", 2000.0),
    })
    post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )
    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"new\n"
    assert (ref_root / "db_a" / "_reference_fp.txt").read_bytes() == b"fp-cloud"


# ---------------------------------------------------------------------------
# Shard completeness (M1): missing _upload_complete or sidecar => ignore
# ---------------------------------------------------------------------------


def test_shard_missing_upload_complete_is_ignored(tmp_path: Path):
    """A shard without `_upload_complete` is in-progress (upload still
    happening or crashed mid-upload). Merger must skip it."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_shard(run_dir, "host-incomplete", "db_a", {
        "models/x.yaml": (b"incomplete\n", 2000.0),
        "_reference_fp.txt": (b"fp", 2000.0),
    }, write_complete=False)

    report = post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )
    assert not (ref_root / "db_a").exists(), (
        "in-progress shard must not be promoted to the warm cache"
    )
    assert "host-incomplete" in report["ignored_shards"]


def test_shard_missing_sidecar_is_ignored(tmp_path: Path):
    """A shard missing `_source_mtimes.json` cannot inform mtime-wins picks;
    merger ignores it (records under ignored_shards)."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_shard(run_dir, "host-nosidecar", "db_a", {
        "models/x.yaml": (b"x\n", 2000.0),
        "_reference_fp.txt": (b"fp", 2000.0),
    }, write_sidecar=False)

    report = post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )
    assert not (ref_root / "db_a").exists()
    assert "host-nosidecar" in report["ignored_shards"]


def test_mixed_complete_and_incomplete_shards(tmp_path: Path):
    """A complete shard alongside an incomplete one: only the complete one
    contributes to the merge."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_shard(run_dir, "host-ok", "db_a", {
        "models/x.yaml": (b"from-ok\n", 2000.0),
        "_reference_fp.txt": (b"fp", 2000.0),
    })
    _make_shard(run_dir, "host-bad", "db_a", {
        "models/x.yaml": (b"from-bad-LATER\n", 3000.0),  # newer but incomplete
        "_reference_fp.txt": (b"fp", 3000.0),
    }, write_complete=False)

    post_run_merge.merge_post_run_into_warm_cache(
        run_dir=run_dir, reference_root=ref_root,
    )
    # `host-bad` had a newer mtime but no `_upload_complete` => ignored.
    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"from-ok\n"


# ---------------------------------------------------------------------------
# Atomicity (per-file): a crashed copy leaves no half-written target
# ---------------------------------------------------------------------------


def test_per_file_atomic_replace_leaves_no_partial(tmp_path: Path):
    """The per-file write must be `tmp + os.replace` in the parent dir, never
    `open(target, 'w').write(...)` in place. If the merger crashes mid-write,
    `target` must still hold the OLD bytes (or be absent) — never partial."""
    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_local(ref_root, "db_a", {
        "models/x.yaml": (b"OLD-COMPLETE\n", 1000.0),
    })
    _make_shard(run_dir, "host-1", "db_a", {
        "models/x.yaml": (b"NEW-CONTENT\n", 2000.0),
    })

    real_replace = os.replace
    call_count = {"n": 0}

    def crash_on_first_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate crash by raising — the tmp file at `src` already holds
            # the new bytes, but the rename never happens, so `dst` stays at
            # its previous content.
            raise RuntimeError("simulated crash mid-merge")
        real_replace(src, dst)

    mp = pytest.MonkeyPatch()
    mp.setattr(os, "replace", crash_on_first_replace)
    try:
        with pytest.raises(RuntimeError):
            post_run_merge.merge_post_run_into_warm_cache(
                run_dir=run_dir, reference_root=ref_root,
            )
    finally:
        mp.undo()

    # The OLD file must still be intact (atomic replace never happened).
    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"OLD-COMPLETE\n"


# ---------------------------------------------------------------------------
# Per-DB cross-process file lock (H1, M5)
# ---------------------------------------------------------------------------


def _spawn_merge_holding_lock(args):
    """Worker that acquires the per-DB build lock (shared with the merger
    under the H4 resolution) and holds it for `hold_s`."""
    run_dir, reference_root, db, hold_s, sentinel_path = args
    import fcntl

    # H4 — merger and `_build_reference` share the SAME per-DB lock file
    # (`<reference_root>/<db>.build.lock`) so an in-flight encoder cannot
    # interleave with a concurrent fetch's merge.
    lock_path = reference_root / f"{db}.build.lock"
    reference_root.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        Path(sentinel_path).write_text("locked")
        time.sleep(hold_s)
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def test_merger_blocks_on_per_db_file_lock(tmp_path: Path):
    """H1+H4 — two concurrent calls to `merge_post_run_into_warm_cache` on
    the same DB (and likewise a merge racing with `_build_reference`) must
    serialize via the shared per-DB `fcntl.flock` on
    `<reference_root>/<db>.build.lock`."""
    import fcntl  # noqa: F401 — fail loudly if unavailable

    run_dir = tmp_path / "run"
    ref_root = tmp_path / "ref"
    _make_shard(run_dir, "host-1", "db_a", {
        "models/x.yaml": (b"x\n", 2000.0),
        "_reference_fp.txt": (b"fp", 2000.0),
    })

    sentinel = tmp_path / "holder_acquired.txt"
    # Background process: take the merge lock and hold it for ~2s.
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_spawn_merge_holding_lock,
        args=((run_dir, ref_root, "db_a", 2.0, str(sentinel)),),
    )
    p.start()
    try:
        # Wait for the holder to acquire the lock.
        deadline = time.time() + 5.0
        while not sentinel.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert sentinel.exists(), "holder failed to acquire the lock"

        t0 = time.time()
        post_run_merge.merge_post_run_into_warm_cache(
            run_dir=run_dir, reference_root=ref_root,
        )
        elapsed = time.time() - t0
        # The merger should have BLOCKED on flock until the holder released
        # (≈2s). Anything under 0.5s means no lock was taken.
        assert elapsed >= 0.5, (
            f"merger did not block on the per-DB lock (elapsed={elapsed:.3f}s)"
        )
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()


# ---------------------------------------------------------------------------
# Module docstring records the fingerprint-coherence invariant (L1)
# ---------------------------------------------------------------------------


def test_module_docstring_documents_fingerprint_invariant():
    """L1 — cross-shard per-file picks are safe because all shards in one run
    have the same fingerprint (the dataset is static during a run). The
    module docstring must state this so a future reader doesn't 'fix' it."""
    doc = post_run_merge.__doc__ or ""
    assert "fingerprint" in doc.lower() and ("coherent" in doc.lower() or "static" in doc.lower()), (
        "post_run_merge module docstring must document the single-fingerprint-"
        "per-run invariant that makes per-file mtime-wins safe across shards"
    )
