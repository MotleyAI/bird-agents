"""DEV-1649: unit contracts for slayer_otf.edited_models — the save side and
the change-detection / manifest / meta primitives.

The full store is snapshotted as a single ``edited_models.tar.gz`` under the
runs/ golden store, keyed by (benchmark, db, instance_id), latest-wins
overwrite. Only written when the run succeeded AND the agent actually changed
the store (manifest diff vs the prepared baseline).
"""

from __future__ import annotations

import asyncio
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from bird_interact_agents.slayer_otf import edited_models as em
from tests._edited_models_fixtures import edit_a_model, make_fake_store


@pytest.fixture()
def checkout(monkeypatch, tmp_path):
    """Anchor runs_root at a tmp main-checkout."""
    import bird_interact_agents.paths as paths_mod

    co = tmp_path / "checkout"
    co.mkdir()
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: co)
    monkeypatch.delenv("BIRD_RUNS_ROOT", raising=False)
    return co


# --------------------------------------------------------------------------
# content_manifest
# --------------------------------------------------------------------------


def test_content_manifest_excludes_transient_and_meta(tmp_path):
    scratch = make_fake_store(tmp_path / "wd", with_wal=True)
    # Drop sidecar artefacts that must never count as content.
    (scratch / "embeddings.db-wal").write_bytes(b"wal")
    (scratch / "embeddings.db-shm").write_bytes(b"shm")
    (scratch.parent / em._BASELINE_MANIFEST).write_text("{}")
    (scratch / em._STORE_META).write_text("{}")

    manifest = em.content_manifest(scratch)
    keys = set(manifest)

    assert "models/alien/foo.yaml" in keys
    # DEV-1668: slayer 0.9.6 stores per-id ``memories/<id>.md``.
    assert "memories/alien_kb_0.md" in keys
    assert "_kb_rows.json" in keys
    assert "embeddings.db" in keys
    # Excluded:
    assert not any(k.endswith(".db-wal") or k.endswith(".db-shm") for k in keys)
    assert em._STORE_META not in keys
    assert em._BASELINE_MANIFEST not in keys


def test_content_manifest_is_deterministic(tmp_path):
    s1 = make_fake_store(tmp_path / "a")
    s2 = make_fake_store(tmp_path / "b")
    assert em.content_manifest(s1) == em.content_manifest(s2)


# --------------------------------------------------------------------------
# baseline / scratch_changed
# --------------------------------------------------------------------------


def test_scratch_unchanged_after_baseline(tmp_path):
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    assert em.scratch_changed(work, scratch) is False


def test_scratch_changed_when_model_edited(tmp_path):
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    assert em.scratch_changed(work, scratch) is True


def test_scratch_changed_when_baseline_missing(tmp_path):
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    # No baseline written -> conservative "changed".
    assert em.scratch_changed(work, scratch) is True


# --------------------------------------------------------------------------
# store_meta
# --------------------------------------------------------------------------


def test_store_meta_shape():
    meta = em.store_meta(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        deleted_kb_ids={3, 1}, cache_fp="fp0",
    )
    assert meta["benchmark"] == "mini-interact"
    assert meta["db"] == "alien"
    assert meta["instance_id"] == "alien_1"
    assert meta["deleted_kb_ids"] == [1, 3]  # sorted
    assert meta["cache_fp"] == "fp0"


# --------------------------------------------------------------------------
# save_edited_store
# --------------------------------------------------------------------------


def _archive_path(checkout, db="alien", iid="alien_1"):
    from bird_interact_agents.eval.annotation_io import run_edited_models_archive

    return run_edited_models_archive(
        benchmark="mini-interact", selected_database=db, instance_id=iid,
    )


def test_save_skips_when_unchanged(tmp_path, checkout):
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    dest = em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids=set(), cache_fp="fp0",
    )
    assert dest is None
    assert not _archive_path(checkout).exists()


def test_save_writes_archive_when_changed(tmp_path, checkout):
    work = tmp_path / "wd"
    scratch = make_fake_store(work, with_wal=True)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    # Simulate a WAL sidecar produced by SLayer writes.
    (scratch / "embeddings.db-wal").write_bytes(b"stale-wal")

    dest = em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids={2}, cache_fp="fp0",
    )
    assert dest == _archive_path(checkout)
    assert dest.is_file()

    with tarfile.open(dest, "r:gz") as tar:
        names = tar.getnames()
    # No loose *.json leaks are visible at the runs/ level, and transient
    # sidecars / the baseline file are excluded from the archive.
    assert not any(n.endswith(".db-wal") or n.endswith(".db-shm") for n in names)
    assert not any(n.endswith(em._BASELINE_MANIFEST) for n in names)
    assert any(n.endswith(em._STORE_META) for n in names)
    assert any(n.endswith("models/alien/foo.yaml") for n in names)


def test_saved_archive_meta_records_deleted_and_fp(tmp_path, checkout):
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    dest = em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids={5}, cache_fp="fpX",
    )
    with tarfile.open(dest, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(em._STORE_META))
        meta = json.loads(tar.extractfile(member).read())
    assert meta["deleted_kb_ids"] == [5]
    assert meta["cache_fp"] == "fpX"


def test_second_save_overwrites_atomically(tmp_path, checkout):
    for _ in range(2):
        work = tmp_path / f"wd{_}"
        scratch = make_fake_store(work)
        em.write_baseline_manifest(work, scratch)
        edit_a_model(scratch)
        em.save_edited_store(
            benchmark="mini-interact", db="alien", instance_id="alien_1",
            work_dir=work, scratch=scratch, deleted_kb_ids=set(), cache_fp="fp0",
        )
    dest = _archive_path(checkout)
    siblings = list(dest.parent.iterdir())
    # Exactly the one archive; no leftover .tmp-* dir/file.
    assert siblings == [dest], siblings


# --------------------------------------------------------------------------
# materialize_from_saved_store (apply mechanics)
# --------------------------------------------------------------------------


def _save_one(tmp_path, checkout, *, deleted, cache_fp):
    work = tmp_path / "save_wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    dest = em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids=deleted, cache_fp=cache_fp,
    )
    return dest, scratch


def test_materialize_round_trip_preserves_agent_edits(tmp_path, checkout):
    dest, orig = _save_one(tmp_path, checkout, deleted=set(), cache_fp="fp0")
    orig_model = (orig / "models" / "alien" / "foo.yaml").read_text()
    # DEV-1668: per-id ``memories/<id>.md`` (make_fake_store seeds alien_kb_0).
    orig_mem = (orig / "memories" / "alien_kb_0.md").read_text()

    work2 = tmp_path / "apply_wd"
    work2.mkdir()
    db_root = tmp_path / "dbroot"
    (db_root / "alien").mkdir(parents=True)
    out = asyncio.run(
        em.materialize_from_saved_store(
            db="alien", archive=dest, work_dir=work2,
            task_deleted_kb_ids=set(), current_cache_fp="fp0",
            mini_interact_root=db_root, db_root=db_root,
        )
    )
    assert out is not None
    # Agent-authored model + memories are preserved byte-for-byte (NOT
    # re-encoded from cache kb_rows — the clobber Codex #1 flagged).
    assert (out / "models" / "alien" / "foo.yaml").read_text() == orig_model
    assert (out / "memories" / "alien_kb_0.md").read_text() == orig_mem
    # And a fresh baseline is written so a no-op run is detected as unchanged.
    assert em.scratch_changed(work2, out) is False


def test_materialize_reanchors_connection_string(tmp_path, checkout):
    dest, _ = _save_one(tmp_path, checkout, deleted=set(), cache_fp="fp0")
    work2 = tmp_path / "apply_wd"
    work2.mkdir()
    db_root = tmp_path / "dbroot"
    (db_root / "alien").mkdir(parents=True)
    out = asyncio.run(
        em.materialize_from_saved_store(
            db="alien", archive=dest, work_dir=work2,
            task_deleted_kb_ids=set(), current_cache_fp="fp0",
            mini_interact_root=db_root, db_root=db_root,
        )
    )
    ds = yaml.safe_load((out / "datasources" / "alien.yaml").read_text())
    # Stale foreign absolute path re-rooted at the current db_root.
    assert str(db_root) in ds["connection_string"]
    assert "/nonexistent/build/machine/" not in ds["connection_string"]


def test_materialize_rejects_deleted_kb_mismatch(tmp_path, checkout):
    dest, _ = _save_one(tmp_path, checkout, deleted={3}, cache_fp="fp0")
    work2 = tmp_path / "apply_wd"
    work2.mkdir()
    out = asyncio.run(
        em.materialize_from_saved_store(
            db="alien", archive=dest, work_dir=work2,
            task_deleted_kb_ids={4},  # differs from saved {3}
            current_cache_fp="fp0",
            mini_interact_root=tmp_path, db_root=tmp_path,
        )
    )
    assert out is None


def test_materialize_rejects_cache_fp_mismatch(tmp_path, checkout):
    dest, _ = _save_one(tmp_path, checkout, deleted=set(), cache_fp="fp0")
    work2 = tmp_path / "apply_wd"
    work2.mkdir()
    out = asyncio.run(
        em.materialize_from_saved_store(
            db="alien", archive=dest, work_dir=work2,
            task_deleted_kb_ids=set(),
            current_cache_fp="fp-DIFFERENT",
            mini_interact_root=tmp_path, db_root=tmp_path,
        )
    )
    assert out is None


# --------------------------------------------------------------------------
# maybe_save_edited_models — the agent-facing gate
# --------------------------------------------------------------------------


def _prep_changed_scratch(tmp_path):
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    return work, scratch


def test_maybe_save_writes_and_stamps_on_success(tmp_path, checkout):
    work, scratch = _prep_changed_scratch(tmp_path)
    row = {"instance_id": "alien_1", "database": "alien", "phase1_passed": True}
    dest = em.maybe_save_edited_models(
        row, benchmark="mini-interact", save_edited_models=True,
        work_dir=work, slayer_storage_dir=str(scratch),
        deleted_kb_ids=set(), cache_fp="fp0",
    )
    assert dest is not None and dest.is_file()
    assert row["edited_models_saved_path"] == str(dest)


def test_maybe_save_noop_when_not_passed(tmp_path, checkout):
    work, scratch = _prep_changed_scratch(tmp_path)
    row = {"instance_id": "alien_1", "database": "alien", "phase1_passed": False}
    dest = em.maybe_save_edited_models(
        row, benchmark="mini-interact", save_edited_models=True,
        work_dir=work, slayer_storage_dir=str(scratch),
        deleted_kb_ids=set(), cache_fp="fp0",
    )
    assert dest is None
    assert not _archive_path(checkout).exists()


def test_maybe_save_noop_when_flag_off(tmp_path, checkout):
    work, scratch = _prep_changed_scratch(tmp_path)
    row = {"instance_id": "alien_1", "database": "alien", "phase1_passed": True}
    dest = em.maybe_save_edited_models(
        row, benchmark="mini-interact", save_edited_models=False,
        work_dir=work, slayer_storage_dir=str(scratch),
        deleted_kb_ids=set(), cache_fp="fp0",
    )
    assert dest is None


def test_maybe_save_noop_when_no_storage_dir(tmp_path, checkout):
    row = {"instance_id": "alien_1", "database": "alien", "phase1_passed": True}
    dest = em.maybe_save_edited_models(
        row, benchmark="mini-interact", save_edited_models=True,
        work_dir=None, slayer_storage_dir=None,
        deleted_kb_ids=set(), cache_fp="fp0",
    )
    assert dest is None


# --------------------------------------------------------------------------
# No-edit baseline nuance (D7 / §5.7): reanchor + mask done by
# prepare_task_storage must be captured in the baseline, NOT counted as edits.
# Also proves the fake store round-trips through the REAL YAMLStorage used by
# prepare_task_storage's _rewrite_datasource_connection_string.
# --------------------------------------------------------------------------


def test_baseline_captures_prepare_mutations(tmp_path):
    from bird_interact_agents.slayer_otf.cache import CacheEntry
    from bird_interact_agents.slayer_otf.runtime import prepare_task_storage

    cache_db = make_fake_store(tmp_path / "cache")  # cache/alien
    entry = CacheEntry(cache_dir=cache_db, fingerprint="fp0", kb_rows=[])
    work = tmp_path / "wd"
    work.mkdir()
    db_root = tmp_path / "dbroot"

    scratch = asyncio.run(
        prepare_task_storage(
            db="alien", deleted_kb_ids=set(), cache_entry=entry, work_dir=work,
            mini_interact_root=db_root, db_root=db_root,
        )
    )
    # The connection_string WAS re-anchored inside prepare (a real mutation),
    # but the baseline is captured after prepare, so no edit is detected.
    ds = yaml.safe_load((scratch / "datasources" / "alien.yaml").read_text())
    assert str(db_root) in ds["connection_string"]

    em.write_baseline_manifest(work, scratch)
    assert em.scratch_changed(work, scratch) is False


# --------------------------------------------------------------------------
# Embeddings self-containment: committed data survives into the archived
# embeddings.db; transient WAL/SHM sidecars are not packed (§5.6).
# --------------------------------------------------------------------------


def test_saved_archive_embeddings_selfcontained(tmp_path, checkout):
    import sqlite3
    import tarfile

    work = tmp_path / "wd"
    scratch = make_fake_store(work, with_wal=True)  # fixture inserts 1 row
    (scratch / "embeddings.db-wal").write_bytes(b"stale")
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    dest = em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids=set(), cache_fp="fp0",
    )
    ex = tmp_path / "ex"
    with tarfile.open(dest, "r:gz") as tar:
        assert not any(n.endswith(".db-wal") for n in tar.getnames())
        tar.extractall(ex, filter="data")
    con = sqlite3.connect(ex / "alien" / "embeddings.db")
    try:
        (n,) = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()
    finally:
        con.close()
    assert n == 1


def test_save_emits_otf_event(tmp_path, checkout, monkeypatch):
    events: list = []
    monkeypatch.setattr(
        em, "log_otf_event", lambda name, **kw: events.append(name), raising=True
    )
    work = tmp_path / "wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids=set(), cache_fp="fp0",
    )
    assert any("edited_models" in e for e in events)
