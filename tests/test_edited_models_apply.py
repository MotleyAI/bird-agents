"""DEV-1649: apply-side wiring — ``apply_or_none`` and the resolver routing.

The resolver (both the shared ``slayer_otf.runtime.resolve_otf_task_storage_dir``
AND the pydantic recursive private ``_resolve_otf_task_storage_dir``) applies a
saved store when present+valid, else falls back to the fresh OTF cache. These
tests drive a REAL saved archive through the REAL ``apply_or_none`` (only the
cache builders are stubbed), so they don't depend on the resolver's internal
dispatch shape.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bird_interact_agents.slayer_otf import edited_models as em
from tests._edited_models_fixtures import edit_a_model, make_fake_store


@pytest.fixture()
def checkout(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod

    co = tmp_path / "checkout"
    co.mkdir()
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: co)
    monkeypatch.delenv("BIRD_RUNS_ROOT", raising=False)
    return co


def _save_store(tmp_path, *, deleted=frozenset(), cache_fp="fp0"):
    work = tmp_path / "save_wd"
    scratch = make_fake_store(work)
    em.write_baseline_manifest(work, scratch)
    edit_a_model(scratch)
    return em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids=set(deleted), cache_fp=cache_fp,
    )


# --------------------------------------------------------------------------
# apply_or_none (pure)
# --------------------------------------------------------------------------


def test_apply_or_none_returns_none_without_archive(tmp_path, checkout):
    out = asyncio.run(
        em.apply_or_none(
            benchmark="mini-interact", db="alien", instance_id="alien_1",
            work_dir=tmp_path / "wd", task_deleted_kb_ids=set(),
            current_cache_fp="fp0", mini_interact_root=tmp_path, db_root=tmp_path,
        )
    )
    assert out is None


def _is_sha256_hex(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def test_apply_or_none_materializes_when_valid(tmp_path, checkout):
    _save_store(tmp_path, cache_fp="fp0")
    work2 = tmp_path / "wd"
    work2.mkdir()
    out = asyncio.run(
        em.apply_or_none(
            benchmark="mini-interact", db="alien", instance_id="alien_1",
            work_dir=work2, task_deleted_kb_ids=set(),
            current_cache_fp="fp0", mini_interact_root=tmp_path, db_root=tmp_path,
        )
    )
    assert out is not None
    # DEV-1778: apply_or_none now returns an AppliedStore(scratch, store_fp).
    assert (Path(out.scratch) / "models" / "alien" / "foo.yaml").is_file()
    assert _is_sha256_hex(out.store_fp)


def test_apply_or_none_falls_back_on_stale(tmp_path, checkout):
    _save_store(tmp_path, cache_fp="fp0")
    out = asyncio.run(
        em.apply_or_none(
            benchmark="mini-interact", db="alien", instance_id="alien_1",
            work_dir=tmp_path / "wd", task_deleted_kb_ids=set(),
            current_cache_fp="fp-STALE", mini_interact_root=tmp_path, db_root=tmp_path,
        )
    )
    assert out is None


# --------------------------------------------------------------------------
# resolver routing — shared runtime resolver
# --------------------------------------------------------------------------


def _stub_cache_builders(monkeypatch, module, tmp_path):
    """Stub ensure_db_cache + prepare_task_storage on `module` so the resolver
    runs without real data. Returns the sentinel cache-scratch path that
    prepare_task_storage yields (the fallback target)."""
    from bird_interact_agents.slayer_otf.cache import CacheEntry

    cache_dir = tmp_path / "cache" / "alien"
    cache_dir.mkdir(parents=True)
    (cache_dir / "_cache_fp.txt").write_text("fpCACHE")

    async def fake_ensure_db_cache(db, **kwargs):
        return CacheEntry(cache_dir=cache_dir, fingerprint="fpCACHE", kb_rows=[])

    cache_scratch = tmp_path / "cache_scratch" / "alien"
    cache_scratch.mkdir(parents=True)

    async def fake_prepare(**kwargs):
        return cache_scratch

    monkeypatch.setattr(module, "ensure_db_cache", fake_ensure_db_cache)
    monkeypatch.setattr(module, "prepare_task_storage", fake_prepare)
    return cache_scratch


def _task_data():
    return {"instance_id": "alien_1", "deleted_knowledge": []}


def test_runtime_resolver_uses_saved_store(monkeypatch, tmp_path, checkout):
    from bird_interact_agents.slayer_otf import runtime

    _stub_cache_builders(monkeypatch, runtime, tmp_path)
    archive = _save_store(tmp_path, cache_fp="fpCACHE")  # matches cache fp
    td = _task_data()

    storage, _deleted = asyncio.run(
        runtime.resolve_otf_task_storage_dir(
            db_name="alien", task_data=td,
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=True,
        )
    )
    # Applied from the saved store (not the cache sentinel).
    assert (Path(storage) / "models" / "alien" / "foo.yaml").is_file()
    assert td.get("_edited_models_applied_from") == str(archive)
    # DEV-1778: the consumed-store fingerprint is stashed for the finalize hook.
    assert _is_sha256_hex(td.get("_edited_models_consumed_store_fp"))


def test_runtime_resolver_falls_back_when_stale(monkeypatch, tmp_path, checkout):
    from bird_interact_agents.slayer_otf import runtime

    cache_scratch = _stub_cache_builders(monkeypatch, runtime, tmp_path)
    _save_store(tmp_path, cache_fp="fpOLD")  # mismatches cache fp -> rejected
    td = _task_data()

    storage, _ = asyncio.run(
        runtime.resolve_otf_task_storage_dir(
            db_name="alien", task_data=td,
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=True,
        )
    )
    assert Path(storage) == cache_scratch
    assert not td.get("_edited_models_applied_from")


def test_runtime_resolver_falls_back_when_absent(monkeypatch, tmp_path, checkout):
    from bird_interact_agents.slayer_otf import runtime

    cache_scratch = _stub_cache_builders(monkeypatch, runtime, tmp_path)
    storage, _ = asyncio.run(
        runtime.resolve_otf_task_storage_dir(
            db_name="alien", task_data=_task_data(),
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=True,
        )
    )
    assert Path(storage) == cache_scratch


def test_runtime_resolver_ignores_apply_when_flag_off(monkeypatch, tmp_path, checkout):
    from bird_interact_agents.slayer_otf import runtime

    cache_scratch = _stub_cache_builders(monkeypatch, runtime, tmp_path)
    _save_store(tmp_path, cache_fp="fpCACHE")  # present + valid, but flag off
    td = _task_data()
    storage, _ = asyncio.run(
        runtime.resolve_otf_task_storage_dir(
            db_name="alien", task_data=td,
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=False,
        )
    )
    assert Path(storage) == cache_scratch
    assert not td.get("_edited_models_applied_from")


# --------------------------------------------------------------------------
# resolver routing — pydantic recursive private resolver (same contract)
# --------------------------------------------------------------------------


def test_recursive_resolver_uses_saved_store(monkeypatch, tmp_path, checkout):
    import bird_interact_agents.agents.pydantic_ai_recursive.agent as pkg

    _stub_cache_builders(monkeypatch, pkg, tmp_path)
    archive = _save_store(tmp_path, cache_fp="fpCACHE")
    td = _task_data()

    storage, _ = asyncio.run(
        pkg._resolve_otf_task_storage_dir(
            db_name="alien", task_data=td,
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=True,
        )
    )
    assert (Path(storage) / "models" / "alien" / "foo.yaml").is_file()
    assert td.get("_edited_models_applied_from") == str(archive)
    assert _is_sha256_hex(td.get("_edited_models_consumed_store_fp"))


def test_recursive_resolver_falls_back_when_absent(monkeypatch, tmp_path, checkout):
    import bird_interact_agents.agents.pydantic_ai_recursive.agent as pkg

    cache_scratch = _stub_cache_builders(monkeypatch, pkg, tmp_path)
    storage, _ = asyncio.run(
        pkg._resolve_otf_task_storage_dir(
            db_name="alien", task_data=_task_data(),
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=True,
        )
    )
    assert Path(storage) == cache_scratch


# --------------------------------------------------------------------------
# DEV-1778: consumed-store fingerprint bound to the CONSUMED snapshot
# --------------------------------------------------------------------------


def _save_variant(tmp_path, tag, *, model_body):
    from tests._edited_models_fixtures import make_fake_store

    work = tmp_path / f"save_{tag}"
    scratch = make_fake_store(work, model_body=model_body)
    # No baseline written -> scratch_changed() is True -> the store is archived
    # with EXACTLY this content (so the applied fp reflects `model_body`).
    return em.save_edited_store(
        benchmark="mini-interact", db="alien", instance_id="alien_1",
        work_dir=work, scratch=scratch, deleted_kb_ids=set(), cache_fp="fp0",
    )


def _apply(tmp_path, wd):
    return asyncio.run(
        em.apply_or_none(
            benchmark="mini-interact", db="alien", instance_id="alien_1",
            work_dir=tmp_path / wd, task_deleted_kb_ids=set(),
            current_cache_fp="fp0", mini_interact_root=tmp_path, db_root=tmp_path,
        )
    )


def test_apply_fp_deterministic_for_identical_content(tmp_path, checkout):
    """Two independent saves of byte-identical store content (distinct gzip
    archives) apply to the SAME store_fp — content-deterministic, not
    archive-byte-dependent (Codex #1/#2)."""
    _save_variant(tmp_path, "a", model_body="name: foo\ncols: [x]\n")
    fp1 = _apply(tmp_path, "wd1").store_fp
    _save_variant(tmp_path, "b", model_body="name: foo\ncols: [x]\n")  # identical content
    fp2 = _apply(tmp_path, "wd2").store_fp
    assert fp1 == fp2


def test_apply_fp_reflects_consumed_content(tmp_path, checkout):
    """The fp is computed from the extracted snapshot, so replacing the archive
    with DIFFERENT content yields a different fp on the next apply (the fp is
    bound to what was consumed, not to the mutable archive path)."""
    _save_variant(tmp_path, "orig", model_body="name: foo\ncols: [x]\n")
    fp_orig = _apply(tmp_path, "wd_orig").store_fp
    _save_variant(tmp_path, "changed", model_body="name: foo\ncols: [DIFFERENT]\n")
    fp_changed = _apply(tmp_path, "wd_changed").store_fp
    assert fp_orig != fp_changed


def test_apply_fp_is_captured_pre_reanchor(tmp_path, checkout):
    """Proves the stamped fp is the CONSUMED archive's content (pre-re-anchor):
    it equals the archive's own content fingerprint, while the live scratch —
    whose datasource was re-anchored during apply — fingerprints differently."""
    import tarfile

    _save_variant(tmp_path, "x", model_body="name: foo\ncols: [x]\n")
    out = _apply(tmp_path, "wd")
    archive = em.run_edited_models_archive(
        benchmark="mini-interact", selected_database="alien", instance_id="alien_1",
    )
    extract = tmp_path / "extract"
    with tarfile.open(archive, "r:gz") as tf:
        # Mirror materialize_from_saved_store: filter="data" only exists on
        # 3.11.4+/3.12+; fall back to the member-validating extractor otherwise.
        if hasattr(tarfile, "data_filter"):
            tf.extractall(extract, filter="data")
        else:  # pragma: no cover - only on pre-3.11.4 interpreters
            em._safe_extractall(tf, extract)
    assert out.store_fp == em.store_content_fingerprint(extract / "alien")
    assert em.store_content_fingerprint(Path(out.scratch)) != out.store_fp


def test_apply_fingerprint_failure_degrades_to_none_and_logs(tmp_path, checkout, monkeypatch):
    """A fingerprint failure must NOT abort apply — it degrades to
    ``store_fp=None`` and logs the failure (Codex #3)."""
    _save_variant(tmp_path, "x", model_body="name: foo\ncols: [x]\n")
    events: list[str] = []

    def _boom(_root):
        raise RuntimeError("fingerprint boom")

    monkeypatch.setattr(em, "store_content_fingerprint", _boom)
    monkeypatch.setattr(em, "log_otf_event", lambda name, **kw: events.append(name))

    out = _apply(tmp_path, "wd")
    assert out is not None            # apply still succeeded
    assert out.store_fp is None
    assert "otf.edited_models.fingerprint_failed" in events
