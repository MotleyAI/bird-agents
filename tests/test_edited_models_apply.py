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
    assert (Path(out) / "models" / "alien" / "foo.yaml").is_file()


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

    storage, deleted = asyncio.run(
        runtime.resolve_otf_task_storage_dir(
            db_name="alien", task_data=td,
            data_path_base=str(tmp_path), benchmark="mini-interact",
            apply_edited_models=True,
        )
    )
    # Applied from the saved store (not the cache sentinel).
    assert (Path(storage) / "models" / "alien" / "foo.yaml").is_file()
    assert td.get("_edited_models_applied_from") == str(archive)


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
