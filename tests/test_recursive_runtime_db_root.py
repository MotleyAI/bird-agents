"""DEV-1462 — Codex (third /process-reviews round): the recursive
on-the-fly path must thread an authoritative ``db_root`` so the harness's
``--db-path`` wins over ``$BIRD_DB_PATH``.

The otf_encode adapter already does this (``ensure_db_reference`` /
``build_task_variant_storage`` accept ``db_root``). The recursive adapter
computes the right root (``mini_interact_root = Path(data_path_base).resolve()``
in ``_resolve_otf_task_storage_dir``) but handed it to
``prepare_task_storage`` only as the *fallback positional* — never as
``db_root`` — so ``runtime._rewrite_datasource_connection_string`` re-anchored
via ``reanchor_connection_string`` with no ``db_root``, letting
``$BIRD_DB_PATH`` (set to mini-interact by conftest / dev shells) win. For a
LiveSQLBench DB whose name collides with a mini-interact DB (``alien``), that
silently re-anchors the per-task SLayer datasource to the WRONG sqlite.

These tests pin the threading: ``prepare_task_storage`` +
``_rewrite_datasource_connection_string`` accept ``db_root``, forward it into
``reanchor_connection_string``, and the recursive agent resolver passes its
resolved ``data_path_base`` as ``db_root``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from slayer.core.models import DatasourceConfig
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf import CacheEntry, prepare_task_storage
from bird_interact_agents.slayer_otf import runtime as runtime_mod


def test_prepare_task_storage_signature_accepts_db_root():
    """Pin the kwarg via signature inspection so it can't silently vanish."""
    sig = inspect.signature(prepare_task_storage)
    assert "db_root" in sig.parameters, (
        f"prepare_task_storage must carry a `db_root` kwarg; "
        f"got params: {list(sig.parameters)!r}"
    )


def test_rewrite_signature_accepts_db_root():
    sig = inspect.signature(runtime_mod._rewrite_datasource_connection_string)
    assert "db_root" in sig.parameters, (
        f"_rewrite_datasource_connection_string must carry a `db_root` kwarg; "
        f"got params: {list(sig.parameters)!r}"
    )


async def _save_datasource(base: Path, db: str, connection_string: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    storage = YAMLStorage(base_dir=str(base))
    await storage.save_datasource(
        DatasourceConfig(name=db, connection_string=connection_string)
    )


@pytest.mark.asyncio
async def test_rewrite_db_root_overrides_env(monkeypatch, tmp_path):
    """``_rewrite_datasource_connection_string`` with an explicit ``db_root``
    must re-anchor at ``db_root`` even when ``$BIRD_DB_PATH`` points
    elsewhere — the LiveSQLBench-vs-mini-interact collision case."""
    env_root = tmp_path / "mini_interact_env"
    env_root.mkdir()
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))

    db_root = tmp_path / "livesqlbench"
    db_root.mkdir()

    scratch = tmp_path / "scratch"
    # A stale FOREIGN-ABSOLUTE connection string (what the OTF cache bakes
    # in) so reanchor force-rewrites to <root>/<db>/<db>.sqlite.
    await _save_datasource(
        scratch, "alien",
        "sqlite:////foreign_machine/data/alien/alien.sqlite",
    )

    await runtime_mod._rewrite_datasource_connection_string(
        db="alien", scratch=scratch, mini_interact_root=db_root,
        db_root=db_root,
    )

    resolved = (await YAMLStorage(base_dir=str(scratch)).get_datasource("alien"))
    assert resolved is not None
    cs = resolved.connection_string or ""
    assert str(db_root) in cs, (
        f"db_root must win over $BIRD_DB_PATH; conn {cs!r} omits {db_root!r}"
    )
    assert str(env_root) not in cs, (
        f"db_root must win over $BIRD_DB_PATH; conn {cs!r} leaked env {env_root!r}"
    )


@pytest.mark.asyncio
async def test_prepare_task_storage_threads_db_root_into_reanchor(
    monkeypatch, tmp_path,
):
    """End-to-end through the public ``prepare_task_storage``: the ``db_root``
    kwarg must reach ``reanchor_connection_string`` (spied) AND the resulting
    per-task datasource must anchor at ``db_root``, not ``$BIRD_DB_PATH``."""
    env_root = tmp_path / "mini_interact_env"
    env_root.mkdir()
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))

    db_root = tmp_path / "livesqlbench"
    db_root.mkdir()

    # Minimal cache dir holding just the datasource — the no-deletion path
    # (deleted_kb_ids=set()) only copytrees + re-anchors, leaving
    # memories/embeddings untouched.
    cache_dir = tmp_path / "cache" / "alien"
    await _save_datasource(
        cache_dir, "alien",
        "sqlite:////foreign_machine/data/alien/alien.sqlite",
    )
    entry = CacheEntry(cache_dir=cache_dir, fingerprint="dead", kb_rows=[])

    seen: list = []
    real = runtime_mod.reanchor_connection_string

    def spy(cs, db, mini, *, db_root=None):
        seen.append(db_root)
        return real(cs, db, mini, db_root=db_root)

    monkeypatch.setattr(runtime_mod, "reanchor_connection_string", spy)

    scratch = await prepare_task_storage(
        db="alien", deleted_kb_ids=set(), cache_entry=entry,
        work_dir=tmp_path / "work", mini_interact_root=db_root,
        db_root=db_root,
    )

    assert seen == [db_root], (
        f"db_root not threaded into reanchor; saw {seen!r}, expected [{db_root!r}]"
    )
    resolved = await YAMLStorage(base_dir=str(scratch)).get_datasource("alien")
    assert resolved is not None
    cs = resolved.connection_string or ""
    assert str(db_root) in cs and str(env_root) not in cs


@pytest.mark.asyncio
async def test_recursive_resolver_passes_resolved_db_path_as_db_root(
    monkeypatch, tmp_path,
):
    """The recursive agent's ``_resolve_otf_task_storage_dir`` must pass
    ``db_root = Path(data_path_base).resolve()`` into ``prepare_task_storage``,
    mirroring the otf_encode adapter."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as rec_agent

    data_path_base = str(tmp_path / "livesqlbench")
    (tmp_path / "livesqlbench").mkdir()

    fake_entry = CacheEntry(
        cache_dir=tmp_path / "cache", fingerprint="x", kb_rows=[],
    )

    async def fake_ensure_db_cache(db, *, cache_root, mini_interact_root, force=False):
        return fake_entry

    captured: dict = {}

    async def fake_prepare_task_storage(*, db, deleted_kb_ids, cache_entry,
                                        work_dir, mini_interact_root, db_root=None):
        captured["db_root"] = db_root
        captured["mini_interact_root"] = mini_interact_root
        return tmp_path / "scratch"

    monkeypatch.setattr(rec_agent, "ensure_db_cache", fake_ensure_db_cache)
    monkeypatch.setattr(rec_agent, "prepare_task_storage", fake_prepare_task_storage)

    await rec_agent._resolve_otf_task_storage_dir(
        db_name="alien",
        task_data={"instance_id": "alien_1"},
        data_path_base=data_path_base,
        benchmark="livesqlbench",
    )

    assert captured["db_root"] == Path(data_path_base).resolve(), (
        f"recursive resolver must pass db_root=resolved(data_path_base); "
        f"got {captured.get('db_root')!r}"
    )
