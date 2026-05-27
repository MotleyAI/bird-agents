"""DEV-1462 — Codex finding #2: the otf_encode reference build's
`_effective_db_root` lets `$BIRD_DB_PATH` override the passed root.

For LiveSQLBench runs that's a footgun — conftest sets `BIRD_DB_PATH`
to the mini-interact root, so an otf_encode livesqlbench resolve would
ingest the wrong sqlite unless the fix lands. The plan threads an
authoritative `db_root` through `ensure_db_reference` → `_effective_db_root`
that overrides the env.

These tests are pure-function on `_effective_db_root` + a thin integration
check that `ensure_db_reference` honours the threaded root.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_effective_db_root_with_explicit_root_overrides_env(monkeypatch, tmp_path):
    from bird_interact_agents.slayer_otf import reference_build

    monkeypatch.setenv("BIRD_DB_PATH", str(tmp_path / "mini_interact"))
    livesqlbench = tmp_path / "livesqlbench"
    livesqlbench.mkdir()
    out = reference_build._effective_db_root(
        livesqlbench, db_root=livesqlbench,
    )
    assert out == livesqlbench, (
        f"explicit db_root MUST override $BIRD_DB_PATH; "
        f"got {out!r}, expected {livesqlbench!r}"
    )


def test_effective_db_root_without_explicit_root_keeps_legacy_env_precedence(
    monkeypatch, tmp_path,
):
    """When no explicit `db_root` is passed (mini-interact path), keep the
    legacy `$BIRD_DB_PATH`-wins semantics so existing behaviour is intact."""
    from bird_interact_agents.slayer_otf import reference_build

    env_root = tmp_path / "via_env"
    env_root.mkdir()
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))
    passed = tmp_path / "via_arg"
    passed.mkdir()
    # No `db_root` → legacy precedence.
    out = reference_build._effective_db_root(passed)
    assert out == env_root


def test_effective_db_root_explicit_root_wins_even_when_env_unset(
    monkeypatch, tmp_path,
):
    """An explicit `db_root` must be returned unchanged when the env is
    absent too — symmetry."""
    from bird_interact_agents.slayer_otf import reference_build

    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    assert (
        reference_build._effective_db_root(
            tmp_path / "fallback", db_root=explicit,
        )
        == explicit
    )


def test_ensure_db_reference_signature_accepts_db_root_kwarg():
    """`ensure_db_reference` MUST accept a `db_root` keyword so callers
    (otf_encode `_resolve_otf_task_storage_dir`) can override the
    `$BIRD_DB_PATH`-precedence semantics for LiveSQLBench runs. Pin via
    signature inspection so the kwarg can't silently disappear."""
    import inspect

    from bird_interact_agents.slayer_otf import reference_build

    sig = inspect.signature(reference_build.ensure_db_reference)
    assert "db_root" in sig.parameters, (
        f"ensure_db_reference signature must carry a `db_root` kwarg; "
        f"got params: {list(sig.parameters)!r}"
    )


@pytest.mark.asyncio
async def test_ensure_db_reference_threads_db_root_into_build_path(
    monkeypatch, tmp_path,
):
    """Behavioural integration: a call with `db_root=<livesqlbench>` MUST
    reach the build path's `_resolve_datasource_for_build` carrying that
    value (which in turn passes it into `_effective_db_root`). Codex
    flagged that a conditional-skip-on-empty-spy can vacuously pass; this
    test drives the BUILD path explicitly (no marker pre-staged) and the
    spy is non-empty by construction."""
    from bird_interact_agents.slayer_otf import reference_build

    monkeypatch.setenv("BIRD_DB_PATH", str(tmp_path / "mini_interact_env"))
    livesqlbench = tmp_path / "livesqlbench"
    (livesqlbench / "alien").mkdir(parents=True)
    received: list = []

    # Spy on the DEEPEST entry point in the build path that consults the
    # threaded root. `_build_reference` is called from inside the lock
    # and receives `db_root` as a kwarg — record it and short-circuit
    # without touching disk or the LLM.
    async def fake_build_reference(**kwargs):
        received.append(kwargs.get("db_root"))
        # Persist the marker so the caller's success path returns cleanly.
        tgt = kwargs["target"]
        tgt.mkdir(parents=True, exist_ok=True)
        (tgt / "_reference_fp.txt").write_text(kwargs["fp"])
        (tgt / "_kb_rows.json").write_text("[]")
        (tgt / "_setup_results.json").write_text("[]")
        return []

    monkeypatch.setattr(
        reference_build, "_build_reference", fake_build_reference,
    )

    # Short-circuit `ensure_db_cache` — we never need to build a real cache.
    from bird_interact_agents.slayer_otf import cache as cache_mod

    (tmp_path / "cache" / "alien").mkdir(parents=True)
    fake_entry = cache_mod.CacheEntry(
        cache_dir=tmp_path / "cache" / "alien",
        fingerprint="dead",
        kb_rows=[],
    )

    async def fake_ensure_db_cache(db, *, cache_root, mini_interact_root, force=False):
        return fake_entry

    monkeypatch.setattr(reference_build, "ensure_db_cache", fake_ensure_db_cache)

    ref_root = tmp_path / "ref"  # marker absent → build path is taken.

    await reference_build.ensure_db_reference(
        "alien",
        reference_root=ref_root,
        cache_root=tmp_path / "cache",
        mini_interact_root=livesqlbench,
        build_encoder=lambda *a, **kw: None,
        db_root=livesqlbench,
    )
    # UNCONDITIONAL: _build_reference MUST have been called (we forced
    # the build path), AND it MUST have received the explicit
    # livesqlbench db_root — not the env value.
    assert received, (
        "_build_reference was never called — ensure_db_reference's "
        "build path skipped it. Cannot prove db_root threading."
    )
    assert received == [livesqlbench], (
        f"_build_reference received db_root={received!r}, expected "
        f"[{livesqlbench!r}] — the explicit kwarg must reach the build path."
    )


@pytest.mark.asyncio
async def test_resolve_datasource_for_build_passes_db_root_to_effective_root(
    monkeypatch, tmp_path,
):
    """Tighter check: ``_resolve_datasource_for_build`` itself MUST
    forward its ``db_root`` kwarg into ``_effective_db_root`` (otherwise
    the build-time resolve would silently ignore the LiveSQLBench root).
    """
    from bird_interact_agents.slayer_otf import reference_build

    monkeypatch.setenv("BIRD_DB_PATH", str(tmp_path / "mini_interact_env"))
    livesqlbench = tmp_path / "livesqlbench"
    livesqlbench.mkdir()
    seen: list = []

    real_effective = reference_build._effective_db_root

    def spy(mini, *, db_root=None):
        seen.append((Path(mini), db_root))
        return real_effective(mini, db_root=db_root)

    monkeypatch.setattr(reference_build, "_effective_db_root", spy)

    # Storage stub that returns None for get_datasource → resolve helper
    # is a no-op past the root computation, which is exactly what we want.
    class _NullStorage:
        async def get_datasource(self, db):
            return None
    await reference_build._resolve_datasource_for_build(
        _NullStorage(), "alien", livesqlbench, db_root=livesqlbench,
    )
    assert seen, "_effective_db_root must be called via _resolve_datasource_for_build"
    assert seen[-1][1] == livesqlbench, (
        f"_resolve_datasource_for_build must forward db_root unchanged; "
        f"got {seen[-1]!r}"
    )
