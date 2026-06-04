"""DEV-1462 — `--otf-rebuild` is benchmark-aware.

The existing `run._maybe_force_wipe_otf` (pinned by
`test_otf_rebuild_wiring.py`) calls
`paths.slayer_otf_cache_root()` + `paths.slayer_models_otf_root()` with
no benchmark scope — it would wipe mini-interact's roots when run with
`--dataset livesqlbench`. The plan extends the helper with a `benchmark`
parameter and routes the per-benchmark `paths.*_root(benchmark=...)`.

This test spies on `purge_caches` / `purge_references` and asserts the
ROOTS they're called with are scoped to the run's dataset — never the
wrong benchmark's.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents import run as run_mod
from bird_interact_agents.slayer_otf import reference_build


@pytest.fixture
def spy_purges(monkeypatch, tmp_path: Path):
    """Stub the path helpers to route per-benchmark and record purge calls."""

    def fake_cache_root(*, benchmark=None):
        if benchmark == "livesqlbench-base-lite-sqlite":
            return tmp_path / "cache_livesqlbench"
        return tmp_path / "cache"

    def fake_ref_root(*, benchmark=None):
        if benchmark == "livesqlbench-base-lite-sqlite":
            return tmp_path / "ref_livesqlbench"
        return tmp_path / "ref"

    monkeypatch.setattr(run_mod.paths, "slayer_otf_cache_root", fake_cache_root)
    monkeypatch.setattr(run_mod.paths, "slayer_models_otf_root", fake_ref_root)

    cache_calls: list = []
    ref_calls: list = []
    monkeypatch.setattr(
        reference_build, "purge_caches",
        lambda root, dbs: (cache_calls.append((Path(root), set(dbs))) or []),
    )
    monkeypatch.setattr(
        reference_build, "purge_references",
        lambda root, dbs: (ref_calls.append((Path(root), set(dbs))) or []),
    )
    return cache_calls, ref_calls, tmp_path


def test_otf_rebuild_with_livesqlbench_uses_scoped_roots(spy_purges):
    cache_calls, ref_calls, tmp_path = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk",
        dbs={"alien", "credit"},
        benchmark="livesqlbench-base-lite-sqlite",
    )
    assert cache_calls == [
        (tmp_path / "cache_livesqlbench", {"alien", "credit"}),
    ], (
        f"livesqlbench rebuild must purge the livesqlbench cache root; "
        f"got cache_calls={cache_calls}"
    )
    assert ref_calls == [
        (tmp_path / "ref_livesqlbench", {"alien", "credit"}),
    ]


def test_otf_rebuild_with_mini_interact_keeps_legacy_roots(spy_purges):
    """No `benchmark` (or `benchmark="mini-interact"`) MUST use the legacy
    roots — never the livesqlbench-scoped ones — so the existing
    mini-interact rebuild path is unchanged."""
    cache_calls, ref_calls, tmp_path = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk",
        dbs={"households"},
        benchmark=None,
    )
    assert cache_calls == [(tmp_path / "cache", {"households"})]
    assert ref_calls == [(tmp_path / "ref", {"households"})]
    # And the livesqlbench dirs were NOT touched.
    assert not any(
        "livesqlbench-base-lite-sqlite" in str(p) for p, _dbs in (cache_calls + ref_calls)
    )


def test_otf_rebuild_livesqlbench_never_touches_mini_interact_roots(spy_purges):
    cache_calls, ref_calls, tmp_path = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk",
        dbs={"alien"},
        benchmark="livesqlbench-base-lite-sqlite",
    )
    # The legacy roots `tmp_path / "cache"` and `tmp_path / "ref"` must
    # never appear in the purge call list — the whole point of B5.
    legacy_cache = tmp_path / "cache"
    legacy_ref = tmp_path / "ref"
    for root, _ in cache_calls + ref_calls:
        assert root != legacy_cache and root != legacy_ref, (
            f"livesqlbench --otf-rebuild MUST NOT purge the mini-interact "
            f"roots; saw a call against {root}"
        )
