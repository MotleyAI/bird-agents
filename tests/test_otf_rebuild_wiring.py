"""DEV-1468: ``--otf-rebuild`` force-wipes BOTH on-the-fly layers (cache AND
reference) for the run's DBs, for BOTH on-the-fly frameworks — not just the
otf_encode reference as before.

These tests target the small ``run._maybe_force_wipe_otf`` helper (the seam
``run_evaluation`` calls once before the task loop) so we don't have to drive a
full evaluation just to prove the wipe wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents import run as run_mod
from bird_interact_agents.slayer_otf import reference_build


@pytest.fixture
def spy_purges(monkeypatch, tmp_path: Path):
    """Redirect the two artifact roots into tmp and record purge calls."""
    # DEV-1462: helpers now take an optional `benchmark` kwarg. The
    # legacy `--otf-rebuild` (no `--dataset livesqlbench`) calls them
    # with `benchmark=None`, so the stub must accept that.
    monkeypatch.setattr(
        run_mod.paths, "slayer_otf_cache_root",
        lambda *, benchmark=None: tmp_path / "cache",
    )
    monkeypatch.setattr(
        run_mod.paths, "slayer_models_otf_root",
        lambda *, benchmark=None: tmp_path / "ref",
    )
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


@pytest.mark.parametrize(
    "framework", ["pydantic_ai_recursive", "pydantic_ai_otf_encode"],
)
def test_otf_rebuild_wipes_both_layers_for_otf_frameworks(spy_purges, framework):
    cache_calls, ref_calls, tmp_path = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework=framework, dbs={"db_a", "db_b"},
    )
    assert cache_calls == [(tmp_path / "cache", {"db_a", "db_b"})]
    assert ref_calls == [(tmp_path / "ref", {"db_a", "db_b"})]


def test_otf_rebuild_off_is_a_noop(spy_purges):
    cache_calls, ref_calls, _ = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=False, framework="pydantic_ai_otf_encode", dbs={"db_a"},
    )
    assert cache_calls == []
    assert ref_calls == []


def test_otf_rebuild_noop_for_non_otf_framework(spy_purges):
    """A non-on-the-fly framework (e.g. pre-encoded pydantic_ai) has no OTF
    artifacts to wipe."""
    cache_calls, ref_calls, _ = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="pydantic_ai", dbs={"db_a"},
    )
    assert cache_calls == []
    assert ref_calls == []
