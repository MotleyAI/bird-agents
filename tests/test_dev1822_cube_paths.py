"""DEV-1822: `paths.cube_local_root` is main-checkout-anchored + per-benchmark."""

from __future__ import annotations

import pytest

from bird_interact_agents import paths


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("BIRD_CUBE_LOCAL_ROOT", raising=False)
    monkeypatch.setattr(paths, "main_checkout_root", lambda: tmp_path)
    return tmp_path


def test_default_under_main_checkout(_isolate):
    # pure resolver (mirrors slayer_otf_cache_root); dir creation is deploy/conf's job
    p = paths.cube_local_root(benchmark="livesqlbench-base-lite")
    assert p == _isolate / "cube_local" / "livesqlbench-base-lite"


def test_env_override(monkeypatch, tmp_path):
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("BIRD_CUBE_LOCAL_ROOT", str(override))
    p = paths.cube_local_root(benchmark="livesqlbench-base-lite")
    assert p == override / "livesqlbench-base-lite"


def test_unknown_benchmark_rejected(_isolate):
    with pytest.raises(ValueError):
        paths.cube_local_root(benchmark="not-a-benchmark")


def test_benchmark_required(_isolate):
    with pytest.raises((ValueError, TypeError)):
        paths.cube_local_root(benchmark=None)  # type: ignore[arg-type]
