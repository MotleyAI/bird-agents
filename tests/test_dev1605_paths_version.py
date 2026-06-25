"""DEV-1605: ``paths.slayer_models_otf_root`` gains an optional ``version=``
segment that sits ABOVE the per-db dir:
``slayer_models_otf/<benchmark>/<version>/<db>/``.

``version=None`` returns the legacy ``<benchmark>`` parent unchanged (so all
existing call sites that append ``/<db>`` themselves keep working, and the
parent is the "list all versions" root). A bad version label (containing
``/`` or ``..`` or empty/dot) is rejected so a malformed slug can't escape the
tree.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import paths
from tests.test_paths import _setup_main_and_worktree


@pytest.fixture(autouse=True)
def _isolate_main_checkout_cache():
    """Clear the memoised main-checkout resolution before AND after each test
    (mirrors test_paths' `_isolate_paths`), so pointing `_LOOKUP_DIR` at a tmp
    worktree neither reads a stale session cache nor leaks the tmp path to a
    later test."""
    paths._main_checkout_root_cached.cache_clear()
    yield
    paths._main_checkout_root_cached.cache_clear()


def test_version_none_is_legacy_parent(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == main / "slayer_models_otf" / "mini-interact"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact", version=None)
        == main / "slayer_models_otf" / "mini-interact"
    )


def test_version_appends_segment_above_db(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact", version="opus-4-7")
        == main / "slayer_models_otf" / "mini-interact" / "opus-4-7"
    )
    # The per-db dir is then root / db (call sites unchanged).
    root = paths.slayer_models_otf_root(benchmark="mini-interact", version="glm-5.2")
    assert root / "alien" == main / "slayer_models_otf" / "mini-interact" / "glm-5.2" / "alien"


def test_version_honours_env_override(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(tmp_path / "ext"))
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact", version="opus-4-7")
        == tmp_path / "ext" / "mini-interact" / "opus-4-7"
    )


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "../escape", "x/../y", "/abs"])
def test_bad_version_label_rejected(tmp_path, monkeypatch, bad):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.slayer_models_otf_root(benchmark="mini-interact", version=bad)


def test_versioned_roots_disjoint_across_benchmarks(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact", version="opus-4-7")
        != paths.slayer_models_otf_root(
            benchmark="livesqlbench-base-lite-sqlite", version="opus-4-7"
        )
    )
