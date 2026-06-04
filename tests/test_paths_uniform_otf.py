"""DEV-1525: uniform nested OTF path structure + gated_gold_root helper.

Tests that FAIL until paths.py is updated:
- slayer_otf_cache_root(benchmark="mini-interact") returns .../slayer_otf_cache/mini-interact/
- slayer_models_otf_root(benchmark="mini-interact") returns .../slayer_models_otf/mini-interact/
- BIRD_OTF_CACHE_ROOT overrides the parent dir for ALL benchmarks
- gated_gold_root(benchmark=) returns .../gated_gold/<benchmark>/
- Shim functions mini_interact_root(), livesqlbench_root(), etc. are gone
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bird_interact_agents import paths


# ---------------------------------------------------------------------------
# Helpers (reuse _init_repo / _setup_main_and_worktree pattern from test_paths)
# ---------------------------------------------------------------------------


def _init_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo_dir)], check=True)
    readme = repo_dir / "README.md"
    readme.write_text("test\n")
    subprocess.run(["git", "-C", str(repo_dir), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir),
         "-c", "user.email=t@example.invalid",
         "-c", "user.name=Test",
         "commit", "-q", "-m", "init"],
        check=True,
    )


def _setup_main_and_worktree(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    main = tmp_path / "main_repo"
    _init_repo(main)
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q", str(wt), "-b", "b1"],
        check=True,
    )
    monkeypatch.setattr(paths, "_LOOKUP_DIR", wt)
    return main.resolve(), wt.resolve()


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch):
    paths._main_checkout_root_cached.cache_clear()
    for var in (
        "BIRD_DB_PATH", "BIRD_DATA_PATH", "BIRD_RESULTS_ROOT",
        "BIRD_SLAYER_MODELS_ROOT",
        "BIRD_OTF_CACHE_ROOT",
        "BIRD_SLAYER_MODELS_OTF_ROOT",
        "BIRD_GATED_GOLD_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    paths._main_checkout_root_cached.cache_clear()


# ---------------------------------------------------------------------------
# slayer_otf_cache_root — uniform nested structure
# ---------------------------------------------------------------------------


def test_otf_cache_mini_interact_is_nested(tmp_path, monkeypatch):
    """mini-interact OTF cache lives at slayer_otf_cache/mini-interact/,
    not the old legacy slayer_otf_cache/ flat root."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == main / "slayer_otf_cache" / "mini-interact"
    )


def test_otf_cache_livesqlbench_sqlite_is_nested(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench-base-lite-sqlite")
        == main / "slayer_otf_cache" / "livesqlbench-base-lite-sqlite"
    )


def test_otf_cache_all_benchmarks_under_same_parent(tmp_path, monkeypatch):
    """All benchmarks share the same parent dir (slayer_otf_cache/);
    only the subdirectory differs."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    parent = main / "slayer_otf_cache"
    for bm in ("mini-interact", "livesqlbench-base-lite-sqlite",
               "livesqlbench-base-lite", "bird-interact-lite-exp"):
        root = paths.slayer_otf_cache_root(benchmark=bm)
        assert root.parent == parent, (
            f"{bm}: expected parent {parent}, got {root.parent}"
        )
        assert root.name == bm


def test_otf_cache_benchmarks_are_disjoint(tmp_path, monkeypatch):
    """Different benchmarks must land at different directories."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    roots = [
        paths.slayer_otf_cache_root(benchmark=bm)
        for bm in ("mini-interact", "livesqlbench-base-lite-sqlite")
    ]
    assert roots[0] != roots[1]


def test_otf_cache_env_override_steers_parent_for_all_benchmarks(
    tmp_path, monkeypatch,
):
    """BIRD_OTF_CACHE_ROOT overrides the parent dir for ALL benchmarks —
    benchmark-specific subdirs are still appended under it."""
    _setup_main_and_worktree(tmp_path, monkeypatch)
    parent_override = tmp_path / "custom_cache_parent"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(parent_override))
    for bm in ("mini-interact", "livesqlbench-base-lite-sqlite"):
        assert paths.slayer_otf_cache_root(benchmark=bm) == parent_override / bm


def test_otf_cache_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", "/data/otf_cache")
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == Path("/data/otf_cache") / "mini-interact"
    )


def test_otf_cache_anchored_to_main_not_worktree(tmp_path, monkeypatch):
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    root = paths.slayer_otf_cache_root(benchmark="mini-interact")
    assert str(root).startswith(str(main))
    assert not str(root).startswith(str(wt))


# ---------------------------------------------------------------------------
# slayer_models_otf_root — same uniform structure
# ---------------------------------------------------------------------------


def test_models_otf_mini_interact_is_nested(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == main / "slayer_models_otf" / "mini-interact"
    )


def test_models_otf_all_benchmarks_under_same_parent(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    parent = main / "slayer_models_otf"
    for bm in ("mini-interact", "livesqlbench-base-lite-sqlite"):
        root = paths.slayer_models_otf_root(benchmark=bm)
        assert root.parent == parent
        assert root.name == bm


def test_models_otf_env_override_steers_parent(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    parent_override = tmp_path / "custom_models_parent"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(parent_override))
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == parent_override / "mini-interact"
    )


def test_models_otf_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", "/data/models_otf")
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == Path("/data/models_otf") / "mini-interact"
    )


def test_models_otf_anchored_to_main_not_worktree(tmp_path, monkeypatch):
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    root = paths.slayer_models_otf_root(benchmark="mini-interact")
    assert str(root).startswith(str(main))
    assert not str(root).startswith(str(wt))


# ---------------------------------------------------------------------------
# gated_gold_root — new helper
# ---------------------------------------------------------------------------


def test_gated_gold_root_default(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.gated_gold_root(benchmark="livesqlbench-base-lite-sqlite")
        == main / "gated_gold" / "livesqlbench-base-lite-sqlite"
    )


def test_gated_gold_root_mini_interact(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.gated_gold_root(benchmark="mini-interact")
        == main / "gated_gold" / "mini-interact"
    )


def test_gated_gold_root_env_override(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    override_parent = tmp_path / "custom_gold_parent"
    monkeypatch.setenv("BIRD_GATED_GOLD_ROOT", str(override_parent))
    assert (
        paths.gated_gold_root(benchmark="livesqlbench-base-lite-sqlite")
        == override_parent / "livesqlbench-base-lite-sqlite"
    )


def test_gated_gold_root_requires_benchmark(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        paths.gated_gold_root()  # benchmark kwarg is required
    with pytest.raises(ValueError):
        paths.gated_gold_root(benchmark=None)


def test_gated_gold_root_anchored_to_main(tmp_path, monkeypatch):
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    root = paths.gated_gold_root(benchmark="livesqlbench-base-lite-sqlite")
    assert str(root).startswith(str(main))
    assert not str(root).startswith(str(wt))


def test_gated_gold_root_unknown_benchmark_raises(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.gated_gold_root(benchmark="not-a-real-benchmark")


def test_gated_gold_root_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv("BIRD_GATED_GOLD_ROOT", "/data/gated_gold")
    assert (
        paths.gated_gold_root(benchmark="livesqlbench-base-lite-sqlite")
        == Path("/data/gated_gold") / "livesqlbench-base-lite-sqlite"
    )


# ---------------------------------------------------------------------------
# Shim functions must NOT exist (removed in this PR)
# ---------------------------------------------------------------------------


def test_mini_interact_root_shim_is_removed():
    assert not hasattr(paths, "mini_interact_root"), (
        "paths.mini_interact_root() shim must be removed; "
        "use benchmark_data_root('mini-interact')"
    )


def test_mini_interact_data_file_shim_is_removed():
    assert not hasattr(paths, "mini_interact_data_file"), (
        "paths.mini_interact_data_file() shim must be removed; "
        "use benchmark_data_file('mini-interact')"
    )


def test_livesqlbench_root_shim_is_removed():
    assert not hasattr(paths, "livesqlbench_root"), (
        "paths.livesqlbench_root() shim must be removed; "
        "use benchmark_data_root('livesqlbench-base-lite-sqlite')"
    )


def test_livesqlbench_data_file_shim_is_removed():
    assert not hasattr(paths, "livesqlbench_data_file"), (
        "paths.livesqlbench_data_file() shim must be removed; "
        "use benchmark_data_file('livesqlbench-base-lite-sqlite')"
    )
