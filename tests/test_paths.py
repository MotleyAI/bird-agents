"""Tests for the central path helper.

These tests have to work from any checkout, including a `git worktree add`
spawned from this repo. They exercise the three behaviours that matter:

* `main_checkout_root` traces through `git rev-parse --git-common-dir` so
  that worktrees still find the canonical checkout.
* `benchmark_data_root` / `benchmark_data_file` use the canonical benchmark
  name as the subdir under the shared `BIRD_BENCHMARKS_ROOT` parent.
* The output-sink helpers (`audited_gold_root`, `slayer_models_root`,
  `results_root`, `benchmarks_root`) are anchored to the main checkout,
  not the worktree, so benchmark results written from a worktree land
  with the rest.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bird_interact_agents import paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_repo(repo_dir: Path) -> None:
    """Initialise a throwaway git repo with one commit (needed for `worktree add`)."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo_dir)],
        check=True,
    )
    readme = repo_dir / "README.md"
    readme.write_text("test\n")
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", "README.md"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C", str(repo_dir),
            "-c", "user.email=t@example.invalid",
            "-c", "user.name=Test",
            "commit", "-q", "-m", "init",
        ],
        check=True,
    )


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch):
    """Clear cache + strip path env vars so the outer pytest env can't leak in.

    Tests that need to assert an env-var override set it explicitly via
    monkeypatch.setenv; everything else sees a clean slate.
    """
    paths._main_checkout_root_cached.cache_clear()
    for var in (
        "BIRD_BENCHMARKS_ROOT", "BIRD_RESULTS_ROOT",
        "BIRD_AUDITED_GOLD_ROOT",
        "BIRD_SLAYER_MODELS_ROOT", "BIRD_OTF_CACHE_ROOT",
        "BIRD_SLAYER_MODELS_OTF_ROOT", "BIRD_GATED_GOLD_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    paths._main_checkout_root_cached.cache_clear()


# ---------------------------------------------------------------------------
# main_checkout_root
# ---------------------------------------------------------------------------


def test_main_checkout_root_from_main(tmp_path, monkeypatch):
    repo = tmp_path / "main_repo"
    _init_repo(repo)
    monkeypatch.setattr(paths, "_LOOKUP_DIR", repo)
    assert paths.main_checkout_root() == repo.resolve()


def test_main_checkout_root_from_worktree(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    wt = tmp_path / "wt_test"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q",
         str(wt), "-b", "wt-branch"],
        check=True,
    )
    monkeypatch.setattr(paths, "_LOOKUP_DIR", wt)
    # The whole point: a worktree resolves to the main checkout, not its own dir.
    assert paths.main_checkout_root() == main.resolve()
    assert paths.main_checkout_root() != wt.resolve()


def test_main_checkout_root_outside_git_falls_back(tmp_path, monkeypatch):
    """When no enclosing git repo, fall back to the source-relative parents[2]."""
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", outside)
    expected = Path(paths.__file__).resolve().parents[2]
    assert paths.main_checkout_root() == expected


def test_main_checkout_root_cached(tmp_path, monkeypatch):
    """Second call must not re-invoke git rev-parse."""
    repo = tmp_path / "cached_repo"
    _init_repo(repo)
    monkeypatch.setattr(paths, "_LOOKUP_DIR", repo)
    first = paths.main_checkout_root()
    # Move the lookup dir to somewhere else; cached result must persist.
    monkeypatch.setattr(paths, "_LOOKUP_DIR", tmp_path)
    second = paths.main_checkout_root()
    assert first == second == repo.resolve()


def test_main_checkout_root_runs_git_only_once(tmp_path, monkeypatch):
    """Stronger version: spy on subprocess.run to prove no second invocation."""
    repo = tmp_path / "spied_repo"
    _init_repo(repo)
    monkeypatch.setattr(paths, "_LOOKUP_DIR", repo)

    real_run = paths.subprocess.run
    call_count = {"n": 0}

    def counting_run(*args, **kwargs):
        call_count["n"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(paths.subprocess, "run", counting_run)
    paths.main_checkout_root()
    paths.main_checkout_root()
    paths.main_checkout_root()
    assert call_count["n"] == 1


def test_main_checkout_root_from_nested_lookup_dir(tmp_path, monkeypatch):
    """Production _LOOKUP_DIR is `src/bird_interact_agents/`, not the repo root.

    Verify the lookup still finds the main checkout when the cwd is a
    deeply-nested subdir of a worktree, not the worktree root itself.
    """
    main = tmp_path / "main_repo"
    _init_repo(main)
    wt = tmp_path / "wt_nested"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q",
         str(wt), "-b", "wt-nested-branch"],
        check=True,
    )
    nested = wt / "src" / "bird_interact_agents"
    nested.mkdir(parents=True)
    monkeypatch.setattr(paths, "_LOOKUP_DIR", nested)
    assert paths.main_checkout_root() == main.resolve()


# ---------------------------------------------------------------------------
# benchmark_data_root / benchmark_data_file — uniform BIRD_BENCHMARKS_ROOT
# ---------------------------------------------------------------------------


def test_benchmark_data_root_default_sibling_of_main(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    (tmp_path / "mini-interact").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert paths.benchmark_data_root("mini-interact") == (tmp_path / "mini-interact").resolve()


def test_benchmark_data_root_from_worktree_points_at_main_sibling(
    tmp_path, monkeypatch,
):
    main = tmp_path / "main_repo"
    _init_repo(main)
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q", str(wt), "-b", "b1"],
        check=True,
    )
    (tmp_path / "mini-interact").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", wt)
    assert paths.benchmark_data_root("mini-interact") == (tmp_path / "mini-interact").resolve()


def test_bird_benchmarks_root_env_overrides_parent_for_all_benchmarks(
    tmp_path, monkeypatch,
):
    """`BIRD_BENCHMARKS_ROOT` overrides the parent dir; benchmark name is appended."""
    parent = tmp_path / "custom_root"
    monkeypatch.setenv("BIRD_BENCHMARKS_ROOT", str(parent))
    assert paths.benchmark_data_root("mini-interact") == parent / "mini-interact"
    assert (
        paths.benchmark_data_root("livesqlbench-base-lite-sqlite")
        == parent / "livesqlbench-base-lite-sqlite"
    )
    assert paths.benchmark_data_root("livesqlbench-base-lite") == parent / "livesqlbench-base-lite"
    assert paths.benchmark_data_root("bird-interact-lite-exp") == parent / "bird-interact-lite-exp"


def test_bird_benchmarks_root_env_wins_when_default_would_fail(tmp_path, monkeypatch):
    """Even if main_checkout_root can't be resolved, BIRD_BENCHMARKS_ROOT wins."""
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    parent = tmp_path / "override_root"
    monkeypatch.setenv("BIRD_BENCHMARKS_ROOT", str(parent))
    assert paths.benchmark_data_root("mini-interact") == parent / "mini-interact"


def test_benchmark_data_root_nonexistent_path_accepted(tmp_path, monkeypatch):
    """Helper is a path-resolver, not a validator — non-existent paths pass through."""
    parent = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("BIRD_BENCHMARKS_ROOT", str(parent))
    result = paths.benchmark_data_root("mini-interact")
    assert result == parent / "mini-interact"
    assert not result.exists()


def test_benchmark_data_file_default(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    (tmp_path / "mini-interact").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert (
        paths.benchmark_data_file("mini-interact")
        == (tmp_path / "mini-interact" / "mini_interact.jsonl").resolve()
    )


# ---------------------------------------------------------------------------
# Output sinks
# ---------------------------------------------------------------------------


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


def test_audited_gold_root_anchored_to_main(tmp_path, monkeypatch):
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert paths.audited_gold_root() == main / "audited_gold"
    assert paths.audited_gold_root() != wt / "audited_gold"


def test_slayer_models_root_anchored_to_main(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert paths.slayer_models_root() == main / "slayer_models"


def test_slayer_models_otf_root_anchored_to_main(tmp_path, monkeypatch):
    """The on-the-fly KB-encode reference root is nested under the main checkout.
    Each benchmark gets its own subdir so artifact roots never collide."""
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == main / "slayer_models_otf" / "mini-interact"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        != wt / "slayer_models_otf" / "mini-interact"
    )


# ---------------------------------------------------------------------------
# DEV-1468: slayer_otf_cache_root (NEW) + env overrides on the three slayer
# artifact roots. The overrides are what the cloud actor uses to repoint each
# root at the per-combo /data/... dir; they must be honoured AND harmless
# locally (default unchanged when the env var is unset).
# ---------------------------------------------------------------------------


def test_slayer_otf_cache_root_anchored_to_main(tmp_path, monkeypatch):
    """The phase-1-3 ingest cache root is nested under the main checkout.
    Each benchmark gets its own subdir so artifact roots never collide."""
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == main / "slayer_otf_cache" / "mini-interact"
    )
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        != wt / "slayer_otf_cache" / "mini-interact"
    )


def test_slayer_models_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "slayer_models"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_ROOT", str(override))
    assert paths.slayer_models_root() == override


def test_slayer_otf_cache_root_env_override(tmp_path, monkeypatch):
    parent = tmp_path / "data" / "otf_cache"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(parent))
    assert paths.slayer_otf_cache_root(benchmark="mini-interact") == parent / "mini-interact"


def test_slayer_models_otf_root_env_override(tmp_path, monkeypatch):
    parent = tmp_path / "data" / "models_otf"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(parent))
    assert paths.slayer_models_otf_root(benchmark="mini-interact") == parent / "mini-interact"


def test_agents_no_longer_define_their_own_otf_cache_root():
    """DEV-1468 (B): the duplicated per-agent ``_otf_cache_root`` helpers are
    deleted — both agents must rely on the single ``paths.slayer_otf_cache_root``
    so the cache root is defined in exactly one place (and is env-overridable
    for the cloud download target)."""
    from bird_interact_agents.agents.pydantic_ai_recursive import (
        agent as recursive_agent,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode import (
        agent as otf_encode_agent,
    )
    assert not hasattr(recursive_agent, "_otf_cache_root"), (
        "pydantic_ai_recursive must use paths.slayer_otf_cache_root()"
    )
    assert not hasattr(otf_encode_agent, "_otf_cache_root"), (
        "pydantic_ai_otf_encode must use paths.slayer_otf_cache_root()"
    )


def test_slayer_root_env_overrides_win_when_default_would_fail(
    tmp_path, monkeypatch,
):
    """Env overrides must resolve without touching git — so they work in a
    wheel install / outside a checkout (the cloud container case)."""
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv("BIRD_SLAYER_MODELS_ROOT", "/data/slayer_models")
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", "/data/otf_cache")
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", "/data/models_otf")
    assert paths.slayer_models_root() == Path("/data/slayer_models")
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == Path("/data/otf_cache") / "mini-interact"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == Path("/data/models_otf") / "mini-interact"
    )


def test_results_root_default_anchored_to_main(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert paths.results_root() == main / "results"


def test_results_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "somewhere" / "shared_results"
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(override))
    assert paths.results_root() == override


def test_results_root_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    override = tmp_path / "explicit_results"
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(override))
    assert paths.results_root() == override


def test_benchmarks_root_anchored_to_main(tmp_path, monkeypatch):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert paths.benchmarks_root() == main / ".benchmarks"


def test_livesqlbench_data_file_default(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    (tmp_path / "livesqlbench-base-lite-sqlite").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert (
        paths.benchmark_data_file("livesqlbench-base-lite-sqlite")
        == (
            tmp_path / "livesqlbench-base-lite-sqlite"
            / "livesqlbench_data_sqlite.jsonl"
        ).resolve()
    )


# ---------------------------------------------------------------------------
# Per-benchmark scoping of the two OTF root helpers
# (`slayer_otf_cache_root` and `slayer_models_otf_root`). `benchmark` is now
# REQUIRED and explicit (no `None` default) — a forgotten benchmark must NOT
# silently fall back to the mini-interact roots and mix artifacts. The
# `benchmark` values are:
#   * `"mini-interact"` → the LEGACY dirs (`slayer_otf_cache/`,
#     `slayer_models_otf/`) + LEGACY env vars (`BIRD_OTF_CACHE_ROOT`,
#     `BIRD_SLAYER_MODELS_OTF_ROOT`) — on-disk layout & cloud contract unchanged,
#   * `"livesqlbench-base-lite-sqlite"` → PARALLEL `_livesqlbench` roots + `_LIVESQLBENCH` env vars,
#   * `None` / unknown → ValueError.
# ---------------------------------------------------------------------------


def test_slayer_otf_cache_root_mini_interact_is_nested(
    tmp_path, monkeypatch,
):
    """Each benchmark gets `slayer_otf_cache/<benchmark>/`, not a flat root."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == main / "slayer_otf_cache" / "mini-interact"
    )


def test_slayer_otf_cache_root_unknown_benchmark_raises(tmp_path, monkeypatch):
    """`benchmark=None`/unknown MUST raise — no silent fallback."""
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.slayer_otf_cache_root(benchmark=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        paths.slayer_otf_cache_root()  # type: ignore[call-arg]


def test_slayer_otf_cache_root_livesqlbench_is_parallel_root(
    tmp_path, monkeypatch,
):
    """Each benchmark gets its own nested subdir — no DB-name collisions."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    lsb_root = paths.slayer_otf_cache_root(benchmark="livesqlbench-base-lite-sqlite")
    assert lsb_root == main / "slayer_otf_cache" / "livesqlbench-base-lite-sqlite"
    assert lsb_root != paths.slayer_otf_cache_root(benchmark="mini-interact")


def test_slayer_otf_cache_root_env_override_is_parent_for_all_benchmarks(
    tmp_path, monkeypatch,
):
    """`BIRD_OTF_CACHE_ROOT` is the parent dir; benchmark is appended as a subdir."""
    _main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    parent = tmp_path / "elsewhere"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(parent))
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == parent / "mini-interact"
    )
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench-base-lite-sqlite")
        == parent / "livesqlbench-base-lite-sqlite"
    )


def test_slayer_otf_cache_root_livesqlbench_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", "/data/otf_cache_parent")
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench-base-lite-sqlite")
        == Path("/data/otf_cache_parent") / "livesqlbench-base-lite-sqlite"
    )


def test_slayer_models_otf_root_mini_interact_is_nested(
    tmp_path, monkeypatch,
):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == main / "slayer_models_otf" / "mini-interact"
    )


def test_slayer_models_otf_root_unknown_benchmark_raises(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.slayer_models_otf_root(benchmark=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        paths.slayer_models_otf_root()  # type: ignore[call-arg]


def test_slayer_models_otf_root_livesqlbench_is_parallel_root(
    tmp_path, monkeypatch,
):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    lsb_root = paths.slayer_models_otf_root(benchmark="livesqlbench-base-lite-sqlite")
    assert lsb_root == main / "slayer_models_otf" / "livesqlbench-base-lite-sqlite"
    assert lsb_root != paths.slayer_models_otf_root(benchmark="mini-interact")


def test_slayer_models_otf_root_env_override_is_parent_for_all_benchmarks(
    tmp_path, monkeypatch,
):
    """`BIRD_SLAYER_MODELS_OTF_ROOT` is the parent dir; benchmark is appended."""
    _main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    parent = tmp_path / "elsewhere_models"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(parent))
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == parent / "mini-interact"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="livesqlbench-base-lite-sqlite")
        == parent / "livesqlbench-base-lite-sqlite"
    )


def test_slayer_models_otf_root_livesqlbench_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", "/data/models_otf_parent")
    assert (
        paths.slayer_models_otf_root(benchmark="livesqlbench-base-lite-sqlite")
        == Path("/data/models_otf_parent") / "livesqlbench-base-lite-sqlite"
    )


def test_unknown_benchmark_value_raises(tmp_path, monkeypatch):
    """An unrecognised benchmark name MUST raise — no silent fallback."""
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        paths.slayer_otf_cache_root(benchmark="this-is-not-a-benchmark")
    assert "benchmark" in str(exc_info.value).lower()
    with pytest.raises(ValueError):
        paths.slayer_models_otf_root(benchmark="this-is-not-a-benchmark")


# ---------------------------------------------------------------------------
# Live invocation from this repo's actual checkout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DEV-1510: `audited_gold_file(benchmark)` — resolves the audited-gold file
# for benchmarks whose `audited_gold_layout == "single_file"`.
# ---------------------------------------------------------------------------


def test_audited_gold_file_livesqlbench_anchored_to_main(tmp_path, monkeypatch):
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.audited_gold_file(benchmark="livesqlbench-base-lite-sqlite")
        == main / "audited_gold" / "livesqlbench-base-lite-sqlite" / "livesqlbench-base-lite-sqlite_audited.jsonl"
    )
    assert (
        paths.audited_gold_file(benchmark="livesqlbench-base-lite-sqlite")
        != wt / "audited_gold" / "livesqlbench-base-lite-sqlite" / "livesqlbench-base-lite-sqlite_audited.jsonl"
    )


def test_audited_gold_file_honours_root_env_override(tmp_path, monkeypatch):
    """`BIRD_AUDITED_GOLD_ROOT` (used by SAR-audit tests) repositions the
    root; the single-file helper must respect it so cloud-side test fixtures
    can point at a tmp dir."""
    override = tmp_path / "elsewhere" / "audited_gold"
    monkeypatch.setenv("BIRD_AUDITED_GOLD_ROOT", str(override))
    assert (
        paths.audited_gold_file(benchmark="livesqlbench-base-lite-sqlite")
        == override / "livesqlbench-base-lite-sqlite" / "livesqlbench-base-lite-sqlite_audited.jsonl"
    )


def test_audited_gold_file_mini_interact_anchored_to_main(tmp_path, monkeypatch):
    """mini-interact uses single_file layout; file is ``mini-interact_audited.jsonl``."""
    main_root, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.audited_gold_file(benchmark="mini-interact")
        == main_root / "audited_gold" / "mini-interact" / "mini-interact_audited.jsonl"
    )


def test_audited_gold_file_unknown_benchmark_raises(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.audited_gold_file(benchmark="this-is-not-a-benchmark")


def test_audited_gold_file_requires_explicit_benchmark(tmp_path, monkeypatch):
    """`benchmark` is required (no default) so a forgotten kwarg cannot
    silently pick a benchmark."""
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        paths.audited_gold_file()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        paths.audited_gold_file(benchmark=None)  # type: ignore[arg-type]


def test_live_main_checkout_root_is_a_git_dir():
    """Smoke test using the real LOOKUP_DIR (no monkeypatch).

    We can't assert an exact path here — that would break in a worktree, which
    is the whole point of this module — but we can assert the result is a
    real directory containing a `.git` entry (file or dir).
    """
    root = paths.main_checkout_root()
    assert root.is_dir()
    git_entry = root / ".git"
    assert git_entry.exists(), f"expected .git under {root}"
