"""Tests for the central path helper.

These tests have to work from any checkout, including a `git worktree add`
spawned from this repo. They exercise the three behaviours that matter:

* `main_checkout_root` traces through `git rev-parse --git-common-dir` so
  that worktrees still find the canonical checkout.
* `mini_interact_root` / `mini_interact_data_file` honour the
  `BIRD_DB_PATH` / `BIRD_DATA_PATH` env-var overrides used by the
  upstream mini-interact-agent harness.
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
        "BIRD_DB_PATH", "BIRD_DATA_PATH", "BIRD_RESULTS_ROOT",
        "BIRD_SLAYER_MODELS_ROOT", "BIRD_OTF_CACHE_ROOT",
        "BIRD_SLAYER_MODELS_OTF_ROOT",
        # DEV-1462: livesqlbench-scoped overrides.
        "BIRD_LIVESQLBENCH_ROOT", "BIRD_LIVESQLBENCH_DATA_FILE",
        "BIRD_OTF_CACHE_ROOT_LIVESQLBENCH",
        "BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH",
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
# mini_interact_root
# ---------------------------------------------------------------------------


def test_mini_interact_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "mini-interact"
    override.mkdir(parents=True)
    monkeypatch.setenv("BIRD_DB_PATH", str(override))
    assert paths.mini_interact_root() == override


def test_mini_interact_root_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    """Even if main_checkout_root can't be resolved, the env override must win.

    Proves the helper short-circuits on the env var *before* attempting the
    git-based default resolution.
    """
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    override = tmp_path / "explicit_db_path"
    monkeypatch.setenv("BIRD_DB_PATH", str(override))
    # Note: override does not need to exist for the helper to return it.
    assert paths.mini_interact_root() == override


def test_mini_interact_root_override_nonexistent_path_accepted(tmp_path, monkeypatch):
    """Helper is a path-resolver, not a validator — non-existent paths pass through."""
    nonexistent = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("BIRD_DB_PATH", str(nonexistent))
    assert paths.mini_interact_root() == nonexistent
    assert not paths.mini_interact_root().exists()


def test_mini_interact_root_default_sibling_of_main(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    # Sibling to the main checkout
    (tmp_path / "mini-interact").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert paths.mini_interact_root() == (tmp_path / "mini-interact").resolve()


def test_mini_interact_root_from_worktree_points_at_main_sibling(
    tmp_path, monkeypatch,
):
    main = tmp_path / "main_repo"
    _init_repo(main)
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q",
         str(wt), "-b", "b1"],
        check=True,
    )
    (tmp_path / "mini-interact").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", wt)
    # Even from the worktree, the sibling is anchored at the main checkout's parent.
    assert paths.mini_interact_root() == (tmp_path / "mini-interact").resolve()


# ---------------------------------------------------------------------------
# mini_interact_data_file
# ---------------------------------------------------------------------------


def test_mini_interact_data_file_env_override(tmp_path, monkeypatch):
    f = tmp_path / "custom_tasks.jsonl"
    f.write_text("")
    monkeypatch.setenv("BIRD_DATA_PATH", str(f))
    assert paths.mini_interact_data_file() == f


def test_mini_interact_data_file_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    override = tmp_path / "tasks.jsonl"
    monkeypatch.setenv("BIRD_DATA_PATH", str(override))
    assert paths.mini_interact_data_file() == override


def test_mini_interact_data_file_default(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    (tmp_path / "mini-interact").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert (
        paths.mini_interact_data_file()
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
    """DEV-1454: the on-the-fly KB-encode reference root is a sibling of
    slayer_models under the main checkout (so it's git-committable and the
    HARD-8 variant builder resolves mini-interact identically)."""
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini_interact")
        == main / "slayer_models_otf"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini_interact")
        != wt / "slayer_models_otf"
    )


# ---------------------------------------------------------------------------
# DEV-1468: slayer_otf_cache_root (NEW) + env overrides on the three slayer
# artifact roots. The overrides are what the cloud actor uses to repoint each
# root at the per-combo /data/... dir; they must be honoured AND harmless
# locally (default unchanged when the env var is unset).
# ---------------------------------------------------------------------------


def test_slayer_otf_cache_root_anchored_to_main(tmp_path, monkeypatch):
    """The phase-1-3 ingest cache root is a sibling of slayer_models under
    the main checkout (shared across worktrees), replacing the duplicated
    per-agent _otf_cache_root helpers."""
    main, wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="mini_interact")
        == main / "slayer_otf_cache"
    )
    assert (
        paths.slayer_otf_cache_root(benchmark="mini_interact")
        != wt / "slayer_otf_cache"
    )


def test_slayer_models_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "slayer_models"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_ROOT", str(override))
    assert paths.slayer_models_root() == override


def test_slayer_otf_cache_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "data" / "slayer_otf_cache"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(override))
    assert paths.slayer_otf_cache_root(benchmark="mini_interact") == override


def test_slayer_models_otf_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(override))
    assert paths.slayer_models_otf_root(benchmark="mini_interact") == override


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
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", "/data/slayer_otf_cache")
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", "/data/slayer_models_otf")
    assert paths.slayer_models_root() == Path("/data/slayer_models")
    assert (
        paths.slayer_otf_cache_root(benchmark="mini_interact")
        == Path("/data/slayer_otf_cache")
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini_interact")
        == Path("/data/slayer_models_otf")
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


# ---------------------------------------------------------------------------
# DEV-1462: livesqlbench_root / livesqlbench_data_file — mirror the
# mini-interact helpers. Same env-override pattern, same worktree-aware
# default anchored at the main checkout's parent.
# ---------------------------------------------------------------------------


def test_livesqlbench_root_default_sibling_of_main(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    (tmp_path / "livesqlbench-base-lite-sqlite").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert (
        paths.livesqlbench_root()
        == (tmp_path / "livesqlbench-base-lite-sqlite").resolve()
    )


def test_livesqlbench_root_env_override(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "livesqlbench"
    override.mkdir(parents=True)
    monkeypatch.setenv("BIRD_LIVESQLBENCH_ROOT", str(override))
    assert paths.livesqlbench_root() == override


def test_livesqlbench_root_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    override = tmp_path / "explicit_livesqlbench_path"
    monkeypatch.setenv("BIRD_LIVESQLBENCH_ROOT", str(override))
    assert paths.livesqlbench_root() == override


def test_livesqlbench_root_from_worktree_points_at_main_sibling(
    tmp_path, monkeypatch,
):
    main = tmp_path / "main_repo"
    _init_repo(main)
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q",
         str(wt), "-b", "b1"],
        check=True,
    )
    (tmp_path / "livesqlbench-base-lite-sqlite").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", wt)
    assert (
        paths.livesqlbench_root()
        == (tmp_path / "livesqlbench-base-lite-sqlite").resolve()
    )


def test_livesqlbench_data_file_env_override(tmp_path, monkeypatch):
    f = tmp_path / "custom_lsb_tasks.jsonl"
    f.write_text("")
    monkeypatch.setenv("BIRD_LIVESQLBENCH_DATA_FILE", str(f))
    assert paths.livesqlbench_data_file() == f


def test_livesqlbench_data_file_default(tmp_path, monkeypatch):
    main = tmp_path / "main_repo"
    _init_repo(main)
    (tmp_path / "livesqlbench-base-lite-sqlite").mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    assert (
        paths.livesqlbench_data_file()
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
#   * `"mini_interact"` → the LEGACY dirs (`slayer_otf_cache/`,
#     `slayer_models_otf/`) + LEGACY env vars (`BIRD_OTF_CACHE_ROOT`,
#     `BIRD_SLAYER_MODELS_OTF_ROOT`) — on-disk layout & cloud contract unchanged,
#   * `"livesqlbench"` → PARALLEL `_livesqlbench` roots + `_LIVESQLBENCH` env vars,
#   * `None` / unknown → ValueError.
# ---------------------------------------------------------------------------


def test_slayer_otf_cache_root_mini_interact_is_legacy_root(
    tmp_path, monkeypatch,
):
    """`benchmark="mini_interact"` returns the legacy mini-interact root so the
    dev-1470 cloud upload/merge contract is unchanged."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="mini_interact")
        == main / "slayer_otf_cache"
    )


def test_slayer_otf_cache_root_none_benchmark_raises(tmp_path, monkeypatch):
    """`benchmark=None` (or omitted) MUST raise — no silent fallback to the
    mini-interact root, which would let a forgotten benchmark mix artifacts."""
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.slayer_otf_cache_root(benchmark=None)
    with pytest.raises(TypeError):
        paths.slayer_otf_cache_root()  # benchmark is required (no default)


def test_slayer_otf_cache_root_livesqlbench_is_parallel_root(
    tmp_path, monkeypatch,
):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench")
        == main / "slayer_otf_cache_livesqlbench"
    )
    # And it is DISJOINT from the legacy mini-interact root: the whole
    # point of the per-benchmark separation is no collision on shared DB
    # names (alien, cross_db, …).
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench")
        != paths.slayer_otf_cache_root(benchmark="mini_interact")
    )


def test_slayer_otf_cache_root_livesqlbench_env_override(tmp_path, monkeypatch):
    override = tmp_path / "data" / "slayer_otf_cache_livesqlbench"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT_LIVESQLBENCH", str(override))
    assert paths.slayer_otf_cache_root(benchmark="livesqlbench") == override


def test_slayer_otf_cache_root_legacy_env_does_not_steer_livesqlbench(
    tmp_path, monkeypatch,
):
    """Setting `BIRD_OTF_CACHE_ROOT` MUST NOT affect the livesqlbench-scoped
    root — they are independent overrides for independent benchmarks."""
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "BIRD_OTF_CACHE_ROOT", str(tmp_path / "elsewhere" / "mini_only"),
    )
    # Mini-interact follows the override…
    assert (
        paths.slayer_otf_cache_root(benchmark="mini_interact")
        == tmp_path / "elsewhere" / "mini_only"
    )
    # …but the livesqlbench root is unaffected.
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench")
        == main / "slayer_otf_cache_livesqlbench"
    )


def test_slayer_otf_cache_root_livesqlbench_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv(
        "BIRD_OTF_CACHE_ROOT_LIVESQLBENCH",
        "/data/slayer_otf_cache_livesqlbench",
    )
    assert (
        paths.slayer_otf_cache_root(benchmark="livesqlbench")
        == Path("/data/slayer_otf_cache_livesqlbench")
    )


def test_slayer_models_otf_root_mini_interact_is_legacy_root(
    tmp_path, monkeypatch,
):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="mini_interact")
        == main / "slayer_models_otf"
    )


def test_slayer_models_otf_root_none_benchmark_raises(tmp_path, monkeypatch):
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        paths.slayer_models_otf_root(benchmark=None)
    with pytest.raises(TypeError):
        paths.slayer_models_otf_root()  # benchmark is required (no default)


def test_slayer_models_otf_root_livesqlbench_is_parallel_root(
    tmp_path, monkeypatch,
):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_models_otf_root(benchmark="livesqlbench")
        == main / "slayer_models_otf_livesqlbench"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="livesqlbench")
        != paths.slayer_models_otf_root(benchmark="mini_interact")
    )


def test_slayer_models_otf_root_livesqlbench_env_override(tmp_path, monkeypatch):
    override = tmp_path / "data" / "slayer_models_otf_livesqlbench"
    monkeypatch.setenv(
        "BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH", str(override),
    )
    assert paths.slayer_models_otf_root(benchmark="livesqlbench") == override


def test_slayer_models_otf_root_legacy_env_does_not_steer_livesqlbench(
    tmp_path, monkeypatch,
):
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "BIRD_SLAYER_MODELS_OTF_ROOT",
        str(tmp_path / "elsewhere" / "mini_only_models_otf"),
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini_interact")
        == tmp_path / "elsewhere" / "mini_only_models_otf"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="livesqlbench")
        == main / "slayer_models_otf_livesqlbench"
    )


def test_slayer_models_otf_root_livesqlbench_env_override_wins_when_default_would_fail(
    tmp_path, monkeypatch,
):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(paths, "_LOOKUP_DIR", not_a_repo)
    monkeypatch.setenv(
        "BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH",
        "/data/slayer_models_otf_livesqlbench",
    )
    assert (
        paths.slayer_models_otf_root(benchmark="livesqlbench")
        == Path("/data/slayer_models_otf_livesqlbench")
    )


def test_unknown_benchmark_value_raises(tmp_path, monkeypatch):
    """An unrecognised benchmark name MUST raise — silent fallback to the
    legacy mini-interact root would let a typo silently mix artifacts."""
    _setup_main_and_worktree(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as exc_info:
        paths.slayer_otf_cache_root(benchmark="this-is-not-a-benchmark")
    assert "benchmark" in str(exc_info.value).lower()
    with pytest.raises(ValueError):
        paths.slayer_models_otf_root(benchmark="this-is-not-a-benchmark")


# ---------------------------------------------------------------------------
# Live invocation from this repo's actual checkout
# ---------------------------------------------------------------------------


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
