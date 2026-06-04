"""DEV-1525: cloud-side path contract after benchmark rename.

Replaces the "unchanged" assertions from test_cloud_paths_unchanged.py with
the new uniform contract:
- mini-interact OTF roots live at slayer_otf_cache/mini-interact/ (nested)
- _cloud_benchmark derives the new canonical name from cfg["dataset"]
- Missing dataset raises, not silently defaults to mini_interact
- gated_gold_root is included in the cloud static call-site audit
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from bird_interact_agents import paths


# ---------------------------------------------------------------------------
# _cloud_benchmark — returns new canonical names
# ---------------------------------------------------------------------------


def test_cloud_benchmark_returns_new_canonical_name_for_mini_interact():
    from bird_interact_agents.cloud import ray_app

    assert ray_app._cloud_benchmark({"dataset": "mini-interact"}) == "mini-interact"


def test_cloud_benchmark_returns_livesqlbench_sqlite():
    from bird_interact_agents.cloud import ray_app

    assert (
        ray_app._cloud_benchmark({"dataset": "livesqlbench-base-lite-sqlite"})
        == "livesqlbench-base-lite-sqlite"
    )


def test_cloud_benchmark_returns_livesqlbench_postgres():
    from bird_interact_agents.cloud import ray_app

    assert (
        ray_app._cloud_benchmark({"dataset": "livesqlbench-base-lite"})
        == "livesqlbench-base-lite"
    )


def test_cloud_benchmark_returns_bird_interact_lite_exp():
    from bird_interact_agents.cloud import ray_app

    assert (
        ray_app._cloud_benchmark({"dataset": "bird-interact-lite-exp"})
        == "bird-interact-lite-exp"
    )


def test_cloud_benchmark_missing_dataset_raises():
    """An absent or empty 'dataset' in cfg must raise — no silent mini_interact default."""
    from bird_interact_agents.cloud import ray_app

    with pytest.raises((ValueError, KeyError)):
        ray_app._cloud_benchmark({})
    with pytest.raises((ValueError, KeyError)):
        ray_app._cloud_benchmark({"dataset": ""})


# ---------------------------------------------------------------------------
# Runtime: ray_app selects nested OTF roots for mini-interact
# ---------------------------------------------------------------------------


def test_ray_app_artifacts_select_mini_interact_nested_root(monkeypatch):
    """ray_app must request mini-interact's NESTED root, not the old flat root."""
    from bird_interact_agents.cloud import ray_app
    from bird_interact_agents import paths as _paths

    seen: list[tuple[str, str]] = []

    def _record_cache(*, benchmark):
        seen.append(("cache", benchmark))
        return Path("/data/otf_cache") / benchmark

    def _record_models(*, benchmark):
        seen.append(("models", benchmark))
        return Path("/data/models_otf") / benchmark

    monkeypatch.setattr(_paths, "slayer_otf_cache_root", _record_cache)
    monkeypatch.setattr(_paths, "slayer_models_otf_root", _record_models)

    cfg = {
        "framework": "claude_sdk",
        "slayer_setup": "on-the-fly",
        "dataset": "mini-interact",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert seen and all(bm == "mini-interact" for _, bm in seen), seen


# ---------------------------------------------------------------------------
# Static call-site: cloud modules pass explicit benchmark= to ALL path helpers
# including gated_gold_root
# ---------------------------------------------------------------------------


def test_cloud_literal_calls_pass_explicit_benchmark_including_gated_gold():
    """Every literal call to slayer_*_root and gated_gold_root in cloud modules
    must pass an explicit benchmark= kwarg."""
    from bird_interact_agents.cloud import driver, post_run_merge, upload_back

    sources = {
        mod.__name__.rsplit(".", 1)[-1]: Path(inspect.getsourcefile(mod) or "").read_text()
        for mod in (driver, post_run_merge, upload_back)
    }
    checked_attrs = ("slayer_otf_cache_root", "slayer_models_otf_root", "gated_gold_root")
    offending: list[tuple[str, int, str]] = []
    for mod_name, src in sources.items():
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in checked_attrs:
                continue
            if not any(kw.arg == "benchmark" for kw in node.keywords):
                offending.append((mod_name, node.lineno, func.attr))
    assert not offending, (
        "Cloud modules must pass benchmark= to path helpers. Offenders: "
        f"{offending!r}"
    )


# ---------------------------------------------------------------------------
# mini-interact resolves to NESTED root (not legacy flat root)
# ---------------------------------------------------------------------------


def test_mini_interact_resolves_to_nested_root(tmp_path, monkeypatch):
    """After the refactor, benchmark='mini-interact' must return the NESTED path
    slayer_otf_cache/mini-interact/ — NOT the old flat slayer_otf_cache/."""
    import subprocess

    def _init_repo(repo_dir):
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo_dir)], check=True)
        (repo_dir / "README.md").write_text("test\n")
        subprocess.run(["git", "-C", str(repo_dir), "add", "README.md"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir),
             "-c", "user.email=t@example.invalid", "-c", "user.name=Test",
             "commit", "-q", "-m", "init"],
            check=True,
        )

    paths._main_checkout_root_cached.cache_clear()
    for var in ("BIRD_OTF_CACHE_ROOT", "BIRD_SLAYER_MODELS_OTF_ROOT"):
        monkeypatch.delenv(var, raising=False)

    main = tmp_path / "main_repo"
    _init_repo(main)
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)

    cache = paths.slayer_otf_cache_root(benchmark="mini-interact")
    models = paths.slayer_models_otf_root(benchmark="mini-interact")

    # Must be NESTED
    assert cache == main / "slayer_otf_cache" / "mini-interact"
    assert models == main / "slayer_models_otf" / "mini-interact"

    # Must NOT be the old flat root
    assert cache != main / "slayer_otf_cache"
    assert models != main / "slayer_models_otf"

    paths._main_checkout_root_cached.cache_clear()
