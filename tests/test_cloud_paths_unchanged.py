"""Cloud-side benchmark contract.

`benchmark` is now REQUIRED on `paths.slayer_otf_cache_root` /
`slayer_models_otf_root` (no `None` default). The cloud is mini-interact-only
by construction today (no `--dataset`/`--gold-file` at submit, no `dataset` in
the run cfg), but it must STILL choose its benchmark EXPLICITLY rather than
rely on a default — so a forgotten benchmark can never silently mix artifacts.

The cloud derives the benchmark from the run config via small helpers
(`driver._submit_benchmark(args)`, `ray_app._cloud_benchmark(cfg)`) that
resolve to `"mini-interact"` today and to `"livesqlbench-base-lite-sqlite"` the moment a
`dataset` is ever plumbed through — without a hardcoded literal scattered
across call sites. `"mini-interact"` maps to the legacy
`slayer_otf_cache/` / `slayer_models_otf/` roots, so on-disk + cloud data
paths are unchanged.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

from bird_interact_agents import paths


# ---------------------------------------------------------------------------
# Benchmark derivation helpers — resolve to mini_interact today (no dataset),
# livesqlbench when a dataset is present.
# ---------------------------------------------------------------------------


def test_submit_benchmark_defaults_mini_interact():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace()  # no `dataset` attr at all
    assert driver._submit_benchmark(args) == "mini-interact"
    args_mi = argparse.Namespace(dataset="mini-interact")
    assert driver._submit_benchmark(args_mi) == "mini-interact"


def test_submit_benchmark_livesqlbench():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(dataset="livesqlbench-base-lite-sqlite")
    assert driver._submit_benchmark(args) == "livesqlbench-base-lite-sqlite"


def test_cloud_benchmark_mini_interact():
    from bird_interact_agents.cloud import ray_app

    assert ray_app._cloud_benchmark({"dataset": "mini-interact"}) == "mini-interact"


def test_cloud_benchmark_livesqlbench():
    from bird_interact_agents.cloud import ray_app

    assert ray_app._cloud_benchmark({"dataset": "livesqlbench-base-lite-sqlite"}) == "livesqlbench-base-lite-sqlite"


# ---------------------------------------------------------------------------
# Static call-site contract: every LITERAL `paths.slayer_*_root(...)` call in
# the cloud modules that build/select artifacts MUST pass an explicit
# `benchmark=` kwarg (no bare default). (ray_app dispatches dynamically via
# `root_fn(benchmark=...)`, covered by the runtime test below.)
# ---------------------------------------------------------------------------


def _cloud_module_sources():
    from bird_interact_agents.cloud import driver, post_run_merge, upload_back

    return {
        mod.__name__.rsplit(".", 1)[-1]: Path(
            inspect.getsourcefile(mod)
        ).read_text()
        for mod in (driver, post_run_merge, upload_back)
    }


def test_cloud_literal_calls_pass_explicit_benchmark():
    sources = _cloud_module_sources()
    offending: list[tuple[str, int, str]] = []
    for mod_name, src in sources.items():
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in (
                "slayer_otf_cache_root", "slayer_models_otf_root",
            ):
                continue
            if not any(kw.arg == "benchmark" for kw in node.keywords):
                offending.append((mod_name, node.lineno, func.attr))
    assert not offending, (
        "Cloud modules must pass an EXPLICIT benchmark= to the path helpers "
        "(no reliance on a default). Offenders: " f"{offending!r}"
    )


# ---------------------------------------------------------------------------
# Runtime: ray_app's dynamic `root_fn(benchmark=...)` selection uses the
# derived benchmark — mini_interact today.
# ---------------------------------------------------------------------------


def test_ray_app_artifacts_select_mini_interact_root(monkeypatch):
    from bird_interact_agents.cloud import ray_app
    from bird_interact_agents import paths as _paths

    seen: list[tuple[str, str]] = []

    def _record_cache(*, benchmark):
        seen.append(("cache", benchmark))
        return Path("/data") / f"otf_cache_{benchmark}"

    def _record_models(*, benchmark):
        seen.append(("models", benchmark))
        return Path("/data") / f"models_otf_{benchmark}"

    monkeypatch.setattr(_paths, "slayer_otf_cache_root", _record_cache)
    monkeypatch.setattr(_paths, "slayer_models_otf_root", _record_models)

    cfg = {"framework": "pydantic_ai_otf_encode", "slayer_setup": "on-the-fly",
           "dataset": "mini-interact"}
    ray_app._slayer_artifacts_for(cfg)
    assert seen and all(bm == "mini-interact" for _, bm in seen), seen


def _spy_path_helpers(monkeypatch):
    """Replace the two path helpers on the shared ``bird_interact_agents.paths``
    module with spies that record the benchmark kwarg they receive. Both
    driver and ray_app resolve through this same module object."""
    from bird_interact_agents import paths as _paths

    seen: list[tuple[str, str]] = []

    def _cache(*, benchmark):
        seen.append(("cache", benchmark))
        return Path("/data") / f"otf_cache_{benchmark}"

    def _models(*, benchmark):
        seen.append(("models", benchmark))
        return Path("/data") / f"models_otf_{benchmark}"

    monkeypatch.setattr(_paths, "slayer_otf_cache_root", _cache)
    monkeypatch.setattr(_paths, "slayer_models_otf_root", _models)
    return seen


def test_ray_app_artifacts_select_livesqlbench_root(monkeypatch):
    """The derived benchmark must actually FLOW into the artifact-root
    selection — a livesqlbench cfg selects the livesqlbench-scoped roots."""
    from bird_interact_agents.cloud import ray_app

    seen = _spy_path_helpers(monkeypatch)
    cfg = {
        "framework": "pydantic_ai_otf_encode", "slayer_setup": "on-the-fly",
        "dataset": "livesqlbench-base-lite-sqlite",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert seen and all(bm == "livesqlbench-base-lite-sqlite" for _, bm in seen), seen


def test_driver_uploads_select_benchmark_from_args(monkeypatch):
    """`driver._slayer_uploads_for` must select roots via the benchmark derived
    from the submit args — mini_interact by default, livesqlbench when the
    submit carries `dataset=livesqlbench` (so it's not a hardcoded literal)."""
    import argparse
    from bird_interact_agents.cloud import driver

    seen_mi = _spy_path_helpers(monkeypatch)
    args_mi = argparse.Namespace(
        framework="pydantic_ai_otf_encode", slayer_setup="on-the-fly",
        dataset="mini-interact",
    )
    driver._slayer_uploads_for(args_mi)
    assert seen_mi and all(bm == "mini-interact" for _, bm in seen_mi), seen_mi

    seen_lsb = _spy_path_helpers(monkeypatch)
    args_lsb = argparse.Namespace(
        framework="pydantic_ai_otf_encode", slayer_setup="on-the-fly",
        dataset="livesqlbench-base-lite-sqlite",
    )
    driver._slayer_uploads_for(args_lsb)
    assert seen_lsb and all(bm == "livesqlbench-base-lite-sqlite" for _, bm in seen_lsb), seen_lsb


def test_mini_interact_resolves_to_nested_roots(tmp_path, monkeypatch):
    """`benchmark="mini-interact"` uses a nested path under the shared cache
    root: `slayer_otf_cache/mini-interact/` (DEV-1525 unified all benchmarks
    to the same `slayer_otf_cache/<benchmark>/` layout)."""
    from tests.test_paths import _setup_main_and_worktree

    paths._main_checkout_root_cached.cache_clear()
    for var in (
        "BIRD_OTF_CACHE_ROOT", "BIRD_SLAYER_MODELS_OTF_ROOT",
        "BIRD_OTF_CACHE_ROOT_LIVESQLBENCH",
        "BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH",
    ):
        monkeypatch.delenv(var, raising=False)
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        == main / "slayer_otf_cache" / "mini-interact"
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        == main / "slayer_models_otf" / "mini-interact"
    )
    assert (
        paths.slayer_otf_cache_root(benchmark="mini-interact")
        != paths.slayer_otf_cache_root(benchmark="livesqlbench-base-lite-sqlite")
    )
    assert (
        paths.slayer_models_otf_root(benchmark="mini-interact")
        != paths.slayer_models_otf_root(benchmark="livesqlbench-base-lite-sqlite")
    )
