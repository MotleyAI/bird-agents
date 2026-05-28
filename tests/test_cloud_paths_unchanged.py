"""DEV-1462 — regression test pinning the cloud-side mini-interact contract.

dev-1470 added `cloud/upload_back.py` + `cloud/post_run_merge.py`. Those
read `paths.slayer_models_otf_root()` (and `slayer_otf_cache_root()`)
with NO `benchmark` argument — that's intentional: cloud-side livesqlbench
support is deferred (plan B8). After B1 parametrizes the helpers with an
optional `benchmark` kwarg, the cloud callers must keep resolving to the
legacy mini-interact roots — no silent shift to a livesqlbench scope.

This is a worktree-safety + deferred-boundary guard: greps the cloud
modules for `paths.slayer_*_root` calls AND verifies the runtime values
match the helpers' default (`benchmark=None`) outputs.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from bird_interact_agents import paths


def _cloud_module_sources():
    """Read the two cloud-side modules' source as text for grep."""
    from bird_interact_agents.cloud import post_run_merge, upload_back

    return {
        "post_run_merge": Path(inspect.getsourcefile(post_run_merge)).read_text(),
        "upload_back": Path(inspect.getsourcefile(upload_back)).read_text(),
    }


def test_cloud_modules_call_helpers_without_benchmark_kwarg():
    """A static AST walk: every `paths.slayer_otf_cache_root(...)` and
    `paths.slayer_models_otf_root(...)` call site in the cloud modules
    must omit the `benchmark` keyword (== legacy mini-interact root)."""
    sources = _cloud_module_sources()
    offending: list[tuple[str, int, str]] = []
    for mod_name, src in sources.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match `<x>.slayer_otf_cache_root` / `<x>.slayer_models_otf_root`
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in (
                "slayer_otf_cache_root", "slayer_models_otf_root",
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "benchmark":
                    offending.append((mod_name, node.lineno, func.attr))
    assert not offending, (
        "Cloud modules must NOT pass benchmark= to the path helpers — "
        "the dev-1470 cloud contract is mini-interact-only and the per-"
        "benchmark plumbing is deferred (B8). Offenders: "
        f"{offending!r}"
    )


def test_runtime_resolution_matches_legacy_root(tmp_path, monkeypatch):
    """A live check: the helpers, called with NO benchmark, must resolve
    to the legacy `slayer_otf_cache/` and `slayer_models_otf/` dirs under
    the main checkout — exactly what the cloud modules read."""
    from tests.test_paths import _setup_main_and_worktree

    # The autouse `_isolate_paths` fixture lives in test_paths.py and
    # doesn't fire here — clear the lru_cache manually before swapping
    # the lookup dir, else `main_checkout_root` returns the real checkout.
    paths._main_checkout_root_cached.cache_clear()
    for var in (
        "BIRD_OTF_CACHE_ROOT", "BIRD_SLAYER_MODELS_OTF_ROOT",
        "BIRD_OTF_CACHE_ROOT_LIVESQLBENCH",
        "BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH",
    ):
        monkeypatch.delenv(var, raising=False)
    main, _wt = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert paths.slayer_otf_cache_root() == main / "slayer_otf_cache"
    assert paths.slayer_models_otf_root() == main / "slayer_models_otf"
    # And these MUST differ from the livesqlbench-scoped roots.
    assert (
        paths.slayer_otf_cache_root()
        != paths.slayer_otf_cache_root(benchmark="livesqlbench")
    )
    assert (
        paths.slayer_models_otf_root()
        != paths.slayer_models_otf_root(benchmark="livesqlbench")
    )
