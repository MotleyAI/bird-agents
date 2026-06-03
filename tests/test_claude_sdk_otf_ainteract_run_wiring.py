"""run.py + cloud wiring for the `claude_sdk_otf_ainteract` framework.

Locks down:
* `run_evaluation` branches to `ClaudeSDKOtfAInteractAgent` and threads
  reasoning_effort + slayer_setup correctly.
* `_validate_slayer_setup` requires on-the-fly for the new flavor.
* `_maybe_force_wipe_otf` purges its cache scoped to mini-interact, with
  the benchmark kwarg threaded through to the path-roots.
* Cloud artifact name + uploads stay cache-only for the new flavor.
* `_build_resubmit_args` preserves `--reasoning-effort` for the new flavor.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# argparse choices
# ---------------------------------------------------------------------------

def _framework_choices_from_parser():
    import ast
    import inspect

    from bird_interact_agents import run as run_mod

    src = inspect.getsource(run_mod.main)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(getattr(node.func, "attr", None), "lower", lambda: "")() == "add_argument"
        ):
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--framework":
                for kw in node.keywords:
                    if kw.arg == "choices" and isinstance(kw.value, ast.List):
                        return {
                            elt.value for elt in kw.value.elts
                            if isinstance(elt, ast.Constant)
                        }
    raise AssertionError("could not find --framework choices in run.main")


def test_framework_choice_accepts_claude_sdk():
    assert "claude_sdk" in _framework_choices_from_parser()


def test_existing_framework_choices_preserved():
    choices = _framework_choices_from_parser()
    assert {"claude_sdk"}.issubset(choices)


# ---------------------------------------------------------------------------
# run_evaluation branch + agent threading
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_evaluation_branches_to_ainteract_agent(monkeypatch, tmp_path):
    from bird_interact_agents import run as run_mod

    constructed = []

    class _Sentinel(Exception):
        pass

    class _FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise _Sentinel("stop")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract."
        "ClaudeSDKOtfAInteractAgent",
        _FakeAgent, raising=False,
    )
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **kw: [])

    data_file = tmp_path / "x.jsonl"
    data_file.write_text("")
    with pytest.raises(_Sentinel):
        await run_mod.run_evaluation(
            data_path=str(data_file), data_dir=str(tmp_path),
            output_path=str(tmp_path / "eval.json"),
            mode="a-interact", query_mode="slayer",
            framework="claude_sdk_otf_ainteract", slayer_setup="on-the-fly",
            reasoning_effort="high",
            dataset="mini-interact",
        )
    assert constructed and constructed[0].get("slayer_setup") == "on-the-fly"
    assert constructed[0].get("reasoning_effort") == "high"


# ---------------------------------------------------------------------------
# Slayer-setup validator
# ---------------------------------------------------------------------------

def test_validate_slayer_setup_requires_on_the_fly_for_ainteract():
    from bird_interact_agents import run as run_mod

    # pre-encoded rejected for the new flavor.
    with pytest.raises(ValueError):
        run_mod._validate_slayer_setup(
            slayer_setup="pre-encoded", framework="claude_sdk_otf_ainteract",
            query_mode="slayer", mode="a-interact",
        )
    # on-the-fly + slayer + a-interact accepted.
    run_mod._validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_ainteract",
        query_mode="slayer", mode="a-interact",
    )


# ---------------------------------------------------------------------------
# _maybe_force_wipe_otf — benchmark kwarg plumbing
# ---------------------------------------------------------------------------

def test_maybe_force_wipe_otf_purges_cache_for_ainteract(monkeypatch):
    from bird_interact_agents import run as run_mod
    from bird_interact_agents.slayer_otf import reference_build as rb

    purged = {}
    monkeypatch.setattr(
        rb, "purge_caches",
        lambda root, dbs: purged.update(cache=set(dbs)) or set(dbs),
    )
    monkeypatch.setattr(rb, "purge_references", lambda root, dbs: set())
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk_otf_ainteract",
        dbs=["shop"], benchmark="mini-interact",
    )
    assert purged.get("cache") == {"shop"}


def test_maybe_force_wipe_otf_passes_benchmark_kwarg_for_ainteract(monkeypatch):
    """Verify the benchmark kwarg threads through to
    `paths.slayer_otf_cache_root(benchmark=...)` and
    `paths.slayer_models_otf_root(benchmark=...)` so the new flavor never
    accidentally targets the legacy / livesqlbench roots."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents import run as run_mod
    from bird_interact_agents.slayer_otf import reference_build as rb

    captured: dict[str, dict] = {}

    def _fake_cache_root(*a, **kwargs):
        captured["cache_kwargs"] = dict(kwargs)
        return "/tmp/fake-cache"

    def _fake_ref_root(*a, **kwargs):
        captured["ref_kwargs"] = dict(kwargs)
        return "/tmp/fake-ref"

    monkeypatch.setattr(paths_mod, "slayer_otf_cache_root", _fake_cache_root)
    monkeypatch.setattr(paths_mod, "slayer_models_otf_root", _fake_ref_root)
    monkeypatch.setattr(rb, "purge_caches", lambda root, dbs: set())
    monkeypatch.setattr(rb, "purge_references", lambda root, dbs: set())

    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk_otf_ainteract",
        dbs=["shop"], benchmark="mini-interact",
    )
    assert captured["cache_kwargs"] == {"benchmark": "mini-interact"}
    assert captured["ref_kwargs"] == {"benchmark": "mini-interact"}


# ---------------------------------------------------------------------------
# CLI rejection paths
# ---------------------------------------------------------------------------

def _argv(framework, mode, dataset, slayer_setup, *, tmp_path, gold_file=None):
    data_file = tmp_path / "x.jsonl"
    data_file.write_text("")
    argv = [
        "prog",
        "--framework", framework,
        "--slayer-setup", slayer_setup,
        "--query-mode", "slayer",
        "--mode", mode,
        "--dataset", dataset,
        "--data", str(data_file),
        "--db-path", str(tmp_path),
    ]
    if gold_file:
        argv += ["--gold-file", str(gold_file)]
    return argv


def test_cli_rejects_slayer_with_pre_encoded(monkeypatch, tmp_path, capsys):
    from bird_interact_agents import run as run_mod

    argv = _argv(
        "claude_sdk", "a-interact", "mini-interact",
        "pre-encoded", tmp_path=tmp_path,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    # The slayer-setup validator must fire.
    assert "on-the-fly" in err or "slayer-setup" in err


def test_cli_rejects_a_interact_with_livesqlbench(monkeypatch, tmp_path, capsys):
    """Dataset×mode gate: livesqlbench doesn't support a-interact."""
    from bird_interact_agents import run as run_mod

    gold = tmp_path / "gold.jsonl"
    gold.write_text("")
    argv = _argv(
        "claude_sdk", "a-interact", "livesqlbench-base-lite-sqlite",
        "on-the-fly", tmp_path=tmp_path, gold_file=gold,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    # The dataset-mode gate fires: livesqlbench rejects a-interact.
    assert "a-interact" in err or "livesqlbench" in err or "mode" in err


def test_cli_rejects_one_shot_with_mini_interact(monkeypatch, tmp_path, capsys):
    """mini-interact doesn't support one-shot."""
    from bird_interact_agents import run as run_mod

    argv = _argv(
        "claude_sdk", "one-shot", "mini-interact",
        "on-the-fly", tmp_path=tmp_path,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    assert "one-shot" in err or "mini-interact" in err or "mode" in err


# ---------------------------------------------------------------------------
# Cloud: cache-only artifact, no reference / upload-back
# ---------------------------------------------------------------------------

def test_cloud_artifact_name_is_cache_only_for_ainteract():
    from bird_interact_agents.cloud import gcs

    assert gcs.slayer_artifact_name(
        "on-the-fly", "claude_sdk_otf_ainteract",
    ) == "slayer_otf_cache"


def test_cloud_actor_downloads_cache_only_for_ainteract():
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "framework": "claude_sdk_otf_ainteract",
        "slayer_setup": "on-the-fly",
        "dataset": "mini-interact",
    }
    artifacts = {a for (a, _root, _req) in ray_app._slayer_artifacts_for(cfg)}
    assert artifacts == {"slayer_otf_cache"}
    assert "slayer_models_otf" not in artifacts


def test_cloud_driver_uploads_cache_only_for_ainteract():
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_ainteract",
        dataset="mini-interact",
    )
    names = {name for (_path, name, _req) in driver._slayer_uploads_for(args)}
    assert names == {"slayer_otf_cache"}


def test_cloud_driver_uploads_pass_mini_interact_benchmark(monkeypatch):
    """Pin the benchmark kwarg flowing into the cache-root helper for the
    ainteract framework."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.cloud import driver

    captured: dict[str, dict] = {}

    def _fake_cache_root(*a, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return "/tmp/fake-cache"

    monkeypatch.setattr(paths_mod, "slayer_otf_cache_root", _fake_cache_root)
    args = SimpleNamespace(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_ainteract",
        dataset="mini-interact",
    )
    driver._slayer_uploads_for(args)
    assert captured["kwargs"] == {"benchmark": "mini-interact"}


def test_cloud_actor_artifacts_pass_mini_interact_benchmark(monkeypatch):
    """The actor-side download (`ray_app._slayer_artifacts_for`) must also
    pass the mini-interact benchmark to `slayer_otf_cache_root` — otherwise
    the cluster would target the legacy / livesqlbench cache root."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.cloud import ray_app

    captured: dict[str, dict] = {}

    def _fake_cache_root(*a, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return "/tmp/fake-cache"

    monkeypatch.setattr(paths_mod, "slayer_otf_cache_root", _fake_cache_root)
    cfg = {
        "framework": "claude_sdk_otf_ainteract",
        "slayer_setup": "on-the-fly",
        "dataset": "mini-interact",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert captured["kwargs"] == {"benchmark": "mini-interact"}


def test_cloud_resubmit_preserves_reasoning_effort_for_ainteract():
    """A db-grouped retry must re-emit --reasoning-effort for the new flavor
    so the second wave runs at the same effort as the original submit."""
    from bird_interact_agents.cloud import driver

    manifest = {
        "framework": "claude_sdk_otf_ainteract",
        "query_mode": "slayer",
        "mode": "a-interact",
        "agent_model": "anthropic/claude-opus-4-7",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "patience": 500,
        "max_depth": 3,
        "dataset": "mini-interact",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
        "reasoning_effort": "high",
        "prompt_cache": True,
        "slayer_setup": "on-the-fly",
        "slayer_storage_root": "/data/slayer_models",
    }
    job_args = driver._build_resubmit_args(
        manifest, "rid", ["households_7"], attempt=1,
    )
    assert "--reasoning-effort" in job_args
    i = job_args.index("--reasoning-effort")
    assert job_args[i + 1] == "high"
    # And the resubmit emits the framework string verbatim.
    assert "--framework" in job_args
    i = job_args.index("--framework")
    assert job_args[i + 1] == "claude_sdk_otf_ainteract"
