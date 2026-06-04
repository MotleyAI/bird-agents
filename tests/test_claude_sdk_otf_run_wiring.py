"""run.py + cloud wiring for the (narrowed) `claude_sdk_otf` framework.

After DEV-1507 `claude_sdk_otf` is livesqlbench / one-shot only. The
mini-interact / a-interact behavior lives in `claude_sdk_otf_ainteract`
(see `tests/test_claude_sdk_otf_ainteract_run_wiring.py`).

Locks down:
* `--framework claude_sdk_otf` is in the CLI choices (existing ones kept).
* `run_evaluation` branches to `ClaudeSDKOtfAgent` for one-shot/livesqlbench.
* `_validate_slayer_setup` requires on-the-fly; `_validate_one_shot_framework`
  accepts it; `_maybe_force_wipe_otf` purges its cache.
* Cloud maps the combo to the cache-only artifact (no reference / upload-back).
"""

from __future__ import annotations

import sys

import pytest


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


@pytest.mark.asyncio
async def test_run_evaluation_branches_to_otf_agent(monkeypatch, tmp_path):
    from bird_interact_agents import run as run_mod

    constructed = []

    class _Sentinel(Exception):
        pass

    class _FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise _Sentinel("stop")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf.ClaudeSDKOtfAgent",
        _FakeAgent, raising=False,
    )
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **kw: [])

    data_file = tmp_path / "x.jsonl"
    data_file.write_text("")
    gold = tmp_path / "gold.jsonl"
    gold.write_text("")
    with pytest.raises(_Sentinel):
        await run_mod.run_evaluation(
            data_path=str(data_file), data_dir=str(tmp_path),
            output_path=str(tmp_path / "eval.json"),
            mode="one-shot", query_mode="slayer",
            framework="claude_sdk_otf", slayer_setup="on-the-fly",
            reasoning_effort="high",
            dataset="livesqlbench-base-lite-sqlite",
        )
    assert constructed and constructed[0].get("slayer_setup") == "on-the-fly"
    # --reasoning-effort must thread through to the agent constructor.
    assert constructed[0].get("reasoning_effort") == "high"


def test_validate_slayer_setup_requires_on_the_fly():
    from bird_interact_agents import run as run_mod

    # pre-encoded must be rejected for claude_sdk_otf
    with pytest.raises(ValueError):
        run_mod._validate_slayer_setup(
            slayer_setup="pre-encoded", framework="claude_sdk_otf",
            query_mode="slayer", mode="one-shot",
        )
    # on-the-fly + slayer + one-shot must pass for the narrowed flavor.
    run_mod._validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk_otf",
        query_mode="slayer", mode="one-shot",
    )


def test_maybe_force_wipe_otf_purges_cache_for_claude_sdk_otf(monkeypatch):
    """Narrowed flavor is livesqlbench-only — wipe must target the
    livesqlbench-scoped cache root."""
    from bird_interact_agents import run as run_mod
    from bird_interact_agents.slayer_otf import reference_build as rb

    purged = {}
    monkeypatch.setattr(
        rb, "purge_caches", lambda root, dbs: purged.update(cache=set(dbs)) or set(dbs),
    )
    monkeypatch.setattr(
        rb, "purge_references", lambda root, dbs: set(),
    )
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk",
        dbs=["museum"], benchmark="livesqlbench-base-lite-sqlite",
    )
    assert purged.get("cache") == {"museum"}


def test_cli_rejects_claude_sdk_otf_with_pre_encoded(monkeypatch, tmp_path):
    from bird_interact_agents import run as run_mod

    data_file = tmp_path / "x.jsonl"
    data_file.write_text("")
    argv = [
        "prog",
        "--framework", "claude_sdk",
        "--slayer-setup", "pre-encoded",
        "--query-mode", "slayer",
        "--mode", "one-shot",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--data", str(data_file),
        "--db-path", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()


# ---------------------------------------------------------------------------
# Cloud: cache-only artifact, no reference / upload-back
# ---------------------------------------------------------------------------

def test_cloud_artifact_name_is_cache_only():
    from bird_interact_agents.cloud import gcs

    assert gcs.slayer_artifact_name("on-the-fly", "claude_sdk_otf") == "slayer_otf_cache"


def test_cloud_actor_downloads_cache_only():
    """Narrowed flavor is livesqlbench-only — actor downloads the cache for
    that benchmark."""
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "framework": "claude_sdk_otf",
        "slayer_setup": "on-the-fly",
        "dataset": "livesqlbench-base-lite-sqlite",
    }
    artifacts = {a for (a, _root, _req) in ray_app._slayer_artifacts_for(cfg)}
    assert artifacts == {"slayer_otf_cache"}
    assert "slayer_models_otf" not in artifacts


def test_cloud_driver_uploads_cache_only():
    from types import SimpleNamespace

    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        slayer_setup="on-the-fly", framework="claude_sdk_otf",
        dataset="livesqlbench-base-lite-sqlite",
    )
    names = {name for (_path, name, _req) in driver._slayer_uploads_for(args)}
    # cache only — no LLM-encoded reference upload-back for this framework
    assert names == {"slayer_otf_cache"}


def test_cloud_resubmit_preserves_reasoning_effort():
    """The manifest carries reasoning_effort and resubmit re-emits the flag so
    a db-grouped retry runs at the same effort as the original submit.
    Narrowed flavor: livesqlbench / one-shot."""
    from bird_interact_agents.cloud import driver

    manifest = {
        "framework": "claude_sdk_otf",
        "query_mode": "slayer",
        "mode": "one-shot",
        "agent_model": "anthropic/claude-opus-4-7",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "patience": 500,
        "max_depth": 3,
        "dataset": "livesqlbench-base-lite-sqlite",
        "gold_file": "/data/gold.jsonl",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
        "reasoning_effort": "high",
        "prompt_cache": True,
        "slayer_setup": "on-the-fly",
        "slayer_storage_root": "/data/slayer_models",
    }
    job_args = driver._build_resubmit_args(
        manifest, "rid", ["museum_1"], attempt=1,
    )
    assert "--reasoning-effort" in job_args
    i = job_args.index("--reasoning-effort")
    assert job_args[i + 1] == "high"
