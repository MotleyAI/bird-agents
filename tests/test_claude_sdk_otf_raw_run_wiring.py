"""run.py + cloud wiring for the raw OTF agent frameworks.

Locks down:
* `_validate_slayer_setup` skips validation for raw frameworks (they don't
  use SLayer at all).
* `run_evaluation` branches to the raw agent classes for the right combos.
* Cloud: raw frameworks have no SLayer artifacts to upload/download.
"""

from __future__ import annotations

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
# _validate_slayer_setup — raw frameworks skip validation entirely
# ---------------------------------------------------------------------------

def test_validate_slayer_setup_skips_for_raw_one_shot():
    """claude_sdk_otf_raw doesn't use SLayer — any slayer_setup value (or
    the default 'pre-encoded') must not raise."""
    from bird_interact_agents import run as run_mod

    # pre-encoded + raw framework: must NOT raise (raw agents ignore slayer_setup)
    run_mod._validate_slayer_setup(
        slayer_setup="pre-encoded", framework="claude_sdk_otf_raw",
        query_mode="raw", mode="one-shot",
    )
    # on-the-fly + raw framework: must also not raise
    run_mod._validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_raw",
        query_mode="raw", mode="one-shot",
    )


def test_validate_slayer_setup_skips_for_raw_ainteract():
    from bird_interact_agents import run as run_mod

    run_mod._validate_slayer_setup(
        slayer_setup="pre-encoded", framework="claude_sdk_otf_ainteract_raw",
        query_mode="raw", mode="a-interact",
    )
    run_mod._validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_ainteract_raw",
        query_mode="raw", mode="a-interact",
    )


def test_validate_slayer_setup_still_requires_on_the_fly_for_slayer_otf():
    """Regression guard: existing slayer OTF behavior must be unchanged."""
    from bird_interact_agents import run as run_mod

    with pytest.raises(ValueError):
        run_mod._validate_slayer_setup(
            slayer_setup="pre-encoded", framework="claude_sdk_otf",
            query_mode="slayer", mode="one-shot",
        )


# ---------------------------------------------------------------------------
# run_evaluation branches to raw agent classes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_evaluation_branches_to_raw_otf_agent(monkeypatch, tmp_path):
    from bird_interact_agents import run as run_mod

    constructed = []

    class _Sentinel(Exception):
        pass

    class _FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise _Sentinel("stop")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_raw.ClaudeSDKOtfRawAgent",
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
            mode="one-shot", query_mode="raw",
            framework="claude_sdk_otf_raw", slayer_setup="pre-encoded",
            reasoning_effort=None,
            dataset="livesqlbench-base-lite-sqlite", gold_file=str(gold),
        )
    assert constructed


@pytest.mark.asyncio
async def test_run_evaluation_branches_to_raw_ainteract_agent(monkeypatch, tmp_path):
    from bird_interact_agents import run as run_mod

    constructed = []

    class _Sentinel(Exception):
        pass

    class _FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise _Sentinel("stop")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.ClaudeSDKOtfAInteractRawAgent",
        _FakeAgent, raising=False,
    )
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **kw: [])

    data_file = tmp_path / "x.jsonl"
    data_file.write_text("")
    with pytest.raises(_Sentinel):
        await run_mod.run_evaluation(
            data_path=str(data_file), data_dir=str(tmp_path),
            output_path=str(tmp_path / "eval.json"),
            mode="a-interact", query_mode="raw",
            framework="claude_sdk_otf_ainteract_raw", slayer_setup="pre-encoded",
            reasoning_effort=None,
            dataset="mini-interact",
        )
    assert constructed


# ---------------------------------------------------------------------------
# Cloud: raw frameworks have no SLayer artifacts
# ---------------------------------------------------------------------------

def test_cloud_actor_has_no_slayer_artifacts_for_raw_otf():
    """claude_sdk_otf_raw uses no slayer_otf_cache or slayer_models_otf."""
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "framework": "claude_sdk_otf_raw",
        "slayer_setup": "pre-encoded",
        "dataset": "livesqlbench-base-lite-sqlite",
    }
    artifacts = list(ray_app._slayer_artifacts_for(cfg))
    assert artifacts == [], (
        f"raw framework must have zero slayer cloud artifacts; got {artifacts!r}"
    )


def test_cloud_actor_has_no_slayer_artifacts_for_raw_ainteract():
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "framework": "claude_sdk_otf_ainteract_raw",
        "slayer_setup": "pre-encoded",
        "dataset": "mini-interact",
    }
    artifacts = list(ray_app._slayer_artifacts_for(cfg))
    assert artifacts == [], (
        f"raw ainteract framework must have zero slayer cloud artifacts; got {artifacts!r}"
    )


def test_cloud_driver_has_no_slayer_uploads_for_raw_otf():
    from types import SimpleNamespace
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        slayer_setup="pre-encoded", framework="claude_sdk_otf_raw",
        dataset="livesqlbench-base-lite-sqlite",
    )
    uploads = list(driver._slayer_uploads_for(args))
    assert uploads == [], (
        f"raw framework must have zero slayer upload artifacts; got {uploads!r}"
    )


def test_cloud_driver_has_no_slayer_uploads_for_raw_ainteract():
    from types import SimpleNamespace
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        slayer_setup="pre-encoded", framework="claude_sdk_otf_ainteract_raw",
        dataset="mini-interact",
    )
    uploads = list(driver._slayer_uploads_for(args))
    assert uploads == [], (
        f"raw ainteract framework must have zero slayer upload artifacts; got {uploads!r}"
    )


def test_cloud_slayer_artifacts_unchanged_for_slayer_otf():
    """Regression guard: slayer OTF still gets its cache artifact."""
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "framework": "claude_sdk_otf",
        "slayer_setup": "on-the-fly",
        "dataset": "livesqlbench-base-lite-sqlite",
    }
    artifacts = {a for (a, _root, _req) in ray_app._slayer_artifacts_for(cfg)}
    assert "slayer_otf_cache" in artifacts
