"""run.py + cloud wiring for the new `claude_sdk_otf_ainteract` framework
(DEV-1507).

Locks down:
* `--framework claude_sdk_otf_ainteract` is in the CLI choices alongside the
  narrowed `claude_sdk_otf`.
* `run_evaluation` branches to `ClaudeSDKOtfAInteractAgent` and threads
  reasoning_effort + slayer_setup correctly.
* `_validate_slayer_setup` requires on-the-fly for the new flavor.
* `_validate_one_shot_framework` REJECTS the new flavor (a-interact only).
* `_maybe_force_wipe_otf` purges its cache scoped to mini_interact, with
  the benchmark kwarg threaded through to the path-roots.
* New `_validate_framework_dataset_mode` rejects every mismatched
  (framework, dataset, mode) tuple — including oracle paths.
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


def test_framework_choice_accepts_claude_sdk_otf_ainteract():
    assert "claude_sdk_otf_ainteract" in _framework_choices_from_parser()


def test_existing_framework_choices_preserved():
    choices = _framework_choices_from_parser()
    assert {
        "claude_sdk", "pydantic_ai", "pydantic_ai_recursive",
        "pydantic_ai_otf_encode", "mcp_agent", "agno", "smolagents",
        "claude_sdk_otf", "claude_sdk_otf_ainteract",
    }.issubset(choices)


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
            dataset="mini_interact",
        )
    assert constructed and constructed[0].get("slayer_setup") == "on-the-fly"
    assert constructed[0].get("reasoning_effort") == "high"


# ---------------------------------------------------------------------------
# Slayer-setup + one-shot framework validators
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


def test_validate_one_shot_framework_rejects_ainteract():
    """ainteract is a-interact-only; the one-shot validator must reject it
    (Codex finding: keep `claude_sdk_otf`, do NOT add the new flavor)."""
    from bird_interact_agents import run as run_mod

    with pytest.raises(ValueError):
        run_mod._validate_one_shot_framework(
            mode="one-shot", query_mode="slayer",
            framework="claude_sdk_otf_ainteract",
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
        dbs=["shop"], benchmark="mini_interact",
    )
    assert purged.get("cache") == {"shop"}


def test_maybe_force_wipe_otf_passes_benchmark_kwarg_for_ainteract(monkeypatch):
    """Codex HIGH#2: verify the benchmark kwarg threads through to
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
        dbs=["shop"], benchmark="mini_interact",
    )
    assert captured["cache_kwargs"] == {"benchmark": "mini_interact"}
    assert captured["ref_kwargs"] == {"benchmark": "mini_interact"}


# ---------------------------------------------------------------------------
# _validate_framework_dataset_mode — NEW gate (Codex HIGH#1)
# ---------------------------------------------------------------------------

def test_validate_framework_dataset_mode_accepts_bound_pairs():
    from bird_interact_agents import run as run_mod

    # claude_sdk_otf bound to livesqlbench + one-shot.
    run_mod._validate_framework_dataset_mode(
        framework="claude_sdk_otf", dataset="livesqlbench", mode="one-shot",
    )
    # claude_sdk_otf_ainteract bound to mini_interact + a-interact.
    run_mod._validate_framework_dataset_mode(
        framework="claude_sdk_otf_ainteract",
        dataset="mini_interact", mode="a-interact",
    )


def test_validate_framework_dataset_mode_rejects_wrong_dataset():
    from bird_interact_agents import run as run_mod

    with pytest.raises(ValueError, match=r"livesqlbench|claude_sdk_otf"):
        run_mod._validate_framework_dataset_mode(
            framework="claude_sdk_otf", dataset="mini_interact",
            mode="one-shot",
        )
    with pytest.raises(ValueError, match=r"mini_interact|ainteract"):
        run_mod._validate_framework_dataset_mode(
            framework="claude_sdk_otf_ainteract", dataset="livesqlbench",
            mode="a-interact",
        )


def test_validate_framework_dataset_mode_rejects_oracle_for_both():
    """Codex HIGH#1: oracle must NOT bypass the new gate, because
    `run_oracle_task` short-circuits framework dispatch and could otherwise
    run a mismatched framework name with no harm but a silent intent
    mismatch. Both flavors reject `mode != bound_mode`, including oracle."""
    from bird_interact_agents import run as run_mod

    # claude_sdk_otf + oracle on the bound dataset is STILL rejected.
    with pytest.raises(ValueError):
        run_mod._validate_framework_dataset_mode(
            framework="claude_sdk_otf", dataset="livesqlbench",
            mode="oracle",
        )
    with pytest.raises(ValueError):
        run_mod._validate_framework_dataset_mode(
            framework="claude_sdk_otf_ainteract", dataset="mini_interact",
            mode="oracle",
        )


def test_validate_framework_dataset_mode_ignores_other_frameworks():
    """Other frameworks are not bound by this gate."""
    from bird_interact_agents import run as run_mod

    # Should NOT raise for non-OTF frameworks regardless of dataset / mode.
    run_mod._validate_framework_dataset_mode(
        framework="pydantic_ai", dataset="mini_interact", mode="a-interact",
    )
    run_mod._validate_framework_dataset_mode(
        framework="pydantic_ai_recursive", dataset="livesqlbench",
        mode="one-shot",
    )
    run_mod._validate_framework_dataset_mode(
        framework="claude_sdk", dataset="mini_interact", mode="oracle",
    )


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


def test_cli_rejects_ainteract_with_pre_encoded(monkeypatch, tmp_path, capsys):
    from bird_interact_agents import run as run_mod

    argv = _argv(
        "claude_sdk_otf_ainteract", "a-interact", "mini_interact",
        "pre-encoded", tmp_path=tmp_path,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    # Codex HIGH#2: the error must mention slayer-setup / on-the-fly so we
    # know the slayer-setup validator fired, not a different gate.
    assert "on-the-fly" in err or "slayer-setup" in err


def test_cli_rejects_ainteract_with_livesqlbench(monkeypatch, tmp_path, capsys):
    """Dataset×framework gate: ainteract is bound to mini_interact."""
    from bird_interact_agents import run as run_mod

    gold = tmp_path / "gold.jsonl"
    gold.write_text("")
    argv = _argv(
        "claude_sdk_otf_ainteract", "a-interact", "livesqlbench",
        "on-the-fly", tmp_path=tmp_path, gold_file=gold,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    # The DATASET-mode gate fires first for ainteract+livesqlbench (because
    # livesqlbench rejects a-interact). Accept either gate's message —
    # they're both load-bearing and the test is in the right "rejected"
    # zone. The dataset-mode gate's message names the offending mode/dataset.
    assert "a-interact" in err or "claude_sdk_otf_ainteract" in err or "livesqlbench" in err


def test_cli_rejects_ainteract_with_one_shot_mode(monkeypatch, tmp_path, capsys):
    """ainteract is a-interact-only; one-shot rejected by
    `_validate_one_shot_framework`."""
    from bird_interact_agents import run as run_mod

    gold = tmp_path / "gold.jsonl"
    gold.write_text("")
    argv = _argv(
        "claude_sdk_otf_ainteract", "one-shot", "livesqlbench",
        "on-the-fly", tmp_path=tmp_path, gold_file=gold,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    # Either the one-shot framework gate or the new dataset×framework gate
    # may win; both name the offending framework.
    assert "claude_sdk_otf_ainteract" in err or "one-shot" in err


def test_cli_rejects_claude_sdk_otf_with_mini_interact_oracle(
    monkeypatch, tmp_path, capsys,
):
    """Codex HIGH#1 / MED#6: claude_sdk_otf + mini_interact + oracle would
    pass `_validate_dataset_mode` (mini_interact supports oracle). The new
    `_validate_framework_dataset_mode` gate is the ONLY validator that
    rejects this combo — pin the error message to the new gate so we know
    it fired."""
    from bird_interact_agents import run as run_mod

    argv = [
        "prog",
        "--framework", "claude_sdk_otf",
        # Oracle doesn't reach _validate_slayer_setup's on-the-fly path
        # because oracle isn't in (a-interact, one-shot). Use pre-encoded so
        # that gate also passes; the new gate is left to do the work.
        "--query-mode", "slayer",
        "--mode", "oracle",
        "--dataset", "mini_interact",
        "--slayer-setup", "pre-encoded",
        "--data", str(tmp_path / "x.jsonl"),
        "--db-path", str(tmp_path),
    ]
    (tmp_path / "x.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    # The new gate's message must name the bound dataset (livesqlbench) for
    # claude_sdk_otf, or the framework itself.
    assert (
        "livesqlbench" in err or "claude_sdk_otf" in err
    ), f"expected the new framework gate to fire; got: {err!r}"


def test_cli_rejects_ainteract_with_livesqlbench_oracle(
    monkeypatch, tmp_path, capsys,
):
    """The symmetric oracle case for the new flavor."""
    from bird_interact_agents import run as run_mod

    gold = tmp_path / "gold.jsonl"
    gold.write_text("")
    argv = [
        "prog",
        "--framework", "claude_sdk_otf_ainteract",
        "--query-mode", "slayer",
        "--mode", "oracle",
        "--dataset", "livesqlbench",
        "--slayer-setup", "pre-encoded",
        "--gold-file", str(gold),
        "--data", str(tmp_path / "x.jsonl"),
        "--db-path", str(tmp_path),
    ]
    (tmp_path / "x.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()
    err = capsys.readouterr().err
    assert (
        "mini_interact" in err or "claude_sdk_otf_ainteract" in err
    ), f"expected the new framework gate to fire; got: {err!r}"


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
        "dataset": "mini_interact",
    }
    artifacts = {a for (a, _root, _req) in ray_app._slayer_artifacts_for(cfg)}
    assert artifacts == {"slayer_otf_cache"}
    assert "slayer_models_otf" not in artifacts


def test_cloud_driver_uploads_cache_only_for_ainteract():
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_ainteract",
        dataset="mini_interact",
    )
    names = {name for (_path, name, _req) in driver._slayer_uploads_for(args)}
    assert names == {"slayer_otf_cache"}


def test_cloud_driver_uploads_pass_mini_interact_benchmark(monkeypatch):
    """Codex LOW#7: pin the benchmark kwarg flowing into the cache-root
    helper for the ainteract framework."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.cloud import driver

    captured: dict[str, dict] = {}

    def _fake_cache_root(*a, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return "/tmp/fake-cache"

    monkeypatch.setattr(paths_mod, "slayer_otf_cache_root", _fake_cache_root)
    args = SimpleNamespace(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_ainteract",
        dataset="mini_interact",
    )
    driver._slayer_uploads_for(args)
    assert captured["kwargs"] == {"benchmark": "mini_interact"}


def test_cloud_actor_artifacts_pass_mini_interact_benchmark(monkeypatch):
    """Codex MED#3: the actor-side download (`ray_app._slayer_artifacts_for`)
    must also pass the mini_interact benchmark to `slayer_otf_cache_root` —
    otherwise the cluster would target the legacy / livesqlbench cache root
    for the new flavor."""
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
        "dataset": "mini_interact",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert captured["kwargs"] == {"benchmark": "mini_interact"}


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
        "dataset": "mini_interact",
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
    # And the resubmit emits the new framework string verbatim.
    assert "--framework" in job_args
    i = job_args.index("--framework")
    assert job_args[i + 1] == "claude_sdk_otf_ainteract"
