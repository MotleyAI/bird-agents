"""DEV-1604: the annotator is provider-aware and bridge-capable.

The annotator was Anthropic-only (`provider_aware=False`). It is now
provider-aware, so it can run a registry open-weight model (Doubleword / z.ai
per-token) through the same Anthropic⇄OpenAI bridge as the OTF agents — with no
annotator-specific bridge code, because it routes through `hermetic_claude_sdk_
session` and the shared `_maybe_ensure_bridge` actor seam.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import driver, ray_app_annotator


_DW = "doubleword/zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"


def test_annotator_agent_session_is_provider_aware():
    """The annotator no longer forces provider_aware=False — a registry model
    must reach the registry session env (and thus the bridge override)."""
    import inspect

    from bird_interact_agents.agents.annotator import agent

    src = inspect.getsource(agent)
    assert "provider_aware=False" not in src


def test_annotator_job_args_emit_subscription_flag():
    import argparse

    args = argparse.Namespace(
        benchmark="mini-interact", agent_model=_DW, effort="medium",
        workers=1, actors_per_worker=1, instance_ids=["alien_1"],
        override=False, no_subscription_auth=True,
    )
    job_args = driver._build_annotator_job_args(args, "r1")
    assert "--no-subscription-auth" in job_args  # per-token bridge
    assert job_args[job_args.index("--model") + 1] == _DW


def test_annotator_resubmit_args_emit_subscription_flag():
    manifest = {
        "dataset": "mini-interact", "agent_model": _GLM, "effort": "medium",
        "override": False, "no_subscription_auth": True,
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }
    job_args = driver._build_annotator_resubmit_args(
        manifest, "r1", ["alien_1"], attempt=2
    )
    assert "--no-subscription-auth" in job_args


def test_annotator_actor_bridges_for_registry_model(monkeypatch):
    """The annotator cloud actor brings up the bridge when its model is a
    registry provider — via the shared `_maybe_ensure_bridge` seam."""
    calls = []
    monkeypatch.setattr(
        ray_app_annotator, "_maybe_ensure_bridge",
        lambda cfg: calls.append((cfg.get("agent_model"),
                                  cfg.get("no_subscription_auth"))),
    )
    cfg = {
        "benchmark": "mini-interact", "agent_model": _DW, "model": _DW,
        "framework": "annotator", "no_subscription_auth": True,
    }
    ray_app_annotator._maybe_ensure_bridge(cfg)
    assert calls == [(_DW, True)]


def test_annotator_main_threads_subscription_flag_into_cfg(monkeypatch):
    """`--subscription-auth` parsed by the in-cluster annotator main reaches the
    actor cfg as no_subscription_auth so the bridge decision is correct."""
    captured = {}
    monkeypatch.setattr(ray_app_annotator, "_load_secrets_file", lambda f: {})
    monkeypatch.setattr(
        ray_app_annotator, "download_benchmark_data", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ray_app_annotator, "_load_annotator_task_data", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        ray_app_annotator, "run_annotator_pool",
        lambda **kw: captured.update(kw),
    )
    ray_app_annotator.main([
        "--run-id", "r1",
        "--benchmark", "mini-interact",
        "--model", _GLM,
        "--instance-ids", "alien_1",
        "--no-subscription-auth",
    ])
    assert captured["cfg"]["no_subscription_auth"] is True
    assert captured["cfg"]["agent_model"] == _GLM


def test_annotator_main_subscription_auth_coding_plan(monkeypatch):
    captured = {}
    monkeypatch.setattr(ray_app_annotator, "_load_secrets_file", lambda f: {})
    monkeypatch.setattr(
        ray_app_annotator, "download_benchmark_data", lambda *a, **k: None
    )
    monkeypatch.setattr(
        ray_app_annotator, "_load_annotator_task_data", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        ray_app_annotator, "run_annotator_pool",
        lambda **kw: captured.update(kw),
    )
    ray_app_annotator.main([
        "--run-id", "r1",
        "--benchmark", "mini-interact",
        "--model", _GLM,
        "--instance-ids", "alien_1",
        "--subscription-auth",
    ])
    assert captured["cfg"]["no_subscription_auth"] is False
