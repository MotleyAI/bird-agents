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


def test_annotator_actor_constructor_calls_bridge_bootstrap():
    """The AnnotatorActor constructor itself must call `_maybe_ensure_bridge`
    (the @ray.remote class can't be instantiated without a live Ray cluster, so
    AST-inspect the constructor source — this fails if the bootstrap call is
    removed, unlike a test that drives the mocked helper directly)."""
    import ast
    import inspect

    src = inspect.getsource(ray_app_annotator._build_annotator_actor_class)
    tree = ast.parse(src)
    # Find the AnnotatorActor.__init__ and assert it calls _maybe_ensure_bridge.
    init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init = node
            break
    assert init is not None, "AnnotatorActor.__init__ not found"
    called = {
        n.func.id
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_maybe_ensure_bridge" in called, (
        "AnnotatorActor.__init__ must call _maybe_ensure_bridge(cfg)"
    )


def test_run_annotator_pool_local_path_starts_bridge(monkeypatch):
    """The sequential/local annotator path (no AnnotatorActor) must also start
    the bridge for a registry model (Codex round-2)."""
    calls = []
    monkeypatch.setattr(
        ray_app_annotator, "_maybe_ensure_bridge",
        lambda cfg: calls.append(cfg.get("agent_model")),
    )

    class _NoHeartbeat:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, _name):  # start / stop_and_flush / … → no-op
            return lambda *a, **k: None

    monkeypatch.setattr(ray_app_annotator, "HeartbeatWriter", _NoHeartbeat)
    ray_app_annotator.run_annotator_pool(
        run_id="r1",
        instance_ids=[],  # no tasks: still must bring up the bridge first
        task_data_by_id={},
        cfg={"benchmark": "mini-interact", "agent_model": _DW, "model": _DW,
             "framework": "annotator", "no_subscription_auth": True},
        data_path_base="/tmp/data",
        num_actors=1,
        gcs_client=object(),
        local_only=True,
    )
    assert calls == [_DW]


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
