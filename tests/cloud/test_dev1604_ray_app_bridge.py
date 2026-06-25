"""DEV-1604: the actor brings up the bridge proxy BEFORE building its runner.

The bridge proxy must exist (and `os.environ[base_url_env]` must point at it)
before the cached SDK runner / first task is built, because the SDK subprocess
inherits `ANTHROPIC_BASE_URL` from `sdk_session_env`, which reads the override
at option-build time (Codex #9). So `ensure_bridge_proxy_for_actor` runs ahead
of `_maybe_build_cached_runner` in BOTH `_LocalActor` and `WorkerActor`.

It runs ONLY when the agent provider needs a bridge — Doubleword (always) and
z.ai per-token — never for Anthropic, Moonshot, or z.ai coding-plan.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import ray_app

RUN_ID = "20260625T1300-claudesdkv1-slayer-abc123"
_DW = "doubleword/zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"


@pytest.fixture
def actor_harness(monkeypatch):
    """Stub the actor's heavy init side-effects and record call ordering."""
    order: list[str] = []
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: object())
    monkeypatch.setattr(ray_app, "download_benchmark_data", lambda *a, **k: None)
    monkeypatch.setattr(
        ray_app, "_snapshot_initial_seed_fps", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        ray_app, "_maybe_build_cached_runner",
        lambda _cfg: order.append("runner"),
    )

    def _fake_ensure(model, cfg):
        order.append("bridge")
        return "http://127.0.0.1:8788"

    monkeypatch.setattr(ray_app, "ensure_bridge_proxy_for_actor", _fake_ensure)
    # Clean ambient Anthropic creds so the OAuth invariant stays quiet for a
    # registry run.
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
                "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    return order


def _cfg(agent_model: str, zai_billing: str = "coding-plan") -> dict:
    return {
        "framework": "claude_sdk_v1",
        "agent_model": agent_model,
        "query_mode": "raw",
        "zai_billing": zai_billing,
    }


def test_doubleword_actor_bridges_before_runner(actor_harness, monkeypatch):
    order = actor_harness
    ray_app._LocalActor(_cfg(_DW), RUN_ID, 1, gcs_client=object())
    assert order == ["bridge", "runner"]


def test_zai_per_token_actor_bridges(actor_harness):
    order = actor_harness
    ray_app._LocalActor(_cfg(_GLM, "per-token"), RUN_ID, 1, gcs_client=object())
    assert order == ["bridge", "runner"]


def test_zai_coding_plan_actor_does_not_bridge(actor_harness):
    order = actor_harness
    ray_app._LocalActor(_cfg(_GLM, "coding-plan"), RUN_ID, 1, gcs_client=object())
    assert order == ["runner"]


def test_anthropic_actor_does_not_bridge(actor_harness):
    order = actor_harness
    ray_app._LocalActor(
        _cfg("anthropic/claude-sonnet-4-6"), RUN_ID, 1, gcs_client=object()
    )
    assert order == ["runner"]


def test_maybe_ensure_bridge_helper(monkeypatch):
    """Both `_LocalActor` AND the real Ray `WorkerActor` route their bridge
    bring-up through this one shared helper, so testing it once covers both
    actor classes (the nested WorkerActor can't be built without a live Ray)."""
    calls = []
    monkeypatch.setattr(
        ray_app, "ensure_bridge_proxy_for_actor",
        lambda model, cfg: calls.append((model, cfg.get("zai_billing"))),
    )
    ray_app._maybe_ensure_bridge(_cfg(_DW))
    ray_app._maybe_ensure_bridge(_cfg(_GLM, "per-token"))
    ray_app._maybe_ensure_bridge(_cfg(_GLM, "coding-plan"))
    ray_app._maybe_ensure_bridge(_cfg("anthropic/claude-sonnet-4-6"))
    assert calls == [(_DW, "coding-plan"), (_GLM, "per-token")]


def test_run_pool_folds_zai_billing_into_actor_cfg(monkeypatch, fake_gcs_bucket):
    """`run_pool` must place `zai_billing` into the cfg the actor reads — not
    merely accept the kwarg. Capture the cfg a dispatched actor receives."""
    client, _store = fake_gcs_bucket
    # A registry (zai/) run forbids ambient Anthropic creds in the actor env.
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
                "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    monkeypatch.setattr(
        ray_app, "ensure_bridge_proxy_for_actor", lambda model, cfg: None
    )

    async def _fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True,
            "total_reward": 1.0, "duration_s": 0.01, "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", _fake_run_one_task
    )

    captured: dict = {}

    class _CaptureActor(ray_app._LocalActor):
        def __init__(self, cfg, run_id, attempt):
            captured.update(cfg)
            super().__init__(cfg, run_id, attempt)

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1"],
        framework="claude_sdk_v1",
        query_mode="raw",
        mode="one-shot",
        agent_model=_GLM,
        num_actors=1,
        attempt=1,
        task_data_by_id={"db_a_1": {"instance_id": "db_a_1",
                                    "selected_database": "db_a"}},
        dataset="mini-interact",
        zai_billing="per-token",
        local_only=True,
        actor_cls=_CaptureActor,
    )
    assert captured["zai_billing"] == "per-token"


def test_ray_app_main_threads_zai_billing_to_run_pool(monkeypatch):
    """`--zai-billing` parsed by the in-cluster ray_app main must reach
    `run_pool` (which folds it into the actor cfg). Capture the kwarg."""
    captured = {}
    monkeypatch.setattr(ray_app, "_load_secrets_file", lambda f: {})
    monkeypatch.setattr(ray_app, "download_benchmark_data", lambda *a, **k: None)
    monkeypatch.setattr(ray_app, "_load_task_data", lambda *a, **k: {})
    monkeypatch.setattr(ray_app, "run_pool", lambda **kw: captured.update(kw))
    ray_app.main([
        "--run-id", RUN_ID,
        "--attempt", "1",
        "--framework", "claude_sdk_v1",
        "--query-mode", "raw",
        "--mode", "one-shot",
        "--agent-model", _GLM,
        "--user-sim-model", "anthropic/claude-haiku-4-5-20251001",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--instance-ids", "alien_1",
        "--zai-billing", "per-token",
    ])
    assert captured["zai_billing"] == "per-token"


def test_ray_app_main_zai_billing_defaults_coding_plan(monkeypatch):
    captured = {}
    monkeypatch.setattr(ray_app, "_load_secrets_file", lambda f: {})
    monkeypatch.setattr(ray_app, "download_benchmark_data", lambda *a, **k: None)
    monkeypatch.setattr(ray_app, "_load_task_data", lambda *a, **k: {})
    monkeypatch.setattr(ray_app, "run_pool", lambda **kw: captured.update(kw))
    ray_app.main([
        "--run-id", RUN_ID,
        "--attempt", "1",
        "--framework", "claude_sdk_v1",
        "--query-mode", "raw",
        "--mode", "one-shot",
        "--agent-model", _DW,
        "--user-sim-model", "anthropic/claude-haiku-4-5-20251001",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--instance-ids", "alien_1",
    ])
    assert captured["zai_billing"] == "coding-plan"
