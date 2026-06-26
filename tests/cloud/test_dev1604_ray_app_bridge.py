"""DEV-1604: the actor brings up the bridge proxy BEFORE building its runner.

The bridge proxy must exist (and `os.environ[base_url_env]` must point at it)
before the cached SDK runner / first task is built, because the SDK subprocess
inherits `ANTHROPIC_BASE_URL` from `sdk_session_env`, which reads the override
at option-build time. So `_maybe_ensure_bridge` runs ahead of
`_maybe_build_cached_runner` in BOTH `_LocalActor` and `WorkerActor`.

It runs ONLY when the agent provider needs a bridge — Doubleword (always) and
z.ai on the per-token path (the recycled `--subscription-auth` flag, carried as
`no_subscription_auth=True`) — never for Anthropic, Moonshot, or z.ai
coding-plan (`no_subscription_auth=False`).
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
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
                "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    return order


def _cfg(agent_model: str, no_subscription_auth: bool = True) -> dict:
    return {
        "framework": "claude_sdk_v1",
        "agent_model": agent_model,
        "query_mode": "raw",
        "no_subscription_auth": no_subscription_auth,
    }


def test_doubleword_actor_bridges_before_runner(actor_harness):
    order = actor_harness
    ray_app._LocalActor(_cfg(_DW), RUN_ID, 1, gcs_client=object())
    assert order == ["bridge", "runner"]


def test_zai_per_token_actor_bridges(actor_harness):
    order = actor_harness
    ray_app._LocalActor(_cfg(_GLM, True), RUN_ID, 1, gcs_client=object())
    assert order == ["bridge", "runner"]


def test_zai_coding_plan_actor_does_not_bridge(actor_harness):
    order = actor_harness
    ray_app._LocalActor(_cfg(_GLM, False), RUN_ID, 1, gcs_client=object())
    assert order == ["runner"]


def test_anthropic_actor_does_not_bridge(actor_harness):
    order = actor_harness
    ray_app._LocalActor(
        _cfg("anthropic/claude-sonnet-4-6"), RUN_ID, 1, gcs_client=object()
    )
    assert order == ["runner"]


def test_maybe_ensure_bridge_skips_non_sdk_framework(monkeypatch):
    """The bridge is claude_sdk-specific. A non-SDK framework (pydantic_ai) on a
    doubleword model must NOT start a proxy — litellm reaches it directly."""
    calls = []
    monkeypatch.setattr(
        ray_app, "ensure_bridge_proxy_for_actor",
        lambda model, cfg: calls.append(model),
    )
    ray_app._maybe_ensure_bridge({
        "framework": "pydantic_ai", "agent_model": _DW, "query_mode": "raw",
        "no_subscription_auth": True,
    })
    assert calls == []


def test_maybe_ensure_bridge_helper(monkeypatch):
    """Both `_LocalActor` AND the real Ray `WorkerActor` route their bridge
    bring-up through this one shared helper, so testing it once covers both
    actor classes (the nested WorkerActor can't be built without a live Ray)."""
    calls = []
    monkeypatch.setattr(
        ray_app, "ensure_bridge_proxy_for_actor",
        lambda model, cfg: calls.append((model, cfg.get("no_subscription_auth"))),
    )
    ray_app._maybe_ensure_bridge(_cfg(_DW))            # doubleword -> bridge
    ray_app._maybe_ensure_bridge(_cfg(_GLM, True))     # per-token -> bridge
    ray_app._maybe_ensure_bridge(_cfg(_GLM, False))    # coding-plan -> no bridge
    ray_app._maybe_ensure_bridge(_cfg("anthropic/claude-sonnet-4-6"))
    assert calls == [(_DW, True), (_GLM, True)]


def test_run_pool_folds_no_subscription_auth_into_actor_cfg(
    monkeypatch, fake_gcs_bucket
):
    """`run_pool` must place `no_subscription_auth` into the cfg the actor reads
    so a per-token z.ai run actually bridges. Capture a dispatched actor's cfg."""
    client, _store = fake_gcs_bucket
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
        no_subscription_auth=True,
        local_only=True,
        actor_cls=_CaptureActor,
    )
    assert captured["no_subscription_auth"] is True


def _main_argv(model: str, extra: list[str] | None = None) -> list[str]:
    return [
        "--run-id", RUN_ID,
        "--attempt", "1",
        "--framework", "claude_sdk_v1",
        "--query-mode", "raw",
        "--mode", "one-shot",
        "--agent-model", model,
        "--user-sim-model", "anthropic/claude-haiku-4-5-20251001",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--instance-ids", "alien_1",
        *(extra or []),
    ]


def _patch_main(monkeypatch, captured):
    monkeypatch.setattr(ray_app, "_load_secrets_file", lambda f: {})
    monkeypatch.setattr(ray_app, "download_benchmark_data", lambda *a, **k: None)
    monkeypatch.setattr(ray_app, "_load_task_data", lambda *a, **k: {})
    monkeypatch.setattr(ray_app, "run_pool", lambda **kw: captured.update(kw))


def test_ray_app_main_subscription_flag_to_run_pool(monkeypatch):
    """--no-subscription-auth parsed by the in-cluster ray_app main must reach
    run_pool as no_subscription_auth=True (per-token bridge)."""
    captured = {}
    _patch_main(monkeypatch, captured)
    ray_app.main(_main_argv(_GLM, ["--no-subscription-auth"]))
    assert captured["no_subscription_auth"] is True


def test_ray_app_main_subscription_auth_coding_plan(monkeypatch):
    captured = {}
    _patch_main(monkeypatch, captured)
    ray_app.main(_main_argv(_GLM, ["--subscription-auth"]))
    assert captured["no_subscription_auth"] is False


def test_ray_app_main_default_is_per_token_bridge(monkeypatch):
    captured = {}
    _patch_main(monkeypatch, captured)
    ray_app.main(_main_argv(_DW))
    assert captured["no_subscription_auth"] is True
