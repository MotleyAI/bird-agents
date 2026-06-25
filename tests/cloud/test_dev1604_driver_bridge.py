"""DEV-1604: driver key-shipping + base_url override suppression + threading.

Two driver responsibilities change for the bridge:

1. The existing registry key-shipping already generalises on `auth_env`, so a
   `doubleword/*` agent ships `DOUBLEWORD_API_KEY` with no new code — pinned
   here so a refactor can't regress it.
2. The submitter forwards a `base_url_env` operator override to the workers
   (so moonshot / z.ai-coding-plan hit the same endpoint the submitter
   validated). When the agent NEEDS the bridge, the actor owns
   `base_url_env` (it points it at the local proxy), so the submitter must
   NOT forward its value — else a stale submitter override would clobber the
   actor's loopback URL. Forwarding stays intact for the non-bridge providers.

Also pins `zai_billing` threading: into the manifest, into the in-cluster job
args, and through resubmit reconstruction.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import cli, driver

_DW = "doubleword/zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"
_KIMI = "moonshot/kimi-k2.7-code"
_SIM = "anthropic/claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Key shipping
# ---------------------------------------------------------------------------


def test_ships_doubleword_key(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    keys = driver.read_api_keys_from_local_env(
        _DW, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True, zai_billing="coding-plan",
    )
    assert keys["DOUBLEWORD_API_KEY"] == "dw-key-1"
    # NEVER ship raw Anthropic creds for a registry agent (existing contract).
    assert "ANTHROPIC_API_KEY" not in keys


# ---------------------------------------------------------------------------
# base_url override suppression when bridged (Codex #8)
# ---------------------------------------------------------------------------


def test_doubleword_override_not_forwarded(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    # Even if the submitter has a stray override exported, it must NOT travel
    # to the workers — the actor sets this var to the local proxy URL.
    monkeypatch.setenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "http://stale:9999")
    keys = driver.read_api_keys_from_local_env(
        _DW, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True, zai_billing="coding-plan",
    )
    assert "BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL" not in keys


def test_zai_per_token_override_not_forwarded(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    monkeypatch.setenv("BIRD_ZAI_ANTHROPIC_BASE_URL", "http://stale:9999")
    keys = driver.read_api_keys_from_local_env(
        _GLM, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True, zai_billing="per-token",
    )
    assert "BIRD_ZAI_ANTHROPIC_BASE_URL" not in keys


def test_zai_coding_plan_override_still_forwarded(monkeypatch):
    """The non-bridge z.ai coding-plan path keeps the existing operator-override
    forwarding."""
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    monkeypatch.setenv(
        "BIRD_ZAI_ANTHROPIC_BASE_URL", "https://op.example/anthropic"
    )
    keys = driver.read_api_keys_from_local_env(
        _GLM, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True, zai_billing="coding-plan",
    )
    assert keys["BIRD_ZAI_ANTHROPIC_BASE_URL"] == "https://op.example/anthropic"


def test_moonshot_override_still_forwarded(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    monkeypatch.setenv(
        "BIRD_MOONSHOT_ANTHROPIC_BASE_URL", "https://op.example/anthropic"
    )
    keys = driver.read_api_keys_from_local_env(
        _KIMI, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True, zai_billing="coding-plan",
    )
    assert keys["BIRD_MOONSHOT_ANTHROPIC_BASE_URL"] == "https://op.example/anthropic"


# ---------------------------------------------------------------------------
# zai_billing threading: manifest + job args
# ---------------------------------------------------------------------------


def _submit_ns(model: str, extra: list[str] | None = None):
    return cli.parse_args(
        [
            "submit",
            "--framework", "claude_sdk_v1",
            "--query-mode", "slayer",
            "--mode", "one-shot",
            "--agent-model", model,
            "--user-sim-model", _SIM,
            "--instance-ids", "alien_1",
            "--dataset", "livesqlbench-base-lite-sqlite",
            "--no-require-annotation",
            "--no-subscription-auth",
            *(extra or []),
        ]
    )


def test_build_manifest_carries_zai_billing():
    ns = _submit_ns(_GLM, ["--zai-billing", "per-token"])
    manifest = driver.build_manifest(ns, image_uri="img:1", run_id="r1")
    assert manifest["zai_billing"] == "per-token"


def test_build_manifest_zai_billing_default():
    ns = _submit_ns(_DW)
    manifest = driver.build_manifest(ns, image_uri="img:1", run_id="r1")
    assert manifest["zai_billing"] == "coding-plan"


def test_job_args_emit_zai_billing():
    ns = _submit_ns(_GLM, ["--zai-billing", "per-token"])
    job_args = driver._build_job_args(ns, "r1", attempt=1)
    assert "--zai-billing" in job_args
    assert job_args[job_args.index("--zai-billing") + 1] == "per-token"


# ---------------------------------------------------------------------------
# resubmit reconstruction must carry zai_billing (Codex #7)
# ---------------------------------------------------------------------------


def _resubmit_manifest(zai_billing: str) -> dict:
    return {
        "run_id": "r1",
        "framework": "claude_sdk_v1",
        "mode": "one-shot",
        "query_mode": "slayer",
        "agent_model": _GLM,
        "user_sim_model": _SIM,
        "dataset": "livesqlbench-base-lite-sqlite",
        "instance_ids": ["alien_1"],
        "no_subscription_auth": True,
        "zai_billing": zai_billing,
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }


def test_resubmit_args_carry_zai_billing():
    """A per-token z.ai resubmit must NOT silently fall back to coding-plan:
    the reconstructed in-cluster job args re-emit `--zai-billing` so the actor
    still bridges to the per-token endpoint."""
    job_args = driver._build_resubmit_args(
        _resubmit_manifest("per-token"), "r1", ["alien_1"], attempt=2
    )
    assert "--zai-billing" in job_args
    assert job_args[job_args.index("--zai-billing") + 1] == "per-token"


def test_resubmit_args_zai_billing_default_for_old_manifest():
    """A pre-DEV-1604 manifest has no `zai_billing` key — resubmit defaults to
    coding-plan (the previous behaviour), never crashing on the missing key."""
    m = _resubmit_manifest("per-token")
    del m["zai_billing"]
    job_args = driver._build_resubmit_args(m, "r1", ["alien_1"], attempt=2)
    assert job_args[job_args.index("--zai-billing") + 1] == "coding-plan"
