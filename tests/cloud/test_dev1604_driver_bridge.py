"""DEV-1604: driver key-shipping + base_url override suppression + threading.

Two driver responsibilities for the bridge:

1. The registry key-shipping generalises on `auth_env`, so a `doubleword/*`
   agent ships `DOUBLEWORD_API_KEY` with no new code.
2. The submitter forwards a `base_url_env` operator override to the workers —
   but when the agent NEEDS the bridge (z.ai per-token), the ACTOR owns
   `base_url_env` (it points it at the loopback proxy), so the submitter must
   NOT forward a stale value. Forwarding stays intact for the non-bridge paths
   (Moonshot, z.ai coding-plan, and DEV-1639: Doubleword-direct).

The bridge selector is the recycled `--subscription-auth` flag, carried as
`no_subscription_auth` (True = per-token/bridge). Also pins that the flag is
re-emitted into the in-cluster job args and survives resubmit.
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
        no_subscription_auth=True,
    )
    assert keys["DOUBLEWORD_API_KEY"] == "dw-key-1"
    # NEVER ship raw Anthropic creds for a registry agent (existing contract).
    assert "ANTHROPIC_API_KEY" not in keys


# ---------------------------------------------------------------------------
# base_url override suppression when bridged
# ---------------------------------------------------------------------------


def test_doubleword_override_is_forwarded(monkeypatch):
    """DEV-1639: Doubleword now talks directly (non-bridge), so an operator
    base-url override IS forwarded to the workers — same as any direct provider
    — instead of being suppressed for a proxy the actor no longer starts."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    monkeypatch.setenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "https://dw.example")
    keys = driver.read_api_keys_from_local_env(
        _DW, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True,
    )
    assert keys["BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"] == "https://dw.example"


def test_zai_per_token_override_not_forwarded(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    monkeypatch.setenv("BIRD_ZAI_ANTHROPIC_BASE_URL", "http://stale:9999")
    # per-token = no_subscription_auth True -> bridge -> suppress.
    keys = driver.read_api_keys_from_local_env(
        _GLM, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=True,
    )
    assert "BIRD_ZAI_ANTHROPIC_BASE_URL" not in keys


def test_zai_coding_plan_override_still_forwarded(monkeypatch):
    """z.ai --subscription-auth (no_subscription_auth False) = coding-plan, the
    non-bridge path — the operator override is still forwarded."""
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-sim")
    monkeypatch.setenv(
        "BIRD_ZAI_ANTHROPIC_BASE_URL", "https://op.example/anthropic"
    )
    keys = driver.read_api_keys_from_local_env(
        _GLM, _SIM, query_mode="raw", framework="claude_sdk",
        no_subscription_auth=False,
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
        no_subscription_auth=True,
    )
    assert keys["BIRD_MOONSHOT_ANTHROPIC_BASE_URL"] == "https://op.example/anthropic"


# ---------------------------------------------------------------------------
# Threading: manifest + in-cluster job args
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
            *(extra or []),
        ]
    )


def test_build_manifest_carries_no_subscription_auth():
    # z.ai default -> per-token bridge -> no_subscription_auth True.
    ns = _submit_ns(_GLM)
    manifest = driver.build_manifest(ns, image_uri="img:1", run_id="r1")
    assert manifest["no_subscription_auth"] is True


def test_build_manifest_zai_coding_plan():
    ns = _submit_ns(_GLM, ["--subscription-auth"])
    manifest = driver.build_manifest(ns, image_uri="img:1", run_id="r1")
    assert manifest["no_subscription_auth"] is False


def test_job_args_emit_subscription_flag_per_token():
    ns = _submit_ns(_GLM)  # default -> per-token bridge
    job_args = driver._build_job_args(ns, "r1", attempt=1)
    assert "--no-subscription-auth" in job_args
    assert "--subscription-auth" not in job_args


def test_job_args_emit_subscription_flag_coding_plan():
    ns = _submit_ns(_GLM, ["--subscription-auth"])  # coding-plan
    job_args = driver._build_job_args(ns, "r1", attempt=1)
    assert "--subscription-auth" in job_args
    assert "--no-subscription-auth" not in job_args


# ---------------------------------------------------------------------------
# resubmit reconstruction must carry the flag
# ---------------------------------------------------------------------------


def _resubmit_manifest(no_subscription_auth: bool) -> dict:
    return {
        "run_id": "r1",
        "framework": "claude_sdk_v1",
        "mode": "one-shot",
        "query_mode": "slayer",
        "agent_model": _GLM,
        "user_sim_model": _SIM,
        "dataset": "livesqlbench-base-lite-sqlite",
        "instance_ids": ["alien_1"],
        "no_subscription_auth": no_subscription_auth,
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }


def test_resubmit_args_carry_per_token_bridge():
    """A per-token z.ai resubmit must NOT silently fall back to coding-plan."""
    job_args = driver._build_resubmit_args(
        _resubmit_manifest(True), "r1", ["alien_1"], attempt=2
    )
    assert "--no-subscription-auth" in job_args


def test_resubmit_args_carry_coding_plan():
    job_args = driver._build_resubmit_args(
        _resubmit_manifest(False), "r1", ["alien_1"], attempt=2
    )
    assert "--subscription-auth" in job_args


def test_resubmit_args_old_manifest_defaults_to_bridge():
    """A pre-DEV-1604 manifest lacks the key — resubmit defaults to the registry
    default (no_subscription_auth True = per-token bridge), never crashing."""
    m = _resubmit_manifest(True)
    del m["no_subscription_auth"]
    job_args = driver._build_resubmit_args(m, "r1", ["alien_1"], attempt=2)
    assert "--no-subscription-auth" in job_args
