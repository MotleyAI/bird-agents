"""DEV-1604: local (`run.py`) bridge wiring.

`bird-interact run` must support `doubleword/*` and `zai/* --zai-billing
per-token` exactly like cloud: a `_maybe_start_bridge_proxy` helper (mirroring
the DEV-1602 `_apply_subscription_auth_env` precedent) starts the loopback
proxy and points `ANTHROPIC_BASE_URL`'s override at it BEFORE any runner is
built. It is called from `main()` right after auth-env setup, so the override
is in place before `run_evaluation` constructs the agent.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import run

_DW = "doubleword/zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"


@pytest.fixture
def spy_ensure(monkeypatch):
    calls = []

    def _fake(model, cfg):
        calls.append((model, dict(cfg)))
        return "http://127.0.0.1:8788"

    monkeypatch.setattr(
        run.bridge_proxy, "ensure_bridge_proxy_for_actor", _fake
    )
    return calls


def _err(msg):  # argparse-style error sink
    raise SystemExit(msg)


def test_doubleword_starts_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model=_DW, zai_billing="coding-plan", error=_err
    )
    assert spy_ensure == [(_DW, {"zai_billing": "coding-plan"})]


def test_zai_per_token_starts_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model=_GLM, zai_billing="per-token", error=_err
    )
    assert spy_ensure == [(_GLM, {"zai_billing": "per-token"})]


def test_zai_coding_plan_no_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model=_GLM, zai_billing="coding-plan", error=_err
    )
    assert spy_ensure == []


def test_anthropic_no_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model="anthropic/claude-sonnet-4-6",
        zai_billing="coding-plan", error=_err,
    )
    assert spy_ensure == []


def test_per_token_rejected_for_non_zai_agent(spy_ensure):
    with pytest.raises(SystemExit):
        run._maybe_start_bridge_proxy(
            agent_model=_DW, zai_billing="per-token", error=_err
        )
    # Validation fired before any proxy was started.
    assert spy_ensure == []
