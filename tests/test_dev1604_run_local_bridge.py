"""DEV-1604: local (`run.py`) bridge wiring.

`bird-interact run` supports `zai/*` bridging like cloud via
`_maybe_start_bridge_proxy` (called from `main()` after `_apply_subscription_
auth_env`, before any runner is built). It recycles `--subscription-auth`:
z.ai `--subscription-auth` -> coding-plan (no bridge); default / `--no-
subscription-auth` -> per-token bridge. DEV-1639: Doubleword now talks its
native Anthropic endpoint directly (no bridge) but still rejects
`--subscription-auth`; Moonshot rejects it too.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import run

_DW = "doubleword/zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"
_KIMI = "moonshot/kimi-k2.7-code"


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


# subscription_auth is the tri-state BooleanOptionalAction value:
# None (unset), True (--subscription-auth), False (--no-subscription-auth).


def test_doubleword_does_not_start_bridge(spy_ensure):
    """DEV-1639: Doubleword talks its native Anthropic endpoint directly, so no
    bridge proxy is started for it (default or --no-subscription-auth)."""
    run._maybe_start_bridge_proxy(
        agent_model=_DW, subscription_auth=None, error=_err
    )
    run._maybe_start_bridge_proxy(
        agent_model=_DW, subscription_auth=False, error=_err
    )
    assert spy_ensure == []


def test_zai_default_starts_per_token_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model=_GLM, subscription_auth=None, error=_err
    )
    assert spy_ensure == [(_GLM, {"no_subscription_auth": True})]


def test_zai_no_subscription_starts_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model=_GLM, subscription_auth=False, error=_err
    )
    assert spy_ensure == [(_GLM, {"no_subscription_auth": True})]


def test_zai_subscription_auth_is_coding_plan_no_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model=_GLM, subscription_auth=True, error=_err
    )
    assert spy_ensure == []


def test_anthropic_no_bridge(spy_ensure):
    run._maybe_start_bridge_proxy(
        agent_model="anthropic/claude-sonnet-4-6",
        subscription_auth=None, error=_err,
    )
    assert spy_ensure == []


def test_doubleword_subscription_auth_rejected(spy_ensure):
    with pytest.raises(SystemExit):
        run._maybe_start_bridge_proxy(
            agent_model=_DW, subscription_auth=True, error=_err
        )
    assert spy_ensure == []  # validation fired before any proxy started


def test_moonshot_subscription_auth_rejected(spy_ensure):
    with pytest.raises(SystemExit):
        run._maybe_start_bridge_proxy(
            agent_model=_KIMI, subscription_auth=True, error=_err
        )
    assert spy_ensure == []
