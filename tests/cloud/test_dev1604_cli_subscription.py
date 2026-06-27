"""DEV-1604: `--subscription-auth` recycled as the z.ai endpoint selector.

Instead of a new flag, z.ai reuses the existing `--subscription-auth` /
`--no-subscription-auth` pair:

* z.ai `--subscription-auth`  -> direct GLM-Coding-Plan Anthropic endpoint
  (no bridge). Still authenticates with `ZAI_API_KEY`, NOT the Claude.ai OAuth
  token — so it must NOT demand `CLAUDE_CODE_OAUTH_TOKEN`.
* z.ai default / `--no-subscription-auth` -> per-token OpenAI endpoint via the
  bridge proxy (`no_subscription_auth=True`).
* Doubleword (OpenAI-only) and Moonshot (provider-key-only) reject
  `--subscription-auth`.

The carried truth is `no_subscription_auth` (already threaded everywhere).
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import cli

_DW = "doubleword/zai-org/GLM-5.2-FP8"
_GLM = "zai/glm-5.2"
_KIMI = "moonshot/kimi-k2.7-code"


def _argv(model: str, extra: list[str] | None = None) -> list[str]:
    return [
        "submit",
        "--framework", "claude_sdk_v1",
        "--query-mode", "slayer",
        "--mode", "one-shot",
        "--agent-model", model,
        "--instance-ids", "alien_1",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--no-require-annotation",
        *(extra or []),
    ]


# ---------------------------------------------------------------------------
# z.ai — the recycled flag selects the endpoint (per-token bridge by default)
# ---------------------------------------------------------------------------


def test_zai_default_is_per_token_bridge(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    ns = cli.parse_args(_argv(_GLM))
    # Default (no flag) -> no_subscription_auth True -> per-token bridge.
    assert ns.no_subscription_auth is True


def test_zai_no_subscription_auth_is_per_token_bridge():
    ns = cli.parse_args(_argv(_GLM, ["--no-subscription-auth"]))
    assert ns.no_subscription_auth is True


def test_zai_subscription_auth_is_coding_plan_no_oauth_required(monkeypatch):
    """z.ai --subscription-auth selects the coding-plan endpoint and must NOT
    require a Claude.ai OAuth token (it auths with ZAI_API_KEY)."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    ns = cli.parse_args(_argv(_GLM, ["--subscription-auth"]))
    assert ns.no_subscription_auth is False


# ---------------------------------------------------------------------------
# Doubleword — OpenAI-only, always bridges; --subscription-auth rejected
# ---------------------------------------------------------------------------


def test_doubleword_default_parses():
    ns = cli.parse_args(_argv(_DW))
    assert ns.no_subscription_auth is True


def test_doubleword_subscription_auth_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(_DW, ["--subscription-auth"]))
    err = capsys.readouterr().err
    assert "subscription-auth" in err
    assert "doubleword" in err.lower()


# ---------------------------------------------------------------------------
# Moonshot — provider-key-only; --subscription-auth still rejected
# ---------------------------------------------------------------------------


def test_moonshot_subscription_auth_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(_KIMI, ["--subscription-auth"]))
    err = capsys.readouterr().err
    assert "subscription-auth" in err
    assert "MOONSHOT_API_KEY" in err


def test_moonshot_default_parses():
    ns = cli.parse_args(_argv(_KIMI))
    assert ns.no_subscription_auth is True
