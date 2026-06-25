"""DEV-1604: `--zai-billing {coding-plan,per-token}` submit flag.

The flag selects z.ai's billing surface: the default `coding-plan` keeps the
existing GLM-Coding-Plan Anthropic endpoint; `per-token` routes the agent
through the bridge proxy to z.ai's per-token OpenAI endpoint (escaping the
`[1313]` Fair-Usage throttle). It is z.ai-only — `per-token` for any other
agent provider (Doubleword, Anthropic, Moonshot) is a hard parse error.
Doubleword auto-bridges from its `api_format=="openai"`; it needs no flag.
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
        "--no-subscription-auth",
        *(extra or []),
    ]


def test_zai_billing_defaults_to_coding_plan():
    ns = cli.parse_args(_argv(_GLM))
    assert ns.zai_billing == "coding-plan"


def test_zai_per_token_with_zai_agent_parses():
    ns = cli.parse_args(_argv(_GLM, ["--zai-billing", "per-token"]))
    assert ns.zai_billing == "per-token"


def test_zai_coding_plan_explicit_parses():
    ns = cli.parse_args(_argv(_GLM, ["--zai-billing", "coding-plan"]))
    assert ns.zai_billing == "coding-plan"


def test_doubleword_coding_plan_default_parses():
    # Doubleword bridges automatically; the default coding-plan value is inert.
    ns = cli.parse_args(_argv(_DW))
    assert ns.zai_billing == "coding-plan"


def test_per_token_rejected_for_doubleword_agent(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(_DW, ["--zai-billing", "per-token"]))
    err = capsys.readouterr().err
    assert "per-token" in err
    assert "zai" in err.lower()


def test_per_token_rejected_for_moonshot_agent(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(_KIMI, ["--zai-billing", "per-token"]))
    err = capsys.readouterr().err
    assert "per-token" in err


def test_per_token_rejected_for_anthropic_agent(capsys):
    argv = [
        "submit",
        "--framework", "claude_sdk_v1",
        "--query-mode", "slayer",
        "--mode", "one-shot",
        "--agent-model", "anthropic/claude-sonnet-4-6",
        "--instance-ids", "alien_1",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--no-require-annotation",
        "--no-subscription-auth",
        "--zai-billing", "per-token",
    ]
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
    err = capsys.readouterr().err
    assert "per-token" in err


def test_invalid_zai_billing_value_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(_GLM, ["--zai-billing", "free-lunch"]))
