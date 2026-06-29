"""DEV-1555 Stage 2: CLI auth-flag conditionality for open-weight models.

`--subscription-auth` is Anthropic-only: for registry agent models on
`submit` it must be rejected with a clear error; omitted (or explicit
`--no-subscription-auth`) resolves to the API-key-style path with
`no_subscription_auth=True`. Anthropic submits keep the existing
explicit-choice requirement. `annotate` is untouched.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import cli

_KIMI = "moonshot/kimi-k2.7-code"


def _submit_argv(
    model: str,
    extra: list[str] | None = None,
    *,
    framework: str = "claude_sdk_v1",
) -> list[str]:
    # Registry models default to ``claude_sdk_v1`` here, but DEV-1579 also
    # wired the provider-aware hermetic session into the v0 ``claude_sdk``
    # aggregator — the dedicated parse test below covers that path.
    return [
        "submit",
        "--framework", framework,
        "--query-mode", "slayer",
        "--mode", "one-shot",
        "--agent-model", model,
        "--instance-ids", "alien_1",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--no-require-annotation",
        *(extra or []),
    ]


def test_moonshot_submit_without_auth_flag_parses(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    ns = cli.parse_args(_submit_argv(_KIMI))
    assert ns.no_subscription_auth is True


def test_moonshot_submit_explicit_no_subscription_auth_parses():
    ns = cli.parse_args(_submit_argv(_KIMI, ["--no-subscription-auth"]))
    assert ns.no_subscription_auth is True


def test_moonshot_submit_with_subscription_auth_rejected(monkeypatch, capsys):
    # DEV-1604: Moonshot is provider-key-only (no subscription concept), so
    # --subscription-auth is still rejected — even with a valid-looking token —
    # and the error names its provider key. (z.ai, by contrast, now ACCEPTS the
    # flag as its endpoint selector; see test_dev1604_cli_subscription.)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    with pytest.raises(SystemExit):
        cli.parse_args(_submit_argv(_KIMI, ["--subscription-auth"]))
    err = capsys.readouterr().err
    assert "subscription-auth" in err
    assert "MOONSHOT_API_KEY" in err


def test_moonshot_submit_with_v0_claude_sdk_parses(monkeypatch):
    """DEV-1579: the v0 ``claude_sdk`` aggregator now carries the
    provider-aware hermetic session env, so ``--framework claude_sdk +
    moonshot/...`` parses at the CLI (no more parse-time rejection) and
    resolves to the API-key path (``no_subscription_auth=True``)."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    ns = cli.parse_args(_submit_argv(_KIMI, framework="claude_sdk"))
    assert ns.no_subscription_auth is True


def test_anthropic_submit_still_requires_explicit_choice():
    with pytest.raises(SystemExit):
        cli.parse_args(_submit_argv("anthropic/claude-sonnet-4-6"))


def test_anthropic_submit_no_subscription_auth_still_parses():
    ns = cli.parse_args(
        _submit_argv("anthropic/claude-sonnet-4-6", ["--no-subscription-auth"])
    )
    assert ns.no_subscription_auth is True


def test_unknown_provider_model_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(
            _submit_argv("cerebras/zai-glm-4.7", ["--no-subscription-auth"])
        )
    err = capsys.readouterr().err
    assert "moonshot" in err  # error lists known providers


def test_annotate_auth_flag_still_required():
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "annotate",
                "--benchmark", "livesqlbench-base-lite-sqlite",
                "--agent-model", "anthropic/claude-sonnet-4-6",
                "--instance-ids", "alien_1",
            ]
        )
