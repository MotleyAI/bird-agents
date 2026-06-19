"""DEV-1555 Stage 2: per-run SDK session env for open-weight backends.

For registry agent models the ClaudeAgentOptions handed to the SDK must
carry `env={ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN}` (per-run — no
os.environ mutation) and the provider-native model id. Anthropic models
keep today's options byte-identical. Unsupported models keep the graceful
skip-row (CLI hard-rejects them before they ever reach an agent in cloud
runs).

Reuses the sibling test modules' `_stub_env` / `_TASK` fakes (same pattern
as tests/test_dev1555_subagent_options.py).
"""

from __future__ import annotations

import importlib

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from bird_interact_agents.agents.claude_sdk.sdk_env import (
    disable_cli_telemetry_env,
)

from tests import test_claude_sdk_otf_v1_agent as otf_t
from tests import test_claude_sdk_otf_ainteract_v1_agent as ainteract_t
from tests import test_claude_sdk_otf_ainteract_raw_v1_agent as ainteract_raw_t
from tests import test_claude_sdk_otf_raw_v1_agent as raw_t

_KIMI = "moonshot/kimi-k2.7-code"

_CASES = [
    pytest.param(
        otf_t, "claude_sdk_otf_v1", "ClaudeSDKOtfAgent", "slayer", "one-shot",
        id="otf",
    ),
    pytest.param(
        ainteract_t, "claude_sdk_otf_ainteract_v1", "ClaudeSDKOtfAInteractAgent",
        "slayer", "a-interact",
        id="ainteract",
    ),
    pytest.param(
        raw_t, "claude_sdk_otf_raw_v1", "ClaudeSDKOtfRawAgent", "raw", "one-shot",
        id="raw",
    ),
    pytest.param(
        ainteract_raw_t, "claude_sdk_otf_ainteract_raw_v1",
        "ClaudeSDKOtfAInteractRawAgent", "raw", "a-interact",
        id="ainteract_raw",
    ),
]


def _agent_module(name):
    return importlib.import_module(f"bird_interact_agents.agents.{name}.agent")


async def _run_and_capture(monkeypatch, tmp_path, *, sibling, module_name,
                           agent_cls_name, query_mode, eval_mode, model):
    m = _agent_module(module_name)
    captured = sibling._stub_env(monkeypatch, m, tmp_path / "store")
    agent = getattr(m, agent_cls_name)(model=model)
    row = await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode,
        eval_mode=eval_mode,
    )
    return captured, row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_moonshot_model_gets_session_env_and_native_id(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    captured, _row = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode, model=_KIMI,
    )
    options = captured["options"]
    assert options.model == "kimi-k2.7-code"
    env = options.env
    assert env["ANTHROPIC_BASE_URL"] == "https://api.moonshot.ai/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ms-key-1"
    # Codex r2: the session env explicitly NEUTRALISES ambient Anthropic
    # API-key / OAuth-token vars by setting them to empty strings.
    # (Empty-string overrides ARE present in the dict so they reach the
    # SDK subprocess via runtime_env and shadow any ambient value the
    # parent process inherits — the prior "must not be in env" contract
    # was actually unsafe because absence let the parent env leak through.)
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    # DEV-1561: the disable-telemetry overlay sits underneath the
    # registry-session auth env. Both must reach the CLI subprocess.
    for k, v in disable_cli_telemetry_env().items():
        assert env.get(k) == v, (k, env.get(k))
    # kimi-k2.7-code rejects requests without thinking enabled (probed
    # live 2026-06-12) — the session must pin it on.
    assert options.thinking == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_anthropic_model_options_env_is_disable_telemetry(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """DEV-1561 byte-equal pin: anthropic runs pass ONLY the disable-CLI-
    telemetry overlay — no registry session env, no Anthropic credentials
    leaked into the SDK subprocess env.
    """
    captured, _row = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode, model="anthropic/claude-sonnet-4-5",
    )
    assert captured["options"].env == disable_cli_telemetry_env()
    assert captured["options"].thinking == ClaudeAgentOptions().thinking


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_unsupported_model_keeps_graceful_skip_row(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """Agent-level contract: unknown providers produce the skip row (no
    crash mid-batch); the CLI registry validation is the hard gate."""
    captured, row = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode, model="cerebras/zai-glm-4.7",
    )
    assert row["phase1_passed"] is False
    assert row.get("error")
    assert "options" not in captured  # SDK session never started
