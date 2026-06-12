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

from tests import test_claude_sdk_otf_agent as otf_t
from tests import test_claude_sdk_otf_ainteract_agent as ainteract_t
from tests import test_claude_sdk_otf_ainteract_raw_agent as ainteract_raw_t
from tests import test_claude_sdk_otf_raw_agent as raw_t

_KIMI = "moonshot/kimi-k2.7-code"

_CASES = [
    pytest.param(
        otf_t, "claude_sdk_otf", "ClaudeSDKOtfAgent", "slayer", "one-shot",
        id="otf",
    ),
    pytest.param(
        ainteract_t, "claude_sdk_otf_ainteract", "ClaudeSDKOtfAInteractAgent",
        "slayer", "a-interact",
        id="ainteract",
    ),
    pytest.param(
        raw_t, "claude_sdk_otf_raw", "ClaudeSDKOtfRawAgent", "raw", "one-shot",
        id="raw",
    ),
    pytest.param(
        ainteract_raw_t, "claude_sdk_otf_ainteract_raw",
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
    # Anthropic credentials must never enter the session env.
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_anthropic_model_options_env_unchanged(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """Byte-identical pin: anthropic runs pass the SDK default env."""
    captured, _row = await _run_and_capture(
        monkeypatch, tmp_path, sibling=sibling, module_name=module_name,
        agent_cls_name=agent_cls_name, query_mode=query_mode,
        eval_mode=eval_mode, model="anthropic/claude-sonnet-4-5",
    )
    assert captured["options"].env == ClaudeAgentOptions().env


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
