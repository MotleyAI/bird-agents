"""DEV-1561: every Claude Agent SDK call site must disable the bundled
CLI's outbound telemetry / error-reporting / auto-updater side channels.

Without this overlay the SDK's ``__aenter__`` blocks for 5-10 minutes
during the initialize handshake while the bundled `claude` Node binary
times out on outbound calls to its analytics / Sentry / version-check
hosts (live-diagnosed against `alien_1` with a warm cache).

Pin the contract in two layers:

* ``disable_cli_telemetry_env()`` returns the set of env knobs
  recognised by the bundled CLI; the value-shape contract (string ``"1"``)
  must hold and successive calls must return INDEPENDENT dicts so an
  agent's per-task mutation of its session env doesn't bleed into the
  next task.

* every claude_sdk* adapter passes those env knobs through
  ``ClaudeAgentOptions.env``. Coverage: the four ``claude_sdk_otf*``
  adapters (the issue's direct target) AND the annotator agent (same
  SDK transport, same hang).
"""

from __future__ import annotations

import importlib

import pytest

from bird_interact_agents.agents.claude_sdk.sdk_env import (
    disable_cli_telemetry_env,
)

from tests import test_claude_sdk_otf_v1_agent as otf_t
from tests import test_claude_sdk_otf_ainteract_v1_agent as ainteract_t
from tests import test_claude_sdk_otf_ainteract_raw_v1_agent as ainteract_raw_t
from tests import test_claude_sdk_otf_raw_v1_agent as raw_t


_EXPECTED_KEYS = {
    "DISABLE_TELEMETRY",
    "DISABLE_ERROR_REPORTING",
    "DISABLE_AUTOUPDATER",
    "DISABLE_BUG_COMMAND",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
}


def test_disable_cli_telemetry_env_returns_expected_keys():
    env = disable_cli_telemetry_env()
    assert set(env) == _EXPECTED_KEYS
    # All knobs are "set to 1 ⇒ disabled"; an empty string would re-enable
    # the channel under the CLI's "treat as unset" semantics, defeating
    # the fix. Pin the value contract.
    assert {v for v in env.values()} == {"1"}


def test_disable_cli_telemetry_env_returns_fresh_copy():
    """Two calls must return distinct dicts so per-call mutation doesn't
    leak into a shared module-level mapping (agent paths update env in
    place when layering registry-provider auth on top)."""
    a = disable_cli_telemetry_env()
    b = disable_cli_telemetry_env()
    assert a == b
    assert a is not b
    a["DISABLE_TELEMETRY"] = "BAD"
    assert b["DISABLE_TELEMETRY"] == "1"
    # And a fresh fetch must still be clean.
    assert disable_cli_telemetry_env()["DISABLE_TELEMETRY"] == "1"


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_anthropic_options_env_contains_disable_telemetry(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """Anthropic model path: the disable-telemetry overlay must reach
    ``ClaudeAgentOptions.env`` unconditionally — the bug DEV-1561
    targets is the Anthropic path."""
    m = importlib.import_module(f"bird_interact_agents.agents.{module_name}.agent")
    captured = sibling._stub_env(monkeypatch, m, tmp_path / "store")
    agent = getattr(m, agent_cls_name)(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode, eval_mode=eval_mode,
    )
    env = captured["options"].env or {}
    for k, v in disable_cli_telemetry_env().items():
        assert env.get(k) == v, (module_name, k, env.get(k))
    # DEV-1579: the hermetic session also pins CLAUDE_CONFIG_DIR to a fresh
    # empty dir so the host's ~/.claude.json connectors never leak in.
    assert env.get("CLAUDE_CONFIG_DIR"), (module_name, "CLAUDE_CONFIG_DIR")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _CASES,
)
async def test_registry_options_env_layers_disable_telemetry_under_session(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """Registry (kimi) model path: the disable-telemetry overlay co-exists
    with the per-run registry session env (ANTHROPIC_BASE_URL /
    ANTHROPIC_AUTH_TOKEN); the session must NOT overwrite the
    telemetry-disable knobs."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    m = importlib.import_module(f"bird_interact_agents.agents.{module_name}.agent")
    captured = sibling._stub_env(monkeypatch, m, tmp_path / "store")
    agent = getattr(m, agent_cls_name)(model="moonshot/kimi-k2.7-code")
    await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode, eval_mode=eval_mode,
    )
    env = captured["options"].env or {}
    # Registry-provider session env present.
    assert env["ANTHROPIC_BASE_URL"] == "https://api.moonshot.ai/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ms-key-1"
    # Telemetry knobs preserved underneath.
    for k, v in disable_cli_telemetry_env().items():
        assert env.get(k) == v, (module_name, k, env.get(k))
    # DEV-1579: CLAUDE_CONFIG_DIR isolation survives the registry layering.
    assert env.get("CLAUDE_CONFIG_DIR"), (module_name, "CLAUDE_CONFIG_DIR")


def test_annotator_routes_through_hermetic_session_anthropic_only():
    """DEV-1579: the annotator must route its SDK session through the shared
    ``hermetic_claude_sdk_session`` helper (which owns the telemetry-disable
    env + CLAUDE_CONFIG_DIR isolation + API-key auth + MCP parity), and must
    pass ``provider_aware=False`` since the annotator is Anthropic-only.

    AST-inspect the annotator source: lower-overhead than driving the full
    annotation pipeline (which needs a gold sidecar + benchmark config), and
    robust against whitespace / multi-line formatting choices.
    """
    import ast
    from pathlib import Path

    ann = importlib.import_module("bird_interact_agents.agents.annotator.agent")
    assert "hermetic_claude_sdk_session" in ann.__dict__, (
        "annotator agent must import hermetic_claude_sdk_session"
    )

    tree = ast.parse(
        Path("src/bird_interact_agents/agents/annotator/agent.py").read_text()
    )
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Name)
            and func.id == "hermetic_claude_sdk_session"
        ):
            continue
        pa = next(
            (k for k in node.keywords if k.arg == "provider_aware"), None,
        )
        assert pa is not None, (
            "annotator's hermetic_claude_sdk_session call must pass "
            "provider_aware=… explicitly"
        )
        # The annotator is Anthropic-only — registry layering must be off.
        assert (
            isinstance(pa.value, ast.Constant) and pa.value.value is False
        ), f"annotator must pass provider_aware=False, got {ast.dump(pa.value)}"
        found = True
        break
    assert found, (
        "no hermetic_claude_sdk_session(...) call found in annotator source"
    )
