"""DEV-1579: the v0 ``claude_sdk`` agents gain registry open-weight support.

Before DEV-1579 the four v0 agents hard-skipped every non-Anthropic model
(``if not is_anthropic(self.model): return <skip row>``). Now they route
through ``hermetic_claude_sdk_session`` (default ``provider_aware=True``),
which layers the registry provider session env — so registry models
(moonshot/…) RUN on v0 too. A genuinely-unsupported provider (not Anthropic,
not in the registry, e.g. cerebras/…) still gets a graceful skip row.

The v0 agents share the exact monkeypatch surface of their v1 siblings, so we
reuse each sibling's ``_stub_env`` / ``_TASK`` to drive the v0 module.
"""

from __future__ import annotations

import importlib

import pytest

from bird_interact_agents.agents.claude_sdk.sdk_env import disable_cli_telemetry_env

from tests import test_claude_sdk_otf_v1_agent as otf_t
from tests import test_claude_sdk_otf_ainteract_v1_agent as ainteract_t
from tests import test_claude_sdk_otf_raw_v1_agent as raw_t
from tests import test_claude_sdk_otf_ainteract_raw_v1_agent as ainteract_raw_t


# (v1 sibling stub module, v0 agent package, v0 class, query_mode, eval_mode)
_V0_CASES = [
    pytest.param(otf_t, "claude_sdk_otf", "ClaudeSDKOtfAgent",
                 "slayer", "one-shot", id="otf"),
    pytest.param(ainteract_t, "claude_sdk_otf_ainteract",
                 "ClaudeSDKOtfAInteractAgent", "slayer", "a-interact",
                 id="ainteract"),
    pytest.param(raw_t, "claude_sdk_otf_raw", "ClaudeSDKOtfRawAgent",
                 "raw", "one-shot", id="raw"),
    pytest.param(ainteract_raw_t, "claude_sdk_otf_ainteract_raw",
                 "ClaudeSDKOtfAInteractRawAgent", "raw", "a-interact",
                 id="ainteract_raw"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _V0_CASES,
)
async def test_v0_registry_model_gets_session_env_and_no_skip(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """A registry (moonshot) model on a v0 agent: the hermetic session layers
    the provider base-url + auth and the hermetic CLAUDE_CONFIG_DIR, and the
    agent does NOT return the unsupported-model skip row."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    m = importlib.import_module(f"bird_interact_agents.agents.{module_name}.agent")
    captured = sibling._stub_env(monkeypatch, m, tmp_path / "store")
    agent = getattr(m, agent_cls_name)(model="moonshot/kimi-k2.7-code")
    row = await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode, eval_mode=eval_mode,
    )
    # The SDK session was actually opened (options captured) — i.e. NOT skipped.
    assert "options" in captured, (module_name, "agent skipped a registry model")
    env = captured["options"].env or {}
    assert env["ANTHROPIC_BASE_URL"] == "https://api.moonshot.ai/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ms-key-1"
    assert env.get("CLAUDE_CONFIG_DIR"), (module_name, "CLAUDE_CONFIG_DIR")
    # Telemetry knobs still present underneath.
    for k, v in disable_cli_telemetry_env().items():
        assert env.get(k) == v, (module_name, k)
    # The native (provider-stripped) model id reached the SDK.
    assert captured["options"].model == "kimi-k2.7-code"
    # Not a skip row.
    assert "Skipped" not in str(row.get("error") or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling,module_name,agent_cls_name,query_mode,eval_mode", _V0_CASES,
)
async def test_v0_unsupported_model_keeps_graceful_skip(
    monkeypatch, tmp_path, sibling, module_name, agent_cls_name,
    query_mode, eval_mode,
):
    """A genuinely-unsupported provider (cerebras — not Anthropic, not in the
    registry) still produces the graceful skip row, never opening a session."""
    m = importlib.import_module(f"bird_interact_agents.agents.{module_name}.agent")
    captured = sibling._stub_env(monkeypatch, m, tmp_path / "store")
    agent = getattr(m, agent_cls_name)(model="cerebras/zai-glm-4.7")
    row = await agent.run_task(
        dict(sibling._TASK), str(tmp_path), 20.0, query_mode, eval_mode=eval_mode,
    )
    assert row.get("error"), (module_name, "expected a skip-shaped error row")
    assert "Skipped" in row["error"]
    # The SDK session must NOT have been opened for an unsupported model.
    assert "options" not in captured, (module_name, "session opened for skip")
