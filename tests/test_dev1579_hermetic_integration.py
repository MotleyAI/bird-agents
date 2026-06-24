"""DEV-1579 integration smoke: prove the hermetic redirect actually isolates
the real Claude Agent SDK subprocess from a host ``~/.claude.json``.

The unit tests stub ``get_mcp_status`` and therefore CANNOT prove that:
  (a) ``CLAUDE_CONFIG_DIR`` redirection stops the bundled CLI from loading
      MCP servers declared in ``~/.claude.json``, and
  (b) ``get_mcp_status()`` is reliable when called right after ``__aenter__``,
      BEFORE the first ``query()`` (no false "missing" abort).

This test launches a REAL ``ClaudeSDKClient`` (bundled CLI, network, valid
``ANTHROPIC_API_KEY``) with a bogus ``~/.claude.json`` on ``HOME`` declaring an
extra MCP server, and asserts the hermetic session loads ONLY the in-process
server we passed — and never blocks on a first-run onboarding/trust prompt.

Marked ``integration`` (excluded from the default run); needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
)


# Module-level: only the integration marker. The per-test auth requirement
# differs (API-key path needs ANTHROPIC_API_KEY; the DEV-1602 OAuth path needs
# CLAUDE_CODE_OAUTH_TOKEN), so each test carries its own skipif.
pytestmark = [pytest.mark.integration]

_NEEDS_API_KEY = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="needs a real ANTHROPIC_API_KEY to launch the bundled CLI",
)
_NEEDS_OAUTH = pytest.mark.skipif(
    not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").startswith("sk-ant-oat01-"),
    reason="needs a real sk-ant-oat01- CLAUDE_CODE_OAUTH_TOKEN (subscription path)",
)


@tool("ping", "Return pong.", {})
async def _ping(_args):
    return {"content": [{"type": "text", "text": "pong"}]}


@_NEEDS_API_KEY
@pytest.mark.asyncio
async def test_hermetic_session_ignores_host_claude_json(monkeypatch, tmp_path):
    """A bogus ``~/.claude.json`` with an extra stdio MCP server must NOT leak
    into the SDK subprocess; the loaded set is exactly our in-process server,
    and ``get_mcp_status()`` works before the first ``query()``."""
    # Bogus HOME whose ~/.claude.json declares a connector that would leak in
    # WITHOUT the CLAUDE_CONFIG_DIR redirect.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".claude.json").write_text(json.dumps({
        "mcpServers": {
            "leaky-connector": {
                "command": "/bin/false", "args": [], "env": {},
            },
        },
    }))
    monkeypatch.setenv("HOME", str(fake_home))
    # Ensure no ambient CLAUDE_CONFIG_DIR shadows the hermetic redirect.
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    server = create_sdk_mcp_server(name="smoke-tools", version="1.0.0", tools=[_ping])
    mcp_servers = {"smoke-tools": server}

    def _build_options(opt_kwargs: dict) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            **opt_kwargs,
            mcp_servers=mcp_servers,
            allowed_tools=["mcp__smoke-tools__ping"],
            tools=[],
            setting_sources=[],
            model="claude-haiku-4-5-20251001",
            max_turns=1,
        )

    # If contamination leaked in, the helper's own parity assertion would raise
    # HermeticEnvError before yielding. We additionally re-check pre-query.
    async with hermetic_claude_sdk_session(
        "anthropic/claude-haiku-4-5-20251001",
        mcp_servers=mcp_servers,
        build_options=_build_options,
    ) as client:
        status = await client.get_mcp_status()  # BEFORE any query()
        loaded = {s["name"] for s in status.get("mcpServers", [])}
        # get_mcp_status reports stdio/external MCP servers (and lists a server
        # by NAME even when its command failed to start, status='failed'). It
        # does NOT report in-process SDK servers like our "smoke-tools" (wired
        # straight into the tool surface). So the contract is "NO server beyond
        # what we passed" — `loaded` must contain nothing outside {smoke-tools}
        # (in practice it's empty, since smoke-tools is in-process). If the CLI
        # had read the ambient ~/.claude.json, "leaky-connector" would appear
        # here regardless of /bin/false failing to launch; its absence proves
        # the redirect kept the ambient config out.
        assert "leaky-connector" not in loaded, (
            f"host ~/.claude.json leaked into the SDK subprocess: {loaded}"
        )
        assert loaded - {"smoke-tools"} == set(), (
            f"unexpected (leaked) MCP server(s): {loaded - {'smoke-tools'}}"
        )
        # The hermetic CLAUDE_CONFIG_DIR seed pre-accepts onboarding/trust, so
        # the non-interactive CLI started without blocking (we got here).
        assert Path(client.options.env["CLAUDE_CONFIG_DIR"]).is_dir()


@_NEEDS_OAUTH
@pytest.mark.asyncio
async def test_hermetic_session_authenticates_via_subscription(monkeypatch, tmp_path):
    """DEV-1602: with BIRD_INTERACT_SUBSCRIPTION_AUTH set and a real
    sk-ant-oat01- token, the bundled CLI authenticates against the Claude.ai
    subscription — WITHOUT any ANTHROPIC_API_KEY — and the hermetic isolation
    still holds (a real ``query`` completes and no host connector leaks in)."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".claude.json").write_text(json.dumps({
        "mcpServers": {
            "leaky-connector": {"command": "/bin/false", "args": [], "env": {}},
        },
    }))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    # The core of the subscription path: NO API key in the env; only the OAuth
    # token + the explicit signal. If sdk_env still demanded ANTHROPIC_API_KEY
    # (the DEV-1579 posture) this would raise ApiKeyAuthError instead of running.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("BIRD_INTERACT_SUBSCRIPTION_AUTH", "1")

    server = create_sdk_mcp_server(name="smoke-tools", version="1.0.0", tools=[_ping])
    mcp_servers = {"smoke-tools": server}

    def _build_options(opt_kwargs: dict) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            **opt_kwargs,
            mcp_servers=mcp_servers,
            allowed_tools=["mcp__smoke-tools__ping"],
            tools=[],
            setting_sources=[],
            model="claude-haiku-4-5-20251001",
            max_turns=1,
        )

    async with hermetic_claude_sdk_session(
        "anthropic/claude-haiku-4-5-20251001",
        mcp_servers=mcp_servers,
        build_options=_build_options,
    ) as client:
        # API key masked in the subprocess env; OAuth token left to inherit.
        assert client.options.env["ANTHROPIC_API_KEY"] == ""
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in client.options.env
        status = await client.get_mcp_status()
        loaded = {s["name"] for s in status.get("mcpServers", [])}
        assert "leaky-connector" not in loaded
        # Prove the subscription token actually authenticates: a real query
        # round-trips (would 401/raise if OAuth were rejected upstream).
        await client.query("Reply with the single word: ok")
        got_text = False
        async for msg in client.receive_response():
            if type(msg).__name__ == "AssistantMessage":
                got_text = True
        assert got_text
