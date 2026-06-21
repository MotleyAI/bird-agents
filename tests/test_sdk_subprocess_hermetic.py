"""DEV-1579: hermetic Claude SDK subprocess env.

The Claude Agent SDK launches a bundled ``claude`` Node CLI. On a developer
machine the CLI reads ``~/.claude.json`` and loads every claude.ai-synced MCP
connector (Linear, Notion, Figma, …) into the model's per-turn tool schema —
making local benchmark runs non-equivalent to cloud (which has no
``~/.claude.json``). ``setting_sources=[]`` + ``allowed_tools=`` do NOT prevent
this; only pointing ``CLAUDE_CONFIG_DIR`` at a dir whose ``.claude.json``
declares no servers does.

This module pins the new ``agents/claude_sdk/sdk_env.py`` surface that all
``claude_sdk*`` agents route through:

* ``hermetic_claude_config_dir`` — fresh empty per-task ``CLAUDE_CONFIG_DIR``.
* ``assert_api_key_auth`` — claude_sdk agents MUST auth via ``ANTHROPIC_API_KEY``
  (Claude.ai subscription/OAuth auth was disabled for the Agent SDK 2026-06-15);
  registry open-weight models are exempt (own provider token).
* ``build_hermetic_session_env`` / ``hermetic_session_option_kwargs`` — the
  ``ClaudeAgentOptions`` env (+ registry session env + thinking) the policy owns.
* ``assert_hermetic_mcp_servers`` — runtime parity assertion (name-set only).
* ``hermetic_claude_sdk_session`` — the single context-manager choke point.

These are pure-unit (no LLM, no real CLI subprocess). The real-CLI behavior is
proven by ``test_run_task_*`` per-agent env capture + the ``integration``-marked
smoke at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_interact_agents.agents.claude_sdk import sdk_env


_ANTHROPIC = "anthropic/claude-sonnet-4-5"
_REGISTRY = "moonshot/kimi-k2.7-code"  # registry, requires_thinking=True


# ---------------------------------------------------------------------------
# hermetic_claude_config_dir
# ---------------------------------------------------------------------------

def test_config_dir_writes_claude_json_with_no_mcp_servers():
    val, path = sdk_env.hermetic_claude_config_dir()
    try:
        assert isinstance(val, str) and Path(val) == path
        cj = path / ".claude.json"
        assert cj.is_file()
        data = json.loads(cj.read_text())
        # The whole point: an explicit EMPTY mcpServers map => the CLI loads
        # ZERO servers beyond options.mcp_servers (not missing-key reliance).
        assert data["mcpServers"] == {}
        # Belt-and-suspenders onboarding/trust flags so the non-interactive
        # CLI never blocks on a first-run prompt.
        assert data.get("hasCompletedOnboarding") is True
        assert data.get("bypassPermissionsModeAccepted") is True
        assert data.get("hasTrustDialogAccepted") is True
    finally:
        import shutil
        shutil.rmtree(path, ignore_errors=True)


def test_config_dir_does_not_copy_credentials():
    """Subscription/OAuth auth is dead for the Agent SDK (2026-06-15); the
    hermetic dir must NOT carry a .credentials.json forward — auth goes via
    ANTHROPIC_API_KEY (process env)."""
    val, path = sdk_env.hermetic_claude_config_dir()
    try:
        assert not (path / ".credentials.json").exists()
    finally:
        import shutil
        shutil.rmtree(path, ignore_errors=True)


def test_config_dir_fresh_each_call():
    v1, p1 = sdk_env.hermetic_claude_config_dir()
    v2, p2 = sdk_env.hermetic_claude_config_dir()
    try:
        assert p1 != p2 and v1 != v2
    finally:
        import shutil
        shutil.rmtree(p1, ignore_errors=True)
        shutil.rmtree(p2, ignore_errors=True)


# ---------------------------------------------------------------------------
# assert_api_key_auth
# ---------------------------------------------------------------------------

def test_api_key_auth_passes_for_anthropic_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    sdk_env.assert_api_key_auth(_ANTHROPIC)  # no raise


def test_api_key_auth_raises_for_anthropic_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(sdk_env.ApiKeyAuthError):
        sdk_env.assert_api_key_auth(_ANTHROPIC)


def test_api_key_auth_raises_on_lone_oauth_token(monkeypatch):
    """A lone CLAUDE_CODE_OAUTH_TOKEN (subscription auth) does NOT satisfy the
    API-key requirement — the dead subscription path must hard-fail."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-deadbeef")
    with pytest.raises(sdk_env.ApiKeyAuthError):
        sdk_env.assert_api_key_auth(_ANTHROPIC)


def test_api_key_auth_exempts_registry_model(monkeypatch):
    """Registry open-weight models authenticate via their own provider token;
    they are exempt from the ANTHROPIC_API_KEY requirement."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sdk_env.assert_api_key_auth(_REGISTRY)  # no raise


def test_api_key_auth_registry_not_exempt_when_provider_unaware(monkeypatch):
    """Codex r2-major: an Anthropic-only call site (provider_aware=False) must
    NOT silently skip auth enforcement on a registry model — the session env
    would not receive registry auth, so require the API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(sdk_env.ApiKeyAuthError):
        sdk_env.assert_api_key_auth(_REGISTRY, provider_aware=False)


# ---------------------------------------------------------------------------
# build_hermetic_session_env
# ---------------------------------------------------------------------------

_TELEMETRY_KEYS = {
    "DISABLE_TELEMETRY", "DISABLE_ERROR_REPORTING", "DISABLE_AUTOUPDATER",
    "DISABLE_BUG_COMMAND", "DISABLE_NON_ESSENTIAL_MODEL_CALLS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
}


def test_session_env_anthropic_has_config_dir_and_telemetry_no_registry():
    env = sdk_env.build_hermetic_session_env(_ANTHROPIC, "/tmp/cfgX")
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/cfgX"
    assert _TELEMETRY_KEYS <= set(env)
    # Ambient subscription/OAuth token is masked so the SDK is forced onto
    # ANTHROPIC_API_KEY (subscription auth is dead).
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
    # No registry layering for an Anthropic model.
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_session_env_registry_layers_provider_auth(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    env = sdk_env.build_hermetic_session_env(_REGISTRY, "/tmp/cfgY")
    # CLAUDE_CONFIG_DIR survives the registry layering (sdk_session_env must
    # not clobber it).
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/cfgY"
    assert _TELEMETRY_KEYS <= set(env)
    assert env["ANTHROPIC_BASE_URL"] == "https://api.moonshot.ai/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "ms-key-1"
    # Ambient Anthropic creds neutralised so the SDK can't route to Anthropic.
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""


def test_session_env_provider_aware_false_skips_registry(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    env = sdk_env.build_hermetic_session_env(
        _REGISTRY, "/tmp/cfgZ", provider_aware=False,
    )
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/cfgZ"
    assert "ANTHROPIC_BASE_URL" not in env


# ---------------------------------------------------------------------------
# hermetic_session_option_kwargs  (+ ClaudeAgentOptions construction)
# ---------------------------------------------------------------------------

def test_option_kwargs_anthropic_no_thinking():
    kw = sdk_env.hermetic_session_option_kwargs(_ANTHROPIC, "/tmp/c")
    assert set(kw) == {"env"}
    assert "thinking" not in kw


def test_option_kwargs_registry_adds_thinking(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    kw = sdk_env.hermetic_session_option_kwargs(_REGISTRY, "/tmp/c")
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8192}


def test_option_kwargs_registry_without_thinking_omits_thinking(monkeypatch):
    """thinking is added ONLY for registry models that require it. The sole
    registry model today requires thinking, so simulate a non-thinking
    registry model by stubbing requires_thinking — pins the `and
    requires_thinking(model)` branch."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.setattr(sdk_env, "requires_thinking", lambda _m: False)
    kw = sdk_env.hermetic_session_option_kwargs(_REGISTRY, "/tmp/c")
    assert "thinking" not in kw
    # Registry session env still layered (it's still a registry model).
    assert kw["env"]["ANTHROPIC_BASE_URL"] == "https://api.moonshot.ai/anthropic"


def test_option_kwargs_construct_real_options(monkeypatch):
    """Codex r2-major(#4): splatting the optional ``thinking`` key alongside
    the agent's other kwargs must produce a valid ClaudeAgentOptions in the
    installed SDK — both for a normal model and a requires_thinking one."""
    from claude_agent_sdk import ClaudeAgentOptions

    opt_a = ClaudeAgentOptions(
        **sdk_env.hermetic_session_option_kwargs(_ANTHROPIC, "/tmp/c"),
        model="claude-sonnet-4-5",
    )
    assert opt_a.env["CLAUDE_CONFIG_DIR"] == "/tmp/c"
    assert opt_a.thinking is None

    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    opt_r = ClaudeAgentOptions(
        **sdk_env.hermetic_session_option_kwargs(_REGISTRY, "/tmp/c"),
        model="kimi-k2.7-code",
    )
    assert opt_r.thinking == {"type": "enabled", "budget_tokens": 8192}


# ---------------------------------------------------------------------------
# assert_hermetic_mcp_servers  (name-set only, contamination not readiness)
# ---------------------------------------------------------------------------

class _StatusClient:
    def __init__(self, servers):
        # servers: list of (name, status)
        self._servers = servers

    async def get_mcp_status(self):
        return {"mcpServers": [
            {"name": n, "status": s} for (n, s) in self._servers
        ]}


@pytest.mark.asyncio
async def test_parity_passes_on_exact_match():
    client = _StatusClient([("slayer", "connected"), ("bird-interact-tools", "connected")])
    await sdk_env.assert_hermetic_mcp_servers(
        client, {"slayer", "bird-interact-tools"},
    )  # no raise


@pytest.mark.asyncio
async def test_parity_raises_on_extra_server():
    client = _StatusClient([
        ("slayer", "connected"), ("bird-interact-tools", "connected"),
        ("Linear", "connected"),
    ])
    with pytest.raises(sdk_env.HermeticEnvError) as ei:
        await sdk_env.assert_hermetic_mcp_servers(
            client, {"slayer", "bird-interact-tools"},
        )
    assert "Linear" in str(ei.value)


@pytest.mark.asyncio
async def test_parity_raises_on_missing_server():
    client = _StatusClient([("slayer", "connected")])
    with pytest.raises(sdk_env.HermeticEnvError) as ei:
        await sdk_env.assert_hermetic_mcp_servers(
            client, {"slayer", "bird-interact-tools"},
        )
    assert "bird-interact-tools" in str(ei.value)


@pytest.mark.asyncio
async def test_parity_is_contamination_not_readiness():
    """Name-set only (product decision): a present-but-FAILED server still
    passes the contamination check. The failed status is surfaced in the
    raised message only when names diverge — here names match so no raise."""
    client = _StatusClient([
        ("slayer", "failed"), ("bird-interact-tools", "connected"),
    ])
    await sdk_env.assert_hermetic_mcp_servers(
        client, {"slayer", "bird-interact-tools"},
    )  # no raise despite slayer=failed


# ---------------------------------------------------------------------------
# hermetic_claude_sdk_session  (the context-manager choke point)
# ---------------------------------------------------------------------------

class _FakeSDKClient:
    """Stand-in for ClaudeSDKClient: captures options, serves get_mcp_status."""

    instances: list = []

    def __init__(self, options):
        self.options = options
        self.entered = False
        type(self).instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *a):
        return None

    async def get_mcp_status(self):
        # Echo whatever mcp_servers the options carry => parity passes by
        # default. Tests inject extras by overriding this.
        names = list((self.options.mcp_servers or {}).keys())
        return {"mcpServers": [{"name": n, "status": "connected"} for n in names]}


def _patch_client(monkeypatch, cls=_FakeSDKClient):
    cls.instances = []
    monkeypatch.setattr(sdk_env, "ClaudeSDKClient", cls)
    return cls


def _build_options(opt_kwargs):
    from claude_agent_sdk import ClaudeAgentOptions
    return ClaudeAgentOptions(
        **opt_kwargs,
        mcp_servers={"bird-interact-tools": object(), "slayer": object()},
        model="claude-sonnet-4-5",
    )


@pytest.mark.asyncio
async def test_session_yields_client_and_cleans_up(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _patch_client(monkeypatch)
    seen = {}
    async with sdk_env.hermetic_claude_sdk_session(
        _ANTHROPIC,
        mcp_servers={"bird-interact-tools": object(), "slayer": object()},
        build_options=_build_options,
    ) as client:
        assert client.entered
        cfg = client.options.env["CLAUDE_CONFIG_DIR"]
        seen["cfg"] = cfg
        assert Path(cfg).is_dir()
    # config dir removed on exit
    assert not Path(seen["cfg"]).exists()


@pytest.mark.asyncio
async def test_session_cleans_up_on_parity_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class _Contaminated(_FakeSDKClient):
        async def get_mcp_status(self):
            return {"mcpServers": [
                {"name": "bird-interact-tools", "status": "connected"},
                {"name": "slayer", "status": "connected"},
                {"name": "Linear", "status": "connected"},
            ]}

    _patch_client(monkeypatch, _Contaminated)
    captured_cfg = {}

    def _bo(opt_kwargs):
        captured_cfg["dir"] = opt_kwargs["env"]["CLAUDE_CONFIG_DIR"]
        return _build_options(opt_kwargs)

    with pytest.raises(sdk_env.HermeticEnvError):
        async with sdk_env.hermetic_claude_sdk_session(
            _ANTHROPIC,
            mcp_servers={"bird-interact-tools": object(), "slayer": object()},
            build_options=_bo,
        ):
            pass
    assert not Path(captured_cfg["dir"]).exists()


@pytest.mark.asyncio
async def test_session_cleans_up_on_build_options_raise(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _patch_client(monkeypatch)
    captured = {}

    def _bo(opt_kwargs):
        captured["dir"] = opt_kwargs["env"]["CLAUDE_CONFIG_DIR"]
        raise RuntimeError("boom during option construction")

    with pytest.raises(RuntimeError, match="boom"):
        async with sdk_env.hermetic_claude_sdk_session(
            _ANTHROPIC, mcp_servers={"slayer": object()}, build_options=_bo,
        ):
            pass
    assert not Path(captured["dir"]).exists()


@pytest.mark.asyncio
async def test_session_cleans_up_on_client_enter_raise(monkeypatch):
    """Codex r2-minor(#7): a failed ClaudeSDKClient.__aenter__ must not leak
    the temp config dir."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class _EnterBoom(_FakeSDKClient):
        async def __aenter__(self):
            raise RuntimeError("enter boom")

    _patch_client(monkeypatch, _EnterBoom)
    captured = {}

    def _bo(opt_kwargs):
        captured["dir"] = opt_kwargs["env"]["CLAUDE_CONFIG_DIR"]
        return _build_options(opt_kwargs)

    with pytest.raises(RuntimeError, match="enter boom"):
        async with sdk_env.hermetic_claude_sdk_session(
            _ANTHROPIC,
            mcp_servers={"bird-interact-tools": object(), "slayer": object()},
            build_options=_bo,
        ):
            pass
    assert not Path(captured["dir"]).exists()


@pytest.mark.asyncio
async def test_session_cleans_up_on_get_mcp_status_raise(monkeypatch):
    """Codex r2-minor(#7): a get_mcp_status() failure must not leak the dir."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    class _StatusBoom(_FakeSDKClient):
        async def get_mcp_status(self):
            raise RuntimeError("status boom")

    _patch_client(monkeypatch, _StatusBoom)
    captured = {}

    def _bo(opt_kwargs):
        captured["dir"] = opt_kwargs["env"]["CLAUDE_CONFIG_DIR"]
        return _build_options(opt_kwargs)

    with pytest.raises(RuntimeError, match="status boom"):
        async with sdk_env.hermetic_claude_sdk_session(
            _ANTHROPIC,
            mcp_servers={"bird-interact-tools": object(), "slayer": object()},
            build_options=_bo,
        ):
            pass
    assert not Path(captured["dir"]).exists()


@pytest.mark.asyncio
async def test_session_cleans_up_on_caller_body_raise(monkeypatch):
    """The finally must also fire when the CALLER raises inside the yielded
    session (the common real path: the agent loop throws)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _patch_client(monkeypatch)
    seen = {}

    class _BodyError(RuntimeError):
        pass

    with pytest.raises(_BodyError):
        async with sdk_env.hermetic_claude_sdk_session(
            _ANTHROPIC,
            mcp_servers={"bird-interact-tools": object(), "slayer": object()},
            build_options=_build_options,
        ) as client:
            seen["cfg"] = client.options.env["CLAUDE_CONFIG_DIR"]
            raise _BodyError("agent loop blew up")
    assert not Path(seen["cfg"]).exists()


@pytest.mark.asyncio
async def test_session_fails_fast_on_missing_api_key(monkeypatch):
    """assert_api_key_auth fires BEFORE any config dir / client work."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    cls = _patch_client(monkeypatch)
    with pytest.raises(sdk_env.ApiKeyAuthError):
        async with sdk_env.hermetic_claude_sdk_session(
            _ANTHROPIC, mcp_servers={"slayer": object()},
            build_options=_build_options,
        ):
            pass
    # No client was ever constructed.
    assert cls.instances == []


@pytest.mark.asyncio
async def test_session_enter_cm_factory_wraps_client_enter(monkeypatch):
    """enter_cm_factory must WRAP the client ``__aenter__`` (so a DEV-1561
    otf_timer attributes enter-time failures), not merely be invoked. Pin the
    order: factory-enter -> client-enter -> factory-exit (the timer closes
    right after the SDK enter, before the parity check / yield)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    import contextlib

    order: list = []

    class _OrderingClient(_FakeSDKClient):
        async def __aenter__(self):
            order.append("client_enter")
            return await super().__aenter__()

    _patch_client(monkeypatch, _OrderingClient)

    @contextlib.contextmanager
    def _timer():
        order.append("factory_enter")
        try:
            yield
        finally:
            order.append("factory_exit")

    async with sdk_env.hermetic_claude_sdk_session(
        _ANTHROPIC,
        mcp_servers={"bird-interact-tools": object(), "slayer": object()},
        build_options=_build_options,
        enter_cm_factory=_timer,
    ):
        order.append("body")
    assert order == ["factory_enter", "client_enter", "factory_exit", "body"]
