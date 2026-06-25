"""Env injected into the Claude Agent SDK CLI subprocess (DEV-1561).

The Agent SDK launches a bundled `claude` Node binary, which by default
makes outbound calls for telemetry, error reporting, and auto-updates as
part of its initialize handshake. In a benchmark / batch run those side
channels add zero value AND can stall the run for minutes — the symptom
DEV-1561 chases:

* every CLI startup ate 5-10 minutes of silence between process start and
  the first agent log line on local runs;
* the SDK's own 60s ``CLAUDE_CODE_STREAM_CLOSE_TIMEOUT`` default would have
  fired sooner, except interactive users tend to bump it to ~10 min, which
  is exactly what the observed wallclock matches.

A live diagnose (``DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1
DISABLE_AUTOUPDATER=1 bird-interact …``) collapses the silence: SDK
``__aenter__`` returns in ~5s, first AssistantMessage in another ~2s.

We unconditionally disable the side channels for every SDK-spawned CLI by
passing the disable knobs through ``ClaudeAgentOptions.env`` — the SDK
transport layers ``options.env`` on top of the inherited environment, so
a user can still re-enable a specific channel by exporting an explicit
override before launching the runner.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from claude_agent_sdk import ClaudeSDKClient

from bird_interact_agents.provider_registry import (
    get_provider,
    requires_thinking,
    sdk_session_env,
)


# Env names recognised by the bundled CLI to opt OUT of outbound side
# channels at initialize time. All "set-to-1 ⇒ disabled" — empty / unset
# keeps the CLI default (= enabled).
#
# - ``DISABLE_TELEMETRY``: analytics ping.
# - ``DISABLE_ERROR_REPORTING``: Sentry crash reporter.
# - ``DISABLE_AUTOUPDATER``: registry version check + background download.
# - ``DISABLE_BUG_COMMAND``: `/bug` slash-command harness (loads at init).
# - ``DISABLE_NON_ESSENTIAL_MODEL_CALLS``: non-task model calls (haiku
#   "thinking" prepass for the bug reporter, etc.).
# - ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC``: umbrella switch added
#   in later CLI versions; gates everything above plus a few smaller
#   probes. Safe to set alongside the per-channel switches.
_DISABLE_OUTBOUND_TELEMETRY_ENV: dict[str, str] = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_BUG_COMMAND": "1",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


def disable_cli_telemetry_env() -> dict[str, str]:
    """Return a fresh copy of the env mapping that disables outbound CLI
    side channels. Merge into ``ClaudeAgentOptions.env`` before passing
    options to ``ClaudeSDKClient``.

    The returned dict is fresh on every call so callers can ``.update(...)``
    additional env vars (e.g. registry-provider auth) without mutating
    the module-level constant.
    """
    return dict(_DISABLE_OUTBOUND_TELEMETRY_ENV)


# ---------------------------------------------------------------------------
# DEV-1579: hermetic SDK subprocess env.
#
# The Agent SDK's bundled `claude` CLI reads `.claude.json` from
# ``$CLAUDE_CONFIG_DIR`` (when set) else ``~/.claude.json``, and loads every
# MCP server it declares into the model's per-turn tool schema. On a developer
# machine that means every claude.ai-synced connector (Linear, Notion, Figma,
# motley-staging, …) bloats the agent's context — 11 servers / 151 tools
# locally vs 2 / 20 in cloud — so local benchmark runs are NOT equivalent to
# cloud (4x cost, different verdicts). `setting_sources=[]` + `allowed_tools=`
# do NOT prevent this; only an empty `CLAUDE_CONFIG_DIR` does.
#
# Every `claude_sdk*` agent routes its SDK session through
# ``hermetic_claude_sdk_session`` below, which owns: a fresh empty
# CLAUDE_CONFIG_DIR, the telemetry-disable + registry-provider session env,
# the per-run `thinking` config, API-key auth enforcement, the runtime MCP
# parity assertion, and config-dir cleanup. Future agents inherit all of it by
# using the same helper.
# ---------------------------------------------------------------------------


class HermeticEnvError(RuntimeError):
    """The SDK subprocess loaded MCP servers we did not pass in.

    Signals that ``CLAUDE_CONFIG_DIR`` isolation broke — typically the host's
    ``~/.claude.json`` claude.ai connectors leaked into the agent's tool
    surface, making the run non-equivalent to cloud.
    """


class ApiKeyAuthError(RuntimeError):
    """A ``claude_sdk`` agent has no usable auth credential.

    On the default API-key path the agent authenticates via ``ANTHROPIC_API_KEY``
    (Anthropic) or its registry provider token. On the DEV-1602 subscription path
    (selected by the ``BIRD_INTERACT_SUBSCRIPTION_AUTH`` signal) it authenticates
    via a Claude.ai OAuth token (``CLAUDE_CODE_OAUTH_TOKEN``, ``sk-ant-oat01-``
    prefix). This is raised when the selected path has no usable credential.
    """


# DEV-1602: the subscription/OAuth path is gated by an EXPLICIT operator signal,
# never inferred from which credential happens to be present in the env. The
# cloud driver sets this on the actor env when ``--subscription-auth`` is chosen;
# the local ``run.py`` sets it from its own ``--subscription-auth`` flag.
_SUBSCRIPTION_AUTH_ENV = "BIRD_INTERACT_SUBSCRIPTION_AUTH"

# A Claude.ai OAuth token always carries this prefix; the driver, prereqs, and
# the cloud actor invariant all validate it, and so do we.
_OAUTH_TOKEN_PREFIX = "sk-ant-oat01-"


def _subscription_auth_selected() -> bool:
    """True iff the operator explicitly opted into the subscription/OAuth path.

    Truthy means the signal var is present and not empty and not ``"0"`` — any
    other non-empty value selects the subscription path.
    """
    val = os.environ.get(_SUBSCRIPTION_AUTH_ENV, "")
    return bool(val) and val != "0"


def assert_api_key_auth(model: str, *, provider_aware: bool = True) -> None:
    """Enforce auth for a ``claude_sdk`` agent (API-key OR subscription path).

    Registry open-weight models authenticate via their own provider token
    (layered by :func:`build_hermetic_session_env`) and are exempt — but only
    when the call site is provider-aware. An Anthropic-only call site
    (``provider_aware=False``) that is somehow handed a registry model would
    NOT receive that provider env, so it must still demand a real credential
    rather than silently skip the check. The registry exemption is checked
    FIRST, so the subscription signal (Anthropic-only) never applies to it.

    On the subscription path (``BIRD_INTERACT_SUBSCRIPTION_AUTH`` set) a valid
    ``CLAUDE_CODE_OAUTH_TOKEN`` (``sk-ant-oat01-`` prefix) satisfies auth and a
    missing/malformed token hard-fails — there is NO silent fall-back to
    ``ANTHROPIC_API_KEY`` even when it is present. Otherwise (API-key path) an
    ``ANTHROPIC_API_KEY`` is required. Raises :class:`ApiKeyAuthError` on
    failure.
    """
    if provider_aware and get_provider(model) is not None:
        return
    if _subscription_auth_selected():
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if not token:
            raise ApiKeyAuthError(
                "claude_sdk subscription auth was selected "
                f"({_SUBSCRIPTION_AUTH_ENV} set) but CLAUDE_CODE_OAUTH_TOKEN is "
                f"not set (model={model!r}). Run `claude setup-token`, or unset "
                f"{_SUBSCRIPTION_AUTH_ENV} to use the ANTHROPIC_API_KEY path."
            )
        if not token.startswith(_OAUTH_TOKEN_PREFIX):
            raise ApiKeyAuthError(
                "CLAUDE_CODE_OAUTH_TOKEN does not look like a Claude.ai OAuth "
                f"token (expected {_OAUTH_TOKEN_PREFIX} prefix, model={model!r}). "
                "Re-run `claude setup-token`."
            )
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ApiKeyAuthError(
            f"claude_sdk agents require ANTHROPIC_API_KEY (model={model!r}) on "
            f"the API-key path. Export ANTHROPIC_API_KEY, or select the "
            f"subscription path ({_SUBSCRIPTION_AUTH_ENV}=1 with a valid "
            "CLAUDE_CODE_OAUTH_TOKEN; cloud: submit with --subscription-auth)."
        )


def hermetic_claude_config_dir() -> tuple[str, Path]:
    """Create a fresh, empty per-task ``CLAUDE_CONFIG_DIR``.

    The bundled CLI reads ``$CLAUDE_CONFIG_DIR/.claude.json``; we seed it with
    a ``.claude.json`` that declares NO ``mcpServers`` key, so the subprocess
    sees only the servers passed via ``ClaudeAgentOptions.mcp_servers``. The
    onboarding / trust flags pre-accept the CLI's first-run prompts so it never
    blocks on interactive stdin in a non-tty subprocess.

    We deliberately do NOT copy a ``.credentials.json`` forward: auth always
    flows from the process env — ``ANTHROPIC_API_KEY`` on the API-key path, or
    ``CLAUDE_CODE_OAUTH_TOKEN`` on the DEV-1602 subscription path (see
    :func:`assert_api_key_auth`) — never from a copied credential file.

    Returns ``(env_var_value, dir_path)``. The caller owns cleanup
    (``shutil.rmtree(dir_path, ignore_errors=True)``);
    :func:`hermetic_claude_sdk_session` does this automatically.
    """
    d = Path(tempfile.mkdtemp(prefix="bird_interact_claude_config_"))
    (d / ".claude.json").write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": True,
                "bypassPermissionsModeAccepted": True,
                "hasTrustDialogAccepted": True,
                # Explicit empty map (not missing-key reliance): the hermetic
                # contract is "this config declares ZERO MCP servers".
                "mcpServers": {},
            }
        )
    )
    return str(d), d


def build_hermetic_session_env(
    model: str, config_dir_val: str, *, provider_aware: bool = True,
) -> dict[str, str]:
    """The ``ClaudeAgentOptions.env`` mapping the isolation policy owns.

    Layers: telemetry-disable knobs (DEV-1561) + the hermetic
    ``CLAUDE_CONFIG_DIR`` + the auth-path credential masking + (for registry
    open-weight models, when ``provider_aware``) the provider session env
    (``ANTHROPIC_BASE_URL`` + Bearer token + ambient-Anthropic-credential
    neutralisation). The registry layer never sets ``CLAUDE_CONFIG_DIR``, so
    isolation survives it.

    DEV-1602 auth-path masking (Anthropic models only — registry is exempt):

    * subscription path (``BIRD_INTERACT_SUBSCRIPTION_AUTH`` set): mask BOTH
      ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN`` (empty strings) so the
      SDK's auth-precedence rule cannot pick either over the OAuth token, and
      leave ``CLAUDE_CODE_OAUTH_TOKEN`` to inherit from the parent env into the
      subprocess. NB the parent process keeps ``ANTHROPIC_API_KEY`` (the litellm
      user-sim needs it on local runs); only the SDK subprocess sees it masked.
    * API-key path (default): mask an ambient ``CLAUDE_CODE_OAUTH_TOKEN`` so the
      SDK is forced onto ``ANTHROPIC_API_KEY``.

    For registry models the provider layer below masks BOTH and sets the
    provider's own bearer token, so the subscription signal is inert for them.
    """
    env = disable_cli_telemetry_env()
    env["CLAUDE_CONFIG_DIR"] = config_dir_val
    is_registry = provider_aware and get_provider(model) is not None
    if _subscription_auth_selected() and not is_registry:
        # Subscription path: mask the API key AND the Bearer auth token (both are
        # SDK credentials the auth-precedence rule would pick over the OAuth
        # token — see the registry path, which uses ANTHROPIC_AUTH_TOKEN for
        # Bearer auth). OAuth is authoritative; the token inherits.
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_AUTH_TOKEN"] = ""
    else:
        # API-key path (and the base layer for registry, overwritten below):
        # mask an ambient OAuth token so the SDK uses ANTHROPIC_API_KEY.
        env["CLAUDE_CODE_OAUTH_TOKEN"] = ""
    if is_registry:
        env.update(sdk_session_env(model))
    return env


def hermetic_session_option_kwargs(
    model: str, config_dir_val: str, *, provider_aware: bool = True,
) -> dict:
    """``ClaudeAgentOptions`` kwargs owned by the policy: ``env`` (+ ``thinking``).

    The ``thinking`` config is added only for registry models that require it
    (probed live — e.g. kimi-k2.7-code rejects requests without thinking). The
    returned dict is splatted into the agent's own ``ClaudeAgentOptions(...)``.
    """
    kwargs: dict = {
        "env": build_hermetic_session_env(
            model, config_dir_val, provider_aware=provider_aware,
        ),
    }
    if (
        provider_aware
        and get_provider(model) is not None
        and requires_thinking(model)
    ):
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}
    return kwargs


async def assert_hermetic_mcp_servers(client, expected: Iterable[str]) -> None:
    """Runtime contamination assertion: the SDK loaded NO server we didn't pass.

    Asserts only the no-EXTRA direction (``loaded - expected == ∅``), NOT exact
    equality. ``get_mcp_status`` reports stdio/external MCP servers (e.g. the
    ``slayer`` stdio server, or a leaked ``~/.claude.json`` connector) but does
    NOT report in-process SDK servers created via ``create_sdk_mcp_server``
    (e.g. ``bird-interact-tools``) — those are wired straight into the model's
    tool surface without an MCP connection. So a loaded set that is a strict
    SUBSET of ``expected`` is normal; only an EXTRA, unexpected server means the
    host's config leaked in. This is a contamination check, not a readiness
    check — a present-but-``failed`` expected server still passes. Statuses are
    surfaced in the message so a leak is not opaque.
    """
    expected_set = set(expected)
    status = await client.get_mcp_status()
    servers = status.get("mcpServers", []) if status else []
    loaded = {s["name"] for s in servers}
    extra = loaded - expected_set
    if extra:
        detail = ", ".join(
            f"{s['name']}={s.get('status', '?')}" for s in servers
        )
        raise HermeticEnvError(
            "SDK subprocess loaded unexpected MCP servers: "
            f"extra={sorted(extra)} (loaded: {detail}). CLAUDE_CONFIG_DIR "
            "isolation likely broken — the host's claude.ai connectors may "
            "have leaked in."
        )


@contextlib.asynccontextmanager
async def hermetic_claude_sdk_session(
    model: str,
    *,
    mcp_servers: Mapping[str, object],
    build_options: Callable[[dict], object],
    enter_cm_factory: Callable[[], object] | None = None,
    provider_aware: bool = True,
):
    """The single choke point every ``claude_sdk*`` agent routes through.

    Owns the full hermetic SDK session lifecycle:

    1. ``assert_api_key_auth`` (fail fast, before any work).
    2. fresh empty ``CLAUDE_CONFIG_DIR``.
    3. ``build_options(opt_kwargs)`` — the agent supplies its own
       ``ClaudeAgentOptions(**opt_kwargs, …unique tools/hooks/agents…)`` given
       the policy-owned ``env`` (+ ``thinking``) kwargs.
    4. enter ``ClaudeSDKClient`` (optionally wrapped in ``enter_cm_factory()``
       to preserve the DEV-1561 ``otf_timer``-around-``__aenter__``
       instrumentation).
    5. parity assertion before yielding the client.
    6. ``rmtree`` the config dir on exit — success OR any failure in steps 3-5.

    ``provider_aware=False`` for Anthropic-only agents (the annotator) so a
    stray registry model never silently gains registry behaviour.
    """
    assert_api_key_auth(model, provider_aware=provider_aware)
    config_val, config_path = hermetic_claude_config_dir()
    try:
        options = build_options(
            hermetic_session_option_kwargs(
                model, config_val, provider_aware=provider_aware,
            )
        )
        async with contextlib.AsyncExitStack() as stack:
            enter_cm = (
                enter_cm_factory() if enter_cm_factory is not None
                else contextlib.nullcontext()
            )
            with enter_cm:
                client = await stack.enter_async_context(
                    ClaudeSDKClient(options=options)
                )
            await assert_hermetic_mcp_servers(client, mcp_servers.keys())
            # Wrap the client so every message streamed through
            # `receive_response()` is recorded into `client.transcript` — making
            # per-session transcript capture an INTRINSIC property of every
            # claude_sdk agent (no agent re-implements it). Wrap AFTER the parity
            # assertion so it runs against the raw client.
            yield _TranscriptClient(client)
    finally:
        shutil.rmtree(config_path, ignore_errors=True)


def serialize_sdk_message(msg: Any) -> dict:
    """Serialise one SDK stream message into a JSON-able ``{type, data}`` dict.

    Mirrors the trajectory shape the claude_sdk agents already build by hand
    (``dataclasses.asdict`` when possible, else ``str``). NEVER raises — capture
    must not break the receive stream, so this owns the full guard (the caller
    appends its result directly)."""
    name = type(msg).__name__
    try:
        return {"type": name, "data": dataclasses.asdict(msg)}
    except Exception:  # noqa: BLE001 — non-dataclass / unserialisable
        try:
            return {"type": name, "data": str(msg)}
        except Exception:  # noqa: BLE001 — pathological __str__
            return {"type": name, "data": "<unserializable>"}


class _TranscriptClient:
    """Transparent proxy over a ``ClaudeSDKClient`` that tees every message
    streamed through ``receive_response()`` into ``.transcript``.

    Everything except ``query``/``receive_response`` delegates to the wrapped
    client via ``__getattr__`` (``get_mcp_status``, ``interrupt``, ``aclose``,
    …), so existing call sites are unaffected. ``.transcript`` accumulates ACROSS
    every ``query()``/``receive_response()`` cycle on the warm client, so a
    multi-turn (or re-prompt-loop) session yields one complete transcript.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.transcript: list[dict] = []

    async def query(self, *args, **kwargs):
        return await self._client.query(*args, **kwargs)

    def receive_response(self):
        return self._record(self._client.receive_response())

    async def _record(self, agen):
        try:
            async for msg in agen:
                # serialize_sdk_message never raises, so the stream is safe.
                self.transcript.append(serialize_sdk_message(msg))
                yield msg
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await agen.aclose()

    def __getattr__(self, name: str):
        # Only reached for attributes not defined on the proxy itself.
        return getattr(self._client, name)
