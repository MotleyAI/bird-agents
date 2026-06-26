"""DEV-1604: local Anthropic-Messages ⇄ OpenAI-Chat-Completions bridge proxy.

`claude_sdk*` agents speak only the Anthropic Messages API (the bundled Node
CLI hits ``ANTHROPIC_BASE_URL/v1/messages``). This module fronts a provider's
OpenAI endpoint with a tiny loopback FastAPI app so those agents can run:

* **Doubleword** — OpenAI-only, no Anthropic endpoint at all;
* **z.ai per-token** — z.ai's Anthropic endpoint is GLM-Coding-Plan/quota
  (Fair-Usage ``[1313]``); per-token billing lives only on its OpenAI endpoint.

The translation is litellm's, not ours: ``litellm.anthropic_messages(...)`` with
``litellm.use_chat_completions_url_for_anthropic_messages = True`` (verified
live — without the flag litellm routes to the OpenAI *Responses* API, which the
open-weight upstreams reject; the per-model ``openai/chat_completions/<id>``
form does NOT work either, only the global flag). We own three thin routes
(``/v1/messages``, ``/v1/messages/count_tokens``, ``/healthz``).

Deliberately Ray-free so ``run.py`` (local) and the cloud actors share it.

Lifecycle (``ensure_bridge_proxy_for_actor``): one proxy per VM on a
deterministic per-target loopback port, brought up under a per-port lock and
identified by ``/healthz`` (reuse an identity match, refuse a healthy
mismatch, start when absent), then ``ANTHROPIC_BASE_URL``'s override env var is
pointed at it. The provider key travels via the subprocess env only — never
argv, on-disk config, or logs.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import litellm
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from bird_interact_agents import provider_registry

# Loopback only. The proxy authenticates upstream with its own provider key and
# ignores the inbound (harmless, loopback) bearer.
SERVE_HOST = "127.0.0.1"

# Deterministic per-target port: PORT_BASE + sha256(provider|upstream|native)%
# PORT_SPAN. Distinct targets get distinct ports, so two providers never fight
# over one socket and a healthy mismatch is a real anomaly (never killed).
PORT_BASE = 8788
PORT_SPAN = 64

# Runtime dir for the per-port lock + log files. A module global so tests can
# redirect it; functions read it at call time.
_RUNTIME_DIR = Path(tempfile.gettempdir())

# Subprocesses started by THIS process (for test/manual teardown). VM-lifetime
# in production, like the shared postgres instance.
_STARTED_PROCS: list = []

# Env channel the parent uses to hand the subprocess its target (no argv).
_ENV_PROVIDER = "BIRD_BRIDGE_PROXY_PROVIDER"
_ENV_UPSTREAM = "BIRD_BRIDGE_PROXY_UPSTREAM"
_ENV_NATIVE_ID = "BIRD_BRIDGE_PROXY_NATIVE_ID"
_ENV_AUTH_ENV = "BIRD_BRIDGE_PROXY_AUTH_ENV"
_ENV_PORT = "BIRD_BRIDGE_PROXY_PORT"

_READY_TIMEOUT_S = 30.0


class BridgeProxyError(RuntimeError):
    """The bridge proxy could not be brought up / validated for this actor."""


class BridgeTarget(BaseModel):
    provider: str  # registry key, e.g. "doubleword" / "zai"
    openai_base_url: str  # upstream OpenAI chat-completions base
    native_id: str  # the model id POSTed upstream (may contain a slash)
    auth_env: str  # env var holding the provider key


def resolve_bridge_target(
    model: str, no_subscription_auth: bool
) -> BridgeTarget:
    """The upstream the proxy fronts for ``model``. Raises if ``model`` does not
    need the bridge (callers gate on ``agent_needs_bridge``; a direct call for a
    non-bridge model is misuse and fails fast)."""
    if not provider_registry.agent_needs_bridge(model, no_subscription_auth):
        raise ValueError(
            f"{model!r} (no_subscription_auth={no_subscription_auth!r}) does "
            "not need the bridge proxy"
        )
    base, native_id, auth_env = provider_registry.per_token_openai_target(model)
    spec = provider_registry.get_provider(model)
    return BridgeTarget(
        provider=spec.key,
        openai_base_url=base,
        native_id=native_id,
        auth_env=auth_env,
    )


# ---------------------------------------------------------------------------
# Pure helpers (port, paths, identity, translation) — unit-tested in isolation
# ---------------------------------------------------------------------------


def deterministic_port(target: BridgeTarget) -> int:
    # hashlib, NOT builtin hash() — hash() is per-process salted, so two actor
    # processes would compute DIFFERENT ports for the same target and miss the
    # shared VM proxy.
    key = f"{target.provider}|{target.openai_base_url}|{target.native_id}"
    digest = hashlib.sha256(key.encode()).digest()
    return PORT_BASE + (int.from_bytes(digest[:4], "big") % PORT_SPAN)


def marker_path(port: int) -> Path:
    # Slash-safe: keyed by the integer port, never the slash-bearing native id,
    # so no nested `/tmp/.../zai-org/` directory is ever created.
    return _RUNTIME_DIR / f"bridge_proxy_{port}.json"


def lock_path(port: int) -> Path:
    return _RUNTIME_DIR / f"bridge_proxy_{port}.lock"


def log_path(port: int) -> Path:
    return _RUNTIME_DIR / f"bridge_proxy_{port}.log"


def healthz_payload(target: BridgeTarget, port: int) -> dict:
    return {
        "provider": target.provider,
        "upstream": target.openai_base_url,
        "native_id": target.native_id,
        "port": port,
    }


def identity_matches(got: object, want: dict) -> bool:
    """Is ``got`` (a /healthz payload from whatever is on the port) OUR proxy
    for the same provider/upstream/native id? Tolerant of garbage payloads."""
    if not isinstance(got, dict):
        return False
    return all(
        got.get(k) == want.get(k)
        for k in ("provider", "upstream", "native_id")
    )


def key_fingerprint(key: str) -> str:
    """Non-secret fingerprint of a provider key, exposed on /healthz so a later
    actor can tell whether a reused VM-lifetime proxy holds the SAME key. A hash
    prefix — never the key itself (Codex: the auth check 401s a reused proxy that
    still carries a rotated key, so we must detect the change and restart)."""
    return hashlib.sha256(key.encode()).hexdigest()[:16] if key else ""


def messages_call_kwargs(
    body: dict, target: BridgeTarget, *, api_key: str
) -> dict:
    """Translate an inbound Anthropic ``/v1/messages`` body into
    ``litellm.anthropic_messages`` kwargs.

    The SDK body carries its own ``model`` (the bare native id); we REPLACE it
    with the ``openai/<native_id>`` route and inject the upstream + key —
    popping first so we never pass ``model`` twice (TypeError). The caller's
    dict is not mutated."""
    kwargs = dict(body)
    kwargs.pop("model", None)
    kwargs["model"] = f"openai/{target.native_id}"
    kwargs["api_base"] = target.openai_base_url
    kwargs["api_key"] = api_key
    kwargs["custom_llm_provider"] = "openai"
    return kwargs


def count_tokens_response(body: dict, target: BridgeTarget) -> dict:
    """Anthropic ``/v1/messages/count_tokens`` shape ``{"input_tokens": N}``,
    estimated offline via litellm's token counter.

    Counts the ``system`` prompt and ``tools`` in addition to ``messages`` — the
    SDK ships large tool schemas + system instructions, so counting messages
    alone materially undercounts the real context (CodeRabbit/Codex).

    ``system``/``tools`` are folded in as a leading text message rather than via
    litellm's ``tools=`` kwarg: the SDK sends Anthropic-format tools
    (``name``/``input_schema``) which litellm's OpenAI-shaped token counter
    can't parse, so we count their serialized form as text — a sound estimate
    that never raises on the inbound shape."""
    messages = list(body.get("messages", []))
    extra: list[str] = []
    system = body.get("system")
    if system:
        extra.append(system if isinstance(system, str) else json.dumps(system))
    tools = body.get("tools")
    if tools:
        extra.append(json.dumps(tools))
    if extra:
        messages = [{"role": "system", "content": "\n".join(extra)}] + messages
    n = litellm.token_counter(
        model=f"openai/{target.native_id}", messages=messages
    )
    return {"input_tokens": int(n)}


def configure_litellm_for_bridge() -> None:
    """Configure litellm's global translation behaviour for the proxy.

    * ``use_chat_completions_url_for_anthropic_messages``: translate
      ``anthropic_messages`` to the OpenAI *chat-completions* endpoint (the only
      form that works for the open-weight upstreams — the Responses-API default
      and the per-model ``openai/chat_completions/`` form both fail).
    * ``drop_params``: the Claude Agent SDK sends Anthropic-only params the
      OpenAI route does not accept (``reasoning_effort``, ``context_management``,
      …). Without this, litellm raises ``UnsupportedParamsError`` and the proxy
      500s on every real SDK request (verified live). Drop them silently."""
    litellm.use_chat_completions_url_for_anthropic_messages = True
    litellm.drop_params = True


def _request_authorized(request: Request, target: BridgeTarget) -> bool:
    """Defence-in-depth on the (loopback-bound) key-spending routes: only accept
    a caller that presents the provider key as its bearer.

    The SDK sends ``ANTHROPIC_AUTH_TOKEN`` (which ``sdk_session_env`` sets to the
    provider key) as ``Authorization: Bearer <key>``, so a legitimate request
    already carries it; a foreign same-host process that discovered the port via
    ``/healthz`` does not know the key and is rejected. If the key env var is
    somehow unset we cannot verify, and fall back to the prior loopback-only
    posture (allow) rather than hard-failing a misconfigured run."""
    expected = os.environ.get(target.auth_env, "")
    if not expected:
        return True
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth[:7].lower() == "bearer " else ""
    # Compare as BYTES: hmac.compare_digest raises TypeError on a non-ASCII str,
    # so a malformed (non-ASCII) bearer would 500 instead of cleanly 401-ing.
    return bool(token) and hmac.compare_digest(
        token.encode("utf-8"), expected.encode("utf-8")
    )


# ---------------------------------------------------------------------------
# The FastAPI app
# ---------------------------------------------------------------------------


def create_app(target: BridgeTarget, port: Optional[int] = None) -> FastAPI:
    configure_litellm_for_bridge()
    health_port = port if port is not None else deterministic_port(target)
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):  # noqa: ANN202
        if not _request_authorized(request, target):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        api_key = os.environ.get(target.auth_env, "")
        kwargs = messages_call_kwargs(body, target, api_key=api_key)
        if body.get("stream"):
            iterator = await litellm.anthropic_messages(**kwargs)

            async def _gen():
                async for chunk in iterator:
                    yield (
                        bytes(chunk)
                        if isinstance(chunk, (bytes, bytearray))
                        else str(chunk).encode()
                    )

            return StreamingResponse(_gen(), media_type="text/event-stream")
        resp = await litellm.anthropic_messages(**kwargs)
        return JSONResponse(resp)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):  # noqa: ANN202
        if not _request_authorized(request, target):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        return JSONResponse(count_tokens_response(body, target))

    @app.get("/healthz")
    async def healthz():  # noqa: ANN202
        payload = healthz_payload(target, health_port)
        # Non-secret key fingerprint + pid so a later actor can detect a rotated
        # key on a reused proxy and restart it (it would otherwise 401).
        payload["key_fp"] = key_fingerprint(os.environ.get(target.auth_env, ""))
        payload["pid"] = os.getpid()
        return JSONResponse(payload)

    return app


def _serve_config_from_env() -> tuple[BridgeTarget, int]:
    try:
        target = BridgeTarget(
            provider=os.environ[_ENV_PROVIDER],
            openai_base_url=os.environ[_ENV_UPSTREAM],
            native_id=os.environ[_ENV_NATIVE_ID],
            auth_env=os.environ[_ENV_AUTH_ENV],
        )
        port = int(os.environ[_ENV_PORT])
    except KeyError as e:  # pragma: no cover - serve() is the subprocess path
        raise BridgeProxyError(
            f"bridge proxy subprocess missing config env var: {e}"
        ) from e
    return target, port


def serve() -> None:  # pragma: no cover - exercised by the integration gate
    import uvicorn

    configure_litellm_for_bridge()
    target, port = _serve_config_from_env()
    app = create_app(target, port)
    uvicorn.run(app, host=SERVE_HOST, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Lifecycle: one proxy per VM, started inside the actor (cloud secret delivery)
# ---------------------------------------------------------------------------


def _probe_identity(port: int) -> Optional[dict]:
    """GET /healthz on the port; return the identity payload, or None if nothing
    healthy answers."""
    try:
        r = httpx.get(f"http://{SERVE_HOST}:{port}/healthz", timeout=1.0)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _start_proxy(target: BridgeTarget, port: int) -> None:  # pragma: no cover
    """Spawn the proxy subprocess, hand it the target via env (never argv), and
    block until it is serving OUR identity. Raises BridgeProxyError on failure.

    Monkeypatched out in unit tests; exercised by the integration gate."""
    env = dict(os.environ)
    env[_ENV_PROVIDER] = target.provider
    env[_ENV_UPSTREAM] = target.openai_base_url
    env[_ENV_NATIVE_ID] = target.native_id
    env[_ENV_AUTH_ENV] = target.auth_env
    env[_ENV_PORT] = str(port)
    # Close the parent's copy of the log fd after spawn — only the child keeps
    # the redirected stdout/stderr handle, so repeated starts can't leak fds.
    with open(log_path(port), "ab") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "bird_interact_agents.cloud.bridge_proxy"],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
    _STARTED_PROCS.append(proc)

    def _reap() -> None:
        # Don't leave a half-started child holding the port after we declare
        # startup failed (Codex).
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)  # reap the killed child, no zombie
        except Exception:
            pass
        with contextlib.suppress(ValueError):
            _STARTED_PROCS.remove(proc)

    want = healthz_payload(target, port)
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _reap()
            raise BridgeProxyError(
                f"bridge proxy for {target.provider} exited early "
                f"(code {proc.returncode}); see {log_path(port)}"
            )
        if identity_matches(_probe_identity(port), want):
            return
        time.sleep(0.25)
    _reap()
    raise BridgeProxyError(
        f"bridge proxy for {target.provider} on {SERVE_HOST}:{port} did not "
        f"become ready within {_READY_TIMEOUT_S}s; see {log_path(port)}"
    )


@contextlib.contextmanager
def _vm_lock(port: int):
    """Serialise concurrent actors on one VM across the whole critical section
    (probe → start → readiness), mirroring the postgres ``/tmp`` lock."""
    lp = lock_path(port)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def ensure_bridge_proxy_for_actor(model: str, cfg: dict) -> str:
    """Bring up (or reuse) the VM's bridge proxy for ``model`` and point the
    provider's ``base_url_env`` override at it. Returns the loopback URL.

    Idempotent and concurrency-safe: a per-port lock spans the probe → start →
    readiness window so concurrent actors share one proxy. A healthy proxy with
    OUR identity AND key is reused; a same-target proxy holding a DIFFERENT (e.g.
    rotated) key is OURS-but-stale and is restarted (it would otherwise 401 the
    auth check); a healthy FOREIGN service on the port is an anomaly we refuse
    (never kill); otherwise we start one. The override is set AFTER the proxy is
    confirmed ready, BEFORE the caller builds the SDK runner.

    ``cfg["no_subscription_auth"]`` (default True — the registry default) is the
    z.ai endpoint selector; Doubleword bridges regardless."""
    target = resolve_bridge_target(
        model, cfg.get("no_subscription_auth", True)
    )
    port = deterministic_port(target)
    url = f"http://{SERVE_HOST}:{port}"
    spec = provider_registry.get_provider(model)
    want = healthz_payload(target, port)
    my_key_fp = key_fingerprint(os.environ.get(spec.auth_env, ""))
    with _vm_lock(port):
        ident = _probe_identity(port)
        if ident is not None:
            if not identity_matches(ident, want):
                raise BridgeProxyError(
                    f"loopback port {port} is occupied by a foreign service "
                    f"(identity={ident!r}); refusing to replace it. Set "
                    f"{spec.base_url_env} or free the port."
                )
            live_fp = ident.get("key_fp")
            if live_fp and my_key_fp and live_fp != my_key_fp:
                # OURS but a different key (rotation / new secret on a reused VM):
                # the auth check would 401 our requests. Restart with our key.
                _replace_stale_proxy(ident.get("pid"), port)
                _start_proxy(target, port)
            # else: healthy, OURS, same key — reuse.
        else:
            _start_proxy(target, port)
        marker_path(port).write_text(json.dumps(want))
    os.environ[spec.base_url_env] = url
    return url


def _replace_stale_proxy(pid: "int | None", port: int) -> None:  # pragma: no cover
    """Terminate a same-target proxy that carries a different key, then wait for
    its port to free so a fresh ``_start_proxy`` can bind it."""
    if pid:
        with contextlib.suppress(Exception):
            os.kill(int(pid), signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _probe_identity(port) is None:
            return
        time.sleep(0.2)


def terminate_local_proxies() -> None:
    """Terminate proxy subprocesses started by THIS process (test/manual
    teardown — production leaves them for the VM lifetime)."""
    global _STARTED_PROCS
    survivors: list = []
    for proc in _STARTED_PROCS:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            if proc.poll() is None:  # still alive — keep the handle
                survivors.append(proc)
        except Exception:
            survivors.append(proc)
    _STARTED_PROCS = survivors


if __name__ == "__main__":  # pragma: no cover
    serve()
