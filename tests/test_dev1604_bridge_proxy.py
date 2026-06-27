"""DEV-1604: the Anthropic-Messages ⇄ OpenAI-Chat-Completions bridge proxy.

The proxy is a tiny FastAPI app that fronts a provider's OpenAI endpoint and
speaks Anthropic `/v1/messages` to the Claude Agent SDK Node CLI. It reuses
litellm's translation (`litellm.anthropic_messages` with the chat-completions
flag) so we own only three thin routes.

These unit tests pin the seams that the blocking integration gate (real SDK,
real upstream) is too coarse to localise — and the specific failure modes
Codex flagged on the plan:

* the inbound SDK body carries `model`, so splatting it into
  `anthropic_messages(model=...)` would raise `TypeError` (Codex #1);
* the slash-bearing native id `zai-org/GLM-5.2-FP8` must never create nested
  `/tmp/.../zai-org/` lock/marker/port paths (Codex #10);
* a deterministic per-target port so distinct providers never collide and a
  healthy mismatch is never silently killed (Codex #2);
* `ensure_bridge_proxy_for_actor` sets `os.environ[base_url_env]` and reuses an
  identity-matching proxy / replaces a dead one / refuses a healthy mismatch.
"""

from __future__ import annotations

import os

import pytest

from bird_interact_agents.cloud import bridge_proxy as bp

_DW_MODEL = "doubleword/zai-org/GLM-5.2-FP8"
_DW_NATIVE = "zai-org/GLM-5.2-FP8"
_ZAI_MODEL = "zai/glm-5.2"


# DEV-1604: the 2nd arg is no_subscription_auth (True = per-token/bridge path).
@pytest.fixture
def dw_target(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    return bp.resolve_bridge_target(_DW_MODEL, True)


@pytest.fixture
def zai_target(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    return bp.resolve_bridge_target(_ZAI_MODEL, True)


# ---------------------------------------------------------------------------
# resolve_bridge_target
# ---------------------------------------------------------------------------


def test_resolve_bridge_target_doubleword(dw_target):
    assert dw_target.provider == "doubleword"
    assert dw_target.openai_base_url == "https://api.doubleword.ai/v1"
    assert dw_target.native_id == _DW_NATIVE
    assert dw_target.auth_env == "DOUBLEWORD_API_KEY"


def test_resolve_bridge_target_zai_per_token(zai_target):
    assert zai_target.provider == "zai"
    assert zai_target.openai_base_url == "https://api.z.ai/api/paas/v4"
    assert zai_target.native_id == "glm-5.2"
    assert zai_target.auth_env == "ZAI_API_KEY"


def test_resolve_bridge_target_rejects_non_bridge_model():
    # z.ai with --subscription-auth (no_subscription_auth=False) = coding-plan,
    # does NOT bridge.
    with pytest.raises((ValueError, RuntimeError)):
        bp.resolve_bridge_target(_ZAI_MODEL, False)
    with pytest.raises((ValueError, RuntimeError)):
        bp.resolve_bridge_target("anthropic/claude-sonnet-4-6", True)


# ---------------------------------------------------------------------------
# messages_call_kwargs — the Codex #1 model-collision fix
# ---------------------------------------------------------------------------


def test_messages_call_kwargs_overrides_inbound_model(dw_target):
    """The SDK body contains `model=zai-org/GLM-5.2-FP8` (the bare native id).
    We must REPLACE it with `openai/<native_id>` — not splat both."""
    body = {
        "model": _DW_NATIVE,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    kwargs = bp.messages_call_kwargs(body, dw_target, api_key="dw-key-1")
    # Exactly one `model`, and it's the openai-routed form.
    assert kwargs["model"] == f"openai/{_DW_NATIVE}"
    assert kwargs["api_base"] == "https://api.doubleword.ai/v1"
    assert kwargs["api_key"] == "dw-key-1"
    assert kwargs["custom_llm_provider"] == "openai"
    # Passes the SDK fields through untouched.
    assert kwargs["max_tokens"] == 256
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_messages_call_kwargs_does_not_mutate_body(dw_target):
    body = {"model": _DW_NATIVE, "max_tokens": 8, "messages": []}
    bp.messages_call_kwargs(body, dw_target, api_key="k")
    assert body["model"] == _DW_NATIVE  # caller's dict is untouched


def test_messages_call_kwargs_can_splat_into_anthropic_messages_signature(
    dw_target,
):
    """Regression for the TypeError: build kwargs, then confirm they bind to
    the real `litellm.anthropic_messages` signature with no duplicate `model`."""
    import inspect

    import litellm

    body = {
        "model": _DW_NATIVE,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    kwargs = bp.messages_call_kwargs(body, dw_target, api_key="k")
    sig = inspect.signature(litellm.anthropic_messages)
    # Would raise TypeError on a duplicate/unknown-without-**kwargs binding.
    sig.bind_partial(**kwargs)


# ---------------------------------------------------------------------------
# count_tokens — real Anthropic response shape (Codex #4)
# ---------------------------------------------------------------------------


def test_count_tokens_response_shape(dw_target):
    body = {
        "model": _DW_NATIVE,
        "messages": [{"role": "user", "content": "count these tokens please"}],
    }
    out = bp.count_tokens_response(body, dw_target)
    assert set(out) == {"input_tokens"}
    assert isinstance(out["input_tokens"], int)
    assert out["input_tokens"] > 0


# ---------------------------------------------------------------------------
# deterministic port — distinct targets never collide (Codex #2)
# ---------------------------------------------------------------------------


def test_deterministic_port_is_stable(dw_target):
    assert bp.deterministic_port(dw_target) == bp.deterministic_port(dw_target)


def test_deterministic_port_distinct_across_all_bridge_targets(
    dw_target, zai_target
):
    # All currently-bridgeable targets must get pairwise-distinct ports so two
    # providers never fight over one loopback socket.
    targets = [dw_target, zai_target]
    ports = [bp.deterministic_port(t) for t in targets]
    assert len(set(ports)) == len(ports)


def test_deterministic_port_in_loopback_range(dw_target, zai_target):
    for t in (dw_target, zai_target):
        p = bp.deterministic_port(t)
        assert isinstance(p, int)
        assert bp.PORT_BASE <= p < bp.PORT_BASE + bp.PORT_SPAN


# ---------------------------------------------------------------------------
# slash-safety: marker/lock paths must not nest on the native-id slash (#10)
# ---------------------------------------------------------------------------


def test_marker_and_lock_paths_are_slash_safe(monkeypatch, tmp_path, dw_target):
    monkeypatch.setattr(bp, "_RUNTIME_DIR", tmp_path)
    port = bp.deterministic_port(dw_target)
    marker = bp.marker_path(port)
    lock = bp.lock_path(port)
    # Both live DIRECTLY under the runtime dir — the embedded `zai-org/` slash
    # in the native id must never have created a nested directory.
    assert marker.parent == tmp_path
    assert lock.parent == tmp_path
    assert "zai-org" not in str(marker)
    assert "zai-org" not in str(lock)
    assert str(port) in marker.name


# ---------------------------------------------------------------------------
# identity / fingerprint
# ---------------------------------------------------------------------------


def test_healthz_payload_carries_identity(dw_target):
    port = bp.deterministic_port(dw_target)
    payload = bp.healthz_payload(dw_target, port)
    assert payload["provider"] == "doubleword"
    assert payload["upstream"] == "https://api.doubleword.ai/v1"
    assert payload["native_id"] == _DW_NATIVE
    assert payload["port"] == port


def test_identity_matches_same_target_mismatch_other(dw_target, zai_target):
    pd = bp.healthz_payload(dw_target, bp.deterministic_port(dw_target))
    pz = bp.healthz_payload(zai_target, bp.deterministic_port(zai_target))
    assert bp.identity_matches(pd, pd) is True
    assert bp.identity_matches(pd, pz) is False
    # A partial/garbage payload from a foreign service is a mismatch, not a crash.
    assert bp.identity_matches({"hello": "world"}, pd) is False


# ---------------------------------------------------------------------------
# FastAPI app routes via TestClient (litellm.anthropic_messages monkeypatched)
# ---------------------------------------------------------------------------


# The SDK presents the provider key as its bearer (sdk_session_env sets
# ANTHROPIC_AUTH_TOKEN to it); the proxy verifies it.
_AUTH = {"authorization": "Bearer dw-key-1"}


def _client(target):
    from starlette.testclient import TestClient

    return TestClient(bp.create_app(target))


def test_app_messages_non_stream_routes_to_openai_native(monkeypatch, dw_target):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    seen = {}

    async def fake_messages(**kwargs):
        seen.update(kwargs)
        return {"type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1}}

    monkeypatch.setattr(bp.litellm, "anthropic_messages", fake_messages)
    resp = _client(dw_target).post(
        "/v1/messages", headers=_AUTH,
        json={"model": _DW_NATIVE, "max_tokens": 16,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "ok"
    # The inbound bare native id was replaced with the openai-routed form,
    # the upstream + key were injected, and NO duplicate `model` was passed.
    assert seen["model"] == f"openai/{_DW_NATIVE}"
    assert seen["api_base"] == "https://api.doubleword.ai/v1"
    assert seen["api_key"] == "dw-key-1"


def test_app_messages_streaming_returns_sse(monkeypatch, dw_target):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")

    async def fake_stream(**kwargs):
        async def gen():
            yield b"event: message_start\ndata: {}\n\n"
            yield b"event: message_stop\ndata: {}\n\n"
        return gen()

    monkeypatch.setattr(bp.litellm, "anthropic_messages", fake_stream)
    with _client(dw_target) as c:
        resp = c.post(
            "/v1/messages", headers=_AUTH,
            json={"model": _DW_NATIVE, "max_tokens": 16, "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert b"message_start" in resp.content
    assert b"message_stop" in resp.content


def test_app_count_tokens_route(monkeypatch, dw_target):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    resp = _client(dw_target).post(
        "/v1/messages/count_tokens", headers=_AUTH,
        json={"model": _DW_NATIVE,
              "messages": [{"role": "user", "content": "hello there"}],
              "system": "You are a helpful SQL assistant.",
              "tools": [{"name": "run_sql", "description": "Run SQL",
                         "input_schema": {"type": "object"}}]},
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0


def test_app_messages_rejects_missing_bearer(monkeypatch, dw_target):
    """A same-host caller that discovered the port via /healthz but doesn't know
    the provider key is rejected — it can't spend the actor's key."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    c = _client(dw_target)
    assert c.post("/v1/messages", json={"messages": []}).status_code == 401
    assert c.post(
        "/v1/messages", headers={"authorization": "Bearer wrong"},
        json={"messages": []},
    ).status_code == 401


def test_app_healthz_route_needs_no_auth(dw_target):
    # /healthz exposes no secret and is the identity probe — it stays open.
    resp = _client(dw_target).get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["provider"] == "doubleword"
    assert resp.json()["native_id"] == _DW_NATIVE


# ---------------------------------------------------------------------------
# ensure_bridge_proxy_for_actor — lifecycle / env / reuse / replace / refuse
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(bp, "_RUNTIME_DIR", tmp_path)
    return tmp_path


def test_ensure_starts_proxy_and_sets_env(
    monkeypatch, isolated_runtime
):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    started = []
    # No existing proxy on the port.
    monkeypatch.setattr(bp, "_probe_identity", lambda port: None)
    monkeypatch.setattr(
        bp, "_start_proxy",
        lambda target, port: started.append((target.provider, port)),
    )
    url = bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})

    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)
    assert url == f"http://127.0.0.1:{port}"
    assert started == [("doubleword", port)]
    # Codex #9: the override is what redirects the SDK subprocess.
    assert os.environ["BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL"] == url


def test_ensure_reuses_identity_matching_proxy(monkeypatch, isolated_runtime):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)

    # A healthy proxy with OUR identity AND our key fingerprint is on the port.
    def _ident(p):
        payload = bp.healthz_payload(target, port)
        payload["key_fp"] = bp.key_fingerprint("dw-key-1")
        return payload

    monkeypatch.setattr(bp, "_probe_identity", _ident)
    spawned = []
    monkeypatch.setattr(
        bp, "_start_proxy", lambda t, p: spawned.append(p)
    )
    url = bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})
    assert url == f"http://127.0.0.1:{port}"
    assert spawned == []  # reused, did NOT respawn


def test_ensure_refuses_healthy_mismatch(monkeypatch, isolated_runtime):
    """Codex #2: a healthy but FOREIGN service on the port is an anomaly — we
    must NOT kill it; fail fast instead."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setattr(
        bp, "_probe_identity",
        lambda p: {"provider": "someone-else", "upstream": "x",
                   "native_id": "y", "port": p},
    )
    monkeypatch.setattr(
        bp, "_start_proxy",
        lambda t, p: pytest.fail("must not spawn over a healthy mismatch"),
    )
    with pytest.raises(bp.BridgeProxyError):
        bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})


def test_key_fingerprint_is_nonsecret_stable(monkeypatch):
    fp = bp.key_fingerprint("secret-key-abc")
    assert fp == bp.key_fingerprint("secret-key-abc")   # stable
    assert "secret-key-abc" not in fp                   # never the key itself
    assert bp.key_fingerprint("other-key") != fp
    assert bp.key_fingerprint("") == ""


def test_ensure_restarts_own_child_on_rotated_key(monkeypatch, isolated_runtime):
    """A reused proxy that is OUR own child but holds a DIFFERENT key (rotation)
    would 401 our auth check — detect it via the key fingerprint and restart
    (Codex r3). Same target, different key_fp, pid IS our child → replace +
    start."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-NEW")
    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)
    stale = bp.healthz_payload(target, port)
    stale["key_fp"] = bp.key_fingerprint("dw-key-OLD")
    stale["pid"] = 4242

    class _Child:
        pid = 4242
    monkeypatch.setattr(bp, "_STARTED_PROCS", [_Child()])
    monkeypatch.setattr(bp, "_probe_identity", lambda p: stale)
    replaced, started = [], []
    monkeypatch.setattr(
        bp, "_replace_stale_proxy", lambda proc, p: replaced.append((proc.pid, p))
    )
    monkeypatch.setattr(bp, "_start_proxy", lambda t, p: started.append(p))
    bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})
    assert replaced == [(4242, port)]
    assert started == [port]


def test_ensure_refuses_non_child_stale_key_never_kills(monkeypatch, isolated_runtime):
    """Codex r5 (HIGH): a same-target proxy with a different/unverifiable key that
    is NOT our child must be REFUSED — never SIGTERM an arbitrary pid taken from
    the unauthenticated /healthz. Preserves 'never kill what we don't own'."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-NEW")
    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)
    spoof = bp.healthz_payload(target, port)
    spoof["key_fp"] = bp.key_fingerprint("attacker-key")
    spoof["pid"] = 13  # an arbitrary same-user pid
    monkeypatch.setattr(bp, "_STARTED_PROCS", [])  # not our child
    monkeypatch.setattr(bp, "_probe_identity", lambda p: spoof)
    monkeypatch.setattr(
        bp, "_start_proxy",
        lambda t, p: pytest.fail("must not start over an unverified process"),
    )
    monkeypatch.setattr(
        bp, "_replace_stale_proxy",
        lambda *a, **k: pytest.fail("must not kill an unverified process"),
    )
    with pytest.raises(bp.BridgeProxyError, match="did not start"):
        bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})


def test_ensure_refuses_same_target_missing_key_fp(monkeypatch, isolated_runtime):
    """CodeRabbit r5: a same-target proxy with NO key_fp (can't verify the key
    matches) must not be silently reused — present-vs-missing never matches, so
    it falls to the non-child refuse path."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)
    ident = bp.healthz_payload(target, port)  # no key_fp / pid set
    monkeypatch.setattr(bp, "_STARTED_PROCS", [])
    monkeypatch.setattr(bp, "_probe_identity", lambda p: ident)
    monkeypatch.setattr(
        bp, "_start_proxy", lambda t, p: pytest.fail("must not reuse/replace"),
    )
    with pytest.raises(bp.BridgeProxyError):
        bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.terminated = False
        self.waited = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


def test_replace_stale_proxy_reaps_own_child(monkeypatch):
    """`_replace_stale_proxy` reaps the OWN-child Popen (terminate/wait/remove)
    and never touches os.kill — the caller guarantees it's our child (Codex)."""
    proc = _FakeProc(4242)
    monkeypatch.setattr(bp, "_STARTED_PROCS", [proc])
    monkeypatch.setattr(
        bp.os, "kill",
        lambda *a, **k: pytest.fail("own child must be reaped, not os.kill'd"),
    )
    monkeypatch.setattr(bp, "_probe_identity", lambda p: None)  # port frees
    bp._replace_stale_proxy(proc, 8800)
    assert proc.terminated and proc.waited
    assert bp._STARTED_PROCS == []  # handle removed → reaped


def test_replace_stale_proxy_raises_if_port_not_freed(monkeypatch):
    """CodeRabbit r5: if the killed proxy never releases the port, raise instead
    of returning silently (a later _start_proxy would fail to bind)."""
    proc = _FakeProc(4242)
    monkeypatch.setattr(bp, "_STARTED_PROCS", [proc])
    # Port stays occupied forever.
    monkeypatch.setattr(
        bp, "_probe_identity", lambda p: {"provider": "x"}
    )
    # Make the wait loop terminate fast by faking the clock budget.
    monkeypatch.setattr(bp.time, "sleep", lambda s: None)
    ticks = iter([0.0, 0.5, 11.0])  # start, one poll, then past the 10s deadline
    monkeypatch.setattr(bp.time, "monotonic", lambda: next(ticks))
    with pytest.raises(bp.BridgeProxyError, match="did not release the port"):
        bp._replace_stale_proxy(proc, 8800)


def test_ensure_reuses_when_key_fp_matches(monkeypatch, isolated_runtime):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)
    ident = bp.healthz_payload(target, port)
    ident["key_fp"] = bp.key_fingerprint("dw-key-1")
    ident["pid"] = 123
    monkeypatch.setattr(bp, "_probe_identity", lambda p: ident)
    spawned = []
    monkeypatch.setattr(bp, "_start_proxy", lambda t, p: spawned.append(p))
    monkeypatch.setattr(
        bp, "_replace_stale_proxy", lambda pid, p: spawned.append("REPLACED")
    )
    bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})
    assert spawned == []  # full match incl key → reuse, no restart


def test_ensure_zai_coding_plan_fails_fast(monkeypatch, isolated_runtime):
    """Callers gate on agent_needs_bridge; a DIRECT call for a non-bridge model
    is misuse and must fail fast (loud), never silently spawn a phantom proxy."""
    monkeypatch.setattr(
        bp, "_start_proxy", lambda t, p: pytest.fail("no proxy for coding-plan")
    )
    # No bridge target resolvable -> error surfaced to the caller.
    with pytest.raises((ValueError, RuntimeError)):
        bp.ensure_bridge_proxy_for_actor(_ZAI_MODEL, {"no_subscription_auth": False})


# ---------------------------------------------------------------------------
# serve()/litellm config + lock usage + termination (Codex #5, #6)
# ---------------------------------------------------------------------------


def test_configure_litellm_sets_global_chat_completions_flag(monkeypatch):
    """Verified live: the per-model `openai/chat_completions/<id>` form does NOT
    work for anthropic_messages — only the global flag does. Pin that the proxy
    sets it."""
    monkeypatch.setattr(
        bp.litellm, "use_chat_completions_url_for_anthropic_messages", False,
        raising=False,
    )
    monkeypatch.setattr(bp.litellm, "drop_params", False, raising=False)
    bp.configure_litellm_for_bridge()
    assert bp.litellm.use_chat_completions_url_for_anthropic_messages is True
    # drop_params: the SDK sends OpenAI-unsupported params (reasoning_effort,
    # context_management) that would otherwise 500 every request.
    assert bp.litellm.drop_params is True


def test_create_app_configures_litellm_flag(monkeypatch, dw_target):
    monkeypatch.setattr(
        bp.litellm, "use_chat_completions_url_for_anthropic_messages", False,
        raising=False,
    )
    bp.create_app(dw_target)
    assert bp.litellm.use_chat_completions_url_for_anthropic_messages is True


def test_serve_host_is_loopback():
    assert bp.SERVE_HOST == "127.0.0.1"


def test_ensure_uses_per_port_lock_file(monkeypatch, isolated_runtime):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.delenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(bp, "_probe_identity", lambda port: None)
    monkeypatch.setattr(bp, "_start_proxy", lambda target, port: None)
    bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})
    target = bp.resolve_bridge_target(_DW_MODEL, True)
    port = bp.deterministic_port(target)
    # The lock file was created under the runtime dir during the critical section.
    assert bp.lock_path(port).exists()


def test_terminate_local_proxies_terminates_tracked(monkeypatch):
    class _FakeProc:
        def __init__(self):
            self.terminated = False

        def poll(self):
            # Alive until terminated, then reports an exit code.
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    proc = _FakeProc()
    monkeypatch.setattr(bp, "_STARTED_PROCS", [proc])
    bp.terminate_local_proxies()
    assert proc.terminated is True
    # A cleanly-terminated proc is dropped from the tracking list.
    assert bp._STARTED_PROCS == []


def test_terminate_local_proxies_keeps_survivor(monkeypatch):
    """A proc that refuses to die (poll stays None even after terminate+kill)
    must NOT be dropped from tracking — we keep the handle."""
    class _StubbornProc:
        def poll(self):
            return None  # never exits

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    proc = _StubbornProc()
    monkeypatch.setattr(bp, "_STARTED_PROCS", [proc])
    bp.terminate_local_proxies()
    assert bp._STARTED_PROCS == [proc]
