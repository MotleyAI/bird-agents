"""DEV-1604 Anthropic-Messages ⇄ OpenAI-Chat-Completions bridge proxy.

The proxy is a tiny FastAPI app that fronts a provider's OpenAI endpoint and
speaks Anthropic `/v1/messages` to the Claude Agent SDK Node CLI. It reuses
litellm's translation (`litellm.anthropic_messages` with the chat-completions
flag) so we own only three thin routes.

DEV-1639: Doubleword now talks its NATIVE Anthropic endpoint directly and no
longer bridges, so the example bridge target here is **z.ai per-token** (the
remaining bridge user). These unit tests pin the seams that the blocking
integration gate (real SDK, real upstream) is too coarse to localise — and the
specific failure modes Codex flagged on the plan:

* the inbound SDK body carries `model`, so splatting it into
  `anthropic_messages(model=...)` would raise `TypeError` (Codex #1);
* a slash-bearing native id must never create nested `/tmp/.../<org>/`
  lock/marker/port paths (Codex #10 — kept via a synthetic slash-bearing target);
* a deterministic per-target port so distinct providers never collide and a
  healthy mismatch is never silently killed (Codex #2);
* `ensure_bridge_proxy_for_actor` sets `os.environ[base_url_env]` and reuses an
  identity-matching proxy / replaces a dead one / refuses a healthy mismatch.
"""

from __future__ import annotations

import os

import pytest

from bird_interact_agents.cloud import bridge_proxy as bp

_ZAI_MODEL = "zai/glm-5.2"
_ZAI_NATIVE = "glm-5.2"
_DW_MODEL = "doubleword/zai-org/GLM-5.2-FP8"


# DEV-1604: the 2nd arg is no_subscription_auth (True = per-token/bridge path).
# DEV-1639: z.ai per-token is now the canonical bridge example.
@pytest.fixture
def bridge_target(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    return bp.resolve_bridge_target(_ZAI_MODEL, True)


# A synthetic slash-bearing target keeps the Codex #10 slash-safety regression
# alive now that no *real* bridgeable provider has a slash in its native id.
@pytest.fixture
def slash_target():
    return bp.BridgeTarget(
        provider="synthetic",
        openai_base_url="https://example.invalid/v1",
        native_id="some-org/Model-Name-FP8",
        auth_env="SYNTHETIC_API_KEY",
    )


# ---------------------------------------------------------------------------
# resolve_bridge_target
# ---------------------------------------------------------------------------


def test_resolve_bridge_target_zai_per_token(bridge_target):
    assert bridge_target.provider == "zai"
    assert bridge_target.openai_base_url == "https://api.z.ai/api/paas/v4"
    assert bridge_target.native_id == _ZAI_NATIVE
    assert bridge_target.auth_env == "ZAI_API_KEY"


def test_resolve_bridge_target_rejects_doubleword(monkeypatch):
    """DEV-1639: Doubleword talks its native Anthropic endpoint directly, so it
    no longer needs the bridge — resolving a bridge target for it is misuse and
    must fail fast (callers gate on agent_needs_bridge)."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    with pytest.raises((ValueError, RuntimeError)):
        bp.resolve_bridge_target(_DW_MODEL, True)
    with pytest.raises((ValueError, RuntimeError)):
        bp.resolve_bridge_target(_DW_MODEL, False)


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


def test_messages_call_kwargs_overrides_inbound_model(bridge_target):
    """The SDK body contains `model=glm-5.2` (the bare native id). We must
    REPLACE it with `openai/<native_id>` — not splat both."""
    body = {
        "model": _ZAI_NATIVE,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    kwargs = bp.messages_call_kwargs(body, bridge_target, api_key="zai-key-1")
    # Exactly one `model`, and it's the openai-routed form.
    assert kwargs["model"] == f"openai/{_ZAI_NATIVE}"
    assert kwargs["api_base"] == "https://api.z.ai/api/paas/v4"
    assert kwargs["api_key"] == "zai-key-1"
    assert kwargs["custom_llm_provider"] == "openai"
    # Passes the SDK fields through untouched.
    assert kwargs["max_tokens"] == 256
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_messages_call_kwargs_does_not_mutate_body(bridge_target):
    body = {"model": _ZAI_NATIVE, "max_tokens": 8, "messages": []}
    bp.messages_call_kwargs(body, bridge_target, api_key="k")
    assert body["model"] == _ZAI_NATIVE  # caller's dict is untouched


def test_messages_call_kwargs_can_splat_into_anthropic_messages_signature(
    bridge_target,
):
    """Regression for the TypeError: build kwargs, then confirm they bind to
    the real `litellm.anthropic_messages` signature with no duplicate `model`."""
    import inspect

    import litellm

    body = {
        "model": _ZAI_NATIVE,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    kwargs = bp.messages_call_kwargs(body, bridge_target, api_key="k")
    sig = inspect.signature(litellm.anthropic_messages)
    # Would raise TypeError on a duplicate/unknown-without-**kwargs binding.
    sig.bind_partial(**kwargs)


# ---------------------------------------------------------------------------
# count_tokens — real Anthropic response shape (Codex #4)
# ---------------------------------------------------------------------------


def test_count_tokens_response_shape(bridge_target):
    body = {
        "model": _ZAI_NATIVE,
        "messages": [{"role": "user", "content": "count these tokens please"}],
    }
    out = bp.count_tokens_response(body, bridge_target)
    assert set(out) == {"input_tokens"}
    assert isinstance(out["input_tokens"], int)
    assert out["input_tokens"] > 0


# ---------------------------------------------------------------------------
# deterministic port — distinct targets never collide (Codex #2)
# ---------------------------------------------------------------------------


def test_deterministic_port_is_stable(bridge_target):
    assert bp.deterministic_port(bridge_target) == bp.deterministic_port(
        bridge_target
    )


def test_deterministic_port_distinct_across_targets(bridge_target, slash_target):
    # Distinct targets must get pairwise-distinct ports so two providers never
    # fight over one loopback socket.
    targets = [bridge_target, slash_target]
    ports = [bp.deterministic_port(t) for t in targets]
    assert len(set(ports)) == len(ports)


def test_deterministic_port_in_loopback_range(bridge_target, slash_target):
    for t in (bridge_target, slash_target):
        p = bp.deterministic_port(t)
        assert isinstance(p, int)
        assert bp.PORT_BASE <= p < bp.PORT_BASE + bp.PORT_SPAN


# ---------------------------------------------------------------------------
# slash-safety: marker/lock paths must not nest on the native-id slash (#10)
# ---------------------------------------------------------------------------


def test_marker_and_lock_paths_are_slash_safe(monkeypatch, tmp_path, slash_target):
    monkeypatch.setattr(bp, "_RUNTIME_DIR", tmp_path)
    port = bp.deterministic_port(slash_target)
    marker = bp.marker_path(port)
    lock = bp.lock_path(port)
    # Both live DIRECTLY under the runtime dir — the embedded `some-org/` slash
    # in the native id must never have created a nested directory.
    assert marker.parent == tmp_path
    assert lock.parent == tmp_path
    assert "some-org" not in str(marker)
    assert "some-org" not in str(lock)
    assert str(port) in marker.name


# ---------------------------------------------------------------------------
# identity / fingerprint
# ---------------------------------------------------------------------------


def test_healthz_payload_carries_identity(bridge_target):
    port = bp.deterministic_port(bridge_target)
    payload = bp.healthz_payload(bridge_target, port)
    assert payload["provider"] == "zai"
    assert payload["upstream"] == "https://api.z.ai/api/paas/v4"
    assert payload["native_id"] == _ZAI_NATIVE
    assert payload["port"] == port


def test_identity_matches_same_target_mismatch_other(bridge_target, slash_target):
    pz = bp.healthz_payload(bridge_target, bp.deterministic_port(bridge_target))
    ps = bp.healthz_payload(slash_target, bp.deterministic_port(slash_target))
    assert bp.identity_matches(pz, pz) is True
    assert bp.identity_matches(pz, ps) is False
    # A partial/garbage payload from a foreign service is a mismatch, not a crash.
    assert bp.identity_matches({"hello": "world"}, pz) is False


# ---------------------------------------------------------------------------
# FastAPI app routes via TestClient (litellm.anthropic_messages monkeypatched)
# ---------------------------------------------------------------------------


def _client(target):
    from starlette.testclient import TestClient

    return TestClient(bp.create_app(target))


def test_app_messages_non_stream_routes_to_openai_native(monkeypatch, bridge_target):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    seen = {}

    async def fake_messages(**kwargs):
        seen.update(kwargs)
        return {"type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1}}

    monkeypatch.setattr(bp.litellm, "anthropic_messages", fake_messages)
    resp = _client(bridge_target).post(
        "/v1/messages",
        json={"model": _ZAI_NATIVE, "max_tokens": 16,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "ok"
    # The inbound bare native id was replaced with the openai-routed form,
    # the upstream + key were injected, and NO duplicate `model` was passed.
    assert seen["model"] == f"openai/{_ZAI_NATIVE}"
    assert seen["api_base"] == "https://api.z.ai/api/paas/v4"
    assert seen["api_key"] == "zai-key-1"


def test_app_messages_streaming_returns_sse(monkeypatch, bridge_target):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")

    async def fake_stream(**kwargs):
        async def gen():
            yield b"event: message_start\ndata: {}\n\n"
            yield b"event: message_stop\ndata: {}\n\n"
        return gen()

    monkeypatch.setattr(bp.litellm, "anthropic_messages", fake_stream)
    with _client(bridge_target) as c:
        resp = c.post(
            "/v1/messages",
            json={"model": _ZAI_NATIVE, "max_tokens": 16, "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert b"message_start" in resp.content
    assert b"message_stop" in resp.content


def test_app_count_tokens_route(monkeypatch, bridge_target):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    resp = _client(bridge_target).post(
        "/v1/messages/count_tokens",
        json={"model": _ZAI_NATIVE,
              "messages": [{"role": "user", "content": "hello there"}],
              "system": "You are a helpful SQL assistant.",
              "tools": [{"name": "run_sql", "description": "Run SQL",
                         "input_schema": {"type": "object"}}]},
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0


def test_app_healthz_route(bridge_target):
    resp = _client(bridge_target).get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["provider"] == "zai"
    assert resp.json()["native_id"] == _ZAI_NATIVE


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
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.delenv("BIRD_ZAI_ANTHROPIC_BASE_URL", raising=False)
    started = []
    # No existing proxy on the port.
    monkeypatch.setattr(bp, "_probe_identity", lambda port: None)
    monkeypatch.setattr(
        bp, "_start_proxy",
        lambda target, port: started.append((target.provider, port)),
    )
    url = bp.ensure_bridge_proxy_for_actor(_ZAI_MODEL, {"no_subscription_auth": True})

    target = bp.resolve_bridge_target(_ZAI_MODEL, True)
    port = bp.deterministic_port(target)
    assert url == f"http://127.0.0.1:{port}"
    assert started == [("zai", port)]
    # Codex #9: the override is what redirects the SDK subprocess.
    assert os.environ["BIRD_ZAI_ANTHROPIC_BASE_URL"] == url


def test_ensure_reuses_identity_matching_proxy(monkeypatch, isolated_runtime):
    """A healthy proxy with OUR identity (same provider/upstream/native id) is
    reused — concurrent actors on one VM share it; the run's key is the same so
    its upstream auth stays valid. No inbound auth / key fingerprint: the proxy
    is loopback-only on a transient single-tenant VM."""
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    target = bp.resolve_bridge_target(_ZAI_MODEL, True)
    port = bp.deterministic_port(target)
    monkeypatch.setattr(
        bp, "_probe_identity", lambda p: bp.healthz_payload(target, port)
    )
    spawned = []
    monkeypatch.setattr(bp, "_start_proxy", lambda t, p: spawned.append(p))
    url = bp.ensure_bridge_proxy_for_actor(_ZAI_MODEL, {"no_subscription_auth": True})
    assert url == f"http://127.0.0.1:{port}"
    assert spawned == []  # reused, did NOT respawn


def test_ensure_refuses_healthy_mismatch(monkeypatch, isolated_runtime):
    """A healthy but FOREIGN service on the port (different identity) is a port
    collision — refuse, never replace something we didn't start."""
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
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
        bp.ensure_bridge_proxy_for_actor(_ZAI_MODEL, {"no_subscription_auth": True})


def test_ensure_zai_coding_plan_fails_fast(monkeypatch, isolated_runtime):
    """Callers gate on agent_needs_bridge; a DIRECT call for a non-bridge model
    is misuse and must fail fast (loud), never silently spawn a phantom proxy."""
    monkeypatch.setattr(
        bp, "_start_proxy", lambda t, p: pytest.fail("no proxy for coding-plan")
    )
    # No bridge target resolvable -> error surfaced to the caller.
    with pytest.raises((ValueError, RuntimeError)):
        bp.ensure_bridge_proxy_for_actor(_ZAI_MODEL, {"no_subscription_auth": False})


def test_ensure_doubleword_fails_fast(monkeypatch, isolated_runtime):
    """DEV-1639: Doubleword no longer bridges; a direct ensure_* call for it is
    misuse and must fail fast, never spawn a phantom proxy."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setattr(
        bp, "_start_proxy", lambda t, p: pytest.fail("no proxy for doubleword")
    )
    with pytest.raises((ValueError, RuntimeError)):
        bp.ensure_bridge_proxy_for_actor(_DW_MODEL, {"no_subscription_auth": True})


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


def test_create_app_configures_litellm_flag(monkeypatch, bridge_target):
    monkeypatch.setattr(
        bp.litellm, "use_chat_completions_url_for_anthropic_messages", False,
        raising=False,
    )
    bp.create_app(bridge_target)
    assert bp.litellm.use_chat_completions_url_for_anthropic_messages is True


def test_serve_host_is_loopback():
    assert bp.SERVE_HOST == "127.0.0.1"


def test_ensure_uses_per_port_lock_file(monkeypatch, isolated_runtime):
    monkeypatch.setenv("ZAI_API_KEY", "zai-key-1")
    monkeypatch.delenv("BIRD_ZAI_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(bp, "_probe_identity", lambda port: None)
    monkeypatch.setattr(bp, "_start_proxy", lambda target, port: None)
    bp.ensure_bridge_proxy_for_actor(_ZAI_MODEL, {"no_subscription_auth": True})
    target = bp.resolve_bridge_target(_ZAI_MODEL, True)
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
