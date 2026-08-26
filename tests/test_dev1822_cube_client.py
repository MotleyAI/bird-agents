"""DEV-1822: the Cube REST client — stdlib HS256 JWT + httpx wrapper with the
`Continue wait` long-poll loop and error surfacing."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import httpx
import pytest

from bird_interact_agents.cube_local.client import (
    CubeApiError,
    CubeClient,
    mint_cube_jwt,
)


def _b64url_json(seg: str) -> dict:
    pad = "=" * (-len(seg) % 4)
    return json.loads(base64.urlsafe_b64decode(seg + pad))


# --- JWT --------------------------------------------------------------------

def test_mint_jwt_payload_and_signature():
    token = mint_cube_jwt("s3cret", "alien", now=1_000_000, ttl_s=3600)
    header_seg, payload_seg, sig_seg = token.split(".")
    assert _b64url_json(header_seg) == {"alg": "HS256", "typ": "JWT"}
    assert _b64url_json(payload_seg) == {"db": "alien", "exp": 1_000_000 + 3600}
    expected = base64.urlsafe_b64encode(
        hmac.new(b"s3cret", f"{header_seg}.{payload_seg}".encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    assert sig_seg == expected


def _client(handler, **kw):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://cube.local/cubejs-api/v1")
    return CubeClient("http://cube.local/cubejs-api/v1", "s3cret", "alien",
                      http_client=http, **kw)


# --- load -------------------------------------------------------------------

def test_load_returns_data():
    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith("/load")
        assert request.headers["authorization"]  # JWT attached
        assert json.loads(request.content) == {"query": {"measures": ["orders.count"]}}
        return httpx.Response(200, json={"data": [{"orders.count": "5"}]})
    c = _client(handler)
    assert c.load({"measures": ["orders.count"]}) == [{"orders.count": "5"}]


def test_load_retries_on_continue_wait_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json={"error": "Continue wait"})
        return httpx.Response(200, json={"data": [{"x": "1"}]})

    c = _client(handler, continue_wait_timeout_s=5.0, sleep=lambda _s: None)
    assert c.load({"measures": ["orders.count"]}) == [{"x": "1"}]
    assert calls["n"] == 3


def test_load_continue_wait_times_out():
    def handler(request):
        return httpx.Response(200, json={"error": "Continue wait"})

    clock = {"t": 0.0}

    def fake_clock():
        clock["t"] += 1.0
        return clock["t"]

    c = _client(handler, continue_wait_timeout_s=2.0, sleep=lambda _s: None, clock=fake_clock)
    with pytest.raises(CubeApiError):
        c.load({"measures": ["orders.count"]})


def test_load_surfaces_server_error():
    def handler(request):
        return httpx.Response(400, json={"error": "Some user error"})
    c = _client(handler)
    with pytest.raises(CubeApiError) as ei:
        c.load({"measures": ["bad"]})
    assert "Some user error" in str(ei.value)


# --- meta / sql -------------------------------------------------------------

def test_meta_passthrough():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path.endswith("/meta")
        assert request.headers["authorization"]
        return httpx.Response(200, json={"cubes": [{"name": "orders"}]})
    c = _client(handler)
    assert c.meta()["cubes"][0]["name"] == "orders"


def test_sql_returns_text_and_params():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path.endswith("/sql")
        # query encoded as a JSON `query` param on /v1/sql
        assert json.loads(request.url.params["query"])["measures"] == ["orders.count"]
        return httpx.Response(200, json={"sql": {"sql": ["SELECT $1", ["US"]]}})
    c = _client(handler)
    text, params = c.sql({"measures": ["orders.count"],
                          "filters": [{"member": "o.region", "operator": "equals", "values": ["US"]}]})
    assert text == "SELECT $1"
    assert params == ["US"]
