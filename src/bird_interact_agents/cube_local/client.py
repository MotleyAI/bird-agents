"""DEV-1822: thin Cube.js REST client — stdlib HS256 JWT (no PyJWT dep) + an
httpx wrapper with the `/v1/load` `Continue wait` long-poll loop."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Callable

import httpx


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def mint_cube_jwt(secret: str, db: str, *, now: int, ttl_s: int = 86400) -> str:
    """Mint an HS256 JWT whose payload security context selects the tenant DB."""
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"},
                                separators=(",", ":")).encode())
    payload = _b64url(json.dumps({"db": db, "exp": now + ttl_s},
                                 separators=(",", ":")).encode())
    sig = _b64url(hmac.new(secret.encode(), f"{header}.{payload}".encode(),
                           hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


class CubeApiError(Exception):
    """A Cube REST call failed (non-2xx, error body, or Continue-wait timeout)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CubeClient:
    """Per-task Cube REST client bound to one tenant DB."""

    def __init__(
        self, base_url: str, api_secret: str, db: str, *,
        http_client: httpx.Client | None = None,
        timeout_s: float = 30.0,
        continue_wait_timeout_s: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_secret = api_secret
        self.db = db
        self._http = http_client or httpx.Client(timeout=timeout_s)
        self._continue_wait_timeout_s = continue_wait_timeout_s
        self._sleep = sleep
        self._clock = clock

    # -- helpers -------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": mint_cube_jwt(self.api_secret, self.db, now=int(time.time()))}

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _err_msg(resp: httpx.Response, body: Any) -> str:
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
        return f"Cube API error {resp.status_code}: {resp.text[:300]}"

    def _raise_for(self, resp: httpx.Response, body: Any) -> None:
        if resp.status_code >= 400:
            raise CubeApiError(self._err_msg(resp, body))
        if isinstance(body, dict) and body.get("error"):
            raise CubeApiError(str(body["error"]))

    # -- endpoints -----------------------------------------------------------

    def meta(self) -> dict:
        resp = self._http.get(f"{self.base_url}/meta", headers=self._headers())
        body = self._safe_json(resp)
        self._raise_for(resp, body)
        return body

    def load(self, query: dict) -> Any:
        deadline = self._clock() + self._continue_wait_timeout_s
        while True:
            resp = self._http.post(f"{self.base_url}/load",
                                   json={"query": query}, headers=self._headers())
            body = self._safe_json(resp)
            if resp.status_code >= 400:
                raise CubeApiError(self._err_msg(resp, body))
            if isinstance(body, dict) and body.get("error") == "Continue wait":
                if self._clock() >= deadline:
                    raise CubeApiError("Cube /v1/load timed out (Continue wait)")
                self._sleep(1.0)
                continue
            if isinstance(body, dict) and body.get("error"):
                raise CubeApiError(str(body["error"]))
            return body.get("data") if isinstance(body, dict) else body

    def sql(self, query: dict) -> tuple[str, list]:
        resp = self._http.get(f"{self.base_url}/sql",
                              params={"query": json.dumps(query)}, headers=self._headers())
        body = self._safe_json(resp)
        self._raise_for(resp, body)
        pair = body["sql"]["sql"]
        return pair[0], pair[1]
