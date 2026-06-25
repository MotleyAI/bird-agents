"""DEV-1589 follow-up: per-session transcript capture is INTRINSIC to the
claude_sdk wrapper.

`hermetic_claude_sdk_session` yields the `ClaudeSDKClient` behind a transparent
recording proxy (`_TranscriptClient`) that tees every message streamed through
`receive_response()` into `.transcript`, accumulating across every
`query()`/`receive_response()` cycle on the warm client. Everything else
delegates to the wrapped client, so existing call sites are unaffected.
"""

from __future__ import annotations

import dataclasses

from bird_interact_agents.agents.claude_sdk import sdk_env


# --- fake SDK messages (only type().__name__ matters for serialisation) ---
class AssistantMessage:
    def __init__(self, text):
        self.text = text


class ResultMessage:
    pass


class _FakeRawClient:
    """Minimal stand-in for a ClaudeSDKClient; `script[i]` = the messages the
    i-th receive_response() yields."""

    def __init__(self, script):
        self._script = script
        self._i = 0
        self.queries = []
        self.closed = 0

    async def query(self, prompt):
        self.queries.append(prompt)

    def receive_response(self):
        msgs = self._script[self._i] if self._i < len(self._script) else []
        self._i += 1
        return self._gen(msgs)

    async def _gen(self, msgs):
        for m in msgs:
            yield m

    async def aclose(self):
        self.closed += 1

    async def get_mcp_status(self):
        return {"mcpServers": []}


async def test_transcript_records_each_message():
    raw = _FakeRawClient([[AssistantMessage("hi"), ResultMessage()]])
    client = sdk_env._TranscriptClient(raw)
    await client.query("go")
    msgs = [m async for m in client.receive_response()]
    assert len(msgs) == 2                       # passthrough intact
    assert [e["type"] for e in client.transcript] == ["AssistantMessage", "ResultMessage"]


async def test_transcript_accumulates_across_query_cycles():
    raw = _FakeRawClient([[AssistantMessage("a")],
                          [AssistantMessage("b"), ResultMessage()]])
    client = sdk_env._TranscriptClient(raw)
    await client.query("1")
    [m async for m in client.receive_response()]
    await client.query("2")
    [m async for m in client.receive_response()]
    # one transcript spans BOTH cycles (the warm-client re-prompt loop)
    assert len(client.transcript) == 3


async def test_proxy_delegates_unknown_attrs_and_query():
    raw = _FakeRawClient([])
    client = sdk_env._TranscriptClient(raw)
    # __getattr__ delegates non-overridden members to the wrapped client
    assert client.queries is raw.queries
    assert await client.get_mcp_status() == {"mcpServers": []}
    await client.query("x")
    assert raw.queries == ["x"]


async def test_record_closes_inner_stream_on_break():
    raw = _FakeRawClient([[AssistantMessage("a"), ResultMessage()]])
    client = sdk_env._TranscriptClient(raw)
    await client.query("go")
    agen = client.receive_response()
    async for _ in agen:
        break                      # consumer breaks early
    await agen.aclose()
    # capture still recorded what streamed before the break
    assert len(client.transcript) >= 1


def test_serialize_sdk_message_dataclass_and_fallback():
    @dataclasses.dataclass
    class DC:
        a: int
        b: str

    out = sdk_env.serialize_sdk_message(DC(a=1, b="x"))
    assert out == {"type": "DC", "data": {"a": 1, "b": "x"}}

    # non-dataclass → str fallback, type name preserved
    out2 = sdk_env.serialize_sdk_message(ResultMessage())
    assert out2["type"] == "ResultMessage"
    assert isinstance(out2["data"], str)
