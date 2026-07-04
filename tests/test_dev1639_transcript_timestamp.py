"""DEV-1639: per-turn timestamps in the claude_sdk transcript.

Each serialized SDK message carries a ``ts`` (epoch receive-time). Because
``AssistantMessage``/``ResultMessage`` already carry their per-turn ``usage``
(incl. cache read/creation tokens), the persisted transcript then supports
per-turn cache-timing analysis (5m vs 1h) with NO usage/CallCost schema change.
"""

from __future__ import annotations

import dataclasses

from bird_interact_agents.agents.claude_sdk.sdk_env import serialize_sdk_message


@dataclasses.dataclass
class _FakeUsageMsg:
    usage: dict


def test_serialize_sdk_message_has_ts_and_data():
    msg = _FakeUsageMsg(usage={"input_tokens": 10, "cache_read_input_tokens": 5})
    out = serialize_sdk_message(msg)
    assert set(out) >= {"type", "data", "ts"}
    assert out["type"] == "_FakeUsageMsg"
    assert out["data"]["usage"]["cache_read_input_tokens"] == 5
    assert isinstance(out["ts"], float)
    assert out["ts"] > 0


def test_serialize_sdk_message_ts_present_on_non_dataclass():
    class _Weird:
        def __repr__(self):
            return "weird"

    out = serialize_sdk_message(_Weird())
    # Falls back to str(msg) for data but STILL stamps ts.
    assert isinstance(out["ts"], float)
    assert out["ts"] > 0
    assert out["type"] == "_Weird"


def test_serialize_sdk_message_ts_present_even_on_unserializable():
    class _Bad:
        def __str__(self):
            raise RuntimeError("boom")

    # Non-dataclass; asdict fails, str() raises → the "<unserializable>" branch,
    # which must still carry a ts.
    out = serialize_sdk_message(_Bad())
    assert out["data"] == "<unserializable>"
    assert isinstance(out["ts"], float)
    assert out["ts"] > 0
