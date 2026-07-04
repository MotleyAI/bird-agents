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


def test_serialize_sdk_message_truncate_variant_keeps_ts():
    """The raw agents pass truncate=N to keep a light trajectory; data becomes a
    truncated str but the per-turn ts is still stamped (so 5m-vs-1h analysis works
    for the raw variants too)."""
    class _Chatty:
        def __str__(self):
            return "Z" * 5000

    out = serialize_sdk_message(_Chatty(), truncate=500)
    assert out["type"] == "_Chatty"
    assert isinstance(out["data"], str)
    assert len(out["data"]) == 500
    assert isinstance(out["ts"], float)
    assert out["ts"] > 0


def test_serialize_sdk_message_truncate_guards_bad_str():
    """truncate path must not raise on a pathological __str__ — it falls back to
    the unserializable marker and still stamps ts."""
    class _Bad:
        def __str__(self):
            raise RuntimeError("boom")

    out = serialize_sdk_message(_Bad(), truncate=500)
    assert out["data"] == "<unserializable>"
    assert isinstance(out["ts"], float)


def test_all_sdk_trajectory_builders_use_serialize():
    """Regression guard for the DEV-1639 gap the live smoke caught: the persisted
    trajectory of every claude_sdk* agent must be built via serialize_sdk_message
    (which stamps ts), NOT a hand-rolled `{"type": str(type(msg).__name__), ...}`
    dict that drops it."""
    import pathlib

    import bird_interact_agents.agents as agents_pkg

    agents_dir = pathlib.Path(agents_pkg.__file__).parent
    # The exact hand-rolled shape the smoke caught (matches single- or multi-line
    # appends since we scan the whole file text). serialize_sdk_message itself
    # lives in sdk_env.py and assigns `name = type(msg).__name__` — a different
    # form — so it is not a false positive.
    anti_pattern = '"type": str(type(msg).__name__)'
    offenders = [
        str(py) for py in agents_dir.rglob("*.py")
        if anti_pattern in py.read_text()
    ]
    assert not offenders, (
        "hand-rolled trajectory appends bypass serialize_sdk_message (no ts): "
        + "; ".join(offenders)
    )
