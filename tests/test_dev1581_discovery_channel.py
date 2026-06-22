"""DEV-1581 R2: ``DiscoveryChannel`` — the warm persistent-discovery bridge.

R2 replaces the SDK-subagent split with two persistent ``ClaudeSDKClient``s
in one process: the main loop, and a long-lived *discovery* client that main
reaches through an in-process ``ask_discovery(question)`` tool. The bridge
object is ``DiscoveryChannel``; these tests pin its contract independent of a
live SDK by injecting a fake discovery client + a fake usage tracker.

Contract (Codex R2 review, findings #1/#4/#5/#6):

* ``ask`` forwards the question to the discovery client and returns its text.
* **Single-flight**: concurrent ``ask`` calls are serialised — the discovery
  client is never driven by two overlapping ``query``/``receive_response``
  cycles (a second consumer of the same stream corrupts session state).
* **Call cap**: after ``max_calls`` the channel stops querying discovery and
  returns a usable tool message telling main to proceed (R2's analog of the
  old Task cap — warm follow-ups must be bounded).
* **Per-stream usage**: a FRESH usage tracker is created and finalised for
  every ``ask`` so multi-call discovery usage is fully counted (the real
  ``SdkUsageTracker`` is idempotent after one ``finalize``).
* **Error handling**: a dead / raising / empty discovery stream yields a
  usable error string, never a hang and never a raised exception into main.
* **Lifecycle**: ``aclose`` closes the underlying client exactly once.
"""

from __future__ import annotations

import asyncio

import pytest

from bird_interact_agents.agents.claude_sdk.discovery_channel import (
    DISCOVERY_CALL_CAP_MESSAGE,
    DiscoveryChannel,
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _Result:
    """Stand-in for the SDK ``ResultMessage`` (loop terminator)."""


class _Assistant:
    """Stand-in for an ``AssistantMessage`` carrying text blocks + usage."""

    def __init__(self, text: str, usage_tokens: int = 0):
        self.content = [type("TextBlock", (), {"text": text})()]
        self.usage_tokens = usage_tokens


class FakeDiscoveryClient:
    """Minimal stand-in for a persistent ClaudeSDKClient.

    ``script`` maps the Nth ``ask`` (0-based) to the list of messages its
    ``receive_response`` will yield. Records concurrency so the single-flight
    test can prove no overlap. ``raise_on_query`` / empty scripts exercise the
    error paths.
    """

    def __init__(self, script, *, hold: asyncio.Event | None = None):
        self._script = script
        self._n = 0
        self.closed = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._hold = hold
        self.queries: list[str] = []

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def receive_response(self):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self._hold is not None:
                await self._hold.wait()
            msgs = self._script[self._n] if self._n < len(self._script) else []
            self._n += 1
            for m in msgs:
                if isinstance(m, Exception):
                    raise m
                yield m
        finally:
            self.in_flight -= 1

    async def aclose(self) -> None:
        self.closed += 1


class FakeTracker:
    """Records that a fresh tracker was finalised per stream, summing tokens
    into a shared accumulator dict so the test can assert total usage."""

    def __init__(self, accum: dict, model: str):
        self._accum = accum
        self._model = model
        self._seen = 0
        self.finalized = 0

    def observe(self, msg) -> None:
        self._seen += getattr(msg, "usage_tokens", 0)

    def finalize(self) -> None:
        self.finalized += 1
        self._accum["tokens"] = self._accum.get("tokens", 0) + self._seen
        self._accum["streams"] = self._accum.get("streams", 0) + 1


def _is_result(msg) -> bool:
    return isinstance(msg, _Result)


def _make_channel(client, accum, *, max_calls=10):
    """Build a DiscoveryChannel wired to the fakes. The factory + result
    predicate are injected so the unit test never needs the real SDK types."""
    return DiscoveryChannel(
        client=client,
        usage_accum=accum,
        model="anthropic/claude-haiku-4-5-20251001",
        max_calls=max_calls,
        tracker_factory=lambda acc, model: FakeTracker(acc, model),
        is_result=_is_result,
    )


# --------------------------------------------------------------------------
# Forwarding
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_forwards_question_and_returns_text():
    client = FakeDiscoveryClient([[_Assistant("HANDOFF-REPORT"), _Result()]])
    ch = _make_channel(client, {})
    out = await ch.ask("what is the schema?")
    assert "HANDOFF-REPORT" in out
    assert client.queries == ["what is the schema?"]


@pytest.mark.asyncio
async def test_ask_concatenates_multiple_text_blocks():
    client = FakeDiscoveryClient([[_Assistant("part-A "), _Assistant("part-B"), _Result()]])
    ch = _make_channel(client, {})
    out = await ch.ask("q")
    assert "part-A" in out and "part-B" in out


# --------------------------------------------------------------------------
# Single-flight (finding #1)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_ask_calls_are_serialised():
    hold = asyncio.Event()
    client = FakeDiscoveryClient(
        [[_Assistant("first"), _Result()], [_Assistant("second"), _Result()]],
        hold=hold,
    )
    ch = _make_channel(client, {})
    t1 = asyncio.create_task(ch.ask("q1"))
    t2 = asyncio.create_task(ch.ask("q2"))
    await asyncio.sleep(0.05)  # both tasks now contending
    hold.set()
    await asyncio.gather(t1, t2)
    # The lock must guarantee the discovery stream is never drained twice at once.
    assert client.max_in_flight == 1, (
        "ask_discovery must single-flight the discovery client — two "
        "overlapping receive_response cycles corrupt the warm session"
    )


# --------------------------------------------------------------------------
# Call cap (finding #6)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_call_cap_stops_querying_and_returns_tool_message():
    script = [[_Assistant(f"ans-{i}"), _Result()] for i in range(3)]
    client = FakeDiscoveryClient(script)
    ch = _make_channel(client, {}, max_calls=2)
    assert "ans-0" in await ch.ask("q0")
    assert "ans-1" in await ch.ask("q1")
    capped = await ch.ask("q2")
    # Third call must NOT hit the client and must return the cap sentinel
    # (a known, actionable "proceed without more discovery" message).
    assert len(client.queries) == 2
    assert capped == DISCOVERY_CALL_CAP_MESSAGE
    assert "ans-2" not in capped


@pytest.mark.asyncio
async def test_failed_ask_counts_toward_cap():
    """A failed discovery stream still consumes a cap slot — otherwise a
    broken discovery client could be retried unboundedly."""
    client = FakeDiscoveryClient(
        [[RuntimeError("boom1")], [RuntimeError("boom2")],
         [_Assistant("late"), _Result()]]
    )
    ch = _make_channel(client, {}, max_calls=2)
    await ch.ask("q0")  # fails, counts (1/2)
    await ch.ask("q1")  # fails, counts (2/2)
    capped = await ch.ask("q2")
    assert capped == DISCOVERY_CALL_CAP_MESSAGE
    # Only the first two attempts ever reached the client.
    assert len(client.queries) == 2


@pytest.mark.asyncio
async def test_truly_empty_stream_returns_usable_message():
    """A stream that yields NOTHING (not even a ResultMessage) returns a
    usable string, never a hang."""
    client = FakeDiscoveryClient([[]])  # empty message list
    ch = _make_channel(client, {})
    out = await ch.ask("q")
    assert out and isinstance(out, str)


# --------------------------------------------------------------------------
# Per-stream usage (finding #5)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_usage_counted_for_every_ask_stream():
    client = FakeDiscoveryClient(
        [[_Assistant("a", usage_tokens=10), _Result()],
         [_Assistant("b", usage_tokens=7), _Result()]]
    )
    accum: dict = {}
    ch = _make_channel(client, accum)
    await ch.ask("q1")
    await ch.ask("q2")
    # A fresh tracker per stream → both calls' usage is aggregated.
    assert accum.get("streams") == 2
    assert accum.get("tokens") == 17


# --------------------------------------------------------------------------
# Error handling (finding #6 / robustness)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_returns_error_string_when_stream_raises():
    client = FakeDiscoveryClient([[RuntimeError("discovery died")]])
    ch = _make_channel(client, {})
    out = await ch.ask("q")
    assert out  # usable string, not a raise
    assert "error" in out.lower() or "discovery" in out.lower()


@pytest.mark.asyncio
async def test_ask_returns_message_when_stream_empty():
    client = FakeDiscoveryClient([[_Result()]])  # no text
    ch = _make_channel(client, {})
    out = await ch.ask("q")
    assert out  # non-empty usable message even with no discovery text


@pytest.mark.asyncio
async def test_error_does_not_consume_a_cap_slot_inconsistently():
    """A raising stream still releases the single-flight lock so the channel
    stays usable for the next call."""
    client = FakeDiscoveryClient([[RuntimeError("boom")], [_Assistant("ok"), _Result()]])
    ch = _make_channel(client, {})
    await ch.ask("q1")
    out = await ch.ask("q2")
    assert "ok" in out


# --------------------------------------------------------------------------
# Lifecycle (finding #4)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_aclose_closes_client_once():
    client = FakeDiscoveryClient([])
    ch = _make_channel(client, {})
    await ch.aclose()
    await ch.aclose()  # idempotent
    assert client.closed == 1
