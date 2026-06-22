"""DEV-1581 R2: the in-process ``ask_discovery`` native must delegate to the
per-task ``DiscoveryChannel`` (Codex test-review #4 — a correct tool-name
constant is worthless if the runtime bridge is not wired).

The native reads the warm channel from the per-task context
(``_ctx["_discovery"]``) and returns ``channel.ask(question)``. We drive the
impl directly with a fake channel installed in the contextvar.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.agents.claude_sdk.agent import _ask_discovery_impl, _ctx_var


class _FakeChannel:
    def __init__(self):
        self.asked: list[str] = []

    async def ask(self, question: str) -> str:
        self.asked.append(question)
        return f"ANSWER<{question}>"


@pytest.mark.asyncio
async def test_ask_discovery_native_delegates_to_channel():
    ch = _FakeChannel()
    _ctx_var.set({"_discovery": ch})
    out = await _ask_discovery_impl("what is the join path?")
    assert ch.asked == ["what is the join path?"]
    assert out == "ANSWER<what is the join path?>"


@pytest.mark.asyncio
async def test_ask_discovery_native_handles_missing_channel():
    """If no discovery channel is wired (shouldn't happen), the native returns
    a usable error string rather than raising a KeyError into the main loop."""
    _ctx_var.set({})  # no "_discovery"
    out = await _ask_discovery_impl("q")
    assert isinstance(out, str) and out
