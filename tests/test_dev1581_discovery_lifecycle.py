"""DEV-1581 R2 (Codex finding #4): the discovery client must be closed on
EVERY exit path — success, body exception, and cancellation — or the second
``claude`` CLI subprocess is orphaned.

R2 wires the warm discovery client through a ``discovery_session`` async
context manager that owns its creation + guaranteed teardown. These tests
drive it with a fake client to prove cleanup happens regardless of how the
``with`` body exits.
"""

from __future__ import annotations

import asyncio

import pytest

from bird_interact_agents.agents.claude_sdk.discovery_channel import (
    DiscoveryChannel,
    discovery_session,
    open_main_and_discovery,
)


class _FakeClient:
    def __init__(self):
        self.closed = 0

    async def aclose(self):
        self.closed += 1


async def _open_fake(client):
    return client


@pytest.mark.asyncio
async def test_session_closes_client_on_normal_exit():
    client = _FakeClient()
    async with discovery_session(
        open_client=lambda: _open_fake(client),
        usage_accum={},
        model="anthropic/claude-haiku-4-5-20251001",
    ) as ch:
        assert isinstance(ch, DiscoveryChannel)
    assert client.closed == 1


@pytest.mark.asyncio
async def test_session_closes_client_on_body_exception():
    client = _FakeClient()
    with pytest.raises(ValueError):
        async with discovery_session(
            open_client=lambda: _open_fake(client),
            usage_accum={},
            model="m",
        ):
            raise ValueError("main loop blew up")
    assert client.closed == 1, "discovery client leaked when main body raised"


@pytest.mark.asyncio
async def test_session_closes_client_on_cancellation():
    client = _FakeClient()
    started = asyncio.Event()

    async def run():
        async with discovery_session(
            open_client=lambda: _open_fake(client),
            usage_accum={},
            model="m",
        ):
            started.set()
            await asyncio.sleep(3600)  # park until cancelled

    task = asyncio.create_task(run())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed == 1, "discovery client orphaned on cancellation"


# --------------------------------------------------------------------------
# Orchestration: both clients under one stack, discovery closes AFTER main
# (Codex test-review #1) — so an in-flight ask_discovery during main shutdown
# never touches a closed discovery client.
# --------------------------------------------------------------------------
class _Recorder:
    def __init__(self, label, order):
        self.label = label
        self.order = order
        self.closed = 0

    async def aclose(self):
        self.closed += 1
        self.order.append(self.label)


@pytest.mark.asyncio
async def test_main_closes_before_discovery():
    order: list[str] = []
    main = _Recorder("main", order)
    disc = _Recorder("discovery", order)

    async with open_main_and_discovery(
        open_main=lambda: _open_fake(main),
        open_discovery=lambda: _open_fake(disc),
        usage_accum={},
        model="m",
    ) as (main_client, channel):
        assert main_client is main
        assert isinstance(channel, DiscoveryChannel)
        assert order == []  # nothing closed inside the body

    assert order == ["main", "discovery"], (
        "main must close before discovery so an in-flight ask_discovery during "
        "main shutdown still has a live discovery client"
    )
    assert main.closed == 1 and disc.closed == 1

