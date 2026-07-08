"""DEV-1654 (D2): dispose the per-task in-process SLayer engine(s) at task
teardown so a reused cloud Ray actor doesn't accumulate asyncpg pools across
tasks (local is process-per-task, so this is a no-op there).

Two engines can exist per v1 task — the MAIN client and the warm DISCOVERY
client each build their own in-process SLayer server (in separate, possibly
copied, asyncio contexts). The disposal must therefore:

* reach the engine handle via the SHARED ``_ctx`` dict (mutated in place, so it
  propagates across context copies) — NOT ``_query.py``'s ``_query_state``
  ContextVar, whose ``.set()`` rebind in a child context is invisible to the
  parent teardown; and
* dispose EVERY stashed server (``_ctx['_slayer_mcps']``), not just the last
  one written (last-writer-wins would leak the other engine).

``dispose_slayer_engine`` calls SLayer 0.9.5's ``mcp._slayer_engine.aclose()``.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from bird_interact_agents.agents import _query
from bird_interact_agents.agents.claude_sdk import agent as csdk


class _FakeEngine:
    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _FakeToolFn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *a, **k):  # pragma: no cover
        return "ok"


def _make_fake_server(storage, names=("query",), with_engine: bool = True):
    tools = {n: type("Tool", (), {"fn": _FakeToolFn(n)})() for n in names}
    tm = type("TM", (), {"_tools": tools})()
    srv = type("Srv", (), {})()
    srv._tool_manager = tm
    if with_engine:
        srv._slayer_engine = _FakeEngine()
    return srv


@pytest.fixture(autouse=True)
def _reset_query_state():
    _query.attach_storage(None)
    yield
    _query.attach_storage(None)


# --- _query.dispose_slayer_engine: the low-level primitive -----------------

@pytest.mark.asyncio
async def test_dispose_slayer_engine_awaits_aclose():
    eng = _FakeEngine()
    mcp = type("M", (), {"_slayer_engine": eng})()
    await _query.dispose_slayer_engine(mcp)
    assert eng.aclose_calls == 1


@pytest.mark.asyncio
async def test_dispose_slayer_engine_noop_on_none_or_missing_engine():
    # None handle, and a server without a `_slayer_engine` (older SLayer), are
    # both silent no-ops — never raise.
    await _query.dispose_slayer_engine(None)
    await _query.dispose_slayer_engine(object())


@pytest.mark.asyncio
async def test_dispose_slayer_engine_swallows_aclose_error():
    class _Boom:
        async def aclose(self):
            raise RuntimeError("boom")

    mcp = type("M", (), {"_slayer_engine": _Boom()})()
    # Disposal must never propagate into the task/teardown.
    await _query.dispose_slayer_engine(mcp)


# --- agent-level stash + dispose -------------------------------------------

@pytest.mark.asyncio
async def test_ensure_storage_attached_stashes_server(monkeypatch):
    server = _make_fake_server(object())
    monkeypatch.setattr(
        "slayer.mcp.server.create_mcp_server", lambda s, **k: server,
    )
    csdk._ctx_var.set({"_slayer_storage": {"id": "s"}})
    csdk._ensure_slayer_storage_attached()
    stash = csdk._ctx.get("_slayer_mcps")
    assert stash and server in stash.values()


@pytest.mark.asyncio
async def test_dispose_task_disposes_all_stashed_servers():
    s1 = _make_fake_server(object(), names=("query",))
    s2 = _make_fake_server(object(), names=("inspect",))
    csdk._ctx_var.set({"_slayer_mcps": {id(s1): s1, id(s2): s2}})
    await csdk.dispose_task_slayer_engine()
    assert s1._slayer_engine.aclose_calls == 1
    assert s2._slayer_engine.aclose_calls == 1


@pytest.mark.asyncio
async def test_dispose_task_noop_when_nothing_stashed():
    # The raw / v0-subprocess case: no in-process server was ever built.
    csdk._ctx_var.set({})
    await csdk.dispose_task_slayer_engine()  # no _slayer_mcps -> no-op, no raise


@pytest.mark.asyncio
async def test_dispose_reaches_servers_built_in_child_contexts(monkeypatch):
    """MAIN and DISCOVERY build their engines in SEPARATE copied contexts
    (child asyncio tasks). Each stashes its server on the SHARED ``_ctx`` dict;
    the parent teardown disposes ALL of them.

    Pins two design points at once: (1) the shared-``_ctx``-dict channel (a
    ``_query_state`` read in the parent would miss a child's rebind), and (2)
    stashing ALL servers (``_slayer_mcps``) rather than last-writer-wins.
    """
    built = []

    def _factory(storage, **_kw):
        srv = _make_fake_server(storage, names=("query",))
        built.append(srv)
        return srv

    monkeypatch.setattr("slayer.mcp.server.create_mcp_server", _factory)
    _query.attach_storage(None)
    csdk._ctx_var.set({"_slayer_storage": {"id": "s"}})

    async def build_in_child():
        # Copied context: attach_storage rebinds _query_state HERE (invisible
        # to the parent), builds a fresh server, stashes it on the shared dict.
        csdk._ensure_slayer_storage_attached()

    await asyncio.create_task(build_in_child())   # "main" context
    await asyncio.create_task(build_in_child())   # "discovery" context

    assert len({id(s) for s in built}) == 2, "expected two distinct servers"
    await csdk.dispose_task_slayer_engine()        # parent teardown
    assert all(s._slayer_engine.aclose_calls == 1 for s in built)


@pytest.mark.asyncio
async def test_static_query_handler_stashes_and_disposes(monkeypatch):
    """The hottest path — the static ``query`` MCP handler (the MAIN client's
    query tool) — must register its SLayer server for disposal. Regression:
    it manually attached storage WITHOUT stashing, so the main client's engine
    leaked on a reused cloud actor.
    """
    eng = _FakeEngine()

    class _AsyncQueryFn:
        async def __call__(self, **kwargs):
            return "ok"

    tool = types.SimpleNamespace(fn=_AsyncQueryFn())
    server = types.SimpleNamespace(
        _tool_manager=types.SimpleNamespace(_tools={"query": tool}),
        _slayer_engine=eng,
    )
    monkeypatch.setattr(
        "slayer.mcp.server.create_mcp_server", lambda s, **k: server,
    )
    _query.attach_storage(None)
    csdk._ctx_var.set({"_slayer_storage": {"id": "s"}})

    await csdk.query.handler({"source_model": "orders"})

    stash = csdk._ctx.get("_slayer_mcps")
    assert stash and server in stash.values()
    await csdk.dispose_task_slayer_engine()
    assert eng.aclose_calls == 1
