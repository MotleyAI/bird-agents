"""DEV-1654 (D1): ONE in-process SLayer MCP server (hence one SlayerQueryEngine
and one asyncpg pool) per task context.

``agents/_query.py::_get_slayer_tool_fn`` historically called
``create_mcp_server(storage)`` on every cache MISS — i.e. once per distinct
tool NAME. Each call builds a fresh ``SlayerQueryEngine`` with its own asyncpg
connection pool, so the ~11 SLayer tool names the claude_sdk v1 surface pulls
opened several pools per task that were never disposed — exhausting the local
Postgres ``max_connections`` under concurrency.

These pin the fix: the per-task ``_TaskQueryState`` caches ONE ``mcp`` server
(``ensure_task_server``) and every tool fn is extracted from it, so
``create_mcp_server`` runs exactly once per task context regardless of how many
distinct names are pulled. A storage-identity change (a new task) still forces
a fresh server (one engine PER task, not shared across tasks).
"""

from __future__ import annotations

import pytest

from bird_interact_agents.agents import _query


class _FakeEngine:
    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _FakeToolFn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *a, **k):  # pragma: no cover - not invoked here
        return "ok"


_DEFAULT_NAMES = (
    "query", "query_nested", "inspect", "validate_models", "recommend_root_model",
)


def _make_fake_server(storage, names=_DEFAULT_NAMES, with_engine: bool = True):
    tools = {n: type("Tool", (), {"fn": _FakeToolFn(n)})() for n in names}
    tm = type("TM", (), {"_tools": tools})()
    srv = type("Srv", (), {})()
    srv._tool_manager = tm
    if with_engine:
        srv._slayer_engine = _FakeEngine()
    return srv


def _counting_factory(calls: dict, **kw):
    def _factory(storage, **_kw):
        calls["n"] += 1
        return _make_fake_server(storage, **kw)
    return _factory


@pytest.fixture(autouse=True)
def _reset_query_state():
    _query.attach_storage(None)
    yield
    _query.attach_storage(None)


def test_one_create_mcp_server_across_distinct_names(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        "slayer.mcp.server.create_mcp_server", _counting_factory(calls),
    )
    _query.attach_storage(object())
    names = list(_DEFAULT_NAMES)
    fns = [_query._get_slayer_tool_fn(n) for n in names]

    assert calls["n"] == 1, (
        f"expected exactly ONE create_mcp_server per task context regardless "
        f"of how many distinct tool names are pulled; got {calls['n']}"
    )
    # Every fn came from that single server (names line up).
    assert [f.name for f in fns] == names


def test_cached_name_is_not_rebuilt(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        "slayer.mcp.server.create_mcp_server", _counting_factory(calls),
    )
    _query.attach_storage(object())
    _query._get_slayer_tool_fn("query")
    _query._get_slayer_tool_fn("query")      # cached -> no rebuild
    _query._get_slayer_tool_fn("inspect")    # same server -> no rebuild
    assert calls["n"] == 1


def test_storage_swap_rebuilds_one_engine_per_task(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        "slayer.mcp.server.create_mcp_server", _counting_factory(calls),
    )
    _query.attach_storage(object())
    _query._get_slayer_tool_fn("query")
    _query.attach_storage(object())          # new identity == new task
    _query._get_slayer_tool_fn("query")
    assert calls["n"] == 2, (
        "a storage-identity change (new task) must rebuild the server so each "
        "task owns its own engine/pool"
    )


def test_ensure_task_server_returns_cached_server(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        "slayer.mcp.server.create_mcp_server", _counting_factory(calls),
    )
    _query.attach_storage(object())
    s1 = _query.ensure_task_server()
    s2 = _query.ensure_task_server()
    assert s1 is s2
    assert calls["n"] == 1
    # The tool fn the agent resolves comes from that same cached server.
    fn = _query._get_slayer_tool_fn("query")
    assert fn is s1._tool_manager._tools["query"].fn
