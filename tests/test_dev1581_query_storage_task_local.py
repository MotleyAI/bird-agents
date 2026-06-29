"""DEV-1581 (Codex R2 finding #2): SLayer query storage must be TASK-LOCAL.

``_query.py`` historically kept the storage handle + the FastMCP-extracted
tool-fn caches as MODULE GLOBALS (``_slayer_storage`` /
``_cached_slayer_query_fn`` / ``_cached_slayer_tool_fns``). Under the R2
design two persistent clients (main + discovery) share one process, and
``run.py`` already runs multiple ``run_task``s concurrently via
``asyncio.gather`` — so two tasks with DIFFERENT ``slayer_storage_dir`` can
interleave: task A attaches storage A, yields inside SLayer query code, task
B attaches storage B (clobbering the global), and task A's next tool lookup
silently resolves against storage B.

These tests pin the post-fix contract: ``attach_storage`` + the tool-fn
cache are scoped to the current task (contextvar-backed), so concurrent
tasks never observe each other's storage. They FAIL against the
module-global implementation (cross-task leak) and pass once the handle is
task-local.
"""

from __future__ import annotations

import asyncio

import pytest

from bird_interact_agents.agents import _query


class _FakeToolFn:
    """A stand-in for a FastMCP tool ``.fn`` that remembers which storage
    object its owning ``create_mcp_server`` was built from."""

    def __init__(self, storage_id: str, name: str):
        self.storage_id = storage_id
        self.name = name

    def __call__(self, *a, **k):  # pragma: no cover - not invoked here
        return self.storage_id


def _install_fake_create_mcp_server(monkeypatch):
    """Patch ``slayer.mcp.server.create_mcp_server`` so each call yields a
    server whose tool ``.fn`` objects are tagged with the storage's id —
    letting a test assert which storage a resolved tool fn came from."""

    class _FakeServer:
        def __init__(self, storage):
            sid = storage["id"]

            class _TM:
                def __init__(self):
                    self._tools = {
                        n: type("T", (), {"fn": _FakeToolFn(sid, n)})()
                        for n in ("query", "inspect_model", "create_model")
                    }

            self._tool_manager = _TM()

    import slayer.mcp.server as srv

    monkeypatch.setattr(srv, "create_mcp_server", lambda storage, **kw: _FakeServer(storage))


@pytest.fixture(autouse=True)
def _reset_query_state():
    """Each test starts from a clean storage handle regardless of impl."""
    # Best-effort reset for whichever backing the impl uses.
    if hasattr(_query, "attach_storage"):
        try:
            _query.attach_storage(None)
        except Exception:
            pass
    yield


def test_attach_storage_resolves_current_storage(monkeypatch):
    """Baseline: after attaching storage S, a tool fn resolves against S."""
    _install_fake_create_mcp_server(monkeypatch)
    storage = {"id": "A"}
    _query.attach_storage(storage)
    fn = _query._get_slayer_tool_fn("query")
    assert fn.storage_id == "A"


@pytest.mark.asyncio
async def test_storage_is_task_local_under_concurrency(monkeypatch):
    """Two concurrent tasks with different storage must NOT clobber each
    other's tool-fn resolution.

    The interleave is forced with an event: task A attaches storage A and
    then awaits; while it is parked, task B attaches storage B and resolves
    a tool fn; task A resumes and resolves its own tool fn. With a
    module-global handle, A resolves against B (regression). Task-local
    storage keeps each correct.
    """
    _install_fake_create_mcp_server(monkeypatch)

    b_attached = asyncio.Event()
    a_resumed = asyncio.Event()
    results: dict[str, str] = {}

    async def task_a():
        _query.attach_storage({"id": "A"})
        # Park so task B can attach its own storage in between.
        await b_attached.wait()
        results["a"] = _query._get_slayer_tool_fn("query").storage_id
        a_resumed.set()

    async def task_b():
        # Let A attach + park first.
        await asyncio.sleep(0)
        _query.attach_storage({"id": "B"})
        results["b"] = _query._get_slayer_tool_fn("inspect_model").storage_id
        b_attached.set()
        await a_resumed.wait()

    await asyncio.gather(task_a(), task_b())

    assert results["b"] == "B"
    # The load-bearing assertion: A's resolution is NOT corrupted by B.
    assert results["a"] == "A", (
        "storage handle leaked across concurrent tasks — it must be "
        "task-local (contextvar-backed), not a module global"
    )
