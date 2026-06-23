"""DEV-1581 R2: the shared in-process SLayer engine + contextvar isolation.

Two facts the R2 design relies on:

1. **Shared engine → coherence (Codex #3, your search-index point).** Main's
   write tools and discovery's introspection tools both resolve through the
   SAME task-local storage handle (one ``create_mcp_server`` engine), so a
   model main creates is visible to discovery's ``inspect_model`` / ``search``.
   We assert the *sharing* mechanically: a write tool fn and an introspection
   tool fn resolve against the same storage object within a task.

2. **contextvar isolation (Codex #7).** The in-process native tools read the
   per-task ``_ctx_var``; two concurrent tasks must each see their own ctx
   (``instance_id`` / ``slayer_storage_dir``) with no cross-talk.

(The end-to-end "discovery sees a model main just wrote" check needs a real
SLayer storage dir and lives in the integration suite; here we pin the
in-process plumbing that makes it possible.)
"""

from __future__ import annotations

import asyncio

import pytest

from bird_interact_agents.agents import _query
from bird_interact_agents.agents.claude_sdk.agent import _ctx_var


# --------------------------------------------------------------------------
# 1. Shared engine: write + introspection tools share one storage handle
# --------------------------------------------------------------------------
class _FakeToolFn:
    def __init__(self, storage_id, name):
        self.storage_id = storage_id
        self.name = name

    def __call__(self, *a, **k):  # pragma: no cover
        return self.storage_id


def _install_fake_create_mcp_server(monkeypatch):
    class _FakeServer:
        def __init__(self, storage):
            sid = storage["id"]

            class _TM:
                def __init__(self):
                    self._tools = {
                        n: type("T", (), {"fn": _FakeToolFn(sid, n)})()
                        for n in ("query", "create_model", "edit_model",
                                  "validate_models", "help", "search",
                                  "models_summary", "inspect_model")
                    }

            self._tool_manager = _TM()

    import slayer.mcp.server as srv

    monkeypatch.setattr(srv, "create_mcp_server", lambda storage, **kw: _FakeServer(storage))


def test_write_and_introspection_tools_share_one_engine(monkeypatch):
    """create_model (a MAIN tool) and inspect_model (a DISCOVERY tool) resolve
    against the same storage object → one engine → coherence."""
    _install_fake_create_mcp_server(monkeypatch)
    _query.attach_storage({"id": "shared"})
    write_fn = _query._get_slayer_tool_fn("create_model")
    introspect_fn = _query._get_slayer_tool_fn("inspect_model")
    assert write_fn.storage_id == introspect_fn.storage_id == "shared"


# --------------------------------------------------------------------------
# 2. contextvar isolation across concurrent tasks
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ctx_var_is_isolated_across_concurrent_tasks():
    seen: dict[str, str] = {}
    gate = asyncio.Event()

    async def task(iid: str, park_first: bool):
        _ctx_var.set({"instance_id": iid, "slayer_storage_dir": f"/dir/{iid}"})
        if park_first:
            await gate.wait()
        else:
            await asyncio.sleep(0)
            gate.set()
        # Read back what a native tool would read.
        ctx = _ctx_var.get()
        seen[iid] = ctx["instance_id"]

    await asyncio.gather(task("alien_1", True), task("alien_2", False))
    assert seen == {"alien_1": "alien_1", "alien_2": "alien_2"}
