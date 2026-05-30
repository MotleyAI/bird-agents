"""SLayer MCP-server startup-handshake timeout must stay generous (DEV-1478).

Root cause of the DEV-1478 alien/credit cloud timeouts: the slayer stdio MCP
server runs `--ingest-on-startup`, whose cost is a datasource schema
RE-REFLECTION + semantic-layer rebuild (CPU, scales with schema size). Under
multi-actor CPU contention that exceeded the prior 300s handshake budget, so
every large-schema task (e.g. alien, 30+ models) failed at MCP `initialize()`.
Embeddings are NOT the cost — they're prebuilt in the reference's
`embeddings.db`, copied into each task variant, and hash-skipped on startup.

These tests lock a generous shared budget and assert each adapter's MCP server
is constructed with it (catches a regression back to a tight per-site value).
"""

from __future__ import annotations

from bird_interact_agents.harness import SLAYER_MCP_STARTUP_TIMEOUT_S


def test_startup_timeout_has_plenty_of_room():
    # ~30-50s uncontended reflection for a big schema; the budget must clear
    # that by a wide margin so multi-actor contention can't trip it.
    assert SLAYER_MCP_STARTUP_TIMEOUT_S >= 1800


def test_otf_encode_server_uses_shared_timeout(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _build_shared_slayer_server,
    )

    server = _build_shared_slayer_server(str(tmp_path))
    assert server.timeout == SLAYER_MCP_STARTUP_TIMEOUT_S


def test_recursive_server_uses_shared_timeout(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        _build_shared_slayer_server,
    )

    server = _build_shared_slayer_server(str(tmp_path))
    assert server.timeout == SLAYER_MCP_STARTUP_TIMEOUT_S


def test_pydantic_ai_adapter_uses_shared_timeout(tmp_path):
    """The base `pydantic_ai` adapter builds its MCP server inline in
    `_build_slayer_agent` (not via a standalone helper), so assert the
    constructed server in the agent's toolsets carries the shared timeout —
    otherwise a per-site `timeout=300` regression here would go uncaught."""
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.models.test import TestModel

    from bird_interact_agents.agents.pydantic_ai.agent import _build_slayer_agent

    agent = _build_slayer_agent(model=TestModel(), slayer_storage_dir=str(tmp_path))
    # `_user_toolsets` is the raw user-supplied list (repo convention — see
    # test_root_clarifier_no_pinning.py), not the derived `agent.toolsets` view.
    servers = [t for t in agent._user_toolsets if isinstance(t, MCPServerStdio)]
    assert len(servers) == 1
    assert servers[0].timeout == SLAYER_MCP_STARTUP_TIMEOUT_S


# ---------------------------------------------------------------------------
# DEV-1508: OTF adapters skip `--ingest-on-startup` so the slayer MCP boots
# fast on a warm OTF cache. The cache by construction IS the post-ingestion
# state; a re-ingest is wasted work AND on the Claude Agent SDK there is no
# timeout knob to wait through it.
# ---------------------------------------------------------------------------


def _spy_slayer_mcp_stdio_config(monkeypatch, target_module):
    """Patch ``slayer_mcp_stdio_config`` on ``target_module`` with a spy that
    records the kwargs and returns a minimal config dict (no real binary
    resolution, no SLAYER_STORAGE check)."""
    captured = {}

    def spy(storage_dir, **kw):
        captured["storage_dir"] = storage_dir
        captured["kw"] = dict(kw)
        return {"command": "slayer", "args": ["mcp"], "env": {}}

    monkeypatch.setattr(target_module, "slayer_mcp_stdio_config", spy)
    return captured


def test_otf_encode_build_shared_slayer_server_keeps_ingest_on_startup(
    tmp_path, monkeypatch,
):
    """pydantic_ai_otf_encode operates against an ``ensure_db_reference``
    artifact (NOT ``ensure_db_cache``). The reference's reuse path is
    presence-gated on ``_reference_fp.txt`` and does NOT carry the
    impl-fingerprint guard added in DEV-1508 (only ``ensure_db_cache``
    does). Until the impl-fp split is extended to ``ensure_db_reference``,
    this adapter must keep ``--ingest-on-startup`` as the safety net
    against a reference built under an older slayer / embedding model
    (e.g. a downloaded artifact from cloud combo 3 — Codex review of
    PR #11)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as m

    captured = _spy_slayer_mcp_stdio_config(monkeypatch, m)
    m._build_shared_slayer_server(str(tmp_path))
    # Either default (no kwarg) or explicit True is acceptable; pin the
    # observable behavior of the helper receiving True.
    assert captured["kw"].get("ingest_on_startup", True) is True


def test_recursive_build_shared_slayer_server_disables_for_on_the_fly(
    tmp_path, monkeypatch,
):
    """pydantic_ai_recursive supports BOTH pre-encoded (committed reference)
    and on-the-fly (OTF cache). When called with ``ingest_on_startup=False``
    (the on-the-fly path), the kwarg must be forwarded to the helper."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as m

    captured = _spy_slayer_mcp_stdio_config(monkeypatch, m)
    m._build_shared_slayer_server(str(tmp_path), ingest_on_startup=False)
    assert captured["kw"].get("ingest_on_startup") is False


def test_recursive_build_shared_slayer_server_default_keeps_ingest_on(
    tmp_path, monkeypatch,
):
    """When called without the kwarg (the pre-encoded / committed-reference
    path), the default MUST stay ``True`` so a committed reference still
    refreshes embeddings against the live embedding model — that's the
    safety net for the non-OTF callers."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as m

    captured = _spy_slayer_mcp_stdio_config(monkeypatch, m)
    m._build_shared_slayer_server(str(tmp_path))
    # Either an explicit True OR the absence of the kwarg is acceptable; we
    # pin the OBSERVABLE behavior — slayer_mcp_stdio_config sees True (or
    # nothing, which means True by default).
    val = captured["kw"].get("ingest_on_startup", True)
    assert val is True


def test_recursive_run_task_call_site_gates_on_slayer_setup():
    """The recursive adapter's ``run_task`` MUST pass
    ``ingest_on_startup=(self.slayer_setup != "on-the-fly")`` to the helper
    — otherwise the helper's new kwarg is useless on this adapter.

    Driving the full ``run_task`` is heavy (clarifier / resolver / constructor
    stubs + an asyncio context manager around the noop server); instead, parse
    the source and assert the literal call pattern. Brittle to a refactor but
    cheap and unambiguous, and it pins the exact wiring decision that the
    field-bug fix depends on.

    Scope: ONLY ``run_task`` bodies are inspected (not anywhere in the module),
    and the value expression MUST encode polarity ``slayer_setup != "on-the-fly"``
    — a bare ``slayer_setup`` reference is not enough (Codex review #2 of
    the tests)."""
    import ast

    from bird_interact_agents.agents.pydantic_ai_recursive import agent as m

    src = ast.parse(__import__("inspect").getsource(m))
    # Locate the ``run_task`` async function definitions (there may be one on
    # the agent class) and inspect ONLY their call bodies, so an unrelated
    # call elsewhere in the module can't satisfy the gate.
    run_task_bodies = [
        node for node in ast.walk(src)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_task"
    ]
    assert run_task_bodies, "recursive adapter must define a run_task"

    found = False
    for fn in run_task_bodies:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else "")
            )
            if name != "_build_shared_slayer_server":
                continue
            for kw in node.keywords:
                if kw.arg != "ingest_on_startup":
                    continue
                text = ast.unparse(kw.value)
                # Polarity check: the expression must explicitly negate the
                # on-the-fly case. Without this, e.g. an ``ingest_on_startup=
                # (self.slayer_setup == "on-the-fly")`` typo would silently
                # pass the previous-weaker version of this test.
                if "slayer_setup" in text and "on-the-fly" in text and "!=" in text:
                    found = True
                    break
            if found:
                break
    assert found, (
        "run_task must pass ingest_on_startup=(self.slayer_setup != \"on-the-fly\") "
        "to _build_shared_slayer_server; the kwarg was added but never wired with "
        "the correct polarity"
    )
