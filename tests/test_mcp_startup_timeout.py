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
