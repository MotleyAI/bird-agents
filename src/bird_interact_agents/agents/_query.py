"""DEV-1534 Fix C: `mcp__bird-interact-tools__query` wrapper.

claude_sdk in slayer mode previously allowed ``mcp__slayer__query``
directly — SLayer's own MCP query tool, served by the slayer subprocess
MCP server. The subprocess server does NOT apply our Mode-B filter
normalization (only the submit path does), so the agent could not
preview the ``LOWER(TRIM(...))``-wrapped SQL it would actually submit
against, and could not opt out at preview time either.

This wrapper replaces ``mcp__slayer__query`` in claude_sdk's allowlist.
It accepts the exact 14 parameters SLayer's MCP ``query`` takes PLUS
``normalize_filters: bool = True``. The wrapper:

* Pre-processes the ``filters`` arg via
  :func:`bird_interact_agents.slayer_pipeline.filter_normalization.normalize_filters_list`
  (governed by ``normalize_filters``).
* Forwards every other parameter verbatim to SLayer's MCP ``query``
  function (we don't reformat dry-run / explain / show_sql / format /
  friendly-DB-error branches — SLayer handles them).
* Does NOT forward ``normalize_filters`` (it's our directive).
* Returns SLayer's output string byte-for-byte.

The SLayer ``query`` function is extracted ONCE per task via
``create_mcp_server(storage)._tool_manager._tools["query"].fn``. The
agent's task setup calls :func:`attach_storage` to wire the storage
in; tests can monkeypatch :func:`_get_slayer_query_fn` directly or
:func:`slayer.mcp.server.create_mcp_server` to swap the function for
a stub.

Out of scope: pydantic_ai_* adapters (their existing
``normalize_tool_filters`` call stays as-is).
"""
from __future__ import annotations

from typing import Any, Optional

from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_filters_list,
)

# Task-local storage handle. ``attach_storage`` is called by the agent
# at task startup; the FastMCP-extracted ``query.fn`` is cached lazily
# in ``_cached_slayer_query_fn``.
_slayer_storage: Any = None
_cached_slayer_query_fn: Any = None


def attach_storage(storage: Any) -> None:
    """Wire the SLayer storage handle for the current task. Invalidates
    the cached ``query.fn`` so subsequent calls re-extract against the
    new storage."""
    global _slayer_storage, _cached_slayer_query_fn
    _slayer_storage = storage
    _cached_slayer_query_fn = None


def _get_slayer_query_fn():
    """Extract SLayer's MCP ``query`` function via
    ``create_mcp_server(storage)._tool_manager._tools["query"].fn``,
    cached per-task. Tests monkeypatch
    ``slayer.mcp.server.create_mcp_server`` to swap in a fake."""
    global _cached_slayer_query_fn
    if _cached_slayer_query_fn is not None:
        return _cached_slayer_query_fn
    from slayer.mcp.server import create_mcp_server

    mcp = create_mcp_server(_slayer_storage)
    _cached_slayer_query_fn = mcp._tool_manager._tools["query"].fn
    return _cached_slayer_query_fn


async def query_impl(
    source_model,
    measures: Optional[list] = None,
    dimensions: Optional[list] = None,
    filters: Optional[list] = None,
    time_dimensions: Optional[list] = None,
    order: Optional[list] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    whole_periods_only: bool = False,
    show_sql: bool = False,
    dry_run: bool = False,
    explain: bool = False,
    format: str = "markdown",  # noqa: A002 — matches SLayer's `query` signature
    variables: Optional[dict] = None,
    normalize_filters: bool = True,
) -> str:
    """SLayer MCP ``query``-compatible wrapper with a 15th parameter:
    ``normalize_filters``.

    The first 14 parameters mirror SLayer's MCP ``query`` exactly (order
    and defaults) so the agent's mid-flight tool call is byte-compatible
    except for the new flag. ``normalize_filters`` is OUR directive and
    is NOT forwarded to SLayer.
    """
    processed_filters = normalize_filters_list(
        filters, normalize=normalize_filters,
    )
    fn = _get_slayer_query_fn()
    return await fn(
        source_model=source_model,
        measures=measures,
        dimensions=dimensions,
        filters=processed_filters,
        time_dimensions=time_dimensions,
        order=order,
        limit=limit,
        offset=offset,
        whole_periods_only=whole_periods_only,
        show_sql=show_sql,
        dry_run=dry_run,
        explain=explain,
        format=format,
        variables=variables,
    )
