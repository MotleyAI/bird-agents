"""DEV-1534 Fix C: `mcp__bird-interact-tools__query` + `query_nested` wrappers.

The claude_sdk OTF agents previously allowed ``mcp__slayer__query`` /
``mcp__slayer__query_nested`` directly — SLayer's own MCP tools, served
by the slayer subprocess MCP server. The subprocess server does NOT
apply our Mode-B filter normalization (only the submit path does), so
the agent could not preview the ``LOWER(TRIM(...))``-wrapped SQL it
would actually submit against, and could not opt out at preview time
either.

These wrappers replace ``mcp__slayer__query`` / ``mcp__slayer__query_nested``
in the OTF allowlist. Each accepts SLayer's MCP parameter list PLUS
``normalize_filters: bool = True``. The wrappers:

* Pre-process the ``filters`` arg via
  :func:`bird_interact_agents.slayer_pipeline.filter_normalization.normalize_filters_list`
  (governed by ``normalize_filters``). For ``query_nested`` each stage's
  ``filters`` is processed independently.
* Forward every other parameter verbatim to SLayer's MCP function
  (dry-run / explain / show_sql / format / friendly-DB-error branches
  are SLayer's responsibility).
* Do NOT forward ``normalize_filters`` (it's our directive).
* Return SLayer's output string byte-for-byte.

The SLayer functions are extracted ONCE per task via
``create_mcp_server(storage)._tool_manager._tools[<name>].fn``. The
agent's task setup calls :func:`attach_storage` to wire the storage in;
tests can monkeypatch :func:`_get_slayer_tool_fn` or
:func:`slayer.mcp.server.create_mcp_server` to swap the functions.

Out of scope: pydantic_ai_* adapters (their existing
``normalize_tool_filters`` call stays as-is).
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_filters_list,
)

# Task-local storage handle. ``attach_storage`` is called by the agent
# at task startup; the FastMCP-extracted ``query``.fn is cached lazily
# in ``_cached_slayer_query_fn`` (kept as a top-level for direct test
# monkeypatching), other tools in ``_cached_slayer_tool_fns``.
_slayer_storage: Any = None
_cached_slayer_query_fn: Any = None
_cached_slayer_tool_fns: dict[str, Any] = {}


def attach_storage(storage: Any) -> None:
    """Wire the SLayer storage handle for the current task.

    Invalidates the cached tool functions ONLY when the storage object
    actually changes — calling this per-tool-invocation with the same
    storage is a no-op. (The claude_sdk handler attaches per-call so the
    wrapper survives storage swaps between tasks; without this guard
    the cache never hits and ``create_mcp_server`` re-runs on every
    query call. Flagged by CodeRabbit on PR #38.)
    """
    global _slayer_storage, _cached_slayer_query_fn
    if storage is _slayer_storage:
        return
    _slayer_storage = storage
    _cached_slayer_query_fn = None
    _cached_slayer_tool_fns.clear()


def _get_slayer_tool_fn(name: str):
    """Extract a SLayer MCP tool function via
    ``create_mcp_server(storage)._tool_manager._tools[name].fn``,
    cached per-task per-name. Tests monkeypatch
    ``slayer.mcp.server.create_mcp_server`` to swap in a fake.

    The ``query`` lookup keeps a module-level alias
    (``_cached_slayer_query_fn``) for back-compat with tests that
    monkeypatch that name directly.
    """
    global _cached_slayer_query_fn
    if name == "query":
        if _cached_slayer_query_fn is not None:
            return _cached_slayer_query_fn
    else:
        cached = _cached_slayer_tool_fns.get(name)
        if cached is not None:
            return cached
    from slayer.mcp.server import create_mcp_server

    mcp = create_mcp_server(_slayer_storage)
    fn = mcp._tool_manager._tools[name].fn
    if name == "query":
        _cached_slayer_query_fn = fn
    else:
        _cached_slayer_tool_fns[name] = fn
    return fn


def _get_slayer_query_fn():
    """Back-compat alias for :func:`_get_slayer_tool_fn` (``query``).

    Existing tests patch this name; keep it stable.
    """
    return _get_slayer_tool_fn("query")


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


def _processed_nested_stages(queries: Any, *, normalize: bool) -> Any:
    """Deep-copy ``queries`` and (optionally) normalize each stage's
    ``filters`` list. Returns non-list inputs (None, dict, etc.)
    unchanged so SLayer can produce its own validation error.

    Each stage dict is ``copy.deepcopy``-ed (NOT ``dict(stage)``) so
    nested mutables — ``dimensions`` / ``measures`` / ``time_dimensions``
    / ``order`` / ``variables`` / nested objects inside ``filters`` —
    are not aliased to the input. Matches the deep-copy contract that
    the sibling ``normalize_query_payload`` already honours for the
    submit-side path.
    """
    if not isinstance(queries, list):
        return queries
    out: list = []
    for stage in queries:
        if isinstance(stage, dict):
            stage_copy = copy.deepcopy(stage)
            if "filters" in stage_copy:
                stage_copy["filters"] = normalize_filters_list(
                    stage_copy.get("filters"), normalize=normalize,
                )
            out.append(stage_copy)
        else:
            out.append(stage)
    return out


async def query_nested_impl(
    queries: list,
    variables: Optional[dict] = None,
    show_sql: bool = False,
    dry_run: bool = False,
    explain: bool = False,
    format: str = "markdown",  # noqa: A002 — matches SLayer's `query_nested` sig
    normalize_filters: bool = True,
) -> str:
    """SLayer MCP ``query_nested``-compatible wrapper with an extra
    ``normalize_filters`` parameter.

    The first 6 parameters mirror SLayer's MCP ``query_nested`` exactly
    (order and defaults). ``normalize_filters`` is OUR directive and is
    NOT forwarded to SLayer; it pre-processes each stage's ``filters``
    list via :func:`normalize_filters_list`.
    """
    processed_queries = _processed_nested_stages(
        queries, normalize=normalize_filters,
    )
    fn = _get_slayer_tool_fn("query_nested")
    return await fn(
        queries=processed_queries,
        variables=variables,
        show_sql=show_sql,
        dry_run=dry_run,
        explain=explain,
        format=format,
    )
