"""DEV-1534 Fix C / DEV-1546: `mcp__bird-interact-tools__query` +
`query_nested` wrappers.

The claude_sdk OTF agents previously allowed ``mcp__slayer__query`` /
``mcp__slayer__query_nested`` directly — SLayer's own MCP tools, served
by the slayer subprocess MCP server. The subprocess server does NOT
apply our Mode-B filter normalization (only the submit path does), so
the agent could not preview the ``LOWER(TRIM(...))``-wrapped SQL it
would actually submit against, and could not opt out at preview time
either.

These wrappers replace ``mcp__slayer__query`` / ``mcp__slayer__query_nested``
in the OTF allowlist.

DEV-1546: ``query_impl`` accepts a SINGLE ``query_json: str`` arg — the
same ``SlayerQuery`` JSON DSL ``submit_query`` accepts — so the agent
uses ONE form across ``query`` / ``query_nested`` / ``submit_query``.
Tool-level options (``show_sql`` / ``dry_run`` / ``explain`` / ``format``)
plus our ``normalize_filters`` directive stay outside the JSON.

Both wrappers:

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
import json
from typing import Any, Optional

from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_filters_list,
)

# DEV-1546: only these SlayerQuery DSL fields are accepted inside
# ``query_json`` for the single-stage preview wrapper — exactly the
# kwargs slayer's MCP ``query`` natively exposes. Tool-level options
# (``show_sql`` / ``dry_run`` / ``explain`` / ``format``) and our
# ``normalize_filters`` directive stay outside the JSON, as wrapper
# kwargs. Any other key (``main_time_dimension``, ``name``, ``version``,
# or a misplaced tool-level kwarg) is rejected with a sharp error.
_QUERY_JSON_ALLOWLIST: frozenset[str] = frozenset({
    "source_model",
    "measures",
    "dimensions",
    "filters",
    "time_dimensions",
    "order",
    "limit",
    "offset",
    "whole_periods_only",
    "variables",
    "distinct_dimension_values",
})

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
    query_json: str,
    *,
    show_sql: bool = False,
    dry_run: bool = False,
    explain: bool = False,
    format: str = "markdown",  # noqa: A002 — matches SLayer's `query` signature
    normalize_filters: bool = True,
) -> str:
    """SLayer MCP ``query``-compatible preview wrapper.

    DEV-1546: takes a single ``query_json`` SlayerQuery DSL string
    (the same shape ``submit_query`` accepts) plus the four tool-level
    options slayer's MCP ``query`` natively exposes (``show_sql``,
    ``dry_run``, ``explain``, ``format``) and our ``normalize_filters``
    directive. Inside ``query_json`` the agent puts the
    ``distinct_dimension_values`` field (and every other SlayerQuery
    DSL field) — the wrapper unpacks it through ``_QUERY_JSON_ALLOWLIST``
    to slayer's ``query.fn`` kwargs.

    Error paths:

    * ``json.JSONDecodeError`` → re-raised as ``ValueError`` naming the
      ``query_json`` arg.
    * Top-level list (nested-DAG shape) → ``ValueError`` pointing the
      agent at ``query_nested``.
    * Missing ``source_model`` → ``ValueError``.
    * Top-level key outside ``_QUERY_JSON_ALLOWLIST`` (unsupported
      SlayerQuery field like ``main_time_dimension`` / ``name`` /
      ``version``, or a tool-level kwarg leaking into the JSON) →
      ``ValueError`` naming the key and pointing at the supported
      alternatives.

    ``normalize_filters`` is OUR directive and is NOT forwarded to
    SLayer.
    """
    try:
        parsed = json.loads(query_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"`query_json` could not be parsed as JSON: {e}. "
            "Pass a SlayerQuery JSON object (the same shape "
            "`submit_query` accepts)."
        ) from e

    if isinstance(parsed, list):
        raise ValueError(
            "`query_json` is a JSON array (nested-DAG shape). The "
            "`query` preview tool is single-stage only — use the "
            "`query_nested` tool for a nested-DAG preview, or "
            "`submit_query` (which accepts both shapes)."
        )
    if not isinstance(parsed, dict):
        raise ValueError(
            "`query_json` must be a SlayerQuery JSON OBJECT; got "
            f"{type(parsed).__name__}."
        )
    if "source_model" not in parsed:
        raise ValueError(
            "`query_json` is missing the required `source_model` field. "
            "Every SlayerQuery names a model — see `models_summary`."
        )

    unknown = set(parsed) - _QUERY_JSON_ALLOWLIST
    if unknown:
        sorted_unknown = sorted(unknown)
        supported = ", ".join(sorted(_QUERY_JSON_ALLOWLIST))
        raise ValueError(
            f"`query_json` contains unsupported top-level field(s) "
            f"{sorted_unknown!r}. The `query` preview tool accepts only "
            f"these SlayerQuery fields: {supported}. Tool-level options "
            "(`show_sql`, `dry_run`, `explain`, `format`, "
            "`normalize_filters`) are separate wrapper kwargs, NOT JSON "
            "fields. For SlayerQuery fields outside this set (e.g. "
            "`main_time_dimension`, `name`, `version`) use "
            "`query_nested` or `submit_query`."
        )

    parsed = copy.deepcopy(parsed)
    if "filters" in parsed:
        parsed["filters"] = normalize_filters_list(
            parsed.get("filters"), normalize=normalize_filters,
        )

    fn = _get_slayer_query_fn()
    return await fn(
        source_model=parsed["source_model"],
        measures=parsed.get("measures"),
        dimensions=parsed.get("dimensions"),
        filters=parsed.get("filters"),
        time_dimensions=parsed.get("time_dimensions"),
        order=parsed.get("order"),
        limit=parsed.get("limit"),
        offset=parsed.get("offset"),
        whole_periods_only=parsed.get("whole_periods_only", False),
        show_sql=show_sql,
        dry_run=dry_run,
        explain=explain,
        format=format,
        variables=parsed.get("variables"),
        distinct_dimension_values=parsed.get("distinct_dimension_values", True),
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
