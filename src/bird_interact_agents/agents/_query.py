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

import contextvars
import copy
import json
import logging
from typing import Any, Optional

from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_filters_list,
)

logger = logging.getLogger(__name__)

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

# DEV-1577: tool-level passthrough kwargs the agent sometimes misplaces
# INSIDE ``query_json`` (e.g. ``{"source_model": ..., "show_sql": true}``)
# instead of as separate wrapper kwargs. Rather than rejecting the whole
# call over a misplacement SlayerQuery itself doesn't care about, we lift
# these out of the JSON into their wrapper kwargs (``query_impl``) or strip
# them before SlayerQuery validation (``submit_slayer_query``). Genuinely
# unknown SlayerQuery fields and unsupported MCP-query fields still raise.
_TOOL_LEVEL_JSON_KEYS: frozenset[str] = frozenset({
    "show_sql",
    "dry_run",
    "explain",
    "format",
    "normalize_filters",
})


def _lift_tool_level_keys(parsed: dict) -> dict:
    """Pop any misplaced tool-level passthrough keys out of ``parsed`` and
    return ``{key: value}`` for the ones that were present.

    Mutates ``parsed`` in place. DEV-1577: keeps a ``query`` call from being
    rejected when the agent nested ``show_sql`` / ``dry_run`` / ``explain`` /
    ``format`` / ``normalize_filters`` inside ``query_json`` instead of
    passing them as wrapper kwargs.
    """
    return {k: parsed.pop(k) for k in list(parsed) if k in _TOOL_LEVEL_JSON_KEYS}


def _strip_tool_level_keys(parsed: Any) -> Any:
    """Drop misplaced tool-level passthrough keys from a parsed submit
    payload so ``SlayerQuery``'s ``extra="forbid"`` doesn't reject the whole
    submission over a misplacement.

    Handles both shapes ``submit_query`` accepts: a single-stage dict and a
    nested-DAG list of stage dicts (each stage stripped independently).
    Mutates in place and returns ``parsed``. Genuinely unknown SlayerQuery
    fields are left untouched so SlayerQuery still rejects them.
    """
    if isinstance(parsed, dict):
        for k in _TOOL_LEVEL_JSON_KEYS:
            parsed.pop(k, None)
    elif isinstance(parsed, list):
        for stage in parsed:
            if isinstance(stage, dict):
                for k in _TOOL_LEVEL_JSON_KEYS:
                    stage.pop(k, None)
    return parsed

# DEV-1581 (Codex R2 finding #2): the SLayer storage handle + the
# FastMCP-extracted tool-fn caches MUST be TASK-LOCAL, not module
# globals. Under the R2 two-persistent-clients design (main + discovery
# in one process) and ``run.py``'s ``asyncio.gather`` over multiple
# ``run_task``s, concurrent tasks with DIFFERENT ``slayer_storage_dir``
# would otherwise interleave and clobber a shared module global: task A
# attaches storage A, yields inside SLayer query code, task B attaches
# storage B, and A's next tool lookup silently resolves against B.
#
# The state lives in a ``ContextVar``. ``attach_storage`` REBINDS the var
# to a fresh state object (never mutates the existing one in place) so an
# ``asyncio.Task`` that inherited the parent context gets its own binding
# the moment it attaches — sibling tasks never observe each other's
# storage. The cache (``query_fn`` + ``tool_fns``) hangs off that
# per-task state object, so it is task-local for free.


class _TaskQueryState:
    """Per-task SLayer storage handle + the ONE MCP server built from it +
    lazily-extracted tool-fn cache.

    DEV-1654: the ``mcp`` server (and hence the single ``SlayerQueryEngine`` /
    asyncpg pool it owns) is cached per task context so every tool fn is
    extracted from the SAME server — see :func:`ensure_task_server`. Before
    this, ``_get_slayer_tool_fn`` built a fresh server per distinct tool NAME,
    leaking one connection pool per name.
    """

    __slots__ = ("storage", "mcp", "query_fn", "tool_fns")

    def __init__(self, storage: Any = None) -> None:
        self.storage = storage
        self.mcp: Any = None
        self.query_fn: Any = None
        self.tool_fns: dict[str, Any] = {}


_query_state: contextvars.ContextVar[_TaskQueryState] = contextvars.ContextVar(
    "bird_interact_slayer_query_state",
)


def _state() -> _TaskQueryState:
    """Return the current task's query state, lazily creating + binding an
    empty one if this context has never attached storage."""
    st = _query_state.get(None)
    if st is None:
        st = _TaskQueryState()
        _query_state.set(st)
    return st


def attach_storage(storage: Any) -> None:
    """Wire the SLayer storage handle for the CURRENT task (contextvar).

    Rebinds the context var to a FRESH state object whenever the storage
    identity changes (or ``storage is None`` — a reset), so concurrent
    ``asyncio.Task``s that inherited the parent context each get their own
    binding and never clobber one another. Calling this repeatedly with
    the SAME storage object is a no-op that preserves the cache (the
    claude_sdk handler attaches per query call; without this guard
    ``create_mcp_server`` would re-run every time — flagged by CodeRabbit
    on PR #38).
    """
    if storage is None:
        # ``None`` is an explicit reset: always rebind to a clean state so
        # no stale cache survives (a plain identity check would early-return
        # when the prior storage was also ``None`` and keep its cache).
        _query_state.set(_TaskQueryState())
        return
    st = _query_state.get(None)
    if st is not None and st.storage is storage:
        return
    _query_state.set(_TaskQueryState(storage))


def ensure_task_server() -> Any:
    """Return the CURRENT task's single SLayer MCP server, building it once
    (``create_mcp_server(storage)``) and caching it on the ``_TaskQueryState``.

    DEV-1654: every ``_get_slayer_tool_fn`` name resolves against THIS one
    server, so exactly one ``SlayerQueryEngine`` (one asyncpg pool) is created
    per task context — instead of one per distinct tool name. The server's
    engine is disposed at task teardown via :func:`dispose_slayer_engine`
    (SLayer 0.9.5 exposes it as ``mcp._slayer_engine``).

    Tests monkeypatch ``slayer.mcp.server.create_mcp_server`` to swap in a fake.
    """
    st = _state()
    if st.mcp is None:
        from slayer.mcp.server import create_mcp_server

        st.mcp = create_mcp_server(st.storage)
    return st.mcp


def _get_slayer_tool_fn(name: str):
    """Extract a SLayer MCP tool function ``.fn`` from the task's ONE server
    (:func:`ensure_task_server`), cached per-task per-name on the current
    ``_TaskQueryState``.
    """
    st = _state()
    if name == "query":
        if st.query_fn is not None:
            return st.query_fn
    else:
        cached = st.tool_fns.get(name)
        if cached is not None:
            return cached

    mcp = ensure_task_server()
    fn = mcp._tool_manager._tools[name].fn
    if name == "query":
        st.query_fn = fn
    else:
        st.tool_fns[name] = fn
    return fn


async def dispose_slayer_engine(mcp: Any) -> None:
    """Best-effort dispose of the asyncpg pool owned by ``mcp``'s
    ``SlayerQueryEngine`` (SLayer 0.9.5 ``mcp._slayer_engine``).

    Awaited at task teardown on the task's own event loop (the asyncpg pool is
    loop-bound). A ``None`` handle, an older SLayer server without
    ``_slayer_engine``, or an engine without ``aclose`` are all silent no-ops;
    a failing ``aclose`` is swallowed — disposal must never break the task.
    """
    if mcp is None:
        return
    engine = getattr(mcp, "_slayer_engine", None)
    if engine is None:
        return
    aclose = getattr(engine, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:  # noqa: BLE001 — teardown must never raise into the task
        logger.warning("dispose_slayer_engine: aclose failed", exc_info=True)


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
      ``version``) → ``ValueError`` naming the key and pointing at the
      supported alternatives.

    DEV-1577: a tool-level passthrough kwarg misplaced INSIDE
    ``query_json`` (``show_sql`` / ``dry_run`` / ``explain`` / ``format``
    / ``normalize_filters``) is NO LONGER rejected — it is lifted out into
    the corresponding wrapper kwarg (the JSON-nested value wins over the
    kwarg default) so the agent doesn't burn a retry on a misplacement
    SlayerQuery would coerce around anyway. String ``measures`` /
    ``dimensions`` shorthands (``["count"]`` → ``[{"formula": "count"}]``)
    ride through to SlayerQuery's own coercion unchanged.

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

    # DEV-1577: tolerate tool-level passthrough kwargs the agent misplaced
    # INSIDE the JSON — lift them out into their wrapper kwargs (JSON-nested
    # value wins over the kwarg default) instead of rejecting the whole call.
    lifted = _lift_tool_level_keys(parsed)
    show_sql = lifted.get("show_sql", show_sql)
    dry_run = lifted.get("dry_run", dry_run)
    explain = lifted.get("explain", explain)
    format = lifted.get("format", format)  # noqa: A001 — matches slayer's `query` signature
    normalize_filters = lifted.get("normalize_filters", normalize_filters)

    unknown = set(parsed) - _QUERY_JSON_ALLOWLIST
    if unknown:
        sorted_unknown = sorted(unknown)
        supported = ", ".join(sorted(_QUERY_JSON_ALLOWLIST))
        raise ValueError(
            f"`query_json` contains unsupported top-level field(s) "
            f"{sorted_unknown!r}. The `query` preview tool accepts only "
            f"these SlayerQuery fields: {supported}. Tool-level options "
            "(`show_sql`, `dry_run`, `explain`, `format`, "
            "`normalize_filters`) are separate wrapper kwargs — if you "
            "place them inside the JSON they are lifted out automatically. "
            "For SlayerQuery fields outside this set (e.g. "
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
