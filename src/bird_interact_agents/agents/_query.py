"""In-process ``query`` / ``submit_query`` wrappers over SLayer's own MCP tools.

SLayer 0.10.2 folded ``query_nested`` into a single ``query`` argument that is
one query object, a multi-stage list, or a run-by-name string. ``query_impl``
takes that value and forwards it to SLayer's ``query`` fn unchanged in shape,
applying only our extras: deterministic Mode-B filter normalisation (governed
by ``normalize_filters``, never forwarded to SLayer) and lifting a tool-level
option the model misplaced inside a single query object (``show_sql`` /
``dry_run`` / ``explain`` / ``format`` / ``normalize_filters``) — for a
multi-stage list such a key is stripped from each stage, never lifted.

The SLayer ``query`` fn is extracted once per task from
``create_mcp_server(storage)``; task setup calls :func:`attach_storage` to wire
storage in. Tests monkeypatch :func:`_get_slayer_query_fn` or
``slayer.mcp.server.create_mcp_server`` to swap it.

Out of scope: pydantic_ai_* adapters (their ``normalize_tool_filters`` call
stays as-is).
"""
from __future__ import annotations

import contextvars
import copy
import logging
from typing import Any, Optional

from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_filters_list,
    normalize_query_payload,
)

logger = logging.getLogger(__name__)

# DEV-1577: tool-level passthrough kwargs the agent sometimes misplaces
# INSIDE a single query object (e.g. ``{"source_model": ..., "show_sql": true}``)
# instead of as separate wrapper kwargs. Rather than rejecting the whole
# call over a misplacement SlayerQuery itself doesn't care about, we lift
# these out into their wrapper kwargs (``query_impl``) or strip them before
# SlayerQuery validation (``submit_slayer_query``).
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
    ``format`` / ``normalize_filters`` inside the query object instead of
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
    query: Any,
    *,
    variables: Optional[dict] = None,
    show_sql: bool = False,
    dry_run: bool = False,
    explain: bool = False,
    format: str = "markdown",  # noqa: A002 — matches SLayer's `query` signature
    normalize_filters: bool = True,
) -> str:
    """Run a SLayer query and return SLayer's formatted result.

    ``query`` is SLayer 0.10.2's single ``query`` argument: a query object, a
    multi-stage list of stage objects, or a run-by-name model-name string,
    forwarded to SLayer's ``query`` fn unchanged in shape.

    A single query object has its Mode-B ``filters`` normalised (unless
    ``normalize_filters`` is false) and any misplaced tool-level option
    (``show_sql`` / ``dry_run`` / ``explain`` / ``format`` /
    ``normalize_filters``) lifted out into the matching argument. In a
    multi-stage list each stage's ``filters`` is normalised and a misplaced
    tool-level key is stripped, never lifted. A run-by-name string is
    forwarded verbatim. ``normalize_filters`` is NOT forwarded to SLayer.
    """
    if isinstance(query, dict):
        parsed = copy.deepcopy(query)
        lifted = _lift_tool_level_keys(parsed)
        show_sql = bool(lifted.get("show_sql", show_sql))
        dry_run = bool(lifted.get("dry_run", dry_run))
        explain = bool(lifted.get("explain", explain))
        format = lifted.get("format", format)  # noqa: A001 — SLayer's `query` sig
        normalize_filters = bool(lifted.get("normalize_filters", normalize_filters))
        query = normalize_query_payload(parsed, normalize=normalize_filters)
    elif isinstance(query, list):
        stripped = _strip_tool_level_keys(copy.deepcopy(query))
        query = normalize_query_payload(stripped, normalize=normalize_filters)
    elif isinstance(query, str):
        pass  # run-by-name: forward verbatim
    else:
        raise ValueError(
            "`query` must be a SlayerQuery object, a list of stage objects, "
            f"or a model-name string; got {type(query).__name__}."
        )

    fn = _get_slayer_query_fn()
    return await fn(
        query=query,
        variables=variables,
        show_sql=show_sql,
        dry_run=dry_run,
        explain=explain,
        format=format,
    )
