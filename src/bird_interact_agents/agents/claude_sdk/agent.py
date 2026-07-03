"""Claude Agent SDK @tool definitions shared by the OTF agents.

This module is the SHARED HOME of the in-process tool functions every
``claude_sdk*`` OTF agent registers on its bird-interact-tools MCP
server. The OTF entrypoints (one_shot / a-interact, slayer / raw) own
their own run loops, prompts, and tool lists — they import the tools
they need from here:

* Raw-mode exploration: ``execute_sql``, ``get_schema``,
  ``get_all_column_meanings`` / ``get_column_meaning``,
  ``get_all_external_knowledge_names`` / ``get_knowledge_definition`` /
  ``get_all_knowledge_definitions``.
* Shared submission: ``ask_user``, ``submit_sql``, ``submit_query``.
* DEV-1534 Fix C SLayer-mode wrappers: ``query`` / ``query_nested``
  (filter-normalization opt-out aware; replace SLayer's MCP
  ``query`` / ``query_nested`` in the OTF agents' allowlist).

The contextvar plumbing (``_ctx_var``, ``_CtxProxy``,
``accumulate_assistant_usage``, ``_state_view``, ``_gate``,
``_run_env``) is also defined here and reused verbatim by every OTF
agent.
"""

import contextvars
import functools
import inspect
import logging
from types import SimpleNamespace

from claude_agent_sdk import create_sdk_mcp_server, tool

from bird_interact_agents.agents import _query as _query_mod
from bird_interact_agents.agents._submit import (
    ask_user_impl,
    submit_raw_sql,
    submit_slayer_query,
)
from bird_interact_agents.agents._tool_specs import (
    BIRD_INTERACT_TOOLS,
    render_action,
)
from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_tool_filters,
)
from bird_interact_agents.harness import (
    ACTION_COSTS,
    SampleStatus,
    execute_env_action,
    update_budget,
)
from bird_interact_agents.usage import TokenUsage


_BY_NAME = {t.name: t for t in BIRD_INTERACT_TOOLS}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-task context — uses contextvars so concurrent task runs don't collide.
# Each invocation of run_task() sets the var, and tools read from it.
# `_ctx` is exposed as a dict-like proxy for backward compat with tests.
# ---------------------------------------------------------------------------
_ctx_var: contextvars.ContextVar[dict] = contextvars.ContextVar("_ctx_var")


class _CtxProxy:
    """Dict-like proxy that reads/writes the current contextvar value.

    Tests do `agent_mod._ctx = {...}` and `agent_mod._ctx["key"]` — both
    work via this proxy by setting/reading the contextvar.
    """

    def __getitem__(self, key):
        return _ctx_var.get()[key]

    def __setitem__(self, key, value):
        _ctx_var.get()[key] = value

    def __contains__(self, key):
        try:
            return key in _ctx_var.get()
        except LookupError:
            return False

    def get(self, key, default=None):
        try:
            return _ctx_var.get().get(key, default)
        except LookupError:
            return default

    def update(self, *args, **kwargs):
        try:
            current = _ctx_var.get()
        except LookupError:
            current = {}
            _ctx_var.set(current)
        current.update(*args, **kwargs)


_ctx = _CtxProxy()


def _text(msg: str) -> dict:
    """Helper to build a tool return value."""
    return {"content": [{"type": "text", "text": str(msg)}]}


def _budget_note(status: SampleStatus) -> str:
    return (
        f"\n\n[Remaining budget: {status.remaining_budget:.1f}"
        f" / {status.total_budget:.1f}]"
    )


def _usage_value(usage: object, key: str) -> int:
    """Read a token field from a Claude Agent SDK usage object.

    The live SDK delivers `message.usage` as a **dict**
    (`{'input_tokens': …, 'output_tokens': …,
    'cache_read_input_tokens': …, 'cache_creation_input_tokens': …}`);
    mocked/older paths use an attribute object. Handle both — reading a
    dict with `getattr` silently returns 0, which is why agent token usage
    (and therefore agent cost) was previously recorded as zero.
    """
    if usage is None:
        return 0
    val = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return val or 0


def accumulate_assistant_usage(accum: TokenUsage, msg: object, model: str) -> None:
    """Legacy per-AssistantMessage accumulator (DEV-1555 follow-up: prefer
    :class:`SdkUsageTracker`).

    The SDK splits one assistant TURN into multiple ``AssistantMessage``
    events (one per content block: thinking, text, tool_use, …), each
    carrying the SAME per-turn ``usage`` dict — naïve summing both
    double-counts cache tokens and under-counts ``output_tokens`` (which
    only fills in on the final block, while prior blocks of the same turn
    report 0). The cumulative authoritative usage lives in the single
    terminal ``ResultMessage`` of the stream.

    Kept for backward compatibility / tests; new agent loops should
    instantiate one :class:`SdkUsageTracker` per task and call
    ``observe()`` in the receive loop, ``finalize()`` on stream end.
    """
    if type(msg).__name__ != "AssistantMessage":
        return
    usage = getattr(msg, "usage", None)
    if usage is None:
        return
    accum.add_call(
        scope="agent",
        model=model,
        prompt=_usage_value(usage, "input_tokens"),
        completion=_usage_value(usage, "output_tokens"),
        cache_read=_usage_value(usage, "cache_read_input_tokens"),
        cache_write=_usage_value(usage, "cache_creation_input_tokens"),
    )


# DEV-1555 (post-stage-2 diagnosis): the per-AssistantMessage accumulator
# both double-counts cache reads (every block of a turn carries the same
# usage dict) and under-counts ``output_tokens`` (which fills in only on
# the last block while we summed every block at the same per-turn total).
# Empirical: Opus alien_1 cache_read 2.21M reported vs 1.39M actual;
# output 3.6K vs 17.6K actual. Kimi alien_1 cache_read 0 vs 3.66M actual
# (Moonshot's per-turn AssistantMessage.usage is all-zero; only the
# terminal ResultMessage reports the cumulative).


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class SdkUsageTracker:
    """Per-task usage tracker for the claude-agent-sdk stream.

    The SDK emits one terminal ``ResultMessage`` whose ``usage`` is the
    cumulative session total — that is the source of truth. While the
    stream is running we also track per-turn ``AssistantMessage.usage``
    (deduplicated by ``message_id`` or usage-dict identity) as a fallback
    estimate for crash paths.

    Usage:
        tracker = SdkUsageTracker(accum, self.model)
        async for msg in client.receive_response():
            tracker.observe(msg)
            ...
        tracker.finalize()   # idempotent; also called by observe() on
                              # ResultMessage so the success path is a
                              # no-op here.
    """

    def __init__(self, accum: TokenUsage, model: str, *, scope: str = "agent"):
        # DEV-1589: ``scope`` is the per-(scope, model) breakdown bucket. Default
        # "agent" keeps every existing caller unchanged; the build-time
        # reference encoder reuses this tracker with scope="setup_encoder" so its
        # tokens stay summable separately (the ``_setup_usage.json`` contract).
        self._accum = accum
        self._model = model
        self._scope = scope
        self._turns: dict = {}
        self._turn_order: list = []
        self._result_usage = None
        self._committed = False
        # DEV-1616: the exact n_calls this tracker contributed to the
        # accum's (scope, model) breakdown row on finalize. For a stream
        # with a terminal ResultMessage that is ``max(1, turns)``; on the
        # crash path it is the summed per-turn count; 0 before finalize /
        # with no activity. The warm-discovery channel sums THIS across
        # asks so ``n_discovery_turns`` matches the breakdown contribution.
        self._committed_n_calls = 0

    @property
    def committed_n_calls(self) -> int:
        """n_calls this tracker committed to its breakdown row (0 until
        finalize)."""
        return self._committed_n_calls

    def observe(self, msg: object) -> None:
        name = type(msg).__name__
        if name == "AssistantMessage":
            usage = getattr(msg, "usage", None)
            if usage is None:
                return
            # The live SDK shares ONE usage dict instance across every
            # content block of a turn — id(usage) is a stable per-turn
            # key. Prefer message_id when the SDK supplies it (older
            # mock paths set it; live 0.1.69 does not on AssistantMessage).
            tid = getattr(msg, "message_id", None)
            if tid is None:
                tid = id(usage)
            if tid in self._turns:
                return
            self._turns[tid] = usage
            self._turn_order.append(tid)
        elif name == "ResultMessage":
            self._result_usage = getattr(msg, "usage", None)
            # Terminal stream message: commit immediately so the receive
            # loop's normal exit doesn't need a separate finalize() call.
            self.finalize()

    def finalize(self) -> None:
        if self._committed:
            return
        self._committed = True
        n_turns = len(self._turn_order)
        if self._result_usage is not None:
            self._commit(self._result_usage, n_calls=max(1, n_turns))
            return
        if not self._turn_order:
            return
        agg = {k: 0 for k in _USAGE_FIELDS}
        for tid in self._turn_order:
            u = self._turns[tid]
            for k in _USAGE_FIELDS:
                agg[k] += _usage_value(u, k)
        self._commit(agg, n_calls=n_turns)
        self._accum.partial = True

    def _commit(self, usage, *, n_calls: int) -> None:
        # DEV-1616: record the committed count so callers (the discovery
        # channel) can sum the exact breakdown contribution.
        self._committed_n_calls = n_calls
        self._accum.add_call(
            scope=self._scope,
            model=self._model,
            prompt=_usage_value(usage, "input_tokens"),
            completion=_usage_value(usage, "output_tokens"),
            cache_read=_usage_value(usage, "cache_read_input_tokens"),
            cache_write=_usage_value(usage, "cache_creation_input_tokens"),
        )
        # ``add_call`` set n_calls=1 — bump to the actual turn count so
        # callers can compare against the legacy per-AssistantMessage
        # counts and against breakdown rows for non-SDK frameworks.
        if n_calls > 1:
            row = self._accum._row_for(scope=self._scope, model=self._model)
            row.n_calls += n_calls - 1
            self._accum.n_calls += n_calls - 1


def _gate(action_name: str, status: SampleStatus) -> str | None:
    """Reject a non-submit tool call when budget would go below submit cost.

    Returns an error message to surface back to the agent, or None if OK.
    Mirrors the `force_submit` gating in the original mini_interact_agent
    and ADK before_tool_callback.
    """
    if action_name.startswith("submit_"):
        return None
    submit_tool = "submit_query" if _ctx.get("query_mode") == "slayer" else "submit_sql"
    submit_cost = ACTION_COSTS[submit_tool]
    cost = ACTION_COSTS.get(action_name, 0)
    if status.force_submit or status.remaining_budget < cost + submit_cost:
        return (
            f"Budget exhausted ({status.remaining_budget:.1f} remaining, "
            f"{action_name} costs {cost}). You MUST call {submit_tool} now "
            "with your best answer."
        )
    return None


async def _run_env(action_name: str, action_str: str) -> dict:
    """Shared body for raw exploration tools: gate → execute → bookkeep → annotate."""
    status: SampleStatus = _ctx["status"]
    err = _gate(action_name, status)
    if err is not None:
        return _text(err)
    observation, _ = execute_env_action(action_str, status, _ctx["data_path_base"])
    update_budget(status, action_name)
    return _text(str(observation) + _budget_note(status))


def _state_view() -> SimpleNamespace:
    """Return a SimpleNamespace adapter onto the current contextvar dict.

    The shared `_submit` helpers expect a state object exposing
    `status`/`data_path_base`/`user_sim_model`/`user_sim_prompt_version`/
    `usage`/`result`/`slayer_storage_dir` as attributes. claude_sdk
    keeps everything in a contextvar dict for legacy reasons; this thin
    view bridges the two.

    Note: only `result` is mutated by the helpers (via submit_*), and a
    SimpleNamespace assignment to `result` won't write back to the dict.
    Helpers that mutate state must use the explicit ctx-writing helpers
    below. Reads of `state.result` DO see the live ctx dict — the submit
    helpers' cross-phase observation-preservation logic relies on it
    (`prior = state.result or {}` → `state.result["phaseN_observation"] =
    prior.get(...)`); hard-coding to None silently blanked the other-phase
    observation on every multi-submit (DEV-1511 follow-up).
    """
    d = _ctx_var.get()
    return SimpleNamespace(
        status=d.get("status"),
        data_path_base=d.get("data_path_base"),
        user_sim_model=d.get("user_sim_model", "anthropic/claude-haiku-4-5-20251001"),
        # DEV-1613: agent model for the in-task N5 insufficient-task judge.
        agent_model=d.get("agent_model"),
        user_sim_prompt_version=d.get("user_sim_prompt_version", "v2"),
        slayer_storage_dir=d.get("slayer_storage_dir", ""),
        usage=d.get("usage") or TokenUsage(),
        result=d.get("result"),
    )


# ---------------------------------------------------------------------------
# Raw-mode tools — direct DB exploration + SQL execution
#
# Each wrapper looks up its action template from BIRD_INTERACT_TOOLS, renders
# it with the model-supplied args, then routes through `_run_env` which
# applies budget gating and bookkeeping.
# ---------------------------------------------------------------------------

@tool("execute_sql", _BY_NAME["execute_sql"].description, {"sql": str})
async def execute_sql(args: dict) -> dict:
    return await _run_env(
        "execute_sql", render_action(_BY_NAME["execute_sql"], sql=args["sql"]),
    )


@tool("get_schema", _BY_NAME["get_schema"].description, {})
async def get_schema(args: dict) -> dict:
    return await _run_env("get_schema", render_action(_BY_NAME["get_schema"]))


@tool(
    "get_all_column_meanings",
    _BY_NAME["get_all_column_meanings"].description,
    {},
)
async def get_all_column_meanings(args: dict) -> dict:
    return await _run_env(
        "get_all_column_meanings",
        render_action(_BY_NAME["get_all_column_meanings"]),
    )


@tool(
    "get_column_meaning",
    _BY_NAME["get_column_meaning"].description,
    {"table_name": str, "column_name": str},
)
async def get_column_meaning(args: dict) -> dict:
    return await _run_env(
        "get_column_meaning",
        render_action(
            _BY_NAME["get_column_meaning"],
            table_name=args["table_name"], column_name=args["column_name"],
        ),
    )


@tool(
    "get_all_external_knowledge_names",
    _BY_NAME["get_all_external_knowledge_names"].description,
    {},
)
async def get_all_external_knowledge_names(args: dict) -> dict:
    return await _run_env(
        "get_all_external_knowledge_names",
        render_action(_BY_NAME["get_all_external_knowledge_names"]),
    )


@tool(
    "get_knowledge_definition",
    _BY_NAME["get_knowledge_definition"].description,
    {"knowledge_name": str},
)
async def get_knowledge_definition(args: dict) -> dict:
    return await _run_env(
        "get_knowledge_definition",
        render_action(
            _BY_NAME["get_knowledge_definition"],
            knowledge_name=args["knowledge_name"],
        ),
    )


@tool(
    "get_all_knowledge_definitions",
    _BY_NAME["get_all_knowledge_definitions"].description,
    {},
)
async def get_all_knowledge_definitions(args: dict) -> dict:
    return await _run_env(
        "get_all_knowledge_definitions",
        render_action(_BY_NAME["get_all_knowledge_definitions"]),
    )


# ---------------------------------------------------------------------------
# Shared tools — user simulator + submission
# ---------------------------------------------------------------------------

async def _ask_user_impl(question: str) -> str:
    """Thin shim that delegates to the shared user-sim helper."""
    return await ask_user_impl(_state_view(), question)


@tool(
    "ask_user",
    "Ask the user a clarification question about their query",
    {"question": str},
)
async def ask_user(args: dict) -> dict:
    status: SampleStatus = _ctx["status"]
    err = _gate("ask_user", status)
    if err is not None:
        return _text(err)
    answer = await _ask_user_impl(args["question"])
    update_budget(status, "ask_user")
    _ctx["asks_used"] = _ctx.get("asks_used", 0) + 1

    suffix = _budget_note(status)
    # In c-interact, the budget IS the turn budget; surface remaining ask
    # rounds explicitly (matches ADK callbacks_cinteract.after_tool_callback).
    if _ctx.get("eval_mode") == "c-interact":
        max_asks = _ctx.get("max_asks", 0)
        remaining = max(0, max_asks - _ctx["asks_used"])
        suffix += (
            f"\n[Clarification turns remaining: {remaining}/{max_asks}]"
        )
    return _text(answer + suffix)


@tool(
    "submit_sql",
    "Submit your final SQL query for evaluation. Only submit when confident.",
    {"sql": str},
)
async def submit_sql(args: dict) -> dict:
    state = _state_view()
    observation = submit_raw_sql(state, args["sql"])
    # `state` is a SimpleNamespace view — `state.result` doesn't write back
    # to the contextvar dict, so persist it explicitly. Budget + budget-note
    # are owned by `submit_raw_sql`.
    _ctx["result"] = {**state.result, "observation": observation}
    return _text(observation)


# ---------------------------------------------------------------------------
# SLayer-mode tools — agent reaches SLayer through its native MCP server
# (configured in run_task via mcp_servers={"slayer": ...}). The only native
# wrapper we keep is `submit_query`, which uses an in-process SlayerClient
# to translate the agent's SLayer query into deterministic SQL and submit
# it through bird-interact's eval pipeline.
# ---------------------------------------------------------------------------

def _slayer_client():
    """Get or build a SlayerClient for the current task's DB (used by submit_query)."""
    client = _ctx.get("_slayer_client")
    if client is None:
        from slayer.client.slayer_client import SlayerClient
        from slayer.storage.yaml_storage import YAMLStorage

        storage_dir = _ctx["slayer_storage_dir"]
        storage = YAMLStorage(base_dir=storage_dir)
        client = SlayerClient(storage=storage)
        _ctx["_slayer_client"] = client
        _ctx["_slayer_storage"] = storage
    return client


@tool(
    "submit_query",
    (
        "Submit your final SLayer query for evaluation. Pass EITHER a "
        "single-stage form (`source_model` + projection fields: "
        "`dimensions`, `measures`, `filters`, `time_dimensions`, `order`, "
        "`limit`, `offset`, `whole_periods_only`, `variables`, "
        "`distinct_dimension_values`) OR a nested-DAG form (`queries`: a "
        "list of stage objects; last element is the DAG root; non-final "
        "stages need a `name`; later stages reference earlier ones via "
        "`source_model: \"<sibling name>\"`). The chosen query is "
        "translated to SQL deterministically and tested against the "
        "ground truth. `normalize_filters` (default true) controls "
        "whether text-equality filter predicates are auto-wrapped in "
        "lower(trim(...)) with the literal lowercased. Pass "
        "`normalize_filters=false` when the gold answer requires "
        "exact-case equality (rare; the default is semantically correct "
        "for most NL questions that carry no casing info)."
    ),
    # CR r1 / O1 follow-up: same unified shape as `query` — caller picks
    # single-stage by populating `source_model`, or nested-DAG by
    # populating `queries`. Mutual exclusion is enforced at the handler
    # level so error messages stay specific.
    {
        "type": "object",
        "properties": {
            "source_model": {
                "oneOf": [{"type": "string"}, {"type": "object"}],
            },
            "measures": {"type": "array"},
            "dimensions": {"type": "array"},
            "filters": {"type": "array"},
            "time_dimensions": {"type": "array"},
            "order": {"type": "array"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "whole_periods_only": {"type": "boolean", "default": False},
            "variables": {"type": "object"},
            "distinct_dimension_values": {"type": "boolean"},
            "queries": {"type": "array"},
            "normalize_filters": {"type": "boolean", "default": True},
        },
        "required": [],
    },
)
async def submit_query(args: dict) -> dict:
    state = _state_view()
    # Build the JSON payload from the structured args. `submit_slayer_query`
    # takes a JSON string (the original CodeRabbit contract); we generate
    # it here from the same structured shape as `query`.
    payload: dict | list
    if args.get("queries") is not None:
        if args.get("source_model") is not None:
            return _text(
                "submit_query: pass either `source_model` (single-stage) "
                "or `queries` (nested DAG), not both."
            )
        payload = args["queries"]
    else:
        if args.get("source_model") is None:
            return _text(
                "submit_query: `source_model` is required for "
                "single-stage queries (or pass `queries` for a "
                "nested DAG)."
            )
        # Single-stage: project the structured args into a SlayerQuery dict.
        # Omit None / falsey values so the downstream JSON matches what the
        # old `query_json` callers wrote by hand.
        single: dict = {"source_model": args["source_model"]}
        for k in (
            "measures", "dimensions", "filters", "time_dimensions",
            "order", "limit", "offset", "variables",
            "distinct_dimension_values",
        ):
            v = args.get(k)
            if v is not None:
                single[k] = v
        if args.get("whole_periods_only"):
            single["whole_periods_only"] = True
        payload = single

    import json as _json
    payload_str = _json.dumps(payload)
    normalize_filters = bool(args.get("normalize_filters", True))
    observation = submit_slayer_query(
        state, payload_str, lambda _s: _slayer_client(),
        normalize_filters=normalize_filters,
    )
    if state.result is None:
        return _text(observation)
    _ctx["result"] = {**state.result, "observation": observation}
    return _text(observation)


# DEV-1534 Fix C + DEV-1546: wrap SLayer's MCP `query` so the agent
# uses the SAME JSON DSL shape as `submit_query` / `query_nested`
# (the field for dedup vs raw rows, `distinct_dimension_values`,
# lives inside that JSON) and so it can opt out of Mode-B filter
# normalization via a separate `normalize_filters` tool parameter.
_QUERY_TOOL_DESC = (
    "Run a SLayer query and return SLayer's formatted result. Pass "
    "EITHER a single-stage form (`source_model` + the usual projection "
    "fields: `dimensions`, `measures`, `filters`, `time_dimensions`, "
    "`order`, `limit`, `offset`, `whole_periods_only`, `variables`, "
    "`distinct_dimension_values`) OR a nested-DAG form (`queries`: a "
    "list of stage objects; last is the DAG root; non-final stages need "
    "a `name`; later stages reference earlier ones via "
    "`source_model: \"<sibling name>\"`). Set "
    "`distinct_dimension_values: false` on a single-stage call to "
    "disable SLayer's default dim-only auto-dedup `GROUP BY`. Tool-level "
    "options live outside the SlayerQuery: `show_sql`, `dry_run`, "
    "`explain`, `format` (markdown/json/csv), and `normalize_filters` "
    "(default true) — when true, every `col == 'X'` filter becomes "
    "`lower(trim(col)) == 'x'` (case/whitespace-tolerant); when false, "
    "filters are forwarded verbatim (exact-case equality)."
)


@tool(
    "query",
    _QUERY_TOOL_DESC,
    # DEV-1555 CR r1 / O1: the wrapper accepts EITHER a single
    # SlayerQuery (set `source_model` + projection fields) OR a
    # nested-DAG list (set `queries`). ``required`` stays empty;
    # the runtime gates on `source_model XOR queries` so the
    # error message is specific. Mirrors `submit_query`'s shape so
    # the agent uses one form across query / submit_query.
    {
        "type": "object",
        "properties": {
            "source_model": {
                "oneOf": [{"type": "string"}, {"type": "object"}],
            },
            "measures": {"type": "array"},
            "dimensions": {"type": "array"},
            "filters": {"type": "array"},
            "time_dimensions": {"type": "array"},
            "order": {"type": "array"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "whole_periods_only": {"type": "boolean", "default": False},
            "variables": {"type": "object"},
            # DEV-1546 (origin/main): inside-SlayerQuery field that
            # disables the default dim-only auto-dedup `GROUP BY`.
            "distinct_dimension_values": {"type": "boolean"},
            "show_sql": {"type": "boolean", "default": False},
            "dry_run": {"type": "boolean", "default": False},
            "explain": {"type": "boolean", "default": False},
            "format": {"type": "string", "default": "markdown"},
            "normalize_filters": {"type": "boolean", "default": True},
            # Nested-DAG: when set, the wrapper routes to
            # `query_nested_impl` and `source_model` + the
            # single-stage projection kwargs must be unset.
            "queries": {"type": "array"},
        },
        "required": [],
    },
)
async def query(args: dict) -> dict:
    # Defer storage attach until first call so the per-task storage
    # set in _slayer_client() is already in _ctx.
    storage = _ctx.get("_slayer_storage")
    if storage is None:
        _slayer_client()  # populates _ctx["_slayer_storage"] as side-effect
        storage = _ctx["_slayer_storage"]
    _query_mod.attach_storage(storage)

    import json as _json

    # Nested-DAG form: when `queries` is supplied, route to
    # `query_nested_impl`. Mutual exclusion with `source_model` is
    # enforced here (rather than in the schema) so a future caller
    # passing both gets a clear error instead of a silent precedence
    # rule.
    if args.get("queries") is not None:
        if args.get("source_model") is not None:
            return _text(
                "query: pass either `source_model` (single-stage) or "
                "`queries` (nested DAG), not both."
            )
        result = await _query_mod.query_nested_impl(
            queries=args["queries"],
            variables=args.get("variables"),
            show_sql=bool(args.get("show_sql", False)),
            dry_run=bool(args.get("dry_run", False)),
            explain=bool(args.get("explain", False)),
            format=args.get("format", "markdown"),
            normalize_filters=bool(args.get("normalize_filters", True)),
        )
        return _text(result if isinstance(result, str) else str(result))

    # Single-stage form: `source_model` is required; build the
    # SlayerQuery JSON from the structured args and hand it to
    # DEV-1546's `query_impl(query_json: str, …)`.
    if args.get("source_model") is None:
        return _text(
            "query: `source_model` is required for single-stage queries "
            "(or pass `queries` for a nested DAG)."
        )
    single: dict = {"source_model": args["source_model"]}
    for k in (
        "measures", "dimensions", "filters", "time_dimensions",
        "order", "limit", "offset", "variables",
        "distinct_dimension_values",
    ):
        v = args.get(k)
        if v is not None:
            single[k] = v
    if args.get("whole_periods_only"):
        single["whole_periods_only"] = True

    result = await _query_mod.query_impl(
        _json.dumps(single),
        show_sql=bool(args.get("show_sql", False)),
        dry_run=bool(args.get("dry_run", False)),
        explain=bool(args.get("explain", False)),
        format=args.get("format", "markdown"),
        normalize_filters=bool(args.get("normalize_filters", True)),
    )
    return _text(result if isinstance(result, str) else str(result))


# DEV-1534 Fix C: same opt-out logic for SLayer's MCP `query_nested`.
# A nested-DAG preview goes through this wrapper so each stage's filters
# can opt out of the default `lower(trim(...))` normalization.
_QUERY_NESTED_TOOL_DESC = (
    "Run a nested-DAG SLayer query (one stage's measure feeding the next "
    "stage's dimension) and return SLayer's formatted result. Same shape "
    "as SLayer's MCP `query_nested` tool — `queries` (required, a list "
    "of stage objects; last is the DAG root; non-final stages need a "
    "`name`; later stages reference earlier ones via `source_model: "
    "\"<sibling name>\"`), `variables`, `show_sql`, `dry_run`, `explain`, "
    "`format` (markdown/json/csv). The 7th parameter `normalize_filters` "
    "(default true) controls our text-equality filter auto-normalization "
    "for every stage's `filters` list: when true, every `col == 'X'` "
    "filter becomes `lower(trim(col)) == 'x'`; when false, filters are "
    "forwarded verbatim (exact-case equality). Pass the stage list in "
    "the `queries` argument."
)


@tool(
    "query_nested",
    _QUERY_NESTED_TOOL_DESC,
    {
        "type": "object",
        "properties": {
            "queries": {"type": "array"},
            "variables": {"type": "object"},
            "show_sql": {"type": "boolean", "default": False},
            "dry_run": {"type": "boolean", "default": False},
            "explain": {"type": "boolean", "default": False},
            "format": {"type": "string", "default": "markdown"},
            "normalize_filters": {"type": "boolean", "default": True},
        },
        "required": ["queries"],
    },
)
async def query_nested(args: dict) -> dict:
    storage = _ctx.get("_slayer_storage")
    if storage is None:
        _slayer_client()
        storage = _ctx["_slayer_storage"]
    _query_mod.attach_storage(storage)

    result = await _query_mod.query_nested_impl(
        queries=args["queries"],
        variables=args.get("variables"),
        show_sql=bool(args.get("show_sql", False)),
        dry_run=bool(args.get("dry_run", False)),
        explain=bool(args.get("explain", False)),
        format=args.get("format", "markdown"),
        normalize_filters=bool(args.get("normalize_filters", True)),
    )
    return _text(result if isinstance(result, str) else str(result))


# ---------------------------------------------------------------------------
# DEV-1581 R2: warm-discovery bridge native + in-process SLayer natives.
#
# R2 drops the SDK-subagent split AND the slayer stdio subprocess. The main
# loop reaches the long-lived *discovery* client only through the in-process
# ``ask_discovery`` tool below; both clients' SLayer tools are in-process
# natives backed by the SAME task-local engine (``_query._get_slayer_tool_fn``),
# so a model main writes is immediately visible to discovery's introspection.
# ---------------------------------------------------------------------------


async def _ask_discovery_impl(question: str) -> str:
    """Forward ``question`` to the per-task warm :class:`DiscoveryChannel`
    stored in ``_ctx['_discovery']`` and return its text answer.

    Never raises into the main loop: if no channel is wired (which should not
    happen in a real run) it returns a usable error string rather than a
    ``KeyError``. The channel itself owns the single-flight lock, the call cap,
    per-stream usage, and its own never-raise contract.
    """
    channel = _ctx.get("_discovery")
    if channel is None:
        return (
            "[discovery unavailable: no discovery channel is wired for this "
            "task] Proceed with the information you already have."
        )
    return await channel.ask(question)


@tool(
    "ask_discovery",
    (
        "Ask the long-lived 'discovery' assistant a focused question and get "
        "back its findings. Use it to clarify request ambiguities with the user "
        "and for broad whole-schema questions; it can also introspect the "
        "schema / sample values / joins / KB definitions on your behalf and "
        "accumulates context across questions, so follow-ups are cheap. (When "
        "your own surface has direct introspection tools, prefer those for "
        "focused lookups.) It cannot submit answers or run your candidate query."
    ),
    {"question": str},
)
async def ask_discovery(args: dict) -> dict:
    return _text(await _ask_discovery_impl(args["question"]))


def _ensure_slayer_storage_attached() -> None:
    """Attach the current task's SLayer storage to the task-local query
    engine so the in-process SLayer natives resolve against it (the shared
    engine both main and discovery read/write)."""
    storage = _ctx.get("_slayer_storage")
    if storage is None:
        _slayer_client()  # populates _ctx["_slayer_storage"] as a side-effect
        storage = _ctx["_slayer_storage"]
    _query_mod.attach_storage(storage)


# SLayer MCP tools we expose as in-process natives (no slayer stdio process).
# create/edit run our filter normalization first — symmetry with the
# pydantic_ai_otf_encode adapter, which already normalizes every write.
_SLAYER_NATIVE_NAMES: frozenset[str] = frozenset({
    "search",
    "models_summary",
    "inspect_model",
    "inspect",
    "recommend_root_model",
    "create_model",
    "edit_model",
    "validate_models",
    "help",
})
_SLAYER_NATIVE_NORMALIZE_WRITE: frozenset[str] = frozenset({
    "create_model",
    "edit_model",
})


@functools.lru_cache(maxsize=None)
def _slayer_tool_metadata(name: str) -> tuple[str, dict]:
    """Extract ``(description, json_schema)`` for a SLayer MCP tool from a
    storage-less ``create_mcp_server`` so the in-process native advertises
    SLayer's real signature to the model.

    The schema is derived from the tool fn's signature and is
    storage-independent; the per-task fn that DOES need storage is resolved
    separately at call time via ``_query._get_slayer_tool_fn`` (the shared,
    task-local engine). Cached per name so we build the storage-less server
    only once per process. Reuses SLayer's own tool definitions rather than
    hand-maintaining a parallel schema.
    """
    from slayer.mcp.server import create_mcp_server

    mcp = create_mcp_server(None)
    t = mcp._tool_manager._tools[name]
    return t.description, t.parameters


def _make_slayer_native(name: str):
    """Build an in-process SDK ``@tool`` for SLayer tool ``name``, bridging
    SLayer's real description + schema and forwarding to the shared task-local
    engine fn. ``create_model`` / ``edit_model`` payloads are filter-normalized
    before SLayer persists them."""
    description, schema = _slayer_tool_metadata(name)
    normalize_write = name in _SLAYER_NATIVE_NORMALIZE_WRITE

    async def _handler(args: dict) -> dict:
        _ensure_slayer_storage_attached()
        payload = dict(args)
        if normalize_write:
            normalized = normalize_tool_filters(name, payload)
            if isinstance(normalized, dict):
                payload = normalized
        fn = _query_mod._get_slayer_tool_fn(name)
        result = fn(**payload)
        if inspect.isawaitable(result):
            result = await result
        return _text(result if isinstance(result, str) else str(result))

    return tool(name, description, schema)(_handler)


# Static in-process natives (module-level singletons defined above).
_STATIC_NATIVE_TOOLS: dict = {
    "execute_sql": execute_sql,
    "get_schema": get_schema,
    "get_all_column_meanings": get_all_column_meanings,
    "get_column_meaning": get_column_meaning,
    "get_all_external_knowledge_names": get_all_external_knowledge_names,
    "get_knowledge_definition": get_knowledge_definition,
    "get_all_knowledge_definitions": get_all_knowledge_definitions,
    "ask_user": ask_user,
    "submit_sql": submit_sql,
    "submit_query": submit_query,
    "query": query,
    "query_nested": query_nested,
    "ask_discovery": ask_discovery,
}

#: Full ``mcp__bird-interact-tools__*`` prefix every in-process native carries.
BIRD_INTERACT_SERVER_NAME = "bird-interact-tools"
_NATIVE_PREFIX = f"mcp__{BIRD_INTERACT_SERVER_NAME}__"


def native_tool_full_name(bare_name: str) -> str:
    """Map a bare native tool name to its full ``mcp__bird-interact-tools__*``
    name (what ``allowed_tools`` / the agents' partition constants use)."""
    return f"{_NATIVE_PREFIX}{bare_name}"


def resolve_native_tool(bare_name: str):
    """Return the in-process ``SdkMcpTool`` for ``bare_name`` (static singleton
    or a freshly-built SLayer native). Raises ``KeyError`` for unknown names so
    a typo in an agent's partition constant fails loudly at build time."""
    if bare_name in _STATIC_NATIVE_TOOLS:
        return _STATIC_NATIVE_TOOLS[bare_name]
    if bare_name in _SLAYER_NATIVE_NAMES:
        return _make_slayer_native(bare_name)
    raise KeyError(f"unknown in-process native tool {bare_name!r}")


def build_bird_interact_server(full_or_bare_names):
    """Build the per-client in-process ``bird-interact-tools`` MCP server for
    the given tool names (full ``mcp__bird-interact-tools__*`` or bare).

    Each ClaudeSDKClient (main, discovery) gets its OWN server holding only
    its half of the partition — because the two clients are separate
    subprocesses, the main client's tool schema never contains discovery's
    introspection tools (the core DEV-1581 guarantee), and vice-versa.
    """
    tools = []
    for n in full_or_bare_names:
        bare = n[len(_NATIVE_PREFIX):] if n.startswith(_NATIVE_PREFIX) else n
        tools.append(resolve_native_tool(bare))
    return create_sdk_mcp_server(
        name=BIRD_INTERACT_SERVER_NAME, version="1.0.0", tools=tools,
    )


# NOTE: this module no longer exports a top-level ``ClaudeSDKAgent``
# class, ``RAW_A_TOOLS`` / ``RAW_C_TOOLS`` / ``SLAYER_A_TOOLS`` /
# ``SLAYER_C_TOOLS`` tool lists, ``SLAYER_MCP_TOOL_NAMES``, or
# ``_select_tools`` / ``_build_prompt`` helpers. The production CLI
# (``--framework claude_sdk``) dispatches via ``run.py`` to one of four
# narrow OTF agents (``claude_sdk_otf{,_ainteract,_raw,_ainteract_raw}``)
# — each owns its own run loop, prompt, tool list, and pre-submit
# guards, and imports the in-process @tool functions above from THIS
# module. The pre-OTF orchestrator class + its tool-list constants /
# prompt-dispatch helpers were unreachable from the CLI after DEV-1507
# (livesqlbench OTF split) and are removed in DEV-1534 to avoid
# duplicate maintenance.
