"""DEV-1534 Fix C: new `mcp__bird-interact-tools__query` wrapper tool.

claude_sdk in slayer mode currently allows `mcp__slayer__query` (SLayer's
own MCP query tool, called directly via the slayer subprocess MCP server)
in the agent's `allowed_tools` list. SLayer's MCP server applies our
filter normalization NEVER — only the submit path does. The agent
therefore cannot preview the LOWER(TRIM(...))-wrapped SQL it'll actually
submit against, and cannot opt out at preview time either.

Fix C introduces a new tool we own — registered on the bird-interact-tools
SDK MCP server — that:

* Accepts SLayer MCP `query`'s exact 14-parameter signature
  (`source_model`, `dimensions`, `measures`, `filters`, `time_dimensions`,
  `order`, `limit`, `offset`, `whole_periods_only`, `show_sql`, `dry_run`,
  `explain`, `format`, `variables`).
* Adds a 15th parameter `normalize_filters: bool = True`.
* Pre-processes the `filters` list via `normalize_filters_list(filters,
  normalize=normalize_filters)` BEFORE forwarding to SLayer's MCP query
  implementation.
* Forwards every other parameter VERBATIM (no reformatting; SLayer's
  function does dry_run/explain/show_sql/format branches and friendly
  DB-error handling itself, so the agent's mid-flight experience is
  identical except for the new flag).
* Does NOT forward `normalize_filters` to SLayer (it's our directive).
* Replaces `mcp__slayer__query` in claude_sdk's `allowed_tools` for the
  slayer-mode query mode.
* Has bird-coin cost 0 (mirrors SLayer's current free `query`).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Wrapper tool exists + signature matches SLayer MCP `query` + extra flag
# ---------------------------------------------------------------------------


def test_query_wrapper_module_exists():
    """Wrapper lives in `agents/_query.py` with a public callable
    that the bird-interact-tools SDK MCP server registers."""
    from bird_interact_agents.agents import _query  # noqa: F401


# DEV-1546: the 15-kwarg signature pin (one kwarg per SlayerQuery field
# + normalize_filters) was replaced by a single ``query_json`` arg in
# DEV-1546 — the wrapper now mirrors ``submit_slayer_query``'s shape so
# the agent uses one JSON DSL form everywhere. The new signature pin
# lives in ``tests/test_dev1546_distinct_dim_values.py``.


# ---------------------------------------------------------------------------
# Behavior: normalize_filters flag controls filter pre-processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_wrapper_default_normalizes_filters(monkeypatch):
    """Default (or explicit True): filters inside the JSON DSL are
    normalized before forwarding to SLayer's MCP `query` function."""
    import json

    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted result>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    result = await _query.query_impl(
        json.dumps({
            "source_model": "widgets",
            "filters": ["category == 'Gadgets'"],
        }),
    )
    assert forwarded["filters"] == ["lower(trim(category)) == 'gadgets'"]
    assert result == "<formatted result>"


@pytest.mark.asyncio
async def test_query_wrapper_normalize_filters_false_passes_verbatim(monkeypatch):
    """`normalize_filters=False`: the filters list passed to SLayer's
    `query.fn` is the agent's ORIGINAL filters (no lower/trim wrap)."""
    import json

    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted result>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(
        json.dumps({
            "source_model": "widgets",
            "filters": ["category == 'Gadgets'"],
        }),
        normalize_filters=False,
    )
    assert forwarded["filters"] == ["category == 'Gadgets'"]


@pytest.mark.asyncio
async def test_query_wrapper_does_not_forward_normalize_filters(monkeypatch):
    """`normalize_filters` is OUR directive — it MUST NOT appear in
    the kwargs forwarded to SLayer's MCP `query` function (which knows
    nothing about it)."""
    import json

    from bird_interact_agents.agents import _query

    forwarded_kwarg_keys = []
    async def fake_slayer_query(**kwargs):
        forwarded_kwarg_keys.extend(kwargs.keys())
        return "<formatted>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(
        json.dumps({
            "source_model": "widgets",
            "filters": ["category == 'X'"],
        }),
        normalize_filters=False,
    )
    assert "normalize_filters" not in forwarded_kwarg_keys


@pytest.mark.asyncio
async def test_query_wrapper_forwards_all_slayer_params(monkeypatch):
    """All SLayer MCP `query` parameters in the JSON DSL plus the
    tool-level kwargs (show_sql/dry_run/explain/format) are forwarded
    verbatim. We don't re-implement dry_run / explain / show_sql /
    format / friendly-DB-error handling — SLayer's own `query` function
    does that and we forward its output."""
    import json

    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(
        json.dumps({
            "source_model": "widgets",
            "dimensions": ["category"],
            "measures": [{"name": "amount", "agg": "sum"}],
            "filters": ["region == 'EU'"],
            "time_dimensions": [
                {"dimension": "created_at", "granularity": "month"},
            ],
            "order": [{"dimension": "category", "direction": "asc"}],
            "limit": 10,
            "offset": 5,
            "whole_periods_only": True,
            "variables": {"k": "v"},
        }),
        show_sql=True,
        dry_run=True,
        explain=True,
        format="json",
        normalize_filters=False,
    )
    assert forwarded["source_model"] == "widgets"
    assert forwarded["dimensions"] == ["category"]
    assert forwarded["measures"] == [{"name": "amount", "agg": "sum"}]
    assert forwarded["filters"] == ["region == 'EU'"]
    assert forwarded["time_dimensions"] == [
        {"dimension": "created_at", "granularity": "month"}
    ]
    assert forwarded["order"] == [{"dimension": "category", "direction": "asc"}]
    assert forwarded["limit"] == 10
    assert forwarded["offset"] == 5
    assert forwarded["whole_periods_only"] is True
    assert forwarded["show_sql"] is True
    assert forwarded["dry_run"] is True
    assert forwarded["explain"] is True
    assert forwarded["format"] == "json"
    assert forwarded["variables"] == {"k": "v"}


@pytest.mark.asyncio
async def test_query_wrapper_returns_slayer_output_verbatim(monkeypatch):
    """We don't reformat SLayer's response — whatever string SLayer's
    `query` function returns (markdown/json/csv/SQL/explain/error) is
    the wrapper's return value, byte-for-byte. A wrapper that prepends/
    appends/reformats output must fail this exact-match assertion."""
    import json

    from bird_interact_agents.agents import _query

    fixture = "## arbitrary slayer-shaped markdown\n| col | val |\n|---|---|\n| a | 1 |"

    async def fake_slayer_query(**kwargs):
        return fixture

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    result = await _query.query_impl(json.dumps({"source_model": "w"}))
    assert result == fixture


@pytest.mark.asyncio
async def test_query_wrapper_filters_none_default_safe(monkeypatch):
    """`filters` is Optional[List[str]] — None (or omitted from the
    JSON) must not blow up the normalizer."""
    import json

    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return ""

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(json.dumps({"source_model": "w"}))
    assert forwarded.get("filters") is None


# ---------------------------------------------------------------------------
# `query_nested` wrapper — same opt-out logic per stage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_nested_wrapper_default_normalizes_each_stage(monkeypatch):
    """Default (normalize_filters=True): every stage's `filters` list is
    pre-processed before forwarding to SLayer's MCP `query_nested`."""
    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake(**kwargs):
        forwarded.update(kwargs)
        return "<formatted>"

    monkeypatch.setattr(
        _query, "_get_slayer_tool_fn",
        lambda name: fake if name == "query_nested" else None,
    )

    stages = [
        {
            "name": "stage1",
            "source_model": "orders",
            "filters": ["region == 'EU'"],
        },
        {
            "source_model": "stage1",
            "filters": ["category == 'Gadgets'"],
        },
    ]
    await _query.query_nested_impl(queries=stages)
    out_stages = forwarded["queries"]
    assert out_stages[0]["filters"] == ["lower(trim(region)) == 'eu'"]
    assert out_stages[1]["filters"] == ["lower(trim(category)) == 'gadgets'"]
    # Inputs were deep-copied (verbatim — never mutated).
    assert stages[0]["filters"] == ["region == 'EU'"]


@pytest.mark.asyncio
async def test_query_nested_wrapper_opt_out_passes_each_stage_verbatim(
    monkeypatch,
):
    """`normalize_filters=False`: every stage's `filters` is forwarded
    BYTE-VERBATIM (no lower/trim wrap, no literal lowercasing)."""
    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake(**kwargs):
        forwarded.update(kwargs)
        return ""

    monkeypatch.setattr(
        _query, "_get_slayer_tool_fn",
        lambda name: fake if name == "query_nested" else None,
    )

    stages = [
        {"source_model": "orders", "filters": ["region == 'EU'"]},
        {"source_model": "products", "filters": ["category == 'Gadgets'"]},
    ]
    await _query.query_nested_impl(queries=stages, normalize_filters=False)
    out_stages = forwarded["queries"]
    assert out_stages[0]["filters"] == ["region == 'EU'"]
    assert out_stages[1]["filters"] == ["category == 'Gadgets'"]


@pytest.mark.asyncio
async def test_query_nested_wrapper_does_not_forward_normalize_filters(
    monkeypatch,
):
    """`normalize_filters` is OUR directive — MUST NOT appear in the
    kwargs forwarded to SLayer's MCP `query_nested`."""
    from bird_interact_agents.agents import _query

    forwarded_keys: list[str] = []
    async def fake(**kwargs):
        forwarded_keys.extend(kwargs.keys())
        return ""

    monkeypatch.setattr(
        _query, "_get_slayer_tool_fn",
        lambda name: fake if name == "query_nested" else None,
    )

    await _query.query_nested_impl(
        queries=[{"source_model": "w"}],
        normalize_filters=False,
    )
    assert "normalize_filters" not in forwarded_keys


# ---------------------------------------------------------------------------
# Extraction path — `_get_slayer_query_fn` uses `create_mcp_server`
# ---------------------------------------------------------------------------


def test_get_slayer_query_fn_extracts_from_create_mcp_server(monkeypatch):
    """The plan: extract SLayer's MCP `query` function via
    `create_mcp_server(storage)._tool_manager._tools["query"].fn`. Pin
    that path so a future implementation can't bypass it (e.g. by
    importing the inner closure from elsewhere) without updating this
    test."""
    from bird_interact_agents.agents import _query
    from slayer.mcp import server as slayer_mcp_server

    # DEV-1581: storage + cache are now task-local (contextvar). Reset to
    # a clean state so a prior test's cached fn can't satisfy this lookup.
    _query.attach_storage(None)

    seen = {"create_mcp_server_called": False}

    class _FakeTool:
        def __init__(self, fn):
            self.fn = fn

    class _FakeManager:
        def __init__(self, tools):
            self._tools = tools

    class _FakeFastMCP:
        def __init__(self, tools):
            self._tool_manager = _FakeManager(tools)

    async def fake_inner_query(**kwargs):
        return "<from create_mcp_server>"

    def fake_create_mcp_server(storage, **kwargs):
        seen["create_mcp_server_called"] = True
        return _FakeFastMCP({"query": _FakeTool(fake_inner_query)})

    monkeypatch.setattr(
        slayer_mcp_server, "create_mcp_server", fake_create_mcp_server,
    )

    fn = _query._get_slayer_query_fn()
    assert callable(fn)
    assert seen["create_mcp_server_called"], (
        "_get_slayer_query_fn must extract via create_mcp_server(storage)"
        "._tool_manager._tools['query'].fn — not by importing the inner "
        "closure from elsewhere."
    )
    # The extracted callable IS the same one we registered.
    assert fn is fake_inner_query


def test_attach_storage_idempotent_keeps_cache(monkeypatch):
    """Post-CR-review: `attach_storage(storage)` called repeatedly with
    the SAME storage object must NOT invalidate the cached `query.fn`.
    Otherwise the per-call `attach_storage` in claude_sdk's `query`
    handler defeats the cache and `create_mcp_server` re-runs every time.
    """
    from bird_interact_agents.agents import _query
    from slayer.mcp import server as slayer_mcp_server

    # DEV-1581: storage + cache are now task-local (contextvar); reset to a
    # clean state so prior tests don't leak cache/storage in.
    _query.attach_storage(None)

    calls = {"count": 0}

    class _FakeTool:
        def __init__(self, fn): self.fn = fn

    class _FakeManager:
        def __init__(self, tools): self._tools = tools

    class _FakeFastMCP:
        def __init__(self, tools): self._tool_manager = _FakeManager(tools)

    async def fake_inner_query(**kwargs): return "ok"

    def fake_create_mcp_server(storage, **kwargs):
        calls["count"] += 1
        return _FakeFastMCP({"query": _FakeTool(fake_inner_query)})

    monkeypatch.setattr(
        slayer_mcp_server, "create_mcp_server", fake_create_mcp_server,
    )

    storage = object()  # the same identity attached three times.
    _query.attach_storage(storage)
    _query._get_slayer_query_fn()
    _query.attach_storage(storage)
    _query._get_slayer_query_fn()
    _query.attach_storage(storage)
    _query._get_slayer_query_fn()

    assert calls["count"] == 1, (
        f"create_mcp_server should run ONCE when attach_storage is called "
        f"repeatedly with the SAME storage object; ran {calls['count']} times."
    )


def test_attach_storage_invalidates_on_storage_swap(monkeypatch):
    """Sanity check the OTHER side: a NEW storage object DOES invalidate
    the cache and triggers a fresh extraction. (The guard mustn't ALSO
    miss real swaps between tasks.)"""
    from bird_interact_agents.agents import _query
    from slayer.mcp import server as slayer_mcp_server

    # DEV-1581: storage + cache are now task-local (contextvar); reset clean.
    _query.attach_storage(None)

    calls = {"count": 0}

    class _FakeTool:
        def __init__(self, fn): self.fn = fn

    class _FakeManager:
        def __init__(self, tools): self._tools = tools

    class _FakeFastMCP:
        def __init__(self, tools): self._tool_manager = _FakeManager(tools)

    async def fake_inner_query(**kwargs): return "ok"

    def fake_create_mcp_server(storage, **kwargs):
        calls["count"] += 1
        return _FakeFastMCP({"query": _FakeTool(fake_inner_query)})

    monkeypatch.setattr(
        slayer_mcp_server, "create_mcp_server", fake_create_mcp_server,
    )

    storage_a = object()
    storage_b = object()
    _query.attach_storage(storage_a)
    _query._get_slayer_query_fn()
    _query.attach_storage(storage_b)  # different identity → invalidates
    _query._get_slayer_query_fn()

    assert calls["count"] == 2, (
        f"create_mcp_server should re-run when storage identity changes; "
        f"ran {calls['count']} times."
    )


# ---------------------------------------------------------------------------
# Allowlist swap — claude_sdk uses bird-interact-tools `query`, not slayer's
# ---------------------------------------------------------------------------


def test_otf_one_shot_slayer_allowlist_excludes_subprocess_query():
    """When the production OTF slayer one-shot agent runs, the allowed
    `mcp__slayer__*` tools MUST NOT include `query` or `query_nested` —
    our bird-interact-tools wrappers replace them so the agent can opt
    out of filter normalization via `normalize_filters=false`."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as otf_mod

    assert "query" not in otf_mod.SLAYER_MCP_TOOLS, (
        "DEV-1534: `query` must come from bird-interact-tools (our wrapper), "
        "NOT from mcp__slayer__query."
    )
    assert "query_nested" not in otf_mod.SLAYER_MCP_TOOLS, (
        "DEV-1534: `query_nested` must come from bird-interact-tools (our "
        "wrapper), NOT from mcp__slayer__query_nested."
    )
    # Spot-check the other slayer MCP tools are still there.
    for kept in (
        "help", "list_datasources", "models_summary", "inspect_model",
        "search", "create_model", "edit_model", "save_memory",
        "validate_models",
    ):
        assert kept in otf_mod.SLAYER_MCP_TOOLS, (
            f"unexpectedly dropped SLayer MCP tool: {kept}"
        )


def test_query_wrapper_in_otf_slayer_one_shot_tool_list():
    """DEV-1555 CR r1 unification: the OTF slayer one-shot agent's
    in-process tool list includes the unified `query` wrapper (no
    `query_nested`); `query` accepts object OR list of stages."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as otf_mod

    names = {getattr(t, "name", None) for t in otf_mod._KNOWLEDGE_TOOLS}
    assert "query" in names
    assert "query_nested" not in names


def test_query_wrapper_in_otf_slayer_ainteract_tool_list():
    """Same for the OTF slayer a-interact agent."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import (
        agent as otf_mod,
    )

    names = {getattr(t, "name", None) for t in otf_mod._KNOWLEDGE_TOOLS}
    assert "query" in names
    assert "query_nested" not in names


def test_otf_slayer_pre_submit_gate_accepts_unified_query_wrapper():
    """DEV-1555 CR r1: the pre-submit verification gate's allow-set
    contains the unified ``query`` wrapper name only — ``query_nested``
    is gone (the same `query` tool now handles both shapes)."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as otf_mod

    assert otf_mod.SLAYER_QUERY_TOOLS == frozenset({
        "mcp__bird-interact-tools__query",
    })


# ---------------------------------------------------------------------------
# SDK tool schema — ONLY `source_model` is required
# ---------------------------------------------------------------------------


def test_query_wrapper_tool_schema_accepts_unified_shape():
    """DEV-1555 (CR r1 / O1) unification: the v1 ``query`` wrapper
    accepts BOTH a single SlayerQuery (set ``source_model`` + projection
    fields) AND a nested-DAG list (set ``queries``). The schema's
    ``required`` is empty; the wrapper enforces ``source_model XOR
    queries`` at runtime so the error message is more useful than a
    JSON-Schema validation failure.

    This supersedes the pre-unification pin "only ``query_json``
    required" that lived here briefly (DEV-1546 follow-up).
    """
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    tool = agent_mod.query
    schema = None
    for attr in ("inputSchema", "input_schema", "schema", "args_schema"):
        s = getattr(tool, attr, None)
        if s is not None:
            schema = s
            break
    assert schema is not None and isinstance(schema, dict) and "properties" in schema, (
        f"query wrapper must expose an explicit JSON Schema dict; got "
        f"{schema!r}"
    )
    required = schema.get("required", [])
    assert required == [], (
        f"query wrapper must accept EITHER source_model OR queries — "
        f"`required` must be empty; got: {required!r}"
    )
    # Every SLayer projection param + `queries` array + `normalize_filters`
    # must be in properties so the agent knows the surface.
    props = schema["properties"]
    for name in (
        "source_model", "queries", "measures", "dimensions", "filters",
        "time_dimensions", "order", "limit", "offset",
        "whole_periods_only", "show_sql", "dry_run", "explain",
        "format", "variables", "normalize_filters",
    ):
        assert name in props, f"property {name!r} missing from query wrapper schema"
    # Booleans declared as `boolean`, not anything else.
    assert props["normalize_filters"]["type"] == "boolean"
    assert props["normalize_filters"].get("default") is True
    assert props["queries"].get("type") == "array"
    # `source_model` must accept BOTH a model-name string AND an inline
    # ModelExtension object (SLayer's MCP query signature is
    # `str | ModelExtension | SlayerModel`; SLAYER_A_INTERACT explicitly
    # documents the inline-object form).
    sm = props["source_model"]
    one_of = sm.get("oneOf")
    assert isinstance(one_of, list) and len(one_of) >= 2, (
        f"source_model must use oneOf to permit str | object; got {sm!r}"
    )
    type_keys = {entry.get("type") for entry in one_of if isinstance(entry, dict)}
    assert {"string", "object"}.issubset(type_keys), (
        f"source_model schema must permit both `string` and `object`; "
        f"types seen: {type_keys}"
    )


# ---------------------------------------------------------------------------
# Cost: the wrapper is FREE (0 bird-coins) — matches current SLayer query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_wrapper_does_not_decrement_budget(monkeypatch):
    """The wrapper must NOT charge bird-coins. SLayer's subprocess MCP
    `query` was effectively free for claude_sdk (the subprocess sits
    outside our budget bookkeeping); the wrapper preserves that
    semantics by never calling `update_budget`.

    We verify the BEHAVIOR (budget unchanged after a wrapper call), not
    the shared `ACTION_COSTS["query"]` dict value — that entry is still
    consulted by other adapters (pydantic_ai_*, smolagents, etc.) and
    is OUT OF SCOPE for DEV-1534."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.agents import _query as _query_mod
    from bird_interact_agents.config import settings
    from bird_interact_agents.harness import SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=20.0, total_budget=20.0,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": object(),  # any non-None so _slayer_client() is skipped
        "_slayer_storage": object(),
        "result": None,
    })

    async def fake_slayer_query(**kwargs):
        return "ok"

    monkeypatch.setattr(_query_mod, "_get_slayer_query_fn", lambda: fake_slayer_query)

    status = agent_mod._ctx_var.get()["status"]
    start = status.remaining_budget

    import json as _json

    await agent_mod.query.handler({"query_json": _json.dumps({"source_model": "w"})})

    assert status.remaining_budget == start, (
        f"query wrapper should be free; budget moved "
        f"{start} -> {status.remaining_budget}"
    )
