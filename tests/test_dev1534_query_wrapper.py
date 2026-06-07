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

import inspect

import pytest


# ---------------------------------------------------------------------------
# Wrapper tool exists + signature matches SLayer MCP `query` + extra flag
# ---------------------------------------------------------------------------


def test_query_wrapper_module_exists():
    """Wrapper lives in `agents/_query.py` with a public callable
    that the bird-interact-tools SDK MCP server registers."""
    from bird_interact_agents.agents import _query  # noqa: F401


def test_query_wrapper_signature_mirrors_slayer_plus_normalize_filters():
    """The wrapper's signature MUST be the EXACT 14 SLayer MCP query
    parameters in order PLUS `normalize_filters: bool = True` as the
    15th (last). Extra params, missing params, wrong order, or wrong
    defaults all fail this test. This pins the contract: the agent's
    mid-flight tool call should be byte-compatible with
    `mcp__slayer__query` except for the new flag.

    SLayer's MCP `query` parameters (verified against slayer 0.7+):
      source_model, measures, dimensions, filters, time_dimensions,
      order, limit, offset, whole_periods_only, show_sql, dry_run,
      explain, format, variables.
    """
    from bird_interact_agents.agents._query import query_impl

    sig = inspect.signature(query_impl)
    expected_order = [
        "source_model",
        "measures",
        "dimensions",
        "filters",
        "time_dimensions",
        "order",
        "limit",
        "offset",
        "whole_periods_only",
        "show_sql",
        "dry_run",
        "explain",
        "format",
        "variables",
        "normalize_filters",
    ]
    assert list(sig.parameters) == expected_order, (
        f"wrapper signature order mismatch:\n"
        f"  expected: {expected_order}\n"
        f"  got:      {list(sig.parameters)}"
    )
    # Defaults must match the SLayer MCP query contract.
    defaults = {n: p.default for n, p in sig.parameters.items()}
    assert defaults["measures"] is None
    assert defaults["dimensions"] is None
    assert defaults["filters"] is None
    assert defaults["time_dimensions"] is None
    assert defaults["order"] is None
    assert defaults["limit"] is None
    assert defaults["offset"] is None
    assert defaults["whole_periods_only"] is False
    assert defaults["show_sql"] is False
    assert defaults["dry_run"] is False
    assert defaults["explain"] is False
    assert defaults["format"] == "markdown"
    assert defaults["variables"] is None
    assert defaults["normalize_filters"] is True
    # source_model has no default (required positional).
    assert sig.parameters["source_model"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Behavior: normalize_filters flag controls filter pre-processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_wrapper_default_normalizes_filters(monkeypatch):
    """Default (or explicit True): filters are normalized before
    forwarding to SLayer's MCP `query` function."""
    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted result>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    result = await _query.query_impl(
        source_model="widgets",
        filters=["category == 'Gadgets'"],
    )
    assert forwarded["filters"] == ["lower(trim(category)) == 'gadgets'"]
    assert result == "<formatted result>"


@pytest.mark.asyncio
async def test_query_wrapper_normalize_filters_false_passes_verbatim(monkeypatch):
    """`normalize_filters=False`: the filters list passed to SLayer's
    `query.fn` is the agent's ORIGINAL filters (no lower/trim wrap)."""
    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted result>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(
        source_model="widgets",
        filters=["category == 'Gadgets'"],
        normalize_filters=False,
    )
    assert forwarded["filters"] == ["category == 'Gadgets'"]


@pytest.mark.asyncio
async def test_query_wrapper_does_not_forward_normalize_filters(monkeypatch):
    """`normalize_filters` is OUR directive — it MUST NOT appear in
    the kwargs forwarded to SLayer's MCP `query` function (which knows
    nothing about it)."""
    from bird_interact_agents.agents import _query

    forwarded_kwarg_keys = []
    async def fake_slayer_query(**kwargs):
        forwarded_kwarg_keys.extend(kwargs.keys())
        return "<formatted>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(
        source_model="widgets",
        filters=["category == 'X'"],
        normalize_filters=False,
    )
    assert "normalize_filters" not in forwarded_kwarg_keys


@pytest.mark.asyncio
async def test_query_wrapper_forwards_all_slayer_params(monkeypatch):
    """All SLayer MCP `query` parameters are forwarded verbatim. We
    don't re-implement dry_run / explain / show_sql / format /
    friendly-DB-error handling — SLayer's own `query` function does
    that and we forward its output."""
    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(
        source_model="widgets",
        dimensions=["category"],
        measures=[{"name": "amount", "agg": "sum"}],
        filters=["region == 'EU'"],
        time_dimensions=[{"dimension": "created_at", "granularity": "month"}],
        order=[{"dimension": "category", "direction": "asc"}],
        limit=10,
        offset=5,
        whole_periods_only=True,
        show_sql=True,
        dry_run=True,
        explain=True,
        format="json",
        variables={"k": "v"},
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
    from bird_interact_agents.agents import _query

    fixture = "## arbitrary slayer-shaped markdown\n| col | val |\n|---|---|\n| a | 1 |"

    async def fake_slayer_query(**kwargs):
        return fixture

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    result = await _query.query_impl(source_model="w")
    assert result == fixture


@pytest.mark.asyncio
async def test_query_wrapper_filters_none_default_safe(monkeypatch):
    """`filters` is Optional[List[str]] — None must not blow up the
    normalizer."""
    from bird_interact_agents.agents import _query

    forwarded = {}
    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return ""

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)

    await _query.query_impl(source_model="w")
    assert forwarded.get("filters") is None


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

    # Reset any cached fn so the test exercises the extraction path
    # cleanly (the wrapper caches by storage; we monkeypatch the
    # factory so the cache key doesn't matter).
    cache_attr = "_cached_slayer_query_fn"
    if hasattr(_query, cache_attr):
        setattr(_query, cache_attr, None)

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


# ---------------------------------------------------------------------------
# Allowlist swap — claude_sdk uses bird-interact-tools `query`, not slayer's
# ---------------------------------------------------------------------------


def test_claude_sdk_slayer_mode_does_not_allowlist_mcp_slayer_query():
    """When the claude_sdk agent runs in slayer mode, the allowed
    `mcp__slayer__*` tools MUST NOT include `query` — our wrapper
    replaces it."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    # The slayer_tools list is defined inline in the run loop; pull it
    # from the module-level constant if one was exposed, otherwise
    # parse the loop. The implementation should expose it as a constant
    # (e.g. `SLAYER_MCP_TOOL_NAMES`) so tests can pin it.
    names = getattr(agent_mod, "SLAYER_MCP_TOOL_NAMES", None)
    assert names is not None, (
        "claude_sdk must expose `SLAYER_MCP_TOOL_NAMES` for the slayer "
        "subprocess MCP allowlist so tests can pin which tools come from "
        "SLayer vs from our wrapper."
    )
    assert "query" not in names, (
        "DEV-1534: `query` must come from bird-interact-tools (our wrapper), "
        "NOT from mcp__slayer__query."
    )
    # Spot-check the other slayer MCP tools are still there.
    for kept in ("help", "list_datasources", "models_summary", "inspect_model"):
        assert kept in names, f"unexpectedly dropped SLayer MCP tool: {kept}"


def test_query_wrapper_in_slayer_a_tools_list():
    """The new wrapper is included in `SLAYER_A_TOOLS` (a-interact slayer
    mode) so claude_sdk registers it on the bird-interact-tools SDK MCP
    server."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    names = {getattr(t, "name", None) for t in agent_mod.SLAYER_A_TOOLS}
    assert "query" in names, (
        "the new `query` wrapper must be in SLAYER_A_TOOLS so it's "
        "registered on bird-interact-tools (visible to claude_sdk as "
        "mcp__bird-interact-tools__query)."
    )


def test_query_wrapper_in_slayer_c_tools_list():
    """Same for c-interact slayer mode."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    names = {getattr(t, "name", None) for t in agent_mod.SLAYER_C_TOOLS}
    assert "query" in names


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

    await agent_mod.query.handler({"source_model": "w"})

    assert status.remaining_budget == start, (
        f"query wrapper should be free; budget moved "
        f"{start} -> {status.remaining_budget}"
    )
