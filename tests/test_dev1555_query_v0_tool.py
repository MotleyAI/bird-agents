"""DEV-1555 v0/v1 split — the v0 ``query`` MCP tool.

Origin/main wrapped SLayer's MCP ``query`` so the agent passed ONE
``query_json`` argument (a stringified ``SlayerQuery`` JSON object). This
PR rewrote that wrapper to use positional fields (``source_model``,
``dimensions``, ``measures``, …). The v0 prompts assume the old shape,
so v0 agents must register a separate MCP tool with the ``query_json``
schema.

The plan puts that tool in ``agents/claude_sdk/_query_v0.py`` exporting
a single ``query`` symbol (function name irrelevant — registered MCP
name comes from the ``@tool("query", ...)`` first arg).

If a future implementer aliases ``query_v0 as query`` instead of naming
the function ``query``, the import still resolves — but the
``test_v0_function_name_is_query`` check fires to flag the cosmetic
inconsistency. The behavioural contract is the SCHEMA, asserted below.
"""

from __future__ import annotations


def test_v0_query_module_exports_query():
    """``_query_v0`` exposes a single public ``query`` symbol."""
    from bird_interact_agents.agents.claude_sdk import _query_v0

    assert hasattr(_query_v0, "query"), (
        "agents/claude_sdk/_query_v0.py must export `query` (the v0 tool)."
    )


def test_v0_query_is_sdk_mcp_tool():
    """``query`` is an ``SdkMcpTool`` instance (decorated with @tool)."""
    from claude_agent_sdk import SdkMcpTool

    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    assert isinstance(query, SdkMcpTool), (
        f"`query` must be an SdkMcpTool, got {type(query).__name__}. "
        "The function must be decorated with `@tool('query', ...)`."
    )


def test_v0_query_registered_name_is_query():
    """Registered MCP tool name is literally ``"query"`` (NOT ``"query_v0"``)."""
    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    assert query.name == "query", (
        f"v0 query tool must register as 'query', got '{query.name}'. "
        "v0 prompts call `query`; mismatching the registered name "
        "breaks v0 in production."
    )


def test_v0_query_schema_requires_query_json_only():
    """v0 schema requires exactly ``query_json``; everything else optional.

    This is the SHAPE the origin/main prompts assume. v0 must NOT require
    ``source_model`` (that's the v1 schema).
    """
    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    schema = query.input_schema
    assert isinstance(schema, dict), (
        f"v0 query schema must be a dict, got {type(schema).__name__}."
    )
    required = schema.get("required", [])
    assert required == ["query_json"], (
        f"v0 query schema must require exactly ['query_json'], got {required}. "
        "Required-only-source_model is the V1 schema."
    )


def test_v0_query_schema_has_query_json_string_property():
    """``query_json`` is typed as a string in the v0 schema (matches origin/main)."""
    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    props = query.input_schema.get("properties", {})
    assert "query_json" in props, (
        f"v0 query schema is missing 'query_json' property: {list(props)}"
    )
    qj = props["query_json"]
    assert qj.get("type") == "string", (
        f"v0 query_json must be type=string, got {qj!r}"
    )


def test_v0_query_schema_has_normalize_filters_opt_out():
    """The DEV-1534 Fix C ``normalize_filters`` knob survives in v0.

    The opt-out flag is part of origin/main's v0 query tool, used by v0
    prompts to disable filter case/whitespace normalization mid-flight.
    """
    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    props = query.input_schema.get("properties", {})
    assert "normalize_filters" in props, (
        "v0 query schema missing `normalize_filters` knob "
        f"(props: {list(props)})"
    )
    nf = props["normalize_filters"]
    assert nf.get("type") == "boolean", (
        f"v0 normalize_filters must be type=boolean, got {nf!r}"
    )
    assert nf.get("default") is True, (
        f"v0 normalize_filters default must be True, got {nf.get('default')!r}"
    )


def test_v0_query_schema_has_legacy_tool_level_knobs():
    """``show_sql`` / ``dry_run`` / ``explain`` / ``format`` are optional kwargs.

    Origin/main exposed these as tool-level kwargs (outside the JSON DSL).
    v0 must preserve them so the v0 prompts that mention them still work.
    """
    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    props = query.input_schema.get("properties", {})
    for kw in ("show_sql", "dry_run", "explain", "format"):
        assert kw in props, (
            f"v0 query schema missing legacy kwarg `{kw}` "
            f"(props: {list(props)})"
        )


def test_v0_query_schema_does_not_require_source_model():
    """v0 must NOT require ``source_model`` (that's the v1 contract).

    Belt-and-braces: even if `required` list is extended, source_model
    isn't there.
    """
    from bird_interact_agents.agents.claude_sdk._query_v0 import query

    required = query.input_schema.get("required", [])
    assert "source_model" not in required, (
        f"v0 must not require `source_model` (got required={required}); "
        "that's the V1 schema."
    )


# ---------------------------------------------------------------------------
# Cosmetic: source-level name. If someone aliased `query_v0 as query` at
# import sites instead of defining the function as `query` in the module,
# the registered MCP name is still correct (handled by `@tool('query', ...)`
# anyway). This test is informational — passing means the cleanest source
# shape was used.
# ---------------------------------------------------------------------------


def test_v0_query_module_exposes_query_in_public_surface():
    """``_query_v0`` has ``query`` in its public surface.

    Mirrors the import sites — v0 agents do
    ``from agents.claude_sdk._query_v0 import query``, so ``query``
    must be a direct attribute on the module (not buried inside an
    internal private name).
    """
    from bird_interact_agents.agents.claude_sdk import _query_v0

    public = {n for n in dir(_query_v0) if not n.startswith("_")}
    assert "query" in public, (
        f"`query` must be in the public surface, got {sorted(public)}"
    )
