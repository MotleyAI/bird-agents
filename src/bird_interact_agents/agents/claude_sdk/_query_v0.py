"""DEV-1555 v0/v1 split — origin/main ``query`` MCP tool.

The v1 (current-branch) ``claude_sdk/agent.py:query`` exposes the SLayer
MCP signature directly (positional ``source_model`` / ``dimensions`` /
``measures`` / ``filters`` / … fields, with ``source_model`` required).
This module preserves the v0 (origin/main) interface where the agent
passes ONE ``query_json`` string — a stringified ``SlayerQuery`` JSON
object — alongside the legacy tool-level kwargs ``show_sql`` /
``dry_run`` / ``explain`` / ``format`` / ``normalize_filters``.

v0 agents (``claude_sdk_otf`` and ``claude_sdk_otf_ainteract``) import
``query`` from this module; the registered MCP tool name remains
``"query"`` (the @tool decorator's first arg), so v0 prompts that say
"call ``query`` with ``query_json``" still resolve.

The implementation defers to ``claude_sdk.agent``'s shared SLayer client
+ storage plumbing (the ``_ctx`` proxy, ``_slayer_client``,
``_query_mod``, ``_text``). The v1 changes to those helpers preserve
backward compatibility — they didn't touch the ``query_impl`` signature
or the ``_ctx`` shape.
"""

from __future__ import annotations

from claude_agent_sdk import tool

from bird_interact_agents.agents import _query as _query_mod
from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx,
    _slayer_client,
    _text,
)


_QUERY_TOOL_DESC = (
    "Run a single-stage SLayer query and return SLayer's formatted "
    "result. `query_json` is a SlayerQuery JSON OBJECT — the same "
    "shape `submit_query` accepts — with `source_model` (required), "
    "and any of `dimensions`, `measures`, `filters`, `time_dimensions`, "
    "`order`, `limit`, `offset`, `whole_periods_only`, `variables`, "
    "`distinct_dimension_values`. Set `distinct_dimension_values: "
    "false` inside the JSON to disable SLayer's default dim-only "
    "auto-dedup `GROUP BY` (emits raw `SELECT <dims/td>` rows). "
    "Tool-level options stay OUTSIDE the JSON as separate kwargs: "
    "`show_sql`, `dry_run`, `explain`, `format` (markdown/json/csv), "
    "and `normalize_filters` (default true) — when true, every "
    "`col == 'X'` filter becomes `lower(trim(col)) == 'x'` "
    "(case/whitespace-tolerant); when false, filters are forwarded "
    "verbatim (exact-case equality). For a nested-DAG preview use "
    "`query_nested` instead."
)


@tool(
    "query",
    _QUERY_TOOL_DESC,
    # Explicit JSON Schema dict so only `query_json` is required. A flat
    # `{key: type}` schema would make the SDK mark every key required
    # (claude_agent_sdk._build_schema → `"required":
    # list(properties.keys())`), forcing every preview call to set
    # show_sql/dry_run/explain/format.
    {
        "type": "object",
        "properties": {
            "query_json": {"type": "string"},
            "show_sql": {"type": "boolean", "default": False},
            "dry_run": {"type": "boolean", "default": False},
            "explain": {"type": "boolean", "default": False},
            "format": {"type": "string", "default": "markdown"},
            "normalize_filters": {"type": "boolean", "default": True},
        },
        "required": ["query_json"],
    },
)
async def query(args: dict) -> dict:
    storage = _ctx.get("_slayer_storage")
    if storage is None:
        _slayer_client()
        storage = _ctx["_slayer_storage"]
    _query_mod.attach_storage(storage)

    result = await _query_mod.query_impl(
        args["query_json"],
        show_sql=bool(args.get("show_sql", False)),
        dry_run=bool(args.get("dry_run", False)),
        explain=bool(args.get("explain", False)),
        format=args.get("format", "markdown"),
        normalize_filters=bool(args.get("normalize_filters", True)),
    )
    return _text(result if isinstance(result, str) else str(result))
