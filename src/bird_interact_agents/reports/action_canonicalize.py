"""MCP-tool-name → upstream-canonical action-string mapping.

The bird-interact-tools MCP wrappers (and the harness's top-level
``ask_user`` tool) map to the same short action names upstream's
``eval_react_bird_interact.py`` emits: ``submit``, ``execute``, ``ask``,
``get_schema``, ``get_all_column_meanings``, ``get_column_meaning``,
``get_all_external_knowledge_names``, ``get_knowledge_definition``,
``get_all_knowledge_definitions``. Anything else falls through to
``<tool_name>(<json_args>)`` so the trajectory is preserved verbatim.

Section VI prose names (``ask_user`` / ``submit_sql`` / ``execute_sql``)
are exposed via ``SECTION_VI_NAME_TO_CANONICAL`` for documentation.
"""

from __future__ import annotations

import json
from typing import Any


# (Tool name, canonical name, has-args). Order: most-specific first.
_KNOWN: list[tuple[str, str]] = [
    ("mcp__bird-interact-tools__submit_query", "submit"),
    ("mcp__bird-interact-tools__execute_sql", "execute"),
    ("ask_user", "ask"),
    ("mcp__bird-interact-tools__get_schema", "get_schema"),
    ("mcp__bird-interact-tools__get_all_column_meanings", "get_all_column_meanings"),
    ("mcp__bird-interact-tools__get_column_meaning", "get_column_meaning"),
    (
        "mcp__bird-interact-tools__get_all_external_knowledge_names",
        "get_all_external_knowledge_names",
    ),
    (
        "mcp__bird-interact-tools__get_knowledge_definition",
        "get_knowledge_definition",
    ),
    (
        "mcp__bird-interact-tools__get_all_knowledge_definitions",
        "get_all_knowledge_definitions",
    ),
]

CANONICAL_NAME_MAP: dict[str, str] = dict(_KNOWN)

# Section VI prose ↔ upstream short form. Documented here for clarity.
SECTION_VI_NAME_TO_CANONICAL: dict[str, str] = {
    "ask_user": "ask",
    "submit_sql": "submit",
    "execute_sql": "execute",
}


def _compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=False)


def _sql_from_input(tool_input: dict[str, Any]) -> str:
    """submit_query carries SQL in ``query_json`` (slayer mode) or
    ``query`` (raw mode); execute_sql uses ``query``."""
    for key in ("query_json", "query", "sql"):
        if key in tool_input:
            return str(tool_input[key])
    return _compact(tool_input)


def canonicalize_action(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Return the canonical action string for one tool call.

    * ``submit_query`` / ``execute_sql`` → ``submit(<sql>)`` / ``execute(<sql>)``
    * ``ask_user`` → ``ask(<question>)``
    * Zero-arg helpers → ``<canonical>()``.
    * Arg-bearing helpers (``get_column_meaning`` /
      ``get_knowledge_definition``) → ``<canonical>(<compact_json>)``.
    * Unknown tools → ``<tool_name>(<compact_json>)``.
    """
    if tool_name == "mcp__bird-interact-tools__submit_query":
        return f"submit({_sql_from_input(tool_input)})"
    if tool_name == "mcp__bird-interact-tools__execute_sql":
        return f"execute({_sql_from_input(tool_input)})"
    if tool_name == "ask_user":
        question = tool_input.get("question", _compact(tool_input))
        return f"ask({question})"

    canonical = CANONICAL_NAME_MAP.get(tool_name)
    if canonical is None:
        return f"{tool_name}({_compact(tool_input)})"
    if not tool_input:
        return f"{canonical}()"
    return f"{canonical}({_compact(tool_input)})"


def action_args_string(tool_name: str, tool_input: dict[str, Any]) -> str:
    """The string that ``count_tokens`` measures for ``action_input_tokens``.

    Per the spec: SQL string for submit/execute; question string for
    ask; ``compact json.dumps`` for everything else.
    """
    if tool_name == "mcp__bird-interact-tools__submit_query":
        return _sql_from_input(tool_input)
    if tool_name == "mcp__bird-interact-tools__execute_sql":
        return _sql_from_input(tool_input)
    if tool_name == "ask_user":
        return str(tool_input.get("question", _compact(tool_input)))
    return _compact(tool_input)
