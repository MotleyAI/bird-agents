"""Tests for the MCP-tool-name → upstream-canonical action-string mapping.

Spec (DEV-1553):
* bird-interact-tools wrappers map to upstream-canonical names: ``ask``,
  ``submit``, ``execute``, ``get_schema``, ``get_all_column_meanings``,
  ``get_column_meaning``, ``get_all_external_knowledge_names``,
  ``get_knowledge_definition``, ``get_all_knowledge_definitions``.
* Unknown MCP tools fall through to ``<tool_name>(<json_args>)``.
* The mapping is the SINGLE source of truth for the leaderboard's
  Section VI cost classifier (``ask``/``submit``/``execute`` are fixed).
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Upstream-canonical mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,tool_input,expected_action",
    [
        # Submit-query — SQL is captured from `query_json` (slayer mode pipes
        # the canonicalized SQL into the wrapper) or from `query` (raw mode).
        (
            "mcp__bird-interact-tools__submit_query",
            {"query_json": "SELECT 1"},
            "submit(SELECT 1)",
        ),
        (
            "mcp__bird-interact-tools__submit_query",
            {"query": "SELECT 2"},
            "submit(SELECT 2)",
        ),
        # Execute-sql (raw mode only)
        (
            "mcp__bird-interact-tools__execute_sql",
            {"query": "SELECT * FROM t"},
            "execute(SELECT * FROM t)",
        ),
        # ask_user (a top-level tool the claude_sdk_otf_* family exposes)
        ("ask_user", {"question": "what is X?"}, "ask(what is X?)"),
        # Zero-arg helpers
        (
            "mcp__bird-interact-tools__get_schema",
            {},
            "get_schema()",
        ),
        (
            "mcp__bird-interact-tools__get_all_column_meanings",
            {},
            "get_all_column_meanings()",
        ),
        (
            "mcp__bird-interact-tools__get_all_external_knowledge_names",
            {},
            "get_all_external_knowledge_names()",
        ),
        (
            "mcp__bird-interact-tools__get_all_knowledge_definitions",
            {},
            "get_all_knowledge_definitions()",
        ),
        # Arg-bearing helpers — args are serialized as compact json.dumps.
        (
            "mcp__bird-interact-tools__get_column_meaning",
            {"table_name": "users", "column_name": "id"},
            'get_column_meaning({"table_name":"users","column_name":"id"})',
        ),
        (
            "mcp__bird-interact-tools__get_knowledge_definition",
            {"name": "kb_007"},
            'get_knowledge_definition({"name":"kb_007"})',
        ),
    ],
)
def test_canonicalize_known_tool(tool_name, tool_input, expected_action):
    from bird_interact_agents.reports.action_canonicalize import (
        canonicalize_action,
    )

    assert canonicalize_action(tool_name, tool_input) == expected_action


# ---------------------------------------------------------------------------
# Unknown MCP tools — fall through to raw <name>(<json>)
# ---------------------------------------------------------------------------


def test_canonicalize_unknown_tool_passes_through():
    from bird_interact_agents.reports.action_canonicalize import (
        canonicalize_action,
    )

    result = canonicalize_action(
        "mcp__slayer__search", {"entities": ["x"], "max_memories": 0}
    )
    assert result.startswith("mcp__slayer__search(")
    args_str = result[len("mcp__slayer__search(") : -1]
    assert json.loads(args_str) == {"entities": ["x"], "max_memories": 0}


def test_canonicalize_unknown_zero_arg_tool():
    from bird_interact_agents.reports.action_canonicalize import (
        canonicalize_action,
    )

    assert canonicalize_action("mcp__slayer__list_datasources", {}) == (
        "mcp__slayer__list_datasources({})"
    )


# ---------------------------------------------------------------------------
# Codex finding #7 — Section VI uses prose names ``ask_user`` / ``submit_sql``
# / ``execute_sql`` while upstream eval_react uses the short forms
# ``ask`` / ``submit`` / ``execute``. We pick the upstream short form and
# expose the mapping for documentation.
# ---------------------------------------------------------------------------


def test_section_vi_action_names_mapping_documented():
    from bird_interact_agents.reports.action_canonicalize import (
        SECTION_VI_NAME_TO_CANONICAL,
    )

    assert SECTION_VI_NAME_TO_CANONICAL == {
        "ask_user": "ask",
        "submit_sql": "submit",
        "execute_sql": "execute",
    }
