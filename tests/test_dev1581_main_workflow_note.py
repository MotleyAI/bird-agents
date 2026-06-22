"""DEV-1581 R2: mechanical contract for ``build_main_workflow_note``.

Per the repo rule (no prompt anchor-phrase / coverage assertions), this pins
ONLY mechanical properties:

* valid ``query_mode`` returns a non-empty string with no unfilled ``{...}``
  format placeholders (placeholder-coverage);
* the mode-appropriate verify/submit tool names are substituted in
  (supplied-value substitution — ``query``/``submit_query`` for slayer,
  ``execute_sql``/``submit_sql`` for raw);
* an unknown ``query_mode`` raises ``ValueError``.

Prompt *wording* (that it steers toward ``ask_discovery``, anti-thrash, etc.)
is validated by cloud smoke, not here.
"""

from __future__ import annotations

import re

import pytest

from bird_interact_agents.agents.claude_sdk.partition import (
    build_main_workflow_note,
)


@pytest.mark.parametrize(
    "query_mode,verify_tool,submit_tool",
    [
        ("slayer", "query", "submit_query"),
        ("raw", "execute_sql", "submit_sql"),
    ],
)
def test_note_substitutes_mode_tool_names(query_mode, verify_tool, submit_tool):
    note = build_main_workflow_note(query_mode=query_mode)
    assert isinstance(note, str) and note.strip()
    # supplied-value substitution: the mode's own tool names appear.
    assert verify_tool in note
    assert submit_tool in note
    # ...and the OTHER mode's finalization tool does NOT (no cross-mode bleed).
    other_submit = "execute_sql" if query_mode == "slayer" else "submit_query"
    assert other_submit not in note


@pytest.mark.parametrize("query_mode", ["slayer", "raw"])
def test_note_has_no_unfilled_placeholders(query_mode):
    note = build_main_workflow_note(query_mode=query_mode)
    # No stray ``{placeholder}`` left after formatting.
    leftover = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", note)
    assert not leftover, f"unfilled placeholders: {leftover}"


def test_unknown_query_mode_raises():
    with pytest.raises(ValueError):
        build_main_workflow_note(query_mode="nonsense")
