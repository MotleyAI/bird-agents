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
    # ...and the OTHER mode's FINALIZATION tool does NOT bleed in. (CodeRabbit
    # PR #56: the slayer branch previously checked ``execute_sql`` — the other
    # mode's *verify* tool — not its finalization tool ``submit_sql``.) We do
    # NOT also assert the other mode's verify tool is absent: the raw verify
    # tool is ``query``, which appears as a common English word in the note's
    # prose ("run the exact query you intend to submit"), so that check would
    # be a guaranteed false positive.
    other_submit = "submit_sql" if query_mode == "slayer" else "submit_query"
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
