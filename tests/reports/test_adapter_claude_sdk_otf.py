"""Tests for the Claude Agent SDK trajectory → Turn iterator.

Spec (DEV-1553):
* Adapter consumes the SDK-native message stream
  (``SystemMessage``/``AssistantMessage``/``UserMessage``/``ResultMessage``)
  and yields one ``Turn`` per ``tool_use`` block.
* Pure-text/thinking assistant messages WITHOUT a tool_use fold into the
  NEXT tool-using turn — they never appear as a no-op row.
* ``UserMessage.tool_result`` content is paired with the Turn that emitted
  the matching ``tool_use_id``.
* Initial task statement (first ``UserMessage`` text before any
  ``AssistantMessage``) is exposed as the ``prompt`` for turn 0.
"""

from __future__ import annotations

from tests.reports._fixtures import (
    assistant_msg,
    system_msg,
    tool_result_msg,
    tool_use_block,
    user_text_msg,
)


def _walk(steps):
    from bird_interact_agents.reports.adapters.claude_sdk_otf import (
        walk_trajectory,
    )

    return list(walk_trajectory(steps))


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_walk_emits_one_turn_per_tool_use():
    steps = [
        system_msg(),
        user_text_msg(text="Task statement."),
        assistant_msg(
            text="Calling submit.",
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": "SELECT 1"},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_1",
            content="Phase 1 SQL Correct! No Phase 2. Task finished.",
        ),
    ]
    turns = _walk(steps)
    assert len(turns) == 1
    t = turns[0]
    assert t.tool_name == "mcp__bird-interact-tools__submit_query"
    assert t.tool_input == {"query_json": "SELECT 1"}
    assert t.tool_use_id == "tu_1"
    assert "Phase 1 SQL Correct" in t.observation
    assert t.model == "claude-opus-4-7"


def test_walk_first_turn_prompt_is_initial_task_text():
    steps = [
        system_msg(),
        user_text_msg(text="Find rows where x = 1."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__get_schema",
                inp={},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="schema text"),
    ]
    turns = _walk(steps)
    assert turns[0].prompt == "Find rows where x = 1."


# ---------------------------------------------------------------------------
# Pure-text/thinking assistant message folds into next tool-using turn
# ---------------------------------------------------------------------------


def test_walk_folds_pure_text_into_next_turn():
    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        # Pure-text assistant message (no tool_use) — must be folded.
        assistant_msg(thinking="thinking aloud", text="I'll ask the user."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="ask_user",
                inp={"question": "What does 'X' mean?"},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="It means foo."),
    ]
    turns = _walk(steps)
    assert len(turns) == 1
    t = turns[0]
    assert "thinking aloud" in t.response_raw
    assert "I'll ask the user." in t.response_raw
    # The tool_use's own text+thinking from this assistant message is also
    # captured.
    assert t.tool_name == "ask_user"


# ---------------------------------------------------------------------------
# Multi-content tool_result (list-of-text-blocks vs plain string)
# ---------------------------------------------------------------------------


def test_walk_tool_result_with_text_block_list_is_concatenated():
    """SDK can deliver tool_result content as a list of {type:text, text:...}
    blocks instead of a plain string. The adapter must concatenate."""
    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__get_schema",
                inp={},
            ),
        ),
        # Hand-craft a UserMessage whose tool_result content is a list.
        {
            "type": "UserMessage",
            "data": {
                "content": [
                    {
                        "tool_use_id": "tu_1",
                        "type": "tool_result",
                        "content": [
                            {"type": "text", "text": "table_a:\n  col_x int\n"},
                            {"type": "text", "text": "table_b:\n  col_y text\n"},
                        ],
                    }
                ],
                "uuid": "u",
                "parent_tool_use_id": None,
                "tool_use_result": None,
            },
        },
    ]
    turns = _walk(steps)
    assert "table_a" in turns[0].observation
    assert "table_b" in turns[0].observation


# ---------------------------------------------------------------------------
# Turn prompt = concatenation of intervening tool_results + free user text
# ---------------------------------------------------------------------------


def test_walk_turn_prompt_concatenates_intervening_user_inputs():
    """The prompt for turn N is the new text observed since turn N-1's
    tool_use. That includes the matched tool_result AND any free
    UserMessage text that arrived before the next tool_use."""
    steps = [
        system_msg(),
        user_text_msg(text="initial."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__get_schema",
                inp={},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="schema X"),
        # Free user-sim turn between tool_use #1 and tool_use #2.
        user_text_msg(text="Now also consider Y."),
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_2",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": "SELECT * FROM x JOIN y"},
            ),
        ),
        tool_result_msg(
            tool_use_id="tu_2", content="Phase 1 SQL Correct! Task finished."
        ),
    ]
    turns = _walk(steps)
    assert len(turns) == 2
    assert turns[0].prompt == "initial."
    # Turn 2's prompt = tool_result of tu_1 + free user text.
    assert "schema X" in turns[1].prompt
    assert "Now also consider Y." in turns[1].prompt


# ---------------------------------------------------------------------------
# response_raw preserves thinking + text + tool_use JSON
# ---------------------------------------------------------------------------


def test_walk_multiple_tool_uses_in_one_message_share_prompt_and_context():
    """Codex round 7: a single AssistantMessage can carry multiple
    ``tool_use`` blocks. Every emitted Turn must share the SAME prompt
    + thinking + text context (the model produced them all together);
    resetting state after the first tool_use would leave subsequent
    Turns with an empty prompt and missing context."""
    steps = [
        system_msg(),
        user_text_msg(text="Initial task statement."),
        # One assistant message, three content items:
        # [thinking, tool_use A, tool_use B].
        {
            "type": "AssistantMessage",
            "data": {
                "content": [
                    {"thinking": "reasoning across both tools", "signature": "sig"},
                    tool_use_block(
                        tool_use_id="tu_A",
                        name="mcp__bird-interact-tools__get_schema",
                        inp={},
                    ),
                    tool_use_block(
                        tool_use_id="tu_B",
                        name="mcp__bird-interact-tools__get_all_column_meanings",
                        inp={},
                    ),
                ],
                "model": "claude-opus-4-7",
                "parent_tool_use_id": None,
                "error": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "message_id": "msg_dual",
                "stop_reason": "tool_use",
                "session_id": "sess_dual",
            },
        },
        tool_result_msg(tool_use_id="tu_A", content="schema text"),
        tool_result_msg(tool_use_id="tu_B", content="column meanings text"),
    ]
    turns = _walk(steps)
    assert len(turns) == 2
    # BOTH turns carry the same prompt (the initial task statement).
    assert turns[0].prompt == "Initial task statement."
    assert turns[1].prompt == "Initial task statement."
    # BOTH turns carry the thinking content (rendered into response_raw
    # — structural envelope check, not content assertion).
    assert "thinking" in turns[0].response_raw
    assert "thinking" in turns[1].response_raw
    # Each turn gets the correct observation.
    assert "schema" in turns[0].observation
    assert "column meanings" in turns[1].observation
    # Different tool names + ids.
    assert turns[0].tool_use_id == "tu_A"
    assert turns[1].tool_use_id == "tu_B"


def test_walk_response_raw_preserves_thinking_and_tool_use():
    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        assistant_msg(
            thinking="let me think...",
            text="Calling submit.",
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": "SELECT 1"},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="Phase 1 SQL Correct!"),
    ]
    turns = _walk(steps)
    raw = turns[0].response_raw
    assert "let me think..." in raw
    assert "Calling submit." in raw
    assert "submit_query" in raw
    assert "SELECT 1" in raw


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


def test_adapter_registry_accepts_real_cloud_metadata():
    """Real cloud runs persist ``framework='claude_sdk'`` (the CLI flag),
    not the internal class name. The adapter must resolve that combo."""
    from bird_interact_agents.reports.adapters import get_adapter

    walker = get_adapter("claude_sdk", query_mode="slayer")
    assert callable(walker)


def test_adapter_registry_covers_slayer_mode_internal_names():
    """Forward-compat: internal agent names also resolve (in case a future
    run_metadata schema promotes them to persisted fields)."""
    from bird_interact_agents.reports.adapters import get_adapter

    family = (
        "claude_sdk",
        "claude_sdk_otf",
        "claude_sdk_otf_ainteract",
    )
    adapters = [get_adapter(f, query_mode="slayer") for f in family]
    assert all(a is adapters[0] for a in adapters)


def test_adapter_registry_unknown_framework_errors():
    import pytest

    from bird_interact_agents.reports.adapters import get_adapter

    with pytest.raises((KeyError, ValueError)):
        get_adapter("pydantic_ai", query_mode="slayer")


def test_adapter_registry_rejects_raw_query_mode():
    """``query_mode='raw'`` runs persist trajectory `data` as a Python
    repr STRING, not a dict — the SLayer walker would crash. Until a
    string-repr parser lands the lookup must error clearly so the
    failure surfaces at source-resolution time (not mid-walk with a
    confusing AttributeError)."""
    import pytest

    from bird_interact_agents.reports.adapters import get_adapter

    with pytest.raises((KeyError, ValueError)):
        get_adapter("claude_sdk", query_mode="raw")
