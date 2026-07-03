"""DEV-1535 follow-up — `extract_tool_stats_from_claude_sdk_trajectory`.

Pre-fix only the pydantic_ai* family produced `tool_call_stats`; every
claude_sdk* adapter left it None, so per-tool error analyses (a major
input to failure-mode bucketing) had to walk the raw trajectory ad-hoc.
This walker centralises that work and the finalize_result_row chokepoint
backfills it for every claude_sdk run.

Output shape MUST match `_run_capture._extract_tool_stats` exactly so
downstream consumers (cascading reports, regrade scripts) don't fork.
"""

from __future__ import annotations

from bird_interact_agents.agents._run_capture import (
    extract_tool_stats_from_claude_sdk_trajectory,
)


def _tool_use(name: str, use_id: str) -> dict:
    return {"type": "tool_use", "name": name, "id": use_id, "input": {}}


def _tool_result(use_id: str, text: str, *, is_error: bool) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "is_error": is_error,
        "content": [{"type": "text", "text": text}],
    }


def _assistant(content: list[dict]) -> dict:
    return {"type": "AssistantMessage", "data": {"content": content}}


def _user(content: list[dict]) -> dict:
    return {"type": "UserMessage", "data": {"content": content}}


def test_counts_tool_use_blocks_by_name():
    """Two `execute_sql` calls + one `get_schema` call → ordered output
    descending by n_calls, then alphabetical on tie."""
    trajectory = [
        _assistant([
            _tool_use("get_schema", "t1"),
            _tool_use("execute_sql", "t2"),
        ]),
        _user([_tool_result("t1", "schema...", is_error=False)]),
        _user([_tool_result("t2", "rows...", is_error=False)]),
        _assistant([_tool_use("execute_sql", "t3")]),
        _user([_tool_result("t3", "rows...", is_error=False)]),
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_calls"] == 3
    assert stats["total_errors"] == 0
    assert stats["per_tool"] == [
        {"tool": "execute_sql", "n_calls": 2, "n_errors": 0},
        {"tool": "get_schema", "n_calls": 1, "n_errors": 0},
    ]
    assert stats["error_samples"] == []


def test_counts_errors_via_tool_use_id_map():
    """`is_error=True` ToolResults are resolved back to their tool name
    via the tool_use_id → name map built from AssistantMessage tool_use
    blocks."""
    trajectory = [
        _assistant([_tool_use("execute_sql", "t1")]),
        _user([_tool_result("t1", "syntax error near 'FROM'", is_error=True)]),
        _assistant([_tool_use("execute_sql", "t2")]),
        _user([_tool_result("t2", "ok", is_error=False)]),
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_calls"] == 2
    assert stats["total_errors"] == 1
    assert stats["per_tool"] == [
        {"tool": "execute_sql", "n_calls": 2, "n_errors": 1},
    ]
    assert stats["error_samples"] == [
        {"tool": "execute_sql", "error": "syntax error near 'FROM'"},
    ]


def test_unresolved_tool_use_id_bucketed_under_unknown():
    """A tool_result whose tool_use_id wasn't seen in any AssistantMessage
    is bucketed under `<unknown>` — defensive against the rare case
    where the trajectory is truncated mid-stream."""
    trajectory = [
        _user([_tool_result("orphan", "oops", is_error=True)]),
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_errors"] == 1
    assert stats["per_tool"] == [
        {"tool": "<unknown>", "n_calls": 0, "n_errors": 1},
    ]


def test_returns_none_on_empty_trajectory():
    assert extract_tool_stats_from_claude_sdk_trajectory([]) is None
    assert extract_tool_stats_from_claude_sdk_trajectory(None) is None


def test_returns_none_on_non_claude_sdk_shape():
    """Pydantic-ai-shaped trajectory: list-of-message-dicts without the
    `{"type":..., "data":...}` wrapper. Walker returns None so the
    dispatcher in finalize_result_row routes to the pydantic_ai path."""
    pydantic_ai_msgs = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt"}]},
        {"kind": "response", "parts": [{"part_kind": "tool-call"}]},
    ]
    assert extract_tool_stats_from_claude_sdk_trajectory(pydantic_ai_msgs) is None


def test_returns_none_on_claude_sdk_raw_string_data_shape():
    """DEV-1535 r2 (Codex): the `claude_sdk_otf*_raw` adapters
    serialize each trajectory item as `{"type": ..., "data": str(msg)[:500]}`
    — `data` is a STRING, not a dataclass dict. Pre-fix the discriminator
    only checked `"data" in item` and accepted these; the walker then
    skipped every non-dict `data` and falsely returned an empty stats
    dict ('0 calls / 0 errors'). Tightened discriminator rejects the
    raw shape — finalize_result_row leaves tool_call_stats absent
    rather than fabricating a misleading-zero record."""
    raw_trajectory = [
        {"type": "AssistantMessage",
         "data": "AssistantMessage(content=[...], parent_tool_use_id=None)"},
        {"type": "ResultMessage",
         "data": "ResultMessage(subtype='success', duration_ms=1234)"},
    ]
    assert extract_tool_stats_from_claude_sdk_trajectory(raw_trajectory) is None


def test_returns_none_when_any_item_has_non_dict_data():
    """Mixed shapes (one dict, one string) also fail the discriminator
    — better to skip the whole walk than to report partial stats from
    only the dict items."""
    mixed = [
        {"type": "AssistantMessage", "data": {"content": []}},
        {"type": "ResultMessage", "data": "stringified"},
    ]
    assert extract_tool_stats_from_claude_sdk_trajectory(mixed) is None


def test_caps_error_samples_at_10():
    """Many errors → error_samples list capped at 10 (mirrors
    `_TOOL_ERROR_SAMPLES_PER_TASK` in the pydantic_ai sibling)."""
    trajectory = []
    for i in range(15):
        trajectory.append(_assistant([_tool_use("execute_sql", f"t{i}")]))
        trajectory.append(_user([_tool_result(f"t{i}", f"err {i}", is_error=True)]))
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_errors"] == 15
    assert len(stats["error_samples"]) == 10


def test_caps_error_sample_text_at_400_chars():
    """Individual error text is truncated to 400 chars."""
    long_err = "X" * 5000
    trajectory = [
        _assistant([_tool_use("execute_sql", "t1")]),
        _user([_tool_result("t1", long_err, is_error=True)]),
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert len(stats["error_samples"][0]["error"]) == 400


def test_string_content_in_tool_result_is_used_as_is():
    """Some SDK paths emit ToolResult with `content` as a bare string
    rather than a list of text blocks — handled without crash."""
    trajectory = [
        _assistant([_tool_use("execute_sql", "t1")]),
        {
            "type": "UserMessage",
            "data": {"content": [{
                "type": "tool_result",
                "tool_use_id": "t1",
                "is_error": True,
                "content": "raw string error",
            }]},
        },
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["error_samples"] == [
        {"tool": "execute_sql", "error": "raw string error"},
    ]


def test_successful_tool_results_dont_count_as_errors():
    """`is_error=False` tool_results are not counted in n_errors."""
    trajectory = [
        _assistant([_tool_use("execute_sql", "t1")]),
        _user([_tool_result("t1", "rows...", is_error=False)]),
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_calls"] == 1
    assert stats["total_errors"] == 0


def test_tool_use_without_explicit_id_still_counted():
    """A tool_use block missing the `id` field still bumps n_calls
    (the call happened); just no error attribution path."""
    trajectory = [
        {
            "type": "AssistantMessage",
            "data": {"content": [{"type": "tool_use", "name": "execute_sql"}]},
        },
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_calls"] == 1


# ---------------------------------------------------------------------------
# REAL production serialization shape (regression for the all-zero bug).
#
# The fixtures above all hand-write `"type": "tool_use"` / `"tool_result"`,
# but the trajectory serializer uses `dataclasses.asdict(msg)` and the SDK's
# ToolUseBlock / ToolResultBlock dataclasses have NO `type` field — so the
# real production trajectory never carries `type` on its content blocks and
# the walker silently counted ZERO tool calls (empty stats in local AND cloud
# results.db). These build fixtures from the ACTUAL SDK dataclasses via
# `dataclasses.asdict`, exactly like claude_sdk_otf/agent.py, so they break if
# the SDK block shape ever regresses.
# ---------------------------------------------------------------------------


def _asdict_blocks(blocks: list) -> list[dict]:
    import dataclasses

    return [dataclasses.asdict(b) for b in blocks]


def test_real_asdict_shape_counts_tool_calls():
    """The real `dataclasses.asdict(ToolUseBlock)` shape ({id,name,input},
    no `type`) must be counted — this is the bug that produced empty stats."""
    from claude_agent_sdk.types import (
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    assistant = _asdict_blocks([
        ToolUseBlock(id="t1", name="recommend_root_model", input={"items": ["a.b"]}),
        TextBlock(text="reasoning..."),
        ToolUseBlock(id="t2", name="inspect", input={"reference": "db.m.c"}),
    ])
    user = _asdict_blocks([
        ToolResultBlock(tool_use_id="t1", content=[{"type": "text", "text": "ok"}],
                        is_error=False),
        ToolResultBlock(tool_use_id="t2", content=[{"type": "text", "text": "boom"}],
                        is_error=True),
    ])
    # Sanity: the real asdict shape truly has NO `type` on the blocks.
    assert all("type" not in b for b in assistant)
    assert all("type" not in b for b in user)

    trajectory = [
        {"type": "AssistantMessage", "data": {"content": assistant}},
        {"type": "UserMessage", "data": {"content": user}},
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_calls"] == 2
    assert stats["total_errors"] == 1
    assert stats["per_tool"] == [
        {"tool": "inspect", "n_calls": 1, "n_errors": 1},
        {"tool": "recommend_root_model", "n_calls": 1, "n_errors": 0},
    ]
    assert stats["error_samples"] == [{"tool": "inspect", "error": "boom"}]


def test_real_asdict_text_block_not_counted_as_tool():
    """A `dataclasses.asdict(TextBlock)` ({text}) must NOT be miscounted as
    a tool call by the structural detector."""
    from claude_agent_sdk.types import TextBlock

    trajectory = [
        {"type": "AssistantMessage",
         "data": {"content": _asdict_blocks([TextBlock(text="just prose")])}},
    ]
    stats = extract_tool_stats_from_claude_sdk_trajectory(trajectory)
    assert stats is not None
    assert stats["total_calls"] == 0
    assert stats["per_tool"] == []
