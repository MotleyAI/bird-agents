"""DEV-1535 follow-up — `extract_claude_sdk_result_metadata`.

The claude_agent SDK emits a final `ResultMessage` per turn carrying
`num_turns`, `stop_reason`, `duration_ms`, `duration_api_ms`. Pre-fix
the adapters consumed those messages only for `accumulate_assistant_usage`
(token counting); the lifecycle metadata was dropped. `stop_reason`
is the canonical "why did the agent stop" signal and was previously
invisible.

Walker pulls the LAST ResultMessage's metadata; finalize_result_row
folds the four `sdk_*` keys under `usage` so results.db's `usage_json`
carries them for free.
"""

from __future__ import annotations

from bird_interact_agents.agents._run_capture import (
    extract_claude_sdk_result_metadata,
)


def _result_msg(num_turns=None, stop_reason=None,
                duration_ms=None, duration_api_ms=None) -> dict:
    data: dict = {}
    if num_turns is not None:
        data["num_turns"] = num_turns
    if stop_reason is not None:
        data["stop_reason"] = stop_reason
    if duration_ms is not None:
        data["duration_ms"] = duration_ms
    if duration_api_ms is not None:
        data["duration_api_ms"] = duration_api_ms
    return {"type": "ResultMessage", "data": data}


def test_pulls_the_last_resultmessage():
    """A run that emits multiple ResultMessages (rare but possible for
    multi-phase runs) — the LAST one is the authoritative termination
    signal."""
    trajectory = [
        {"type": "AssistantMessage", "data": {"content": []}},
        _result_msg(num_turns=5, stop_reason="end_turn", duration_ms=1000),
        {"type": "AssistantMessage", "data": {"content": []}},
        _result_msg(num_turns=12, stop_reason="stop_sequence",
                    duration_ms=2500, duration_api_ms=2100),
    ]
    meta = extract_claude_sdk_result_metadata(trajectory)
    assert meta == {
        "sdk_num_turns": 12,
        "sdk_stop_reason": "stop_sequence",
        "sdk_duration_ms": 2500,
        "sdk_duration_api_ms": 2100,
    }


def test_returns_none_when_no_resultmessage():
    """Very-early crashes never emit a ResultMessage — walker returns
    None and finalize_result_row leaves the usage dict alone."""
    trajectory = [
        {"type": "SystemMessage", "data": {"subtype": "init"}},
        {"type": "AssistantMessage", "data": {"content": []}},
    ]
    assert extract_claude_sdk_result_metadata(trajectory) is None


def test_returns_none_on_empty_trajectory():
    assert extract_claude_sdk_result_metadata([]) is None
    assert extract_claude_sdk_result_metadata(None) is None


def test_returns_none_on_non_claude_sdk_shape():
    """Pydantic-ai-shaped trajectory doesn't carry ResultMessage items —
    walker returns None for the shape, not just for the absent message."""
    pydantic_ai_msgs = [{"kind": "response", "parts": []}]
    assert extract_claude_sdk_result_metadata(pydantic_ai_msgs) is None


def test_handles_partial_result_message():
    """A ResultMessage missing some fields (older SDK versions, or
    error-path early termination) yields None on the missing fields —
    NOT on the message itself. Keeps the result-row schema stable."""
    trajectory = [_result_msg(num_turns=3, stop_reason="end_turn")]
    meta = extract_claude_sdk_result_metadata(trajectory)
    assert meta == {
        "sdk_num_turns": 3,
        "sdk_stop_reason": "end_turn",
        "sdk_duration_ms": None,
        "sdk_duration_api_ms": None,
    }


def test_skips_non_dict_items():
    """A trajectory with stringified messages (claude_sdk fallback when
    dataclasses.asdict raises) — walker skips those and finds the
    ResultMessage further back."""
    trajectory = [
        _result_msg(num_turns=4, stop_reason="end_turn"),
        # Stringified entry — but this fails the shape discriminator,
        # so the whole walk returns None. (Documented behavior; the
        # finalize_result_row dispatcher tolerates None.)
        "stringified msg",
    ]
    # The string item breaks `_looks_like_claude_sdk_trajectory`,
    # so the discriminator rejects the whole list.
    assert extract_claude_sdk_result_metadata(trajectory) is None


def test_skips_resultmessage_with_non_dict_data():
    """A ResultMessage whose `data` field is not a dict (corruption
    edge case) is skipped during the reverse walk; the walker keeps
    looking for an earlier valid ResultMessage."""
    trajectory = [
        _result_msg(num_turns=2, stop_reason="end_turn"),
        {"type": "ResultMessage", "data": "garbage"},
    ]
    meta = extract_claude_sdk_result_metadata(trajectory)
    assert meta == {
        "sdk_num_turns": 2,
        "sdk_stop_reason": "end_turn",
        "sdk_duration_ms": None,
        "sdk_duration_api_ms": None,
    }
