"""Per-task tool-call stats extracted from PydanticAI's message history.

The harness records call counts and validation/retry errors per tool name
so offline failure analysis can spot patterns like "agent never called
inspect_model" or "10 retry-prompts on submit_query — args kept failing
validation". Tests here pin the extractor's contract against synthesised
message histories that mimic PydanticAI's runtime shapes.
"""

from __future__ import annotations

from types import SimpleNamespace

from bird_interact_agents.agents.pydantic_ai.agent import _extract_tool_stats


class _FakeRun:
    def __init__(self, messages):
        self._messages = messages

    def all_messages(self):
        return self._messages


def _msg(*parts):
    """Wrap parts in a stand-in message — `_extract_tool_stats` reads
    `getattr(msg, 'parts', None)` so any object with a `parts` attribute
    works."""
    return SimpleNamespace(parts=list(parts))


def _tool_call(tool_name: str):
    return SimpleNamespace(part_kind="tool-call", tool_name=tool_name)


def _retry(tool_name: str | None, content: str):
    return SimpleNamespace(part_kind="retry-prompt", tool_name=tool_name,
                           content=content)


def _text(text: str):
    return SimpleNamespace(part_kind="text", content=text)


def test_extract_counts_calls_per_tool_name():
    run = _FakeRun([
        _msg(_tool_call("models_summary")),
        _msg(_tool_call("inspect_model"), _tool_call("inspect_model")),
        _msg(_tool_call("submit_query")),
    ])
    stats = _extract_tool_stats(run)
    assert stats is not None
    by_tool = {row["tool"]: row for row in stats["per_tool"]}
    assert by_tool["inspect_model"]["n_calls"] == 2
    assert by_tool["models_summary"]["n_calls"] == 1
    assert by_tool["submit_query"]["n_calls"] == 1
    assert stats["total_calls"] == 4
    assert stats["total_errors"] == 0


def test_extract_counts_retry_prompts_as_errors():
    run = _FakeRun([
        _msg(_tool_call("submit_query")),
        # PydanticAI sends a retry-prompt back to the model when args
        # don't validate — this is the cleanest "tool errored" signal.
        _msg(_retry("submit_query", "validation error: missing measures")),
        _msg(_tool_call("submit_query")),
        _msg(_retry("submit_query", "validation error: bad source_model")),
    ])
    stats = _extract_tool_stats(run)
    by_tool = {row["tool"]: row for row in stats["per_tool"]}
    assert by_tool["submit_query"] == {
        "tool": "submit_query", "n_calls": 2, "n_errors": 2,
    }
    assert stats["total_errors"] == 2
    assert {s["error"] for s in stats["error_samples"]} == {
        "validation error: missing measures",
        "validation error: bad source_model",
    }


def test_extract_handles_missing_tool_retry():
    """If the model invents a tool name, PydanticAI emits a retry with
    the offending tool_name. Tools that ONLY appear as errors must still
    show up in `per_tool` with n_calls=0."""
    run = _FakeRun([
        _msg(_tool_call("query")),
        _msg(_retry("get_all_external_knowledge_names",
                    "Tool 'get_all_external_knowledge_names' not found")),
    ])
    stats = _extract_tool_stats(run)
    by_tool = {row["tool"]: row for row in stats["per_tool"]}
    assert by_tool["get_all_external_knowledge_names"] == {
        "tool": "get_all_external_knowledge_names",
        "n_calls": 0, "n_errors": 1,
    }
    assert by_tool["query"]["n_calls"] == 1


def test_extract_caps_error_samples():
    """`error_samples` is capped at 10 to keep results.db rows compact."""
    parts = [_retry("submit_query", f"err {i}") for i in range(25)]
    run = _FakeRun([_msg(*parts)])
    stats = _extract_tool_stats(run)
    assert stats["total_errors"] == 25
    assert len(stats["error_samples"]) == 10


def test_extract_ignores_non_tool_parts():
    """ThinkingPart / TextPart / SystemPromptPart shouldn't count."""
    run = _FakeRun([
        _msg(_text("planning my approach")),
        _msg(SimpleNamespace(part_kind="thinking", content="hmm")),
        _msg(_tool_call("inspect_model")),
    ])
    stats = _extract_tool_stats(run)
    assert stats["total_calls"] == 1
    assert stats["per_tool"] == [
        {"tool": "inspect_model", "n_calls": 1, "n_errors": 0},
    ]


def test_extract_returns_none_when_history_unwalkable():
    """Defensive: if all_messages() raises, return None rather than crash."""
    class _Boom:
        def all_messages(self):
            raise RuntimeError("boom")
    assert _extract_tool_stats(_Boom()) is None


def test_results_db_round_trips_tool_call_stats_json(tmp_path):
    import json
    from bird_interact_agents.results_db import (
        TaskResultRow, insert_task_result, open_db,
    )

    blob = json.dumps({
        "per_tool": [{"tool": "inspect_model", "n_calls": 3, "n_errors": 0}],
        "total_calls": 3, "total_errors": 0, "error_samples": [],
    })
    conn = open_db(tmp_path / "results.db")
    insert_task_result(conn, TaskResultRow(
        run_id="r", framework="pydantic_ai", mode="a-interact",
        query_mode="slayer", instance_id="x", database="d",
        started_at=0.0, duration_s=0.0,
        phase1_passed=False, phase2_passed=False, total_reward=0.0,
        tool_call_stats_json=blob,
    ))
    rows = list(conn.execute(
        "SELECT instance_id, tool_call_stats_json FROM task_results"
    ))
    assert rows == [("x", blob)]
