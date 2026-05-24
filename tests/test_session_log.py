"""Tests for ``agents._session_log`` (DEV-1454 per-sub-agent session logging).

The goal of the module is that ANY single sub-agent's session is trivially
isolatable — one ``.md`` file with a header + a top-to-bottom tool-call trace,
one ``.json`` sidecar with the raw history, and an ``INDEX.md`` for triage.
These tests pin that behaviour with framework-only fakes (no LLM/MCP).
"""

from __future__ import annotations

from bird_interact_agents.agents import _session_log as sl


def _messages() -> list[dict]:
    """A realistic serialised history: user prompt → tool call → return →
    a retried (erroring) tool call → text."""
    return [
        {"parts": [{"part_kind": "user-prompt", "content": "encode kb 9"}]},
        {"parts": [
            {"part_kind": "text", "content": "I'll search first."},
            {"part_kind": "tool-call", "tool_name": "search",
             "args": "{'question': 'social support'}"},
        ]},
        {"parts": [{"part_kind": "tool-return", "tool_name": "search",
                    "outcome": "success", "content": "memory households_kb_9"}]},
        {"parts": [{"part_kind": "tool-call", "tool_name": "edit_model",
                    "args": "{'model_name': 'service_types'}"}]},
        {"parts": [{"part_kind": "retry-prompt", "tool_name": "edit_model",
                    "content": "VALIDATION FAILED: near 'ILIKE': syntax error"}]},
        {"parts": [{"part_kind": "text", "content": "Fixing to use LOWER()."}]},
    ]


def test_write_session_renders_header_and_step_trace(tmp_path):
    row = sl.write_session(
        tmp_path, "setup__kb_009",
        messages=_messages(),
        tool_call_stats={
            "total_calls": 2, "total_errors": 1,
            "per_tool": [
                {"tool": "edit_model", "n_calls": 1, "n_errors": 1},
                {"tool": "search", "n_calls": 1, "n_errors": 0},
            ],
        },
        n_turns=3, role="setup_encoder", meta={"kb_id": 9},
        status="deferred", output={"status": "deferred"}, error=None,
    )
    md = (tmp_path / "setup__kb_009.md").read_text()
    # Header carries the thrashing fingerprint.
    assert "role: setup_encoder" in md
    assert "kb_id: 9" in md
    assert "status: deferred" in md
    assert "tool calls: 2 | tool errors: 1" in md
    assert "edit_model: 1 calls, 1 errors" in md
    # Step trace renders each part kind, in order.
    assert "→ TOOL CALL `search`" in md
    assert "← return `search`" in md
    assert "→ TOOL CALL `edit_model`" in md
    assert "⚠️ RETRY/ERROR `edit_model`" in md
    assert "near 'ILIKE': syntax error" in md
    # The user prompt appears before the edit_model call (ordering preserved).
    # Use the trace marker — "edit_model" also appears in the header stats.
    assert md.index("USER") < md.index("→ TOOL CALL `edit_model`")
    # JSON sidecar holds the raw history.
    import json
    raw = json.loads((tmp_path / "setup__kb_009.json").read_text())
    assert len(raw) == len(_messages())
    # Index row summarises it.
    assert row["session_id"] == "setup__kb_009"
    assert row["kb_id"] == 9
    assert row["status"] == "deferred"
    assert row["tool_calls"] == 2 and row["tool_errors"] == 1


def test_write_session_with_error_and_no_messages(tmp_path):
    """A failed run (e.g. UsageLimitExceeded) with no captured messages still
    yields a session file carrying the error in the header."""
    row = sl.write_session(
        tmp_path, "setup__kb_010",
        messages=[], tool_call_stats=None, n_turns=None,
        role="setup_encoder", meta={"kb_id": 10}, status="error",
        output={"status": "error"},
        error="UsageLimitExceeded: request_limit of 120",
    )
    md = (tmp_path / "setup__kb_010.md").read_text()
    assert "status: error" in md
    assert "UsageLimitExceeded: request_limit of 120" in md
    assert "_(no messages captured)_" in md
    assert row["status"] == "error"
    assert "UsageLimitExceeded" in row["error"]


def test_session_from_run_extracts_parts():
    """session_from_run pulls messages/stats/turns off any object exposing
    all_messages(); None is handled."""
    class _Run:
        def all_messages(self):
            return []

    parts = sl.session_from_run(_Run())
    assert parts == {"messages": [], "tool_call_stats": {
        "per_tool": [], "total_calls": 0, "total_errors": 0, "error_samples": [],
    }, "n_turns": 0}
    none_parts = sl.session_from_run(None)
    assert none_parts == {"messages": [], "tool_call_stats": None, "n_turns": None}


def test_write_index_lists_all_sessions(tmp_path):
    rows = [
        {"session_id": "setup__kb_001", "role": "setup_encoder", "kb_id": 1,
         "status": "encoded", "n_turns": 4, "tool_calls": 6, "tool_errors": 0,
         "error": ""},
        {"session_id": "setup__kb_010", "role": "setup_encoder", "kb_id": 10,
         "status": "error", "n_turns": 60, "tool_calls": 118, "tool_errors": 40,
         "error": "UsageLimitExceeded"},
    ]
    sl.write_index(tmp_path, rows)
    idx = (tmp_path / "INDEX.md").read_text()
    assert "2 session(s)." in idx
    assert "setup__kb_001" in idx and "setup__kb_010" in idx
    # The thrashing item's high counts are visible in the triage table.
    assert "118" in idx and "UsageLimitExceeded" in idx


def test_write_session_never_raises_on_bad_input(tmp_path):
    """Best-effort: malformed messages must not blow up the run."""
    row = sl.write_session(
        tmp_path, "weird", messages=[{"parts": [{"part_kind": "mystery",
                                                  "blob": object()}]}],
        role="x", status="ok",
    )
    assert (tmp_path / "weird.md").exists()
    assert row["session_id"] == "weird"
