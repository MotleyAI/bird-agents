"""Local/cloud parity: local runs must persist the raw per-turn trajectory
(and the tool_call_stats derived from it) the SAME way cloud runs do.

The cloud actor writes the finalized row (incl. `trajectory` + `tool_call_stats`)
to `runs/<run_id>/rows/<iid>/attempt-<n>.json` via `_gcs.write_row`. Locally,
`run.py::write_local_attempt_row` writes the identical `rows/<iid>/attempt-1.json`
so `submission.trajectory_path` resolves and downstream consumers (autopsy
regen, tool-stats analysis) get the same data. Before this, the file was never
written locally and the trajectory was discarded on process exit.
"""

from __future__ import annotations

import json

from bird_interact_agents.agents._run_capture import (
    extract_tool_stats_from_claude_sdk_trajectory,
)
from bird_interact_agents.run import write_local_attempt_row


def _row_with_trajectory() -> dict:
    return {
        "instance_id": "households_10",
        "database": "households",
        "trajectory": [
            {"type": "AssistantMessage", "data": {"content": [
                {"id": "t1", "name": "recommend_root_model", "input": {"items": ["a.b"]}},
            ]}},
            {"type": "UserMessage", "data": {"content": [
                {"tool_use_id": "t1", "content": [{"type": "text", "text": "ok"}],
                 "is_error": False},
            ]}},
        ],
        "tool_call_stats": {
            "per_tool": [{"tool": "recommend_root_model", "n_calls": 1, "n_errors": 0}],
            "total_calls": 1, "total_errors": 0, "error_samples": [],
        },
        "predicted_row_count": 3,
    }


def test_write_local_attempt_row_creates_file(tmp_path):
    rows_dir = tmp_path / "rows"
    write_local_attempt_row(rows_dir, "households_10", _row_with_trajectory())
    dest = rows_dir / "households_10" / "attempt-1.json"
    assert dest.exists(), "local run must write the same attempt-1.json cloud writes"


def test_written_row_round_trips_trajectory_and_stats(tmp_path):
    """The persisted attempt row carries the raw trajectory AND the
    tool_call_stats, and the trajectory re-derives the same stats — i.e. a
    local run captures exactly what cloud captures."""
    rows_dir = tmp_path / "rows"
    row = _row_with_trajectory()
    write_local_attempt_row(rows_dir, "households_10", row)
    loaded = json.loads((rows_dir / "households_10" / "attempt-1.json").read_text())

    assert loaded["trajectory"] == row["trajectory"]
    assert loaded["tool_call_stats"] == row["tool_call_stats"]
    # The trajectory in the file is enough to reconstruct the stats offline
    # (the cloud collation / autopsy-regen path relies on exactly this).
    rederived = extract_tool_stats_from_claude_sdk_trajectory(loaded["trajectory"])
    assert rederived["total_calls"] == 1
    assert rederived["per_tool"] == [
        {"tool": "recommend_root_model", "n_calls": 1, "n_errors": 0},
    ]


def test_trajectory_path_matches_annotation_convention(tmp_path):
    """The written path must match the `rows/<iid>/attempt-1.json` string the
    local grader stamps into `submission.trajectory_path` (run.py), so the
    annotation's pointer actually resolves."""
    rows_dir = tmp_path / "rows"
    write_local_attempt_row(rows_dir, "households_10", _row_with_trajectory())
    # submission.trajectory_path is relative to output_dir; rows_dir is
    # output_dir/"rows", so the row file is rows_dir/<iid>/attempt-1.json.
    assert (rows_dir / "households_10" / "attempt-1.json").exists()


def test_write_is_best_effort_on_unserializable(tmp_path):
    """A stray non-JSON object must not crash the run loop — default=str
    guards it (mirrors cloud popping Pydantic objects before json.dumps)."""
    rows_dir = tmp_path / "rows"

    class _Weird:
        def __repr__(self) -> str:
            return "<weird>"

    row = _row_with_trajectory()
    row["_leftover_obj"] = _Weird()
    # Must not raise.
    write_local_attempt_row(rows_dir, "households_10", row)
    dest = rows_dir / "households_10" / "attempt-1.json"
    assert dest.exists()
    loaded = json.loads(dest.read_text())
    assert loaded["trajectory"] == row["trajectory"]
