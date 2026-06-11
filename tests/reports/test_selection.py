"""Tests for selection.jsonl loading + per-instance source resolution.

Spec (DEV-1553):
* Duplicate ``instance_id`` in a selection file is a hard error listing
  every duplicate.
* Missing trajectory.json for any (instance_id, run_id) is a hard error
  listing every missing entry.
* Stub-only trajectory (no ``trajectory`` array) is a hard error.
"""

from __future__ import annotations

import json

import pytest

from tests.reports._fixtures import trajectory_one_phase_pass


# ---------------------------------------------------------------------------
# Selection.jsonl loader
# ---------------------------------------------------------------------------


def test_selection_loads_well_formed_file(tmp_path):
    from bird_interact_agents.reports.selection import load_selection

    sel_path = tmp_path / "selection.jsonl"
    sel_path.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "alien_1", "run_id": "run_a"}),
                json.dumps({"instance_id": "alien_2", "run_id": "run_b"}),
            ]
        )
    )
    sel = load_selection(sel_path)
    assert sel == [
        ("alien_1", "run_a"),
        ("alien_2", "run_b"),
    ]


def test_selection_duplicate_instance_id_is_hard_error(tmp_path):
    """Codex finding #1 + spec: dupes must be flagged with every duplicate
    listed in the error message."""
    from bird_interact_agents.reports.selection import (
        DuplicateInstanceError,
        load_selection,
    )

    sel_path = tmp_path / "selection.jsonl"
    sel_path.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "alien_1", "run_id": "run_a"}),
                json.dumps({"instance_id": "alien_1", "run_id": "run_b"}),
                json.dumps({"instance_id": "alien_3", "run_id": "run_c"}),
                json.dumps({"instance_id": "alien_3", "run_id": "run_d"}),
            ]
        )
    )
    with pytest.raises(DuplicateInstanceError) as exc_info:
        load_selection(sel_path)
    msg = str(exc_info.value)
    assert "alien_1" in msg
    assert "alien_3" in msg


def test_selection_malformed_line_is_hard_error(tmp_path):
    from bird_interact_agents.reports.selection import load_selection

    sel_path = tmp_path / "selection.jsonl"
    sel_path.write_text('{"instance_id": "alien_1"}\n')  # missing run_id
    with pytest.raises((ValueError, KeyError)):
        load_selection(sel_path)


# ---------------------------------------------------------------------------
# Source resolution (locate trajectory.json + results.db)
# ---------------------------------------------------------------------------


def test_resolve_sources_finds_existing_trajectory(stage):
    _runs_root, _results_root = stage(
        benchmark="bird-interact-lite-exp",
        run_id="r1",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    from bird_interact_agents.reports.sources import resolve_sources

    sources = resolve_sources(
        selection=[("alien_1", "r1")],
        benchmark="bird-interact-lite-exp",
    )
    src = sources["alien_1"]
    assert src.trajectory_path.exists()
    assert src.results_db_path.exists()
    assert src.database == "alien"
    # Real cloud runs persist `framework="claude_sdk"` (the CLI flag).
    assert src.framework == "claude_sdk"
    assert src.agent_model == "anthropic/claude-opus-4-7"
    assert src.user_sim_model == "anthropic/claude-sonnet-4-6"
    assert src.mode == "a-interact"
    assert src.query_mode == "slayer"


def test_resolve_sources_missing_trajectory_lists_every_missing(stage, tmp_path):
    stage(
        benchmark="bird-interact-lite-exp",
        run_id="r1",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    from bird_interact_agents.reports.sources import (
        MissingTrajectoryError,
        resolve_sources,
    )

    with pytest.raises(MissingTrajectoryError) as exc_info:
        resolve_sources(
            selection=[
                ("alien_1", "r1"),  # present
                ("alien_2", "r1"),  # missing
                ("alien_3", "r1"),  # missing
            ],
            benchmark="bird-interact-lite-exp",
        )
    msg = str(exc_info.value)
    assert "alien_2" in msg
    assert "alien_3" in msg
    assert "alien_1" not in msg  # the present one should not appear


def test_resolve_sources_stub_only_trajectory_is_hard_error(stage):
    """A trajectory.json whose top-level lacks a ``trajectory`` array
    (older mini-interact placeholder) cannot be reconstructed."""
    import json as _json

    runs_root, _ = stage(
        benchmark="mini-interact",
        run_id="r2",
        instances=[
            (
                "db_a",
                "db_a_0",
                {"instance_id": "db_a_0", "phase1_passed": False,
                 "phase2_passed": False, "submitted_sql": "",
                 "submission_status": "error"},
            ),
        ],
    )
    # Overwrite the trajectory file with a stub (no `trajectory` array).
    stub_path = runs_root / "mini-interact" / "db_a" / "db_a_0" / "r2.trajectory.json"
    stub_path.write_text(_json.dumps({"trajectory_path": "rows/db_a_0/attempt-1.json"}))

    from bird_interact_agents.reports.sources import (
        StubTrajectoryError,
        resolve_sources,
    )

    with pytest.raises(StubTrajectoryError) as exc_info:
        resolve_sources(
            selection=[("db_a_0", "r2")],
            benchmark="mini-interact",
        )
    assert "db_a_0" in str(exc_info.value)


def test_resolve_sources_missing_task_results_row_is_hard_error(stage):
    """Codex round 3 finding: a trajectory.json + results.db that exist
    but lack the selected instance_id in task_results must NOT silently
    produce an InstanceSource with empty mode (which would bypass the
    a-Interact gate). Refuse here."""
    import sqlite3

    _runs_root, results_root = stage(
        benchmark="bird-interact-lite-exp",
        run_id="r1",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    # Drop the task_results row but keep the trajectory + run_metadata.
    db = results_root / "bird-interact-lite-exp" / "cloud" / "r1" / "results.db"
    con = sqlite3.connect(db)
    con.execute("DELETE FROM task_results WHERE instance_id = ?", ("alien_1",))
    con.commit()
    con.close()

    from bird_interact_agents.reports.sources import (
        MissingTaskResultsError,
        resolve_sources,
    )

    with pytest.raises(MissingTaskResultsError) as exc_info:
        resolve_sources(
            selection=[("alien_1", "r1")],
            benchmark="bird-interact-lite-exp",
        )
    assert "alien_1" in str(exc_info.value)


def test_resolve_sources_missing_results_db_is_hard_error(stage, tmp_path):
    _runs_root, results_root = stage(
        benchmark="bird-interact-lite-exp",
        run_id="r1",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    # Remove the results.db.
    (results_root / "bird-interact-lite-exp" / "cloud" / "r1" / "results.db").unlink()

    from bird_interact_agents.reports.sources import resolve_sources

    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_sources(
            selection=[("alien_1", "r1")],
            benchmark="bird-interact-lite-exp",
        )
