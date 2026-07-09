"""DEV-1649 (Codex #5): the two observability columns must be part of the
additive-migration path AND round-trip through insert_task_result.

open_db() only ALTERs columns listed in _DIAGNOSTIC_COLUMNS; insert uses an
explicit column list; TaskResultRow must carry the fields. All three must be
touched together, so pin the contract.
"""

from __future__ import annotations

import sqlite3

from bird_interact_agents import results_db


_NEW_COLS = ("edited_models_saved_path", "edited_models_applied_from")


def _base_row_kwargs() -> dict:
    return dict(
        run_id="r1", framework="claude_sdk_otf_ainteract", mode="a-interact",
        query_mode="slayer", instance_id="alien_1", database="alien",
        started_at=0.0, duration_s=1.0, phase1_passed=True, phase2_passed=False,
        total_reward=1.0,
    )


def test_new_columns_in_diagnostic_migration():
    names = {n for n, _ in results_db._DIAGNOSTIC_COLUMNS}
    for col in _NEW_COLS:
        assert col in names


def test_task_result_row_has_fields():
    row = results_db.TaskResultRow(
        **_base_row_kwargs(),
        edited_models_saved_path="/runs/x/edited_models.tar.gz",
        edited_models_applied_from=None,
    )
    assert row.edited_models_saved_path.endswith("edited_models.tar.gz")
    assert row.edited_models_applied_from is None


def test_open_db_alters_legacy_db(tmp_path):
    # Simulate a pre-existing DB WITHOUT the new columns.
    db_path = tmp_path / "results.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE task_results (run_id TEXT, framework TEXT, mode TEXT, "
        "query_mode TEXT, instance_id TEXT, database TEXT, started_at REAL, "
        "duration_s REAL, phase1_passed INTEGER, phase2_passed INTEGER, "
        "total_reward REAL)"
    )
    con.commit()
    con.close()

    conn = results_db.open_db(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(task_results)")}
        for col in _NEW_COLS:
            assert col in cols
    finally:
        conn.close()


def test_collation_row_carries_new_columns():
    """The default local process-pool path + cloud fetch build the row via
    collation._row_to_task_result_row; it must carry the edited-models
    provenance (DEV-1649 / process-reviews Codex #1)."""
    from bird_interact_agents.cloud.collation import _row_to_task_result_row

    manifest = {
        "run_id": "r1", "framework": "claude_sdk_otf_ainteract",
        "mode": "a-interact", "query_mode": "slayer",
    }
    r = {
        "instance_id": "alien_1", "database": "alien",
        "phase1_passed": True, "phase2_passed": False, "total_reward": 1.0,
        "edited_models_saved_path": "/runs/alien/alien_1/edited_models.tar.gz",
        "edited_models_applied_from": "/runs/alien/alien_1/edited_models.tar.gz",
    }
    row = _row_to_task_result_row(manifest, r)
    assert row.edited_models_saved_path.endswith("edited_models.tar.gz")
    assert row.edited_models_applied_from.endswith("edited_models.tar.gz")


def test_insert_round_trips_new_columns(tmp_path):
    conn = results_db.open_db(tmp_path / "results.db")
    try:
        row = results_db.TaskResultRow(
            **_base_row_kwargs(),
            edited_models_saved_path="/runs/alien/alien_1/edited_models.tar.gz",
            edited_models_applied_from="/runs/alien/alien_1/edited_models.tar.gz",
        )
        results_db.insert_task_result(conn, row)
        got = conn.execute(
            "SELECT edited_models_saved_path, edited_models_applied_from "
            "FROM task_results WHERE instance_id='alien_1'"
        ).fetchone()
        assert got == (
            "/runs/alien/alien_1/edited_models.tar.gz",
            "/runs/alien/alien_1/edited_models.tar.gz",
        )
    finally:
        conn.close()
