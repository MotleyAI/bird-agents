"""Ingest upstream's `original/results.jsonl` + per-turn JSONLs into
`results.db` so the original leg sits in the same SQLite store as
raw/slayer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _write_orig_dir(
    orig_dir: Path,
    *,
    instance_id: str = "alien_1",
    submitted_sql: str | None = "SELECT id FROM telescopes",
    sol_sql: str | None = "SELECT id FROM telescopes",
    phase1_passed: bool = False,
    n_turns: int = 2,
    prompt_tokens_per_turn: int = 1000,
    completion_tokens_per_turn: int = 200,
) -> None:
    """Mimic the on-disk layout that upstream `mini_interact_agent` writes
    after `batch_run_bird_interact.main`."""
    orig_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "idx": 0,
        "instance_id": instance_id,
        "original_data": {
            "instance_id": instance_id,
            "selected_database": "alien",
            "sol_sql": [sol_sql] if sol_sql else [],
        },
        "phase1_completed": phase1_passed,
        "task_finished": False,
        "last_reward": 1.0 if phase1_passed else 0.0,
        "successful_phase1_sql": submitted_sql if phase1_passed else None,
        "interaction_history": [
            {"turn": n_turns, "phase": 1, "action": f"submit_sql({submitted_sql})"},
        ] if submitted_sql and not phase1_passed else [],
    }
    (orig_dir / "results.jsonl").write_text(json.dumps(record) + "\n")
    for i in range(1, n_turns + 1):
        (orig_dir / f"results.jsonl.agent_raw_turn_{i}.jsonl").write_text(
            json.dumps({
                "prompt": "...",
                "id": instance_id,
                "response": "...",
                "token_usage": {
                    "prompt_tokens": prompt_tokens_per_turn,
                    "completion_tokens": completion_tokens_per_turn,
                    "reasoning_tokens": 0,
                    "total_tokens": prompt_tokens_per_turn + completion_tokens_per_turn,
                },
            }) + "\n"
        )
    (orig_dir / "duration.json").write_text(
        json.dumps({
            "total_duration_s": 60.0,
            "n_tasks": 1,
            "avg_duration_s": 60.0,
            "agent_model": "anthropic/claude-haiku-4-5-20251001",
        })
    )


def test_ingest_writes_one_row_per_task(tmp_path):
    from bird_interact_agents.original_ingest import ingest_original_to_db

    orig_dir = tmp_path / "original"
    _write_orig_dir(orig_dir)
    db_path = tmp_path / "results.db"

    ingest_original_to_db(orig_dir=orig_dir, db_path=db_path, run_id="3way_smoke")

    with sqlite3.connect(db_path) as c:
        rows = list(c.execute(
            "SELECT instance_id, query_mode, framework, submitted_sql, "
            "ground_truth_sql, phase1_passed, duration_s "
            "FROM task_results"
        ))
    assert rows == [(
        "alien_1", "raw", "original", "SELECT id FROM telescopes",
        "SELECT id FROM telescopes", 0, 60.0,
    )]


def test_ingest_uses_successful_phase1_sql_when_passed(tmp_path):
    from bird_interact_agents.original_ingest import ingest_original_to_db

    orig_dir = tmp_path / "original"
    _write_orig_dir(orig_dir, phase1_passed=True, submitted_sql="GOOD SQL")
    db_path = tmp_path / "results.db"
    ingest_original_to_db(orig_dir=orig_dir, db_path=db_path, run_id="x")

    with sqlite3.connect(db_path) as c:
        row = c.execute(
            "SELECT submitted_sql, phase1_passed FROM task_results"
        ).fetchone()
    assert row == ("GOOD SQL", 1)


def test_ingest_sums_token_usage_across_turns(tmp_path):
    from bird_interact_agents.original_ingest import ingest_original_to_db

    orig_dir = tmp_path / "original"
    _write_orig_dir(
        orig_dir, n_turns=3, prompt_tokens_per_turn=500,
        completion_tokens_per_turn=100,
    )
    db_path = tmp_path / "results.db"
    ingest_original_to_db(orig_dir=orig_dir, db_path=db_path, run_id="x")

    with sqlite3.connect(db_path) as c:
        usage_json = c.execute(
            "SELECT usage_json FROM task_results"
        ).fetchone()[0]
    usage = json.loads(usage_json)
    # 3 turns × 500 prompt + 100 completion each = 1500 / 300
    assert usage["prompt_tokens"] == 1500
    assert usage["completion_tokens"] == 300


def test_ingest_skips_when_results_jsonl_missing(tmp_path):
    """No-op (don't crash) when the original leg never produced its
    output — e.g. a fail-fast bash exit before upstream wrote anything."""
    from bird_interact_agents.original_ingest import ingest_original_to_db

    orig_dir = tmp_path / "original"
    orig_dir.mkdir()
    db_path = tmp_path / "results.db"
    ingest_original_to_db(orig_dir=orig_dir, db_path=db_path, run_id="x")
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM task_results"
            ).fetchone()[0]
        assert count == 0


@pytest.mark.integration
def test_compare_results_picks_up_ingested_original_row(tmp_path):
    """End-to-end: after ingest, the original leg's submitted_sql shows
    up in `comparison.json["per_task"]` and `per_task_sql.md`.

    Marked integration: shells out to the real `scripts.compare_results`
    subprocess. The other tests in this module are fast in-process sqlite
    checks and stay in the default run."""
    import subprocess
    import sys

    from bird_interact_agents.original_ingest import ingest_original_to_db

    base = tmp_path / "3way"
    orig_dir = base / "original"
    _write_orig_dir(
        orig_dir, submitted_sql="ORIG SQL",
        sol_sql="GROUND SQL", phase1_passed=True,
    )
    # ingest writes to <orig_dir>/results.db so compare_results picks it up
    # via the same per-leg DB path used for raw/slayer.
    db_path = orig_dir / "results.db"
    ingest_original_to_db(orig_dir=orig_dir, db_path=db_path, run_id="3way")

    # No raw/slayer dirs — pass --allow-missing so the script doesn't
    # bail on the absent eval.json files.
    res = subprocess.run(
        [
            sys.executable, "-m", "scripts.compare_results",
            "--allow-missing", str(base),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, check=True,
    )
    assert "Wrote" in res.stdout

    blob = json.loads((base / "comparison.json").read_text())
    orig_row = blob["per_task"]["alien_1"]["original"]
    assert orig_row["submitted_sql"] == "ORIG SQL"
    assert orig_row["ground_truth_sql"] == "GROUND SQL"

    md = (base / "per_task_sql.md").read_text()
    assert "ORIG SQL" in md
    assert "GROUND SQL" in md
