"""T20–T23: collation — latest attempt wins, INSERT OR REPLACE semantics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from bird_interact_agents.cloud import collation  # noqa: E402


RUN_ID = "20260521T1422-pydanticai-raw-a1b2c3"


def _write_attempt(run_dir: Path, iid: str, attempt: int, row: dict) -> None:
    p = run_dir / "rows" / iid
    p.mkdir(parents=True, exist_ok=True)
    (p / f"attempt-{attempt}.json").write_text(json.dumps(row))


def _read_rows(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT instance_id, error, phase1_passed FROM task_results"
    )]
    conn.close()
    return rows


def _read_dual_cols(db_path: Path) -> dict[str, dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    out = {
        r["instance_id"]: dict(r)
        for r in conn.execute(
            "SELECT instance_id, phase1_passed_audited, phase1_passed_original, "
            "phase1_observation_audited, phase1_observation_original "
            "FROM task_results"
        )
    }
    conn.close()
    return out


# ---------------------------------------------------------------------------
# T20 — multiple attempts per iid: latest wins in results.db; older stays on disk.
# ---------------------------------------------------------------------------


def test_collate_picks_latest_attempt(tmp_path: Path, sample_task_result_row):
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()

    err_row = {
        **sample_task_result_row,
        "error": "boom",
        "phase1_passed": False,
        "phase2_passed": False,
    }
    ok_row = {**sample_task_result_row, "error": None}

    _write_attempt(run_dir, "db_a_1", 1, err_row)
    _write_attempt(run_dir, "db_a_1", 2, ok_row)

    manifest = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1"],
    }
    collation.collate(run_dir, manifest)

    rows = _read_rows(run_dir / "results.db")
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "db_a_1"
    assert rows[0]["error"] is None  # latest attempt won
    assert rows[0]["phase1_passed"] == 1

    # Older attempt's file is still on disk.
    assert (run_dir / "rows" / "db_a_1" / "attempt-1.json").exists()
    assert (run_dir / "rows" / "db_a_1" / "attempt-2.json").exists()


# ---------------------------------------------------------------------------
# T21 — re-running collate after a newer attempt REPLACES the existing row.
# ---------------------------------------------------------------------------


def test_collate_replaces_on_rerun(tmp_path: Path, sample_task_result_row):
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    err_row = {**sample_task_result_row, "error": "boom", "phase1_passed": False}
    _write_attempt(run_dir, "db_a_1", 1, err_row)
    manifest = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1"],
    }
    collation.collate(run_dir, manifest)
    rows1 = _read_rows(run_dir / "results.db")
    assert rows1[0]["error"] == "boom"

    # New attempt lands on disk (e.g. via fetch); re-collate.
    ok_row = {**sample_task_result_row, "error": None}
    _write_attempt(run_dir, "db_a_1", 2, ok_row)
    eval2 = collation.collate(run_dir, manifest)
    rows2 = _read_rows(run_dir / "results.db")
    assert rows2[0]["error"] is None  # replaced, not skipped

    # eval.json bookkeeping for resubmits
    assert eval2.get("n_resubmitted", 0) >= 1
    assert "db_a_1" in eval2.get("resubmitted_ids", [])


# ---------------------------------------------------------------------------
# T22 — eval.json matches what local `bird-interact run` produces on the
#       same per-task row set (key set AND values for the metrics fields
#       the aggregator computes deterministically from `results`).
# ---------------------------------------------------------------------------


def test_eval_json_matches_local_aggregator(
    tmp_path: Path, sample_task_result_row
):
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    # Two rows: one full success, one full failure — gives non-trivial rates.
    row_ok = {**sample_task_result_row, "instance_id": "db_a_1",
              "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0}
    row_fail = {**sample_task_result_row, "instance_id": "db_a_2",
                "phase1_passed": False, "phase2_passed": False,
                "total_reward": 0.0, "error": "boom"}
    _write_attempt(run_dir, "db_a_1", 1, row_ok)
    _write_attempt(run_dir, "db_a_2", 1, row_fail)

    manifest = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1", "db_a_2"],
    }
    metrics = collation.collate(run_dir, manifest)
    on_disk = json.loads((run_dir / "eval.json").read_text())

    # Same key set on both surfaces.
    assert set(metrics) == set(on_disk)
    # All keys produced by local `bird-interact run`'s aggregator are present.
    expected_keys = {
        "framework", "mode", "query_mode",
        "total_tasks",
        "phase1_count", "phase1_rate",
        "phase2_count", "phase2_rate",
        "total_reward", "average_reward",
        "total_usage", "results",
        "total_duration_s", "avg_duration_s",
        "p50_duration_s", "max_duration_s",
    }
    assert expected_keys <= set(metrics), (
        f"missing keys: {expected_keys - set(metrics)}"
    )
    # Values the aggregator computes purely from the row set.
    assert metrics["total_tasks"] == 2
    assert metrics["phase1_count"] == 1
    assert metrics["phase1_rate"] == 0.5
    assert metrics["phase2_count"] == 1
    assert metrics["phase2_rate"] == 0.5
    assert metrics["total_reward"] == 1.0
    assert metrics["average_reward"] == 0.5
    # And results carries per-row entries in input order.
    assert {r["instance_id"] for r in metrics["results"]} == {"db_a_1", "db_a_2"}


# ---------------------------------------------------------------------------
# Dual-eval columns must survive collation into results.db. Regression: the
# row JSON / eval.json carried phase1_passed_audited/original but
# `_row_to_task_result_row` dropped them, so the cloud results.db showed NULL
# for the original-gold score on every row.
# ---------------------------------------------------------------------------


def test_collate_persists_dual_eval_columns(tmp_path: Path, sample_task_result_row):
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()

    # Edited row: audited gold passed, original gold failed (the diverging
    # case that makes the original score informative).
    diverging = {
        **sample_task_result_row,
        "instance_id": "db_a_1",
        "phase1_passed": True,
        "phase1_passed_audited": True,
        "phase1_passed_original": False,
        "phase1_observation_audited": "audited OK",
        "phase1_observation_original": "original FAILED",
    }
    # Clean row: both golds identical → audited == original == phase1.
    clean = {
        **sample_task_result_row,
        "instance_id": "db_a_2",
        "phase1_passed": True,
        "phase1_passed_audited": True,
        "phase1_passed_original": True,
        "phase1_observation_audited": "OK",
        "phase1_observation_original": "OK",
    }
    _write_attempt(run_dir, "db_a_1", 1, diverging)
    _write_attempt(run_dir, "db_a_2", 1, clean)

    manifest = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai_otf_encode",
        "mode": "a-interact",
        "query_mode": "slayer",
        "agent_model": "anthropic/claude-opus-4-7",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "instance_ids": ["db_a_1", "db_a_2"],
    }
    metrics = collation.collate(run_dir, manifest)

    cols = _read_dual_cols(run_dir / "results.db")
    # The diverging row must keep BOTH verdicts distinct in results.db.
    assert cols["db_a_1"]["phase1_passed_audited"] == 1
    assert cols["db_a_1"]["phase1_passed_original"] == 0
    assert cols["db_a_1"]["phase1_observation_audited"] == "audited OK"
    assert cols["db_a_1"]["phase1_observation_original"] == "original FAILED"
    # The clean row also records the original score (always-score directive).
    assert cols["db_a_2"]["phase1_passed_audited"] == 1
    assert cols["db_a_2"]["phase1_passed_original"] == 1

    # eval.json carries the dual aggregate (parity with local run.py).
    assert metrics["n_dual_eval_tasks"] == 2
    assert metrics["phase1_count_audited"] == 2
    assert metrics["phase1_count_original"] == 1
    assert metrics["phase1_rate_original"] == 0.5


def test_collate_dual_columns_null_when_single_eval(
    tmp_path: Path, sample_task_result_row
):
    """A non-audited run (single-eval) has no dual fields in its row JSON;
    collation must leave the columns NULL and the dual aggregate at 0."""
    run_dir = tmp_path / RUN_ID
    run_dir.mkdir()
    _write_attempt(run_dir, "db_a_1", 1, sample_task_result_row)
    manifest = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1"],
    }
    metrics = collation.collate(run_dir, manifest)

    cols = _read_dual_cols(run_dir / "results.db")
    assert cols["db_a_1"]["phase1_passed_audited"] is None
    assert cols["db_a_1"]["phase1_passed_original"] is None
    assert metrics["n_dual_eval_tasks"] == 0
    assert metrics["phase1_rate_original"] == 0


# ---------------------------------------------------------------------------
# T23 — fetch idempotency at the driver level: running `driver.fetch` twice
#       doesn't duplicate rows in results.db or corrupt eval.json.
# ---------------------------------------------------------------------------


def test_driver_fetch_twice_idempotent(
    monkeypatch, tmp_path: Path, sample_task_result_row, fake_gcs_bucket
):
    from bird_interact_agents.cloud import driver  # noqa: E402

    client, store = fake_gcs_bucket
    # Seed GCS with manifest + one row attempt.
    manifest = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1"],
    }
    store[f"runs/{RUN_ID}/manifest.json"] = json.dumps(manifest).encode()
    store[f"runs/{RUN_ID}/rows/db_a_1/attempt-1.json"] = json.dumps(
        sample_task_result_row
    ).encode()

    monkeypatch.setattr(driver, "default_gcs_client", lambda: client)
    monkeypatch.setattr(
        driver, "local_results_root",
        lambda: tmp_path,
    )

    driver.fetch(RUN_ID)
    rows1 = _read_rows(tmp_path / RUN_ID / "results.db")
    eval1 = json.loads((tmp_path / RUN_ID / "eval.json").read_text())

    driver.fetch(RUN_ID)
    rows2 = _read_rows(tmp_path / RUN_ID / "results.db")
    eval2 = json.loads((tmp_path / RUN_ID / "eval.json").read_text())

    assert len(rows1) == 1
    assert len(rows2) == 1
    # eval.json's row-derived metrics are identical across the two fetches.
    for k in ("total_tasks", "phase1_count", "phase1_rate",
              "phase2_count", "phase2_rate", "total_reward", "average_reward"):
        assert eval1[k] == eval2[k]
