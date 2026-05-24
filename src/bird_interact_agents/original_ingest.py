"""Backfill `task_results` rows for the original (upstream) leg.

The upstream `mini_interact_agent` writes its own
`<output_dir>/original/results.jsonl` (one SampleStatus per task) plus
per-turn `agent_raw_turn_*.jsonl` files carrying `token_usage`. None
of that lands in our `results.db` automatically, so `compare_results`
can't query the original leg the same way it queries raw/slayer.

`ingest_original_to_db` fills the gap: parse the upstream artefacts,
insert one row per task with the `framework="original"` tag, and use
the same column schema raw/slayer use. After this runs, the original
leg is queryable from `results.db` with no special-case in callers.
"""

from __future__ import annotations

import json
from pathlib import Path

from bird_interact_agents.results_db import (
    TaskResultRow,
    insert_task_result,
    open_db,
)


def _last_submit_sql(history: list) -> str | None:
    """Walk a SampleStatus.interaction_history for the most recent
    `submit_sql(...)` action. Returns None if the agent never submitted.
    """
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") or ""
        if action.startswith("submit_sql(") and action.endswith(")"):
            return action[len("submit_sql("):-1]
    return None


def _extract_ground_truth(record: dict) -> str | None:
    od = record.get("original_data") or {}
    sol = od.get("sol_sql")
    if isinstance(sol, list) and sol:
        return sol[0]
    if isinstance(sol, str):
        return sol
    return None


def _sum_token_usage(orig_dir: Path) -> dict:
    """Sum `token_usage` across all per-turn JSONLs.

    Returns a TokenUsage-shaped dict that's compatible with what
    `bird_interact_agents.usage.TokenUsage.model_dump()` produces, so
    `compare_results._load_ours_from_db` can read it back via the
    `usage_json` column without a special-case.
    """
    prompt = 0
    completion = 0
    reasoning = 0
    n_calls = 0
    for f in sorted(orig_dir.glob("results.jsonl.agent_raw_turn_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                u = row.get("token_usage") or {}
                if not u:
                    continue
                prompt += int(u.get("prompt_tokens") or 0)
                completion += int(u.get("completion_tokens") or 0)
                reasoning += int(u.get("reasoning_tokens") or 0)
                n_calls += 1
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cache_read_tokens": 0,
        "n_calls": n_calls,
        "cost_usd": 0.0,
        "agent_cost_usd": 0.0,
        "user_sim_cost_usd": 0.0,
        "partial": True,  # upstream doesn't separately track user-sim usage
        "breakdown": [],
    }


def _read_duration(orig_dir: Path) -> tuple[float, str]:
    """Return (avg_duration_s, agent_model) from the leg-total
    `duration.json`, or (0.0, "") if missing."""
    path = orig_dir / "duration.json"
    if not path.is_file():
        return 0.0, ""
    blob = json.loads(path.read_text())
    return float(blob.get("avg_duration_s") or 0.0), blob.get("agent_model") or ""


def ingest_original_to_db(
    *,
    orig_dir: Path,
    db_path: Path,
    run_id: str,
    mode: str = "a-interact",
) -> int:
    """Read upstream's results.jsonl + per-turn JSONLs and write one
    `task_results` row per task into `db_path`. Returns the number of
    rows inserted (0 if the upstream output is missing).

    Idempotent: re-running replaces prior rows by primary key
    (run_id, framework, query_mode, instance_id).
    """
    results_jsonl = orig_dir / "results.jsonl"
    if not results_jsonl.is_file():
        return 0

    avg_duration, _ = _read_duration(orig_dir)
    usage_json = json.dumps(_sum_token_usage(orig_dir))

    conn = open_db(db_path)
    inserted = 0
    try:
        with open(results_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                phase1_passed = bool(record.get("phase1_completed"))
                submitted_sql = (
                    record.get("successful_phase1_sql")
                    or _last_submit_sql(record.get("interaction_history") or [])
                )
                instance_id = (
                    record.get("instance_id")
                    or (record.get("original_data") or {}).get("instance_id")
                    or ""
                )
                database = (record.get("original_data") or {}).get(
                    "selected_database"
                ) or ""
                insert_task_result(conn, TaskResultRow(
                    run_id=run_id,
                    framework="original",
                    mode=mode,
                    query_mode="raw",
                    instance_id=instance_id,
                    database=database,
                    started_at=0.0,
                    duration_s=avg_duration,
                    phase1_passed=phase1_passed,
                    phase2_passed=bool(record.get("task_finished")),
                    total_reward=float(record.get("last_reward") or 0.0),
                    submitted_sql=submitted_sql,
                    submitted_query=None,
                    ground_truth_sql=_extract_ground_truth(record),
                    error=None,
                    usage_json=usage_json,
                ))
                inserted += 1
    finally:
        conn.close()
    return inserted


def main() -> None:
    """CLI entry point — used by `run_three_way.sh` after the original leg."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orig-dir", required=True,
        help="Directory containing original/results.jsonl + agent_raw_turn_*.jsonl",
    )
    parser.add_argument(
        "--db-path", required=True,
        help="SQLite results.db path; created if missing",
    )
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--mode", default="a-interact")
    args = parser.parse_args()

    n = ingest_original_to_db(
        orig_dir=Path(args.orig_dir),
        db_path=Path(args.db_path),
        run_id=args.run_id,
        mode=args.mode,
    )
    print(f"Ingested {n} original-leg row(s) into {args.db_path}")


if __name__ == "__main__":
    main()
