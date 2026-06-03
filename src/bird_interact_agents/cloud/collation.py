"""Collate per-task GCS objects (after `fetch`) into a local `results.db`
+ `eval.json`.

Latest attempt per iid wins; older attempts stay on disk under
`rows/<iid>/attempt-*.json` for forensics but are NOT written to the DB.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from bird_interact_agents.results_db import (
    TaskResultRow,
    insert_run_metadata,
    insert_task_result,
    open_db,
)
from bird_interact_agents.usage import TokenUsage


_ATTEMPT_RE = re.compile(r"^attempt-(?P<n>\d+)\.json$")


def _discover_canonical_rows(run_dir: Path) -> tuple[dict[str, dict],
                                                       dict[str, list[int]]]:
    """Walk `run_dir/rows/<iid>/attempt-*.json` and return (canonical, attempts).
    canonical[iid] = the row dict from the latest attempt.
    attempts[iid]  = sorted list of attempt numbers seen for that iid.
    """
    canonical: dict[str, dict] = {}
    attempts: dict[str, list[int]] = {}
    rows_dir = run_dir / "rows"
    if not rows_dir.exists():
        return {}, {}
    for iid_dir in sorted(rows_dir.iterdir()):
        if not iid_dir.is_dir():
            continue
        seen: list[int] = []
        for f in iid_dir.iterdir():
            m = _ATTEMPT_RE.match(f.name)
            if not m:
                continue
            seen.append(int(m.group("n")))
        if not seen:
            continue
        seen.sort()
        attempts[iid_dir.name] = seen
        latest_path = iid_dir / f"attempt-{seen[-1]}.json"
        canonical[iid_dir.name] = json.loads(latest_path.read_text())
    return canonical, attempts


def _row_to_task_result_row(manifest: dict, r: dict) -> TaskResultRow:
    """Build a TaskResultRow from a row dict (the same shape `_persist`
    consumes in run.py)."""
    sol = r.get("ground_truth_sql")
    usage_blob = r.get("usage")
    usage_json = json.dumps(usage_blob) if usage_blob is not None else "{}"
    n_turns = r.get("n_agent_turns")
    stats_blob = r.get("tool_call_stats")
    tool_call_stats_json = (
        json.dumps(stats_blob) if stats_blob is not None else r.get("tool_call_stats_json")
    )
    return TaskResultRow(
        run_id=manifest["run_id"],
        framework=manifest["framework"],
        mode=manifest["mode"],
        query_mode=manifest["query_mode"],
        instance_id=str(r.get("instance_id") or ""),
        database=str(r.get("database") or ""),
        started_at=float(r.get("started_at") or 0.0),
        duration_s=float(r.get("duration_s") or 0.0),
        phase1_passed=bool(r.get("phase1_passed")),
        phase2_passed=bool(r.get("phase2_passed")),
        total_reward=float(r.get("total_reward") or 0.0),
        submitted_sql=r.get("submitted_sql"),
        submitted_query=r.get("submitted_query"),
        ground_truth_sql=sol,
        error=r.get("error"),
        usage_json=usage_json,
        user_query=r.get("user_query"),
        submission_status=str(r.get("submission_status") or "never_submitted"),
        phase1_observation=r.get("phase1_observation"),
        phase2_observation=r.get("phase2_observation"),
        predicted_result_json=r.get("predicted_result_json"),
        gold_result_json=r.get("gold_result_json"),
        n_agent_turns=int(n_turns) if isinstance(n_turns, int) else None,
        tool_call_stats_json=tool_call_stats_json,
        # Codex r9: parity with the local ``run.py::_persist`` shape.
        # Cloud workers emit these observation strings on the per-row
        # JSON blob (every agent flavor's submit helper +
        # ``agents/_submit.py`` populate them); without the explicit
        # plumb-through, cloud ``fetch``+collation drops them at the
        # results.db boundary while local runs retain them — an
        # asymmetry that hid the audited/original observation diag
        # for everything that ran in the cloud.
        phase1_observation_audited=r.get("phase1_observation_audited"),
        phase1_observation_original=r.get("phase1_observation_original"),
    )


def _build_metrics(manifest: dict, canonical_rows: list[dict],
                    attempts: dict[str, list[int]]) -> dict[str, Any]:
    n = len(canonical_rows)
    p1_count = sum(1 for r in canonical_rows if r.get("phase1_passed"))
    p2_count = sum(1 for r in canonical_rows if r.get("phase2_passed"))
    total_reward = sum(float(r.get("total_reward") or 0.0) for r in canonical_rows)

    total_usage = TokenUsage()
    for r in canonical_rows:
        u_blob = r.get("usage")
        if u_blob is not None:
            try:
                total_usage.merge(TokenUsage.model_validate(u_blob))
            except Exception:  # noqa: BLE001
                pass

    durations = [float(r.get("duration_s") or 0.0) for r in canonical_rows]
    resubmitted_ids = sorted(iid for iid, lst in attempts.items() if len(lst) > 1)

    # DEV-1515: the legacy dual-eval breakdown is replaced by the
    # cascading_phase1 block built downstream by
    # `emit_cascading_eval_json` over per-row submission_annotation.json
    # files (merged in by `driver.fetch` via `merge_submission_annotations`).
    return {
        "mode": manifest["mode"],
        "query_mode": manifest["query_mode"],
        "framework": manifest["framework"],
        "total_tasks": n,
        "phase1_count": p1_count,
        "phase1_rate": p1_count / n if n else 0,
        "phase2_count": p2_count,
        "phase2_rate": p2_count / n if n else 0,
        "total_reward": total_reward,
        "average_reward": total_reward / n if n else 0,
        "total_usage": total_usage.model_dump(),
        "total_duration_s": sum(durations),
        "avg_duration_s": (sum(durations) / len(durations)) if durations else 0.0,
        "p50_duration_s": statistics.median(durations) if durations else 0.0,
        "max_duration_s": max(durations) if durations else 0.0,
        "results": canonical_rows,
        "n_resubmitted": len(resubmitted_ids),
        "resubmitted_ids": resubmitted_ids,
    }


def collate(run_dir: Path, manifest: dict) -> dict[str, Any]:
    """Build/refresh `run_dir/results.db` and write `run_dir/eval.json`.
    Returns the metrics dict written to eval.json."""
    canonical, attempts = _discover_canonical_rows(run_dir)
    instance_ids = manifest.get("instance_ids", [])
    # Order canonical rows by manifest order so eval.json is stable.
    ordered_rows = [canonical[iid] for iid in instance_ids if iid in canonical]

    db_path = run_dir / "results.db"
    db = open_db(db_path)
    try:
        insert_run_metadata(
            db,
            run_id=manifest["run_id"],
            agent_model=manifest.get("agent_model", ""),
            user_sim_model=manifest.get("user_sim_model", ""),
            framework=manifest["framework"],
            mode=manifest["mode"],
            started_at=time.time(),
        )
        for r in ordered_rows:
            insert_task_result(db, _row_to_task_result_row(manifest, r))
    finally:
        db.close()

    metrics = _build_metrics(manifest, ordered_rows, attempts)
    eval_path = run_dir / "eval.json"
    rows_dir = run_dir / "rows"
    has_per_row_anns = rows_dir.exists() and any(
        (rd / "submission_annotation.json").exists()
        for rd in rows_dir.iterdir() if rd.is_dir()
    )
    if has_per_row_anns:
        from bird_interact_agents.eval import cascading_report as _cr
        try:
            metrics = _cr.emit_cascading_eval_json(
                rows_dir=rows_dir, out_path=eval_path, base_metrics=metrics,
            )
        except FileNotFoundError as exc:
            metrics["cascading_phase1_error"] = str(exc)
            eval_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    else:
        eval_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    return metrics
