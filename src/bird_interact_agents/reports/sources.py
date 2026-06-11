"""Source resolution: locate trajectory.json + results.db for each
``(instance_id, run_id)`` and read run metadata.

Hard errors:
* Missing trajectory.json → ``MissingTrajectoryError`` listing every
  missing entry.
* Trajectory.json present but ``trajectory`` array absent (older mini-
  interact placeholder) → ``StubTrajectoryError``.
* Missing results.db → ``FileNotFoundError``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from bird_interact_agents import paths


class MissingTrajectoryError(FileNotFoundError):
    pass


class StubTrajectoryError(ValueError):
    pass


class MissingTaskResultsError(ValueError):
    pass


@dataclass
class InstanceSource:
    instance_id: str
    run_id: str
    database: str
    trajectory_path: Path
    results_db_path: Path
    framework: str
    agent_model: str
    user_sim_model: str
    trajectory_obj: dict
    task_results_row: dict
    # Per-row (framework can also be in run_metadata but query_mode + mode
    # are per-task; we use the task_results values as authoritative).
    mode: str
    query_mode: str


def _read_results_db(results_db_path: Path, run_id: str) -> tuple[dict, dict[str, dict]]:
    con = sqlite3.connect(results_db_path)
    con.row_factory = sqlite3.Row
    try:
        meta_rows = con.execute(
            "SELECT * FROM run_metadata WHERE run_id = ?", (run_id,)
        ).fetchall()
        if not meta_rows:
            raise ValueError(
                f"{results_db_path}: run_metadata has no row for run_id={run_id!r}"
            )
        meta = dict(meta_rows[0])
        task_rows = con.execute(
            "SELECT * FROM task_results WHERE run_id = ?", (run_id,)
        ).fetchall()
        # Codex round 5 finding: task_results' composite key includes
        # framework/mode/query_mode, so in principle a single run_id
        # could carry multiple rows per instance_id. The harness never
        # writes such rows today, but a silent dict overwrite would
        # pick whichever row SQLite returns last, potentially landing
        # on the wrong mode + flipping adapter / a-Interact-gate
        # decisions downstream. Raise instead.
        by_instance: dict[str, dict] = {}
        dupes: list[str] = []
        for row in task_rows:
            iid = row["instance_id"]
            if iid in by_instance:
                dupes.append(iid)
            by_instance[iid] = dict(row)
        if dupes:
            raise ValueError(
                f"{results_db_path}: task_results has multiple rows for "
                f"run_id={run_id!r} on instance_id(s): "
                f"{', '.join(sorted(set(dupes)))}. Each (run_id, instance_id) "
                f"must be unique — a duplicate indicates a corrupt or "
                f"mixed-mode run."
            )
    finally:
        con.close()
    return meta, by_instance


def _instance_dir(benchmark: str, db: str, instance_id: str) -> Path:
    return paths.runs_root() / benchmark / db / instance_id


def _find_database_for_instance(
    benchmark: str, instance_id: str
) -> str | None:
    """Walk ``runs/<benchmark>/*/<instance_id>/`` to find the ``db``
    subdirectory that contains this instance's trajectory."""
    bench_root = paths.runs_root() / benchmark
    if not bench_root.is_dir():
        return None
    for db_dir in bench_root.iterdir():
        if not db_dir.is_dir():
            continue
        if (db_dir / instance_id).is_dir():
            return db_dir.name
    return None


def resolve_sources(
    *,
    selection: list[tuple[str, str]],
    benchmark: str,
) -> dict[str, InstanceSource]:
    """Return ``{instance_id: InstanceSource}`` for the entire selection.

    Raises ``MissingTrajectoryError`` / ``StubTrajectoryError`` /
    ``FileNotFoundError`` for any unresolvable entry, listing all
    offenders.
    """
    # Group by run_id so we open each results.db once.
    by_run: dict[str, list[str]] = {}
    for inst_id, run_id in selection:
        by_run.setdefault(run_id, []).append(inst_id)

    sources: dict[str, InstanceSource] = {}
    missing_traj: list[tuple[str, str, Path]] = []
    stub_traj: list[tuple[str, str, Path]] = []
    missing_task: list[tuple[str, str]] = []

    for run_id, inst_ids in by_run.items():
        results_db_path = (
            paths.results_root() / benchmark / "cloud" / run_id / "results.db"
        )
        if not results_db_path.is_file():
            raise FileNotFoundError(
                f"{results_db_path} does not exist (run_id={run_id!r}, "
                f"benchmark={benchmark!r})"
            )
        meta, task_rows_by_inst = _read_results_db(results_db_path, run_id)

        for inst_id in inst_ids:
            # Find the database subdir.
            db_name = task_rows_by_inst.get(inst_id, {}).get("database")
            if not db_name:
                db_name = _find_database_for_instance(benchmark, inst_id)
            if not db_name:
                missing_traj.append(
                    (inst_id, run_id, paths.runs_root() / benchmark)
                )
                continue

            traj_path = (
                _instance_dir(benchmark, db_name, inst_id)
                / f"{run_id}.trajectory.json"
            )
            if not traj_path.is_file():
                missing_traj.append((inst_id, run_id, traj_path))
                continue
            traj_obj = json.loads(traj_path.read_text())
            if "trajectory" not in traj_obj or not isinstance(
                traj_obj.get("trajectory"), list
            ):
                stub_traj.append((inst_id, run_id, traj_path))
                continue

            task_row = task_rows_by_inst.get(inst_id)
            if not task_row:
                # No row in results.db for this instance — Codex round 3
                # finding: empty task_row makes mode='' which silently
                # bypasses the a-Interact gate. Refuse here.
                missing_task.append((inst_id, run_id))
                continue
            sources[inst_id] = InstanceSource(
                instance_id=inst_id,
                run_id=run_id,
                database=db_name,
                trajectory_path=traj_path,
                results_db_path=results_db_path,
                framework=str(meta.get("framework") or ""),
                agent_model=str(meta.get("agent_model") or ""),
                user_sim_model=str(meta.get("user_sim_model") or ""),
                trajectory_obj=traj_obj,
                task_results_row=task_row,
                mode=str(task_row.get("mode") or meta.get("mode") or ""),
                query_mode=str(task_row.get("query_mode") or ""),
            )

    if missing_traj:
        lines = "\n".join(
            f"  - instance_id={iid} run_id={rid} expected at {p}"
            for iid, rid, p in missing_traj
        )
        raise MissingTrajectoryError(
            f"trajectory.json missing for {len(missing_traj)} selection "
            f"entries (benchmark={benchmark!r}):\n{lines}"
        )
    if stub_traj:
        lines = "\n".join(
            f"  - instance_id={iid} run_id={rid} at {p}"
            for iid, rid, p in stub_traj
        )
        raise StubTrajectoryError(
            f"stub-only trajectory.json (no `trajectory` array) for "
            f"{len(stub_traj)} entries — cannot reconstruct prompt_flow:\n{lines}"
        )
    if missing_task:
        lines = "\n".join(
            f"  - instance_id={iid} run_id={rid}"
            for iid, rid in missing_task
        )
        raise MissingTaskResultsError(
            f"results.db has no task_results row for {len(missing_task)} "
            f"selection entries — the run never recorded these instances, "
            f"so mode / query_mode / final SQL / phase results are all "
            f"unknown:\n{lines}"
        )
    return sources
