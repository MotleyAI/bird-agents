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
        by_instance = {row["instance_id"]: dict(row) for row in task_rows}
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
                task_results_row=task_rows_by_inst.get(inst_id, {}),
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
    return sources
