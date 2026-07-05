"""DEV-1640: ``local_pool`` — process-per-task dispatch (``spawn``) that
gives ``--concurrency N`` real OS-process isolation, plus the real
per-task worker's soft-error handling and cancellation cleanup.

Spawned children do NOT inherit pytest monkeypatches, so process-mechanics
tests use REAL importable stub workers from ``tests/_dev1640_workers.py``;
the real ``_local_task_worker`` body (which needs a monkeypatched
``run_one_task`` / grader) is exercised IN-PROCESS.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import threading
from pathlib import Path

import pytest

from bird_interact_agents import local_pool
from tests import _dev1640_workers as workers


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for t0, t1 in intervals:
        events.append((t0, +1))
        events.append((t1, -1))
    events.sort()
    cur = peak = 0
    for _t, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _tasks(n: int) -> list[dict]:
    return [
        {"instance_id": f"t{i}", "selected_database": "db_a"} for i in range(n)
    ]


def test_dispatch_never_exceeds_concurrency(tmp_path: Path):
    """Safety: no more than `concurrency` workers are ever busy at once."""
    tasks = _tasks(6)
    local_pool.dispatch_local_process_pool(
        tasks=tasks, cfg={}, run_dir=str(tmp_path), attempt=1,
        run_id="R", concurrency=2, worker=workers.interval_worker,
    )
    intervals = []
    for t in tasks:
        blob = tmp_path / "rows" / t["instance_id"] / "attempt-1.json"
        assert blob.exists()
        d = json.loads(blob.read_text())
        intervals.append((d["t0"], d["t1"]))
    assert _max_overlap(intervals) <= 2


def test_dispatch_achieves_real_concurrency(tmp_path: Path):
    """Liveness (immune to spawn-import jitter): with a Barrier(parties=N),
    every task's row blob appears ONLY if N workers were simultaneously
    alive — a serial impl would deadlock the barrier and drop blobs."""
    mgr = mp.Manager()
    cfg = {"barrier": mgr.Barrier(2)}
    tasks = _tasks(6)  # 3 waves of 2
    local_pool.dispatch_local_process_pool(
        tasks=tasks, cfg=cfg, run_dir=str(tmp_path), attempt=1,
        run_id="R", concurrency=2, worker=workers.barrier_worker,
    )
    for t in tasks:
        assert (tmp_path / "rows" / t["instance_id"] / "attempt-1.json").exists(), (
            f"{t['instance_id']} missing -> barrier never reached -> not concurrent"
        )


def test_dispatch_runs_in_child_processes_not_parent(tmp_path: Path):
    import os
    tasks = _tasks(4)
    local_pool.dispatch_local_process_pool(
        tasks=tasks, cfg={}, run_dir=str(tmp_path), attempt=1,
        run_id="R", concurrency=2, worker=workers.pid_worker,
    )
    pids = {
        json.loads((tmp_path / "rows" / t["instance_id"] / "attempt-1.json").read_text())["pid"]
        for t in tasks
    }
    assert os.getpid() not in pids, "a task ran in the PARENT process"
    assert len(pids) >= 2, "tasks did not fan out across worker processes"


def test_dispatch_hard_crash_is_isolated_and_reported(tmp_path: Path):
    """A hard crash (os._exit) in one task must NOT prevent siblings from
    completing, must NOT raise, and must be REPORTED as row-not-written."""
    tasks = [
        {"instance_id": "ok1", "selected_database": "db_a"},
        {"instance_id": "boom", "selected_database": "db_a", "crash": True},
        {"instance_id": "ok2", "selected_database": "db_a"},
    ]
    statuses = local_pool.dispatch_local_process_pool(
        tasks=tasks, cfg={}, run_dir=str(tmp_path), attempt=1,
        run_id="R", concurrency=3, worker=workers.branching_worker,
    )
    by_iid = {s["instance_id"]: s for s in statuses}
    assert by_iid["ok1"]["row_written"] is True
    assert by_iid["ok2"]["row_written"] is True
    assert by_iid["boom"]["row_written"] is False
    assert (tmp_path / "rows" / "ok1" / "attempt-1.json").exists()
    assert not (tmp_path / "rows" / "boom" / "attempt-1.json").exists()


def test_dispatch_leaves_no_new_child_processes(tmp_path: Path):
    before = set(mp.active_children())
    local_pool.dispatch_local_process_pool(
        tasks=_tasks(3), cfg={}, run_dir=str(tmp_path), attempt=1,
        run_id="R", concurrency=2, worker=workers.branching_worker,
    )
    new = set(mp.active_children()) - before
    assert new == set(), f"orphaned children: {new}"


def test_spawn_children_inherit_parent_env(tmp_path: Path, monkeypatch):
    """The subscription-auth signal, bridge base-url and BIRD_PG_* are set
    in the PARENT before dispatch; spawn children must see them (this is
    what replaces the old in-process env wiring — no new plumbing)."""
    monkeypatch.setenv("BIRD_INTERACT_SUBSCRIPTION_AUTH", "1")
    monkeypatch.setenv("BIRD_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9911")

    local_pool.dispatch_local_process_pool(
        tasks=[{"instance_id": "e1", "selected_database": "db_a"}],
        cfg={}, run_dir=str(tmp_path), attempt=1,
        run_id="R", concurrency=1, worker=workers.env_echo_worker,
    )
    got = json.loads((tmp_path / "rows" / "e1" / "attempt-1.json").read_text())
    assert got["BIRD_INTERACT_SUBSCRIPTION_AUTH"] == "1"
    assert got["BIRD_PG_HOST"] == "127.0.0.1"
    assert got["ANTHROPIC_BASE_URL"] == "http://localhost:9911"


# ---------------------------------------------------------------------------
# Cancellation cleanup (Codex M3).
# ---------------------------------------------------------------------------


def test_terminate_all_kills_live_children():
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=workers.slow_worker,
                    args=({}, {"instance_id": f"s{i}"}, "/tmp", 1, "R"))
        for i in range(2)
    ]
    for p in procs:
        p.start()
    assert any(p.is_alive() for p in procs)
    local_pool.terminate_all(procs)
    assert all(not p.is_alive() for p in procs)
    assert all(p.exitcode is not None for p in procs)


def test_dispatch_terminates_live_children_on_keyboard_interrupt(tmp_path: Path):
    """A Ctrl-C mid-run must not orphan a live worker: the still-running
    child is terminated in the dispatcher's cleanup path."""
    ctx = mp.get_context("spawn")

    class _KbOnFirstJoin:
        def __init__(self, real):
            self._real = real
            self._raised = False

        def start(self):
            self._real.start()

        def join(self, *a, **k):
            if not self._raised:
                self._raised = True
                raise KeyboardInterrupt
            return self._real.join(*a, **k)

        def terminate(self):
            self._real.terminate()

        def is_alive(self):
            return self._real.is_alive()

        @property
        def exitcode(self):
            return self._real.exitcode

        @property
        def pid(self):
            return self._real.pid

    holder: dict = {}

    def _factory(**kw):
        real = ctx.Process(**kw)
        wrapped = _KbOnFirstJoin(real)
        holder.setdefault("procs", []).append((wrapped, real))
        return wrapped

    with pytest.raises(KeyboardInterrupt):
        local_pool.dispatch_local_process_pool(
            tasks=[{"instance_id": "s0", "selected_database": "db_a"}],
            cfg={}, run_dir=str(tmp_path), attempt=1,
            run_id="R", concurrency=1, worker=workers.slow_worker,
            process_factory=_factory,
        )
    _wrapped, real = holder["procs"][0]
    assert not real.is_alive(), "live child was not terminated on interrupt"


# ---------------------------------------------------------------------------
# The REAL worker body — exercised in-process so monkeypatches apply.
# ---------------------------------------------------------------------------


def _minimal_cfg() -> dict:
    return {
        "framework": "pydantic_ai", "query_mode": "raw", "mode": "c-interact",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 3, "strict": False, "use_audited_gold_sql": False,
        "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
        "slayer_storage_root": None, "dataset": "mini-interact",
        "data_dir": "/data/mini-interact",
    }


def test_local_task_worker_writes_error_row_on_soft_failure(tmp_path, monkeypatch):
    """A normal exception inside the task must be caught by the worker body
    and persisted as an ERROR row (not crash the process)."""
    from bird_interact_agents.cloud import ray_app

    async def boom(task_data, **_kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", boom)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    local_pool._local_task_worker(
        _minimal_cfg(),
        {"instance_id": "db_a_1", "selected_database": "db_a"},
        str(tmp_path), 1, "R",
    )
    row_path = tmp_path / "rows" / "db_a_1" / "attempt-1.json"
    assert row_path.exists()
    assert json.loads(row_path.read_text()).get("error")


def test_local_task_worker_success_writes_row_and_annotation(tmp_path, monkeypatch):
    from bird_interact_agents.cloud import ray_app

    async def ok(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a", "phase1_passed": True, "phase2_passed": False,
            "total_reward": 1.0, "duration_s": 0.01, "error": None,
            "submitted_sql": "SELECT 1",
        }

    def fake_grade(*, task_data, rows_dir, run_id, **_kw):
        d = Path(rows_dir) / task_data["instance_id"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / "submission_annotation.json"
        p.write_text(json.dumps({"evaluation": {"verdict": "correct"}}))
        return p

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", ok)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    monkeypatch.setattr(ray_app, "_grade_one_submission", fake_grade, raising=True)

    local_pool._local_task_worker(
        _minimal_cfg(),
        {"instance_id": "db_a_1", "selected_database": "db_a", "amb_user_query": "q"},
        str(tmp_path), 1, "R",
    )
    rowdir = tmp_path / "rows" / "db_a_1"
    assert (rowdir / "attempt-1.json").exists()
    assert (rowdir / "submission_annotation.json").exists()
