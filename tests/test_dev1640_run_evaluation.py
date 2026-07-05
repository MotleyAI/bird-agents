"""DEV-1640: ``run_evaluation`` orchestration for the process-pool path.

Spawned workers can't see pytest monkeypatches, so these tests fake the
DISPATCHER (``dispatch_local_process_pool``) with an in-process stand-in
that writes the SAME row blobs + runs/ annotations a real worker would,
then assert the parent-side orchestration (cfg + manifest -> reconcile ->
collate -> emit_cascading_eval_json) produces ``results.db`` + ``eval.json``
with an honest cascade denominator, and that ``started_at`` / ``user_query``
survive collate. Also pins the ``BIRD_INTERACT_LOCAL_INPROCESS=1`` escape
hatch and that the parent does NOT run the legacy in-process machinery on
the default path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import bird_interact_agents.run as run_mod
from bird_interact_agents import paths as paths_mod
from bird_interact_agents.cloud.persistence import LocalFsStore
from bird_interact_agents.eval.annotation_io import (
    read_submission_annotation,
    run_annotation_path,
    write_run_annotation,
)
from bird_interact_agents.eval.grade_in_place import (
    write_failed_submission_annotation,
)

BENCHMARK = "mini-interact"


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    root = tmp_path / "runs_store"
    root.mkdir()
    monkeypatch.setattr(paths_mod, "runs_root", lambda: root)
    return root


def _fake_dispatch_writing_results(**kwargs):
    """Simulate the worker pool: write a graded row blob + a runs/
    annotation per task (as a real worker would via LocalFsStore +
    grade_one_submission)."""
    tasks = kwargs["tasks"]
    run_dir = Path(kwargs["run_dir"])
    run_id = kwargs["run_id"]
    attempt = kwargs["attempt"]
    store = LocalFsStore(run_dir)
    statuses = []
    for i, td in enumerate(tasks):
        iid = td["instance_id"]
        db = td["selected_database"]
        store.write_row(run_id, iid, attempt, {
            "instance_id": iid, "database": db,
            "phase1_passed": True, "phase2_passed": False,
            "total_reward": 1.0, "duration_s": 0.01,
            "started_at": 1000.0 + i, "user_query": td["amb_user_query"],
            "submitted_sql": "SELECT 1", "usage": {},
        })
        p = write_failed_submission_annotation(
            rows_dir=run_dir / "rows", instance_id=iid, selected_database=db,
            benchmark=BENCHMARK, run_id=run_id,
            trajectory_path=f"rows/{iid}/attempt-{attempt}.json",
            failure_details="graded",
        )
        ann = read_submission_annotation(p)
        write_run_annotation(
            ann,
            run_annotation_path(benchmark=BENCHMARK, selected_database=db,
                                instance_id=iid, run_id=run_id),
            benchmark=BENCHMARK, run_id=run_id, allow_manifest_fallback=False,
        )
        statuses.append({"instance_id": iid, "row_written": True})
    return statuses


def _tasks():
    return [
        {"instance_id": "db_a_1", "selected_database": "db_a",
         "amb_user_query": "q1", "sol_sql": ["SELECT 1"]},
        {"instance_id": "db_a_2", "selected_database": "db_a",
         "amb_user_query": "q2", "sol_sql": ["SELECT 2"]},
    ]


@pytest.mark.asyncio
async def test_process_pool_path_builds_results_db_and_cascade(
    tmp_path, monkeypatch, runs_root,
):
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **k: _tasks())
    monkeypatch.setattr(
        run_mod, "dispatch_local_process_pool", _fake_dispatch_writing_results,
    )

    out = tmp_path / "out" / "eval.json"
    await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored", output_path=str(out),
        mode="c-interact", query_mode="raw", framework="pydantic_ai",
        dataset=BENCHMARK, concurrency=2,
    )

    conn = sqlite3.connect(str(tmp_path / "out" / "results.db"))
    try:
        rows = conn.execute(
            "SELECT instance_id, started_at, user_query FROM task_results "
            "ORDER BY instance_id"
        ).fetchall()
    finally:
        conn.close()
    assert [r[0] for r in rows] == ["db_a_1", "db_a_2"]
    # H3: started_at / user_query survive the collate boundary.
    assert rows[0][1] == 1000.0
    assert rows[0][2] == "q1"

    on_disk = json.loads(out.read_text())
    # The runs/ handoff worked: the cascade denominator counts BOTH tasks.
    assert on_disk["cascading_phase1"]["n_dual_eval_tasks"] == 2


@pytest.mark.asyncio
async def test_process_pool_cascade_excludes_stale_out_of_run_annotations(
    tmp_path, monkeypatch, runs_root,
):
    """A filtered rerun reusing the same run_id must NOT fold a prior task's
    stale runs/ annotation into the cascade denominator (Codex: emit must be
    instance_filter-scoped)."""
    tasks = [
        {"instance_id": "db_a_1", "selected_database": "db_a",
         "amb_user_query": "q1", "sol_sql": ["SELECT 1"]},
    ]
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(
        run_mod, "dispatch_local_process_pool", _fake_dispatch_writing_results,
    )

    out = tmp_path / "out" / "eval.json"
    run_id = out.parent.name  # run_evaluation derives run_id from output_dir.name

    # Seed a stale annotation for an instance NOT in this run, same run_id.
    stale = write_failed_submission_annotation(
        rows_dir=tmp_path / "_stale", instance_id="db_a_99",
        selected_database="db_a", benchmark=BENCHMARK, run_id=run_id,
        trajectory_path="rows/db_a_99/attempt-1.json", failure_details="stale",
    )
    ann = read_submission_annotation(stale)
    write_run_annotation(
        ann,
        run_annotation_path(benchmark=BENCHMARK, selected_database="db_a",
                            instance_id="db_a_99", run_id=run_id),
        benchmark=BENCHMARK, run_id=run_id, allow_manifest_fallback=False,
    )

    await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored", output_path=str(out),
        mode="c-interact", query_mode="raw", framework="pydantic_ai",
        dataset=BENCHMARK, concurrency=1,
    )
    on_disk = json.loads(out.read_text())
    # Only the ONE current task counts — the stale db_a_99 is filtered out.
    assert on_disk["cascading_phase1"]["n_dual_eval_tasks"] == 1


@pytest.mark.asyncio
async def test_process_pool_is_default_and_bypasses_legacy_machinery(
    tmp_path, monkeypatch, runs_root,
):
    """The default path dispatches through the process pool and must NOT
    touch the legacy in-parent ``_make_runner`` / ``insert_task_result``,
    and must thread the caller's ``data_dir`` into the worker cfg."""
    monkeypatch.delenv("BIRD_INTERACT_LOCAL_INPROCESS", raising=False)
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **k: _tasks())

    seen = {}

    def _spy(**kwargs):
        seen["called"] = True
        seen["cfg"] = kwargs["cfg"]
        seen["task_ids"] = [t["instance_id"] for t in kwargs["tasks"]]
        seen["concurrency"] = kwargs["concurrency"]
        return _fake_dispatch_writing_results(**kwargs)

    monkeypatch.setattr(run_mod, "dispatch_local_process_pool", _spy)

    def _boom_runner(*a, **k):
        raise AssertionError("parent must NOT build a runner on the process path")

    def _boom_insert(*a, **k):
        raise AssertionError("parent must NOT insert_task_result on the process path")

    monkeypatch.setattr(run_mod, "_make_runner", _boom_runner)
    monkeypatch.setattr(run_mod, "insert_task_result", _boom_insert)

    out = tmp_path / "out" / "eval.json"
    await run_mod.run_evaluation(
        data_path="ignored", data_dir="/my/data/root", output_path=str(out),
        mode="c-interact", query_mode="raw", framework="pydantic_ai",
        dataset=BENCHMARK, concurrency=3,
    )
    assert seen["called"] is True
    assert seen["concurrency"] == 3
    assert seen["task_ids"] == ["db_a_1", "db_a_2"]
    # Codex H1: the worker cfg carries the caller's data_dir, not a cloud default.
    assert seen["cfg"]["data_dir"] == "/my/data/root"


@pytest.mark.asyncio
async def test_escape_hatch_uses_legacy_inprocess_path(tmp_path, monkeypatch, runs_root):
    """BIRD_INTERACT_LOCAL_INPROCESS=1 keeps the legacy single-loop path
    (for debugging) and must NOT touch the process pool."""
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")

    from bird_interact_agents import usage as usage_mod
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    tasks = [
        {"instance_id": "t1", "selected_database": "fake", "amb_user_query": "q1"},
    ]
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(run_mod, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        return {
            "task_id": td["instance_id"], "instance_id": td["instance_id"],
            "database": "fake", "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "trajectory": [], "error": None,
            "usage": usage_mod.TokenUsage().model_dump(),
        }

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    def _boom(**kwargs):
        raise AssertionError("process pool must NOT run under the escape hatch")

    monkeypatch.setattr(run_mod, "dispatch_local_process_pool", _boom)

    out = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        data_path="ignored", data_dir="ignored", output_path=str(out),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
    )
    assert out.exists()
    assert metrics is not None
