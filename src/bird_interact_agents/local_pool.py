"""DEV-1640: process-per-task local dispatch.

The local ``run_evaluation`` used to run ``--concurrency N`` as coroutines
under a SINGLE ``asyncio`` event loop. A blocking / deadlocking call in the
SLayer per-task path (the MCP stdio subprocess bridge under ≥3 concurrent
sessions, sync sqlite, per-DB locks) stalled the WHOLE loop, freezing every
agent at 0% CPU. Cloud never hit this because Ray actors are separate OS
processes.

This module gives the local runner the same isolation WITHOUT Ray: each
task runs in its own ``spawn``-ed worker process (its own event loop + its
own ``claude`` / MCP subprocesses), capped at ``concurrency`` by a
``ThreadPoolExecutor`` whose threads each own one child at a time. The
worker reuses the exact cloud per-task body (:func:`ray_app._run_one_in_actor`)
against a :class:`~bird_interact_agents.cloud.persistence.LocalFsStore`, so
the task body + artifact layout + grading converge with cloud; only the
dispatch/isolation mechanism differs.

``spawn`` (never ``fork``) is mandatory: the parent holds an asyncio loop,
open sqlite handles and threads that ``fork`` would corrupt. ``spawn``
children inherit the parent's ``os.environ`` at start, so the
subscription-auth signal, the bridge base-url override and ``BIRD_PG_*``
(all set in ``run.main`` BEFORE the pool starts) reach the workers with no
extra plumbing.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _local_task_worker(cfg: dict, task_data: dict, run_dir: str, attempt: int,
                       run_id: str) -> str:
    """Top-level (picklable) worker body run in a spawned child process.

    Builds a local persistence backend and delegates to the shared cloud
    per-task body. Does NOT bring up the bridge proxy — the parent already
    started it and the base-url override is inherited via ``os.environ``.
    """
    # Imported here (not at module top) purely to keep the child's import of
    # this module cheap when a caller only needs the dispatcher helpers; the
    # cloud extra is always present for a local run (the ``all`` extra pulls
    # ray + google-cloud-storage).
    from bird_interact_agents.cloud.persistence import LocalFsStore
    from bird_interact_agents.cloud.ray_app import _run_one_in_actor

    store = LocalFsStore(run_dir)
    return _run_one_in_actor(
        task_data=task_data,
        cfg=cfg,
        run_id=run_id,
        attempt=attempt,
        store=store,
        cached_runner=None,
        uploaded_dbs=set(),
        initial_seed_fp_by_db={},
    )


def terminate_all(procs) -> None:
    """Terminate + reap a batch of worker processes. Defensive: never raises
    (an unstarted or already-dead process is skipped). Used on cancellation
    (Ctrl-C) so a stalled task can't orphan its child process."""
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
    for p in procs:
        try:
            p.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def dispatch_local_process_pool(
    *,
    tasks: list[dict],
    cfg: dict,
    run_dir: str,
    attempt: int,
    run_id: str,
    concurrency: int,
    worker: Callable[..., Any] = _local_task_worker,
    mp_context: Any = None,
    process_factory: Callable[..., Any] | None = None,
) -> list[dict]:
    """Run every task in its own ``spawn``-ed process, ≤ ``concurrency`` at a
    time, and return a per-task ``{"instance_id", "row_written", "exitcode"}``
    status list.

    The worker communicates results ONLY via the filesystem (the row blob it
    writes through ``LocalFsStore``); its return value is discarded across the
    process boundary. A hard crash (segfault / OOM-kill / ``os._exit``) leaves
    no row blob — reported as ``row_written=False`` and turned into an error
    row by :func:`reconcile_local_run`. Sibling tasks are unaffected.
    """
    ctx = mp_context or mp.get_context("spawn")
    factory = process_factory or (lambda **kw: ctx.Process(**kw))
    run_dir_p = Path(run_dir)

    live: set = set()
    live_lock = threading.Lock()

    def _run_one(td: dict) -> dict:
        iid = str(td.get("instance_id") or "")
        p = factory(target=worker, args=(cfg, td, str(run_dir_p), attempt, run_id))
        with live_lock:
            live.add(p)
        p.start()
        # A KeyboardInterrupt raised here (Ctrl-C) leaves ``p`` in ``live`` on
        # purpose, so the outer cleanup can terminate it.
        p.join()
        with live_lock:
            live.discard(p)
        row_written = (
            run_dir_p / "rows" / iid / f"attempt-{attempt}.json"
        ).exists()
        return {"instance_id": iid, "row_written": row_written,
                "exitcode": p.exitcode}

    results: list[dict] = []
    ex = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    try:
        # Submit INSIDE the try so a mid-submission failure/interrupt still
        # hits the finally cleanup (no leaked executor / orphaned child).
        futures = [ex.submit(_run_one, td) for td in tasks]
        for fut in futures:
            results.append(fut.result())
        return results
    finally:
        # On ANY exit — success, Ctrl-C, or a dispatch-side error — terminate
        # any child still alive so a stalled task can't orphan its process
        # (their worker threads' join() then unblocks), and tear the executor
        # down WITHOUT waiting on wedged join() calls. For this to catch a
        # real Ctrl-C, the caller runs this synchronously (not via to_thread),
        # so the KeyboardInterrupt lands in fut.result() above.
        with live_lock:
            procs = list(live)
        if procs:
            terminate_all(procs)
        ex.shutdown(wait=False, cancel_futures=True)


def reconcile_local_run(
    *,
    run_dir: str | Path,
    tasks: list[dict],
    benchmark: str,
    run_id: str,
    attempt: int,
    submission_config: Any = None,
) -> None:
    """Close the ONE gap process isolation introduces: a hard-crashed worker
    that died before persisting anything.

    Graded and never-submitted tasks already populate the ``runs/`` golden
    store from inside the worker (``grade_one_submission`` /
    ``write_failed_submission_annotation`` both call ``_write_to_runs``). Only
    a task with NO row blob crashed hard. For each, write an error row (so
    ``collate`` counts it in ``results.db``) and write / OVERWRITE a
    fail-everything ``runs/`` annotation (so the cascade counts it, repairing
    any stale success verdict left by a crash-after-grade). Tasks that DID
    persist a row are left exactly as the worker wrote them.
    """
    # Imported here to avoid a run-time import cycle (ray_app imports lots of
    # cloud machinery; run.py imports this module at top level).
    from bird_interact_agents.cloud.ray_app import _build_error_row
    from bird_interact_agents.eval.grade_in_place import (
        write_failed_submission_annotation,
    )

    run_dir_p = Path(run_dir)
    rows_dir = run_dir_p / "rows"
    for td in tasks:
        iid = str(td.get("instance_id") or "")
        db = str(td.get("selected_database") or "")
        row_path = rows_dir / iid / f"attempt-{attempt}.json"
        if row_path.exists():
            continue  # worker persisted a row -> already counted everywhere
        _msg = "worker process crashed before persisting a row"
        err = _build_error_row(iid, db, _msg)
        (rows_dir / iid).mkdir(parents=True, exist_ok=True)
        (rows_dir / iid / f"attempt-{attempt}.json").write_text(
            json.dumps(err, default=str)
        )
        try:
            # This ALSO writes the runs/ golden store (via _write_to_runs),
            # overwriting any stale annotation for this (iid, run_id).
            write_failed_submission_annotation(
                rows_dir=rows_dir,
                instance_id=iid,
                selected_database=db or "<unknown>",
                benchmark=benchmark,
                run_id=run_id,
                trajectory_path=f"rows/{iid}/attempt-{attempt}.json",
                failure_details=(
                    f"{_msg}; counted as 0-pass at every cascade tier"
                ),
                config=submission_config,
            )
        except Exception:  # noqa: BLE001 — reconciliation must never abort a run
            logger.exception(
                "reconcile_local_run: failed to write crash annotation for %s",
                iid,
            )
