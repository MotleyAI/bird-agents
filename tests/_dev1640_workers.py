"""Top-level, picklable stub workers for the DEV-1640 process-pool tests.

These live in a dedicated importable module (NOT the test module) so a
``spawn``-started child can re-import them by qualified name. Each mirrors
the ``(cfg, task_data, run_dir, attempt, run_id)`` signature of the real
``local_pool._local_task_worker`` and communicates results ONLY via the
filesystem (row blobs) — exactly like the real worker, whose return value
is discarded across the process boundary.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _row_path(run_dir, iid: str, attempt: int) -> Path:
    d = Path(run_dir) / "rows" / iid
    d.mkdir(parents=True, exist_ok=True)
    return d / f"attempt-{attempt}.json"


def interval_worker(cfg, task_data, run_dir, attempt, run_id):
    """Sleep briefly and record the [t0, t1] busy interval into the row
    blob, so a test can compute the peak number of overlapping workers."""
    iid = task_data["instance_id"]
    t0 = time.time()
    time.sleep(0.25)
    t1 = time.time()
    _row_path(run_dir, iid, attempt).write_text(
        json.dumps({"instance_id": iid, "t0": t0, "t1": t1})
    )


def branching_worker(cfg, task_data, run_dir, attempt, run_id):
    """Hard-crash (no row blob) when ``task_data['crash']`` is truthy;
    otherwise write a normal row blob. Proves crash isolation."""
    iid = task_data["instance_id"]
    if task_data.get("crash"):
        os._exit(1)  # simulate segfault/OOM-kill: no chance to persist
    _row_path(run_dir, iid, attempt).write_text(json.dumps({"instance_id": iid}))


def env_echo_worker(cfg, task_data, run_dir, attempt, run_id):
    """Echo selected env vars the parent set BEFORE spawn into the row
    blob, to prove ``spawn`` children inherit ``os.environ``."""
    iid = task_data["instance_id"]
    _row_path(run_dir, iid, attempt).write_text(
        json.dumps({
            "instance_id": iid,
            "BIRD_INTERACT_SUBSCRIPTION_AUTH": os.environ.get(
                "BIRD_INTERACT_SUBSCRIPTION_AUTH"
            ),
            "BIRD_PG_HOST": os.environ.get("BIRD_PG_HOST"),
            "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
        })
    )


def pid_worker(cfg, task_data, run_dir, attempt, run_id):
    """Record the worker's own pid into the row blob, so a test can prove
    tasks ran in child processes distinct from the parent."""
    iid = task_data["instance_id"]
    # A short sleep so sibling workers overlap -> distinct pids appear.
    time.sleep(0.1)
    _row_path(run_dir, iid, attempt).write_text(
        json.dumps({"instance_id": iid, "pid": os.getpid()})
    )


def barrier_worker(cfg, task_data, run_dir, attempt, run_id):
    """Rendezvous on a shared Barrier(parties=concurrency) BEFORE writing
    the row blob. If fewer than ``concurrency`` workers are ever alive at
    once (a serial implementation), the barrier times out, the worker
    raises, and no row blob is written — so a test asserting all blobs
    present deterministically proves real concurrency, immune to
    spawn-import jitter.
    """
    iid = task_data["instance_id"]
    cfg["barrier"].wait(timeout=60)
    _row_path(run_dir, iid, attempt).write_text(json.dumps({"instance_id": iid}))


def slow_worker(cfg, task_data, run_dir, attempt, run_id):
    """Sleep long enough to still be alive when a test terminates it."""
    time.sleep(30)
