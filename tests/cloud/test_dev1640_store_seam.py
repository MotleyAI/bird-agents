"""DEV-1640: ``_run_one_in_actor`` runs against an injected
``PersistenceStore`` (not a hard-wired ``gcs_client``): it uses the store
for the row, annotation, LOG and PARTIAL TRANSCRIPT, stamps
``started_at`` / ``user_query`` onto the row (Codex H3), and forwards
``task_start_ts`` (== the row's ``started_at``) plus the OTF upload-back
state to ``store.upload_back`` (Codex M1/M4).
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest

from bird_interact_agents.cloud import ray_app


class _RecordingStore:
    """Captures every seam call; satisfies the duck-typed store surface."""

    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self.annotations: list[tuple] = []
        self.logs: list[tuple] = []
        self.partials: list[tuple] = []
        self.upload_backs: list[dict] = []

    def write_row(self, run_id, iid, attempt, row):
        self.rows.append((run_id, iid, attempt, dict(row)))

    def write_submission_annotation(self, run_id, iid, annotation):
        self.annotations.append((run_id, iid, annotation))

    def write_log(self, run_id, iid, attempt, log_bytes):
        self.logs.append((run_id, iid, attempt, log_bytes))

    def write_partial_transcript(self, run_id, iid, data):
        self.partials.append((run_id, iid, data))

    def upload_back(self, run_id, cfg, iid, attempt, *, task_start_ts,
                    uploaded_dbs, initial_seed_fp_by_db):
        self.upload_backs.append({
            "iid": iid, "task_start_ts": task_start_ts,
            "uploaded_dbs": uploaded_dbs,
            "initial_seed_fp_by_db": initial_seed_fp_by_db,
        })


def _minimal_cfg() -> dict:
    return {
        "framework": "pydantic_ai",  # not claude_sdk -> no partial uploader
        "query_mode": "raw",
        "mode": "c-interact",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 3,
        "strict": False,
        "use_audited_gold_sql": False,
        "prompt_cache": True,
        "max_depth": 3,
        "slayer_setup": "on-the-fly",
        "slayer_storage_root": None,
        "dataset": "mini-interact",
        "data_dir": "/data/mini-interact",
    }


def _patch_grade(monkeypatch):
    def fake_grade(*, task_data, rows_dir, run_id, **_kw):
        d = Path(rows_dir) / task_data["instance_id"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / "submission_annotation.json"
        p.write_text(json.dumps({"evaluation": {"verdict": "correct"}}))
        return p

    monkeypatch.setattr(ray_app, "_grade_one_submission", fake_grade, raising=True)


@pytest.fixture
def _fake_task(monkeypatch):
    async def fake_run_one_task(task_data, **_kw):
        # Write to the OS-level fd 1 (which _run_one_in_actor's fd_capture
        # redirects to the task log) so the LOG seam is exercised — a
        # Python-level print would be swallowed by pytest's capture instead.
        os.write(1, b"task log line\n")
        return {
            "instance_id": task_data["instance_id"],
            "database": task_data.get("selected_database", "db_a"),
            "phase1_passed": True,
            "phase2_passed": False,
            "total_reward": 1.0,
            "duration_s": 0.02,
            "error": None,
            "submitted_sql": "SELECT 1",
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task,
    )
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    _patch_grade(monkeypatch)
    # If the seam is broken and falls back to gcs.*, make that fail loudly.
    from bird_interact_agents.cloud import gcs as _gcs
    for name in ("write_row", "write_log", "write_submission_annotation",
                 "write_partial_transcript"):
        monkeypatch.setattr(
            _gcs, name,
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must go through the store, not gcs.*")
            ),
            raising=True,
        )


def _run(store, **overrides):
    td = {
        "instance_id": "db_a_1",
        "selected_database": "db_a",
        "amb_user_query": "How many aliens?",
    }
    cfg = _minimal_cfg()
    cfg.update(overrides.pop("cfg", {}))
    return ray_app._run_one_in_actor(
        task_data=td, cfg=cfg, run_id="R1", attempt=1, store=store,
    )


def test_run_one_in_actor_uses_injected_store_for_row_and_annotation(_fake_task):
    store = _RecordingStore()
    iid = _run(store)
    assert iid == "db_a_1"
    assert len(store.rows) == 1 and store.rows[0][1] == "db_a_1"
    assert len(store.annotations) == 1


def test_run_one_in_actor_uses_store_for_log(_fake_task):
    store = _RecordingStore()
    _run(store)
    assert store.logs, "captured task log must be persisted via store.write_log"
    assert b"task log line" in store.logs[0][3]


def test_run_one_in_actor_stamps_started_at_and_user_query(_fake_task):
    store = _RecordingStore()
    _run(store)
    (_run_id, _iid, _attempt, row) = store.rows[0]
    assert row.get("started_at", 0) > 0, "started_at must be stamped from task_start_ts"
    assert row.get("user_query") == "How many aliens?"


def test_run_one_in_actor_forwards_full_state_to_upload_back(_fake_task):
    store = _RecordingStore()
    _run(store)
    assert len(store.upload_backs) == 1
    ub = store.upload_backs[0]
    (_run_id, _iid, _attempt, row) = store.rows[0]
    # task_start_ts forwarded to upload_back IS the row's started_at.
    assert ub["task_start_ts"] == row["started_at"]
    assert ub["uploaded_dbs"] is not None
    assert ub["initial_seed_fp_by_db"] is not None


def test_run_one_in_actor_partial_transcript_uses_store(monkeypatch):
    """For claude_sdk* frameworks on a store WITHOUT a local partial path
    (the cloud ``GcsStore`` shape — ``_RecordingStore`` here has no
    ``partial_transcript_local_path``), the streamed partial transcript MUST go
    through ``store.write_partial_transcript`` (not gcs.*). DEV-1642: a
    ``LocalFsStore`` instead appends per-message to ``rows/<iid>/`` and bypasses
    this snapshot — see ``tests/cloud/test_dev1642_local_streaming.py``."""
    store = _RecordingStore()
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    _patch_grade(monkeypatch)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": False, "total_reward": 1.0,
            "duration_s": 0.01, "error": None, "submitted_sql": "SELECT 1",
        }

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", fake_run_one_task)

    from bird_interact_agents.agents.claude_sdk import sdk_env

    def fake_record(uploader):
        @contextlib.contextmanager
        def _cm():
            uploader({"turn": 1, "text": "hello"})
            yield
        return _cm()

    monkeypatch.setattr(sdk_env, "record_partial_transcript", fake_record)

    cfg = _minimal_cfg()
    cfg["framework"] = "claude_sdk"
    ray_app._run_one_in_actor(
        task_data={"instance_id": "db_a_1", "selected_database": "db_a"},
        cfg=cfg, run_id="R1", attempt=1, store=store,
    )
    assert store.partials, "partial transcript must be persisted via the store"
