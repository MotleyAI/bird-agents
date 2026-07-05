"""DEV-1640: the persistence-backend seam (``cloud/persistence.py``).

Two implementations of ``PersistenceStore`` share the per-task body
``_run_one_in_actor``:

* ``GcsStore`` — the cloud backend; every method delegates to the
  module-level ``gcs.*`` / ``upload_back.*`` functions (so existing
  monkeypatch-based cloud tests stay green) and is behaviourally
  identical to today's inline ``_gcs.*`` calls.
* ``LocalFsStore`` — the local backend; writes the SAME on-disk layout
  the cloud row blobs use (``rows/<iid>/attempt-<n>.json`` etc.) so
  ``collate()`` can build ``results.db`` + ``eval.json`` unchanged.

These tests pin the seam contract. They import ``persistence`` before any
ray-skip so a missing cloud extra fails-for-the-right-reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_interact_agents.cloud import persistence


# ---------------------------------------------------------------------------
# GcsStore — delegates to the module-level gcs.* / upload_back.* functions.
# ---------------------------------------------------------------------------


def test_gcs_store_write_row_delegates(monkeypatch, fake_gcs_bucket):
    client, _store = fake_gcs_bucket
    calls: list[tuple] = []

    from bird_interact_agents.cloud import gcs as _gcs
    monkeypatch.setattr(
        _gcs, "write_row",
        lambda run_id, iid, attempt, row, client=None: calls.append(
            ("write_row", run_id, iid, attempt, row, client)
        ),
        raising=True,
    )

    store = persistence.GcsStore(client)
    store.write_row("R", "iid1", 1, {"instance_id": "iid1"})

    assert len(calls) == 1
    name, run_id, iid, attempt, row, passed_client = calls[0]
    assert (run_id, iid, attempt) == ("R", "iid1", 1)
    assert row == {"instance_id": "iid1"}
    # The store threads ITS client through to the module function.
    assert passed_client is client


def test_gcs_store_write_submission_annotation_delegates(monkeypatch, fake_gcs_bucket):
    client, _store = fake_gcs_bucket
    calls: list[tuple] = []
    from bird_interact_agents.cloud import gcs as _gcs
    monkeypatch.setattr(
        _gcs, "write_submission_annotation",
        lambda run_id, iid, ann, client=None: calls.append((run_id, iid, ann, client)),
        raising=True,
    )
    persistence.GcsStore(client).write_submission_annotation("R", "iid1", {"a": 1})
    assert calls == [("R", "iid1", {"a": 1}, client)]


def test_gcs_store_write_log_and_partial_transcript_delegate(monkeypatch, fake_gcs_bucket):
    client, _store = fake_gcs_bucket
    log_calls: list[tuple] = []
    pt_calls: list[tuple] = []
    from bird_interact_agents.cloud import gcs as _gcs
    monkeypatch.setattr(
        _gcs, "write_log",
        lambda run_id, iid, attempt, data, client=None: log_calls.append(
            (run_id, iid, attempt, data, client)
        ),
        raising=True,
    )
    monkeypatch.setattr(
        _gcs, "write_partial_transcript",
        lambda run_id, iid, data, client=None: pt_calls.append(
            (run_id, iid, data, client)
        ),
        raising=True,
    )
    s = persistence.GcsStore(client)
    s.write_log("R", "iid1", 2, b"logbytes")
    s.write_partial_transcript("R", "iid1", "line\n")
    assert log_calls == [("R", "iid1", 2, b"logbytes", client)]
    assert pt_calls == [("R", "iid1", "line\n", client)]


def test_gcs_store_upload_back_runs_the_dev1470_triple(monkeypatch, fake_gcs_bucket):
    """``upload_back`` must run all three DEV-1470 helpers in order and
    forward ``task_start_ts`` to the setup-sessions helper (Codex M1)."""
    client, _store = fake_gcs_bucket
    from bird_interact_agents.cloud import upload_back as _ub

    order: list[str] = []
    captured: dict = {}

    def _debug(**kw):
        order.append("debug")

    def _sessions(**kw):
        order.append("sessions")
        captured["task_start_ts"] = kw.get("task_start_ts")

    def _delta(**kw):
        order.append("delta")
        captured["uploaded_dbs"] = kw.get("uploaded_dbs")
        captured["initial_seed_fp_by_db"] = kw.get("initial_seed_fp_by_db")

    monkeypatch.setattr(_ub, "upload_per_task_debug", _debug, raising=True)
    monkeypatch.setattr(_ub, "upload_per_task_setup_sessions", _sessions, raising=True)
    monkeypatch.setattr(_ub, "upload_otf_reference_delta", _delta, raising=True)

    s = persistence.GcsStore(client)
    s.upload_back(
        "R", {"query_mode": "slayer"}, "iid1", 1,
        task_start_ts=123.5,
        uploaded_dbs={"db_a"},
        initial_seed_fp_by_db={"db_a": "fp"},
    )
    assert order == ["debug", "sessions", "delta"]
    assert captured["task_start_ts"] == 123.5
    assert captured["uploaded_dbs"] == {"db_a"}
    assert captured["initial_seed_fp_by_db"] == {"db_a": "fp"}


# ---------------------------------------------------------------------------
# LocalFsStore — writes the GCS-mirroring on-disk layout; upload_back no-op.
# ---------------------------------------------------------------------------


def test_local_fs_store_write_row(tmp_path: Path):
    store = persistence.LocalFsStore(tmp_path)
    store.write_row("R", "iid1", 1, {"instance_id": "iid1", "phase1_passed": True})
    p = tmp_path / "rows" / "iid1" / "attempt-1.json"
    assert p.exists()
    assert json.loads(p.read_text())["phase1_passed"] is True


def test_local_fs_store_write_row_is_atomic_no_tmp_left(tmp_path: Path):
    store = persistence.LocalFsStore(tmp_path)
    store.write_row("R", "iid1", 1, {"instance_id": "iid1"})
    rowdir = tmp_path / "rows" / "iid1"
    # tmp+rename: only the final file remains, no ``.tmp`` sibling.
    leftovers = [p.name for p in rowdir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_local_fs_store_write_submission_annotation(tmp_path: Path):
    store = persistence.LocalFsStore(tmp_path)
    store.write_submission_annotation("R", "iid1", {"evaluation": {"verdict": "correct"}})
    p = tmp_path / "rows" / "iid1" / "submission_annotation.json"
    assert json.loads(p.read_text())["evaluation"]["verdict"] == "correct"


def test_local_fs_store_write_log_and_partial_transcript(tmp_path: Path):
    store = persistence.LocalFsStore(tmp_path)
    store.write_log("R", "iid1", 1, b"some log bytes")
    store.write_partial_transcript("R", "iid1", '{"turn": 1}\n')
    rowdir = tmp_path / "rows" / "iid1"
    assert (rowdir / "task-1.log").read_bytes() == b"some log bytes"
    assert (rowdir / "partial_transcript.jsonl").read_text() == '{"turn": 1}\n'


def test_local_fs_store_upload_back_is_noop(tmp_path: Path):
    store = persistence.LocalFsStore(tmp_path)
    # Must not raise and must not create any GCS-ish artefacts.
    store.upload_back(
        "R", {"query_mode": "slayer"}, "iid1", 1,
        task_start_ts=1.0, uploaded_dbs=set(), initial_seed_fp_by_db={},
    )
    # No rows/ dir is created by upload_back alone.
    assert not (tmp_path / "rows").exists() or list((tmp_path / "rows").iterdir()) == []


def test_both_stores_satisfy_the_protocol():
    """Both concrete stores are ``PersistenceStore`` subclasses so the
    shared per-task body can type against the ABC."""
    assert issubclass(persistence.GcsStore, persistence.PersistenceStore)
    assert issubclass(persistence.LocalFsStore, persistence.PersistenceStore)
