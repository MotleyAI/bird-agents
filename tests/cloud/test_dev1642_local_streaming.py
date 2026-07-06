"""DEV-1642: the LOCAL per-task body appends the claude_sdk trajectory to
``rows/<iid>/partial_transcript.jsonl`` as each message streams (no throttle,
no temp-dir hop, no store snapshot), while the CLOUD path keeps its throttled
full-snapshot GCS re-upload. The divergence lives in the ``PersistenceStore``
seam (``partial_transcript_local_path``) and is selected by
``ray_app._build_partial_transcript_recorder``.
"""
from __future__ import annotations

import json
from pathlib import Path

from bird_interact_agents.cloud import ray_app
from bird_interact_agents.cloud.persistence import (
    GcsStore,
    LocalFsStore,
    PersistenceStore,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import LocalTranscriptAppender


# --------------------------------------------------------------------------
# persistence seam: where does the durable local partial transcript live?
# --------------------------------------------------------------------------


def test_localfsstore_advertises_rows_partial_path(tmp_path):
    store = LocalFsStore(tmp_path)
    dest = store.partial_transcript_local_path("db_a_1")
    assert dest == tmp_path / "rows" / "db_a_1" / "partial_transcript.jsonl"
    # The row dir is created so the append can open the file immediately.
    assert dest.parent.is_dir()


def test_gcsstore_has_no_local_partial_path():
    assert GcsStore(object()).partial_transcript_local_path("db_a_1") is None


def test_base_store_default_local_partial_path_is_none():
    # A minimal concrete subclass exercising ONLY the base default.
    class _Bare(PersistenceStore):
        def write_row(self, *a, **k):
            ...

        def write_submission_annotation(self, *a, **k):
            ...

        def write_log(self, *a, **k):
            ...

        def write_partial_transcript(self, *a, **k):
            ...

        def upload_back(self, *a, **k):
            ...

    assert _Bare().partial_transcript_local_path("iid") is None


# --------------------------------------------------------------------------
# recorder selection: local -> append-only; cloud/unknown -> throttled uploader
# --------------------------------------------------------------------------


def test_recorder_selection_local_is_append_only(tmp_path):
    store = LocalFsStore(tmp_path)
    recorder, flush = ray_app._build_partial_transcript_recorder(
        store=store, run_id="R1", iid="db_a_1", log_dir=tmp_path / "log",
    )
    assert isinstance(recorder, LocalTranscriptAppender)
    assert recorder.path == tmp_path / "rows" / "db_a_1" / "partial_transcript.jsonl"
    # Append is the durable write; there is nothing to flush at the end.
    assert flush is None


def test_recorder_selection_cloud_is_throttled_uploader(tmp_path):
    store = GcsStore(object())
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    recorder, flush = ray_app._build_partial_transcript_recorder(
        store=store, run_id="R1", iid="db_a_1", log_dir=log_dir,
    )
    assert isinstance(recorder, ray_app.PartialTranscriptUploader)
    assert flush == recorder.flush
    # The throttled uploader still scratches to the temp log dir (not rows/).
    assert recorder.local_path == log_dir / "partial_transcript.jsonl"


def test_recorder_selection_store_without_method_falls_back_to_uploader(tmp_path):
    class _StoreNoMethod:
        def write_partial_transcript(self, *a, **k):
            ...

    log_dir = tmp_path / "log"
    log_dir.mkdir()
    recorder, flush = ray_app._build_partial_transcript_recorder(
        store=_StoreNoMethod(), run_id="R1", iid="db_a_1", log_dir=log_dir,
    )
    assert isinstance(recorder, ray_app.PartialTranscriptUploader)
    assert flush == recorder.flush
    assert recorder.local_path == log_dir / "partial_transcript.jsonl"


# --------------------------------------------------------------------------
# _run_one_in_actor: end-to-end local streaming through a real LocalFsStore
# --------------------------------------------------------------------------


def _minimal_cfg() -> dict:
    return {
        "framework": "claude_sdk",
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


def test_run_one_in_actor_local_appends_incrementally(monkeypatch, tmp_path):
    """The claude_sdk trajectory lands in rows/<iid>/partial_transcript.jsonl
    AS IT STREAMS (before the task returns / before attempt-1.json), one line
    per message, WITHOUT going through store.write_partial_transcript."""
    store = LocalFsStore(tmp_path)
    dest = tmp_path / "rows" / "db_a_1" / "partial_transcript.jsonl"

    # Spy: local append MUST NOT route through the store snapshot.
    snapshot_calls: list = []
    orig = store.write_partial_transcript

    def _spy(run_id, iid, data):
        snapshot_calls.append((iid, data))
        return orig(run_id, iid, data)

    monkeypatch.setattr(store, "write_partial_transcript", _spy)

    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    _patch_grade(monkeypatch)

    from bird_interact_agents.agents.claude_sdk import sdk_env

    async def fake_run_one_task(task_data, **_kw):
        sink = sdk_env._partial_transcript_sink.get()
        assert sink is not None, "the local recorder must be installed around the task"
        # First message: assert it is ON DISK immediately (incremental, not
        # buffered until task end).
        sink({"type": "AssistantMessage", "data": "msg1"})
        mid = dest.read_text(encoding="utf-8").splitlines()
        assert len(mid) == 1 and json.loads(mid[0])["data"] == "msg1"
        assert not (dest.parent / "attempt-1.json").exists(), (
            "partial must precede the finalized attempt row"
        )
        sink({"type": "ResultMessage", "data": "msg2"})
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": False, "total_reward": 1.0,
            "duration_s": 0.01, "error": None, "submitted_sql": "SELECT 1",
        }

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", fake_run_one_task)

    ray_app._run_one_in_actor(
        task_data={"instance_id": "db_a_1", "selected_database": "db_a"},
        cfg=_minimal_cfg(), run_id="R1", attempt=1, store=store,
    )

    lines = dest.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["data"] for x in lines] == ["msg1", "msg2"]
    # The local branch never touches the throttled store snapshot.
    assert snapshot_calls == []
    # Finalized row still written afterwards.
    assert (dest.parent / "attempt-1.json").exists()


def test_run_one_in_actor_cloud_still_uses_store_snapshot(monkeypatch, tmp_path):
    """Regression: a store WITHOUT partial_transcript_local_path (the cloud
    GcsStore shape) keeps the throttled PartialTranscriptUploader path, i.e.
    store.write_partial_transcript IS called."""
    partials: list = []

    class _RecordingStore:
        def write_row(self, run_id, iid, attempt, row):
            ...

        def write_submission_annotation(self, run_id, iid, annotation):
            ...

        def write_log(self, run_id, iid, attempt, log_bytes):
            ...

        def write_partial_transcript(self, run_id, iid, data):
            partials.append((iid, data))

        def upload_back(self, *a, **k):
            ...

    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    _patch_grade(monkeypatch)

    from bird_interact_agents.agents.claude_sdk import sdk_env

    async def fake_run_one_task(task_data, **_kw):
        sink = sdk_env._partial_transcript_sink.get()
        assert sink is not None
        sink({"type": "AssistantMessage", "data": "cloudmsg"})
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": False, "total_reward": 1.0,
            "duration_s": 0.01, "error": None, "submitted_sql": "SELECT 1",
        }

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", fake_run_one_task)

    ray_app._run_one_in_actor(
        task_data={"instance_id": "db_a_1", "selected_database": "db_a"},
        cfg=_minimal_cfg(), run_id="R1", attempt=1, store=_RecordingStore(),
    )
    # The throttled uploader's flush() forces a final store snapshot.
    assert partials, "cloud-shaped store must persist the partial via write_partial_transcript"
    assert "cloudmsg" in partials[-1][1]


def test_run_one_in_actor_non_claude_framework_installs_no_recorder(monkeypatch, tmp_path):
    """A non-claude_sdk framework gets NO partial-transcript recorder — no sink
    is installed and no partial_transcript.jsonl is written."""
    store = LocalFsStore(tmp_path)
    dest = tmp_path / "rows" / "db_a_1" / "partial_transcript.jsonl"
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    _patch_grade(monkeypatch)

    from bird_interact_agents.agents.claude_sdk import sdk_env

    async def fake_run_one_task(task_data, **_kw):
        assert sdk_env._partial_transcript_sink.get() is None, (
            "non-claude frameworks must not install the partial-transcript sink"
        )
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": False, "total_reward": 1.0,
            "duration_s": 0.01, "error": None, "submitted_sql": "SELECT 1",
        }

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", fake_run_one_task)

    cfg = _minimal_cfg()
    cfg["framework"] = "pydantic_ai"
    ray_app._run_one_in_actor(
        task_data={"instance_id": "db_a_1", "selected_database": "db_a"},
        cfg=cfg, run_id="R1", attempt=1, store=store,
    )
    assert not dest.exists()
