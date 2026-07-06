"""In-flight transcript streaming: the claude_sdk recorder feeds each message
to a sink as it streams, and PartialTranscriptUploader appends locally + uploads
to GCS on a throttle — so a hung/slow task leaves an inspectable partial.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bird_interact_agents.cloud import gcs, ray_app
from bird_interact_agents.agents.claude_sdk import sdk_env


# --------------------------------------------------------------------------
# sdk_env: the central per-message hook
# --------------------------------------------------------------------------


class _FakeMsg:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return f"MSG:{self.text}"


class _FakeClient:
    """Minimal stand-in: receive_response() streams a fixed message list."""

    def __init__(self, msgs):
        self._msgs = msgs

    async def query(self, *a, **k):  # pragma: no cover - unused here
        return None

    def receive_response(self):
        async def _agen():
            for m in self._msgs:
                yield m
        return _agen()


def test_record_partial_transcript_feeds_each_message():
    got: list = []
    client = sdk_env._TranscriptClient(_FakeClient([_FakeMsg("a"), _FakeMsg("b")]))

    async def _drive():
        with sdk_env.record_partial_transcript(lambda d: got.append(d)):
            async for _ in client.receive_response():
                pass

    asyncio.run(_drive())
    assert [d["data"] for d in got] == ["MSG:a", "MSG:b"]
    # The transcript is still accumulated as before (sink is additive).
    assert len(client.transcript) == 2


def test_sink_exception_never_breaks_stream():
    seen = []

    def _boom(_d):
        raise RuntimeError("sink blew up")

    client = sdk_env._TranscriptClient(_FakeClient([_FakeMsg("x"), _FakeMsg("y")]))

    async def _drive():
        with sdk_env.record_partial_transcript(_boom):
            async for m in client.receive_response():
                seen.append(m)

    asyncio.run(_drive())  # must not raise
    assert len(seen) == 2


def test_no_sink_is_pure_noop():
    client = sdk_env._TranscriptClient(_FakeClient([_FakeMsg("z")]))

    async def _drive():
        async for _ in client.receive_response():
            pass

    asyncio.run(_drive())  # no contextvar set → no-op
    assert len(client.transcript) == 1


# --------------------------------------------------------------------------
# PartialTranscriptUploader: local append + throttled GCS upload
# --------------------------------------------------------------------------


def _patch_gcs(monkeypatch):
    uploads: list = []

    def _fake_write(run_id, instance_id, text, *, client=None):
        uploads.append((instance_id, text))

    monkeypatch.setattr(ray_app._gcs, "write_partial_transcript", _fake_write)
    return uploads


def test_uploader_appends_locally_and_throttles_uploads(monkeypatch, tmp_path):
    uploads = _patch_gcs(monkeypatch)
    p = tmp_path / "partial.jsonl"
    up = ray_app.PartialTranscriptUploader(
        run_id="r", instance_id="alpha", store=ray_app.GcsStore(object()),
        local_path=p, min_upload_interval_s=1_000.0,  # long throttle
    )
    up({"type": "AssistantMessage", "data": "t1"})
    up({"type": "ResultMessage", "data": "t2"})

    # Both appended locally (cheap, per-message).
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["data"] == "t1"

    # The FIRST message uploads immediately (so the partial appears in GCS as
    # soon as the task produces output); the second is throttled (within the
    # 1000s window) — so exactly one upload so far, and it predates t2.
    assert len(uploads) == 1
    assert "t1" in uploads[0][1] and "t2" not in uploads[0][1]

    # flush() forces a final upload carrying the full content.
    up.flush()
    assert len(uploads) == 2
    assert "t1" in uploads[1][1] and "t2" in uploads[1][1]


def test_uploader_uploads_when_interval_elapsed(monkeypatch, tmp_path):
    uploads = _patch_gcs(monkeypatch)
    up = ray_app.PartialTranscriptUploader(
        run_id="r", instance_id="beta", store=ray_app.GcsStore(object()),
        local_path=tmp_path / "p.jsonl", min_upload_interval_s=0.0,  # always due
    )
    up({"type": "X", "data": "1"})
    assert len(uploads) == 1  # interval=0 → uploads immediately


def test_failed_first_upload_does_not_throttle_retry(monkeypatch, tmp_path):
    # The throttle must only advance after a SUCCESSFUL upload. If the first
    # GCS write fails, the next message must retry immediately rather than being
    # suppressed for the whole window — otherwise a task that wedges right after
    # a failed first write leaves nothing behind.
    attempts: list = []

    def _flaky_write(run_id, instance_id, text, *, client=None):
        attempts.append(text)
        if len(attempts) == 1:
            raise RuntimeError("transient GCS error")

    monkeypatch.setattr(ray_app._gcs, "write_partial_transcript", _flaky_write)
    up = ray_app.PartialTranscriptUploader(
        run_id="r", instance_id="delta", store=ray_app.GcsStore(object()),
        local_path=tmp_path / "p.jsonl", min_upload_interval_s=1_000.0,  # long throttle
    )
    up({"type": "X", "data": "1"})  # first upload raises → throttle NOT stamped
    up({"type": "Y", "data": "2"})  # within the window, but must still retry

    assert len(attempts) == 2  # the failed first attempt did not suppress the retry
    assert "2" in attempts[1]


def test_flush_without_messages_uploads_nothing(monkeypatch, tmp_path):
    uploads = _patch_gcs(monkeypatch)
    up = ray_app.PartialTranscriptUploader(
        run_id="r", instance_id="gamma", store=ray_app.GcsStore(object()),
        local_path=tmp_path / "p.jsonl",
    )
    up.flush()
    assert uploads == []


# --------------------------------------------------------------------------
# gcs path
# --------------------------------------------------------------------------


def test_partial_transcript_blob_under_rows_dir():
    assert (
        gcs.partial_transcript_blob("run-1", "alpha")
        == "runs/run-1/rows/alpha/partial_transcript.jsonl"
    )


def test_partial_transcript_updated_ts_returns_blob_mtime():
    import datetime

    when = datetime.datetime(2026, 6, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)

    class _Blob:
        updated = when

        def reload(self):
            pass

    class _Bucket:
        def blob(self, _path):
            return _Blob()

    class _Client:
        def bucket(self, _name):
            return _Bucket()

    ts = gcs.partial_transcript_updated_ts("r", "a", client=_Client())
    assert ts == when.timestamp()


def test_partial_transcript_updated_ts_none_on_error():
    class _Blob:
        def reload(self):
            raise RuntimeError("404 not found")

    class _Bucket:
        def blob(self, _path):
            return _Blob()

    class _Client:
        def bucket(self, _name):
            return _Bucket()

    assert gcs.partial_transcript_updated_ts("r", "a", client=_Client()) is None
