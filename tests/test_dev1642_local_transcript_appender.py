"""DEV-1642: ``LocalTranscriptAppender`` — the append-per-message sink for
LOCAL runs. Each streamed claude_sdk message is written as one JSON line to a
durable path (``rows/<iid>/partial_transcript.jsonl``) the instant it arrives:
no throttle, no temp-dir hop, no store round-trip. Best-effort (never raises
into the receive stream) and truncate-once at construction so a stale file from
a prior run/attempt cannot bleed into this task's transcript.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from bird_interact_agents.agents.claude_sdk import sdk_env
from bird_interact_agents.agents.claude_sdk.sdk_env import LocalTranscriptAppender


# --------------------------------------------------------------------------
# append semantics
# --------------------------------------------------------------------------


def test_single_call_writes_one_json_line(tmp_path):
    p = tmp_path / "rows" / "iid" / "partial_transcript.jsonl"
    p.parent.mkdir(parents=True)
    app = LocalTranscriptAppender(p)
    app({"type": "AssistantMessage", "data": "t1"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"type": "AssistantMessage", "data": "t1"}


def test_calls_append_and_preserve_order_not_overwrite(tmp_path):
    p = tmp_path / "partial_transcript.jsonl"
    app = LocalTranscriptAppender(p)
    app({"type": "A", "data": "1"})
    app({"type": "B", "data": "2"})
    app({"type": "C", "data": "3"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["data"] for x in lines] == ["1", "2", "3"]


def test_construction_creates_empty_file_when_missing(tmp_path):
    """The sink 'opens' at construction: a missing file (parent exists) becomes
    an empty file BEFORE any message, so an absent file unambiguously means
    'the recorder was never installed' rather than 'no message yet'."""
    p = tmp_path / "partial_transcript.jsonl"
    assert not p.exists()
    LocalTranscriptAppender(p)
    assert p.exists()
    assert p.read_text(encoding="utf-8") == ""


def test_truncates_stale_file_at_construction(tmp_path):
    """A leftover partial_transcript.jsonl from a prior run/attempt must not
    bleed into this task — the sink resets the file ONCE at construction, then
    appends per message (Codex review)."""
    p = tmp_path / "partial_transcript.jsonl"
    p.write_text("STALE-LINE-FROM-PRIOR-RUN\n", encoding="utf-8")
    app = LocalTranscriptAppender(p)
    # Cleared before any message streams.
    assert p.read_text(encoding="utf-8") == ""
    app({"type": "X", "data": "fresh"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["data"] == "fresh"


# --------------------------------------------------------------------------
# best-effort: never raises into the receive stream
# --------------------------------------------------------------------------


def test_unserialisable_message_no_raise_no_line(tmp_path):
    """A message that genuinely defeats json.dumps(..., default=str) — a
    circular reference — must be dropped silently, never raised, and must not
    write a partial/garbage line."""
    p = tmp_path / "partial_transcript.jsonl"
    app = LocalTranscriptAppender(p)
    circular: dict = {}
    circular["self"] = circular  # json.dumps raises ValueError even with default=str
    app(circular)  # must not raise
    # Nothing (beyond the truncate) written.
    assert p.read_text(encoding="utf-8") == ""


def test_custom_object_serialises_via_default_str(tmp_path):
    class _Weird:
        def __str__(self):
            return "WEIRD"

    p = tmp_path / "partial_transcript.jsonl"
    app = LocalTranscriptAppender(p)
    app({"type": "T", "data": _Weird()})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["data"] == "WEIRD"


def test_io_error_does_not_raise(tmp_path):
    # Path points at a directory → open("a") raises IsADirectoryError, which
    # the sink must swallow.
    d = tmp_path / "iam_a_dir"
    d.mkdir()
    app = LocalTranscriptAppender(d)  # construction truncate also swallowed
    app({"type": "T", "data": "x"})  # must not raise


def test_write_failure_logged_once_not_per_message(monkeypatch, tmp_path, caplog):
    """A permanently-broken path must leave ONE diagnostic line, not one per
    streamed message (first-failure-only guard; CodeRabbit / Ruff S110)."""
    import builtins

    p = tmp_path / "partial_transcript.jsonl"
    app = LocalTranscriptAppender(p)  # construction (truncate) succeeds

    real_open = builtins.open

    def boom_open(file, mode="r", *a, **k):
        if str(file) == str(p) and "a" in mode:
            raise OSError("disk full")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", boom_open)
    with caplog.at_level("DEBUG", logger="bird_interact_agents.agents.claude_sdk.sdk_env"):
        app({"data": "1"})
        app({"data": "2"})
        app({"data": "3"})

    hits = [r for r in caplog.records if "LocalTranscriptAppender" in r.getMessage()]
    assert len(hits) == 1


def test_flush_is_noop_and_returns_none(tmp_path):
    p = tmp_path / "partial_transcript.jsonl"
    app = LocalTranscriptAppender(p)
    app({"type": "T", "data": "x"})
    assert app.flush() is None
    # flush changed nothing (append is already durable).
    assert len(p.read_text(encoding="utf-8").splitlines()) == 1


def test_concurrent_calls_never_tear_lines(tmp_path):
    """The sink serialises writes under a lock, so concurrent callers (the
    warm-discovery + main clients share ONE sink) can never interleave into a
    torn/garbage line. Large payloads force multi-syscall buffered writes so a
    lockless implementation would actually corrupt lines here."""
    p = tmp_path / "partial_transcript.jsonl"
    app = LocalTranscriptAppender(p)
    n_threads, per_thread = 8, 40
    blob = "x" * 200_000  # big enough to split the write across syscalls

    def worker(wid: int) -> None:
        for i in range(per_thread):
            app({"w": wid, "i": i, "blob": blob})

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_threads * per_thread
    # Every line is a complete, parseable JSON object (no interleaving).
    for ln in lines:
        json.loads(ln)


# --------------------------------------------------------------------------
# sink → appender → file chain (the path both local runners rely on)
# --------------------------------------------------------------------------


class _FakeMsg:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return f"MSG:{self.text}"


class _FakeClient:
    def __init__(self, msgs):
        self._msgs = msgs

    async def query(self, *a, **k):  # pragma: no cover - unused
        return None

    def receive_response(self):
        async def _agen():
            for m in self._msgs:
                yield m
        return _agen()


def test_record_partial_transcript_streams_into_appender_file(tmp_path):
    p = tmp_path / "rows" / "iid" / "partial_transcript.jsonl"
    p.parent.mkdir(parents=True)
    app = LocalTranscriptAppender(p)
    client = sdk_env._TranscriptClient(_FakeClient([_FakeMsg("a"), _FakeMsg("b")]))

    async def _drive():
        with sdk_env.record_partial_transcript(app):
            async for _ in client.receive_response():
                pass

    asyncio.run(_drive())
    lines = p.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["data"] for x in lines] == ["MSG:a", "MSG:b"]
