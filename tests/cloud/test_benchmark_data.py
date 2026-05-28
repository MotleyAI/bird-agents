"""GCS benchmark-data delivery: content-hash keyed, upload-once (marker LAST),
download-if-absent. The dataset never changes, so it lives at a stable
`benchmark-data/<benchmark>/<hash>/` prefix and is uploaded/downloaded at most
once per content hash per location."""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents.cloud import benchmark_data as bd
from bird_interact_agents.cloud import gcs


# --- in-memory fake GCS (just enough for the marker blob) ------------------

class _FakeBlob:
    def __init__(self, store: dict, name: str):
        self._s = store
        self.name = name

    def exists(self) -> bool:
        return self.name in self._s

    def upload_from_string(self, data) -> None:
        self._s[self.name] = data.encode() if isinstance(data, str) else data


class _FakeBucket:
    def __init__(self, store: dict):
        self._s = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._s, name)


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self.store)


def _make_dataset(root: Path) -> None:
    (root / "alien").mkdir(parents=True)
    (root / "alien" / "alien.sqlite").write_bytes(b"SQLITEDATA")
    (root / "data.jsonl").write_text('{"instance_id": "alien_1"}\n')


# --- content_hash ----------------------------------------------------------

def test_content_hash_deterministic_and_sensitive(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root)
    h1 = bd.content_hash(root)
    assert h1 == bd.content_hash(root)  # deterministic
    (root / "data.jsonl").write_text('{"instance_id": "alien_2"}\n')
    assert bd.content_hash(root) != h1  # any change flips the hash


def test_prefix_is_benchmark_and_hash_keyed(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root)
    chash = bd.content_hash(root)
    prefix = bd.benchmark_data_prefix("livesqlbench", chash)
    assert prefix == f"benchmark-data/livesqlbench/{chash}/"


# --- ensure_uploaded -------------------------------------------------------

def test_ensure_uploaded_uploads_then_marks_when_absent(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_dataset(root)
    client = _FakeClient()
    calls: list = []
    monkeypatch.setattr(
        gcs, "upload_dir_prefix",
        lambda local_dir, prefix, **kw: calls.append((Path(local_dir), prefix)),
    )

    prefix = bd.ensure_uploaded("livesqlbench", root=root, client=client)

    chash = bd.content_hash(root)
    assert prefix == f"benchmark-data/livesqlbench/{chash}/"
    # data uploaded to the prefix (no trailing slash, matching upload_dir_prefix)
    assert calls == [(root, prefix.rstrip("/"))]
    # marker written LAST, at the prefix root
    assert (prefix + bd._MARKER) in client.store


def test_ensure_uploaded_skips_when_marker_present(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_dataset(root)
    client = _FakeClient()
    chash = bd.content_hash(root)
    prefix = bd.benchmark_data_prefix("livesqlbench", chash)
    client.store[prefix + bd._MARKER] = chash.encode()  # pretend already uploaded

    def _boom(*a, **k):
        raise AssertionError("must not re-upload when marker present")

    monkeypatch.setattr(gcs, "upload_dir_prefix", _boom)
    assert bd.ensure_uploaded("livesqlbench", root=root, client=client) == prefix


# --- ensure_downloaded -----------------------------------------------------

def test_ensure_downloaded_downloads_then_marks(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "livesqlbench"
    client = _FakeClient()

    def _fake_dl(prefix, d, **kw):
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "data.jsonl").write_text("{}\n")

    monkeypatch.setattr(gcs, "download_prefix", _fake_dl)
    out = bd.ensure_downloaded("benchmark-data/livesqlbench/abc/", dest, client=client)
    assert out == dest
    assert (dest / "data.jsonl").is_file()
    assert (dest / bd._MARKER).is_file()  # local marker written after download


def test_ensure_downloaded_skips_when_local_marker_present(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "livesqlbench"
    dest.mkdir(parents=True)
    (dest / bd._MARKER).write_text("benchmark-data/livesqlbench/abc/")

    def _boom(*a, **k):
        raise AssertionError("must not re-download when local marker present")

    monkeypatch.setattr(gcs, "download_prefix", _boom)
    assert bd.ensure_downloaded("benchmark-data/livesqlbench/abc/", dest, client=_FakeClient()) == dest
