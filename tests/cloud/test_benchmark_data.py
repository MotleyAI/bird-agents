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


# livesqlbench's tasks file — `ensure_uploaded` requires the benchmark's data
# file to be present in the root (so it never stamps a complete EMPTY prefix).
_LSB_DATA_FILE = "livesqlbench_data_sqlite.jsonl"


def _make_dataset(root: Path) -> None:
    (root / "alien").mkdir(parents=True)
    (root / "alien" / "alien.sqlite").write_bytes(b"SQLITEDATA")
    (root / _LSB_DATA_FILE).write_text('{"instance_id": "alien_1"}\n')


# --- content_hash ----------------------------------------------------------

def test_content_hash_deterministic_and_sensitive(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root)
    h1 = bd.content_hash(root)
    assert h1 == bd.content_hash(root)  # deterministic
    (root / _LSB_DATA_FILE).write_text('{"instance_id": "alien_2"}\n')
    assert bd.content_hash(root) != h1  # any change flips the hash


def test_content_hash_excludes_git_dir(tmp_path):
    """A benchmark data dir can be its own git checkout (livesqlbench). The
    content hash must ignore `.git/` so it's stable across upstream commits —
    otherwise the upload-once prefix would churn every time the dataset repo
    advances."""
    root = tmp_path / "ds"
    _make_dataset(root)
    h1 = bd.content_hash(root)
    # Mutate .git/ as an upstream commit would; the hash must NOT change.
    (root / ".git" / "objects").mkdir(parents=True)
    (root / ".git" / "objects" / "deadbeef").write_bytes(b"\x01\x02")
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert bd.content_hash(root) == h1
    # A real dataset change still flips it.
    (root / "alien" / "alien.sqlite").write_bytes(b"CHANGED")
    assert bd.content_hash(root) != h1


def test_content_hash_excludes_transient_sqlite_sidecars(tmp_path):
    """Transient SQLite WAL sidecars (`-shm` / `-wal` / `-journal`) are created
    live while a `.sqlite` DB is open. They must NOT enter the content hash —
    else the upload-once prefix churns run-to-run depending on whether a DB
    happened to be open during hashing."""
    root = tmp_path / "ds"
    _make_dataset(root)
    h1 = bd.content_hash(root)
    (root / "alien" / "alien.sqlite-shm").write_bytes(b"\x00\x01")
    (root / "alien" / "alien.sqlite-wal").write_bytes(b"\x02\x03")
    (root / "gaming").mkdir(exist_ok=True)
    (root / "gaming" / "gaming.sqlite-journal").write_bytes(b"\x04")
    assert bd.content_hash(root) == h1  # sidecars ignored
    # A real `.sqlite` change still flips it.
    (root / "alien" / "alien.sqlite").write_bytes(b"CHANGED")
    assert bd.content_hash(root) != h1


def test_exclude_predicate_drops_sidecars_and_git(tmp_path):
    from pathlib import Path as _P
    assert bd._is_excluded_path(_P("alien/alien.sqlite-shm")) is True
    assert bd._is_excluded_path(_P("alien/alien.sqlite-wal")) is True
    assert bd._is_excluded_path(_P("gaming/gaming.sqlite-journal")) is True
    assert bd._is_excluded_path(_P(".git/config")) is True
    # Real dataset files are kept.
    assert bd._is_excluded_path(_P("alien/alien.sqlite")) is False
    assert bd._is_excluded_path(_P("mini_interact.jsonl")) is False


def _no_gated_gold(tmp_path):
    """Return a lambda that monkeypatches `gated_gold_root` to a non-existent
    path so upload tests are isolated from any real gated gold on disk."""
    absent = tmp_path / "no_gated_gold"
    return lambda **_kw: absent


def test_ensure_uploaded_excludes_git_from_upload(tmp_path, monkeypatch):
    """The upload must drop `.git/` files too (same set as the content hash),
    so the GCS dataset tree never carries repo history."""
    root = tmp_path / "ds"
    _make_dataset(root)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    client = _FakeClient()
    seen_excludes: list = []
    monkeypatch.setattr(bd.paths, "gated_gold_root", _no_gated_gold(tmp_path))

    def _fake_upload(local_dir, prefix, *, exclude=None, **kw):
        seen_excludes.append(exclude)

    monkeypatch.setattr(gcs, "upload_dir_prefix", _fake_upload)
    bd.ensure_uploaded("livesqlbench-base-lite-sqlite", root=root, client=client)
    assert len(seen_excludes) == 1
    exclude = seen_excludes[0]
    assert exclude is not None
    from pathlib import Path as _P
    assert exclude(_P(".git/config")) is True
    assert exclude(_P("alien/alien.sqlite")) is False


def test_prefix_is_benchmark_and_hash_keyed(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root)
    chash = bd.content_hash(root)
    prefix = bd.benchmark_data_prefix("livesqlbench-base-lite-sqlite", chash)
    assert prefix == f"benchmark-data/livesqlbench-base-lite-sqlite/{chash}/"


# --- ensure_uploaded -------------------------------------------------------

def test_ensure_uploaded_uploads_then_marks_when_absent(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_dataset(root)
    client = _FakeClient()
    calls: list = []
    monkeypatch.setattr(bd.paths, "gated_gold_root", _no_gated_gold(tmp_path))
    monkeypatch.setattr(
        gcs, "upload_dir_prefix",
        lambda local_dir, prefix, **kw: calls.append((Path(local_dir), prefix)),
    )

    prefix = bd.ensure_uploaded("livesqlbench-base-lite-sqlite", root=root, client=client)

    chash = bd.content_hash(root)
    assert prefix == f"benchmark-data/livesqlbench-base-lite-sqlite/{chash}/"
    # data uploaded to the prefix (no trailing slash, matching upload_dir_prefix)
    assert calls == [(root, prefix.rstrip("/"))]
    # marker written LAST, at the prefix root
    assert (prefix + bd._MARKER) in client.store


def test_ensure_uploaded_skips_when_marker_present(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    _make_dataset(root)
    client = _FakeClient()
    monkeypatch.setattr(bd.paths, "gated_gold_root", _no_gated_gold(tmp_path))
    chash = bd.content_hash(root)
    prefix = bd.benchmark_data_prefix("livesqlbench-base-lite-sqlite", chash)
    client.store[prefix + bd._MARKER] = chash.encode()  # pretend already uploaded

    def _boom(*a, **k):
        raise AssertionError("must not re-upload when marker present")

    monkeypatch.setattr(gcs, "upload_dir_prefix", _boom)
    assert bd.ensure_uploaded("livesqlbench-base-lite-sqlite", root=root, client=client) == prefix


def test_ensure_uploaded_includes_gated_gold_in_hash_and_upload(tmp_path, monkeypatch):
    """When gated_gold_root exists, its files are included in the content hash
    (so adding/changing gold triggers a new prefix) and uploaded to the
    GATED_GOLD_SUBDIR sub-prefix."""
    root = tmp_path / "ds"
    _make_dataset(root)
    gated_root = tmp_path / "gated" / "livesqlbench-base-lite-sqlite"
    gated_root.mkdir(parents=True)
    (gated_root / "gt_kg.jsonl").write_text('{"id": 1}\n')
    monkeypatch.setattr(bd.paths, "gated_gold_root", lambda **_kw: gated_root)

    client = _FakeClient()
    calls: list = []
    monkeypatch.setattr(
        gcs, "upload_dir_prefix",
        lambda local_dir, prefix, **kw: calls.append((Path(local_dir), prefix)),
    )

    prefix = bd.ensure_uploaded("livesqlbench-base-lite-sqlite", root=root, client=client)

    # Prefix differs from content_hash(root) because gated gold is included.
    assert prefix != f"benchmark-data/livesqlbench-base-lite-sqlite/{bd.content_hash(root)}/"
    # Two uploads: data root + gated gold subdir.
    assert calls[0] == (root, prefix.rstrip("/"))
    assert calls[1] == (gated_root, f"{prefix.rstrip('/')}/{bd.GATED_GOLD_SUBDIR}/livesqlbench-base-lite-sqlite")
    assert (prefix + bd._MARKER) in client.store


# --- ensure_downloaded -----------------------------------------------------

def test_ensure_downloaded_downloads_then_marks(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "livesqlbench-base-lite-sqlite"
    client = _FakeClient()
    prefix = "benchmark-data/livesqlbench/abc/"
    # The remote completeness marker must be present for the download to be
    # trusted (the upload writes it LAST).
    client.store[prefix + bd._MARKER] = b"abc"

    def _fake_dl(p, d, **kw):
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "data.jsonl").write_text("{}\n")

    monkeypatch.setattr(gcs, "download_prefix", _fake_dl)
    out = bd.ensure_downloaded(prefix, dest, client=client)
    assert out == dest
    assert (dest / "data.jsonl").is_file()
    assert (dest / bd._MARKER).is_file()  # local marker written after download


def test_ensure_downloaded_refuses_prefix_without_remote_marker(tmp_path, monkeypatch):
    """No remote completeness marker → partial/wrong/GC'd prefix → must raise,
    never cache a local marker over an empty/partial download."""
    dest = tmp_path / "data" / "livesqlbench-base-lite-sqlite"
    client = _FakeClient()  # store empty → no remote marker

    def _boom(*a, **k):
        raise AssertionError("must not download when remote marker absent")

    monkeypatch.setattr(gcs, "download_prefix", _boom)
    with pytest.raises(FileNotFoundError, match="no completeness marker"):
        bd.ensure_downloaded("benchmark-data/livesqlbench/missing/", dest, client=client)
    assert not (dest / bd._MARKER).exists()


def test_ensure_downloaded_skips_when_local_marker_present(tmp_path, monkeypatch):
    dest = tmp_path / "data" / "livesqlbench-base-lite-sqlite"
    dest.mkdir(parents=True)
    (dest / bd._MARKER).write_text("benchmark-data/livesqlbench/abc/")

    def _boom(*a, **k):
        raise AssertionError("must not re-download when local marker present")

    monkeypatch.setattr(gcs, "download_prefix", _boom)
    assert bd.ensure_downloaded("benchmark-data/livesqlbench/abc/", dest, client=_FakeClient()) == dest


def test_ensure_downloaded_redownloads_on_prefix_mismatch(tmp_path, monkeypatch):
    """A stale local marker written for a DIFFERENT content-hash prefix is NOT
    a cache hit — `dest` is benchmark-scoped, not hash-scoped, so a benchmark
    update lands a new prefix into the same dir. ensure_downloaded must clear
    the stale tree and re-download (CodeRabbit)."""
    dest = tmp_path / "data" / "livesqlbench-base-lite-sqlite"
    dest.mkdir(parents=True)
    (dest / bd._MARKER).write_text("benchmark-data/livesqlbench/OLDHASH/")
    (dest / "stale.jsonl").write_text("old")  # a file the new dataset removed
    client = _FakeClient()
    new_prefix = "benchmark-data/livesqlbench/NEWHASH/"
    client.store[new_prefix + bd._MARKER] = b"NEWHASH"
    dl_calls: list = []

    def _fake_dl(prefix, d, **kw):
        dl_calls.append(prefix)
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "fresh.jsonl").write_text("{}\n")

    monkeypatch.setattr(gcs, "download_prefix", _fake_dl)
    bd.ensure_downloaded(new_prefix, dest, client=client)
    assert dl_calls == [new_prefix]            # re-downloaded under the new prefix
    assert not (dest / "stale.jsonl").exists()  # stale tree cleared
    assert (dest / "fresh.jsonl").is_file()      # new content present
    assert (dest / bd._MARKER).read_text() == new_prefix


def test_ensure_downloaded_clears_partial_dest_without_marker(tmp_path, monkeypatch):
    """A partial dest (files present, NO marker — a crashed prior download) is
    cleared + re-downloaded under the lock; the marker (written LAST) is the
    only completeness signal (Codex)."""
    dest = tmp_path / "data" / "livesqlbench-base-lite-sqlite"
    dest.mkdir(parents=True)
    (dest / "partial.jsonl").write_text("half")  # leftover from a crash, no marker
    client = _FakeClient()
    prefix = "benchmark-data/livesqlbench/abc/"
    client.store[prefix + bd._MARKER] = b"abc"

    def _fake_dl(p, d, **kw):
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "full.jsonl").write_text("{}\n")

    monkeypatch.setattr(gcs, "download_prefix", _fake_dl)
    bd.ensure_downloaded(prefix, dest, client=client)
    assert not (dest / "partial.jsonl").exists()
    assert (dest / "full.jsonl").is_file()
    assert (dest / bd._MARKER).read_text() == prefix
