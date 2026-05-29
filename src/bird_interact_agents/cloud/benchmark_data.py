"""Benchmark dataset delivery to GCS — uploaded ONCE, downloaded per node.

A benchmark's dataset (per-DB sqlite + tasks JSONL + KB/column-meaning) never
changes, so baking it into every code image is wasteful. Instead it lives at a
STABLE, content-hashed GCS prefix ``benchmark-data/<benchmark>/<hash>/`` — NOT
under a per-run prefix — uploaded *if-absent* at submit and downloaded
*if-absent* on each node (head + every worker).

A completeness marker is written LAST on upload and only trusted on download,
so a partial or concurrent upload can never be mistaken for a finished one
(mirrors the OTF cache's ``_cache_fp.txt`` invariant).
"""

from __future__ import annotations

import fcntl
import hashlib
import shutil
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.benchmark import Benchmark, get_benchmark
from bird_interact_agents.cloud import gcs

# Sibling marker blob at the prefix root (NOT under the data tree), written last.
_MARKER = "_benchmark_data.marker"


def _is_vcs_path(rel: Path) -> bool:
    """VCS metadata to exclude from BOTH the content hash and the upload.

    A benchmark data dir can be its own git checkout (e.g. livesqlbench's
    ``livesqlbench-base-lite-sqlite/.git/``). Including ``.git/`` would (a)
    churn the content hash on every upstream commit — defeating the
    upload-ONCE property — and (b) bloat the upload with repo history. The
    actual dataset (per-DB sqlite + tasks JSONL + KB) is what matters."""
    return ".git" in rel.parts


def _as_benchmark(benchmark: str | Benchmark) -> Benchmark:
    return benchmark if isinstance(benchmark, Benchmark) else get_benchmark(benchmark)


def content_hash(root: Path) -> str:
    """Deterministic content hash over every (non-VCS) file under ``root``
    (sorted by relative path, hashing path + bytes). Two identical dataset
    trees hash the same regardless of host layout; any change flips the hash →
    a new prefix. ``.git/`` is excluded (see :func:`_is_vcs_path`) so the hash
    is stable across upstream commits — same set as the upload."""
    root = Path(root)
    h = hashlib.sha256()
    for f in sorted(
        (p for p in root.rglob("*")
         if p.is_file() and not _is_vcs_path(p.relative_to(root))),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(b"\x00")
        h.update(f.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def benchmark_data_prefix(benchmark: str | Benchmark, chash: str) -> str:
    """Stable GCS prefix for a benchmark's dataset at a given content hash."""
    return f"benchmark-data/{_as_benchmark(benchmark).name}/{chash}/"


def ensure_uploaded(
    benchmark: str | Benchmark, *, root: Path | None = None, client=None,
) -> str:
    """Upload the benchmark's dataset to its content-hashed GCS prefix if not
    already present (marker check), and return the prefix. Idempotent: a second
    submit of the same dataset is a no-op marker check."""
    b = _as_benchmark(benchmark)
    root = Path(root) if root is not None else paths.benchmark_data_root(b)
    # Refuse to stamp a "complete" prefix for a missing dataset: an absent or
    # data-file-less root would otherwise hash to a stable (empty) value,
    # upload nothing, and write the marker — caching a fake-complete EMPTY
    # dataset that makes every in-cluster task fail mysteriously (Codex).
    if not root.is_dir():
        raise FileNotFoundError(
            f"benchmark {b.name!r} data root not found: {root}"
        )
    if not (root / b.data_file).is_file():
        raise FileNotFoundError(
            f"benchmark {b.name!r} data root {root} is missing its tasks file "
            f"{b.data_file!r}; refusing to upload an incomplete dataset"
        )
    chash = content_hash(root)
    prefix = benchmark_data_prefix(b, chash)
    client = client or gcs.default_gcs_client()
    marker = client.bucket(gcs.BUCKET_NAME).blob(prefix + _MARKER)
    if marker.exists():
        return prefix
    gcs.upload_dir_prefix(
        root, prefix.rstrip("/"), client=client, exclude=_is_vcs_path,
    )
    marker.upload_from_string(chash)  # marker LAST — completeness invariant
    return prefix


def _marker_matches(local_marker: Path, prefix: str) -> bool:
    """The local marker stores the prefix it was downloaded for. ``dest`` is
    benchmark-scoped (``/data/<benchmark>``), NOT hash-scoped, so a benchmark
    update lands a NEW content-hash prefix into the same dir — only a marker
    whose content equals THIS prefix is a real cache hit (CodeRabbit)."""
    return local_marker.is_file() and local_marker.read_text() == prefix


def ensure_downloaded(prefix: str, dest: Path, *, client=None) -> Path:
    """Download the dataset at ``prefix`` into ``dest`` if not already present
    for THIS prefix. Concurrency-safe across the multiple Ray actors that share
    one VM's ``dest`` (``--actors-per-worker > 1``): the whole
    check→download→mark sequence runs under a per-``dest`` ``fcntl`` lock, so a
    task can never observe a half-written tree (Codex). The completeness marker
    (``_MARKER``, holding the prefix) is written LAST, so an interrupted
    download leaves no marker and is re-attempted on the next call.

    Refuses to cache a prefix that lacks the upload-completeness marker on the
    GCS side (written LAST by :func:`ensure_uploaded`): a missing remote marker
    means the upload was partial/concurrent OR the prefix is wrong /
    lifecycle-GC'd. Trusting it would mark an empty/partial download complete
    and silently fail every in-cluster task (Codex)."""
    dest = Path(dest)
    local_marker = dest / _MARKER
    # Lock-free fast path: this exact prefix is already fully present.
    if _marker_matches(local_marker, prefix):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Sibling lock (NOT under dest, so it survives an rmtree of dest). Serialises
    # concurrent actors on one VM: the first downloads + marks; the rest block,
    # then re-check and hit the cache.
    lock_path = dest.parent / f".{dest.name}.dl.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        # Re-check under the lock — a peer may have just finished.
        if _marker_matches(local_marker, prefix):
            return dest
        client = client or gcs.default_gcs_client()
        remote_marker = client.bucket(gcs.BUCKET_NAME).blob(prefix + _MARKER)
        if not remote_marker.exists():
            raise FileNotFoundError(
                f"benchmark-data prefix {prefix!r} has no completeness marker "
                f"({_MARKER}); refusing to cache an incomplete/missing dataset"
            )
        # Clear any stale (older-prefix) or partial (crashed mid-download) tree
        # before re-downloading, so removed files don't linger and a partial
        # tree can't be mistaken for complete. Safe under the lock.
        if dest.exists():
            shutil.rmtree(dest)
        gcs.download_prefix(prefix, dest, client=client)
        local_marker.write_text(prefix)  # marker LAST — completeness invariant
    return dest
