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


def ensure_downloaded(prefix: str, dest: Path, *, client=None) -> Path:
    """Download the dataset at ``prefix`` into ``dest`` if not already present
    locally (local marker check). Writes the local marker only AFTER a complete
    download, so an interrupted download is re-attempted next call.

    Refuses to cache a prefix that lacks the upload-completeness marker
    (``_MARKER``, written LAST by :func:`ensure_uploaded`): a missing remote
    marker means the upload was partial/concurrent OR the prefix is wrong /
    lifecycle-GC'd. Trusting it would write a local marker over an empty/partial
    download and silently fail every in-cluster task (Codex)."""
    dest = Path(dest)
    local_marker = dest / _MARKER
    if local_marker.is_file():
        # The marker stores the prefix it was downloaded for. ``dest`` is
        # benchmark-scoped (``/data/<benchmark>``), NOT hash-scoped, so a
        # benchmark update lands a NEW content-hash prefix into the same dir.
        # Only treat it as a cache hit when the marker matches THIS prefix;
        # on a mismatch the cached tree is stale (and may retain files the new
        # dataset removed), so clear it and re-download (CodeRabbit).
        if local_marker.read_text() == prefix:
            return dest
        shutil.rmtree(dest)
    client = client or gcs.default_gcs_client()
    remote_marker = client.bucket(gcs.BUCKET_NAME).blob(prefix + _MARKER)
    if not remote_marker.exists():
        raise FileNotFoundError(
            f"benchmark-data prefix {prefix!r} has no completeness marker "
            f"({_MARKER}); refusing to cache an incomplete/missing dataset"
        )
    gcs.download_prefix(prefix, dest, client=client)
    local_marker.write_text(prefix)
    return dest
