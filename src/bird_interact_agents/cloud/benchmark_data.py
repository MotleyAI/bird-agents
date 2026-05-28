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
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.benchmark import Benchmark, get_benchmark
from bird_interact_agents.cloud import gcs

# Sibling marker blob at the prefix root (NOT under the data tree), written last.
_MARKER = "_benchmark_data.marker"


def _as_benchmark(benchmark: str | Benchmark) -> Benchmark:
    return benchmark if isinstance(benchmark, Benchmark) else get_benchmark(benchmark)


def content_hash(root: Path) -> str:
    """Deterministic content hash over every file under ``root`` (sorted by
    relative path, hashing path + bytes). Two identical dataset trees hash the
    same regardless of host layout; any change flips the hash → a new prefix."""
    h = hashlib.sha256()
    for f in sorted(
        (p for p in Path(root).rglob("*") if p.is_file()),
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
    chash = content_hash(root)
    prefix = benchmark_data_prefix(b, chash)
    client = client or gcs.default_gcs_client()
    marker = client.bucket(gcs.BUCKET_NAME).blob(prefix + _MARKER)
    if marker.exists():
        return prefix
    gcs.upload_dir_prefix(root, prefix.rstrip("/"), client=client)
    marker.upload_from_string(chash)  # marker LAST — completeness invariant
    return prefix


def ensure_downloaded(prefix: str, dest: Path, *, client=None) -> Path:
    """Download the dataset at ``prefix`` into ``dest`` if not already present
    locally (local marker check). Writes the local marker only AFTER a complete
    download, so an interrupted download is re-attempted next call."""
    dest = Path(dest)
    local_marker = dest / _MARKER
    if local_marker.is_file():
        return dest
    client = client or gcs.default_gcs_client()
    gcs.download_prefix(prefix, dest, client=client)
    local_marker.write_text(prefix)
    return dest
