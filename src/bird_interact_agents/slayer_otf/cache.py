"""Per-DB cache for the on-the-fly setup path.

Materialises orchestrator phases 1-3 (slayer ingest + column-meaning
overlay + JSONB-leaf expansion) into a SINGLE authoritative dir per DB,
``<cache_root>/<db>/``, and caches the parsed ``*_kb.jsonl`` rows
alongside.

Phase 4 (LLM TEXT-as-date detection) is intentionally skipped on this
path — corpus audit (DEV-1455) showed zero retypes across all 28
ingested DBs, so paying for an LLM call per first-task-touching-DB
buys nothing.

Lifecycle (DEV-1468 consolidation — presence-gated reuse + explicit
force-wipe):

- The completeness marker is ``_cache_fp.txt`` (written last). Its
  presence means "complete"; its content is the build-time fingerprint
  (provenance only). ``_kb_rows.json`` is written before it.
- Reuse is gated solely on the marker's presence: a present marker →
  reuse (no rebuild, ``fingerprint_of`` is NOT recomputed). Fingerprint
  *gating* is removed — this makes an uploaded artifact reusable in the
  cloud by construction (a recomputed fingerprint would differ on a
  different sqlite mtime / abs root and force a needless rebuild).
- Rebuild happens only when the marker is ABSENT or ``force=True``.
  Accepted tradeoff: editing a KB/schema/DB and re-running WITHOUT
  ``force`` reuses the stale artifact; reingest is now explicit.
- ``fingerprint_of`` is retained as a pure provenance function (written
  to ``_cache_fp.txt`` at build, loaded back on reuse into
  ``CacheEntry.fingerprint``); it no longer names the dir or gates reuse.

Concurrency / atomicity:

- Build into a unique sibling tmp ``<cache_root>/.<db>.tmp-<random>/``
  and atomic-rename onto ``<cache_root>/<db>`` on success. The target is
  NEVER pre-created (renaming onto a pre-created dir fails).
- Migration / force: a pre-existing target with NO marker (old
  ``<db>/<fp>/`` layout, or an incomplete dir) — or any target under
  ``force`` — is ``rmtree``'d before the rename. The cache is
  regenerable, so this uses the same single-process-per-run assumption
  as the reference's force path.
- A per-DB :class:`asyncio.Lock` serialises concurrent builds within the
  same process+event-loop. For cross-process concurrency the atomic
  rename is the source of truth: the loser of a rename race sees
  ``OSError`` and treats it as success iff the target now carries the
  marker (else re-raises), discarding its own tmp dir.
- Failed builds wipe their tmp dir before re-raising, so a crash
  mid-build never leaves a half-baked dir behind.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import slayer
from slayer.embeddings.client import (
    current_model as _embedding_current_model,
    embed_batch,
    is_available as _embeddings_available,
)
from slayer.embeddings.models import Embedding
from slayer.memories.models import MEMORY_CANONICAL_PREFIX, Memory
from slayer.search.render import render_memory_text_for_embedding
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)
from bird_interact_agents.slayer_pipeline.orchestrator import (
    _phase1_ingest,
    _phase2_overlay,
    _phase3_jsonb,
    _phase4_dates,  # imported only so tests can monkeypatch + assert non-call
)

# Re-export _phase4_dates for tests' monkeypatch + assertion. The cache
# layer must NEVER call it.
_ = _phase4_dates


logger = logging.getLogger(__name__)

# Completeness marker for a per-DB cache dir. Present ⇒ complete; written
# LAST in the build tmp dir. Content = the build-time fingerprint (provenance).
_CACHE_MARKER = "_cache_fp.txt"

# DEV-1508: implementation-only fingerprint marker. Captures the components
# that MUST match between cache-build time and runtime — slayer package
# version + active embedding model name — but NONE of the input components
# that the cloud's "build local, reuse remote" lifecycle deliberately
# tolerates differing (sqlite mtime, absolute mini-interact root, KB / column-
# meaning content). On reuse, this is recomputed and rebuilds on mismatch;
# the full fingerprint stays marker-presence-gated (see `ensure_db_cache`).
_IMPL_MARKER = "_impl_fp.txt"


@dataclass(frozen=True)
class CacheEntry:
    """Result of a successful (or no-op) cache materialisation."""

    cache_dir: Path
    fingerprint: str
    kb_rows: list[dict] = field(default_factory=list)


# In-process serialisation key = (id(event_loop), db). asyncio.Locks
# are bound to the loop that created them, so we recreate per-loop
# (pytest-asyncio runs each test in its own loop). Cross-process
# concurrency is handled by the atomic-rename collision check.
_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


def _get_lock(db: str) -> asyncio.Lock:
    loop = asyncio.get_event_loop()
    key = (id(loop), db)
    if key not in _LOCKS:
        _LOCKS[key] = asyncio.Lock()
    return _LOCKS[key]


def _slayer_version() -> str:
    """Return the active ``slayer`` package version. Pulled out so tests
    can monkeypatch it and prove the fingerprint changes on upgrade."""
    return getattr(slayer, "__version__", "unknown")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_stat(path: Path) -> str:
    """Hash a stat tuple — used for the sqlite file where we don't want
    to read the whole DB just to compute a fingerprint. Size + mtime
    nanos is enough to detect every realistic change (ingest writes
    update both)."""
    st = path.stat()
    return hashlib.sha256(
        f"{st.st_size}:{int(st.st_mtime_ns)}".encode()
    ).hexdigest()


def _active_embedding_model_or_none() -> str:
    """Return the active embedding model name when the channel is
    configured, or ``"none"`` when it isn't. Used as a fingerprint
    component so caches built without embeddings — or with a different
    model — rebuild when the configuration changes (Codex finding on
    PR #19: silently reusing an embeddings-less cache after enabling
    the channel would leave channel 3 dark forever)."""
    if not _embeddings_available():
        return "none"
    try:
        return _embedding_current_model()
    except Exception:  # pragma: no cover — config-detect fallback
        return "unknown"


def fingerprint_of(
    *,
    db_name: str,
    data_root: Path | None = None,
    benchmark: object | None = None,
    mini_interact_root: Path | None = None,
) -> str:
    """Compute the cache fingerprint for one DB.

    Exposed publicly so tests can predict the expected cache path
    without having to mirror the algorithm. Pure function — no I/O
    beyond stat / read of the three fingerprint inputs.

    Args:
        db_name: DB identifier.
        data_root: Root containing ``<db_name>/`` sub-dir (preferred).
        benchmark: Benchmark descriptor (optional). Postgres benchmarks
            hash the schema text file instead of the sqlite file stat.
        mini_interact_root: Legacy alias for ``data_root`` (kept for
            backward compat with existing callers).

    Fingerprint components:

    - ``slayer.__version__`` — orchestrator phase behaviour can change
      across releases.
    - ``data_root.resolve()`` — two roots with identical file stats
      must not share a cache.
    - active embedding model name (or ``"none"``) — cache built
      without channel-3 embeddings must rebuild when embeddings are
      later enabled, OR when ``SLAYER_EMBEDDING_MODEL`` changes.
    - For SQLite: sqlite size+mtime.
      For Postgres: schema-text content hash (no sqlite file to stat).
    - column-meaning content, KB JSONL content.
    """
    effective_root: Path = data_root or mini_interact_root  # type: ignore[assignment]
    if effective_root is None:
        raise ValueError("fingerprint_of: data_root (or mini_interact_root) is required")
    meanings_path = effective_root / db_name / f"{db_name}_column_meaning_base.json"
    kb_path = effective_root / db_name / f"{db_name}_kb.jsonl"
    h = hashlib.sha256()
    h.update(f"slayer={_slayer_version()}\n".encode())
    h.update(f"root={effective_root.resolve().as_posix()}\n".encode())
    h.update(f"embed={_active_embedding_model_or_none()}\n".encode())
    if getattr(benchmark, "db_backend", "sqlite") == "postgres":
        schema_path = effective_root / db_name / f"{db_name}_schema.txt"
        h.update(f"schema={_hash_file(schema_path)}\n".encode())
    else:
        sqlite_path = effective_root / db_name / f"{db_name}.sqlite"
        h.update(f"sqlite={_hash_stat(sqlite_path)}\n".encode())
    h.update(f"meanings={_hash_file(meanings_path)}\n".encode())
    h.update(f"kb={_hash_file(kb_path)}\n".encode())
    return h.hexdigest()[:16]


def _impl_fingerprint_of() -> str:
    """Return the implementation-version fingerprint half.

    Includes ONLY the components that must agree between cache-build time
    and runtime; the input-half (sqlite stat, abs root, KB content,
    column-meaning content) is intentionally excluded so the cloud's "build
    local, reuse remote" lifecycle keeps working.

    Components:

    - ``slayer.__version__`` — semantic-layer schema / renderer / ingestion
      behaviour can change across releases; reusing a cache built against
      a different slayer would silently serve stale entity embeddings or
      models. (Mitigates DEV-1508 risk #1 — Codex review of the plan.)
    - active embedding model name (or ``"none"``) — search ranks would
      degrade silently if embeddings were built against a different model.

    Used by :func:`ensure_db_cache` on the reuse path: recompute, compare
    against the persisted ``_impl_fp.txt`` marker, and fall through to a
    full rebuild on mismatch.
    """
    h = hashlib.sha256()
    h.update(f"slayer={_slayer_version()}\n".encode())
    h.update(f"embed={_active_embedding_model_or_none()}\n".encode())
    return h.hexdigest()[:16]


def _load_kb_rows(kb_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in kb_path.read_text().splitlines()
        if line.strip()
    ]


def _load_cache_entry(target: Path) -> CacheEntry:
    """Build a :class:`CacheEntry` from a complete on-disk cache dir.

    Only called when the ``_cache_fp.txt`` marker is present; since the marker
    and ``_kb_rows.json`` are committed together by one atomic rename, both are
    guaranteed present here.
    """
    fp = (target / _CACHE_MARKER).read_text().strip()
    kb_rows = json.loads((target / "_kb_rows.json").read_text())
    return CacheEntry(cache_dir=target, fingerprint=fp, kb_rows=kb_rows)


async def ensure_db_cache(
    db: str,
    *,
    cache_root: Path,
    mini_interact_root: Path | None = None,
    data_root: Path | None = None,
    benchmark: object | None = None,
    force: bool = False,
) -> CacheEntry:
    """Materialise (or reuse) the single authoritative per-DB cache at
    ``<cache_root>/<db>/``.

    Reuse gating:

    - ``_cache_fp.txt`` (full fingerprint) is marker-presence gated only.
      Recomputing it on reuse would always-mismatch in the cloud (different
      absolute root + sqlite mtime than the laptop that built the cache),
      forcing a needless rebuild — that's the lifecycle the cloud was
      designed around.
    - ``_impl_fp.txt`` (DEV-1508) IS recomputed and forces a rebuild on
      mismatch. It only captures the impl-version components
      (``slayer.__version__`` + active embedding model) so the cloud
      lifecycle keeps working; an actual slayer-version / embed-model drift
      between cache-build and runtime triggers a rebuild instead of
      silently serving stale storage. Missing ``_impl_fp.txt`` is treated
      as a mismatch (legacy-cache migration).

    Rebuild happens when the marker is ABSENT, ``force=True``, or the
    impl fingerprint diverges. See the module docstring for the full
    lifecycle/atomicity contract.

    Returns a :class:`CacheEntry` whose ``cache_dir`` is ``<cache_root>/<db>``
    and whose ``fingerprint`` is loaded from the marker (reuse) or freshly
    computed (build), as provenance.
    """
    target = cache_root / db
    marker = target / _CACHE_MARKER

    def _impl_ok() -> bool:
        """Persisted impl fp matches the current implementation. The legacy
        case (no ``_impl_fp.txt`` because the cache was built before the
        split landed) is treated as a mismatch → rebuild — safer than
        assuming compat with an unknown prior implementation."""
        impl_marker = target / _IMPL_MARKER
        if not impl_marker.is_file():
            return False
        return impl_marker.read_text().strip() == _impl_fingerprint_of()

    # Fast path — reuse a complete cache without recomputing the FULL
    # fingerprint. The IMPL half IS recomputed; mismatch falls through to
    # rebuild.
    if not force and marker.is_file() and _impl_ok():
        return _load_cache_entry(target)

    async with _get_lock(db):
        # Double-check under the lock — a peer coroutine may have built it
        # while we waited.
        if not force and marker.is_file() and _impl_ok():
            return _load_cache_entry(target)

        effective_root: Path = data_root or mini_interact_root  # type: ignore[assignment]
        if effective_root is None:
            raise ValueError("ensure_db_cache: data_root (or mini_interact_root) is required")

        is_postgres = getattr(benchmark, "db_backend", "sqlite") == "postgres"

        meanings_path = effective_root / db / f"{db}_column_meaning_base.json"
        kb_path = effective_root / db / f"{db}_kb.jsonl"
        sqlite_path: Path | None = None

        required_files: list[tuple[Path, str]] = [
            (meanings_path, "column-meaning"),
            (kb_path, "kb"),
        ]
        if is_postgres:
            schema_path = effective_root / db / f"{db}_schema.txt"
            required_files.insert(0, (schema_path, "schema"))
        else:
            sqlite_path = effective_root / db / f"{db}.sqlite"
            required_files.insert(0, (sqlite_path, "sqlite"))

        for p, label in required_files:
            if not p.is_file():
                raise FileNotFoundError(
                    f"slayer_otf cache: required {label} file missing for "
                    f"db={db}: {p}"
                )

        fp = fingerprint_of(db_name=db, data_root=effective_root, benchmark=benchmark)
        kb_rows = _load_kb_rows(kb_path)

        # Build into a unique tmp sibling under cache_root so a crash leaves no
        # partial target. NEVER pre-create the target itself.
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{db}.tmp-", dir=str(cache_root)))
        try:
            await _build_async(
                build_dir=tmp_dir, db=db,
                sqlite_path=sqlite_path,
                meanings_path=meanings_path,
                kb_rows=kb_rows,
                benchmark=benchmark,
            )
            # DEV-1508: persist the impl-fp half BEFORE the completeness
            # marker so a successful reuse (marker present) is guaranteed
            # to find a usable ``_impl_fp.txt`` to validate. The marker is
            # still written LAST and remains the sole completeness signal.
            (tmp_dir / _IMPL_MARKER).write_text(_impl_fingerprint_of())
            # Provenance marker, written LAST.
            (tmp_dir / _CACHE_MARKER).write_text(fp)

            # Migration / force / drift: a pre-existing target with NO
            # marker (old <db>/<fp>/ layout or an incomplete dir), or any
            # target under force, OR a target whose persisted impl
            # fingerprint diverges from the current one (DEV-1508) — is
            # wiped before the rename. Without the impl-drift branch,
            # ``os.rename(tmp_dir, target)`` would fail with ENOTEMPTY and
            # the OSError handler would silently return the stale target,
            # defeating the impl-fp protection (Codex review of PR #11).
            if target.exists() and (force or not marker.is_file() or not _impl_ok()):
                shutil.rmtree(target, ignore_errors=True)

            # Atomic-rename onto the (now-absent) target. Cross-process race:
            # if a peer won the rename while we built, ours fails with OSError;
            # treat that as success iff the target now carries the marker AND
            # the peer's impl fingerprint matches ours — otherwise we'd be
            # accepting a peer that built under a stale impl. Tight invariant
            # in practice (single uv.lock per process tree) but explicit so
            # the cross-process race can't reintroduce DEV-1508's bug class.
            try:
                os.rename(tmp_dir, target)
            except OSError:
                if not marker.is_file() or not _impl_ok():
                    raise
                # A peer won the rename; the target is THEIR complete dir
                # AND its impl fp matches ours. Return their on-disk
                # fp/kb_rows (ours may differ on input-half components) so
                # the CacheEntry matches cache_dir's actual contents.
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return _load_cache_entry(target)
        except BaseException:
            # Clean up the half-built tmp on any failure — keep the cache
            # root tidy and the next call's double-check honest.
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return CacheEntry(cache_dir=target, fingerprint=fp, kb_rows=kb_rows)


async def _build_async(
    *,
    build_dir: Path,
    db: str,
    sqlite_path: Path | None,
    meanings_path: Path,
    kb_rows: list[dict],
    benchmark: object | None = None,
) -> None:
    """Async equivalent of the orchestrator phases 1-3 build, used by
    ``ensure_db_cache``. Phase 1 is a sync subprocess; we wrap it in
    ``asyncio.to_thread`` so the event loop stays responsive while
    ``slayer ingest`` runs.

    Also pre-encodes the full (no-deletion) memories.yaml and populates
    embeddings.db so the per-task copy inherits both. Tasks with
    ``deleted_kb_ids`` will overwrite memories.yaml + prune embeddings
    rows at prepare time; tasks with no deletions reuse the cache
    verbatim and pay zero embedding API cost.
    """
    import os as _os

    is_postgres = getattr(benchmark, "db_backend", "sqlite") == "postgres"
    build_dir.mkdir(parents=True, exist_ok=True)

    if is_postgres:
        host = _os.environ.get("BIRD_PG_HOST", "localhost")
        port = _os.environ.get("BIRD_PG_PORT", "5432")
        user = _os.environ.get("BIRD_PG_USER", "bird_interact")
        password = _os.environ.get("BIRD_PG_PASSWORD", "bird_interact")
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        await asyncio.to_thread(_phase1_ingest, db, build_dir, db_url=db_url)
    else:
        if sqlite_path is None:
            raise ValueError("_build_async: sqlite_path is required for non-postgres benchmarks")
        await asyncio.to_thread(_phase1_ingest, db, build_dir, sqlite_path=sqlite_path)

    storage = YAMLStorage(base_dir=str(build_dir))
    _touched, p2_warns = await _phase2_overlay(storage, db, meanings_path)
    if p2_warns:
        logger.info(
            "[slayer_otf] phase2 produced %d warnings for db=%s",
            len(p2_warns), db,
        )
    _added, jsonb_typing, drift = await _phase3_jsonb(
        storage, db, meanings_path=meanings_path, sqlite_path=sqlite_path,
        benchmark=benchmark,
    )
    if jsonb_typing or drift:
        logger.info(
            "[slayer_otf] phase3 produced %d typing warnings, %d drift "
            "findings for db=%s",
            len(jsonb_typing), len(drift), db,
        )

    (build_dir / "_kb_rows.json").write_text(json.dumps(kb_rows, indent=2))

    # Pre-encode the no-deletion memories + their embeddings into the
    # cache. Per-task prepare copies these verbatim (no deletions) or
    # overwrites memories.yaml + prunes the embedding rows for the
    # deleted memory ids (with deletions).
    await _materialise_cache_memories(
        db=db, build_dir=build_dir, kb_rows=kb_rows,
    )


async def _materialise_cache_memories(
    *,
    db: str,
    build_dir: Path,
    kb_rows: list[dict],
) -> None:
    """Write ``memories.yaml`` for the no-deletion case AND populate the
    embedding rows for each memory so SearchService channel 3 (dense
    embedding similarity) can rank them.

    Embedding population is gated on ``embedding_client.is_available()``:
    when the channel is not configured (no API key / no extra), this is
    a no-op and tantivy (channel 2) remains the sole search path. That
    matches SLayer's own write-side semantics for memory creation.

    The embedding API call is **batched** across all KB memories in one
    ``embed_batch`` round-trip, and the resulting rows are persisted via
    a single ``save_embeddings`` write. Calling ``EmbeddingService.
    refresh_memory`` per memory (as the prior version did) issues one
    API request per row, which for a ~60-row KB blows up cost and
    rate-limit risk during cache build.
    """
    import hashlib

    import yaml

    mems = encode_kb_as_memories(db, kb_rows, deleted_kb_ids=set())
    (build_dir / "memories.yaml").write_text(
        yaml.safe_dump(mems, sort_keys=False)
    )

    if not _embeddings_available():
        # Channel disabled (no extra installed, or no API key for the
        # active embedding model). Matches EmbeddingService's own
        # write-side semantics — silently skip, search still works via
        # tantivy.
        return

    model_name = _embedding_current_model()
    # Memory.model_validate is cheap; the encoder's round-trip test
    # already proves all dicts are valid.
    memories = [Memory.model_validate(d) for d in mems]
    texts = [
        render_memory_text_for_embedding(memory=m) for m in memories
    ]
    vectors = await embed_batch(texts, model=model_name)
    rows: list[Embedding] = []
    # strict=True so an embed_batch length mismatch raises instead of
    # silently truncating (would otherwise leave the tail of memories
    # without embedding rows).
    for memory, text, vec in zip(memories, texts, vectors, strict=True):
        if vec is None:
            logger.warning(
                "[slayer_otf] embedding refresh failed for memory %s "
                "in db=%s — falling back to tantivy-only retrieval for "
                "this memory",
                memory.id, db,
            )
            continue
        rows.append(Embedding(
            canonical_id=f"{MEMORY_CANONICAL_PREFIX}{memory.id}",
            embedding_model_name=model_name,
            entity_kind="memory",
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            embedding=vec,
        ))
    if rows:
        storage = YAMLStorage(base_dir=str(build_dir))
        await storage.save_embeddings(rows)
