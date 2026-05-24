"""Per-DB reference build for the DEV-1454 on-the-fly KB-encode setup pass.

Takes the phases-1-3 ingest cache (:func:`cache.ensure_db_cache`), preloads the
full KB as memories, runs the **setup encoder** over the whole KB dependency
DAG (encoding the confidently-encodable items, deferring the ambiguous ones),
annotates each KB's memory with its encode status, and writes a durable,
reviewable reference at ``slayer_models_otf/<db>/`` — never touching the
hand-built ``slayer_models/``.

Lifecycle / concurrency (mirrors :mod:`cache`):

* **Build-if-absent only.** A present reference is reused; a *stale* fingerprint
  marker triggers a WARNING + reuse (never an in-place clobber of a
  reviewed/committed dir). Rebuild is explicit via ``force=True`` or deleting
  the dir.
* We only ever atomic-rename onto an **absent** target (cross-process safe: the
  loser of a rename race discards its tmp and reuses the winner's dir). The
  ``_reference_fp.txt`` marker is written **last**, so a present target is
  always complete.
* A per-DB :class:`asyncio.Lock` serialises concurrent first callers in-process.

The setup encoder itself is injected via ``build_encoder`` (a callable
``(storage, build_dir) -> run_one``) so this module never imports the agent
package — no circular import, and no model/MCP construction lives here.

Embeddings are NOT built explicitly: they auto-create as a side effect of the
encoder's SLayer write tools and the memory-service ``save_memory`` (both gated
on ``embeddings.client.is_available()``); stale ones are inert at search time,
so HARD-8 needs no embedding pruning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.cache import (
    _get_lock,
    ensure_db_cache,
)
from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    _normalise_children,
    encode_kb_as_memories,
)
from bird_interact_agents.slayer_pipeline.portable_connection import (
    resolve_committed_connection_string,
    to_portable_connection_string,
)
from bird_interact_agents.agents._session_log import write_index

logger = logging.getLogger(__name__)

_MARKER = "_reference_fp.txt"
_SETUP_RESULTS = "_setup_results.json"

# `run_one(kb_id, row, deps_results) -> EncoderResult`
_RunOne = Callable[..., Awaitable[Any]]
# `build_encoder(storage, build_dir) -> run_one`
_BuildEncoder = Callable[[Any, Path], _RunOne]


class ReferenceEntry(BaseModel):
    """Result of a successful (or reused) reference materialisation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reference_dir: Path
    fingerprint: str
    kb_rows: list[dict] = Field(default_factory=list)
    setup_results: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dependency edges + cycle detection
# ---------------------------------------------------------------------------


def _edges_from_kb_rows(kb_rows: list[dict]) -> dict[int, list[int]]:
    """Map each KB id to its dependency ids, normalising the raw
    ``children_knowledge`` field (``-1`` / ``None`` / int / list) the same way
    :func:`kb_memory_encoder.encode_kb_as_memories` does."""
    edges: dict[int, list[int]] = {}
    for row in kb_rows:
        edges[int(row["id"])] = _normalise_children(row.get("children_knowledge"))
    return edges


def _cyclic_ids(ids: set[int], edges: dict[int, list[int]]) -> set[int]:
    """Return the ids that cannot be topologically ordered — i.e. every id
    that participates in a dependency cycle OR transitively depends on one.
    Edges to ids outside ``ids`` are treated as already satisfied."""
    deps: dict[int, set[int]] = {
        i: {c for c in edges.get(i, []) if c in ids} for i in ids
    }
    dependents: dict[int, set[int]] = {i: set() for i in ids}
    for i, ds in deps.items():
        for c in ds:
            dependents[c].add(i)

    q: deque[int] = deque(sorted(i for i, ds in deps.items() if not ds))
    seen: set[int] = set()
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        for d in dependents[n]:
            deps[d].discard(n)
            if not deps[d] and d not in seen:
                q.append(d)
    return ids - seen


# ---------------------------------------------------------------------------
# Parallel scheduler — per-KB spawn-time lock + dependency-wait
# ---------------------------------------------------------------------------


async def _encode_all(
    *,
    kb_rows: list[dict],
    edges: dict[int, list[int]],
    run_one: _RunOne,
    concurrency: int = 6,
) -> list[Any]:
    """Encode every KB across the dependency DAG.

    * cycle members (and their transitive dependents) are NOT scheduled — they
      come back ``status="deferred"`` (so the recursive scheme cannot deadlock);
    * each acyclic KB is encoded **exactly once** (per-KB spawn-time lock),
      after its dependencies finish (their results are passed to ``run_one``);
    * independent subtrees run concurrently, bounded by ``concurrency``.
    """
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    rows_by_id = {int(r["id"]): r for r in kb_rows}
    all_ids = set(rows_by_id)
    cyclic = _cyclic_ids(all_ids, edges)
    acyclic = all_ids - cyclic

    # Reverse edges (DEV-1466): parents_by_id[c] = the KBs that list c as a
    # child. Surfaced (acyclic-filtered) to each encoder as `reverse_deps` so a
    # value_illustration can defer an embedded scoring scheme to a
    # calculation_knowledge parent that OWNS that score. Only scheduled
    # (acyclic) parents are surfaced — never a parent that will only cycle-defer.
    parents_by_id: dict[int, set[int]] = {kid: set() for kid in rows_by_id}
    for parent_id, child_ids in edges.items():
        for c in child_ids:
            if c in parents_by_id:
                parents_by_id[c].add(parent_id)

    sem = asyncio.Semaphore(concurrency)
    locks: dict[int, asyncio.Lock] = {}
    results: dict[int, Any] = {}

    async def ensure(kb_id: int) -> Any:
        # Resolve deps first, OUTSIDE this id's lock (deps use their own locks;
        # acyclic ⇒ no deadlock). Blocks if a shared dep is mid-encode.
        dep_ids = [c for c in edges.get(kb_id, []) if c in acyclic]
        dep_results = (
            list(await asyncio.gather(*(ensure(d) for d in dep_ids)))
            if dep_ids else []
        )
        reverse_deps = [
            rows_by_id[p]
            for p in sorted(parents_by_id.get(kb_id, ()))
            if p in acyclic
        ]
        lock = locks.setdefault(kb_id, asyncio.Lock())
        async with lock:
            if kb_id in results:  # another waiter already encoded it
                return results[kb_id]
            async with sem:
                res = await run_one(
                    kb_id, rows_by_id[kb_id], dep_results,
                    reverse_deps=reverse_deps,
                )
            results[kb_id] = res
            return res

    if acyclic:
        await asyncio.gather(*(ensure(i) for i in acyclic))

    out: list[Any] = []
    for kb_id in sorted(all_ids):
        if kb_id in results:
            out.append(results[kb_id])
        else:
            out.append(EncoderResult(
                kb_id=kb_id, status="deferred", entities=[],
                notes=f"dependency cycle: kb {kb_id} cannot be topologically "
                      f"ordered",
                clarifying_questions=[],
            ))
    return out


# ---------------------------------------------------------------------------
# Post-build collision integrity check
# ---------------------------------------------------------------------------


async def _collision_check(results: list[Any], storage: Any, db: str) -> list[Any]:
    """Downgrade every KB whose encoded entity name collides with another KB's
    (same ``(host_model, name, kind)`` written under a different ``kb_id``) AND
    **remove the ambiguous entity from storage**. A clash means one writer
    silently overwrote the other; downgrading the result alone would still leave
    the colliding entity in the committed reference (exposed via models_summary/
    search, with whichever ``meta.kb_id`` survived) — so we delete it and defer
    both KBs for per-task re-encoding (Codex)."""
    owners: dict[tuple[str | None, str, str], set[int]] = {}
    for r in results:
        if r.status != "encoded":
            continue
        for ent in r.entities:
            owners.setdefault(
                (ent.host_model, ent.name, ent.kind), set(),
            ).add(r.kb_id)

    colliding_keys = {k for k, ids in owners.items() if len(ids) > 1}
    if not colliding_keys:
        return results
    colliding_kb_ids: set[int] = set()
    for k in colliding_keys:
        colliding_kb_ids |= owners[k]

    logger.warning(
        "reference_build: entity-name collision across kb_ids %s — removing the "
        "ambiguous entities and deferring", sorted(colliding_kb_ids),
    )
    await _remove_entities_from_storage(storage, db, colliding_keys)

    out: list[Any] = []
    for r in results:
        if r.kb_id in colliding_kb_ids and r.status == "encoded":
            out.append(r.model_copy(update={
                "status": "deferred",
                "entities": [],
                "notes": (r.notes + " | " if r.notes else "")
                + "setup collision: another KB wrote the same entity name; "
                "removed from reference, deferred for per-task encoding",
            }))
        else:
            out.append(r)
    return out


async def _remove_entities_from_storage(
    storage: Any, db: str, keys: set[tuple[str | None, str, str]],
) -> None:
    """Drop each ``(host_model, name, kind)`` entity from the reference. Column/
    measure/aggregation entities are stripped from their host model; a
    query-backed ``model`` entity (host=None) is deleted outright."""
    # Group leaf removals by host model so each model is rewritten once.
    by_host: dict[str, set[tuple[str, str]]] = {}
    model_drops: set[str] = set()
    for host, name, kind in keys:
        if kind == "model" or host is None:
            model_drops.add(name)
        else:
            by_host.setdefault(host, set()).add((name, kind))

    for host, leaves in by_host.items():
        model = await storage.get_model(host)
        if model is None:
            continue
        drop_cols = {n for n, k in leaves if k == "column"}
        drop_meas = {n for n, k in leaves if k == "measure"}
        drop_aggs = {n for n, k in leaves if k == "aggregation"}
        await storage.save_model(model.model_copy(update={
            "columns": [c for c in (model.columns or []) if c.name not in drop_cols],
            "measures": [m for m in (model.measures or []) if m.name not in drop_meas],
            "aggregations": [
                a for a in (model.aggregations or []) if a.name not in drop_aggs
            ],
        }))

    for name in model_drops:
        try:
            await storage.delete_model(name)
        except Exception:  # noqa: BLE001 — already absent is fine
            logger.debug("reference_build: model %r already absent on collision drop", name)


# ---------------------------------------------------------------------------
# Memory annotation
# ---------------------------------------------------------------------------


def _format_setup_section(result: Any) -> str:
    if result.status == "encoded":
        lines = ["--- setup-encode: encoded ---", "entities:"]
        for ent in result.entities:
            lines.append(
                f"  - {ent.entity_ref} (kind={ent.kind}, host={ent.host_model})"
            )
        return "\n".join(lines)
    if result.status == "deferred":
        lines = ["--- setup-encode: deferred ---"]
        if result.notes:
            lines.append(f"ambiguous because: {result.notes}")
        if result.clarifying_questions:
            lines.append("clarify:")
            lines.extend(f"  - {q}" for q in result.clarifying_questions)
        return "\n".join(lines)
    return f"--- setup-encode: error ---\n{result.error or 'unknown error'}"


async def _annotate_memories(
    *, storage: Any, db: str, setup_results: list[Any], kb_rows: list[dict],
) -> None:
    """Re-save each KB's memory with its setup-encode status appended.

    Writes via ``storage.save_memory(id=…)`` (preserves the ``<db>_kb_<n>`` id
    and ``created_at``) then fires the embedding refresh hook directly (gated on
    availability) — keeping the DEV-1455 ``entities`` invariant exactly (no
    resolver mangling) while still populating ``embeddings.db`` as a side
    effect. A KB's concrete entity refs are added ONLY to its own memory
    (Codex #6)."""
    from slayer.embeddings.service import EmbeddingService

    for result in setup_results:
        mem_id = f"{db}_kb_{result.kb_id}"
        mem = await storage.get_memory_row(mem_id)
        if mem is None:
            continue
        new_learning = mem.learning.rstrip() + "\n\n" + _format_setup_section(result)
        new_entities = list(mem.entities)
        if result.status == "encoded":
            for ent in result.entities:
                if ent.entity_ref not in new_entities:
                    new_entities.append(ent.entity_ref)
        saved = await storage.save_memory(
            learning=new_learning, entities=new_entities,
            query=mem.query, id=mem_id,
        )
        try:
            await EmbeddingService(storage=storage).refresh_memory(saved)
        except Exception:  # noqa: BLE001 — embedding refresh is best-effort
            logger.debug("reference_build: embedding refresh skipped for %s", mem_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _load_reference_entry(
    target: Path, fingerprint: str, kb_rows: list[dict],
) -> ReferenceEntry:
    setup_results: list[Any] = []
    sr_path = target / _SETUP_RESULTS
    if sr_path.is_file():
        from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
            EncoderResult,
        )
        try:
            setup_results = [
                EncoderResult.model_validate(d)
                for d in json.loads(sr_path.read_text())
            ]
        except Exception:  # noqa: BLE001 — best-effort reuse metadata
            setup_results = []
    return ReferenceEntry(
        reference_dir=target, fingerprint=fingerprint,
        kb_rows=kb_rows, setup_results=setup_results,
    )


def purge_references(reference_root: Path, dbs) -> list[str]:
    """Delete the per-DB OTF reference dir(s) under ``reference_root`` so the
    next :func:`ensure_db_reference` rebuilds them from scratch.

    The db-level KB-encoded reference is otherwise PRESERVED across runs (reused
    when the fingerprint matches). The ``--otf-rebuild-reference`` run option
    calls this ONCE before the task loop to explicitly drop it, so the lazy
    build regenerates it and every task reuses the fresh copy (no per-task
    ``force``, no concurrency window). Returns the db names actually removed."""
    removed: list[str] = []
    for db in dbs:
        target = Path(reference_root) / db
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(db)
    return removed


async def ensure_db_reference(
    db: str,
    *,
    reference_root: Path,
    cache_root: Path,
    mini_interact_root: Path,
    build_encoder: _BuildEncoder,
    force: bool = False,
) -> ReferenceEntry:
    """Materialise (or reuse) the durable per-DB reference at
    ``reference_root / db``. See module docstring for the lifecycle contract."""
    cache_entry = await ensure_db_cache(
        db, cache_root=cache_root, mini_interact_root=mini_interact_root,
    )
    fp = cache_entry.fingerprint
    kb_rows = cache_entry.kb_rows
    target = reference_root / db
    marker = target / _MARKER
    reference_root.mkdir(parents=True, exist_ok=True)

    # Fast path — reuse only a CURRENT reference. The fingerprint encodes the
    # DB root + sqlite size/mtime + KB + column-meanings (see cache.fingerprint_of),
    # so a marker mismatch means the reference was built against different inputs
    # (changed KB/schema, or a different --db-path) and MUST be rebuilt, not
    # reused (Codex finding) — otherwise a task gets models from one dataset and
    # data resolved against another.
    if not force and marker.is_file() and marker.read_text().strip() == fp:
        return _load_reference_entry(target, fp, kb_rows)

    async with _get_lock(db):
        # Double-check under the lock — a peer may have built a CURRENT one.
        if not force and marker.is_file() and marker.read_text().strip() == fp:
            return _load_reference_entry(target, fp, kb_rows)

        # Rebuild. Overwrite any existing target (an explicit ``force`` or a
        # stale one whose fingerprint no longer matches).
        results = await _build_reference(
            db=db, fp=fp, cache_entry=cache_entry, kb_rows=kb_rows,
            reference_root=reference_root, target=target,
            mini_interact_root=mini_interact_root,
            build_encoder=build_encoder, force=force or target.exists(),
        )
        return ReferenceEntry(
            reference_dir=target, fingerprint=fp,
            kb_rows=kb_rows, setup_results=results,
        )


async def _build_reference(
    *, db, fp, cache_entry, kb_rows, reference_root, target,
    mini_interact_root, build_encoder, force,
) -> list[Any]:
    tmp = Path(tempfile.mkdtemp(prefix=f".{fp}.tmp-", dir=str(reference_root)))
    try:
        # 1. Copy phases-1-3 datasources/ + models/ into tmp.
        shutil.copytree(cache_entry.cache_dir, tmp, dirs_exist_ok=True)

        # 2. RESOLVE the datasource connection to a LIVE absolute path for the
        # build, re-anchored at THIS run's mini_interact_root. The setup
        # encoder must be able to query/validate against the real DB while it
        # encodes — without it, queries fail and the agent thrashes its request
        # budget on un-validatable encodings (UsageLimitExceeded). The cache's
        # baked-in absolute path is NOT trusted: the cache is shared across
        # checkouts/worktrees, so we portabilise-then-resolve to re-anchor at
        # the current root (paths.mini_interact_root() is git-common-dir based,
        # so identical from the main checkout or any worktree). The committed
        # reference is portabilised again at step 6 so the on-disk artifact
        # stays machine-agnostic.
        storage = YAMLStorage(base_dir=str(tmp))
        await _resolve_datasource_for_build(storage, db, mini_interact_root)

        # 3. Preload the full KB as memories (no deletions at build time).
        mems = encode_kb_as_memories(db, kb_rows, deleted_kb_ids=set())
        (tmp / "memories.yaml").write_text(yaml.safe_dump(mems, sort_keys=False))

        # 4. Run the setup encoder over the whole DAG. Pass the real `db` —
        # the encoder must NOT infer it from the tmp build-dir name (Codex).
        # Per-DB setup-encoder session logs (ephemeral /tmp, NOT committed into
        # the reference): one file per kb so any encoder's session is trivially
        # isolatable. Mirrors agent._otf_work_dir's tmp scheme.
        sessions_dir = (
            Path(tempfile.gettempdir())
            / "bird_interact_slayer_otf" / "_setup_sessions" / db
        )
        run_one = build_encoder(storage, tmp, db, sessions_dir)
        # Enter the shared MCP server HERE, in this (parent) task, before the
        # concurrent fan-out — and close it below in the same task. Entering
        # inside a gather'd child task (the old lazy-entry) and exiting here
        # raised "exit cancel scope in a different task" (anyio cancel scopes
        # are task-bound).
        opener = getattr(run_one, "aopen", None)
        if opener is not None:
            await opener()
        try:
            edges = _edges_from_kb_rows(kb_rows)
            results = await _encode_all(
                kb_rows=kb_rows, edges=edges, run_one=run_one,
            )
        finally:
            closer = getattr(run_one, "aclose", None)
            if closer is not None:
                await closer()
        # One-glance triage table over every setup-encoder session.
        write_index(sessions_dir, getattr(run_one, "index_rows", []))

        # 5. Post-build integrity + memory annotation. (Neither queries the
        # DB — they operate on the YAML/memories — so the server is already
        # closed above.)
        results = await _collision_check(results, storage, db)
        await _annotate_memories(
            storage=storage, db=db, setup_results=results, kb_rows=kb_rows,
        )

        # 5b. PORTABILISE the datasource connection back to the relative form
        # so the committed reference is machine-agnostic (the live absolute
        # path from step 2 must never be committed). build_task_variant_storage
        # re-resolves it at task time.
        await _portabilise_datasource(storage, db, mini_interact_root)

        # 6. Persist setup results + marker (marker LAST).
        (tmp / _SETUP_RESULTS).write_text(
            json.dumps([r.model_dump() for r in results], indent=2, default=str)
        )
        (tmp / _MARKER).write_text(fp)

        # 7. Atomic-rename onto the (absent) target.
        _commit_reference(tmp, target, force=force)
        return results
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _effective_db_root(mini_interact_root: Path) -> Path:
    """The root the LIVE SQLite path is anchored at. ``$BIRD_DB_PATH`` wins (it
    is what ``resolve_committed_connection_string`` honours), else
    ``mini_interact_root``. The build-time resolve (step 2) and the
    commit-time portabilise (step 5b) MUST agree on this root — otherwise an
    absolute path resolved under ``$BIRD_DB_PATH`` can't be stripped back to
    the portable relative form when those two roots differ, and the committed
    reference leaks a machine-specific absolute path."""
    env_root = os.environ.get("BIRD_DB_PATH")
    return Path(env_root).expanduser() if env_root else mini_interact_root


async def _portabilise_datasource(
    storage: YAMLStorage, db: str, mini_interact_root: Path,
) -> None:
    root = _effective_db_root(mini_interact_root)
    ds = await storage.get_datasource(db)
    if ds is None or ds.connection_string is None:
        return
    portable = to_portable_connection_string(ds.connection_string, root)
    if portable != ds.connection_string:
        await storage.save_datasource(
            ds.model_copy(update={"connection_string": portable})
        )


async def _resolve_datasource_for_build(
    storage: YAMLStorage, db: str, mini_interact_root: Path,
) -> None:
    """Rewrite the datasource connection to a LIVE absolute SQLite path,
    re-anchored at THIS run's effective DB root, so the setup encoder can
    query/validate the real DB during the build.

    The cache's baked-in absolute path is NOT trusted: the on-the-fly cache is
    shared across checkouts/worktrees, so an absolute path written by another
    checkout would point at a stale (but possibly valid-looking) location. We
    portabilise first (strip the effective-root prefix → relative) and then
    resolve (relative → absolute at the effective root), which re-anchors the
    path no matter which checkout last wrote the cache. The effective root is
    ``$BIRD_DB_PATH`` or ``mini_interact_root`` (the latter is git-common-dir
    based, so identical from the main checkout or any worktree).
    """
    root = _effective_db_root(mini_interact_root)
    ds = await storage.get_datasource(db)
    if ds is None or ds.connection_string is None:
        return
    portable = to_portable_connection_string(ds.connection_string, root)
    resolved = resolve_committed_connection_string(portable, root)
    if resolved != ds.connection_string:
        await storage.save_datasource(
            ds.model_copy(update={"connection_string": resolved})
        )


def _commit_reference(tmp: Path, target: Path, *, force: bool) -> None:
    """Atomic-rename ``tmp`` onto ``target``. We only ever rename onto an
    absent target; ``force`` with an existing target removes it first.

    KNOWN LIMITATION (cross-process, Codex finding — intentionally not hardened):
    the ``force``/stale-rebuild path (rmtree-then-rename) is only safe within a
    SINGLE process. ``ensure_db_reference``'s ``_get_lock(db)`` is an in-process
    asyncio lock, so two SEPARATE ``run`` processes rebuilding the same DB's
    reference at the same time can race — the loser's rmtree could remove the
    winner's freshly committed reference after the winner's tasks began using it.
    The OTF flow assumes single-process-per-run (each run owns its DBs); making
    concurrent multi-process rebuilds of one DB safe would need an inter-process
    file lock or a fully atomic replace. Not done here — documented instead."""
    if force and target.exists():
        shutil.rmtree(target, ignore_errors=True)
    try:
        os.rename(tmp, target)
    except OSError:
        # Cross-process race: a peer won the rename. Treat as success if the
        # target is now complete; otherwise re-raise.
        if not (target / _MARKER).is_file():
            raise
        shutil.rmtree(tmp, ignore_errors=True)
