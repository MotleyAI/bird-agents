"""End-to-end smoke test for the on-the-fly setup path.

Picks the smallest real mini-interact DB by KB row count and exercises
``ensure_db_cache`` + ``prepare_task_storage`` + ``SearchService`` on it.
No LLM agent — this test is intentionally cheap so it can run in CI as
long as the mini-interact data submodule is present.

Skipped cleanly when ``paths.benchmark_data_root("mini-interact")`` doesn't resolve.

The behavioural contract under test (Codex finding): the agent's
``search(question=..., datasource=db)`` MUST surface the KB memories
we encode. Asserting only YAML round-trip is insufficient — the
``SearchService._filter_memories_by_datasource`` filter (slayer/search/
service.py:267) rejects memories whose entities list has no
datasource-rooted entry. The bare ``db`` entity is the cheap fix.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from slayer.search.service import SearchService
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents import paths


# Real `slayer ingest` subprocess + real SearchService → integration.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# DB picker
# ---------------------------------------------------------------------------


def _has_resolvable_child(row: dict, kb_ids: set[int]) -> bool:
    """Mirror the encoder's drop rules: a child is "expected" only when
    it is present in ``kb_ids`` and is not a self-reference."""
    cks = row.get("children_knowledge")
    self_id = int(row["id"])
    if isinstance(cks, list):
        return any(int(c) in kb_ids and int(c) != self_id for c in cks)
    if isinstance(cks, int) and cks >= 0:
        return cks in kb_ids and cks != self_id
    return False


def _pick_smallest_db() -> tuple[str, Path, list[dict]] | None:
    """Return ``(db_name, kb_path, kb_rows)`` for the mini-interact DB
    with the fewest KB rows, or ``None`` if the data isn't present."""
    root = paths.benchmark_data_root("mini-interact")
    if not root.exists():
        return None
    candidates: list[tuple[int, str, Path, list[dict]]] = []
    for db_dir in root.iterdir():
        if not db_dir.is_dir() or db_dir.name.startswith("."):
            continue
        kb_path = db_dir / f"{db_dir.name}_kb.jsonl"
        sqlite_path = db_dir / f"{db_dir.name}.sqlite"
        column_meaning_path = (
            db_dir / f"{db_dir.name}_column_meaning_base.json"
        )
        if not kb_path.is_file() or not sqlite_path.is_file():
            continue
        if not column_meaning_path.is_file():
            continue
        rows = [
            json.loads(line)
            for line in kb_path.read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            continue
        candidates.append((len(rows), db_dir.name, kb_path, rows))
    if not candidates:
        return None
    candidates.sort()
    _, db, kb_path, rows = candidates[0]
    return db, kb_path, rows


@pytest.fixture(scope="module")
def smallest_db():
    found = _pick_smallest_db()
    if found is None:
        pytest.skip("mini-interact data not available")
    return found


@pytest.fixture(scope="module")
def shared_cache_root(tmp_path_factory):
    """One cache root for the whole module. ``ensure_db_cache`` is
    idempotent under a stable fingerprint, so every test that ingests
    the same ``smallest_db`` reuses the same physical cache dir built
    by the first test that runs. Drops `slayer ingest` cost from N×~8s
    to a single build."""
    return tmp_path_factory.mktemp("slayer_otf_cache_shared")


@pytest.fixture(autouse=True)
def _fake_embedding_api(monkeypatch):
    """Replace the real embedding API with a deterministic fake for the
    entire smoke-test module.

    Why: ``_materialise_cache_memories`` calls ``embed_batch`` whenever
    ``is_available()`` is True. Letting that hit the real API per test
    means ~60 vector calls per cache build × 6 tests = ~360 API calls
    per smoke-test run, plus minutes of wall time and dollars. The
    behaviour we're verifying (rows land in embeddings.db, deletion
    prunes them, fingerprint includes the model name) doesn't depend
    on the vectors being real — only that they're shaped correctly
    and persist through the storage round-trip.
    """
    from bird_interact_agents.slayer_otf import cache as otf_cache

    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model", lambda: "fake-embed-v1",
    )

    async def fake_embed_batch(texts, *, model=None):
        # Return a unique, deterministic 4-d vector per input. Determinism
        # matters for the fingerprint test (same input → same vector →
        # same hash); uniqueness keeps the embedding rows distinguishable.
        return [
            [float(len(t) % 7), float(hash(t) & 0xFFFF) / 65535.0, 0.5, -0.5]
            for t in texts
        ]

    monkeypatch.setattr(otf_cache, "embed_batch", fake_embed_batch)
    return None


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


async def test_on_the_fly_build_produces_expected_layout(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """``ensure_db_cache`` + ``prepare_task_storage`` materialise a
    scratch dir with: ``datasources/<db>.yaml`` (absolute connection
    string), ``models/<db>/*.yaml`` (one per sqlite table), and
    ``memories/<id>.md`` (one per KB row, no deletions — slayer 0.9.6)."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db,
        cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )

    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db,
        deleted_kb_ids=set(),
        cache_entry=entry,
        work_dir=work_dir,
    )

    # Layout
    assert (scratch / "datasources" / f"{db}.yaml").is_file()
    # DEV-1668: slayer 0.9.6 stores per-id ``memories/<id>.md`` (not flat).
    assert any((scratch / "memories").glob("*.md"))

    # Datasource connection string resolves to an existing absolute file.
    storage = YAMLStorage(base_dir=str(scratch))
    ds = await storage.get_datasource(db)
    assert ds is not None and ds.connection_string is not None
    conn = ds.connection_string
    assert conn.startswith("sqlite:///"), (
        f"connection_string should be a sqlite:/// URL, got {conn!r}"
    )
    # Extract the file path and assert it points at a real file.
    file_part = conn.removeprefix("sqlite:///")
    # SLayer's portable connection resolver uses 4-slash for absolutes;
    # both 3-slash (relative) and 4-slash (absolute) are sqlite:/// prefixed.
    # Strip leading slashes that come from absolute-URL trickery.
    if file_part.startswith("/"):
        target = Path(file_part)
    else:
        target = Path("/" + file_part)
    assert target.is_file(), (
        f"datasource sqlite file should exist post-prepare, missing: {target}"
    )

    # Models count matches sqlite table count.
    sqlite_path = paths.benchmark_data_root("mini-interact") / db / f"{db}.sqlite"
    con = sqlite3.connect(sqlite_path)
    try:
        # Exclude SQLite engine-internal tables (sqlite_sequence,
        # sqlite_stat*, etc.) — they're never modelled in our YAML
        # output, so including them here would false-fail on any DB
        # that happens to use autoincrement.
        sqlite_tables = {
            r[0] for r in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        con.close()
    yaml_models = set(await storage.list_models())
    assert yaml_models == sqlite_tables, (
        f"models on disk should match sqlite tables exactly; "
        f"sqlite-only={sqlite_tables - yaml_models}, "
        f"yaml-only={yaml_models - sqlite_tables}"
    )


async def test_memories_match_kb_rows_and_carry_bare_db_entity(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """One memory per KB row; ``entities[0] == db`` on every memory
    (datasource eligibility for ``SearchService``)."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db, deleted_kb_ids=set(), cache_entry=entry, work_dir=work_dir,
    )

    storage = YAMLStorage(base_dir=str(scratch))
    memories = await storage.list_memories(entities=None)
    assert len(memories) == len(kb_rows), (
        f"expected {len(kb_rows)} memories, got {len(memories)}"
    )
    for m in memories:
        assert m.entities and m.entities[0] == db, (
            f"memory {m.id} must lead with bare datasource entity; "
            f"got {m.entities}"
        )


async def test_at_least_one_cross_ref_when_corpus_has_one(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """If the smallest DB's source data references at least one other KB,
    the encoded memories must carry at least one ``memory:<db>_kb_<n>``
    cross-ref. Defends against silently dropping the cross-ref graph."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    # Only count cross-refs the encoder would actually emit (child in
    # kb id set, not a self-ref). Raw children_knowledge can have refs
    # that are correctly dropped, so a raw-length check would falsely
    # expect cross-refs the encoder is required to omit.
    kb_ids = {int(r["id"]) for r in kb_rows}
    expected = any(_has_resolvable_child(r, kb_ids) for r in kb_rows)
    if not expected:
        pytest.skip("smallest DB has no resolvable cross-refs to verify")

    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db, deleted_kb_ids=set(), cache_entry=entry, work_dir=work_dir,
    )
    storage = YAMLStorage(base_dir=str(scratch))
    memories = await storage.list_memories(entities=None)
    cross_refs = [
        e for m in memories for e in m.entities
        if e.startswith(f"memory:{db}_kb_")
    ]
    assert cross_refs, (
        f"expected at least one cross-ref entity in encoded memories; "
        f"corpus has expected children_knowledge entries"
    )

    # Sanity: every cross-ref points at an actual memory in the same store.
    mem_ids = {m.id for m in memories}
    for ref in cross_refs:
        target = ref.removeprefix("memory:")
        assert target in mem_ids, (
            f"dangling memory ref in encoded output: {ref} (not in {mem_ids})"
        )


# ---------------------------------------------------------------------------
# SearchService end-to-end — the actual contract
# ---------------------------------------------------------------------------


async def test_search_service_recency_fallback_under_datasource_filter(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """The recency-fallback path (no question, no entities, no query) is
    also datasource-scoped. With the bare-``db`` entity we attach, the
    fallback must return > 0 memories — otherwise our memories aren't
    eligible at all and the implementation is broken regardless of
    the question channel's behaviour."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db, deleted_kb_ids=set(), cache_entry=entry, work_dir=work_dir,
    )

    storage = YAMLStorage(base_dir=str(scratch))
    service = SearchService(storage=storage)

    # DEV-1546: slayer 0.7.2 collapsed per-kind caps into ``max_results``
    # and the unified ``SearchResponse.results`` list; filter to memory hits.
    response = await service.search(
        datasource=db,
        max_results=5,
        cypher_filter="MATCH (n:Memory) RETURN n.id AS id",
    )
    memory_hits = [h for h in response.results if h.kind == "memory"]
    assert memory_hits, (
        f"datasource-scoped recency fallback returned 0 memory hits — the "
        f"bare {db!r} entity is missing from one or more memories, OR the "
        f"datasource filter is rejecting them. Either way, the agent will "
        f"never see KB content under this mode."
    )


async def test_search_service_surfaces_kb_memory_under_datasource_filter(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """The Codex-flagged contract: ``SearchService.search(question=...,
    datasource=db)`` must return at least one of our KB memories.

    Picks the first KB row's ``knowledge`` title as the query string, so
    the tantivy full-text channel (channel 2) has a real signal to rank on.
    """
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db, deleted_kb_ids=set(), cache_entry=entry, work_dir=work_dir,
    )

    storage = YAMLStorage(base_dir=str(scratch))
    service = SearchService(storage=storage)

    # Query for a distinctive token from the first KB row's title.
    first = kb_rows[0]
    query = first["knowledge"]

    # DEV-1546: slayer 0.7.2 unified the per-kind buckets into
    # ``SearchResponse.results``; filter to memory hits.
    response = await service.search(
        question=query, datasource=db,
        max_results=10,
        cypher_filter="MATCH (n:Memory) RETURN n.id AS id",
    )

    memory_hits = [h for h in response.results if h.kind == "memory"]
    assert memory_hits, (
        f"SearchService returned 0 memory hits under datasource={db!r} for "
        f"question={query!r}; this is the Codex-flagged datasource-filter "
        f"regression — every KB memory needs the bare {db!r} as its first "
        f"entity to be eligible."
    )
    # And the first KB row's own memory is among the hits.
    expected_id = f"{db}_kb_{first['id']}"
    hit_ids = {h.id for h in memory_hits}
    assert expected_id in hit_ids, (
        f"expected memory {expected_id!r} in search hits; got {hit_ids!r}"
    )


# ---------------------------------------------------------------------------
# Deletion smoke
# ---------------------------------------------------------------------------


async def test_memory_embeddings_are_populated_in_cache_when_available(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """Codex finding (group C): KB memories must have embedding rows in
    ``embeddings.db`` so SearchService channel 3 (dense embedding
    similarity) can rank them. The cache builds embeddings once per DB;
    per-task scratch copies them via copytree.

    The autouse ``_fake_embedding_api`` fixture monkeypatches the
    embedding API to return deterministic fake vectors, so this test
    runs offline and without API cost — but still exercises the real
    persist path through ``save_embeddings`` and the sqlite write."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db, deleted_kb_ids=set(),
        cache_entry=entry, work_dir=work_dir,
    )

    emb_path = scratch / "embeddings.db"
    assert emb_path.is_file(), (
        f"embeddings.db should exist in scratch when embeddings channel "
        f"is configured; got missing"
    )

    con = sqlite3.connect(emb_path)
    try:
        kb_memory_canonicals = {
            r[0] for r in con.execute(
                "SELECT canonical_id FROM embeddings "
                "WHERE canonical_id LIKE 'memory:' || ? || '_kb_%'",
                (db,),
            )
        }
    finally:
        con.close()
    expected = {f"memory:{db}_kb_{r['id']}" for r in kb_rows}
    assert kb_memory_canonicals == expected, (
        f"every KB memory must have an embedding row; "
        f"missing={expected - kb_memory_canonicals}, "
        f"extra={kb_memory_canonicals - expected}"
    )


async def test_deletion_prunes_embedding_rows(smallest_db, shared_cache_root: Path, tmp_path: Path):
    """When a task deletes KB memories, the per-task scratch's
    embeddings.db must NOT carry the deleted memory's embedding row
    (otherwise channel 3 search would return ghost hits).

    Runs against the autouse fake embedding API — see the
    ``_fake_embedding_api`` fixture."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db
    deleted = {int(kb_rows[0]["id"])}
    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db, deleted_kb_ids=deleted,
        cache_entry=entry, work_dir=work_dir,
    )
    con = sqlite3.connect(scratch / "embeddings.db")
    try:
        present = {
            r[0] for r in con.execute(
                "SELECT canonical_id FROM embeddings "
                "WHERE canonical_id LIKE 'memory:' || ? || '_kb_%'",
                (db,),
            )
        }
    finally:
        con.close()
    deleted_canonical = f"memory:{db}_kb_{kb_rows[0]['id']}"
    assert deleted_canonical not in present, (
        f"deleted KB memory's embedding row should be pruned; still "
        f"present in {present}"
    )


async def test_prepare_rewrites_stale_absolute_connection_string(tmp_path: Path):
    """Codex finding (group B follow-through): when the cache's
    datasource YAML carries an absolute sqlite path that points OUTSIDE
    the current mini-interact root, ``prepare_task_storage`` must
    re-anchor it. This is the belt-and-suspenders behind the
    fingerprint fix — if a stale absolute path ever slips through
    (e.g. cache reused across two roots that the fingerprint didn't
    actually distinguish), runtime catches it before SLayer opens the
    wrong sqlite.

    Builds a fake cache dir by hand (no slayer ingest subprocess), so
    this test runs in <100ms instead of ~10s. The path under test —
    ``_rewrite_datasource_connection_string`` inside
    ``prepare_task_storage`` — doesn't care whether the cache was
    produced by the orchestrator or hand-crafted; it just reads the
    datasource YAML and rewrites the connection string.
    """
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db = "fakedb"
    fake_root = tmp_path / "mini-interact-fake"
    sqlite_file = fake_root / db / f"{db}.sqlite"
    sqlite_file.parent.mkdir(parents=True)
    sqlite_file.write_bytes(b"")  # presence-only; not opened in this test

    # Hand-build a minimal cache: datasources/<db>.yaml carrying a
    # bogus absolute path that points at a different mini-interact
    # location, plus an empty models/<db>/ dir so YAMLStorage init is
    # happy.
    fake_cache = tmp_path / "fake_cache"
    (fake_cache / "datasources").mkdir(parents=True)
    (fake_cache / "models" / db).mkdir(parents=True)
    bogus_sqlite = tmp_path / "elsewhere" / db / f"{db}.sqlite"
    bogus_conn = f"sqlite:////{bogus_sqlite.as_posix().lstrip('/')}"
    (fake_cache / "datasources" / f"{db}.yaml").write_text(
        f"name: {db}\ntype: sqlite\nconnection_string: {bogus_conn}\n"
    )

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    fake_entry = otf_cache.CacheEntry(
        cache_dir=fake_cache, fingerprint="fake", kb_rows=[],
    )

    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=fake_root,
        db=db, deleted_kb_ids=set(),
        cache_entry=fake_entry, work_dir=work_dir,
    )
    scratch_storage = YAMLStorage(base_dir=str(scratch))
    ds_fixed = await scratch_storage.get_datasource(db)
    assert ds_fixed is not None and ds_fixed.connection_string is not None
    assert "elsewhere" not in ds_fixed.connection_string, (
        f"stale absolute path was not rewritten; ds.connection_string="
        f"{ds_fixed.connection_string!r}"
    )
    # And it now points at the expected file under the real root.
    file_part = ds_fixed.connection_string.removeprefix("sqlite:///")
    target = Path(file_part) if file_part.startswith("/") else Path("/" + file_part)
    assert target == sqlite_file.resolve(), (
        f"rewritten path should point at {sqlite_file.resolve()}; got {target}"
    )


async def test_deletion_removes_memory_and_strips_dangling_refs(
    smallest_db, shared_cache_root: Path, tmp_path: Path,
):
    """End-to-end deletion: pick a KB row that is referenced by another
    row, mark it as ``deleted``, and assert (a) its memory is absent,
    (b) the referring memory's entities no longer contains its
    ``memory:<id>`` token."""
    from bird_interact_agents.slayer_otf import (
        cache as otf_cache,
        runtime as otf_runtime,
    )

    db, _kb_path, kb_rows = smallest_db

    # Find a (referrer, referred) pair to exercise the strip path.
    by_id = {int(r["id"]): r for r in kb_rows}
    referrer = None
    referred = None
    for r in kb_rows:
        cks = r.get("children_knowledge")
        if isinstance(cks, list) and cks:
            for c in cks:
                if int(c) in by_id and int(c) != int(r["id"]):
                    referrer = int(r["id"])
                    referred = int(c)
                    break
        elif (
            isinstance(cks, int)
            and cks >= 0
            and cks in by_id
            and cks != int(r["id"])
        ):
            referrer = int(r["id"])
            referred = int(cks)
        if referrer is not None:
            break
    if referrer is None or referred is None:
        pytest.skip("smallest DB has no parent→child KB pair to exercise")

    cache_root = shared_cache_root
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    entry = await otf_cache.ensure_db_cache(
        db, cache_root=cache_root,
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
    )
    scratch = await otf_runtime.prepare_task_storage(
        mini_interact_root=paths.benchmark_data_root("mini-interact"),
        db=db,
        deleted_kb_ids={referred},
        cache_entry=entry,
        work_dir=work_dir,
    )

    storage = YAMLStorage(base_dir=str(scratch))
    memories = await storage.list_memories(entities=None)
    mem_ids = {m.id for m in memories}
    referred_memory_id = f"{db}_kb_{referred}"
    referrer_memory_id = f"{db}_kb_{referrer}"

    assert referred_memory_id not in mem_ids, (
        f"deleted KB {referred} should be absent; ids={sorted(mem_ids)}"
    )

    referrer_mem = next(m for m in memories if m.id == referrer_memory_id)
    assert f"memory:{referred_memory_id}" not in referrer_mem.entities, (
        f"referrer {referrer_memory_id} should not carry a stale ref to "
        f"deleted {referred_memory_id}; entities={referrer_mem.entities}"
    )
