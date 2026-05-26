"""Tests for ``slayer_otf.reference_build.ensure_db_reference`` (DEV-1454).

The reference build is the once-per-DB setup pass: it takes the phases-1-3
ingest cache (``ensure_db_cache``), preloads the full KB as memories, runs the
**setup encoder** over every KB item across the dependency DAG (encoding the
confidently-encodable ones, deferring the ambiguous ones), annotates each KB's
memory, and writes a durable, reviewable reference at
``slayer_models_otf/<db>/`` — never touching the hand-built ``slayer_models/``.

These tests:
* monkeypatch ``reference_build.ensure_db_cache`` so the slayer ingest
  subprocess never runs — the reference build's responsibility is the encode +
  annotate + durable-write orchestration, not ingest.
* inject the setup encoder via the ``build_encoder`` seam (a callable
  ``(storage, build_dir) -> run_one``) so no real LLM / MCP server is spun up.
* disable embeddings (``is_available`` → False) by default so the
  annotation's service ``save_memory`` never calls OpenAI (no-real-APIs rule).
  A dedicated test re-enables a *spy* EmbeddingService.
"""

from __future__ import annotations

import json

import pytest


DB = "fakedb"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _kb_rows() -> list[dict]:
    # KB 2 depends on KB 1 (children_knowledge=[1]); KB 3 is independent.
    return [
        {"id": 1, "knowledge": "K1", "definition": "income > 0",
         "description": "d", "type": "calc", "children_knowledge": -1},
        {"id": 2, "knowledge": "K2", "definition": "uses kb1",
         "description": "d", "type": "calc", "children_knowledge": [1]},
        {"id": 3, "knowledge": "K3", "definition": "tier label",
         "description": "d", "type": "calc", "children_knowledge": -1},
    ]


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """Build a fake phases-1-3 cache dir and monkeypatch
    ``reference_build.ensure_db_cache`` to return a CacheEntry pointing at
    it, so the reference build never runs a real ingest."""
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.cache import CacheEntry

    # conftest sets BIRD_DB_PATH to the REAL mini-interact; drop it so these
    # tests anchor connection strings at the per-test tmp_path root only
    # (deterministic + isolated from the ambient dataset).
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)

    # DEV-1468: the cache is a single authoritative dir per DB (no <fp> level).
    cache_dir = tmp_path / "cache" / DB
    (cache_dir / "datasources").mkdir(parents=True)
    (cache_dir / "models" / DB).mkdir(parents=True)
    # Absolute conn string anchored under the mini_interact_root the build is
    # told about, so the portabilise step can strip it to a relative path.
    abs_sqlite = tmp_path / "mini-interact" / DB / f"{DB}.sqlite"
    (cache_dir / "datasources" / f"{DB}.yaml").write_text(
        f"name: {DB}\ntype: sqlite\n"
        f"connection_string: sqlite:///{abs_sqlite}\n"
    )
    (cache_dir / "models" / DB / "households.yaml").write_text(
        "name: households\ndata_source: %s\nsql_table: households\n"
        "columns:\n  - name: id\n    primary_key: true\n  - name: income\n"
        "measures: []\naggregations: []\njoins: []\n" % DB
    )
    rows = _kb_rows()
    (cache_dir / "_kb_rows.json").write_text(json.dumps(rows))

    entry = CacheEntry(cache_dir=cache_dir, fingerprint="fp_abc123", kb_rows=rows)

    async def fake_ensure_db_cache(
        db, *, cache_root, mini_interact_root, force=False,
    ):
        return entry

    monkeypatch.setattr(
        reference_build, "ensure_db_cache", fake_ensure_db_cache,
    )
    return entry


def _encoded_build_encoder(record):
    """A ``build_encoder`` seam whose ``run_one`` simulates a confident
    encode: it writes a Column tagged ``meta.kb_id`` to ``households`` and
    returns ``status='encoded'``. ``record`` collects the kb_ids in call
    order."""
    from slayer.core.models import Column
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    def build_encoder(storage, build_dir, db, sessions_dir=None):
        async def run_one(kb_id, row, deps_results, **_):
            record.append(kb_id)
            model = await storage.get_model("households")
            name = f"kb{kb_id}_col"
            model.columns.append(Column(name=name, sql="1", meta={"kb_id": kb_id}))
            await storage.save_model(model)
            return EncoderResult(
                kb_id=kb_id, status="encoded",
                entities=[EncodedEntity(
                    kind="column", host_model="households", name=name,
                    entity_ref=f"{DB}.households.{name}",
                )],
                notes="",
            )
        return run_one

    return build_encoder


async def _build(tmp_path, build_encoder, *, force=False):
    from bird_interact_agents.slayer_otf import reference_build

    return await reference_build.ensure_db_reference(
        DB,
        reference_root=tmp_path / "slayer_models_otf",
        cache_root=tmp_path / "cache",
        mini_interact_root=tmp_path / "mini-interact",
        build_encoder=build_encoder,
        force=force,
    )


# ---------------------------------------------------------------------------
# Build-if-absent + encode-all
# ---------------------------------------------------------------------------


async def test_build_when_absent_runs_encoder_over_full_kb(fake_cache, tmp_path):
    """First call builds the reference and runs the setup encoder over EVERY
    KB id (the full set, not a per-task subset)."""
    record: list[int] = []
    entry = await _build(tmp_path, _encoded_build_encoder(record))

    assert sorted(record) == [1, 2, 3], "encoder must run over the full KB set"
    ref = tmp_path / "slayer_models_otf" / DB
    assert entry.reference_dir == ref
    assert (ref / "memories.yaml").exists()
    assert (ref / "models" / DB / "households.yaml").exists()
    assert (ref / "_reference_fp.txt").exists()
    assert {r.kb_id for r in entry.setup_results} == {1, 2, 3}
    assert all(r.status == "encoded" for r in entry.setup_results)


async def test_dependency_encoded_before_dependent(fake_cache, tmp_path):
    """KB 2 depends on KB 1 — the encoder must see 1 finished before 2."""
    record: list[int] = []
    await _build(tmp_path, _encoded_build_encoder(record))
    assert record.index(1) < record.index(2)


async def test_second_call_is_noop_reuse(fake_cache, tmp_path):
    """A present reference with a matching fingerprint marker is reused —
    the encoder is NOT run again."""
    record: list[int] = []
    await _build(tmp_path, _encoded_build_encoder(record))
    first = list(record)
    record.clear()
    await _build(tmp_path, _encoded_build_encoder(record))
    assert record == [], "second call must not re-run the encoder"
    assert sorted(first) == [1, 2, 3]


async def test_force_is_forwarded_to_ensure_db_cache(
    fake_cache, tmp_path, monkeypatch,
):
    """Codex (round-2): ``ensure_db_reference(force=True)`` must FORWARD force
    to ``ensure_db_cache``, else a forced reference rebuild can encode stale
    phase-1-3 cache contents (the CLI ``--otf-rebuild`` covers this via a
    separate purge, but a programmatic ``force=True`` caller shouldn't get
    that footgun)."""
    from bird_interact_agents.slayer_otf import reference_build

    inner = reference_build.ensure_db_cache
    seen: dict = {}

    async def recording_cache(db, *, cache_root, mini_interact_root, force=False):
        seen["force"] = force
        return await inner(
            db, cache_root=cache_root, mini_interact_root=mini_interact_root,
        )

    monkeypatch.setattr(reference_build, "ensure_db_cache", recording_cache)

    # First build (force=False) primes the marker + records seen["force"]=False.
    await _build(tmp_path, _encoded_build_encoder([]))
    assert seen["force"] is False

    # Now force=True must reach ensure_db_cache too.
    seen.clear()
    await _build(tmp_path, _encoded_build_encoder([]), force=True)
    assert seen.get("force") is True, (
        "ensure_db_reference(force=True) must forward force to ensure_db_cache"
    )


async def test_force_rebuilds_even_when_present(fake_cache, tmp_path):
    record: list[int] = []
    await _build(tmp_path, _encoded_build_encoder(record))
    record.clear()
    await _build(tmp_path, _encoded_build_encoder(record), force=True)
    assert sorted(record) == [1, 2, 3], "force=True must re-run the encoder"


async def test_present_marker_is_reused_without_fingerprint_check(
    fake_cache, tmp_path,
):
    """DEV-1468 consolidation: a present ``_reference_fp.txt`` marker is REUSED
    regardless of whether its fingerprint still matches the current inputs.
    Fingerprint gating is removed (accepted tradeoff + provenance warning) —
    a "stale" marker no longer triggers a rebuild; reingest is explicit via
    --otf-rebuild."""
    record: list[int] = []
    await _build(tmp_path, _encoded_build_encoder(record))
    ref = tmp_path / "slayer_models_otf" / DB
    # Scribble a non-matching fingerprint — under the OLD contract this forced
    # a rebuild; under the new one it is reused as-is.
    (ref / "_reference_fp.txt").write_text("STALE_fingerprint")

    record.clear()
    await _build(tmp_path, _encoded_build_encoder(record))
    assert record == [], "a present marker must be reused, not rebuilt"
    # Marker left untouched (no rebuild rewrote it).
    assert (ref / "_reference_fp.txt").read_text().strip() == "STALE_fingerprint"


async def test_reuse_does_not_call_ensure_db_cache(
    fake_cache, tmp_path, monkeypatch,
):
    """Load-bearing for cloud combo 3: when ``slayer_models_otf/<db>/`` is
    present (downloaded), reuse must happen BEFORE ``ensure_db_cache`` — the
    cache is NOT downloaded in cloud, so calling ensure_db_cache would try to
    ingest in-cluster. Monkeypatch ensure_db_cache to explode and prove the
    reuse path never reaches it."""
    from bird_interact_agents.slayer_otf import reference_build

    record: list[int] = []
    await _build(tmp_path, _encoded_build_encoder(record))

    async def boom(*_a, **_k):
        raise AssertionError("ensure_db_cache must not run on reference reuse")

    monkeypatch.setattr(reference_build, "ensure_db_cache", boom)
    record.clear()
    entry = await _build(tmp_path, _encoded_build_encoder(record))
    assert record == [], "encoder must not run on reuse"
    assert entry.reference_dir == tmp_path / "slayer_models_otf" / DB


async def test_no_self_deadlock_when_cache_takes_same_lock(
    fake_cache, tmp_path, monkeypatch,
):
    """Regression (CodeRabbit CRITICAL): ``ensure_db_reference`` must NOT hold
    the per-db ``_get_lock(db)`` while calling ``ensure_db_cache``, which
    acquires the SAME non-reentrant lock. We wrap the fake cache so it takes
    the real lock (mimicking the production cache); a fresh build must COMPLETE,
    not hang. Under the old code (cache called inside the reference lock) this
    times out."""
    import asyncio

    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.cache import _get_lock

    inner = reference_build.ensure_db_cache  # the fake installed by fake_cache

    async def lock_taking_cache(
        db, *, cache_root, mini_interact_root, force=False,
    ):
        async with _get_lock(db):  # the real ensure_db_cache takes this lock
            return await inner(
                db, cache_root=cache_root, mini_interact_root=mini_interact_root,
                force=force,
            )

    monkeypatch.setattr(reference_build, "ensure_db_cache", lock_taking_cache)

    record: list[int] = []
    entry = await asyncio.wait_for(
        _build(tmp_path, _encoded_build_encoder(record)), timeout=10,
    )
    assert sorted(record) == [1, 2, 3]
    assert (entry.reference_dir / "_reference_fp.txt").exists()


async def test_reuse_loads_kb_rows_from_reference_dir(fake_cache, tmp_path):
    """On reuse, kb_rows come from the on-disk ``_kb_rows.json`` in the
    reference dir (self-contained loader), not from a fresh cache build."""
    await _build(tmp_path, _encoded_build_encoder([]))
    entry = await _build(tmp_path, _encoded_build_encoder([]))
    assert [r["id"] for r in entry.kb_rows] == [1, 2, 3]


async def test_marker_present_but_kb_rows_missing_raises(fake_cache, tmp_path):
    """Codex r2 Med#5: a present marker alone is not sufficient. If the marker
    is there but ``_kb_rows.json`` is gone, that's corruption — surface it
    loudly, do NOT silently rebuild or reuse a half-broken reference."""
    await _build(tmp_path, _encoded_build_encoder([]))
    ref = tmp_path / "slayer_models_otf" / DB
    (ref / "_kb_rows.json").unlink()
    with pytest.raises(RuntimeError):
        await _build(tmp_path, _encoded_build_encoder([]))


async def test_marker_present_but_kb_rows_corrupt_raises(fake_cache, tmp_path):
    await _build(tmp_path, _encoded_build_encoder([]))
    ref = tmp_path / "slayer_models_otf" / DB
    (ref / "_kb_rows.json").write_text("{ not json ]")
    with pytest.raises(RuntimeError):
        await _build(tmp_path, _encoded_build_encoder([]))


async def test_empty_kb_rows_is_valid_reuse(fake_cache, tmp_path):
    """Codex (round-3): an empty KB (``[]``) is structurally valid — no
    invariant forbids a DB with zero KB rows. Reuse must accept it, not
    treat it as corruption."""
    await _build(tmp_path, _encoded_build_encoder([]))
    ref = tmp_path / "slayer_models_otf" / DB
    (ref / "_kb_rows.json").write_text("[]")
    entry = await _build(tmp_path, _encoded_build_encoder([]))
    assert entry.kb_rows == []


async def test_reuse_emits_provenance_warning(fake_cache, tmp_path, caplog):
    """Reuse logs a one-line provenance WARNING naming the db so the operator
    knows the fingerprint was not re-checked."""
    import logging

    await _build(tmp_path, _encoded_build_encoder([]))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await _build(tmp_path, _encoded_build_encoder([]))
    assert any(
        r.levelno == logging.WARNING and DB in r.getMessage()
        for r in caplog.records
    ), "reuse must emit a WARNING-level provenance line naming the db"


async def test_purge_caches_removes_only_named_dbs(tmp_path):
    """purge_caches drops the named per-DB cache dirs (the --otf-rebuild path
    wipes BOTH layers) and leaves others + returns what it removed."""
    from bird_interact_agents.slayer_otf.reference_build import purge_caches

    root = tmp_path / "slayer_otf_cache"
    for db in ("households", "crypto", "museum"):
        (root / db).mkdir(parents=True)
        (root / db / "_cache_fp.txt").write_text("fp")

    removed = purge_caches(root, {"households", "crypto", "absent_db"})
    assert sorted(removed) == ["crypto", "households"]
    assert not (root / "households").exists()
    assert not (root / "crypto").exists()
    assert (root / "museum").exists()


async def test_purge_references_removes_only_named_dbs(tmp_path):
    """purge_references drops the named per-DB reference dirs (the
    --otf-rebuild-reference path) and leaves others + returns what it removed."""
    from bird_interact_agents.slayer_otf.reference_build import purge_references

    root = tmp_path / "slayer_models_otf"
    for db in ("households", "crypto", "museum"):
        (root / db).mkdir(parents=True)
        (root / db / "_reference_fp.txt").write_text("fp")

    removed = purge_references(root, {"households", "crypto", "absent_db"})
    assert sorted(removed) == ["crypto", "households"]   # absent_db not removed
    assert not (root / "households").exists()
    assert not (root / "crypto").exists()
    assert (root / "museum").exists()                    # untouched


# ---------------------------------------------------------------------------
# Durability invariants
# ---------------------------------------------------------------------------


async def test_never_writes_under_slayer_models(fake_cache, tmp_path):
    """The build must only ever write under ``slayer_models_otf/`` — the
    hand-built ``slayer_models/`` must be left entirely untouched."""
    committed = tmp_path / "slayer_models" / DB
    committed.mkdir(parents=True)
    (committed / "sentinel.yaml").write_text("name: handbuilt\n")
    before = (committed / "sentinel.yaml").read_text()

    await _build(tmp_path, _encoded_build_encoder([]))

    assert (committed / "sentinel.yaml").read_text() == before
    assert list(committed.iterdir()) == [committed / "sentinel.yaml"]


async def test_reference_datasource_connection_string_is_portable(
    fake_cache, tmp_path,
):
    """The committed reference must carry a PORTABLE (relative) connection
    string so the dir is committable and re-anchored at task time."""
    from slayer.storage.yaml_storage import YAMLStorage

    await _build(tmp_path, _encoded_build_encoder([]))
    ref = tmp_path / "slayer_models_otf" / DB
    ds = await YAMLStorage(base_dir=str(ref)).get_datasource(DB)
    assert ds is not None
    # Portable form is the 3-slash relative scheme, not an absolute path.
    assert not ds.connection_string.startswith("sqlite:////"), (
        f"reference conn string must be portable, got {ds.connection_string!r}"
    )


async def test_datasource_connection_is_live_during_encode(fake_cache, tmp_path):
    """The setup encoder must see a LIVE (absolute, re-anchored) connection
    string while it runs, so it can query/validate against the real DB. The
    old code portabilised BEFORE the encode, leaving the build server with an
    unresolvable relative path — queries failed and the agent thrashed its
    request budget (UsageLimitExceeded). Regression guard: capture the
    connection string the encoder sees, assert it is absolute AND re-anchored
    at THIS run's mini_interact_root (not the cache's baked-in path)."""
    from slayer.storage.yaml_storage import YAMLStorage

    seen: dict[str, str] = {}

    def build_encoder(storage, build_dir, db, sessions_dir=None):
        async def run_one(kb_id, row, deps_results, **_):
            ds = await storage.get_datasource(db)
            # First call wins; all encoders share one storage/datasource.
            seen.setdefault("conn", ds.connection_string)
            from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
                EncoderResult,
            )
            return EncoderResult(
                kb_id=kb_id, status="deferred", entities=[], notes="x",
            )
        return run_one

    entry = await _build(tmp_path, build_encoder)

    # During the encode: absolute (4-slash) and pointing at this run's root.
    conn = seen["conn"]
    assert conn.startswith("sqlite:////"), (
        f"encoder must see a live absolute connection, got {conn!r}"
    )
    expected_abs = (tmp_path / "mini-interact" / DB / f"{DB}.sqlite").resolve()
    assert str(expected_abs) in conn, (
        f"connection must be re-anchored at the current mini_interact_root; "
        f"got {conn!r}, expected to contain {expected_abs}"
    )

    # After commit: portabilised back to the relative form.
    ds = await YAMLStorage(base_dir=str(entry.reference_dir)).get_datasource(DB)
    assert not ds.connection_string.startswith("sqlite:////"), (
        f"committed reference must be portable, got {ds.connection_string!r}"
    )


async def test_concurrent_first_callers_build_once(fake_cache, tmp_path):
    """Two ``asyncio.gather``ed first callers for the same DB must serialise
    on the per-DB lock so the setup encoder runs once and both get the same
    reference (mirrors ``test_slayer_otf_cache``'s concurrency guard)."""
    import asyncio

    record: list[int] = []
    be = _encoded_build_encoder(record)
    a, b = await asyncio.gather(
        _build(tmp_path, be), _build(tmp_path, be),
    )
    assert sorted(record) == [1, 2, 3], (
        f"encoder must run once across concurrent callers; got {record}"
    )
    assert a.reference_dir == b.reference_dir


async def test_no_embeddings_computed_when_unavailable(fake_cache, tmp_path):
    """With embeddings unavailable (the autouse fixture), the write-side
    refresh hooks short-circuit so NO embedding rows are computed (no OpenAI
    call) — search falls back to BM25/tantivy (today's behaviour). The
    sidecar db file may be auto-created empty; what matters is zero rows."""
    import sqlite3

    entry = await _build(tmp_path, _encoded_build_encoder([]))
    db_file = entry.reference_dir / "embeddings.db"
    n = 0
    if db_file.exists():
        con = sqlite3.connect(db_file)
        try:
            n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0  # table never created — also "no embeddings"
        finally:
            con.close()
    assert n == 0, f"no embeddings should be computed when unavailable; got {n}"


async def test_failed_build_leaves_no_reference_dir(fake_cache, tmp_path):
    """If the encoder raises mid-build, the target reference dir must NOT
    exist (build-into-tmp + atomic-rename-onto-absent)."""
    def build_encoder(storage, build_dir, db, sessions_dir=None):
        async def run_one(kb_id, learning, deps_results, **_):
            raise RuntimeError("encoder died")
        return run_one

    with pytest.raises(RuntimeError, match="encoder died"):
        await _build(tmp_path, build_encoder)

    assert not (tmp_path / "slayer_models_otf" / DB).exists()


# ---------------------------------------------------------------------------
# Collision integrity check (Codex #1/#8 residual)
# ---------------------------------------------------------------------------


async def test_same_entity_name_collision_is_downgraded(fake_cache, tmp_path):
    """Two independent KBs that write the SAME entity name on the same host
    model are flagged by the post-build integrity check and downgraded so
    the collision surfaces for review rather than silently overwriting."""
    from slayer.core.models import Column
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    def build_encoder(storage, build_dir, db, sessions_dir=None):
        async def run_one(kb_id, row, deps_results, **_):
            model = await storage.get_model("households")
            # Every KB claims the SAME entity name -> collision. Upsert by
            # name (as edit_model does) so storage stays valid; the clash is
            # in the EncoderResults the collision check inspects.
            if not any(c.name == "dup" for c in model.columns):
                model.columns.append(
                    Column(name="dup", sql="1", meta={"kb_id": kb_id}),
                )
                await storage.save_model(model)
            return EncoderResult(
                kb_id=kb_id, status="encoded",
                entities=[EncodedEntity(
                    kind="column", host_model="households", name="dup",
                    entity_ref=f"{DB}.households.dup",
                )],
                notes="",
            )
        return run_one

    entry = await _build(tmp_path, build_encoder)
    statuses = {r.kb_id: r.status for r in entry.setup_results}
    # Every collision participant (all three claim "dup") must be downgraded
    # to deferred so the clash surfaces for review, not silently overwrite.
    assert all(s == "deferred" for s in statuses.values()), (
        f"all name-collision participants must be deferred; got {statuses}"
    )
    # ...and the ambiguous entity must be REMOVED from the reference storage,
    # not just downgraded in the result metadata (Codex).
    from slayer.storage.yaml_storage import YAMLStorage
    households = await YAMLStorage(
        base_dir=str(entry.reference_dir)
    ).get_model("households")
    assert households is not None
    assert not any(c.name == "dup" for c in (households.columns or [])), (
        "colliding entity 'dup' must be removed from the reference"
    )


# ---------------------------------------------------------------------------
# _edges_from_kb_rows
# ---------------------------------------------------------------------------


def test_edges_from_kb_rows_normalises_children_variants():
    from bird_interact_agents.slayer_otf.reference_build import (
        _edges_from_kb_rows,
    )

    rows = [
        {"id": 1, "children_knowledge": -1},      # sentinel -> no deps
        {"id": 2, "children_knowledge": 1},       # scalar int
        {"id": 3, "children_knowledge": [1, 2]},  # list
        {"id": 4, "children_knowledge": None},    # missing -> no deps
    ]
    edges = _edges_from_kb_rows(rows)
    assert edges[1] == []
    assert edges[2] == [1]
    assert sorted(edges[3]) == [1, 2]
    assert edges[4] == []


# ---------------------------------------------------------------------------
# Reverse-dependency wiring (DEV-1466): `_encode_all` must hand each KB the rows
# of the KBs that REFERENCE it (its parents), so a value_illustration can defer
# an embedded scoring scheme to a calculation_knowledge parent. Only SCHEDULED
# (acyclic) parents are surfaced — never a parent that will only cycle-defer.
# ---------------------------------------------------------------------------


async def test_encode_all_passes_reverse_deps_to_run_one():
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    rows = [
        {"id": 3, "knowledge": "Water", "type": "value_illustration",
         "children_knowledge": -1},
        {"id": 4, "knowledge": "Road", "type": "value_illustration",
         "children_knowledge": -1},
        {"id": 5, "knowledge": "Park", "type": "value_illustration",
         "children_knowledge": -1},
        {"id": 6, "knowledge": "Dwelling Type", "type": "value_illustration",
         "children_knowledge": -1},
        {"id": 13, "knowledge": "Infra Score", "type": "calculation_knowledge",
         "children_knowledge": [3, 4, 5]},
        {"id": 44, "knowledge": "Dwelling Type Score",
         "type": "calculation_knowledge", "children_knowledge": [6]},
        {"id": 20, "knowledge": "Living Condition Score",
         "type": "calculation_knowledge", "children_knowledge": [6, 13]},
    ]
    edges = reference_build._edges_from_kb_rows(rows)
    seen: dict[int, list[dict]] = {}

    # keyword-only `reverse_deps` enforces the plan's kwarg contract: if the
    # implementation passes it positionally, this run_one raises.
    async def run_one(kb_id, row, deps_results, *, reverse_deps=None):
        seen[kb_id] = list(reverse_deps or [])
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    await reference_build._encode_all(kb_rows=rows, edges=edges, run_one=run_one)

    def ids(kb):
        return {p["id"] for p in seen[kb]}

    assert ids(6) == {20, 44}      # KB-6 is referenced by KB-20 and KB-44
    assert ids(3) == {13}          # component score referenced only by its averager
    assert ids(13) == {20}
    assert ids(44) == set()        # nobody references KB-44
    assert ids(20) == set()
    # full parent ROWS are passed (type/knowledge/definition), not bare ids —
    # the reverse-deps block needs them to drive the DUPLICATE-SCORE GUARD.
    kb44 = next(p for p in seen[6] if p["id"] == 44)
    assert kb44["type"] == "calculation_knowledge"
    assert kb44["knowledge"] == "Dwelling Type Score"


async def test_encode_all_reverse_deps_excludes_cyclic_parents():
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    # 6 is an acyclic leaf; 7 is an acyclic parent of 6; 50<->51 form a cycle and
    # 50 also lists 6 as a child; 52 transitively depends on the cycle (via 50)
    # and also lists 6. Only 7 may surface in 6's reverse_deps — 50/51 (cycle
    # members) and 52 (transitive dependent) are never scheduled.
    rows = [
        {"id": 6, "knowledge": "leaf", "type": "value_illustration",
         "children_knowledge": -1},
        {"id": 7, "knowledge": "acyclic parent", "type": "calculation_knowledge",
         "children_knowledge": [6]},
        {"id": 50, "knowledge": "cyc a", "type": "calculation_knowledge",
         "children_knowledge": [6, 51]},
        {"id": 51, "knowledge": "cyc b", "type": "calculation_knowledge",
         "children_knowledge": [50]},
        {"id": 52, "knowledge": "transitive dependent of cycle",
         "type": "calculation_knowledge", "children_knowledge": [6, 50]},
    ]
    edges = reference_build._edges_from_kb_rows(rows)
    seen: dict[int, set[int]] = {}

    async def run_one(kb_id, row, deps_results, *, reverse_deps=None):
        seen[kb_id] = {p["id"] for p in (reverse_deps or [])}
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    await reference_build._encode_all(kb_rows=rows, edges=edges, run_one=run_one)

    assert seen[6] == {7}          # only the acyclic parent surfaces
    # cycle members + transitive dependents are never scheduled (so absent here)
    assert 50 not in seen and 51 not in seen and 52 not in seen
