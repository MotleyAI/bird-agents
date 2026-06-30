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
  A dedicated test re-enables a *spy* ``SearchService.upsert_memory`` (SLayer
  0.7.3+; the legacy ``EmbeddingService.refresh_memory`` path was removed).
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
        db, *, cache_root, mini_interact_root, force=False, benchmark=None,
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
    # DEV-1609: a fresh build reports built=True (drives encode-usage attribution).
    assert entry.built is True


async def test_setup_encode_usage_is_persisted(fake_cache, tmp_path):
    """DEV-1478: the per-DB setup-encode token usage (exposed on run_one.usage)
    is written to `_setup_usage.json` next to the reference, so the otherwise-
    uninstrumented reference-build encode cost is recoverable."""
    import json

    from slayer.core.models import Column

    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.usage import TokenUsage

    def build_encoder(storage, build_dir, db, sessions_dir=None):
        usage = TokenUsage()

        async def run_one(kb_id, row, deps_results, **_):
            usage.add_call(
                scope="setup_encoder", model="anthropic/claude-opus-4-7",
                prompt=1000, completion=200,
            )
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

        run_one.usage = usage
        return run_one

    await _build(tmp_path, build_encoder)
    usage_file = tmp_path / "slayer_models_otf" / DB / "_setup_usage.json"
    assert usage_file.exists(), "reference build must persist _setup_usage.json"
    data = json.loads(usage_file.read_text())
    assert data["n_calls"] == 3  # one add_call per KB (the fake cache has 3)
    assert any(r["scope"] == "setup_encoder" for r in data["breakdown"])
    # setup-encode cost is isolated from the per-task subtotals
    assert data["agent_cost_usd"] == 0.0
    assert data["user_sim_cost_usd"] == 0.0


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

    async def recording_cache(db, *, cache_root, mini_interact_root, force=False, benchmark=None):
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
        db, *, cache_root, mini_interact_root, force=False, benchmark=None,
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


async def test_encoder_connection_reanchored_from_foreign_absolute_cache(
    tmp_path, monkeypatch
):
    """DEV-1478 cloud bug: when the deterministic cache was built on ANOTHER
    machine (absolute path under a foreign root, e.g. a laptop's
    /home/<user>/...), the setup encoder running in a cloud container must
    still see a connection re-anchored at THIS run's root — not the foreign
    path, which doesn't exist here and gives "unable to open database file".

    The old portabilise→resolve logic couldn't strip a foreign prefix, so it
    leaked the stale absolute path. `reanchor_connection_string` force-rewrites
    it. Regression guard."""
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.cache import CacheEntry

    monkeypatch.delenv("BIRD_DB_PATH", raising=False)

    # Cache carries a FOREIGN absolute path (a different machine's layout).
    foreign_abs = "/home/someone-else/Dropbox/SLayer/mini-interact"
    cache_dir = tmp_path / "cache" / DB
    (cache_dir / "datasources").mkdir(parents=True)
    (cache_dir / "models" / DB).mkdir(parents=True)
    (cache_dir / "datasources" / f"{DB}.yaml").write_text(
        f"name: {DB}\ntype: sqlite\n"
        f"connection_string: sqlite:////{foreign_abs.lstrip('/')}/{DB}/{DB}.sqlite\n"
    )
    (cache_dir / "models" / DB / "households.yaml").write_text(
        "name: households\ndata_source: %s\nsql_table: households\n"
        "columns:\n  - name: id\n    primary_key: true\n  - name: income\n"
        "measures: []\naggregations: []\njoins: []\n" % DB
    )
    rows = _kb_rows()
    (cache_dir / "_kb_rows.json").write_text(json.dumps(rows))
    entry = CacheEntry(cache_dir=cache_dir, fingerprint="fp_foreign", kb_rows=rows)

    async def fake_ensure_db_cache(db, *, cache_root, mini_interact_root, force=False, benchmark=None):
        return entry

    monkeypatch.setattr(reference_build, "ensure_db_cache", fake_ensure_db_cache)

    seen: dict[str, str] = {}

    def build_encoder(storage, build_dir, db, sessions_dir=None):
        async def run_one(kb_id, row, deps_results, **_):
            ds = await storage.get_datasource(db)
            seen.setdefault("conn", ds.connection_string)
            from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
                EncoderResult,
            )
            return EncoderResult(
                kb_id=kb_id, status="deferred", entities=[], notes="x",
            )
        return run_one

    await _build(tmp_path, build_encoder)

    conn = seen["conn"]
    expected_abs = (tmp_path / "mini-interact" / DB / f"{DB}.sqlite").resolve()
    assert str(expected_abs) in conn, (
        f"encoder must see a connection re-anchored at the current root; "
        f"got {conn!r}, expected to contain {expected_abs}"
    )
    assert "someone-else" not in conn, (
        f"foreign-machine path must NOT leak into the encoder's connection; "
        f"got {conn!r}"
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


@pytest.mark.asyncio
async def test_collision_downgrade_prunes_memory_backrefs(tmp_path):
    """A collision-downgraded KB must not ship dangling memory backrefs: the
    `<db>_kb_<id>` memory of EVERY collision participant must have the removed
    entity ref pruned (Codex). The surviving storage entity carries only one
    KB's `meta.kb_id`, so pruning must be by REF across all owners — a
    kb_id-keyed purge would miss the loser's backref."""
    from slayer.core.models import Column, DatasourceConfig, SlayerModel
    from slayer.storage.yaml_storage import YAMLStorage
    from bird_interact_agents.slayer_otf.reference_build import _collision_check
    from bird_interact_agents.slayer_otf.encoder_types import (
        EncodedEntity, EncoderResult,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    # The survivor 'dup' carries only kb 8's tag (last writer wins).
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="dup", sql="1", meta={"kb_id": 8}),
        ],
    ))
    # Both KBs autowired a backref to the colliding entity during encoding.
    for kb in (5, 8):
        await storage.save_memory(
            learning=f"KB {kb}",
            entities=[f"{DB}.households.dup", f"{DB}.households.keep_{kb}"],
            query=None, id=f"{DB}_kb_{kb}", description="",
        )

    def _result(kb):
        return EncoderResult(
            kb_id=kb, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="households", name="dup",
                entity_ref=f"{DB}.households.dup",
            )],
            notes="",
        )

    out = await _collision_check([_result(5), _result(8)], storage, tmp_path, DB)

    assert all(r.status == "deferred" for r in out)
    s2 = YAMLStorage(base_dir=str(tmp_path))
    hm = await s2.get_model("households", data_source=DB)
    assert not any(c.name == "dup" for c in (hm.columns or []))
    for kb in (5, 8):
        mem = await s2.get_memory_row(f"{DB}_kb_{kb}")
        assert f"{DB}.households.dup" not in mem.entities, (
            f"stale backref to removed entity survived in kb {kb} memory"
        )
        assert f"{DB}.households.keep_{kb}" in mem.entities, (
            "unrelated backref must be left untouched"
        )


@pytest.mark.asyncio
async def test_collision_downgrade_purges_noncolliding_entities(tmp_path):
    """A KB downgraded for ONE colliding entity must not leave its OTHER
    (non-colliding) entities committed — once deferred, the KB owns no
    reference entities, so the orphan tagged entity + its backref must go
    too (Codex r2)."""
    from slayer.core.models import Column, DatasourceConfig, SlayerModel
    from slayer.storage.yaml_storage import YAMLStorage
    from bird_interact_agents.slayer_otf.reference_build import _collision_check
    from bird_interact_agents.slayer_otf.encoder_types import (
        EncodedEntity, EncoderResult,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="dup", sql="1", meta={"kb_id": 8}),    # collides
            Column(name="solo", sql="2", meta={"kb_id": 5}),   # kb 5 only
        ],
    ))
    await storage.save_memory(
        learning="KB 5", entities=[f"{DB}.households.dup", f"{DB}.households.solo"],
        query=None, id=f"{DB}_kb_5", description="",
    )
    await storage.save_memory(
        learning="KB 8", entities=[f"{DB}.households.dup"],
        query=None, id=f"{DB}_kb_8", description="",
    )

    results = [
        EncoderResult(kb_id=5, status="encoded", entities=[
            EncodedEntity(kind="column", host_model="households", name="dup",
                          entity_ref=f"{DB}.households.dup"),
            EncodedEntity(kind="column", host_model="households", name="solo",
                          entity_ref=f"{DB}.households.solo"),
        ], notes=""),
        EncoderResult(kb_id=8, status="encoded", entities=[
            EncodedEntity(kind="column", host_model="households", name="dup",
                          entity_ref=f"{DB}.households.dup"),
        ], notes=""),
    ]

    out = await _collision_check(results, storage, tmp_path, DB)

    assert all(r.status == "deferred" for r in out)
    s2 = YAMLStorage(base_dir=str(tmp_path))
    cols = {c.name for c in (await s2.get_model("households", data_source=DB)).columns or []}
    assert "dup" not in cols          # colliding entity removed
    assert "solo" not in cols         # non-colliding orphan of a deferred KB removed
    assert "id" in cols               # untagged base column untouched
    mem5 = await s2.get_memory_row(f"{DB}_kb_5")
    assert f"{DB}.households.solo" not in mem5.entities
    assert f"{DB}.households.dup" not in mem5.entities


@pytest.mark.asyncio
async def test_collision_downgrade_prunes_model_leaf_backrefs(tmp_path):
    """When a colliding query-backed MODEL is deleted, leaf backrefs
    `<db>.<model>.<leaf>` in EVERY participant's memory must be pruned by
    prefix, not just the exact `<db>.<model>` ref (Codex r2)."""
    from slayer.core.models import Column, DatasourceConfig, SlayerModel
    from slayer.storage.yaml_storage import YAMLStorage
    from bird_interact_agents.slayer_otf.reference_build import _collision_check
    from bird_interact_agents.slayer_otf.encoder_types import (
        EncodedEntity, EncoderResult,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="qm", data_source=DB, sql_table="qm",
        columns=[Column(name="id", primary_key=True)],
        meta={"kb_id": 8},   # whole query-backed model tagged for the survivor
    ))
    for kb in (5, 8):
        await storage.save_memory(
            learning=f"KB {kb}",
            entities=[f"{DB}.qm", f"{DB}.qm.id", f"{DB}.other.keep"],
            query=None, id=f"{DB}_kb_{kb}", description="",
        )

    def _model_result(kb):
        return EncoderResult(kb_id=kb, status="encoded", entities=[
            EncodedEntity(kind="model", host_model=None, name="qm",
                          entity_ref=f"{DB}.qm"),
        ], notes="")

    out = await _collision_check([_model_result(5), _model_result(8)], storage, tmp_path, DB)

    assert all(r.status == "deferred" for r in out)
    s2 = YAMLStorage(base_dir=str(tmp_path))
    assert await s2.get_model("qm", data_source=DB) is None   # model deleted
    for kb in (5, 8):
        mem = await s2.get_memory_row(f"{DB}_kb_{kb}")
        assert f"{DB}.qm" not in mem.entities          # exact ref pruned
        assert f"{DB}.qm.id" not in mem.entities       # leaf ref pruned by prefix
        assert f"{DB}.other.keep" in mem.entities       # unrelated ref untouched


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
# Edges also come from the `definition` formula (the cross-references that
# `children_knowledge` frequently omits). A `\text{ABBR}` token that names
# another KB term (by its parenthetical abbreviation) becomes a dependency
# edge — so the referent is encoded first and its column exists when the
# dependent's SQL is validated.
# ---------------------------------------------------------------------------


def test_edges_include_formula_derived_deps():
    from bird_interact_agents.slayer_otf.reference_build import (
        _edges_from_kb_rows,
    )

    rows = [
        {"id": 0, "knowledge": "Signal-to-Noise Quality Indicator (SNQI)",
         "definition": r"$\text{SNQI} = \text{SnrRatio} - 0.1$",
         "children_knowledge": -1},
        {"id": 5, "knowledge": "Composite Detection Score (CDS)",
         "definition": r"$\text{CDS} = \text{SNQI} \times 2$",
         "children_knowledge": -1},
    ]
    edges = _edges_from_kb_rows(rows)
    # KB5's formula references SNQI (KB0) even though children_knowledge=-1.
    assert 0 in edges[5]
    # KB0 references only SnrRatio (not a KB term) + itself -> no KB edge.
    assert edges[0] == []


def test_formula_edges_union_with_children_knowledge():
    from bird_interact_agents.slayer_otf.reference_build import (
        _edges_from_kb_rows,
    )

    rows = [
        {"id": 0, "knowledge": "Base (B0)", "definition": "x",
         "children_knowledge": -1},
        {"id": 1, "knowledge": "Mid (M1)", "definition": "y",
         "children_knowledge": -1},
        {"id": 2, "knowledge": "Top (T2)",
         "definition": r"$\text{T2} = \text{M1} + 1$",
         "children_knowledge": [0]},  # declared dep on 0, formula dep on 1
    ]
    edges = _edges_from_kb_rows(rows)
    assert set(edges[2]) == {0, 1}


def test_ambiguous_abbreviation_does_not_create_edge():
    """If a parenthetical abbreviation maps to >1 KB term it's ambiguous —
    emitting an edge on it could create a spurious cycle that defers a large
    valid subtree, so it must be dropped."""
    from bird_interact_agents.slayer_otf.reference_build import (
        _edges_from_kb_rows,
    )

    rows = [
        {"id": 1, "knowledge": "Alpha Metric (AM)", "definition": "x",
         "children_knowledge": -1},
        {"id": 2, "knowledge": "Another Measure (AM)", "definition": "x",
         "children_knowledge": -1},
        {"id": 3, "knowledge": "Uses It (UI)",
         "definition": r"$\text{UI} = \text{AM} + 1$",
         "children_knowledge": -1},
    ]
    edges = _edges_from_kb_rows(rows)
    assert edges[3] == []  # "AM" is ambiguous -> no edge


def test_raw_column_token_suppresses_kb_edge():
    """A `\\text{token}` that names a raw base column must NOT create a KB edge
    even when it ALSO matches a KB abbreviation — raw-column suppression wins,
    so a formula variable that happens to collide with an abbreviation can't
    fabricate a spurious dependency (and cycle)."""
    from bird_interact_agents.slayer_otf.reference_build import (
        _edges_from_kb_rows,
    )

    rows = [
        # "AOI" is BOTH a KB abbreviation (KB1) AND a raw base column below.
        {"id": 1, "knowledge": "Atmospheric Observability Index (AOI)",
         "definition": "x", "children_knowledge": -1},
        {"id": 2, "knowledge": "Detection (DET)",
         "definition": r"$\text{DET} = \text{AOI} \times 2$",
         "children_knowledge": -1},
    ]
    # Without suppression, AOI would resolve to KB1.
    assert _edges_from_kb_rows(rows)[2] == [1]
    # With AOI present as a raw column, the token is suppressed -> no edge.
    assert _edges_from_kb_rows(rows, raw_columns={"AOI"})[2] == []


async def test_formula_only_dep_encoded_before_dependent():
    """End-to-end: a formula-only dependency (children_knowledge=-1) must
    still gate encode order — the referent runs before the dependent."""
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    rows = [
        {"id": 0, "knowledge": "Signal-to-Noise Quality Indicator (SNQI)",
         "definition": r"$\text{SNQI} = \text{SnrRatio}$",
         "type": "calculation_knowledge", "children_knowledge": -1},
        {"id": 5, "knowledge": "Composite Detection Score (CDS)",
         "definition": r"$\text{CDS} = \text{SNQI} \times 2$",
         "type": "calculation_knowledge", "children_knowledge": -1},
    ]
    edges = reference_build._edges_from_kb_rows(rows)
    order: list[int] = []

    async def run_one(kb_id, row, deps_results, *, reverse_deps=None):
        order.append(kb_id)
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    await reference_build._encode_all(kb_rows=rows, edges=edges, run_one=run_one)
    assert order.index(0) < order.index(5)


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


# ---------------------------------------------------------------------------
# DEV-1470 — H4: cross-process per-DB build lock on `_build_reference`.
# Two Ray actors are separate processes on one VM; without `fcntl.flock`,
# `_commit_reference(force=True)` can rmtree a peer's freshly-committed
# reference (the known-limitation comment at reference_build.py:651-663).
# ---------------------------------------------------------------------------


def _holder_proc(args):
    """Acquire `<reference_root>/<db>.build.lock` and hold for `hold_s`, then
    release. Signals readiness by writing a sentinel file."""
    import fcntl

    reference_root_str, db, hold_s, sentinel_path = args
    from pathlib import Path as _P
    import time as _t

    reference_root = _P(reference_root_str)
    reference_root.mkdir(parents=True, exist_ok=True)
    lock_path = reference_root / f"{db}.build.lock"
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        _P(sentinel_path).write_text("locked")
        _t.sleep(hold_s)
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _holder_proc_commit_then_release(args):
    """Acquire the per-DB build flock, COMMIT a complete reference (marker +
    minimal scaffolding files), then release. Models the in-cloud race the
    Codex r2 fix addresses: a peer actor process finishes encoding while
    another is blocked on this same flock; the blocked actor must observe
    the freshly-committed marker once it acquires the lock and reuse instead
    of re-encoding the whole KB."""
    import fcntl
    import json

    reference_root_str, db, target_str, sentinel_path = args
    from pathlib import Path as _P

    reference_root = _P(reference_root_str)
    target = _P(target_str)
    reference_root.mkdir(parents=True, exist_ok=True)
    lock_path = reference_root / f"{db}.build.lock"
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        _P(sentinel_path).write_text("locked")
        # Commit a minimal complete reference: marker LAST, kb rows next to it.
        target.mkdir(parents=True, exist_ok=True)
        (target / "_kb_rows.json").write_text(json.dumps([]))
        (target / "_setup_results.json").write_text(json.dumps([]))
        (target / "_reference_fp.txt").write_text("peer-committed-fp")
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def test_build_reference_takes_cross_process_flock(tmp_path):
    """H4 — `_build_reference` must acquire `fcntl.flock(LOCK_EX)` on
    `<reference_root>/<db>.build.lock` BEFORE it touches `target` or runs the
    encoder, so two concurrent Ray actor processes building the same DB
    serialize instead of `rmtree`-ing each other's committed reference."""
    import asyncio
    import multiprocessing
    import time as _t
    import fcntl  # noqa: F401 — fail loudly if unavailable

    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.cache import CacheEntry

    reference_root = tmp_path / "ref"
    reference_root.mkdir(parents=True)
    db = "db_a"
    target = reference_root / db
    sentinel = tmp_path / "holder_acquired.txt"

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_holder_proc,
        args=((str(reference_root), db, 2.0, str(sentinel)),),
    )
    p.start()
    try:
        # Wait for the holder to take the lock.
        deadline = _t.time() + 5.0
        while not sentinel.exists() and _t.time() < deadline:
            _t.sleep(0.05)
        assert sentinel.exists(), "holder failed to acquire the lock"

        # Stub the encoder + cache so `_build_reference` runs quickly when it
        # finally gets the lock; we only care that it BLOCKED until then.
        cache_dir = tmp_path / "cache" / db
        cache_dir.mkdir(parents=True)
        (cache_dir / "_cache_fp.txt").write_text("fp-x")
        (cache_dir / "datasources").mkdir()
        cache_entry = CacheEntry(
            cache_dir=cache_dir, fingerprint="fp-x", kb_rows=[],
        )

        def fake_build_encoder(_storage, _build_dir, _db, _sessions_dir):
            """Synchronous factory returning the encoder callable, matching
            `_BuildEncoder = Callable[[Any, Path], _RunOne]` at
            reference_build.py:71."""
            class _RunOne:
                def __init__(self):
                    # Instance attribute, not class attribute — avoids the
                    # mutable-default-shared-across-instances trap.
                    self.index_rows: list[dict] = []

                async def __call__(self, *a, **kw):
                    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
                        EncoderResult,
                    )
                    return EncoderResult(
                        kb_id=0, status="encoded", entities=[], notes="",
                    )
            return _RunOne()

        async def run_build():
            t0 = _t.time()
            await reference_build._build_reference(
                db=db, fp="fp-x", cache_entry=cache_entry, kb_rows=[],
                reference_root=reference_root, target=target,
                mini_interact_root=tmp_path / "mini",
                build_encoder=fake_build_encoder,
                force=False,
            )
            return _t.time() - t0

        elapsed = asyncio.run(run_build())
        assert elapsed >= 0.5, (
            f"_build_reference did not block on cross-process flock "
            f"(elapsed={elapsed:.3f}s); H4 race against peer rmtree remains open"
        )
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()


def test_ensure_db_reference_reuses_peer_commit_when_markerless_scrap_exists(tmp_path):
    """CR r2 — when a stale markerless `target` dir exists locally,
    `ensure_db_reference` previously passed `force=force or target.exists()`
    to `_build_reference`, which forwarded `force=True` into
    `_build_reference_inside_lock`'s peer-reuse check (`if not force`),
    DISABLING reuse. With a peer process committing a complete reference
    during the flock wait, this caused rmtree of the peer's freshly
    committed reference (lost work + redundant LLM encode).

    After the fix: `_build_reference_inside_lock` always reuses a present
    marker when the USER didn't explicitly request `force`. The scrap-clear
    responsibility moved into `_commit_reference`, which is never reached
    in this scenario."""
    import asyncio
    import multiprocessing
    import time as _t
    import fcntl  # noqa: F401

    from bird_interact_agents.slayer_otf import reference_build

    reference_root = tmp_path / "ref"
    reference_root.mkdir(parents=True)
    db = "db_a"
    target = reference_root / db
    sentinel = tmp_path / "holder_acquired.txt"

    # Lay down stale markerless scrap locally — this triggers the buggy
    # `force=True` plumbing in the pre-fix code.
    target.mkdir(parents=True)
    (target / "stale_scrap.txt").write_text("leftover from a prior crash")

    # Holder process: hold flock + commit a complete reference, then release.
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_holder_proc_commit_then_release,
        args=((str(reference_root), db, str(target), str(sentinel)),),
    )
    p.start()
    try:
        deadline = _t.time() + 5.0
        while not sentinel.exists() and _t.time() < deadline:
            _t.sleep(0.05)
        assert sentinel.exists(), "holder failed to acquire the lock"

        # Stub the rest of the pipeline so `ensure_db_reference` reaches
        # `_build_reference` via the under-lock path. The peer's commit
        # MUST be honored when we acquire the flock.
        encoder_invocations = {"n": 0}

        def counting_build_encoder(_storage, _build_dir, _db, _sessions_dir):
            encoder_invocations["n"] += 1

            class _RunOne:
                def __init__(self):
                    # Instance attribute, not class attribute — avoids the
                    # mutable-default-shared-across-instances trap.
                    self.index_rows: list[dict] = []

                async def __call__(self, *a, **kw):
                    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
                        EncoderResult,
                    )
                    return EncoderResult(
                        kb_id=0, status="encoded", entities=[], notes="",
                    )
            return _RunOne()

        # Minimal ensure_db_cache stub so we don't depend on dataset on disk.
        from bird_interact_agents.slayer_otf import cache as _cache
        from bird_interact_agents.slayer_otf.cache import CacheEntry

        cache_dir = tmp_path / "cache" / db
        cache_dir.mkdir(parents=True)
        (cache_dir / "_cache_fp.txt").write_text("fp-x")
        (cache_dir / "datasources").mkdir()

        async def fake_ensure_db_cache(db_, *, cache_root, mini_interact_root,
                                        force=False, benchmark=None):
            return CacheEntry(
                cache_dir=cache_dir, fingerprint="fp-x", kb_rows=[],
            )

        import unittest.mock as _mock
        with _mock.patch.object(
            reference_build, "ensure_db_cache", side_effect=fake_ensure_db_cache,
        ):
            entry = asyncio.run(reference_build.ensure_db_reference(
                db,
                reference_root=reference_root,
                cache_root=tmp_path / "cache",
                mini_interact_root=tmp_path / "mini",
                build_encoder=counting_build_encoder,
                force=False,
            ))

        assert encoder_invocations["n"] == 0, (
            f"`force=True` leaked from `target.exists()` plumbing — "
            f"build_encoder invoked {encoder_invocations['n']}x even though "
            f"a peer process committed the reference during the flock wait. "
            f"The user did NOT request force; the peer's marker should have "
            f"been honored."
        )
        # Peer's marker is intact and we returned a valid entry.
        # NOTE: the stale `stale_scrap.txt` remains alongside the peer's
        # committed files — that's by design. The contract is "marker
        # present ⇒ content complete"; unreferenced stale files alongside a
        # complete commit don't violate it. A user-requested rebuild
        # (`force=True`) is the only path that rmtrees the target.
        assert (target / "_reference_fp.txt").read_text() == "peer-committed-fp"
        assert entry.reference_dir == target
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()


def test_build_reference_rechecks_marker_inside_flock(tmp_path):
    """Codex r2 — once `_build_reference_inside_lock` acquires the flock, it
    MUST re-check `target / _MARKER`. If a peer process committed a complete
    reference while we were blocked on flock, we must REUSE it instead of
    invoking the (expensive, LLM-driven) `build_encoder`.

    Without the re-check, two Ray actor processes that both pass
    `ensure_db_reference`'s asyncio-level marker check will BOTH run the
    entire setup encode — wasting one whole LLM run per DB. The marker
    re-check is the only defence inside the same process.
    """
    import asyncio
    import multiprocessing
    import time as _t
    import fcntl  # noqa: F401 — fail loudly if unavailable

    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.cache import CacheEntry

    reference_root = tmp_path / "ref"
    reference_root.mkdir(parents=True)
    db = "db_a"
    target = reference_root / db
    sentinel = tmp_path / "holder_acquired.txt"

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_holder_proc_commit_then_release,
        args=((str(reference_root), db, str(target), str(sentinel)),),
    )
    p.start()
    try:
        deadline = _t.time() + 5.0
        while not sentinel.exists() and _t.time() < deadline:
            _t.sleep(0.05)
        assert sentinel.exists(), "holder failed to acquire the lock"

        cache_dir = tmp_path / "cache" / db
        cache_dir.mkdir(parents=True)
        (cache_dir / "_cache_fp.txt").write_text("fp-x")
        (cache_dir / "datasources").mkdir()
        cache_entry = CacheEntry(
            cache_dir=cache_dir, fingerprint="fp-x", kb_rows=[],
        )

        encoder_invocations = {"n": 0}

        def counting_build_encoder(_storage, _build_dir, _db, _sessions_dir):
            encoder_invocations["n"] += 1

            class _RunOne:
                def __init__(self):
                    # Instance attribute, not class attribute — avoids the
                    # mutable-default-shared-across-instances trap.
                    self.index_rows: list[dict] = []

                async def __call__(self, *a, **kw):
                    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
                        EncoderResult,
                    )
                    return EncoderResult(
                        kb_id=0, status="encoded", entities=[], notes="",
                    )
            return _RunOne()

        async def run_build():
            return await reference_build._build_reference(
                db=db, fp="fp-x", cache_entry=cache_entry, kb_rows=[],
                reference_root=reference_root, target=target,
                mini_interact_root=tmp_path / "mini",
                build_encoder=counting_build_encoder,
                force=False,
            )

        # The peer's commit is in flight; we block on flock until it
        # finishes, then must see the marker and short-circuit.
        entry = asyncio.run(run_build())
        assert encoder_invocations["n"] == 0, (
            f"build_encoder was invoked {encoder_invocations['n']}x after "
            f"peer process committed the reference — Codex r2 marker "
            f"re-check is missing from _build_reference_inside_lock"
        )
        # Reused metadata loads from the peer's _setup_results.json (empty
        # list in this fixture).
        assert entry.setup_results == []
        # DEV-1609: a cross-process peer-reuse is NOT a build — `built` must be
        # False so the caller never re-attributes the peer's setup-encode usage.
        assert entry.built is False
        # And the peer's marker is intact.
        assert (target / "_reference_fp.txt").read_text() == "peer-committed-fp"
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()
