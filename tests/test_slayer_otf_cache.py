"""Tests for ``slayer_otf.cache.ensure_db_cache`` (DEV-1468 consolidation).

The cache materialises orchestrator phases 1-3 (slayer ingest +
column-meaning overlay + JSONB-leaf expansion) into a SINGLE authoritative
dir per DB: ``<cache_root>/<db>/`` (the ``<fingerprint>`` path level is
gone). Reuse is **presence-gated** on the ``_cache_fp.txt`` completeness
marker (written last); a rebuild happens only when the marker is ABSENT or
``force=True``. ``fingerprint_of`` is retained as provenance only — written
to ``_cache_fp.txt`` at build time, loaded back on reuse — and no longer
names the dir or gates reuse.

These tests monkeypatch the orchestrator phase functions so the slayer
CLI subprocess is never actually invoked — the cache layer's
responsibility is *orchestration*, not the orchestrator's behaviour.

Phase 4 (LLM date detection) is explicitly NOT called by the cache layer.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from bird_interact_agents.slayer_otf import cache as otf_cache


# ---------------------------------------------------------------------------
# Fixture: minimal fake mini-interact root with one DB
# ---------------------------------------------------------------------------


DB = "fakedb"
MARKER = "_cache_fp.txt"


@pytest.fixture(autouse=True)
def _disable_embeddings_by_default(monkeypatch):
    """Pin the embedding channel to "off" for the whole module so the cache
    tests never call the real embedding API and fingerprints don't pick up a
    host-dependent model name. The one test that flips embeddings ON does its
    own monkeypatch on top of this."""
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: False)


@pytest.fixture
def fake_mini_interact_root(tmp_path: Path) -> Path:
    """A throwaway mini-interact layout with one DB folder containing the
    three fingerprint inputs (sqlite / column-meaning / kb jsonl)."""
    root = tmp_path / "mini-interact"
    db_dir = root / DB
    db_dir.mkdir(parents=True)

    (db_dir / f"{DB}.sqlite").write_bytes(b"")
    (db_dir / f"{DB}_column_meaning_base.json").write_text("{}")
    (db_dir / f"{DB}_kb.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "id": 1, "knowledge": "K1", "description": "d",
                    "definition": "f", "type": "x", "children_knowledge": -1,
                }),
                json.dumps({
                    "id": 2, "knowledge": "K2", "description": "d",
                    "definition": "f", "type": "x", "children_knowledge": [1],
                }),
            ]
        ) + "\n"
    )
    return root


@pytest.fixture
def mock_orchestrator(monkeypatch):
    """Replace the orchestrator phase functions on ``otf_cache`` with
    in-memory stubs that record call counts and write a believable output
    layout under the build dir. Returns the call-counter dict."""
    calls = {"phase1": 0, "phase2": 0, "phase3": 0, "phase4": 0}

    def fake_phase1(db, sqlite_path, storage):
        calls["phase1"] += 1
        models_dir = Path(storage) / "models" / db
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "table_a.yaml").write_text("name: table_a\ncolumns: []\n")
        ds_dir = Path(storage) / "datasources"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / f"{db}.yaml").write_text(
            f"name: {db}\ntype: sqlite\nconnection_string: sqlite:///{db}.sqlite\n"
        )

    async def fake_phase2(storage, db, meanings_path):
        calls["phase2"] += 1
        return 0, []

    async def fake_phase3(storage, db, meanings_path, sqlite_path):
        calls["phase3"] += 1
        return 0, [], []

    async def fake_phase4(storage, db, sqlite_path, llm_model):  # pragma: no cover
        calls["phase4"] += 1
        return 0, []

    monkeypatch.setattr(otf_cache, "_phase1_ingest", fake_phase1)
    monkeypatch.setattr(otf_cache, "_phase2_overlay", fake_phase2)
    monkeypatch.setattr(otf_cache, "_phase3_jsonb", fake_phase3)
    monkeypatch.setattr(otf_cache, "_phase4_dates", fake_phase4)
    return calls


# ---------------------------------------------------------------------------
# Happy path — single authoritative dir, no <fp> level
# ---------------------------------------------------------------------------


async def test_first_call_builds_into_db_dir_no_fp_level(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """First call runs phases 1-3 and lands the cache at ``<cache_root>/<db>/``
    directly — NOT under a ``<fingerprint>`` sub-dir."""
    cache_root = tmp_path / "cache"
    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    assert mock_orchestrator["phase1"] == 1
    assert mock_orchestrator["phase2"] == 1
    assert mock_orchestrator["phase3"] == 1

    # The authoritative dir is <cache_root>/<db>/ — its parent is cache_root.
    assert entry.cache_dir == cache_root / DB
    assert entry.cache_dir.parent == cache_root
    assert entry.fingerprint  # non-empty provenance

    # Orchestrator outputs + the cached KB rows live directly under <db>/.
    assert (entry.cache_dir / "datasources" / f"{DB}.yaml").exists()
    assert (entry.cache_dir / "models" / DB / "table_a.yaml").exists()
    assert (entry.cache_dir / "_kb_rows.json").exists()
    assert [r["id"] for r in entry.kb_rows] == [1, 2]


async def test_cache_fp_marker_written_and_matches_fingerprint(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """The completeness marker ``_cache_fp.txt`` is written and carries the
    build-time fingerprint (provenance)."""
    cache_root = tmp_path / "cache"
    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    marker = entry.cache_dir / MARKER
    assert marker.is_file(), "completeness marker must be written"
    assert marker.read_text().strip() == entry.fingerprint


async def test_phase_4_is_never_invoked(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    await otf_cache.ensure_db_cache(
        DB, cache_root=tmp_path / "cache",
        mini_interact_root=fake_mini_interact_root,
    )
    assert mock_orchestrator["phase4"] == 0


async def test_kb_rows_match_parsed_jsonl(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=tmp_path / "cache",
        mini_interact_root=fake_mini_interact_root,
    )
    expected = [
        json.loads(line)
        for line in (fake_mini_interact_root / DB / f"{DB}_kb.jsonl")
        .read_text().splitlines()
        if line.strip()
    ]
    assert entry.kb_rows == expected
    assert json.loads((entry.cache_dir / "_kb_rows.json").read_text()) == expected


# ---------------------------------------------------------------------------
# Presence-gated reuse — the consolidation contract
# ---------------------------------------------------------------------------


async def test_second_call_reuses_without_rebuild(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """A present ``_cache_fp.txt`` marker → reuse: no orchestrator phase
    re-invoked, same cache_dir."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    snapshot = dict(mock_orchestrator)
    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert mock_orchestrator == snapshot, "reuse must not re-invoke any phase"
    assert a.cache_dir == b.cache_dir == cache_root / DB
    assert a.fingerprint == b.fingerprint


async def test_reuse_does_not_call_fingerprint_of(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """The reuse path must NOT compute ``fingerprint_of`` — that's the whole
    point of dropping fingerprint gating (a recompute in-cloud would stat a
    different sqlite mtime + abs root and is exactly what we removed)."""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    def boom(**_kw):
        raise AssertionError("fingerprint_of must not be called on reuse")

    monkeypatch.setattr(otf_cache, "fingerprint_of", boom)
    # Must reuse cleanly without invoking fingerprint_of.
    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert entry.cache_dir == cache_root / DB


async def test_reuse_loads_fingerprint_from_marker_not_recompute(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """On reuse, ``CacheEntry.fingerprint`` comes from the on-disk
    ``_cache_fp.txt`` (provenance), not a recomputation — so the reference
    BUILD that reuses an existing cache still gets a coherent fingerprint
    (Codex r2 Med#4)."""
    cache_root = tmp_path / "cache"
    first = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    # If reuse recomputed, it would return this bogus value; it must not.
    monkeypatch.setattr(otf_cache, "fingerprint_of", lambda **_k: "WRONG_FP")
    reused = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert reused.fingerprint == first.fingerprint != "WRONG_FP"


async def test_force_rebuilds_even_when_marker_present(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """``force=True`` re-runs the orchestrator even when a complete cache
    is present."""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        force=True,
    )
    assert mock_orchestrator["phase1"] == 2
    assert mock_orchestrator["phase2"] == 2
    assert mock_orchestrator["phase3"] == 2


async def test_changed_input_without_force_reuses_stale(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """Accepted tradeoff (user-chosen): editing a KB/schema/DB locally and
    re-running WITHOUT force reuses the stale artifact. Reingest is explicit.
    The fingerprint *would* change, but it no longer gates reuse."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # Mutate a fingerprint input.
    (fake_mini_interact_root / DB / f"{DB}_column_meaning_base.json").write_text(
        '{"some": "new content"}'
    )
    snapshot = dict(mock_orchestrator)
    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert mock_orchestrator == snapshot, "changed input must NOT rebuild w/o force"
    assert a.cache_dir == b.cache_dir
    assert a.fingerprint == b.fingerprint  # stale fp reused from marker


# ---------------------------------------------------------------------------
# Migration: old <db>/<fp>/ layout (or incomplete dir) must not crash
# ---------------------------------------------------------------------------


async def test_markerless_existing_dir_is_wiped_and_rebuilt(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """An old-layout cache (``cache_root/<db>/<fp>/...`` with no
    ``_cache_fp.txt`` directly under ``<db>/``) — or any incomplete dir —
    is rmtree'd and rebuilt, NOT crashed on the rename-onto-existing path."""
    cache_root = tmp_path / "cache"
    old_db_dir = cache_root / DB
    (old_db_dir / "deadbeefcafe1234").mkdir(parents=True)  # old <fp> subdir
    (old_db_dir / "deadbeefcafe1234" / "_kb_rows.json").write_text("[]")
    assert not (old_db_dir / MARKER).exists()  # no db-level marker

    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    assert mock_orchestrator["phase1"] == 1, "must rebuild the markerless dir"
    assert entry.cache_dir == cache_root / DB
    assert (entry.cache_dir / MARKER).is_file()
    # The stale old-layout subdir is gone (the dir was wiped before rename).
    assert not (cache_root / DB / "deadbeefcafe1234").exists()


# ---------------------------------------------------------------------------
# Atomicity + concurrency
# ---------------------------------------------------------------------------


async def test_failed_build_leaves_no_db_dir(
    fake_mini_interact_root: Path, monkeypatch, tmp_path: Path,
):
    """If phase 1 raises mid-build, the target ``<cache_root>/<db>/`` must
    NOT exist — the build is into a tmp sibling, atomic-renamed only on
    success. (tmp dirs may linger; the strict invariant is "no final dir".)"""
    def boom(db, sqlite_path, storage):
        raise RuntimeError("orchestrator died")

    async def fake_phase2(storage, db, meanings_path):  # pragma: no cover
        return 0, []

    async def fake_phase3(storage, db, meanings_path, sqlite_path):  # pragma: no cover
        return 0, [], []

    monkeypatch.setattr(otf_cache, "_phase1_ingest", boom)
    monkeypatch.setattr(otf_cache, "_phase2_overlay", fake_phase2)
    monkeypatch.setattr(otf_cache, "_phase3_jsonb", fake_phase3)

    cache_root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="orchestrator died"):
        await otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        )
    assert not (cache_root / DB).exists()
    # The plan requires the half-built tmp sibling be rmtree'd on failure —
    # not left to accumulate. cache_root may exist (mkdir'd) but must hold no
    # leftover ".<db>.tmp-*" build dir.
    leftover = [p for p in cache_root.glob(".*") if p.is_dir()] if cache_root.exists() else []
    assert leftover == [], f"failed build left tmp dirs behind: {leftover}"


async def test_rename_oserror_without_marker_reraises(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """If os.rename fails with OSError but the target has NO completeness
    marker (i.e. it's a genuine error, not a peer winning the race), the
    error must propagate — we must not silently treat a markerless target
    as success."""
    cache_root = tmp_path / "cache"

    def failing_rename(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(otf_cache.os, "rename", failing_rename)

    with pytest.raises(OSError, match="disk on fire"):
        await otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        )
    assert not (cache_root / DB / MARKER).exists()


async def test_cross_process_race_rename_onto_marked_dir_is_success(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """If a peer process won the rename while we built (target now exists +
    marked), our ``os.rename`` raises OSError; we must treat that as success
    (the peer's content is equivalent) and discard our tmp dir, not re-raise."""
    cache_root = tmp_path / "cache"
    target = cache_root / DB

    real_rename = os.rename

    def racing_rename(src, dst):
        # Simulate a peer that already committed a complete dir at dst.
        if Path(dst) == target and not target.exists():
            target.mkdir(parents=True)
            (target / "_kb_rows.json").write_text("[]")
            (target / MARKER).write_text("peerfp")
            raise OSError("Directory not empty")
        return real_rename(src, dst)

    monkeypatch.setattr(otf_cache.os, "rename", racing_rename)

    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # No exception; returns the (peer's) target dir AND the peer's on-disk
    # metadata — not our local fp/kb_rows — so CacheEntry matches cache_dir
    # (CodeRabbit).
    assert entry.cache_dir == target
    assert (target / MARKER).is_file()
    assert entry.fingerprint == "peerfp"
    assert entry.kb_rows == []


async def test_concurrent_calls_for_same_db_build_once(
    fake_mini_interact_root: Path, monkeypatch, tmp_path: Path,
):
    """Two gathered calls for the same db serialise on the per-DB lock so the
    orchestrator runs once and both see the same cache_dir."""
    cache_root = tmp_path / "cache"
    calls = {"phase1": 0}

    def fake_phase1(db, sqlite_path, storage):
        calls["phase1"] += 1
        import time as _time
        _time.sleep(0.05)
        models_dir = Path(storage) / "models" / db
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "table_a.yaml").write_text("name: table_a\ncolumns: []\n")
        ds_dir = Path(storage) / "datasources"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / f"{db}.yaml").write_text(
            f"name: {db}\ntype: sqlite\nconnection_string: sqlite:///{db}.sqlite\n"
        )

    async def fake_phase2(storage, db, meanings_path):
        return 0, []

    async def fake_phase3(storage, db, meanings_path, sqlite_path):
        return 0, [], []

    monkeypatch.setattr(otf_cache, "_phase1_ingest", fake_phase1)
    monkeypatch.setattr(otf_cache, "_phase2_overlay", fake_phase2)
    monkeypatch.setattr(otf_cache, "_phase3_jsonb", fake_phase3)

    a, b = await asyncio.gather(
        otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        ),
        otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        ),
    )
    assert a.cache_dir == b.cache_dir == cache_root / DB
    assert a.fingerprint == b.fingerprint
    assert calls["phase1"] == 1


# ---------------------------------------------------------------------------
# fingerprint_of — retained as a pure provenance function (no gating)
# ---------------------------------------------------------------------------


def test_fingerprint_of_changes_on_column_meaning(
    fake_mini_interact_root: Path,
):
    a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    (fake_mini_interact_root / DB / f"{DB}_column_meaning_base.json").write_text(
        '{"some": "new content"}'
    )
    b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert a != b


def test_fingerprint_of_changes_on_kb_jsonl(fake_mini_interact_root: Path):
    a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    jsonl = fake_mini_interact_root / DB / f"{DB}_kb.jsonl"
    jsonl.write_text(jsonl.read_text() + json.dumps({
        "id": 3, "knowledge": "K3", "description": "d", "definition": "f",
        "type": "x", "children_knowledge": -1,
    }) + "\n")
    b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert a != b


def test_fingerprint_of_changes_on_sqlite_mtime(fake_mini_interact_root: Path):
    a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    sqlite_path = fake_mini_interact_root / DB / f"{DB}.sqlite"
    old = sqlite_path.stat().st_mtime
    os.utime(sqlite_path, (old + 60, old + 60))
    b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert a != b


def test_fingerprint_of_changes_on_mini_interact_root(
    fake_mini_interact_root: Path, tmp_path: Path,
):
    """Two roots with byte-identical files + identical stat tuples must still
    differ (the stored absolute sqlite path differs)."""
    import shutil as _shutil
    twin = tmp_path / "mini-interact-twin"
    _shutil.copytree(fake_mini_interact_root, twin)
    for p in twin.rglob("*"):
        if p.is_file():
            src = fake_mini_interact_root / p.relative_to(twin)
            st = src.stat()
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    fp_a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    fp_b = otf_cache.fingerprint_of(db_name=DB, mini_interact_root=twin)
    assert fp_a != fp_b


def test_fingerprint_of_changes_on_embedding_model(
    fake_mini_interact_root: Path, monkeypatch,
):
    fp_off = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(otf_cache, "_embedding_current_model", lambda: "embed-v1")
    fp_a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert fp_off != fp_a
    monkeypatch.setattr(otf_cache, "_embedding_current_model", lambda: "embed-v2")
    fp_b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert fp_a != fp_b


def test_fingerprint_of_changes_on_slayer_version(
    fake_mini_interact_root: Path, monkeypatch,
):
    a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    monkeypatch.setattr(otf_cache, "_slayer_version", lambda: "999.0.0-fake")
    b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert a != b


# ---------------------------------------------------------------------------
# Impl-fingerprint split (DEV-1508): dropping `--ingest-on-startup` from the
# OTF MCP launch removes the last defensive refresh path for slayer / embed-
# model drift. The marker-presence reuse contract MUST stay (cloud lifecycle
# would otherwise always-rebuild, since stat-mtime and abs-root differ). The
# narrower fix is a SECOND marker `_impl_fp.txt` that holds only the
# implementation-version components (slayer version + embedding model name)
# — recomputed on reuse, mismatched → rebuild.
# ---------------------------------------------------------------------------


IMPL_MARKER = "_impl_fp.txt"


async def test_impl_fp_marker_written_on_build(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """After a fresh build BOTH markers exist; `_impl_fp.txt` carries the
    impl-only fingerprint (a string)."""
    cache_root = tmp_path / "cache"
    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    impl_marker = entry.cache_dir / IMPL_MARKER
    assert impl_marker.is_file(), "impl-fp marker must be written on build"
    content = impl_marker.read_text().strip()
    assert content, "impl-fp marker must be non-empty"
    # Impl fp is a separate symbol from the full fingerprint and excludes
    # root/stat-mtime/file-content, so it should NOT equal entry.fingerprint
    # (which carries those input components).
    assert content != entry.fingerprint


async def test_impl_fp_helper_excludes_root_and_inputs(
    fake_mini_interact_root: Path, tmp_path: Path, monkeypatch,
):
    """The new ``_impl_fingerprint_of`` helper depends ONLY on the
    implementation-version components (slayer version + embedding model).
    Changing the mini-interact root or any input file MUST NOT change it —
    that's the whole point of the split (cloud reuse with a different abs
    root must keep working)."""
    a = otf_cache._impl_fingerprint_of()
    # Mutate a file that affects the FULL fingerprint but not impl.
    (fake_mini_interact_root / DB / f"{DB}_column_meaning_base.json").write_text(
        '{"some": "new content"}'
    )
    b = otf_cache._impl_fingerprint_of()
    assert a == b
    # And changing the slayer version DOES change it.
    monkeypatch.setattr(otf_cache, "_slayer_version", lambda: "999.0.0-fake")
    c = otf_cache._impl_fingerprint_of()
    assert a != c


async def test_impl_fp_helper_changes_on_embed_model(monkeypatch):
    """Switching the embedding model name (or flipping channel on/off)
    changes the impl fingerprint."""
    off = otf_cache._impl_fingerprint_of()
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(otf_cache, "_embedding_current_model", lambda: "embed-v1")
    a = otf_cache._impl_fingerprint_of()
    assert off != a
    monkeypatch.setattr(otf_cache, "_embedding_current_model", lambda: "embed-v2")
    b = otf_cache._impl_fingerprint_of()
    assert a != b


async def test_reuse_rebuilds_on_slayer_version_mismatch(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """Build cache. Bump slayer version. Reuse must rebuild — otherwise we'd
    silently serve a cache built under a different slayer."""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    snapshot = dict(mock_orchestrator)

    # Simulate a slayer upgrade between cache-build and runtime.
    monkeypatch.setattr(otf_cache, "_slayer_version", lambda: "999.0.0-fake")

    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # Each orchestrator phase ran AGAIN — total = snapshot + 1 per phase.
    assert mock_orchestrator["phase1"] == snapshot["phase1"] + 1
    assert mock_orchestrator["phase2"] == snapshot["phase2"] + 1
    assert mock_orchestrator["phase3"] == snapshot["phase3"] + 1


async def test_reuse_rebuilds_on_embed_model_mismatch(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """Build cache with embeddings OFF (autouse fixture default). Flip
    embeddings ON. Reuse must rebuild."""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    snapshot = dict(mock_orchestrator)

    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model", lambda: "embed-v1-fake",
    )

    # Stub embed_batch so we don't hit any real client.
    async def fake_embed_batch(texts, model):
        return [[0.0] for _ in texts]

    monkeypatch.setattr(otf_cache, "embed_batch", fake_embed_batch)

    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # Symmetric with the slayer-version test: full rebuild, all phases re-run.
    assert mock_orchestrator["phase1"] == snapshot["phase1"] + 1
    assert mock_orchestrator["phase2"] == snapshot["phase2"] + 1
    assert mock_orchestrator["phase3"] == snapshot["phase3"] + 1


async def test_reuse_impl_match_keeps_fast_path(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """Build cache. Re-call with no impl change. The orchestrator phases
    MUST NOT re-run. (The fast path is the whole reason the cache exists.)"""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    snapshot = dict(mock_orchestrator)
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert mock_orchestrator == snapshot, (
        "impl-fp match must keep the fast (no-rebuild) path"
    )


async def test_reuse_does_not_call_full_fingerprint_of_after_split(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """The split must not regress the explicit "no full fingerprint recompute
    on reuse" invariant (the cloud lifecycle would otherwise mismatch on
    sqlite mtime + abs root). Only the impl-only helper is allowed."""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    def boom(**_kw):
        raise AssertionError(
            "full fingerprint_of must not be called on reuse — only _impl_fingerprint_of"
        )

    monkeypatch.setattr(otf_cache, "fingerprint_of", boom)
    # Must reuse cleanly.
    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert entry.cache_dir == cache_root / DB
