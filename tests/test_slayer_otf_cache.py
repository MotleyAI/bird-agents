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

    def fake_phase1(db, storage, *, sqlite_path=None, db_url=None, pg_password=None):
        calls["phase1"] += 1
        models_dir = Path(storage) / "models" / db
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "table_a.yaml").write_text("name: table_a\ncolumns: []\n")
        ds_dir = Path(storage) / "datasources"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / f"{db}.yaml").write_text(
            f"name: {db}\ntype: sqlite\nconnection_string: sqlite:///{db}.sqlite\n"
        )

    async def fake_phase2(storage, db, meanings_path, *, backend="sqlite", pg_sampler=None):
        calls["phase2"] += 1
        return 0, []

    async def fake_phase3(storage, db, meanings_path=None, sqlite_path=None, *, benchmark=None, backend=None, pg_extract_sampler=None):
        calls["phase3"] += 1
        return 0, [], []

    async def fake_phase4(storage, db, sqlite_path=None, llm_model=None, *, benchmark=None):  # pragma: no cover
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
    def boom(db, storage, *, sqlite_path=None, db_url=None):
        raise RuntimeError("orchestrator died")

    async def fake_phase2(storage, db, meanings_path, *, backend="sqlite", pg_sampler=None):  # pragma: no cover
        return 0, []

    async def fake_phase3(storage, db, meanings_path=None, sqlite_path=None, *, benchmark=None, backend=None, pg_extract_sampler=None):  # pragma: no cover
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
    marked + impl-fp matches), our ``os.rename`` raises OSError; we must treat
    that as success (the peer's content is equivalent) and discard our tmp dir,
    not re-raise. DEV-1508 tightens the contract: the peer's `_impl_fp.txt`
    must ALSO match ours — otherwise we'd be accepting a peer that built
    under a stale impl, reintroducing the bug class Codex flagged."""
    cache_root = tmp_path / "cache"
    target = cache_root / DB

    real_rename = os.rename
    current_impl = otf_cache._impl_fingerprint_of()

    def racing_rename(src, dst):
        # Simulate a peer that already committed a complete dir at dst —
        # WITH a matching impl marker (the "compatible peer" case).
        if Path(dst) == target and not target.exists():
            target.mkdir(parents=True)
            (target / "_kb_rows.json").write_text("[]")
            (target / IMPL_MARKER).write_text(current_impl)
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


async def test_cross_process_race_peer_with_mismatched_impl_is_rejected(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """If a peer won the rename but built under a DIFFERENT impl
    fingerprint (e.g. older slayer version that lacked a feature), we must
    NOT silently accept their stale cache — that's the DEV-1508 bug class.
    Re-raise the OSError so the caller knows the rebuild failed."""
    cache_root = tmp_path / "cache"
    target = cache_root / DB

    real_rename = os.rename

    def racing_rename(src, dst):
        if Path(dst) == target and not target.exists():
            target.mkdir(parents=True)
            (target / "_kb_rows.json").write_text("[]")
            # Peer's impl marker DIFFERS from ours.
            (target / IMPL_MARKER).write_text("STALE_PEER_IMPL")
            (target / MARKER).write_text("peerfp")
            raise OSError("Directory not empty")
        return real_rename(src, dst)

    monkeypatch.setattr(otf_cache.os, "rename", racing_rename)

    with pytest.raises(OSError):
        await otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        )


async def test_concurrent_calls_for_same_db_build_once(
    fake_mini_interact_root: Path, monkeypatch, tmp_path: Path,
):
    """Two gathered calls for the same db serialise on the per-DB lock so the
    orchestrator runs once and both see the same cache_dir."""
    cache_root = tmp_path / "cache"
    calls = {"phase1": 0}

    def fake_phase1(db, storage, *, sqlite_path=None, db_url=None, pg_password=None):
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

    async def fake_phase2(storage, db, meanings_path, *, backend="sqlite", pg_sampler=None):
        return 0, []

    async def fake_phase3(storage, db, meanings_path=None, sqlite_path=None, *, benchmark=None, backend=None, pg_extract_sampler=None):
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


def test_impl_fp_invariant_across_postgres_connection(monkeypatch):
    """DEV-1685 B2: the postgres connection (host/port/user) is RUNTIME-supplied
    and reanchored per task, so it must NOT feed the impl fingerprint. A cache
    built on one port must be reused verbatim on another — no `pg_conn` block.
    Guards against the auto-port change thrashing the OTF cache."""
    class _PgBench:
        db_backend = "postgres"

    monkeypatch.setenv("BIRD_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("BIRD_PG_PORT", "5544")
    monkeypatch.setenv("BIRD_PG_USER", "bird_interact")
    a = otf_cache._impl_fingerprint_of(_PgBench())

    monkeypatch.setenv("BIRD_PG_HOST", "otherhost")
    monkeypatch.setenv("BIRD_PG_PORT", "5433")
    monkeypatch.setenv("BIRD_PG_USER", "someone_else")
    b = otf_cache._impl_fingerprint_of(_PgBench())
    assert a == b

    # And a postgres benchmark now fingerprints identically to a non-postgres
    # one (given the same slayer/embed state) — the connection is gone entirely.
    assert otf_cache._impl_fingerprint_of(_PgBench()) == otf_cache._impl_fingerprint_of(None)


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


async def test_impl_mismatch_actually_replaces_target_on_disk(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
    monkeypatch,
):
    """Codex review: when the impl fingerprint drifts, the rebuild must
    actually REPLACE the on-disk target — incrementing phase counters is
    not enough. Without this assertion a prior version of the rebuild path
    appeared to "work" (phase counts went up) while ``os.rename`` silently
    failed because the marked target still existed, and the OSError handler
    returned the stale target. Pin both observable effects:
    (a) ``_impl_fp.txt`` on disk reflects the NEW impl,
    (b) the returned ``cache_dir`` is the target (not a stranded tmp dir)."""
    cache_root = tmp_path / "cache"
    first = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    impl_marker = first.cache_dir / IMPL_MARKER
    old_impl = impl_marker.read_text().strip()

    monkeypatch.setattr(otf_cache, "_slayer_version", lambda: "999.0.0-fake")
    new_expected_impl = otf_cache._impl_fingerprint_of()
    assert new_expected_impl != old_impl  # sanity

    second = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # Target replaced; the on-disk impl marker reflects the new build.
    assert second.cache_dir == cache_root / DB
    assert impl_marker.read_text().strip() == new_expected_impl


async def test_legacy_cache_without_impl_marker_rebuilds(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """A cache built BEFORE the impl-fp split landed has ``_cache_fp.txt``
    but no ``_impl_fp.txt``. On reuse, that must trigger a rebuild (the
    safer-than-assume-compat semantics) — and the rebuild must actually
    replace the legacy target on disk."""
    cache_root = tmp_path / "cache"
    await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # Simulate a pre-split cache by deleting the impl marker.
    impl_marker = cache_root / DB / IMPL_MARKER
    assert impl_marker.is_file()
    impl_marker.unlink()
    snapshot = dict(mock_orchestrator)

    entry = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    # Orchestrator re-ran AND the impl marker is back, populated with the
    # current impl fingerprint.
    assert mock_orchestrator["phase1"] == snapshot["phase1"] + 1
    assert entry.cache_dir == cache_root / DB
    assert impl_marker.is_file()
    assert impl_marker.read_text().strip() == otf_cache._impl_fingerprint_of()


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


# ---------------------------------------------------------------------------
# DEV-1557 / Stage-2: cache builder delegates embedding-text truncation to
# SLayer 0.7.4+. Migrated from the deleted test_slayer_otf_cache_embed_truncate
# file with a delegation-style test replacing the per-helper unit tests.
# ---------------------------------------------------------------------------


def test_embedding_builder_version_in_cache_fingerprint(monkeypatch):
    """Bumping ``_EMBEDDING_BUILDER_VERSION`` MUST change
    ``_impl_fingerprint_of(...)`` so already-built caches invalidate
    automatically when the embedding-text pipeline changes (e.g. we
    delegated truncation to SLayer in version 3; a future change to the
    pipeline lifts to version 4 and the bump alone forces a rebuild).
    Migrated from test_slayer_otf_cache_embed_truncate.py before that
    file was deleted."""
    fp_before = otf_cache._impl_fingerprint_of(None)
    monkeypatch.setattr(otf_cache, "_EMBEDDING_BUILDER_VERSION", 999)
    fp_after = otf_cache._impl_fingerprint_of(None)
    assert fp_before != fp_after


def test_embedding_builder_version_is_3_after_stage_2_delegation():
    """Codex /spec review (f): assert the literal version value so a
    partial implementation that bumps everything else but forgets the
    constant gets caught. The migration contract is encoded here."""
    assert otf_cache._EMBEDDING_BUILDER_VERSION == 3


def test_bird_side_truncation_helpers_were_deleted():
    """Codex /spec review (a) / (f): direct absence assertions on the
    symbols the Stage-2 migration deleted. No finite-input
    "passes-through-50k-chars" test can prove "no truncation ever";
    only the absence of the helper API gives us that guarantee."""
    assert not hasattr(otf_cache, "_truncate_for_embedding"), (
        "_truncate_for_embedding must be deleted — SLayer 0.7.4+ is the "
        "single source of truth for per-text token truncation"
    )
    assert not hasattr(otf_cache, "_EMBEDDING_MAX_TOKENS"), (
        "_EMBEDDING_MAX_TOKENS must be deleted with the helper"
    )
    assert not hasattr(otf_cache, "_EMBEDDING_FALLBACK_MAX_CHARS"), (
        "_EMBEDDING_FALLBACK_MAX_CHARS must be deleted with the helper"
    )


def test_materialise_cache_memories_passes_raw_text_to_embed_batch(
    monkeypatch, tmp_path: Path,
):
    """DEV-1557 / Stage-2: bird-agents no longer pre-truncates the
    rendered memory text before ``embed_batch``. SLayer 0.7.4's own
    ``embed_batch`` handles per-text token truncation + per-input retry.

    Verify by: monkeypatching ``embed_batch`` to capture inputs;
    forcing one of the rendered memories to be 50k chars long (way
    over any plausible cap); asserting the EXACT raw text is what
    ``embed_batch`` receives. If a future refactor reintroduces
    bird-side truncation, this test fails immediately with a
    length mismatch.

    Replaces the prior per-helper unit tests, which were tightly
    coupled to the deleted ``_truncate_for_embedding`` and would
    otherwise drift into testing SLayer's private helpers."""
    seen_inputs: list[str] = []

    async def stub_embed_batch(texts, *, model=None):
        seen_inputs.extend(texts)
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model",
        lambda: "openai/text-embedding-3-small",
    )

    long_raw = "x " * 25_000  # ~50k chars; well above any prior bird-side cap
    short_raw = "short text"

    def fake_render(*, memory):
        return long_raw if memory.id == "long" else short_raw

    monkeypatch.setattr(otf_cache, "render_memory_text_for_embedding", fake_render)

    # DEV-1668: real Memory dicts so the per-id persist step can serialize
    # them; the embedding path keys on ``memory.id`` (via fake_render).
    monkeypatch.setattr(
        otf_cache, "encode_kb_as_memories",
        lambda *a, **kw: [
            {"id": "long", "learning": "kb long", "entities": []},
            {"id": "short", "learning": "kb short", "entities": []},
        ],
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    asyncio.run(otf_cache._materialise_cache_memories(
        db="alien", build_dir=build_dir, kb_rows=[{"id": 1}, {"id": 2}],
    ))

    # Both texts arrived at embed_batch.
    assert len(seen_inputs) == 2
    # The long memory's text arrives VERBATIM — no bird-side truncation.
    # If somebody reintroduces a `_truncate_for_embedding` step, the
    # length comparison fails immediately.
    assert long_raw in seen_inputs, (
        "long memory text was not passed through to embed_batch verbatim — "
        "did somebody re-add bird-side truncation? Slayer 0.7.4+ is the "
        "single source of truth for per-text token truncation."
    )
    assert short_raw in seen_inputs


def test_materialise_cache_memories_logs_per_memory_observability(
    monkeypatch, tmp_path: Path, caplog,
):
    """DEV-1557 / Stage-2: keep per-memory observability after deleting
    the truncation helper. SLayer 0.7.4 logs truncation events with a
    sha256_prefix but doesn't know our memory id / db; emit an INFO log
    near the embed_batch call mapping ``memory.id``, ``db``,
    ``len(text)``, and a sha256 prefix so the two log streams can be
    correlated."""
    import logging
    async def stub_embed_batch(texts, *, model=None):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model",
        lambda: "openai/text-embedding-3-small",
    )
    monkeypatch.setattr(
        otf_cache, "render_memory_text_for_embedding",
        lambda *, memory: f"text for {memory.id}",
    )

    # DEV-1668: real Memory dicts so the per-id persist step can serialize them.
    monkeypatch.setattr(
        otf_cache, "encode_kb_as_memories",
        lambda *a, **kw: [
            {"id": "alpha", "learning": "kb alpha", "entities": []},
            {"id": "beta", "learning": "kb beta", "entities": []},
        ],
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    with caplog.at_level(logging.INFO, logger="bird_interact_agents.slayer_otf.cache"):
        asyncio.run(otf_cache._materialise_cache_memories(
            db="alien_db", build_dir=build_dir,
            kb_rows=[{"id": 1}, {"id": 2}],
        ))

    import hashlib

    records = [r for r in caplog.records
               if "[slayer_otf]" in r.message
               and "embedding" in r.message.lower()
               and ("alpha" in r.message or "beta" in r.message)]
    assert len(records) == 2, (
        "expected one INFO log per memory mapping memory.id + db + chars + sha256 prefix; "
        f"got {len(records)} matching records"
    )
    # Codex /spec review (b): assert level explicitly. ``caplog.at_level(INFO)``
    # still captures warnings, so without this a regression to a
    # warning-shaped log would slip through.
    for r in records:
        assert r.levelno == logging.INFO, (
            f"per-memory observability log must be INFO, got level={r.levelname}"
        )
    # Compute the digest WE expect (from the rendered text) and verify
    # the logged sha256 prefix is at least 8 hex chars of that digest —
    # ties the log line to the right text without coupling to
    # slayer's private formatting.
    expected_digests = {
        "alpha": hashlib.sha256(b"text for alpha").hexdigest(),
        "beta": hashlib.sha256(b"text for beta").hexdigest(),
    }
    for r in records:
        if "alpha" in r.message:
            mid = "alpha"
        elif "beta" in r.message:
            mid = "beta"
        else:
            continue
        assert "alien_db" in r.message, f"db not carried: {r.message}"
        assert "chars=" in r.message or "len=" in r.message, (
            f"length not carried: {r.message}"
        )
        # Look for a non-trivial prefix of the expected digest in the
        # message — at least 8 hex chars proves we hashed THIS text.
        digest = expected_digests[mid]
        # Try a few common prefix lengths; the impl picks the budget.
        assert any(digest[:k] in r.message for k in (8, 12, 16, 24, 32)), (
            f"expected a sha256 prefix of {digest!r} in {r.message!r}"
        )


def test_materialise_cache_memories_partial_batch_persists_good_skips_failed(
    monkeypatch, tmp_path: Path, caplog,
):
    """Codex /spec review (d): bird-side resilience contract — when
    `embed_batch` returns `[vec, None]` (good + failed), the cache
    builder must persist the good embedding row and skip / warn for the
    failed one. This stays independent of SLayer's internal retry
    mechanics (we don't assert on BadRequestError flows); we just
    contract on the per-input None we receive."""
    import logging
    persisted: list = []

    async def stub_embed_batch(texts, *, model=None):
        # First memory got a vector; second came back None (slayer
        # exhausted its per-input retry or the input was unrecoverable).
        return [[0.1] * 8, None]

    async def stub_save_embeddings(self, rows):
        # YAMLStorage.save_embeddings is an async method on the storage
        # object (self + rows); just record what bird tries to persist.
        persisted.extend(rows)

    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model", lambda: "openai/test-model",
    )
    monkeypatch.setattr(
        otf_cache, "render_memory_text_for_embedding",
        lambda *, memory: f"text for {memory.id}",
    )

    # DEV-1668: real Memory dicts so the per-id persist step can serialize them.
    monkeypatch.setattr(
        otf_cache, "encode_kb_as_memories",
        lambda *a, **kw: [
            {"id": "good_mem", "learning": "kb good", "entities": []},
            {"id": "bad_mem", "learning": "kb bad", "entities": []},
        ],
    )

    # Patch YAMLStorage.save_embeddings to capture, NOT write to disk.
    monkeypatch.setattr(
        otf_cache.YAMLStorage, "save_embeddings",
        stub_save_embeddings,
        raising=False,
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    with caplog.at_level(logging.WARNING,
                        logger="bird_interact_agents.slayer_otf.cache"):
        asyncio.run(otf_cache._materialise_cache_memories(
            db="alien", build_dir=build_dir, kb_rows=[{"id": 1}, {"id": 2}],
        ))

    # Exactly one row persisted — the good one.
    assert len(persisted) == 1, (
        f"expected one persisted embedding (the good memory); "
        f"got {len(persisted)}"
    )
    persisted_id = persisted[0].canonical_id
    assert "good_mem" in persisted_id, (
        f"expected `good_mem` in canonical_id, got {persisted_id!r}"
    )
    # The failed memory's id surfaces in a warning.
    fail_warns = [r for r in caplog.records
                  if r.levelno >= logging.WARNING and "bad_mem" in r.message]
    assert fail_warns, (
        "expected a WARNING naming the failed memory id so the operator "
        "can find which memory slayer couldn't embed"
    )
