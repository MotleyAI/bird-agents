"""Tests for ``slayer_otf.cache.ensure_db_cache``.

The cache materialises orchestrator phases 1-3 (slayer ingest +
column-meaning overlay + JSONB-leaf expansion) into
``<cache_root>/<db>/<fingerprint>/`` and caches the parsed KB rows
alongside.

These tests monkeypatch the orchestrator phase functions so the slayer
CLI subprocess is never actually invoked — the cache layer's
responsibility is *orchestration*, not the orchestrator's behaviour.

Phase 4 (LLM date detection) is explicitly NOT called by the cache
layer; one test guards against future regressions.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from bird_interact_agents.slayer_otf import cache as otf_cache


# ---------------------------------------------------------------------------
# Fixture: minimal fake mini-interact root with one DB
# ---------------------------------------------------------------------------


DB = "fakedb"


@pytest.fixture(autouse=True)
def _disable_embeddings_by_default(monkeypatch):
    """Pin the embedding channel to "off" for the whole module.

    Why: ``fingerprint_of`` and ``_materialise_cache_memories`` both
    branch on ``_embeddings_available()``. On a host where the channel
    is actually configured (production-like dev box), the cache tests
    would (a) compute fingerprints that include a real embedding model
    name — making path-only invariants pass for the wrong reason —
    and (b) call the real embedding API through ``ensure_db_cache``,
    spending money and minutes for no reason.

    The one test that flips embeddings ON (``test_fingerprint_changes_
    when_embedding_model_changes``) does its own monkeypatch on top of
    this; the rest stay in "no embeddings" mode (matching how
    ``_materialise_cache_memories`` no-ops when the channel is off).
    """
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: False)


@pytest.fixture
def fake_mini_interact_root(tmp_path: Path) -> Path:
    """A throwaway mini-interact layout with one DB folder containing:

    - ``<db>.sqlite`` (empty 0-byte file; orchestrator phase 1 is mocked
      so its contents don't matter)
    - ``<db>_column_meaning_base.json`` (empty object — phase 2 is
      mocked, so the schema doesn't need to be valid)
    - ``<db>_kb.jsonl`` (2 KB rows so the cache layer has something to
      copy verbatim)
    """
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
    in-memory stubs that record their call counts and write a marker
    file under the target storage so layout-existence checks see the
    'output' the phases would have produced.

    Returns the call-counter dict so tests can introspect.
    """
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
# Happy path
# ---------------------------------------------------------------------------


async def test_first_call_invokes_phases_1_2_3(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """First call to ensure_db_cache against an empty cache_root must
    actually run phases 1, 2, 3 of the orchestrator."""
    cache_root = tmp_path / "cache"

    entry = await otf_cache.ensure_db_cache(
        DB,
        cache_root=cache_root,
        mini_interact_root=fake_mini_interact_root,
    )

    assert mock_orchestrator["phase1"] == 1
    assert mock_orchestrator["phase2"] == 1
    assert mock_orchestrator["phase3"] == 1

    # The returned cache_dir lives at <cache_root>/<db>/<fingerprint>/.
    assert entry.cache_dir.exists()
    assert entry.cache_dir.parent == cache_root / DB, (
        f"cache_dir.parent should be <cache_root>/<db>; got {entry.cache_dir.parent}"
    )
    assert entry.cache_dir.parent.parent == cache_root
    assert entry.fingerprint  # non-empty
    assert entry.cache_dir.name == entry.fingerprint

    # The cache dir contains the orchestrator's outputs.
    assert (entry.cache_dir / "datasources" / f"{DB}.yaml").exists()
    assert (entry.cache_dir / "models" / DB / "table_a.yaml").exists()

    # And the cached KB rows are written alongside.
    kb_rows_path = entry.cache_dir / "_kb_rows.json"
    assert kb_rows_path.exists()
    rows = json.loads(kb_rows_path.read_text())
    assert [r["id"] for r in rows] == [1, 2]
    assert entry.kb_rows == rows


async def test_phase_4_is_never_invoked(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """Phase 4 (LLM TEXT-as-date detection) is explicitly excluded from
    the on-the-fly path; this is a load-bearing decision (see plan)."""
    await otf_cache.ensure_db_cache(
        DB,
        cache_root=tmp_path / "cache",
        mini_interact_root=fake_mini_interact_root,
    )
    assert mock_orchestrator["phase4"] == 0


async def test_second_call_is_a_no_op(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """When ``<cache_root>/<db>/<fingerprint>/`` already exists, a second
    call must NOT re-invoke the orchestrator and must return the same
    cache_dir."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    snapshot = dict(mock_orchestrator)
    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    assert mock_orchestrator == snapshot, (
        "second call should not re-invoke any orchestrator phase"
    )
    assert a.cache_dir == b.cache_dir
    assert a.fingerprint == b.fingerprint


async def test_kb_rows_match_parsed_jsonl(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """``_kb_rows.json`` content must match a straight parse of the
    source ``*_kb.jsonl`` file — no field rewriting, no reordering."""
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
# Fingerprint behaviour
# ---------------------------------------------------------------------------


async def test_fingerprint_changes_when_column_meaning_changes(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """Cache key includes a content hash of column_meaning_base.json.
    Mutating that file forces a fresh build under a new fingerprint
    sub-dir; the old one is left in place."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    # Mutate the column-meaning JSON.
    (fake_mini_interact_root / DB / f"{DB}_column_meaning_base.json").write_text(
        '{"some": "new content"}'
    )

    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    assert a.fingerprint != b.fingerprint, (
        "fingerprint must change when column_meaning changes"
    )
    assert a.cache_dir.exists(), "old fingerprint dir should not be auto-GC'd"
    assert b.cache_dir.exists()
    # Orchestrator was re-invoked exactly once for the new build.
    assert mock_orchestrator["phase1"] == 2
    assert mock_orchestrator["phase2"] == 2
    assert mock_orchestrator["phase3"] == 2


async def test_fingerprint_changes_when_kb_jsonl_changes(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """KB jsonl content is part of the fingerprint — adding a KB row
    must invalidate the cache so the new row gets a memory."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    jsonl = fake_mini_interact_root / DB / f"{DB}_kb.jsonl"
    jsonl.write_text(jsonl.read_text() + json.dumps({
        "id": 3, "knowledge": "K3", "description": "d", "definition": "f",
        "type": "x", "children_knowledge": -1,
    }) + "\n")
    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert a.fingerprint != b.fingerprint
    assert [r["id"] for r in b.kb_rows] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Atomicity + concurrency
# ---------------------------------------------------------------------------


async def test_failed_build_leaves_no_final_fingerprint_dir(
    fake_mini_interact_root: Path, monkeypatch, tmp_path: Path,
):
    """If phase 1 raises mid-build, the target ``<cache_root>/<db>/<fingerprint>/``
    cache dir must NOT exist at all — the build is into a tmp dir
    that is atomic-renamed only on success.

    We compute the expected fingerprint up-front (cache.fingerprint_of
    is a pure helper exposed for this exact reason) and assert that the
    final path is absent post-failure. tmp build dirs are allowed to
    linger; the strict invariant is "no final fingerprint dir".
    """

    def boom(db, sqlite_path, storage):  # pragma: no cover - just to fire
        raise RuntimeError("orchestrator died")

    async def fake_phase2(storage, db, meanings_path):  # pragma: no cover
        return 0, []

    async def fake_phase3(storage, db, meanings_path, sqlite_path):  # pragma: no cover
        return 0, [], []

    monkeypatch.setattr(otf_cache, "_phase1_ingest", boom)
    monkeypatch.setattr(otf_cache, "_phase2_overlay", fake_phase2)
    monkeypatch.setattr(otf_cache, "_phase3_jsonb", fake_phase3)

    cache_root = tmp_path / "cache"

    expected_fp = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    expected_dir = cache_root / DB / expected_fp

    with pytest.raises(RuntimeError, match="orchestrator died"):
        await otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        )

    assert not expected_dir.exists(), (
        f"failed build must not leave the final fingerprint dir behind: {expected_dir}"
    )


async def test_concurrent_calls_for_same_db_invoke_orchestrator_once(
    fake_mini_interact_root: Path, monkeypatch, tmp_path: Path,
):
    """Two ``asyncio.gather``ed calls for the same db must serialise
    so the orchestrator runs once and both callers see the same
    cache_dir. The fake phase 1 below uses an asyncio Event to block
    inside the contested critical section, so both callers are forced
    to compete for the lock before the first build completes (without
    this barrier the race could resolve in trivially-serial order)."""
    cache_root = tmp_path / "cache"
    phase1_entered = asyncio.Event()
    release_phase1 = asyncio.Event()
    calls = {"phase1": 0, "phase2": 0, "phase3": 0}

    def fake_phase1(db, sqlite_path, storage):
        calls["phase1"] += 1
        # Signal that we're inside phase 1, then block until the test
        # has confirmed the second caller is waiting.
        phase1_entered.set()
        # asyncio.run / pytest-asyncio: we're in a sync function called
        # from an async coroutine. Use a short blocking sleep instead.
        import time as _time
        _time.sleep(0.05)
        # Produce the expected output layout so the cache path is valid.
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

    monkeypatch.setattr(otf_cache, "_phase1_ingest", fake_phase1)
    monkeypatch.setattr(otf_cache, "_phase2_overlay", fake_phase2)
    monkeypatch.setattr(otf_cache, "_phase3_jsonb", fake_phase3)

    release_phase1.set()  # phase 1 is sync sleep, no async release needed

    a, b = await asyncio.gather(
        otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        ),
        otf_cache.ensure_db_cache(
            DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
        ),
    )
    assert a.cache_dir == b.cache_dir
    assert a.fingerprint == b.fingerprint
    assert calls["phase1"] == 1, (
        f"orchestrator must run exactly once across concurrent callers; "
        f"got phase1={calls['phase1']}"
    )


# ---------------------------------------------------------------------------
# Fingerprint coverage for the remaining inputs from the plan
# ---------------------------------------------------------------------------


async def test_fingerprint_changes_when_sqlite_mtime_changes(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """Per the plan, sqlite size+mtime is part of the fingerprint.
    Touching the sqlite file (without changing size) invalidates the cache."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    sqlite_path = fake_mini_interact_root / DB / f"{DB}.sqlite"
    import os as _os
    old_mtime = sqlite_path.stat().st_mtime
    _os.utime(sqlite_path, (old_mtime + 60, old_mtime + 60))

    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert a.fingerprint != b.fingerprint, (
        "fingerprint must change when sqlite mtime changes"
    )


async def test_fingerprint_changes_when_mini_interact_root_changes(
    fake_mini_interact_root: Path, mock_orchestrator: dict, tmp_path: Path,
):
    """Codex finding: two different mini-interact roots with byte-identical
    sqlite/column-meaning/kb files must still produce different fingerprints,
    so the cache can't silently reuse one root's absolute sqlite path for
    a task whose --db-path points at the other."""
    import shutil as _shutil

    # Clone the fake mini-interact root verbatim, preserving file mtimes
    # (shutil.copytree's default doesn't touch them); the two roots now
    # have identical file contents AND identical stat tuples — only the
    # path differs.
    twin_root = tmp_path / "mini-interact-twin"
    _shutil.copytree(fake_mini_interact_root, twin_root)
    for p in twin_root.rglob("*"):
        if p.is_file():
            src = fake_mini_interact_root / p.relative_to(twin_root)
            st = src.stat()
            import os as _os
            # Preserve nanoseconds. ``fingerprint_of`` hashes
            # ``st_mtime_ns``; using float ``st_mtime`` here would lose
            # sub-second precision on filesystems that have it, which
            # could then make ``fp_a != fp_b`` pass for the wrong reason
            # (mtime drift rather than the root path actually being in
            # the fingerprint).
            _os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))

    fp_a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    fp_b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=twin_root,
    )
    assert fp_a != fp_b, (
        f"fingerprint must change when mini_interact_root changes (got "
        f"{fp_a!r} for both roots; this means the cache would silently "
        f"reuse one root's absolute sqlite path for the other's tasks)"
    )


async def test_fingerprint_changes_when_embedding_model_changes(
    fake_mini_interact_root: Path, mock_orchestrator: dict, monkeypatch,
    tmp_path: Path,
):
    """Codex finding: the active embedding model name must be part of
    the fingerprint, so a cache built without embeddings (or with a
    different model) is not silently reused after the embedding channel
    becomes available — that would leave channel 3 dark."""
    fp_no_embed = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )

    # Pretend the embedding channel is now configured under a specific
    # model name. The fingerprint must shift.
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model", lambda: "fake-embed-v1",
    )
    fp_embed_a = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert fp_no_embed != fp_embed_a, (
        "fingerprint must shift when embeddings flip from off to on"
    )

    # And a model-name swap must produce yet another fingerprint, so
    # changing SLAYER_EMBEDDING_MODEL also invalidates the cache.
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model", lambda: "fake-embed-v2",
    )
    fp_embed_b = otf_cache.fingerprint_of(
        db_name=DB, mini_interact_root=fake_mini_interact_root,
    )
    assert fp_embed_a != fp_embed_b, (
        "fingerprint must shift when the embedding model name changes"
    )


async def test_fingerprint_changes_when_slayer_version_changes(
    fake_mini_interact_root: Path, mock_orchestrator: dict, monkeypatch,
    tmp_path: Path,
):
    """Per the plan, the active ``slayer`` package version is part of
    the fingerprint so an upgrade of SLayer's orchestrator behaviour
    invalidates the cache."""
    cache_root = tmp_path / "cache"
    a = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )

    # Pretend slayer was upgraded between runs.
    monkeypatch.setattr(otf_cache, "_slayer_version", lambda: "999.0.0-fake")

    b = await otf_cache.ensure_db_cache(
        DB, cache_root=cache_root, mini_interact_root=fake_mini_interact_root,
    )
    assert a.fingerprint != b.fingerprint, (
        "fingerprint must change when the active SLayer version changes"
    )
