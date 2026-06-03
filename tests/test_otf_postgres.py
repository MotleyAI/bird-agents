"""Tests for OTF (on-the-fly) SLayer setup with postgres-backed benchmarks.

Verifies:
- fingerprint_of for postgres uses schema-text hash, not sqlite file stat.
- ensure_db_cache for postgres does NOT require <db>.sqlite to exist.
- _phase1_ingest accepts db_url and uses it instead of sqlite_path.
- Phases 3/4 are skipped for postgres (native types make them a no-op).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from bird_interact_agents.benchmark import get_benchmark


# ---------------------------------------------------------------------------
# fingerprint_of — postgres branch
# ---------------------------------------------------------------------------


def test_fingerprint_of_postgres_does_not_require_sqlite_file(tmp_path):
    """For postgres benchmarks, fingerprint_of must not require or read
    a <db>.sqlite file — it fingerprints schema text + KB + column meanings."""
    from bird_interact_agents.slayer_otf.cache import fingerprint_of

    b = get_benchmark("livesqlbench_postgres")
    db_name = "alien"
    data_root = tmp_path / "livesqlbench-base-lite-postgres"
    db_dir = data_root / db_name
    db_dir.mkdir(parents=True)

    # Write schema text and KB — no .sqlite file
    (db_dir / f"{db_name}_schema.txt").write_text("CREATE TABLE t (id INTEGER)")
    (db_dir / f"{db_name}_kb.jsonl").write_text("")
    (db_dir / f"{db_name}_column_meaning_base.json").write_text("{}")

    # Should not raise even though there is no .sqlite file
    fp = fingerprint_of(db_name=db_name, data_root=data_root, benchmark=b)
    assert isinstance(fp, str) and len(fp) > 0


def test_fingerprint_of_postgres_changes_when_schema_changes(tmp_path):
    """Changing schema text must change the fingerprint."""
    from bird_interact_agents.slayer_otf.cache import fingerprint_of

    b = get_benchmark("livesqlbench_postgres")
    db_name = "alien"
    data_root = tmp_path / "livesqlbench-base-lite-postgres"
    db_dir = data_root / db_name
    db_dir.mkdir(parents=True)
    (db_dir / f"{db_name}_kb.jsonl").write_text("")
    (db_dir / f"{db_name}_column_meaning_base.json").write_text("{}")

    (db_dir / f"{db_name}_schema.txt").write_text("CREATE TABLE t (id INTEGER)")
    fp1 = fingerprint_of(db_name=db_name, data_root=data_root, benchmark=b)

    (db_dir / f"{db_name}_schema.txt").write_text("CREATE TABLE t (id INTEGER, name TEXT)")
    fp2 = fingerprint_of(db_name=db_name, data_root=data_root, benchmark=b)

    assert fp1 != fp2


def test_fingerprint_of_sqlite_backward_compat(tmp_path):
    """fingerprint_of for sqlite benchmark still uses the sqlite file stat."""
    from bird_interact_agents.slayer_otf.cache import fingerprint_of
    import sqlite3

    b = get_benchmark("mini_interact")
    db_name = "alien"
    mini_root = tmp_path / "mini-interact"
    db_dir = mini_root / db_name
    db_dir.mkdir(parents=True)

    sqlite_path = db_dir / f"{db_name}.sqlite"
    con = sqlite3.connect(str(sqlite_path))
    con.execute("CREATE TABLE t (id INTEGER)")
    con.commit()
    con.close()
    (db_dir / f"{db_name}_kb.jsonl").write_text("")
    (db_dir / f"{db_name}_column_meaning_base.json").write_text("{}")

    # Should use the standard signature (backward compat: data_root = mini_interact_root)
    fp = fingerprint_of(db_name=db_name, data_root=mini_root, benchmark=b)
    assert isinstance(fp, str) and len(fp) > 0


# ---------------------------------------------------------------------------
# _phase1_ingest — db_url parameter
# ---------------------------------------------------------------------------


def test_phase1_ingest_accepts_db_url(tmp_path):
    """_phase1_ingest must accept a db_url parameter and pass it to slayer ingest
    instead of constructing a sqlite URL from sqlite_path."""
    from bird_interact_agents.slayer_pipeline.orchestrator import _phase1_ingest

    storage = tmp_path / "storage"
    storage.mkdir()
    db_url = "postgresql://bird_interact:bird_interact@localhost:5432/alien"

    calls: list = []

    def fake_slayer_ingest(conn_str, storage_path, **kw):
        calls.append(conn_str)

    with patch("bird_interact_agents.slayer_pipeline.orchestrator._slayer_ingest", side_effect=fake_slayer_ingest):
        _phase1_ingest("alien", storage, db_url=db_url)

    assert len(calls) == 1
    assert calls[0] == db_url


def test_phase1_ingest_sqlite_path_still_works(tmp_path):
    """Existing sqlite_path callers still work after adding db_url parameter."""
    from bird_interact_agents.slayer_pipeline.orchestrator import _phase1_ingest
    import sqlite3

    sqlite_path = tmp_path / "alien.sqlite"
    con = sqlite3.connect(str(sqlite_path))
    con.execute("CREATE TABLE t (id INTEGER)")
    con.commit()
    con.close()

    storage = tmp_path / "storage"
    storage.mkdir()

    calls: list = []

    def fake_slayer_ingest(conn_str, storage_path, **kw):
        calls.append(conn_str)

    with patch("bird_interact_agents.slayer_pipeline.orchestrator._slayer_ingest", side_effect=fake_slayer_ingest):
        _phase1_ingest("alien", storage, sqlite_path=sqlite_path)

    assert len(calls) == 1
    assert "sqlite" in calls[0].lower()


# ---------------------------------------------------------------------------
# ensure_db_cache — postgres does not require sqlite file
# ---------------------------------------------------------------------------


def test_ensure_db_cache_postgres_skips_sqlite_check(tmp_path):
    """ensure_db_cache for postgres benchmark must not fail with
    'SQLite not found' even if no .sqlite file exists."""
    from bird_interact_agents.slayer_otf.cache import ensure_db_cache
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("livesqlbench_postgres")
    db_name = "alien"
    cache_root = tmp_path / "cache"
    data_root = tmp_path / "livesqlbench-base-lite-postgres"
    db_dir = data_root / db_name
    db_dir.mkdir(parents=True)

    (db_dir / f"{db_name}_schema.txt").write_text("CREATE TABLE t (id INTEGER)")
    (db_dir / f"{db_name}_kb.jsonl").write_text("")
    (db_dir / f"{db_name}_column_meaning_base.json").write_text("{}")

    async def _fake_build(**kw):
        pass

    with patch("bird_interact_agents.slayer_otf.cache._build_async", side_effect=_fake_build):
        import asyncio
        asyncio.run(ensure_db_cache(
            db_name,
            cache_root=cache_root,
            data_root=data_root,
            benchmark=b,
            force=True,
        ))
    # If we get here without FileNotFoundError, the sqlite check was skipped correctly.


# ---------------------------------------------------------------------------
# Phases 3 and 4 are skipped for postgres
# ---------------------------------------------------------------------------


def test_phase3_skipped_for_postgres(tmp_path):
    """_phase3_jsonb must be a no-op for postgres benchmarks (postgres has
    native JSONB; text-as-JSON detection is meaningless)."""
    import asyncio
    from bird_interact_agents.slayer_pipeline.orchestrator import _phase3_jsonb

    b = get_benchmark("livesqlbench_postgres")
    storage = MagicMock()

    with patch("bird_interact_agents.slayer_pipeline.orchestrator._detect_jsonb_columns") as mock_detect:
        added, _, _ = asyncio.run(_phase3_jsonb(storage, "alien", benchmark=b))
        mock_detect.assert_not_called()
    assert added == 0


def test_phase4_skipped_for_postgres(tmp_path):
    """_phase4_dates must be a no-op for postgres benchmarks (postgres has
    native date/timestamp types; text-as-date retyping is meaningless)."""
    import asyncio
    from bird_interact_agents.slayer_pipeline.orchestrator import _phase4_dates

    b = get_benchmark("livesqlbench_postgres")
    storage = MagicMock()

    with patch("bird_interact_agents.slayer_pipeline.orchestrator.detect_and_apply") as mock_detect:
        retyped, _ = asyncio.run(_phase4_dates(storage, "alien", benchmark=b, sqlite_path=None))
        mock_detect.assert_not_called()
    assert retyped == 0
