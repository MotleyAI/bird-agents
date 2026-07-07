"""Tests for OTF (on-the-fly) SLayer setup with postgres-backed benchmarks.

Verifies:
- fingerprint_of for postgres uses schema-text hash, not sqlite file stat.
- ensure_db_cache for postgres does NOT require <db>.sqlite to exist.
- _phase1_ingest accepts db_url and uses it instead of sqlite_path.
- Phase 3 RUNS JSONB leaf expansion for postgres (DEV-1648) but skips the
  SQLite-only drift sampling; phase 4 (LLM date detection) stays skipped
  for postgres (its date refinement flows through phase 2).
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

    b = get_benchmark("livesqlbench-base-lite")
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

    b = get_benchmark("livesqlbench-base-lite")
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

    b = get_benchmark("mini-interact")
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

    b = get_benchmark("livesqlbench-base-lite")
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


def test_phase3_runs_jsonb_expansion_for_postgres(tmp_path):
    """DEV-1648: _phase3_jsonb now RUNS leaf expansion for postgres
    (livesqlbench has real JSONB columns), reading the meanings file. It
    only skips the SQLite-native drift sampling. (Behaviour-detail tests
    live in tests/slayer_pipeline/test_orchestrator_pg_phases.py.)"""
    import asyncio
    from bird_interact_agents.slayer_pipeline.orchestrator import _phase3_jsonb

    b = get_benchmark("livesqlbench-base-lite")
    storage = MagicMock()
    meanings_path = tmp_path / "alien_column_meaning_base.json"
    meanings_path.write_text("{}")

    with patch(
        "bird_interact_agents.slayer_pipeline.orchestrator._detect_jsonb_columns",
        return_value=[],
    ) as mock_detect:
        added, _, _ = asyncio.run(
            _phase3_jsonb(storage, "alien", meanings_path=meanings_path,
                          sqlite_path=None, benchmark=b)
        )
        # Postgres no longer short-circuits before reading the meanings.
        mock_detect.assert_called_once()
    assert added == 0


def test_phase4_skipped_for_postgres(tmp_path):
    """_phase4_dates must be a no-op for postgres benchmarks (postgres has
    native date/timestamp types; text-as-date retyping is meaningless)."""
    import asyncio
    from bird_interact_agents.slayer_pipeline.orchestrator import _phase4_dates

    b = get_benchmark("livesqlbench-base-lite")
    storage = MagicMock()

    with patch("bird_interact_agents.slayer_pipeline.orchestrator.detect_and_apply") as mock_detect:
        retyped, _ = asyncio.run(_phase4_dates(storage, "alien", benchmark=b, sqlite_path=None))
        mock_detect.assert_not_called()
    assert retyped == 0


# ---------------------------------------------------------------------------
# Password isolation — pg_password must not appear in subprocess args or YAML
# ---------------------------------------------------------------------------


def test_slayer_ingest_sets_pgpassword_env_not_url(tmp_path):
    """_slayer_ingest must inject pg_password via PGPASSWORD env var and NOT
    embed it in the subprocess command-line args."""
    import subprocess as _subprocess
    from bird_interact_agents.slayer_pipeline.orchestrator import _slayer_ingest

    captured_args: list = []
    captured_env: dict = {}

    def fake_run(args, *, env, **_kw):
        captured_args.extend(args)
        captured_env.update(env or {})
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    storage = tmp_path / "storage"
    storage.mkdir()
    conn_str = "postgresql://bird@localhost:5432/alien"  # no password in URL

    with patch("bird_interact_agents.slayer_pipeline.orchestrator.subprocess.run", side_effect=fake_run):
        _slayer_ingest(conn_str, storage, db="alien", pg_password="s3cr3t")

    # Password must NOT appear in command-line args
    for arg in captured_args:
        assert "s3cr3t" not in str(arg), f"password leaked into subprocess arg: {arg!r}"
    # Password MUST be in env
    assert captured_env.get("PGPASSWORD") == "s3cr3t"


def test_slayer_ingest_no_pgpassword_when_none(tmp_path):
    """_slayer_ingest must not set PGPASSWORD when pg_password is None (SQLite path)."""
    from bird_interact_agents.slayer_pipeline.orchestrator import _slayer_ingest

    captured_env: dict = {}

    def fake_run(args, *, env, **_kw):
        captured_env.update(env or {})
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        return result

    storage = tmp_path / "storage"
    storage.mkdir()

    with patch("bird_interact_agents.slayer_pipeline.orchestrator.subprocess.run", side_effect=fake_run):
        import os
        env_before = os.environ.copy()
        env_before.pop("PGPASSWORD", None)
        with patch.dict("os.environ", env_before, clear=True):
            _slayer_ingest("sqlite:///foo.sqlite", storage, db="foo")

    assert "PGPASSWORD" not in captured_env, "PGPASSWORD must not be set for SQLite ingest"


def test_build_async_postgres_url_excludes_password(tmp_path):
    """_build_async must pass a password-free URL to _phase1_ingest and
    supply the password via pg_password so it never appears in the URL."""
    import asyncio
    import os
    from bird_interact_agents.slayer_otf import cache as otf_cache
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("livesqlbench-base-lite")
    build_dir = tmp_path / "build"

    phase1_calls: list[dict] = []
    phase2_calls: list[dict] = []
    phase3_calls: list[dict] = []

    def fake_phase1(db, storage, *, sqlite_path=None, db_url=None, pg_password=None):
        phase1_calls.append({"db_url": db_url, "pg_password": pg_password})

    async def fake_phase2(storage, db, meanings_path, *, backend="sqlite"):
        # Capture PGPASSWORD as seen DURING the build (the postgres branch must
        # export it before phase 2/3 so the in-process engine refresh authns).
        phase2_calls.append({
            "backend": backend,
            "pgpassword": os.environ.get("PGPASSWORD"),
        })
        return 0, []

    async def fake_phase3(storage, db, meanings_path=None, sqlite_path=None, *,
                          benchmark=None, backend=None, pg_extract_sampler=None):
        phase3_calls.append({"backend": backend, "pg_extract_sampler": pg_extract_sampler})
        return 0, [], []

    async def fake_phase4(storage, db, sqlite_path=None, llm_model=None, *, benchmark=None):
        return 0, []

    env_patch = {
        "BIRD_PG_HOST": "pghost",
        "BIRD_PG_PORT": "5432",
        "BIRD_PG_USER": "pguser",
        "BIRD_PG_PASSWORD": "pgpass",
    }
    with patch.dict("os.environ", env_patch):
        os.environ.pop("PGPASSWORD", None)  # simulate a standalone caller (unset)
        with patch.object(otf_cache, "_phase1_ingest", side_effect=fake_phase1):
            with patch.object(otf_cache, "_phase2_overlay", side_effect=fake_phase2):
                with patch.object(otf_cache, "_phase3_jsonb", side_effect=fake_phase3):
                    with patch.object(otf_cache, "_phase4_dates", side_effect=fake_phase4):
                        asyncio.run(otf_cache._build_async(
                            build_dir=build_dir,
                            db="alien",
                            sqlite_path=None,
                            meanings_path=tmp_path / "meanings.json",
                            kb_rows=[],
                            benchmark=b,
                        ))

    assert len(phase1_calls) == 1
    url = phase1_calls[0]["db_url"]
    pw = phase1_calls[0]["pg_password"]
    assert url is not None
    assert "pgpass" not in url, f"password leaked into db_url: {url!r}"

    # DEV-1648: postgres backend threaded into phases 2 & 3; the JSON-leaf
    # extract sampler is threaded into phase 3 (top-level columns are not
    # refined on postgres, so phase 2 needs no sampler).
    assert phase2_calls and phase2_calls[0]["backend"] == "postgres"
    assert phase3_calls and phase3_calls[0]["backend"] == "postgres"
    assert phase3_calls[0]["pg_extract_sampler"] is not None
    # DEV-1648: when PGPASSWORD is unset (standalone caller), set-if-absent
    # fills it from BIRD_PG_PASSWORD so phase-3's in-process engine refresh
    # authenticates against the passwordless persisted datasource. (In real
    # runs harness.py already sets it, so setdefault is a no-op — never
    # clobbering a caller's value, and race-free across concurrent builds.)
    assert phase2_calls[0]["pgpassword"] == "pgpass"
    assert pw == "pgpass", f"pg_password not passed separately: {pw!r}"
