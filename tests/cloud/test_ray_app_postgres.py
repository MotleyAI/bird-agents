"""Tests for _ensure_postgres_loaded in ray_app."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bird_interact_agents.cloud import ray_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg_dumps_dir(tmp_path: Path, dbs: list[str]) -> Path:
    """Create a pg_dumps/<db>/<db>.sql tree under tmp_path; return data_dir."""
    for db in dbs:
        db_dir = tmp_path / "pg_dumps" / db
        db_dir.mkdir(parents=True)
        (db_dir / f"{db}.sql").write_text(f"-- {db} dump\nCREATE TABLE t (id int);\n")
    return tmp_path


def _runuser_postgres(*args: str) -> list[str]:
    return ["runuser", "-u", "postgres", "--", *args]


def _patch_pg_markers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Monkeypatch both marker functions to use tmp_path instead of /tmp."""
    monkeypatch.setattr(
        ray_app, "_pg_loaded_marker",
        lambda data_dir: tmp_path / "pg_loaded.marker",
    )
    monkeypatch.setattr(
        ray_app, "_pg_db_marker",
        lambda db, data_dir: tmp_path / f"pg_db_loaded_{db}.marker",
    )


# ---------------------------------------------------------------------------
# _ensure_postgres_loaded — basic: calls pg_ctlcluster, createdb, psql -f
# ---------------------------------------------------------------------------


def test_ensure_postgres_loaded_calls_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """_ensure_postgres_loaded invokes pg_ctlcluster start, dropdb, createdb,
    and psql -f for each SQL file, in order."""
    data_dir = _make_pg_dumps_dir(tmp_path, ["alien", "bird"])
    monkeypatch.setattr(ray_app, "_pg_version", lambda: "17")
    monkeypatch.setattr(ray_app, "_PG_INIT_LOCK", tmp_path / "pg_init.lock")
    _patch_pg_markers(monkeypatch, tmp_path)

    calls_seen: list[list[str]] = []

    def fake_run(cmd, *, check=False, capture_output=False):  # noqa: ARG001
        calls_seen.append(list(cmd))
        return MagicMock(returncode=0)

    monkeypatch.setattr(ray_app.subprocess, "run", fake_run)

    ray_app._ensure_postgres_loaded(data_dir)

    assert calls_seen[0] == ["pg_ctlcluster", "17", "main", "start"]

    # role-creation call appears before any createdb
    role_calls = [c for c in calls_seen if "psql" in c and "-c" in c]
    assert len(role_calls) == 1
    assert "bird_interact" in role_calls[0][-1]

    # Each DB: dropdb --if-exists then createdb then psql -f
    assert _runuser_postgres("dropdb", "--if-exists", "alien") in calls_seen
    assert _runuser_postgres("dropdb", "--if-exists", "bird") in calls_seen
    assert _runuser_postgres("createdb", "alien") in calls_seen
    assert _runuser_postgres("createdb", "bird") in calls_seen

    alien_sql = str(data_dir / "pg_dumps" / "alien" / "alien.sql")
    bird_sql = str(data_dir / "pg_dumps" / "bird" / "bird.sql")
    assert _runuser_postgres("psql", "-d", "alien", "-f", alien_sql) in calls_seen
    assert _runuser_postgres("psql", "-d", "bird", "-f", bird_sql) in calls_seen

    # Both per-DB markers and the global marker must be written.
    assert (tmp_path / "pg_db_loaded_alien.marker").exists()
    assert (tmp_path / "pg_db_loaded_bird.marker").exists()
    assert (tmp_path / "pg_loaded.marker").exists()


# ---------------------------------------------------------------------------
# _ensure_postgres_loaded — idempotent: marker prevents re-init
# ---------------------------------------------------------------------------


def test_ensure_postgres_loaded_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If the marker already exists, _ensure_postgres_loaded is a no-op."""
    data_dir = _make_pg_dumps_dir(tmp_path, ["alien"])
    marker_path = tmp_path / "pg_loaded.marker"
    marker_path.touch()

    monkeypatch.setattr(ray_app, "_pg_version", lambda: "17")
    monkeypatch.setattr(ray_app, "_PG_INIT_LOCK", tmp_path / "pg_init.lock")
    monkeypatch.setattr(
        ray_app, "_pg_loaded_marker",
        lambda data_dir: marker_path,
    )

    calls_seen: list = []
    monkeypatch.setattr(
        ray_app.subprocess, "run",
        lambda *a, **_kw: calls_seen.append(a) or MagicMock(returncode=0),
    )

    ray_app._ensure_postgres_loaded(data_dir)

    assert calls_seen == [], "subprocess.run should NOT be called when marker exists"


# ---------------------------------------------------------------------------
# _ensure_postgres_loaded — missing pg_dumps/ raises
# ---------------------------------------------------------------------------


def test_ensure_postgres_loaded_missing_pg_dumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Raises RuntimeError when pg_dumps/ is absent."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(ray_app, "_pg_version", lambda: "17")
    monkeypatch.setattr(ray_app, "_PG_INIT_LOCK", tmp_path / "pg_init.lock")
    monkeypatch.setattr(
        ray_app, "_pg_loaded_marker",
        lambda data_dir: tmp_path / "pg_loaded.marker",
    )

    with pytest.raises(RuntimeError, match="pg_dumps/ directory missing"):
        ray_app._ensure_postgres_loaded(data_dir)


# ---------------------------------------------------------------------------
# _ensure_postgres_loaded — concurrent actors serialised via flock
# ---------------------------------------------------------------------------


def test_ensure_postgres_loaded_retry_skips_completed_dbs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """On retry, DBs whose per-DB marker already exists are skipped; only
    the remaining DB is dropped, created, and loaded."""
    data_dir = _make_pg_dumps_dir(tmp_path, ["alien", "bird"])
    monkeypatch.setattr(ray_app, "_pg_version", lambda: "17")
    monkeypatch.setattr(ray_app, "_PG_INIT_LOCK", tmp_path / "pg_init.lock")
    _patch_pg_markers(monkeypatch, tmp_path)
    # Simulate "alien" completed in a prior attempt.
    (tmp_path / "pg_db_loaded_alien.marker").touch()

    calls_seen: list[list[str]] = []

    def fake_run(cmd, *, check=False, capture_output=False):  # noqa: ARG001
        calls_seen.append(list(cmd))
        return MagicMock(returncode=0)

    monkeypatch.setattr(ray_app.subprocess, "run", fake_run)

    ray_app._ensure_postgres_loaded(data_dir)

    # alien must not be touched at all.
    assert _runuser_postgres("dropdb", "--if-exists", "alien") not in calls_seen
    assert _runuser_postgres("createdb", "alien") not in calls_seen
    alien_sql = str(data_dir / "pg_dumps" / "alien" / "alien.sql")
    assert _runuser_postgres("psql", "-d", "alien", "-f", alien_sql) not in calls_seen

    # bird must be fully loaded.
    assert _runuser_postgres("dropdb", "--if-exists", "bird") in calls_seen
    assert _runuser_postgres("createdb", "bird") in calls_seen
    bird_sql = str(data_dir / "pg_dumps" / "bird" / "bird.sql")
    assert _runuser_postgres("psql", "-d", "bird", "-f", bird_sql) in calls_seen

    assert (tmp_path / "pg_db_loaded_bird.marker").exists()
    assert (tmp_path / "pg_loaded.marker").exists()


def test_ensure_postgres_loaded_concurrent_actors_serialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two threads calling _ensure_postgres_loaded concurrently must not
    double-init: pg_ctlcluster is called exactly once."""
    data_dir = _make_pg_dumps_dir(tmp_path, ["alien"])

    monkeypatch.setattr(ray_app, "_pg_version", lambda: "17")
    monkeypatch.setattr(ray_app, "_PG_INIT_LOCK", tmp_path / "pg_init.lock")
    _patch_pg_markers(monkeypatch, tmp_path)

    pg_ctlcluster_call_count = [0]

    def fake_run(cmd, *, check=False, capture_output=False):  # noqa: ARG001
        if "pg_ctlcluster" in cmd:
            pg_ctlcluster_call_count[0] += 1
        return MagicMock(returncode=0)

    monkeypatch.setattr(ray_app.subprocess, "run", fake_run)

    errors: list[Exception] = []

    def run_actor():
        try:
            ray_app._ensure_postgres_loaded(data_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=run_actor)
    t2 = threading.Thread(target=run_actor)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive(), f"t1 still alive; errors: {errors}"
    assert not t2.is_alive(), f"t2 still alive; errors: {errors}"
    assert errors == [], f"thread errors: {errors}"
    assert pg_ctlcluster_call_count[0] == 1, (
        f"pg_ctlcluster called {pg_ctlcluster_call_count[0]} times, expected 1"
    )
    assert (tmp_path / "pg_loaded.marker").exists()
