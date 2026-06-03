"""Regression: `_phase1_ingest` must build an absolute, openable SQLite
connection string even when handed a RELATIVE `sqlite_path` (which happens
when the harness runs with a relative `--db-path`, e.g. the README's
`--db-path ../livesqlbench-base-lite-sqlite/`).

Previously `_phase1_ingest` hard-coded `f"sqlite:////{sqlite_path}"`; with a
relative path that yields `sqlite:////../…` → SQLAlchemy resolves it at the
filesystem root → "unable to open database file". The fix routes through
`portable_connection.absolute_sqlite_url`, and canonicalises the datasource
YAML the slayer CLI writes (later phases read that YAML).

These tests mock only `subprocess.run`, so the real conn-string + YAML logic
runs without spawning the slayer CLI.
"""

from __future__ import annotations

import types
from pathlib import Path

import yaml

from bird_interact_agents.slayer_pipeline import orchestrator
from bird_interact_agents.slayer_pipeline.portable_connection import (
    absolute_sqlite_url,
)

DB = "alien"


def _install_fake_slayer(monkeypatch, storage: Path, *, cli_writes: str):
    """Patch orchestrator.subprocess.run: `datasources create` records the
    conn-string arg and writes a datasource YAML carrying ``cli_writes`` as
    the connection_string (simulating whatever slash form the CLI persists);
    `ingest` is a no-op success."""
    captured: dict[str, str] = {}

    def fake_run(cmd, *a, **kw):
        # Fidelity: the orchestrator MUST point the CLI at the build storage.
        assert kw.get("env", {}).get("SLAYER_STORAGE") == str(storage)
        if cmd[:3] == ["slayer", "datasources", "create"]:
            captured["conn"] = cmd[3]
            ds_dir = storage / "datasources"
            ds_dir.mkdir(parents=True, exist_ok=True)
            (ds_dir / f"{DB}.yaml").write_text(
                f"name: {DB}\ntype: sqlite\nconnection_string: {cli_writes}\n"
            )
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        if cmd[:2] == ["slayer", "ingest"]:
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    return captured


def test_phase1_conn_string_is_absolute_from_relative_root(
    tmp_path, monkeypatch,
):
    db_dir = tmp_path / "lsb" / DB
    db_dir.mkdir(parents=True)
    (db_dir / f"{DB}.sqlite").touch()
    monkeypatch.chdir(tmp_path)

    storage = tmp_path / "build"
    captured = _install_fake_slayer(
        monkeypatch, storage, cli_writes=f"sqlite:///{DB}/{DB}.sqlite",
    )

    rel_sqlite = Path("lsb") / DB / f"{DB}.sqlite"
    orchestrator._phase1_ingest(DB, storage, sqlite_path=rel_sqlite)

    want = absolute_sqlite_url(rel_sqlite)
    # 1) the conn string passed to `datasources create` is canonical absolute
    assert captured["conn"] == want
    assert "/../" not in captured["conn"]
    assert not captured["conn"].startswith("sqlite:////..")
    # 2) the persisted datasource YAML is canonicalised (later phases read it)
    persisted = yaml.safe_load(
        (storage / "datasources" / f"{DB}.yaml").read_text()
    )
    assert persisted["connection_string"] == want


def test_phase1_canonicalises_cli_written_yaml_regardless_of_form(
    tmp_path, monkeypatch,
):
    """Even if the CLI persists a malformed 5-slash / relative form, the
    datasource YAML ends up canonical (robust rewrite, not a string-replace
    keyed on the exact input path)."""
    db_dir = tmp_path / "lsb" / DB
    db_dir.mkdir(parents=True)
    (db_dir / f"{DB}.sqlite").touch()
    monkeypatch.chdir(tmp_path)
    storage = tmp_path / "build"
    rel_sqlite = Path("lsb") / DB / f"{DB}.sqlite"
    want = absolute_sqlite_url(rel_sqlite)

    _install_fake_slayer(
        monkeypatch, storage,
        cli_writes="sqlite://///wrong/garbage/path.sqlite",
    )
    orchestrator._phase1_ingest(DB, storage, sqlite_path=rel_sqlite)
    persisted = yaml.safe_load(
        (storage / "datasources" / f"{DB}.yaml").read_text()
    )
    assert persisted["connection_string"] == want
