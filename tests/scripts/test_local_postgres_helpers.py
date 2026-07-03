"""Pure-helper tests for the local-postgres tooling.

DEV-1638: the logic moved from ``scripts/setup_local_postgres.py`` into the
package (``bird_interact_agents.local_postgres``) so the installed
``bird-interact`` console script can provision on demand. These tests now import
the PACKAGE module directly (no more ``importlib`` loading of the un-packaged
script). The dotenv parser moved to ``bird_interact_agents.env_file`` and is
covered by ``tests/test_env_file.py``; ``run_local_postgres`` is retired.

Covers only the deterministic logic + the marker policy with subprocess faked.
"""
from __future__ import annotations

import pytest

from bird_interact_agents import local_postgres as setup_local_postgres


def test_env_exports_shape():
    exp = setup_local_postgres.env_exports(5544)
    assert exp == {
        "BIRD_PG_HOST": "127.0.0.1",
        "BIRD_PG_PORT": "5544",
        "BIRD_PG_USER": "bird_interact",
        "BIRD_PG_PASSWORD": "bird_interact",
    }
    # Port flows through verbatim.
    assert setup_local_postgres.env_exports(6000)["BIRD_PG_PORT"] == "6000"


def test_cluster_dir_is_under_main_checkout():
    # Worktree-safe: cluster lives in the main checkout, name is stable.
    assert setup_local_postgres.cluster_dir().name == ".local_pg"


def test_required_roles_include_owner_and_login():
    # Dumps do `OWNER TO root`; harness connects as bird_interact. Both needed.
    assert set(setup_local_postgres._REQUIRED_ROLES) == {"bird_interact", "root"}


class _FakeProc:
    def __init__(self, stderr=""):
        self.stdout = ""
        self.stderr = stderr
        self.returncode = 0


def _prep_load_databases(monkeypatch, tmp_path, *, load_stderr, table_count):
    """Wire load_databases against a fake psql so we can assert the marker
    policy (touch only on a clean, non-empty load)."""
    slp = setup_local_postgres
    markers = tmp_path / "markers"
    markers.mkdir()
    monkeypatch.setattr(slp, "_paths", lambda: {"markers": markers,
                                                "sock": tmp_path / "sock"})
    # A benchmark whose pg_dumps/<db>/<db>.sql exists (content is irrelevant —
    # _psql is faked).
    pg = tmp_path / "data" / "pg_dumps" / "somedb"
    pg.mkdir(parents=True)
    (pg / "somedb.sql").write_text("-- fake dump\n")

    class _BM:
        name = "livesqlbench-large"

    monkeypatch.setattr(slp, "get_benchmark", lambda _n: _BM())
    monkeypatch.setattr(slp.paths, "benchmark_data_root",
                        lambda _n: tmp_path / "data")
    monkeypatch.setattr(slp, "_db_exists", lambda *a, **k: False)
    monkeypatch.setattr(slp, "_table_count", lambda *a, **k: table_count)

    def _fake_psql(bindir, port, sock, *args, **kw):
        # The load call carries "-f"; everything else (CREATE DATABASE, …) is
        # a no-op fake.
        if "-f" in args:
            return _FakeProc(stderr=load_stderr)
        return _FakeProc()

    monkeypatch.setattr(slp, "_psql", _fake_psql)
    return slp, markers / "livesqlbench-large__somedb.done"


def test_resolve_dbs_empty_list_is_empty_not_whole_benchmark(monkeypatch, tmp_path):
    """Codex PR #75 r3: an EXPLICIT empty instance_ids list resolves to no DBs
    (not every dump). `None` vs `[]` must not collapse."""
    slp = setup_local_postgres

    class _BM:
        name = "livesqlbench-large"

    monkeypatch.setattr(slp, "get_benchmark", lambda _n: _BM())
    monkeypatch.setattr(slp.paths, "benchmark_data_root", lambda _n: tmp_path)
    # load_benchmark_tasks must NOT be consulted for an empty subset.
    monkeypatch.setattr(
        slp, "load_benchmark_tasks",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("empty subset must not load tasks")),
    )
    assert slp.resolve_dbs_for("livesqlbench-large", []) == []


def test_resolve_dbs_raises_when_no_dumps(monkeypatch, tmp_path):
    slp = setup_local_postgres

    class _BM:
        name = "livesqlbench-large"

    monkeypatch.setattr(slp, "get_benchmark", lambda _n: _BM())
    # benchmark_data_root/pg_dumps does not exist → available is empty.
    monkeypatch.setattr(slp.paths, "benchmark_data_root", lambda _n: tmp_path)
    with pytest.raises(SystemExit):
        slp.resolve_dbs_for("livesqlbench-large", None)


def test_running_port_parses_postmaster_pid(tmp_path):
    slp = setup_local_postgres
    (tmp_path / "postmaster.pid").write_text(
        "12345\n" + str(tmp_path) + "\n1700000000\n5544\n/sock\n")
    assert slp._running_port(tmp_path) == 5544
    # No pidfile → None.
    assert slp._running_port(tmp_path / "empty") is None


def test_start_cluster_rejects_port_mismatch(monkeypatch, tmp_path):
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_paths", lambda: {"data": tmp_path})
    monkeypatch.setattr(slp, "_server_running", lambda *a, **k: True)
    monkeypatch.setattr(slp, "_running_port", lambda _d: 9999)
    with pytest.raises(SystemExit):
        slp.start_cluster(bindir=None, port=5544)
    # Matching port → no raise, returns cleanly (no subprocess start attempted).
    monkeypatch.setattr(slp, "_running_port", lambda _d: 5544)
    slp.start_cluster(bindir=None, port=5544)


def test_load_marks_done_on_clean_load(monkeypatch, tmp_path):
    slp, marker = _prep_load_databases(
        monkeypatch, tmp_path, load_stderr="", table_count=50)
    slp.load_databases(bindir=None, port=5544, benchmark="livesqlbench-large",
                       dbs=["somedb"])
    assert marker.exists()  # clean, many tables → marked done


def test_load_skips_marker_on_dump_error(monkeypatch, tmp_path):
    slp, marker = _prep_load_databases(
        monkeypatch, tmp_path,
        load_stderr='psql:somedb.sql:29771: ERROR:  relation '
                    '"public.Household_Telecomm_Metrix" does not exist\n',
        table_count=1)
    slp.load_databases(bindir=None, port=5544, benchmark="livesqlbench-large",
                       dbs=["somedb"])
    # Partial dump (FK ALTER to a missing table) → NOT marked, so a re-provision
    # after the dump is fixed will retry.
    assert not marker.exists()
