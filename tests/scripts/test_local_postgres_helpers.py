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

import socket
import subprocess

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


# --------------------------------------------------------------------------- #
# DEV-1685: auto-select a free port so the .local_pg cluster never collides
# with the IAP tunnel (:5432) or a stray listener.
# --------------------------------------------------------------------------- #
def test_port_scan_constants_are_pinned():
    """The scan contract: :5432 reserved, 64-wide inclusive window, capped at
    the max TCP port."""
    slp = setup_local_postgres
    assert slp._RESERVED_PORTS == frozenset({5432})
    assert slp._PORT_SCAN_SPAN == 64
    assert slp._MAX_PORT == 65535


def test_port_available_true_for_unbound_and_false_while_held():
    slp = setup_local_postgres
    # Bind an ephemeral port so we KNOW something is listening on it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        held = s.getsockname()[1]
        # A live listener ⇒ the port reads as unavailable.
        assert slp._port_available(held) is False
    # Once the socket is closed the port frees up again (may briefly linger in
    # TIME_WAIT, but a fresh ephemeral port bound above is chosen to be free).


def test_port_available_true_for_a_definitely_free_port():
    slp = setup_local_postgres
    # Grab an OS-assigned free port, release it, then assert it probes free.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert slp._port_available(free) is True


def test_resolve_port_adopts_running_verbatim(monkeypatch):
    """A running cluster's port wins unconditionally — even over an explicit
    preferred port, and without probing availability at all."""
    slp = setup_local_postgres

    def _boom(*a, **k):
        raise AssertionError("_port_available must not be consulted when a "
                             "cluster is already running")

    monkeypatch.setattr(slp, "_port_available", _boom)
    assert slp.resolve_port(preferred=5544, running_port=6001) == 6001
    # Even when the explicit preferred differs from the running port.
    assert slp.resolve_port(preferred=5432, running_port=5533) == 5533


def test_resolve_port_rejects_reserved_running_cluster(monkeypatch):
    """A legacy .local_pg cluster on a reserved port (:5432) must NOT be
    adopted — that would collide with the IAP tunnel. Hard-fail instead so the
    operator stops + migrates it (CodeRabbit, DEV-1685)."""
    slp = setup_local_postgres

    def _boom(*a, **k):
        raise AssertionError("_port_available must not be probed on the "
                             "reserved-running-port reject path")

    monkeypatch.setattr(slp, "_port_available", _boom)
    with pytest.raises(SystemExit, match="reserved"):
        slp.resolve_port(preferred=5544, running_port=5432)


def test_resolve_port_preferred_free_unchanged(monkeypatch):
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_port_available", lambda *a, **k: True)
    assert slp.resolve_port(preferred=5544, running_port=None) == 5544


def test_resolve_port_preferred_busy_scans_to_next_free(monkeypatch):
    slp = setup_local_postgres
    # 5544 busy, 5545 free.
    monkeypatch.setattr(slp, "_port_available", lambda port, *a, **k: port != 5544)
    assert slp.resolve_port(preferred=5544, running_port=None) == 5545


def test_resolve_port_skips_reserved_5432_even_when_free(monkeypatch):
    """Preferred == 5432 must never be adopted even if 5432 probes free —
    it belongs to the IAP tunnel."""
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_port_available", lambda *a, **k: True)
    assert slp.resolve_port(preferred=5432, running_port=None) == 5433


def test_resolve_port_scans_inclusive_to_preferred_plus_span(monkeypatch):
    """The window is inclusive of preferred+_PORT_SCAN_SPAN — the LAST port in
    the window must be reachable (an exclusive range(preferred, preferred+64)
    would miss it and wrongly SystemExit)."""
    slp = setup_local_postgres
    last = 6000 + slp._PORT_SCAN_SPAN  # 6064
    monkeypatch.setattr(slp, "_port_available", lambda port, *a, **k: port == last)
    assert slp.resolve_port(preferred=6000, running_port=None) == last


def test_resolve_port_raises_when_window_exhausted(monkeypatch):
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_port_available", lambda *a, **k: False)
    with pytest.raises(SystemExit):
        slp.resolve_port(preferred=5544, running_port=None)


def test_resolve_port_single_candidate_at_max_port(monkeypatch):
    """preferred == _MAX_PORT: window is just {65535}; free → adopt it."""
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_port_available", lambda *a, **k: True)
    assert slp.resolve_port(preferred=slp._MAX_PORT, running_port=None) == slp._MAX_PORT


def test_resolve_port_never_probes_above_max_port(monkeypatch):
    """Near the ceiling the scan must cap at _MAX_PORT and never probe 65536+
    (which would OverflowError in socket.bind). All in-range busy → SystemExit,
    NOT a traceback."""
    slp = setup_local_postgres
    probed = []

    def _spy(port, *a, **k):
        probed.append(port)
        return False

    monkeypatch.setattr(slp, "_port_available", _spy)
    with pytest.raises(SystemExit):
        slp.resolve_port(preferred=slp._MAX_PORT - 2, running_port=None)
    assert max(probed) <= slp._MAX_PORT
    assert all(p <= slp._MAX_PORT for p in probed)


@pytest.mark.parametrize("bad", [0, -1, 70000, 65536])
def test_resolve_port_rejects_invalid_preferred(monkeypatch, bad):
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_port_available", lambda *a, **k: True)
    with pytest.raises(SystemExit):
        slp.resolve_port(preferred=bad, running_port=None)


def test_running_cluster_port_none_when_down(monkeypatch, tmp_path):
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_paths", lambda: {"data": tmp_path})
    monkeypatch.setattr(slp, "_server_running", lambda *a, **k: False)
    # _running_port must not even be consulted when the server is down.
    monkeypatch.setattr(slp, "_running_port", lambda _d: 5544)
    assert slp.running_cluster_port(bindir=None) is None


def test_running_cluster_port_returns_port_when_up(monkeypatch, tmp_path):
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_paths", lambda: {"data": tmp_path})
    monkeypatch.setattr(slp, "_server_running", lambda *a, **k: True)
    monkeypatch.setattr(slp, "_running_port", lambda _d: 5544)
    assert slp.running_cluster_port(bindir=None) == 5544


def test_port_change_note_unchanged_is_none():
    slp = setup_local_postgres
    assert slp._port_change_note(5544, 5544, running_port=None) is None


def test_port_change_note_adopting_running():
    slp = setup_local_postgres
    note = slp._port_change_note(5544, 5533, running_port=5533)
    assert note is not None
    assert "5533" in note and "5544" in note
    # Reason-aware: an adoption reads as adopting/running, not "busy".
    assert ("adopt" in note.lower()) or ("running" in note.lower())
    assert "busy" not in note.lower()
    assert "tunnel" not in note.lower()


def test_port_change_note_reserved():
    slp = setup_local_postgres
    note = slp._port_change_note(5432, 5433, running_port=None)
    assert note is not None
    assert "reserved" in note.lower()
    assert "5432" in note and "5433" in note
    assert "tunnel" not in note.lower()


def test_port_change_note_busy():
    slp = setup_local_postgres
    note = slp._port_change_note(5544, 5545, running_port=None)
    assert note is not None
    assert "5544" in note and "5545" in note
    # Reason-aware: a busy preferred reads as busy/unavailable, not "reserved".
    assert ("busy" in note.lower()) or ("unavailable" in note.lower())
    assert "reserved" not in note.lower()
    assert "tunnel" not in note.lower()


def test_start_cluster_raises_when_running_but_port_unreadable(monkeypatch, tmp_path):
    """A running cluster whose postmaster.pid line 4 is missing/malformed must
    hard-fail (we can't know its port), NOT silently return and export a wrong
    BIRD_PG_PORT (Codex 4/6)."""
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_paths", lambda: {"data": tmp_path})
    monkeypatch.setattr(slp, "_server_running", lambda *a, **k: True)
    monkeypatch.setattr(slp, "_running_port", lambda _d: None)
    with pytest.raises(SystemExit):
        slp.start_cluster(bindir=None, port=5544)


def test_start_cluster_wraps_pg_ctl_bind_failure(monkeypatch, tmp_path):
    """A TOCTOU race (port taken between probe and pg_ctl start) must surface
    as a clear SystemExit naming the port + log, not a raw CalledProcessError
    (Codex 7)."""
    slp = setup_local_postgres
    monkeypatch.setattr(slp, "_paths", lambda: {
        "data": tmp_path / "data", "log": tmp_path / "server.log",
        "sock": tmp_path / "sock",
    })
    monkeypatch.setattr(slp, "_server_running", lambda *a, **k: False)

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(
            1, ["pg_ctl"], output="", stderr="bind: Address already in use")

    monkeypatch.setattr(slp.subprocess, "run", _boom)
    with pytest.raises(SystemExit) as ei:
        slp.start_cluster(bindir=tmp_path / "bin", port=5544)
    msg = str(ei.value)
    assert "5544" in msg
    # Points the operator at the server log for the real bind error.
    assert "server.log" in msg


def test_provision_and_export_threads_resolved_port_everywhere(monkeypatch):
    """The crux of DEV-1685: when the requested port is unavailable, the
    RESOLVED port must flow into start_cluster/ensure_roles/load_databases/
    write_env AND the returned BIRD_PG_* dict — so the harness connects where
    the cluster actually listens, not where it was asked to."""
    slp = setup_local_postgres
    calls: dict[str, object] = {}

    monkeypatch.setattr(slp, "resolve_bindir", lambda: "BINDIR")
    monkeypatch.setattr(slp, "running_cluster_port", lambda _b: None)
    # Simulate requested 5544 busy → resolver picks 5601.
    monkeypatch.setattr(slp, "resolve_port",
                        lambda preferred, running: 5601)
    monkeypatch.setattr(slp, "resolve_dbs_for", lambda _bm, _ids: ["db1"])
    monkeypatch.setattr(slp, "ensure_cluster", lambda *a, **k: None)
    monkeypatch.setattr(slp, "start_cluster",
                        lambda bindir, port: calls.__setitem__("start", port))
    monkeypatch.setattr(slp, "ensure_roles",
                        lambda bindir, port: calls.__setitem__("roles", port))
    monkeypatch.setattr(
        slp, "load_databases",
        lambda bindir, port, bm, dbs: calls.__setitem__("load", port))
    monkeypatch.setattr(slp, "write_env",
                        lambda port: calls.__setitem__("write", port))

    exports = slp.provision_and_export("livesqlbench-large", ["x_1"], 5544)

    assert exports["BIRD_PG_PORT"] == "5601"
    assert exports["BIRD_PG_HOST"] == "127.0.0.1"
    assert calls == {"start": 5601, "roles": 5601, "load": 5601, "write": 5601}


def test_provision_and_export_adopts_running_cluster_port(monkeypatch):
    """When a cluster is already running, provision_and_export must adopt its
    port (singleton) and export THAT, ignoring the requested port."""
    slp = setup_local_postgres
    seen: dict[str, object] = {}

    monkeypatch.setattr(slp, "resolve_bindir", lambda: "BINDIR")
    monkeypatch.setattr(slp, "running_cluster_port", lambda _b: 5533)
    # Real resolve_port: running_port wins verbatim.
    monkeypatch.setattr(slp, "resolve_dbs_for", lambda _bm, _ids: ["db1"])
    monkeypatch.setattr(slp, "ensure_cluster", lambda *a, **k: None)
    monkeypatch.setattr(slp, "start_cluster",
                        lambda bindir, port: seen.__setitem__("start", port))
    monkeypatch.setattr(slp, "ensure_roles", lambda *a, **k: None)
    monkeypatch.setattr(slp, "load_databases", lambda *a, **k: None)
    monkeypatch.setattr(slp, "write_env", lambda *a, **k: None)

    exports = slp.provision_and_export("livesqlbench-large", None, 5544)
    assert exports["BIRD_PG_PORT"] == "5533"
    assert seen["start"] == 5533
