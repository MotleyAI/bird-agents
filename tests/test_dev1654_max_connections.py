"""DEV-1654 (D3): the local ``.local_pg`` cluster is started with a raised
``max_connections`` so the shared single-tenant dev cluster has headroom for
concurrent agents + grading (the stock Postgres default is 100).

``max_connections`` is a postmaster start parameter, so it is passed as a
``-c max_connections=N`` entry in the ``pg_ctl ... -o "..."`` options string.
NB: ``start_cluster`` early-returns when a cluster is already running, so an
existing ``.local_pg`` only picks up the new value after ``--stop`` /
``--recreate``; this test covers the fresh-start subprocess path only.
"""

from __future__ import annotations

from pathlib import Path

from bird_interact_agents import local_postgres as lp


def test_pg_max_connections_constant_is_200():
    assert lp.PG_MAX_CONNECTIONS == 200


def test_start_cluster_passes_max_connections(monkeypatch):
    captured: dict = {}

    # Not already running -> proceed to the pg_ctl start subprocess.
    monkeypatch.setattr(lp, "_server_running", lambda bindir, data: False)

    def _fake_run(args, **kwargs):
        captured["args"] = list(args)

        class _R:
            returncode = 0
            stdout = b""
            stderr = b""

        return _R()

    monkeypatch.setattr(lp.subprocess, "run", _fake_run)

    lp.start_cluster(Path("/opt/pg/bin"), 5544)

    args = captured["args"]
    # The postmaster options ride in the single string after ``-o``.
    opts = args[args.index("-o") + 1]
    assert f"-c max_connections={lp.PG_MAX_CONNECTIONS}" in opts
    # ... alongside the pre-existing options (regression: don't drop them).
    assert "listen_addresses=127.0.0.1" in opts
    assert "-p 5544" in opts
