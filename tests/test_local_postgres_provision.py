"""DEV-1638: the local-postgres cluster bootstrap moved into the package.

``scripts/setup_local_postgres.py``'s logic now lives at
``bird_interact_agents.local_postgres`` so the installed ``bird-interact``
console script can provision on demand. ``provision_and_export`` is the new
high-level orchestrator the CLI calls.

These pin the ORCHESTRATION contract with the subprocess-touching steps
monkeypatched (no real cluster): step order, the returned ``BIRD_PG_*`` dict,
that it does NOT mutate ``os.environ`` (the CLI does that), and that step
failures (`resolve_dbs_for` / `start_cluster` SystemExit) propagate. A real
end-to-end provision is exercised only by the ``@pytest.mark.integration``
smoke at the bottom.
"""

from __future__ import annotations

import os

import pytest

from bird_interact_agents import local_postgres as lp


_BENCH = "livesqlbench-large"


def _patch_steps(monkeypatch, calls: list[str]):
    monkeypatch.setattr(
        lp, "_resolve_bindir", lambda: calls.append("_resolve_bindir") or "BINDIR",
    )
    monkeypatch.setattr(
        lp, "resolve_dbs_for",
        lambda bench, ids: (calls.append(f"resolve:{ids}") or ["db_a"]),
    )
    monkeypatch.setattr(
        lp, "ensure_cluster", lambda bindir: calls.append("ensure_cluster"),
    )
    monkeypatch.setattr(
        lp, "start_cluster", lambda bindir, port: calls.append("start_cluster"),
    )
    monkeypatch.setattr(
        lp, "ensure_roles", lambda bindir, port: calls.append("ensure_roles"),
    )
    monkeypatch.setattr(
        lp, "load_databases",
        lambda bindir, port, bench, dbs: calls.append("load_databases"),
    )
    monkeypatch.setattr(
        lp, "write_env", lambda port: calls.append("write_env"),
    )


def test_provision_and_export_step_order_and_return(monkeypatch):
    calls: list[str] = []
    _patch_steps(monkeypatch, calls)
    before = dict(os.environ)

    exports = lp.provision_and_export(_BENCH, ["solar_panel_6"], 5544)

    # Steps run in dependency order; bindir + DB resolution precede cluster ops.
    assert calls == [
        "_resolve_bindir",
        "resolve:['solar_panel_6']",
        "ensure_cluster",
        "start_cluster",
        "ensure_roles",
        "load_databases",
        "write_env",
    ]
    # Returns exactly the BIRD_PG_* export dict for the chosen port ...
    assert exports == lp.env_exports(5544)
    assert exports["BIRD_PG_PORT"] == "5544"
    # ... and does NOT mutate the process environment itself.
    assert dict(os.environ) == before


def test_resolve_failure_propagates(monkeypatch):
    monkeypatch.setattr(lp, "_resolve_bindir", lambda: "BINDIR")

    def _boom(bench, ids):
        raise SystemExit("pg_dumps/ missing")

    monkeypatch.setattr(lp, "resolve_dbs_for", _boom)
    with pytest.raises(SystemExit):
        lp.provision_and_export(_BENCH, ["x_1"], 5544)


def test_start_cluster_failure_propagates(monkeypatch):
    calls: list[str] = []
    _patch_steps(monkeypatch, calls)

    def _boom(bindir, port):
        raise SystemExit("already running on another port")

    monkeypatch.setattr(lp, "start_cluster", _boom)
    with pytest.raises(SystemExit):
        lp.provision_and_export(_BENCH, None, 5544)
    # load_databases must NOT run once the cluster failed to start.
    assert "load_databases" not in calls


@pytest.mark.integration
def test_real_provision_smoke(tmp_path):
    """Opt-in: a real provision on a spare port when the postgres server
    toolchain is present. Skips cleanly otherwise."""
    try:
        lp._resolve_bindir()
    except SystemExit:
        pytest.skip("postgres server toolchain (initdb/pg_ctl) not present")
    # A full DB load needs staged pg_dumps/; this smoke only asserts the
    # cluster can be brought up + BIRD_PG_* exported without sudo.
    bindir = lp._resolve_bindir()
    try:
        lp.ensure_cluster(bindir)
        lp.start_cluster(bindir, 5546)
        lp.ensure_roles(bindir, 5546)
        assert lp.env_exports(5546)["BIRD_PG_PORT"] == "5546"
    finally:
        lp.stop_cluster(bindir)
