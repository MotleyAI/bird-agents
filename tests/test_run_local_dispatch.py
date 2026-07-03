"""DEV-1638: the local ``bird-interact`` unification wiring in ``run.py``.

One local entrypoint. ``bird-interact --dataset <name>`` derives
``--data``/``--db-path`` from the registry when omitted (both backends), and —
for a postgres benchmark with no pre-set ``BIRD_PG_*`` — provisions a local
cluster + exports ``BIRD_PG_*``, then best-effort syncs task annotations, then
runs. These pin the helpers + the CLI boundary (mechanical wiring only, per
`feedback_no_prompt_content_tests`).
"""

from __future__ import annotations

import argparse
import os

import pytest

from bird_interact_agents import paths, run


class _CalledError(RuntimeError):
    """Stand-in for argparse's ``parser.error`` (which exits)."""


def _err(msg: str):
    raise _CalledError(msg)


# --------------------------------------------------------------------------- #
# _resolve_data_paths — derive both / passthrough / both-or-neither
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bench", ["mini-interact", "livesqlbench-large"])
def test_resolve_data_paths_derives_both_backends(bench):
    args = argparse.Namespace(dataset=bench, data=None, db_path=None)
    run._resolve_data_paths(args, error=_err)
    assert args.data == str(paths.benchmark_data_file(bench))
    assert args.db_path == str(paths.benchmark_data_root(bench))


def test_resolve_data_paths_passthrough_when_both_given():
    args = argparse.Namespace(
        dataset="mini-interact", data="/x/data.jsonl", db_path="/x/db",
    )
    run._resolve_data_paths(args, error=_err)
    assert args.data == "/x/data.jsonl"
    assert args.db_path == "/x/db"


@pytest.mark.parametrize("data,db_path", [("/x", None), (None, "/x")])
def test_resolve_data_paths_exactly_one_errors(data, db_path):
    args = argparse.Namespace(dataset="mini-interact", data=data, db_path=db_path)
    with pytest.raises(_CalledError):
        run._resolve_data_paths(args, error=_err)


# --------------------------------------------------------------------------- #
# _effective_instance_ids — filter_ids / --limit / whole benchmark
# --------------------------------------------------------------------------- #
def test_effective_ids_prefers_filter_ids():
    args = argparse.Namespace(dataset="livesqlbench-large", data="d", limit=None)
    assert run._effective_instance_ids(args, ["a", "b"]) == ["a", "b"]


def test_effective_ids_from_limit_loads_first_n(monkeypatch):
    rows = [{"instance_id": f"x_{i}", "selected_database": "x"} for i in range(5)]

    def _load(dataset, data_path, limit=None, filter_ids=None):
        return rows[:limit] if limit else rows

    monkeypatch.setattr(run, "load_benchmark_tasks", _load)
    args = argparse.Namespace(dataset="livesqlbench-large", data="d", limit=2)
    assert run._effective_instance_ids(args, None) == ["x_0", "x_1"]


def test_effective_ids_whole_benchmark_is_none():
    args = argparse.Namespace(dataset="livesqlbench-large", data="d", limit=None)
    assert run._effective_instance_ids(args, None) is None


# --------------------------------------------------------------------------- #
# _maybe_bootstrap_local_postgres — gated on backend + BIRD_PG_HOST (Codex #1)
# --------------------------------------------------------------------------- #
@pytest.fixture
def _stub_provision(monkeypatch):
    calls = []

    def _prov(dataset, ids, port):
        calls.append((dataset, ids, port))
        return {"BIRD_PG_HOST": "127.0.0.1", "BIRD_PG_PORT": str(port)}

    monkeypatch.setattr(run, "provision_and_export", _prov)
    return calls


@pytest.fixture(autouse=True)
def _clear_pg_host(monkeypatch):
    monkeypatch.delenv("BIRD_PG_HOST", raising=False)
    monkeypatch.delenv("BIRD_PG_PORT", raising=False)


def test_bootstrap_fires_for_postgres_when_pg_host_unset(_stub_provision, monkeypatch):
    args = argparse.Namespace(dataset="livesqlbench-large", pg_port=5544)
    run._maybe_bootstrap_local_postgres(args, ["solar_panel_6"])
    assert _stub_provision == [("livesqlbench-large", ["solar_panel_6"], 5544)]
    assert os.environ["BIRD_PG_HOST"] == "127.0.0.1"


def test_bootstrap_skips_for_sqlite(_stub_provision):
    args = argparse.Namespace(dataset="mini-interact", pg_port=5544)
    run._maybe_bootstrap_local_postgres(args, None)
    assert _stub_provision == []


def test_bootstrap_skips_when_pg_host_already_set(_stub_provision, monkeypatch):
    monkeypatch.setenv("BIRD_PG_HOST", "my-own-pg")
    args = argparse.Namespace(dataset="livesqlbench-large", pg_port=5544)
    run._maybe_bootstrap_local_postgres(args, None)
    assert _stub_provision == []


def test_bootstrap_fires_for_postgres_even_with_explicit_paths(_stub_provision):
    """Codex #1 regression: explicit --data/--db-path must NOT suppress
    provisioning — postgres connectivity is BIRD_PG_*, not the data paths."""
    args = argparse.Namespace(dataset="livesqlbench-large", pg_port=5544)
    # (explicit paths live on args.data/db_path but are irrelevant to the gate)
    run._maybe_bootstrap_local_postgres(args, None)
    assert len(_stub_provision) == 1


# --------------------------------------------------------------------------- #
# _maybe_sync_annotations — best-effort, opt-out, swallows failures
# --------------------------------------------------------------------------- #
def test_sync_called_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run, "sync_annotations",
        lambda ds, ids: calls.append((ds, ids)) or {
            "fetched": 0, "already_local": 0, "missing_in_gcs": 0},
    )
    args = argparse.Namespace(dataset="mini-interact", skip_annotations=False)
    run._maybe_sync_annotations(args, ["a"])
    assert calls == [("mini-interact", ["a"])]


def test_sync_skipped_when_opted_out(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "sync_annotations", lambda ds, ids: calls.append(1))
    args = argparse.Namespace(dataset="mini-interact", skip_annotations=True)
    run._maybe_sync_annotations(args, None)
    assert calls == []


def test_sync_failure_is_swallowed(monkeypatch):
    def _boom(ds, ids):
        raise RuntimeError("GCS down")

    monkeypatch.setattr(run, "sync_annotations", _boom)
    args = argparse.Namespace(dataset="mini-interact", skip_annotations=False)
    # Must not raise — best-effort.
    run._maybe_sync_annotations(args, None)


def test_sync_warns_on_still_missing(monkeypatch, caplog):
    """Plan: warn loudly on any id still missing after sync (surfaces the
    silent implicit-N1 fallback)."""
    monkeypatch.setattr(
        run, "sync_annotations",
        lambda ds, ids: {"fetched": 0, "already_local": 0, "missing_in_gcs": 2},
    )
    args = argparse.Namespace(dataset="mini-interact", skip_annotations=False)
    import logging
    with caplog.at_level(logging.WARNING):
        run._maybe_sync_annotations(args, ["a", "b"])
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)


# --------------------------------------------------------------------------- #
# CLI boundary (Codex #4) — drive main() with argv, everything downstream stubbed
# --------------------------------------------------------------------------- #
@pytest.fixture
def _stub_run(monkeypatch):
    """Stubs every side-effecting main() step and records an ORDERED event log,
    so CLI tests can assert both that a step fired and its relative order."""
    captured: dict = {"events": [], "run_kwargs": None,
                      "provision_calls": [], "sync_calls": [],
                      "env_file_calls": []}

    async def _fake_run_eval(**kwargs):
        captured["run_kwargs"] = kwargs
        captured["events"].append("run")
        return {}

    def _fake_sync(ds, ids):
        captured["sync_calls"].append((ds, ids))
        captured["events"].append("sync")
        return {"fetched": 0, "already_local": 0, "missing_in_gcs": 0}

    def _fake_provision(ds, ids, port):
        captured["provision_calls"].append((ds, ids, port))
        captured["events"].append("provision")
        # Full BIRD_PG_* dict so we can assert os.environ.update applies all keys.
        from bird_interact_agents.local_postgres import env_exports
        return env_exports(port)

    def _fake_load_env(path):
        captured["env_file_calls"].append(str(path))
        return 0

    monkeypatch.setattr(run, "run_evaluation", _fake_run_eval)
    monkeypatch.setattr(run, "sync_annotations", _fake_sync)
    monkeypatch.setattr(run, "provision_and_export", _fake_provision)
    monkeypatch.setattr(run, "load_env_file", _fake_load_env)
    monkeypatch.setattr(run, "_maybe_start_bridge_proxy", lambda **k: None)
    return captured


def _argv(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["bird-interact", *args])


def _sqlite_argv(monkeypatch, *extra):
    _argv(
        monkeypatch,
        "--framework", "pydantic_ai", "--mode", "a-interact",
        "--dataset", "mini-interact", "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        *extra,
    )


def _pg_argv(monkeypatch, *extra):
    _argv(
        monkeypatch,
        "--framework", "pydantic_ai", "--mode", "one-shot",
        "--dataset", "livesqlbench-large", "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-id", "solar_panel_6",
        *extra,
    )


def test_cli_sqlite_no_paths_derives_syncs_and_runs(_stub_run, monkeypatch):
    _sqlite_argv(monkeypatch)  # no --data/--db-path → must NOT SystemExit
    run.main()
    rk = _stub_run["run_kwargs"]
    assert rk["data_path"] == str(paths.benchmark_data_file("mini-interact"))
    from pathlib import Path
    assert rk["data_dir"] == str(
        Path(paths.benchmark_data_root("mini-interact")).resolve()
    )
    # sqlite → NO provision; annotation sync DID fire, and BEFORE the run.
    assert _stub_run["provision_calls"] == []
    assert _stub_run["sync_calls"] == [("mini-interact", None)]
    assert _stub_run["events"].index("sync") < _stub_run["events"].index("run")


def test_cli_postgres_no_paths_provisions_then_syncs_then_runs(_stub_run, monkeypatch):
    _pg_argv(monkeypatch)
    run.main()
    assert _stub_run["provision_calls"] == [
        ("livesqlbench-large", ["solar_panel_6"], 5544),  # default pg-port
    ]
    assert _stub_run["sync_calls"] == [("livesqlbench-large", ["solar_panel_6"])]
    # Plan order: provision (export BIRD_PG_*) → sync → run.
    assert _stub_run["events"] == ["provision", "sync", "run"]
    # os.environ.update applied ALL BIRD_PG_* keys from the returned dict.
    assert os.environ["BIRD_PG_PORT"] == "5544"
    assert os.environ["BIRD_PG_USER"] == "bird_interact"


def test_cli_postgres_explicit_paths_still_provisions(_stub_run, monkeypatch, tmp_path):
    """Codex #1 regression at the CLI boundary: explicit --data/--db-path must
    NOT suppress provisioning (postgres connects via BIRD_PG_*, not paths)."""
    data = tmp_path / "d.jsonl"
    dbp = tmp_path / "dbroot"
    data.write_text("")
    dbp.mkdir()
    _pg_argv(monkeypatch, "--data", str(data), "--db-path", str(dbp))
    run.main()
    assert len(_stub_run["provision_calls"]) == 1


def test_cli_postgres_skips_provision_when_pg_host_set(_stub_run, monkeypatch):
    monkeypatch.setenv("BIRD_PG_HOST", "my-own-pg")
    _pg_argv(monkeypatch)
    run.main()
    assert _stub_run["provision_calls"] == []


def test_cli_pg_port_flag_propagates(_stub_run, monkeypatch):
    _pg_argv(monkeypatch, "--pg-port", "6001")
    run.main()
    assert _stub_run["provision_calls"] == [
        ("livesqlbench-large", ["solar_panel_6"], 6001),
    ]


def test_cli_env_file_is_loaded(_stub_run, monkeypatch, tmp_path):
    envf = tmp_path / ".env.agents"
    envf.write_text("ANTHROPIC_API_KEY=sk-x\n")
    _sqlite_argv(monkeypatch, "--env-file", str(envf))
    run.main()
    assert _stub_run["env_file_calls"] == [str(envf)]


def test_cli_env_file_defaults_from_bird_env_file(_stub_run, monkeypatch, tmp_path):
    envf = tmp_path / ".env.agents"
    envf.write_text("ANTHROPIC_API_KEY=sk-x\n")
    monkeypatch.setenv("BIRD_ENV_FILE", str(envf))
    _sqlite_argv(monkeypatch)  # no --env-file → default from BIRD_ENV_FILE
    run.main()
    assert _stub_run["env_file_calls"] == [str(envf)]


def test_cli_skip_annotations_flag_disables_sync(_stub_run, monkeypatch):
    _sqlite_argv(monkeypatch, "--skip-annotations")
    run.main()
    assert _stub_run["sync_calls"] == []


def test_cli_explicit_both_paths_passthrough(_stub_run, monkeypatch, tmp_path):
    data = tmp_path / "d.jsonl"
    dbp = tmp_path / "dbroot"
    data.write_text("")
    dbp.mkdir()
    _sqlite_argv(monkeypatch, "--data", str(data), "--db-path", str(dbp))
    run.main()
    assert _stub_run["run_kwargs"]["data_path"] == str(data)
    from pathlib import Path
    assert _stub_run["run_kwargs"]["data_dir"] == str(Path(dbp).resolve())


def test_cli_exactly_one_path_errors(_stub_run, monkeypatch, tmp_path):
    _sqlite_argv(monkeypatch, "--data", str(tmp_path / "d.jsonl"))  # no --db-path
    with pytest.raises(SystemExit):
        run.main()


def test_cli_empty_filter_ids_file_errors_before_provision(_stub_run, monkeypatch, tmp_path):
    """Codex PR #75: an empty --filter-ids file must error BEFORE provisioning +
    sync (not fall through to a whole-benchmark provision)."""
    empty = tmp_path / "ids.txt"
    empty.write_text("\n  \n")  # only blanks
    _pg_argv_no_id = [
        "--framework", "pydantic_ai", "--mode", "one-shot",
        "--dataset", "livesqlbench-large", "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--filter-ids", str(empty),
    ]
    _argv(monkeypatch, *_pg_argv_no_id)
    with pytest.raises(SystemExit):
        run.main()
    # Must NOT have provisioned or synced the whole benchmark.
    assert _stub_run["provision_calls"] == []
    assert _stub_run["sync_calls"] == []
