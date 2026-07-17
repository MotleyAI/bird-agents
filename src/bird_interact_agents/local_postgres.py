"""Provision a private, sudo-free PostgreSQL cluster for local postgres benchmarks.

The cloud worker auto-loads postgres benchmarks via
``cloud/ray_app.py::_ensure_postgres_loaded`` (needs ``runuser -u postgres`` /
root). There is no equivalent for a laptop run, so a local
``bird-interact --dataset livesqlbench-large`` (or any ``db_backend ==
"postgres"`` benchmark) has nowhere to connect. This module fills that gap
WITHOUT sudo: it runs an entirely separate PostgreSQL *cluster* owned by the
current OS user, on its own port, under ``<main_checkout>/.local_pg/``.

DEV-1638: this logic used to live in ``scripts/setup_local_postgres.py``. It
moved into the package so the installed ``bird-interact`` console script can
provision on demand (``provision_and_export``). ``scripts/setup_local_postgres.py``
is now a thin admin CLI (``--stop`` / ``--recreate`` / standalone provision)
that imports these functions.

What it does (all idempotent — safe to re-run):
  1. ``initdb`` a cluster at ``<main_checkout>/.local_pg/data`` (trust auth on
     loopback — a transient, single-tenant dev cluster, no network exposure).
  2. Start it on ``BIRD_PG_PORT`` (default 5544) so it never collides with a
     system postgres on 5432.
  3. Create the roles the dumps + harness need: ``bird_interact`` (the LOGIN
     role the harness connects as) and ``root`` (the ``OWNER TO root`` target in
     every livesqlbench dump). Both SUPERUSER.
  4. ``createdb`` + ``psql -f <db>/*.sql`` for each database the benchmark needs
     (all of them, or just those backing ``--instance-ids``). Per-DB markers
     skip already-loaded databases on re-run.
  5. Write ``<main_checkout>/.local_pg/env.sh`` with the ``BIRD_PG_*`` exports.

The load recipe mirrors ``_ensure_postgres_loaded`` (createdb + ``psql -f`` per
``*.sql``, no ``ON_ERROR_STOP`` — matches the cloud, and lets a stray
``OWNER TO root`` degrade gracefully even if the role step were skipped).
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.harness import load_benchmark_tasks

DEFAULT_PORT = 5544
PG_USER = "bird_interact"
PG_PASSWORD = "bird_interact"
# Roles every cluster needs: the harness LOGIN role + the dump OWNER role.
_REQUIRED_ROLES = (PG_USER, "root")
# DEV-1654: raise max_connections above the stock Postgres default of 100 so the
# shared single-tenant dev cluster has headroom for concurrent agents (each
# SLayer query path holds a bounded async+sync pool) plus grading, without
# masking a real leak. Passed as a postmaster `-c` override at start; a cluster
# that is ALREADY running only picks up the new value after `--stop`/`--recreate`.
PG_MAX_CONNECTIONS = 200

# DEV-1685: :5432 is reserved for the `gcloud start-iap-tunnel` mapping to a
# REMOTE PostgreSQL (the benchmark's default BIRD_PG_PORT=5432 reaches the
# remote DB through that tunnel). Our self-hosted cluster must NEVER bind it,
# even when free. When the target port is occupied, scan a bounded window for
# the next free port instead of dying with a cryptic pg_ctl bind error.
_RESERVED_PORTS = frozenset({5432})
_PORT_SCAN_SPAN = 64  # scan preferred .. preferred+_PORT_SCAN_SPAN (inclusive)
_MAX_PORT = 65535


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable — no subprocess / filesystem side effects)
# --------------------------------------------------------------------------- #
def _port_available(port: int, host: str = "127.0.0.1") -> bool:
    """True iff a TCP listener can bind ``(host, port)`` right now.

    Plain bind-probe with NO ``SO_REUSEADDR`` so an active listener (the IAP
    tunnel, a stray process, or a foreign postgres) reads as unavailable. This
    is inherently TOCTOU — a port free at probe time can be taken before
    ``pg_ctl`` binds it — so ``start_cluster`` still surfaces a clear error on
    the losing race (the window is narrow and this only degrades to the old
    failure mode, never to a silent wrong-port).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def resolve_port(preferred: int, running_port: "int | None") -> int:
    """The port to provision/connect on (DEV-1685).

    * ``running_port`` not None → adopt it verbatim: the private cluster is a
      singleton, so whatever port it is already listening on wins — even over
      an explicit ``preferred`` (caller logs the adoption; migrate with
      ``--stop``). This also keeps the exported ``BIRD_PG_PORT`` stable across
      runs, so the OTF cache is not thrashed.
    * else scan ``preferred .. preferred+_PORT_SCAN_SPAN`` (inclusive, capped at
      ``_MAX_PORT``), skipping ``_RESERVED_PORTS``, and return the first free
      port. ``SystemExit`` if the whole window is occupied.
    """
    if running_port is not None:
        return running_port
    if not (1 <= preferred <= _MAX_PORT):
        raise SystemExit(
            f"invalid preferred postgres port {preferred}; must be "
            f"1..{_MAX_PORT}. Set --pg-port / BIRD_PG_PORT to a valid port."
        )
    last = min(preferred + _PORT_SCAN_SPAN, _MAX_PORT)
    for port in range(preferred, last + 1):
        if port in _RESERVED_PORTS:
            continue
        if _port_available(port):
            return port
    raise SystemExit(
        f"no free port in {preferred}..{last} "
        f"(reserved: {sorted(_RESERVED_PORTS)}); free one up or set "
        f"--pg-port / BIRD_PG_PORT to an open port."
    )


def _port_change_note(
    requested: int, resolved: int, running_port: "int | None"
) -> "str | None":
    """A human-readable, reason-aware log line when the resolved port differs
    from the requested one, else ``None``. Never names the tunnel."""
    if resolved == requested:
        return None
    if running_port is not None:
        return (
            f"adopting the already-running local postgres cluster on port "
            f"{resolved} (requested {requested}); stop it (--stop) to migrate."
        )
    if requested in _RESERVED_PORTS:
        return f"port {requested} is reserved; using free port {resolved}"
    return f"port {requested} unavailable (busy); using free port {resolved}"
def cluster_dir() -> Path:
    return paths.main_checkout_root() / ".local_pg"


def resolve_dbs_for(
    benchmark: str, instance_ids: list[str] | None = None
) -> list[str]:
    """Return the sorted unique ``selected_database`` names to load.

    A non-empty ``instance_ids`` → only the DBs backing those tasks. ``None``
    (omitted) → every DB that has a ``pg_dumps/<db>/`` directory. An EXPLICIT
    empty list → ``[]`` (no scope), NOT the whole benchmark (Codex PR #75 r3):
    `None` and `[]` must not collapse, else a malformed
    ``setup_local_postgres.py --instance-ids ",,,"`` would provision every DB.
    """
    bm = get_benchmark(benchmark)
    pg_dumps = paths.benchmark_data_root(bm.name) / "pg_dumps"
    available = {
        p.name for p in pg_dumps.iterdir() if p.is_dir()
    } if pg_dumps.is_dir() else set()

    if instance_ids is not None:
        if not instance_ids:
            return []  # explicit empty subset → load nothing (not everything)
        rows = {
            r["instance_id"]: r
            for r in load_benchmark_tasks(
                bm.name, str(paths.benchmark_data_file(bm.name)),
                filter_ids=instance_ids,
            )
        }
        missing = [i for i in instance_ids if i not in rows]
        if missing:
            raise SystemExit(f"instance_ids not found in {bm.name}: {missing}")
        wanted = {rows[i]["selected_database"] for i in instance_ids}
        absent = sorted(wanted - available)
        if absent:
            raise SystemExit(
                f"pg_dumps/ missing for DBs {absent} under {pg_dumps}"
            )
        return sorted(wanted)

    dbs = sorted(available)
    if not dbs:
        # Mirror the subset path's fail-fast: a full run with no dumps would
        # otherwise "provision 0 DB(s)" and launch bird-interact against an
        # empty cluster.
        raise SystemExit(
            f"no pg_dumps/<db>/ dumps found under {pg_dumps}; "
            "fetch them with scripts/download_pg_dumps.py first"
        )
    return dbs


def resolve_bindir() -> Path:
    """Locate the PostgreSQL server bin dir (initdb/pg_ctl live here, unlike
    the client tools which are on PATH). Public: imported across the module
    boundary by ``scripts/setup_local_postgres.py`` (CodeRabbit PR #75)."""
    if os.environ.get("PG_BINDIR"):
        return Path(os.environ["PG_BINDIR"])
    candidates = sorted(
        Path("/usr/lib/postgresql").glob("*/bin"),
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else -1,
        reverse=True,
    )
    for c in candidates:
        if (c / "initdb").exists() and (c / "pg_ctl").exists():
            return c
    raise SystemExit(
        "Could not find initdb/pg_ctl under /usr/lib/postgresql/*/bin. "
        "Set PG_BINDIR to your PostgreSQL server bin directory."
    )


def env_exports(port: int) -> dict[str, str]:
    return {
        "BIRD_PG_HOST": "127.0.0.1",
        "BIRD_PG_PORT": str(port),
        "BIRD_PG_USER": PG_USER,
        "BIRD_PG_PASSWORD": PG_PASSWORD,
    }


# --------------------------------------------------------------------------- #
# Cluster lifecycle
# --------------------------------------------------------------------------- #
def _paths() -> dict[str, Path]:
    root = cluster_dir()
    return {
        "root": root,
        "data": root / "data",
        "sock": root / "sock",
        "log": root / "server.log",
        "markers": root / "markers",
        "env": root / "env.sh",
    }


def _server_running(bindir: Path, data: Path) -> bool:
    r = subprocess.run(
        [str(bindir / "pg_ctl"), "-D", str(data), "status"],
        capture_output=True, text=True,
    )
    return r.returncode == 0  # 3 = not running, 4 = no data dir


def _psql(bindir: Path, port: int, sock: Path, *args: str, db: str = "postgres",
          check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(bindir / "psql") if (bindir / "psql").exists() else "psql",
         "-h", str(sock), "-p", str(port), "-U", PG_USER, "-d", db, *args],
        check=check, **kw,
    )


def ensure_cluster(bindir: Path, recreate: bool = False) -> None:
    p = _paths()
    p["root"].mkdir(parents=True, exist_ok=True)
    p["sock"].mkdir(parents=True, exist_ok=True)
    p["markers"].mkdir(parents=True, exist_ok=True)
    if recreate and p["data"].exists():
        stop_cluster(bindir)
        shutil.rmtree(p["data"])
    if not (p["data"] / "PG_VERSION").exists():
        subprocess.run(
            [str(bindir / "initdb"), "-D", str(p["data"]),
             "-U", PG_USER, "-A", "trust", "--encoding=UTF8"],
            check=True, capture_output=True,
        )


def _running_port(data: Path) -> "int | None":
    """The port a running cluster is listening on, from postmaster.pid line 4
    (1=pid, 2=datadir, 3=start-time, 4=port). None if not derivable."""
    pidfile = data / "postmaster.pid"
    if not pidfile.exists():
        return None
    lines = pidfile.read_text().splitlines()
    if len(lines) >= 4:
        try:
            return int(lines[3].strip())
        except ValueError:
            return None
    return None


def running_cluster_port(bindir: Path) -> "int | None":
    """The port our ``.local_pg`` cluster is CURRENTLY listening on, or ``None``
    if it is not running (DEV-1685).

    Gates on ``pg_ctl status`` so a stale ``postmaster.pid`` left by a crashed
    cluster reports ``None`` (not the dead port). Used to adopt an
    already-running singleton instead of spawning a second cluster.
    """
    data = _paths()["data"]
    if not _server_running(bindir, data):
        return None
    return _running_port(data)


def start_cluster(bindir: Path, port: int) -> None:
    p = _paths()
    if _server_running(bindir, p["data"]):
        actual = _running_port(p["data"])
        if actual is None:
            # Running but its port is unreadable (missing/malformed
            # postmaster.pid line 4). Returning here would export
            # BIRD_PG_PORT=<port> while the server listens on an unknown port,
            # so every downstream connection could fail silently.
            raise SystemExit(
                "local postgres appears to be running but its port could not "
                f"be read from {p['data'] / 'postmaster.pid'}. Stop it first: "
                "scripts/setup_local_postgres.py --stop"
            )
        if actual != port:
            # A cluster is up on a different port than requested; returning
            # here would export BIRD_PG_PORT=<port> while the server listens on
            # <actual>, so every downstream psql/agent connection would fail.
            raise SystemExit(
                f"local postgres is already running on port {actual}, not "
                f"{port}. Re-run with --port {actual}, or stop it first: "
                f"scripts/setup_local_postgres.py --stop"
            )
        return
    try:
        subprocess.run(
            [str(bindir / "pg_ctl"), "-D", str(p["data"]), "-l", str(p["log"]),
             "-o", (f"-p {port} -k {p['sock']} -c listen_addresses=127.0.0.1 "
                    f"-c max_connections={PG_MAX_CONNECTIONS}"),
             "-w", "start"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Most likely a TOCTOU race: the port we probed free was taken before
        # pg_ctl bound it. Surface a clear message (requested port + log path)
        # instead of a raw CalledProcessError traceback.
        raise SystemExit(
            f"failed to start local postgres on port {port} "
            f"(data={p['data']}); the port may have been taken after it was "
            f"probed free. See {p['log']} for the postmaster's bind error.\n"
            f"{(exc.stderr or '').strip()}"
        ) from exc


def stop_cluster(bindir: Path) -> None:
    p = _paths()
    if _server_running(bindir, p["data"]):
        subprocess.run(
            [str(bindir / "pg_ctl"), "-D", str(p["data"]), "-m", "fast", "stop"],
            check=False, capture_output=True,
        )


def ensure_roles(bindir: Path, port: int) -> None:
    p = _paths()
    pw = PG_PASSWORD.replace("'", "''")
    for role in _REQUIRED_ROLES:
        _psql(
            bindir, port, p["sock"], "-c",
            f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname="
            f"'{role}') THEN CREATE ROLE {role} LOGIN SUPERUSER PASSWORD "
            f"'{pw}'; END IF; END $$;",
            capture_output=True,
        )


def _db_exists(bindir: Path, port: int, db: str) -> bool:
    p = _paths()
    r = _psql(
        bindir, port, p["sock"], "-tAc",
        f"SELECT 1 FROM pg_database WHERE datname='{db}'",
        capture_output=True, text=True,
    )
    return r.stdout.strip() == "1"


def _table_count(bindir: Path, port: int, db: str) -> int:
    p = _paths()
    r = _psql(
        bindir, port, p["sock"], "-tAc",
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema='public'",
        db=db, capture_output=True, text=True, check=False,
    )
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def load_databases(
    bindir: Path, port: int, benchmark: str, dbs: list[str]
) -> None:
    p = _paths()
    bm = get_benchmark(benchmark)
    pg_dumps = paths.benchmark_data_root(bm.name) / "pg_dumps"
    for db in dbs:
        marker = p["markers"] / f"{bm.name}__{db}.done"
        if marker.exists() and _db_exists(bindir, port, db):
            print(f"  [skip] {db} (already loaded)", file=sys.stderr)
            continue
        if _db_exists(bindir, port, db):
            _psql(bindir, port, p["sock"], "-c", f'DROP DATABASE "{db}"',
                  capture_output=True)
        _psql(bindir, port, p["sock"], "-c", f'CREATE DATABASE "{db}"',
              capture_output=True)
        sql_files = sorted((pg_dumps / db).glob("*.sql"))
        if not sql_files:
            raise SystemExit(f"no *.sql under {pg_dumps / db}")
        had_errors = False
        for sql in sql_files:
            print(f"  [load] {db} <- {sql.name}", file=sys.stderr)
            # -o /dev/null: dumps run `SELECT setval(...)` etc. whose result
            # rows would otherwise spam stdout. Capture stderr so a partial /
            # corrupt dump (e.g. a single-table livesqlbench-large dump that
            # ALTERs FKs to tables it never creates) is SURFACED, not silently
            # loaded and left to fail later as a cryptic per-task dry_run_error.
            r = _psql(bindir, port, p["sock"], "-q", "-o", os.devnull,
                      "-v", "ON_ERROR_STOP=0", "-f", str(sql), db=db,
                      capture_output=True, text=True, check=False)
            errs = [ln for ln in (r.stderr or "").splitlines() if "ERROR:" in ln]
            if errs:
                had_errors = True
                print(f"  [WARN] {db}: {len(errs)} error(s) loading {sql.name}; "
                      f"first: {errs[0].strip()}", file=sys.stderr)
        n_tables = _table_count(bindir, port, db)
        if had_errors or n_tables == 0:
            # Do NOT touch the marker: a re-provision after the dump is fixed
            # must retry. Warn loudly so the operator does not mistake the
            # downstream dry_run_errors for agent failures.
            print(f"  [WARN] {db}: load INCOMPLETE (tables={n_tables}, "
                  f"errors={'yes' if had_errors else 'no'}); NOT marking done. "
                  f"The dump under {pg_dumps / db} is likely partial/corrupt — "
                  f"re-fetch it (scripts/download_pg_dumps.py) and re-provision.",
                  file=sys.stderr)
        else:
            marker.touch()


def write_env(port: int) -> Path:
    p = _paths()
    lines = [f'export {k}="{v}"' for k, v in env_exports(port).items()]
    p["env"].write_text("\n".join(lines) + "\n")
    return p["env"]


# --------------------------------------------------------------------------- #
# High-level orchestrator (DEV-1638): the single entry the CLI composes.
# --------------------------------------------------------------------------- #
def provision_and_export(
    benchmark: str, instance_ids: list[str] | None, port: int,
) -> dict[str, str]:
    """Idempotently provision + load the DBs backing ``instance_ids`` (all DBs
    if ``None``) and RETURN the ``BIRD_PG_*`` export dict.

    Does NOT mutate ``os.environ`` — the caller applies the returned dict. This
    keeps the function pure-return (testable) and lets the caller decide when
    the connection env becomes visible.
    """
    bindir = resolve_bindir()
    running = running_cluster_port(bindir)
    resolved = resolve_port(port, running)
    note = _port_change_note(port, resolved, running)
    if note:
        print(f"[bird-interact] {note}", file=sys.stderr)
    port = resolved
    dbs = resolve_dbs_for(benchmark, instance_ids)
    print(f"[bird-interact] provisioning {len(dbs)} DB(s) for {benchmark} "
          f"on port {port}", file=sys.stderr)
    ensure_cluster(bindir)
    start_cluster(bindir, port)
    ensure_roles(bindir, port)
    load_databases(bindir, port, benchmark, dbs)
    write_env(port)
    return env_exports(port)
