"""Provision a private, sudo-free PostgreSQL cluster for local postgres-benchmark runs.

The cloud worker auto-loads postgres benchmarks via
``cloud/ray_app.py::_ensure_postgres_loaded`` (needs ``runuser -u postgres`` /
root). There is no equivalent for a laptop run, so a local
``--dataset livesqlbench-large`` (or any ``db_backend == "postgres"`` benchmark)
has nowhere to connect. This script fills that gap WITHOUT sudo: it runs an
entirely separate PostgreSQL *cluster* owned by the current OS user, on its own
port, under ``<main_checkout>/.local_pg/``.

What it does (all idempotent — safe to re-run):
  1. ``initdb`` a cluster at ``<main_checkout>/.local_pg/data`` (trust auth on
     loopback — this is a transient, single-tenant dev cluster, no network
     exposure, so no password hardening; cf. the project's "don't overdo
     security for self-controlled envs" posture).
  2. Start it on ``BIRD_PG_PORT`` (default 5544) so it never collides with a
     system postgres on 5432.
  3. Create the roles the dumps + harness need: ``bird_interact`` (the LOGIN
     role the harness connects as) and ``root`` (the ``OWNER TO root`` target in
     every livesqlbench dump). Both SUPERUSER.
  4. ``createdb`` + ``psql -f <db>/*.sql`` for each database the benchmark needs
     (all of them, or just those backing ``--instance-ids``). Per-DB markers
     skip already-loaded databases on re-run.
  5. Write ``<main_checkout>/.local_pg/env.sh`` with the ``BIRD_PG_*`` exports and
     print them, so a subsequent ``bird-interact`` run "just works":

        eval "$(uv run python scripts/setup_local_postgres.py --benchmark livesqlbench-large)"
        uv run bird-interact --dataset livesqlbench-large ...   # connects to :5544

Other subcommands: ``--stop`` (stop the cluster), ``--recreate`` (wipe + rebuild).

The load recipe mirrors ``_ensure_postgres_loaded`` (createdb + ``psql -f`` per
``*.sql``, no ``ON_ERROR_STOP`` — matches the cloud, and lets a stray
``OWNER TO root`` degrade gracefully even if the role step were skipped).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.benchmark import Benchmark, get_benchmark
from bird_interact_agents.harness import load_benchmark_tasks

DEFAULT_PORT = 5544
PG_USER = "bird_interact"
PG_PASSWORD = "bird_interact"
# Roles every cluster needs: the harness LOGIN role + the dump OWNER role.
_REQUIRED_ROLES = (PG_USER, "root")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable — no subprocess / filesystem side effects)
# --------------------------------------------------------------------------- #
def cluster_dir() -> Path:
    return paths.main_checkout_root() / ".local_pg"


def resolve_dbs_for(
    benchmark: str, instance_ids: list[str] | None = None
) -> list[str]:
    """Return the sorted unique ``selected_database`` names to load.

    With ``instance_ids`` → only the DBs backing those tasks. Otherwise every
    DB that has a ``pg_dumps/<db>/`` directory for the benchmark.
    """
    bm = get_benchmark(benchmark)
    pg_dumps = paths.benchmark_data_root(bm.name) / "pg_dumps"
    available = {
        p.name for p in pg_dumps.iterdir() if p.is_dir()
    } if pg_dumps.is_dir() else set()

    if instance_ids:
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

    return sorted(available)


def _resolve_bindir() -> Path:
    """Locate the PostgreSQL server bin dir (initdb/pg_ctl live here, unlike
    the client tools which are on PATH)."""
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
        subprocess.run(["rm", "-rf", str(p["data"])], check=True)
    if not (p["data"] / "PG_VERSION").exists():
        subprocess.run(
            [str(bindir / "initdb"), "-D", str(p["data"]),
             "-U", PG_USER, "-A", "trust", "--encoding=UTF8"],
            check=True, capture_output=True,
        )


def start_cluster(bindir: Path, port: int) -> None:
    p = _paths()
    if _server_running(bindir, p["data"]):
        return
    subprocess.run(
        [str(bindir / "pg_ctl"), "-D", str(p["data"]), "-l", str(p["log"]),
         "-o", f"-p {port} -k {p['sock']} -c listen_addresses=127.0.0.1",
         "-w", "start"],
        check=True, capture_output=True,
    )


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
        for sql in sql_files:
            print(f"  [load] {db} <- {sql.name}", file=sys.stderr)
            # -o /dev/null: dumps run `SELECT setval(...)` etc. whose result
            # rows would otherwise spam stdout.
            _psql(bindir, port, p["sock"], "-q", "-o", os.devnull,
                  "-f", str(sql), db=db)
        marker.touch()


def write_env(port: int) -> Path:
    p = _paths()
    lines = [f'export {k}="{v}"' for k, v in env_exports(port).items()]
    p["env"].write_text("\n".join(lines) + "\n")
    return p["env"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", help="Benchmark token, e.g. livesqlbench-large.")
    ap.add_argument("--instance-ids", default=None,
                    help="Comma-separated ids; load only their DBs (default: all).")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("BIRD_PG_PORT", DEFAULT_PORT)))
    ap.add_argument("--recreate", action="store_true",
                    help="Wipe the cluster and rebuild from scratch.")
    ap.add_argument("--stop", action="store_true",
                    help="Stop the local cluster and exit.")
    args = ap.parse_args(argv)

    bindir = _resolve_bindir()

    if args.stop:
        stop_cluster(bindir)
        print("stopped local postgres cluster", file=sys.stderr)
        return 0

    if not args.benchmark:
        ap.error("--benchmark is required (unless --stop)")
    bm = get_benchmark(args.benchmark)
    if getattr(bm, "db_backend", "sqlite") != "postgres":
        ap.error(f"{bm.name} is not a postgres benchmark (db_backend="
                 f"{getattr(bm, 'db_backend', 'sqlite')})")

    ids = ([s.strip() for s in args.instance_ids.split(",")]
           if args.instance_ids else None)
    dbs = resolve_dbs_for(bm.name, ids)
    print(f"provisioning {len(dbs)} DB(s) for {bm.name} on port {args.port}",
          file=sys.stderr)

    ensure_cluster(bindir, recreate=args.recreate)
    start_cluster(bindir, args.port)
    ensure_roles(bindir, args.port)
    load_databases(bindir, args.port, bm.name, dbs)
    env_path = write_env(args.port)
    print(f"env written to {env_path}", file=sys.stderr)

    # stdout = shell-evalable exports so callers can `eval "$(... )"`.
    for k, v in env_exports(args.port).items():
        print(f'export {k}="{v}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
