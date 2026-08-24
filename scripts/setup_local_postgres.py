"""Admin CLI for the local, sudo-free PostgreSQL cluster.

DEV-1638: the provisioning logic moved into the package
(``bird_interact_agents.local_postgres``) so the installed ``bird-interact``
console script can provision on demand. This script is retained as the ADMIN
tool for the operations ``bird-interact`` does NOT do:

  * ``--stop``     — stop the local cluster.
  * ``--recreate`` — wipe the cluster and rebuild from scratch.
  * standalone provision of a benchmark's DBs (all, or ``--instance-ids``).

    eval "$(uv run python scripts/setup_local_postgres.py --benchmark livesqlbench-large)"
    uv run bird-interact --dataset livesqlbench-large ...   # connects to :5544

(For a turnkey local run you no longer need this — ``bird-interact --dataset
<postgres benchmark>`` auto-provisions unless ``BIRD_PG_HOST`` is already set.)
"""
from __future__ import annotations

import argparse
import os
import sys

from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.local_postgres import (
    DEFAULT_PORT,
    _port_change_note,
    ensure_cluster,
    ensure_roles,
    env_exports,
    load_databases,
    resolve_bindir,
    resolve_dbs_for,
    resolve_port,
    running_cluster_port,
    start_cluster,
    stop_cluster,
    write_env,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    bindir = resolve_bindir()

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

    # Distinguish OMITTED (None → whole benchmark) from an explicitly-provided
    # empty value (`--instance-ids ""` / `",,,"` → [] → no DBs), so a degenerate
    # value never silently provisions every DB (Codex PR #75 r5).
    ids = (None if args.instance_ids is None
           else [s.strip() for s in args.instance_ids.split(",") if s.strip()])
    dbs = resolve_dbs_for(bm.name, ids)

    # DEV-1685: auto-select a free port. Resolve AFTER ensure_cluster so a
    # --recreate (which stops + wipes the old cluster) doesn't adopt the port
    # of a cluster we're about to destroy — after a wipe running_cluster_port
    # is None, so the requested port is honoured (bumped off a busy / reserved
    # :5432 target). Without --recreate, an already-running singleton is
    # adopted. The RESOLVED port flows into every lifecycle call + env_exports
    # so the harness connects where the cluster actually listens.
    ensure_cluster(bindir, recreate=args.recreate)
    running = running_cluster_port(bindir)
    port = resolve_port(args.port, running)
    note = _port_change_note(args.port, port, running)
    if note:
        print(note, file=sys.stderr)

    print(f"provisioning {len(dbs)} DB(s) for {bm.name} on port {port}",
          file=sys.stderr)

    start_cluster(bindir, port)
    ensure_roles(bindir, port)
    load_databases(bindir, port, bm.name, dbs)
    env_path = write_env(port)
    print(f"env written to {env_path}", file=sys.stderr)

    # stdout = shell-evalable exports so callers can `eval "$(... )"`.
    for k, v in env_exports(port).items():
        print(f'export {k}="{v}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
