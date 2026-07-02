"""One-shot orchestrator: run a postgres-backed benchmark LOCALLY, end to end.

Postgres benchmarks (livesqlbench-*, bird-interact-lite-exp, …) have no turnkey
local path — the DB-load logic lives only in the cloud worker, and cloud auth
env isn't in a laptop shell. This script wires the three missing pieces so a
local run "just works":

  1. Load auth env from a dotenv-style file (``--env-file`` / ``BIRD_ENV_FILE``)
     so ``--subscription-auth`` (CLAUDE_CODE_OAUTH_TOKEN) or the API-key path has
     credentials. Missing file → skipped (assume the shell already has them).
  2. Provision a private, sudo-free PostgreSQL cluster and load the benchmark's
     DBs (delegates to ``setup_local_postgres``), then export ``BIRD_PG_*``.
  3. Sync the stable task annotations from GCS (delegates to
     ``fetch_local_annotations``) so ``--require-annotation`` passes.

Then it execs ``bird-interact`` with ``--data`` / ``--db-path`` auto-derived from
the benchmark registry. Everything after ``--`` is passed straight through to
``bird-interact`` (framework, mode, query-mode, agent-model, auth flag, …).

Example — the 15-task livesqlbench-large raw claude_sdk run:

    uv run python scripts/run_local_postgres.py \
        --benchmark livesqlbench-large \
        --instance-ids solar_panel_6,fake_account_15,... \
        -- \
        --framework claude_sdk --query-mode raw --mode one-shot \
        --agent-model anthropic/claude-opus-4-8 --subscription-auth \
        --use-audited-gold-sql --concurrency 1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark

import fetch_local_annotations
import setup_local_postgres as slp

# Machine-specific default location of the auth dotenv (holds
# CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY). Override with --env-file or
# BIRD_ENV_FILE; a missing file is not fatal.
_DEFAULT_ENV_FILE = os.environ.get(
    "BIRD_ENV_FILE",
    str(Path.home() / "Dropbox/PyCharmProjects/shared/.env.ubuntu"),
)


def load_env_file(path: Path) -> int:
    """Merge ``KEY=VALUE`` / ``export KEY=VALUE`` lines into os.environ.

    Returns the count of vars set. Silently returns 0 if the file is absent.
    """
    if not path.is_file():
        return 0
    n = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ[key] = val
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--benchmark", required=True)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--instance-ids", default=None,
                     help="Comma-separated ids. Default: whole benchmark.")
    grp.add_argument("--instance-ids-file", default=None,
                     help="File with one instance_id per line.")
    ap.add_argument("--env-file", default=_DEFAULT_ENV_FILE,
                    help=f"Auth dotenv to load (default: {_DEFAULT_ENV_FILE}).")
    ap.add_argument("--pg-port", type=int,
                    default=int(os.environ.get("BIRD_PG_PORT", slp.DEFAULT_PORT)))
    ap.add_argument("--skip-annotations", action="store_true",
                    help="Do not sync annotations from GCS.")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="Everything after `--` is passed to bird-interact.")
    args = ap.parse_args(argv)

    bm = get_benchmark(args.benchmark)
    if getattr(bm, "db_backend", "sqlite") != "postgres":
        ap.error(f"{bm.name} is not a postgres benchmark; run bird-interact directly.")

    # `bird-interact`'s own args come after the leading `--` that REMAINDER keeps.
    bi_args = list(args.passthrough)
    if bi_args and bi_args[0] == "--":
        bi_args = bi_args[1:]

    ids: list[str] | None = None
    if args.instance_ids:
        ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    elif args.instance_ids_file:
        ids = [ln.strip() for ln in Path(args.instance_ids_file).read_text().splitlines()
               if ln.strip()]

    # 1) auth env
    n_env = load_env_file(Path(args.env_file))
    print(f"[run-local-pg] loaded {n_env} vars from {args.env_file}"
          if n_env else f"[run-local-pg] no env file at {args.env_file} (skipping)",
          file=sys.stderr)

    # 2) provision postgres + load DBs, then export BIRD_PG_*
    bindir = slp._resolve_bindir()
    dbs = slp.resolve_dbs_for(bm.name, ids)
    print(f"[run-local-pg] provisioning {len(dbs)} DB(s) on port {args.pg_port}",
          file=sys.stderr)
    slp.ensure_cluster(bindir)
    slp.start_cluster(bindir, args.pg_port)
    slp.ensure_roles(bindir, args.pg_port)
    slp.load_databases(bindir, args.pg_port, bm.name, dbs)
    os.environ.update(slp.env_exports(args.pg_port))

    # 3) sync annotations from GCS (best-effort)
    if not args.skip_annotations:
        try:
            fa_argv = ["--benchmark", bm.name]
            if ids:
                fa_argv += ["--instance-ids", ",".join(ids)]
            fetch_local_annotations.main(fa_argv)
        except Exception as exc:  # noqa: BLE001
            print(f"[run-local-pg] annotation sync failed ({exc}); "
                  "pass --no-require-annotation to bird-interact if intended.",
                  file=sys.stderr)

    # 4) exec bird-interact with derived data/db-path + passthrough
    data_file = str(paths.benchmark_data_file(bm.name))
    db_path = str(paths.benchmark_data_root(bm.name))
    cmd = [
        "bird-interact",
        "--dataset", bm.name,
        "--data", data_file,
        "--db-path", db_path,
        *bi_args,
    ]
    if ids:
        cmd += ["--instance-id", ",".join(ids)]
    print(f"[run-local-pg] exec: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd, env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
