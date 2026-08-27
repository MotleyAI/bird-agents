#!/usr/bin/env python3
"""Admin CLI over ``bird_interact_agents.cube_local`` (mirrors
``setup_local_postgres.py``): provision / stop / recreate the local Cube
container and (re)generate per-DB Cube models.

``bird-interact --query-mode cube`` auto-provisions this at task prep; this
script is for the things it does NOT do (explicit stop / recreate / regen).
Requires ``BIRD_PG_*`` (or a running ``.local_pg``) to point at Postgres.
"""

from __future__ import annotations

import argparse
import os
import sys

from bird_interact_agents import local_postgres, paths
from bird_interact_agents.cube_local import deploy, model_gen


def _pg_env() -> dict:
    env = {k: os.environ[k] for k in
           ("BIRD_PG_HOST", "BIRD_PG_PORT", "BIRD_PG_USER", "BIRD_PG_PASSWORD")
           if k in os.environ}
    if "BIRD_PG_HOST" not in env:
        sys.exit("BIRD_PG_* not set; start .local_pg (setup_local_postgres.py) first.")
    return env


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="setup_local_cube")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--instance-ids", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    p.add_argument("--stop", action="store_true", help="Stop + remove the container.")
    p.add_argument("--recreate", action="store_true", help="Force-remove before starting.")
    p.add_argument("--regenerate-models", action="store_true",
                   help="Rebuild model dirs even if the fingerprint is unchanged.")
    args = p.parse_args(argv)

    if args.stop:
        deploy.stop_cube(args.benchmark)
        print(f"stopped cube container for {args.benchmark}")
        return

    pg_env = _pg_env()
    dbs = sorted(set(local_postgres.resolve_dbs_for(args.benchmark, args.instance_ids)))
    if args.recreate:
        deploy.stop_cube(args.benchmark)
    if args.regenerate_models:
        root = paths.cube_local_root(benchmark=args.benchmark) / "conf" / "model"
        for db in dbs:
            fp = root / db / "_model_fp.txt"
            if fp.exists():
                fp.unlink()
    model_gen.ensure_models(args.benchmark, dbs, pg_env)
    info = deploy.ensure_cube_running(args.benchmark, pg_env)
    deploy.poll_models_ready(info, dbs)
    print(f"cube ready: {info.base_url} (dbs: {', '.join(dbs)})")


if __name__ == "__main__":
    main()
