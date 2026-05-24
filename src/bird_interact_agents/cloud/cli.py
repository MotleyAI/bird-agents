"""`bird-interact-cloud` CLI."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from bird_interact_agents.cloud import driver


_VALID_MODES = ("a-interact", "c-interact", "oracle")


def _instance_ids(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bird-interact-cloud")
    sub = p.add_subparsers(dest="subcommand", required=True)

    sp_submit = sub.add_parser("submit")
    sp_submit.add_argument("--framework", required=True)
    sp_submit.add_argument("--query-mode", required=True, choices=("raw", "slayer"))
    sp_submit.add_argument("--agent-model", required=True)
    sp_submit.add_argument(
        "--user-sim-model", default="anthropic/claude-haiku-4-5-20251001"
    )
    grp = sp_submit.add_mutually_exclusive_group(required=True)
    grp.add_argument("--instance-ids", type=_instance_ids)
    grp.add_argument("--instance-ids-file", type=str)
    sp_submit.add_argument("--mode", required=True, choices=_VALID_MODES)
    sp_submit.add_argument("--patience", type=int, default=3)
    sp_submit.add_argument("--strict", action="store_true")
    sp_submit.add_argument(
        "--use-audited-gold-sql", action="store_true"
    )
    sp_submit.add_argument("--max-depth", type=int, default=3)
    sp_submit.add_argument(
        "--prompt-cache", dest="prompt_cache", action="store_true", default=True
    )
    sp_submit.add_argument(
        "--no-prompt-cache", dest="prompt_cache", action="store_false"
    )
    sp_submit.add_argument("--workers", type=int, default=4)
    sp_submit.add_argument("--actors-per-worker", type=int, default=4)
    sp_submit.add_argument("--worker-type", default="e2-standard-4")
    sp_submit.add_argument("--max-runtime-hours", type=int, default=8)
    sp_submit.add_argument("--run-id", default=None)
    sp_submit.add_argument("--detach", action="store_true")
    sp_submit.add_argument("--allow-dirty", action="store_true")

    for name in ("fetch", "kill", "resubmit"):
        spx = sub.add_parser(name)
        spx.add_argument("run_id")

    sub.add_parser("list")
    sp_build = sub.add_parser("build")
    sp_build.add_argument("--force", action="store_true")

    ns = p.parse_args(argv)

    if ns.subcommand == "submit":
        if ns.detach and ns.allow_dirty:
            p.error("--detach and --allow-dirty are mutually exclusive")
        # Cloud slayer mode isn't wired yet. The image doesn't ship the
        # SLayer setup files and the in-cluster actor has no storage path to
        # point the agent at, so slayer tasks would fail per-task mid-run
        # (FileNotFoundError in `_setup_per_task_slayer`). Reject it at submit
        # — fail fast and cheap, before building/pushing the image and
        # bringing up a cluster. Tracked for a follow-up PR that ships the
        # local SLayer setup (`slayer_models/`, `slayer_models_otf/`) into the
        # image and reuses `run_one_task`'s per-task storage path (no
        # cloud-specific slayer code). Remove this guard when that lands.
        if ns.query_mode == "slayer":
            p.error(
                "cloud slayer mode is not yet implemented (tracked for a "
                "follow-up PR); use --query-mode raw for now"
            )
        if ns.instance_ids_file and not ns.instance_ids:
            with open(ns.instance_ids_file) as f:
                ns.instance_ids = [
                    s.strip() for s in f.read().split() if s.strip()
                ]
        # CR#2 — reject empty id sets: `--instance-ids ""` and empty files
        # both reach this point with an empty list and would otherwise
        # produce a no-op run that still spins up + tears down the cluster.
        if not ns.instance_ids:
            p.error(
                "--instance-ids resolved to an empty list "
                "(no ids parsed from `--instance-ids` or `--instance-ids-file`)"
            )

    return ns


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    if ns.subcommand == "submit":
        run_id = driver.submit(ns)
        print(f"submitted: {run_id}")
        return 0
    if ns.subcommand == "fetch":
        driver.fetch(ns.run_id)
        return 0
    if ns.subcommand == "kill":
        driver.kill(ns.run_id)
        return 0
    if ns.subcommand == "resubmit":
        driver.resubmit(ns.run_id)
        return 0
    if ns.subcommand == "list":
        for row in driver.list_runs():
            print(
                f"{row['run_id']}  {row['framework']}/{row['query_mode']}/"
                f"{row['mode']}  {row['status']}  {row['done']}/{row['total']}"
            )
        return 0
    if ns.subcommand == "build":
        from bird_interact_agents import paths
        from bird_interact_agents.cloud import image

        repo_root = driver.submitter_repo_root()
        tag = image.image_tag(
            repo_root,
            paths.mini_interact_root(),
            allow_dirty=False,
        )
        uri = image.build_and_push(tag, repo_root, force=ns.force)
        print(uri)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
