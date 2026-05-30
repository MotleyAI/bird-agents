"""`bird-interact-cloud` CLI."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from bird_interact_agents.benchmark import cli_dataset_tokens, get_benchmark
from bird_interact_agents.cloud import driver


_VALID_MODES = ("a-interact", "c-interact", "oracle", "one-shot")


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
    sp_submit.add_argument(
        "--dataset", choices=cli_dataset_tokens(), default="mini_interact",
        help=(
            "Which benchmark to run (registry token; `mini-interact` accepted "
            "as an alias). `livesqlbench` REQUIRES --gold-file and --mode "
            "{one-shot, oracle}."
        ),
    )
    sp_submit.add_argument(
        "--gold-file", default=None,
        help=(
            "Path to the gated gold sidecar for benchmarks whose data JSONL "
            "ships gold empty (e.g. livesqlbench). MUST live under the "
            "benchmark data root so it rides along in the GCS dataset upload; "
            "merged by instance_id at task load in-cluster."
        ),
    )
    sp_submit.add_argument("--patience", type=int, default=500)
    sp_submit.add_argument("--strict", action="store_true")
    sp_submit.add_argument(
        "--use-audited-gold-sql", action=argparse.BooleanOptionalAction,
        default=True,
    )
    sp_submit.add_argument(
        "--require-audited-gold", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --use-audited-gold-sql is on, fail at submit time if any "
            "passed instance_id lacks an audited-gold entry (audit_status "
            "missing-row / missing-file). Default on — prevents silent "
            "fall-back to the un-audited gold. Pass --no-require-audited-gold "
            "to allow the fallback."
        ),
    )
    sp_submit.add_argument("--max-depth", type=int, default=3)
    sp_submit.add_argument(
        "--reasoning-effort", dest="reasoning_effort", default=None,
        choices=("low", "medium", "high", "max"),
        help="Reasoning-effort level for claude_sdk_otf and "
             "claude_sdk_otf_ainteract (ClaudeAgentOptions.effort). Ignored "
             "by other frameworks; unset = SDK default.",
    )
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
    sp_submit.add_argument(
        "--slayer-setup", choices=("pre-encoded", "on-the-fly"),
        default="pre-encoded",
    )
    sp_submit.add_argument(
        "--slayer-storage-root", default="/data/slayer_models",
    )

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
        # Cloud slayer mode mirrors local: validate the
        # (framework, query_mode, mode, slayer_setup) tuple via the SAME guard
        # the local CLI uses, so an unsupported combo fails fast at submit —
        # before building/pushing the image or bringing up a cluster — rather
        # than per-task mid-run. (run._validate_slayer_setup raises ValueError;
        # translate to the standard argparse exit-2 + stderr.)
        # Normalize the benchmark token to canonical (e.g. mini-interact alias
        # → mini_interact) before any benchmark-keyed logic.
        ns.dataset = get_benchmark(ns.dataset).name
        from bird_interact_agents.run import (
            _validate_dataset_mode,
            _validate_framework_dataset_mode,
            _validate_one_shot_framework,
            _validate_slayer_setup,
        )
        try:
            # Same dataset⟺mode⟺framework gates the local CLI uses — one source
            # of truth, so an unsupported cloud combo fails fast at submit.
            _validate_dataset_mode(dataset=ns.dataset, mode=ns.mode)
            _validate_framework_dataset_mode(
                framework=ns.framework, dataset=ns.dataset, mode=ns.mode,
            )
            _validate_one_shot_framework(
                mode=ns.mode, query_mode=ns.query_mode, framework=ns.framework,
            )
            _validate_slayer_setup(
                slayer_setup=ns.slayer_setup, framework=ns.framework,
                query_mode=ns.query_mode, mode=ns.mode,
            )
            if get_benchmark(ns.dataset).gold_required and not ns.gold_file:
                raise ValueError(
                    f"--dataset {ns.dataset} requires --gold-file (the gated "
                    "gold sidecar).",
                )
        except ValueError as e:
            p.error(str(e))
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
        # DEV-1478 follow-up: when `use_audited_gold_sql` is on AND the
        # default `require_audited_gold` guard is on, fail at submit if any
        # passed instance_id has no audited-gold row. Default ON because
        # silently falling back to the un-audited gold mid-cloud-run is a
        # foot-gun: the cluster comes up, encodes (10+ min), and only THEN
        # surfaces the missing-audit via a warning in the actor log. Fail
        # before bringing up the cluster instead.
        # The audited-gold overlay is a mini-interact concept (its data JSONL
        # carries gold inline + a separate audited_gold/ sidecar). Benchmarks
        # with their own gated gold sidecar (gold_required, e.g. livesqlbench)
        # have no audited_gold — skip the check rather than report every id
        # missing.
        if (
            ns.use_audited_gold_sql
            and ns.require_audited_gold
            and not get_benchmark(ns.dataset).gold_required
        ):
            from bird_interact_agents.cloud._audited_gold_check import (
                missing_audited_gold_ids,
            )
            missing = missing_audited_gold_ids(ns.instance_ids)
            if missing:
                p.error(
                    "the following instance_ids have no audited-gold entry "
                    "(audit_status missing-row / missing-file): "
                    f"{', '.join(missing)}. Either remove them, pass "
                    "--no-require-audited-gold to allow the harness fallback "
                    "to the original gold, or pass --no-use-audited-gold-sql "
                    "to evaluate against the original gold for all tasks."
                )

    return ns


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    if ns.subcommand == "submit":
        run_id = driver.submit(ns)
        print(f"submitted: {run_id}")
        return 0
    if ns.subcommand == "fetch":
        metrics = driver.fetch(ns.run_id)
        # Codex r6: surface the merge report so post-run merge failures
        # (ignored shards, skipped dbs) are visible at fetch time rather
        # than buried in the on-disk merge_report.json.
        merge_report = metrics.get("merge_report") or {}
        merged_dbs = merge_report.get("merged_dbs") or []
        skipped_dbs = merge_report.get("skipped_dbs") or []
        ignored_shards = merge_report.get("ignored_shards") or []
        if merged_dbs or skipped_dbs or ignored_shards:
            for entry in merged_dbs:
                print(
                    f"merged slayer_models_otf/{entry['db']}: "
                    f"{entry['files_updated']} files updated, "
                    f"{entry['files_skipped']} skipped"
                )
            for entry in skipped_dbs:
                # Codex r7: surface dbs that existed in shards but couldn't
                # be merged (e.g. one actor uploaded fully but a peer died
                # mid-upload for this db, and no other shard covers it).
                print(
                    f"SKIPPED slayer_models_otf/{entry['db']}: "
                    f"{entry['reason']}"
                )
            if ignored_shards:
                print(f"ignored {len(ignored_shards)} incomplete shard(s): "
                      f"{', '.join(ignored_shards)}")
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
            paths.audited_gold_root(),
            allow_dirty=False,
        )
        uri = image.build_and_push(
            tag, repo_root,
            audited_gold_root=paths.audited_gold_root(),
            force=ns.force,
        )
        print(uri)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
