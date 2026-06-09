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
    sp_submit.add_argument("--framework", required=True, choices=["claude_sdk"])
    sp_submit.add_argument("--query-mode", required=True, choices=("raw", "slayer"))
    sp_submit.add_argument("--agent-model", required=True)
    sp_submit.add_argument(
        "--user-sim-model", default="anthropic/claude-sonnet-4-6"
    )
    grp = sp_submit.add_mutually_exclusive_group(required=True)
    grp.add_argument("--instance-ids", type=_instance_ids)
    grp.add_argument("--instance-ids-file", type=str)
    sp_submit.add_argument("--mode", required=True, choices=_VALID_MODES)
    sp_submit.add_argument(
        "--dataset", choices=cli_dataset_tokens(), required=True,
        help=(
            "Which benchmark to run (registry token; `mini-interact` accepted "
            "as an alias). REQUIRED — no default, to prevent silently running "
            "mini-interact when --mode/--instance-ids happen to be consistent "
            "with both."
        ),
    )
    sp_submit.add_argument("--patience", type=int, default=250)
    sp_submit.add_argument("--strict", action="store_true")
    sp_submit.add_argument(
        "--use-audited-gold-sql", action=argparse.BooleanOptionalAction,
        default=True,
    )
    sp_submit.add_argument(
        "--require-annotation", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail at submit time if any passed instance_id lacks a task "
            "annotation file at "
            "<annotations_root>/<benchmark>/<db>/<iid>.task.json. Default on "
            "— the annotation is the authoritative source for grading "
            "(gold_variants, evaluator_prompt, masked_terms, "
            "original_gold_is_correct); without it the harness silently "
            "loses access to alternate-reading grading, the LLM judge for "
            "insufficient tasks, and the audited-gold overlay decision. "
            "Pass --no-require-annotation for smoke tests on un-annotated "
            "ids."
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
    sp_submit.add_argument(
        "--subscription-auth", action=argparse.BooleanOptionalAction,
        required=True, dest="subscription_auth",
        help=(
            "REQUIRED. `--subscription-auth` uses the Claude.ai OAuth "
            "subscription (CLAUDE_CODE_OAUTH_TOKEN must be set in the "
            "submitter's env, with the sk-ant-oat01- prefix); submit fails "
            "if the token is absent or malformed. `--no-subscription-auth` "
            "uses the legacy API-key path (ANTHROPIC_API_KEY). No default — "
            "an explicit choice is required to prevent a silent fall-back "
            "to the API-key path burning credits when the operator meant "
            "to hit the subscription."
        ),
    )

    sp_annotate = sub.add_parser("annotate")
    sp_annotate.add_argument("--benchmark", required=True,
                              help="Benchmark to annotate (registry token).")
    sp_annotate.add_argument("--agent-model", required=True,
                              help="Model for the annotator agent.")
    sp_annotate.add_argument(
        "--effort", default="medium", choices=("low", "medium", "high"),
        help="Reasoning effort for the annotator agent.",
    )
    grp_ann = sp_annotate.add_mutually_exclusive_group(required=True)
    grp_ann.add_argument("--instance-ids", type=_instance_ids)
    grp_ann.add_argument("--instance-ids-file", type=str)
    sp_annotate.add_argument("--override", action="store_true",
                              help="Re-annotate even when stable blobs already exist.")
    sp_annotate.add_argument("--workers", type=int, default=4)
    sp_annotate.add_argument("--actors-per-worker", type=int, default=4)
    sp_annotate.add_argument("--worker-type", default="e2-standard-4")
    sp_annotate.add_argument("--max-runtime-hours", type=int, default=8)
    sp_annotate.add_argument("--run-id", default=None)
    sp_annotate.add_argument("--detach", action="store_true")
    sp_annotate.add_argument("--allow-dirty", action="store_true")
    sp_annotate.add_argument(
        "--subscription-auth", action=argparse.BooleanOptionalAction,
        required=True, dest="subscription_auth",
        help=(
            "REQUIRED. Same shape as `submit`: `--subscription-auth` requires "
            "a valid CLAUDE_CODE_OAUTH_TOKEN in the env, "
            "`--no-subscription-auth` uses ANTHROPIC_API_KEY. No default."
        ),
    )

    sp_fetch = sub.add_parser("fetch")
    sp_fetch.add_argument("run_id")
    sp_fetch.add_argument(
        "--no-kill", action="store_true",
        help="Do not shut down the cluster after a successful fetch.",
    )
    for name in ("kill", "resubmit"):
        spx = sub.add_parser(name)
        spx.add_argument("run_id")

    sub.add_parser("list")
    sp_build = sub.add_parser("build")
    sp_build.add_argument("--force", action="store_true")

    ns = p.parse_args(argv)

    # `--subscription-auth` / `--no-subscription-auth` is required on submit
    # AND annotate (above). Mirror the bool into the legacy `no_subscription_auth`
    # attribute that the driver/prereqs/manifest plumbing still reads — keeps the
    # rename surface limited to the CLI surface. If `--subscription-auth` was
    # chosen, validate the OAuth token at parse time so the operator sees the
    # failure before the cluster comes up. (The driver re-validates as
    # defence-in-depth; see `read_api_keys_from_local_env`.)
    if ns.subcommand in ("submit", "annotate"):
        import os
        ns.no_subscription_auth = not ns.subscription_auth
        if ns.subscription_auth:
            token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
            if not token:
                p.error(
                    "--subscription-auth requires CLAUDE_CODE_OAUTH_TOKEN to "
                    "be set in the submitter's env. Either source the env "
                    "file that exports it (e.g. `set -a; source .env.ubuntu; "
                    "set +a`), run `claude setup-token`, or pass "
                    "`--no-subscription-auth` to use the ANTHROPIC_API_KEY "
                    "path."
                )
            if not token.startswith("sk-ant-oat01-"):
                p.error(
                    "CLAUDE_CODE_OAUTH_TOKEN does not look like a Claude.ai "
                    "OAuth token (expected sk-ant-oat01- prefix). Re-run "
                    "`claude setup-token`."
                )

    if ns.subcommand == "annotate":
        if ns.detach and ns.allow_dirty:
            p.error("--detach and --allow-dirty are mutually exclusive")
        ns.benchmark = get_benchmark(ns.benchmark).name
        if ns.instance_ids_file and not ns.instance_ids:
            with open(ns.instance_ids_file) as f:
                ns.instance_ids = [
                    s.strip() for s in f.read().split() if s.strip()
                ]
        if not ns.instance_ids:
            p.error(
                "--instance-ids resolved to an empty list "
                "(no ids parsed from `--instance-ids` or `--instance-ids-file`)"
            )

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
            _validate_framework_mode,
            _validate_slayer_setup,
        )
        try:
            # Same dataset⟺mode gates the local CLI uses — one source
            # of truth, so an unsupported cloud combo fails fast at submit.
            _validate_dataset_mode(dataset=ns.dataset, mode=ns.mode)
            _validate_framework_mode(
                framework=ns.framework, dataset=ns.dataset, mode=ns.mode,
            )
            _validate_slayer_setup(
                slayer_setup=ns.slayer_setup, framework=ns.framework,
                query_mode=ns.query_mode, mode=ns.mode,
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
        # DEV-1515 follow-up: require every passed instance_id to have a
        # task annotation file. The annotation is the authoritative source
        # for grading (gold_variants, evaluator_prompt, masked_terms,
        # original_gold_is_correct). Default ON because a missing annotation
        # silently degrades grading mid-cloud-run: no alternate-reading
        # match, no LLM judge for `insufficient` tasks, no audited-gold
        # overlay decision. Fail at submit, before the 10+ min cluster
        # warm-up. Decoupled from `--use-audited-gold-sql` because the
        # annotation is needed for grading regardless of overlay use.
        if ns.require_annotation:
            from bird_interact_agents.cloud._annotation_check import (
                missing_annotation_ids,
            )
            missing = missing_annotation_ids(
                ns.instance_ids,
                benchmark=get_benchmark(ns.dataset),
            )
            if missing:
                p.error(
                    "the following instance_ids have no task annotation "
                    "(annotations/<benchmark>/<db>/<iid>.task.json absent): "
                    f"{', '.join(missing)}. Either remove them, annotate "
                    "them first (`bird-interact-cloud annotate`), or pass "
                    "--no-require-annotation to bypass for smoke runs."
                )

    return ns


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    if ns.subcommand == "annotate":
        run_id = driver.submit_annotator(ns)
        print(f"submitted: {run_id}")
        return 0
    if ns.subcommand == "submit":
        run_id = driver.submit(ns)
        print(f"submitted: {run_id}")
        return 0
    if ns.subcommand == "fetch":
        metrics = driver.fetch(ns.run_id, kill_after_fetch=not ns.no_kill)
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
        for report_key, label in [
            ("task_annotation_merge_report", "task annotation merge"),
            ("audited_gold_variants_merge_report", "audited gold merge"),
        ]:
            report = metrics.get(report_key) or {}
            n_errors = report.get("errors", 0)
            if n_errors:
                print(f"WARNING: {label}: {n_errors} error(s)")
                for detail in report.get("error_details", []):
                    print(f"  {detail}")
        if kill_err := metrics.get("kill_after_fetch_error"):
            print(f"WARNING: auto-kill failed — cluster may still be running: {kill_err}")
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
            annotations_root=paths.annotations_root(),
        )
        uri = image.build_and_push(
            tag, repo_root,
            audited_gold_root=paths.audited_gold_root(),
            annotations_root=paths.annotations_root(),
            force=ns.force,
        )
        print(uri)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
