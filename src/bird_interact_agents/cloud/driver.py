"""Laptop-side driver: submit, fetch, kill, list, resubmit, build."""

from __future__ import annotations

import datetime
import logging
import secrets
import signal
import sys
import time
from pathlib import Path
from typing import Any

from bird_interact_agents import paths
from bird_interact_agents.cloud import cluster, config, gcs, image, prereqs
from bird_interact_agents.cloud import collation as _collation


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local paths (overridable in tests)
# ---------------------------------------------------------------------------


def yaml_cache_dir() -> Path:
    return Path.home() / ".bird-interact-cloud"


def local_results_root() -> Path:
    return paths.results_root() / "cloud"


def default_gcs_client():
    return gcs.default_gcs_client()


def submitter_repo_root() -> Path:
    """The repo tree whose code we should hash + build + push.

    `paths.main_checkout_root()` returns the *canonical* checkout shared
    across worktrees — fine for results sinks, but wrong for cloud
    builds: when the user invokes `bird-interact-cloud` from a worktree,
    we want to ship the worktree's branch, not the main checkout's HEAD.

    Resolve via `git rev-parse --show-toplevel` against the cwd; fall
    back to `main_checkout_root()` when not in a git tree (e.g. installed
    wheel)."""
    import subprocess as _sp
    try:
        res = _sp.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip())
    except (FileNotFoundError, OSError):
        pass
    return paths.main_checkout_root()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mint_run_id(framework: str, query_mode: str) -> str:
    """Mint a run-id that's also a valid GCE label value AND instance-name
    fragment: lowercase letters / digits / `-` only. The timestamp uses
    `t` (lowercase) as the date/time separator because uppercase `T` is
    forbidden in GCE label values (`The value can only contain lowercase
    letters, numeric characters, underscores and dashes`)."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M")
    slug = framework.replace("_", "").lower()
    return f"{ts}-{slug}-{query_mode.lower()}-{secrets.token_hex(3)}"


def read_api_keys_from_local_env(agent_model: str, user_sim_model: str) -> dict[str, str]:
    import os

    needed: set[str] = set()
    for model in (agent_model, user_sim_model):
        needed.update(prereqs._required_api_keys(model))
    return {k: os.environ[k] for k in needed if k in os.environ}


def build_manifest(args, *, image_uri: str, run_id: str) -> dict:
    """Build the manifest dict from a SubmitArgs-like object."""
    instance_ids = list(args.instance_ids)
    return {
        "run_id": run_id,
        "framework": args.framework,
        "mode": args.mode,
        "query_mode": args.query_mode,
        "agent_model": args.agent_model,
        "user_sim_model": args.user_sim_model,
        "instance_ids": instance_ids,
        "patience": args.patience,
        "strict": bool(args.strict),
        "use_audited_gold_sql": bool(args.use_audited_gold_sql),
        "max_depth": args.max_depth,
        "prompt_cache": bool(args.prompt_cache),
        "render_inputs": {
            "workers": args.workers,
            "actors_per_worker": args.actors_per_worker,
            "worker_type": args.worker_type,
            "zone": cluster.DEFAULT_ZONE,
            "worker_sa": cluster.DEFAULT_WORKER_SA,
            "max_runtime_hours": args.max_runtime_hours,
            "image_uri": image_uri,
            "project": config.PROJECT,
            "region": config.REGION,
        },
    }


# ---------------------------------------------------------------------------
# Signal handlers / teardown
# ---------------------------------------------------------------------------


class _Handler:
    def __init__(self, *, run_id: str, yaml_path: Path):
        self.run_id = run_id
        self.yaml_path = yaml_path
        self._torn_down = False
        self._sigint_count = 0

    def teardown(self, *, reason: str) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        try:
            cluster.down(self.yaml_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("teardown ray down failed: %s (reason=%s)", e, reason)
            try:
                cluster.fallback_delete_by_label(self.run_id)
            except Exception:  # noqa: BLE001
                pass

    def _on_sigint(self, _signum, _frame):
        self._sigint_count += 1
        if self._sigint_count == 1:
            sys.stderr.write(
                "tearing down cluster; Ctrl-C again to detach.\n"
            )
            self.teardown(reason="sigint-1")
            raise SystemExit(130)
        sys.stderr.write(
            "abandoning teardown; VM self-delete timer will clean up.\n"
        )
        raise SystemExit(130)


def install_signal_handlers(*, run_id: str, yaml_path: Path) -> _Handler:
    h = _Handler(run_id=run_id, yaml_path=yaml_path)
    signal.signal(signal.SIGINT, h._on_sigint)
    signal.signal(signal.SIGTERM, h._on_sigint)
    return h


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class WaitResult:
    def __init__(self, *, terminal_state: str, hint: str = ""):
        self.terminal_state = terminal_state
        self.hint = hint


def submit(args) -> str:
    prereqs.check(args)
    repo_root = submitter_repo_root()
    tag = image.image_tag(
        repo_root,
        paths.mini_interact_root(),
        allow_dirty=args.allow_dirty,
    )
    image_uri = image.build_and_push(
        tag, repo_root,
        mini_interact_root=paths.mini_interact_root(),
        force=False,
    )
    run_id = args.run_id or mint_run_id(args.framework, args.query_mode)
    manifest = build_manifest(args, image_uri=image_uri, run_id=run_id)
    gcs.write_manifest(run_id, manifest)
    yaml_path = cluster.render_from_manifest(manifest, cache_dir=yaml_cache_dir())
    h = install_signal_handlers(run_id=run_id, yaml_path=yaml_path)
    submit_succeeded = False
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        env_vars = read_api_keys_from_local_env(args.agent_model, args.user_sim_model)
        job_args = _build_job_args(args, run_id, attempt=1)
        ray_job_id = cluster.submit_job(
            head_address=head, args=job_args, env_vars=env_vars,
            yaml_path=yaml_path,
        )
        submit_succeeded = True
        gcs.write_status(run_id, {
            "ray_job_id": ray_job_id,
            "last_heartbeat_ts": time.time(),
            "rows_done": 0,
            "rows_total": len(args.instance_ids),
            "terminal_state": None,
            "attempt": 1,
        })
        if not args.detach:
            wait_until_done(run_id, manifest)
            fetch(run_id)
    finally:
        # Detach skips teardown ONLY on successful submit — if we never
        # got the Ray job in flight, the cluster is just burning money
        # without anything running. (CR#4 — without this, `--detach` +
        # `submit_job` failure orphans the cluster.)
        if (not args.detach) or (not submit_succeeded):
            h.teardown(reason="finally")
    return run_id


def _build_job_args(args, run_id: str, *, attempt: int) -> list[str]:
    job_args = [
        "--run-id", run_id,
        "--attempt", str(attempt),
        "--framework", args.framework,
        "--query-mode", args.query_mode,
        "--mode", args.mode,
        "--agent-model", args.agent_model,
        "--user-sim-model", args.user_sim_model,
        "--patience", str(args.patience),
        "--max-depth", str(args.max_depth),
        "--num-actors", str(args.workers * args.actors_per_worker),
        "--instance-ids", ",".join(args.instance_ids),
    ]
    if args.strict:
        job_args.append("--strict")
    if args.use_audited_gold_sql:
        job_args.append("--use-audited-gold-sql")
    if args.prompt_cache:
        job_args.append("--prompt-cache")
    else:
        job_args.append("--no-prompt-cache")
    return job_args


# ---------------------------------------------------------------------------
# wait_until_done
# ---------------------------------------------------------------------------


HEARTBEAT_STALL_SECONDS = 300.0


def wait_until_done(run_id: str, manifest: dict, *,
                     poll_interval_s: float = 10.0,
                     no_progress_deadline_s: float = 3600.0) -> WaitResult:
    total = len(manifest.get("instance_ids", []))
    # `no_progress_deadline_s` is a NO-PROGRESS deadline, not a wall-clock
    # cap: it resets every time a new row lands. A wedged job (workers never
    # autoscale, actor sits PENDING, 0 rows) trips it; a healthy long run —
    # `max_runtime_hours` can be many hours — keeps resetting it and runs to
    # completion. (The VM self-delete timer + the headless check below bound
    # the absolute runtime.)
    last_progress = -1
    last_progress_ts = time.time()
    while True:
        status = gcs.read_status(run_id) or {}
        attempts = gcs.list_attempts(run_id)
        terminal = status.get("terminal_state")
        if terminal in ("done", "error"):
            return WaitResult(terminal_state=terminal)
        if len(attempts) >= total and total > 0:
            return WaitResult(terminal_state="done")
        # Progress = number of iids with at least one attempt row in GCS.
        if len(attempts) > last_progress:
            last_progress = len(attempts)
            last_progress_ts = time.time()
        last = status.get("last_heartbeat_ts")
        if last is not None and (time.time() - float(last)) > HEARTBEAT_STALL_SECONDS:
            if cluster.head_is_alive(run_id):
                return WaitResult(
                    terminal_state="stalled",
                    hint="job appears stalled; fetch + resubmit",
                )
        if not cluster.head_is_alive(run_id) and total > 0:
            return WaitResult(
                terminal_state="headless",
                hint="head node is gone; fetch + resubmit",
            )
        # No-progress deadline: even with a fresh heartbeat, if NO new rows
        # have landed for this long the job is wedged (e.g. workers never
        # autoscaled up, so actors sit PENDING forever while the heartbeat
        # keeps writing `terminal_state: null`). Resets on every new row, so
        # a slow-but-progressing run is never falsely timed out.
        if (time.time() - last_progress_ts) > no_progress_deadline_s:
            return WaitResult(
                terminal_state="timed-out",
                hint=(
                    "no new rows within the no-progress deadline; check "
                    "`ray status` on the head (workers may not have scaled "
                    "up); resubmit"
                ),
            )
        if poll_interval_s <= 0:
            return WaitResult(terminal_state="poll-once-exit")
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# fetch / kill / list / resubmit
# ---------------------------------------------------------------------------


def fetch(run_id: str) -> dict:
    dest = local_results_root() / run_id
    gcs.concurrent_download_prefix(run_id, dest, client=default_gcs_client())
    manifest_path = dest / "manifest.json"
    if not manifest_path.exists():
        import json
        manifest = gcs.read_manifest(run_id, client=default_gcs_client())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        import json
        manifest = json.loads(manifest_path.read_text())
    metrics = _collation.collate(dest, manifest)
    return metrics


def kill(run_id: str) -> None:
    cache = yaml_cache_dir()
    yaml_path = cache / f"{run_id}.yaml"
    if not yaml_path.exists():
        manifest = gcs.read_manifest(run_id)
        yaml_path = cluster.render_from_manifest(manifest, cache_dir=cache)
    try:
        cluster.down(yaml_path)
    except Exception:  # noqa: BLE001
        cluster.fallback_delete_by_label(run_id)


def list_runs() -> list[dict]:
    client = default_gcs_client()
    bucket = client.bucket(gcs.BUCKET_NAME)
    seen: set[str] = set()
    for blob in bucket.list_blobs(prefix="runs/"):
        parts = blob.name.split("/", 2)
        if len(parts) >= 2:
            seen.add(parts[1])
    out: list[dict] = []
    for rid in sorted(seen):
        try:
            mf = gcs.read_manifest(rid, client=client)
        except Exception:  # noqa: BLE001
            continue
        attempts = gcs.list_attempts(rid, client=client)
        status = "live" if cluster.head_is_alive(rid) else "done"
        out.append({
            "run_id": rid,
            "framework": mf.get("framework"),
            "mode": mf.get("mode"),
            "query_mode": mf.get("query_mode"),
            "status": status,
            "done": len(attempts),
            "total": len(mf.get("instance_ids", [])),
        })
    return out


def resubmit(run_id: str) -> None:
    manifest = gcs.read_manifest(run_id)
    attempts = gcs.list_attempts(run_id)
    done: set[str] = set()
    max_attempt = 0
    for iid, lst in attempts.items():
        if not lst:
            continue
        max_attempt = max(max_attempt, max(lst))
        latest_n = lst[-1]
        try:
            row = gcs.read_row(run_id, iid, latest_n)
        except Exception:  # noqa: BLE001
            continue
        if not row.get("error"):
            done.add(iid)
    missing = [iid for iid in manifest.get("instance_ids", []) if iid not in done]
    if not missing:
        return
    next_attempt = max_attempt + 1
    yaml_path = cluster.render_from_manifest(manifest, cache_dir=yaml_cache_dir())
    h = install_signal_handlers(run_id=run_id, yaml_path=yaml_path)
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        env_vars = read_api_keys_from_local_env(
            manifest["agent_model"], manifest["user_sim_model"],
        )
        job_args = _build_resubmit_args(manifest, run_id, missing, next_attempt)
        cluster.submit_job(
            head_address=head, args=job_args, env_vars=env_vars,
            yaml_path=yaml_path,
        )
        wait_until_done(run_id, manifest)
        fetch(run_id)
    finally:
        h.teardown(reason="resubmit-finally")


def _build_resubmit_args(manifest: dict, run_id: str, missing: list[str],
                          attempt: int) -> list[str]:
    job_args = [
        "--run-id", run_id,
        "--attempt", str(attempt),
        "--framework", manifest["framework"],
        "--query-mode", manifest["query_mode"],
        "--mode", manifest["mode"],
        "--agent-model", manifest["agent_model"],
        "--user-sim-model", manifest["user_sim_model"],
        "--patience", str(manifest.get("patience", 3)),
        "--max-depth", str(manifest.get("max_depth", 3)),
        "--num-actors", str(
            manifest["render_inputs"]["workers"]
            * manifest["render_inputs"]["actors_per_worker"]
        ),
        "--instance-ids", ",".join(missing),
    ]
    if manifest.get("strict"):
        job_args.append("--strict")
    if manifest.get("use_audited_gold_sql"):
        job_args.append("--use-audited-gold-sql")
    if manifest.get("prompt_cache", True):
        job_args.append("--prompt-cache")
    else:
        job_args.append("--no-prompt-cache")
    return job_args
