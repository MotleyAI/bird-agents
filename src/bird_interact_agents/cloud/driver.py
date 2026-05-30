"""Laptop-side driver: submit, fetch, kill, list, resubmit, build."""

from __future__ import annotations

import asyncio
import datetime
import logging
import secrets
import signal
import sys
import time
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.cloud import benchmark_data, cluster, config, gcs, image, prereqs
from bird_interact_agents.cloud import collation as _collation
from bird_interact_agents.cloud import post_run_merge as _post_run_merge
# Imported by NAME (not via the `gcs` module attr) so tests that mock
# `driver.gcs` still get the real pure mapping — only the I/O helpers
# (`gcs.upload_dir_prefix` etc.) need to be mockable.
from bird_interact_agents.cloud.gcs import slayer_artifact_name
# Imported by NAME so they survive tests that mock `driver.prereqs`. PrereqError
# must be the real class for the raise in `read_api_keys_from_local_env`, and
# `_required_api_keys` must be the REAL provider→key mapping so submit/resubmit
# tests that mock `driver.prereqs` still exercise the actual key selection
# (CodeRabbit) — otherwise the mock returns an empty MagicMock-iterable and
# key-selection silently no-ops in those tests.
from bird_interact_agents.cloud.prereqs import PrereqError, _required_api_keys
# Imported by NAME so `_build_missing_otf_caches` can be exercised with a mock
# (`monkeypatch.setattr(driver, "ensure_db_cache", ...)`) without a real build.
from bird_interact_agents.slayer_otf.cache import ensure_db_cache


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


# GCP's compute-instance-name regex caps the FULL name at 63 chars
# (`[a-z]([-a-z0-9]{0,61}[a-z0-9])?` → 1+61+1 = 63). Ray composes the name as
# `ray-<run_id>-worker-<uuid8>-compute` (worst case: "worker" 6 > "head" 4;
# "compute" 7 > "tpu" 3), so 4 + len(run_id) + 7 + 9 + 8 = 28 + len(run_id).
# Ray's INTERNAL assertion (`name_label <= 55`, INSTANCE_NAME_MAX_LEN-1-UUID)
# is off-by-one and looser than GCP's real 63 — observed: Ray accepted a
# 50-char label but GCP rejected the resulting 66-char name. So compute the
# slug cap from GCP's true 63 limit, not Ray's 55.
_INSTANCE_NAME_MAX = 63
_NAME_OVERHEAD = 28  # ray- + -worker + -<uuid8> + -compute


def mint_run_id(framework: str, query_mode: str) -> str:
    """Mint a run-id that's also a valid GCE label value AND instance-name
    fragment: lowercase letters / digits / `-` only. The timestamp uses
    `t` (lowercase) as the date/time separator because uppercase `T` is
    forbidden in GCE label values (`The value can only contain lowercase
    letters, numeric characters, underscores and dashes`). The framework slug
    is length-capped so Ray's `ray-<run_id>-worker-<uuid8>-compute` GCE name
    fits GCP's 63-char compute-instance-name limit."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M")
    qm = query_mode.lower()
    token = secrets.token_hex(3)
    # run_id format: <ts>-<slug>-<qm>-<token> (3 dashes). Cap the slug so
    # the full GCE instance name fits 63.
    slug_max = max(
        1, _INSTANCE_NAME_MAX - _NAME_OVERHEAD - len(ts) - len(qm) - len(token) - 3
    )
    slug = framework.replace("_", "").lower()[:slug_max]
    return f"{ts}-{slug}-{qm}-{token}"


def read_api_keys_from_local_env(
    agent_model: str, user_sim_model: str, *, query_mode: str = "raw",
) -> dict[str, str]:
    import os

    needed: set[str] = set()
    for model in (agent_model, user_sim_model):
        needed.update(_required_api_keys(model))
    # DEV-1468: slayer mode needs OPENAI_API_KEY for channel-3 embeddings,
    # regardless of the agent/user-sim providers.
    if query_mode == "slayer":
        needed.add("OPENAI_API_KEY")
    # Fail fast on a missing required key instead of silently dropping it —
    # `resubmit` does NOT run prereq checks, so an absent key would otherwise
    # surface much later as an opaque per-actor auth failure (CodeRabbit).
    missing = [k for k in sorted(needed) if not os.environ.get(k)]
    if missing:
        cmds = "\n".join(f"export {k}=<your-key>" for k in missing)
        raise PrereqError(
            f"missing API key env vars for job submission: {missing}",
            remediation=cmds,
        )
    return {k: os.environ[k] for k in needed}


def build_manifest(
    args, *, image_uri: str, run_id: str, benchmark_data_prefix: str | None = None,
) -> dict:
    """Build the manifest dict from a SubmitArgs-like object.

    ``gold_file`` is stored as the IN-CLUSTER path (under the benchmark's
    ``container_data_dir``), not the submitter's local path, so ``resubmit``
    on any machine re-derives identical job args. ``benchmark_data_prefix`` is
    the content-hashed GCS prefix the dataset was uploaded to at submit; the
    actor downloads from it per node."""
    instance_ids = list(args.instance_ids)
    return {
        "run_id": run_id,
        "framework": args.framework,
        "mode": args.mode,
        "dataset": _submit_benchmark(args),
        "gold_file": _in_cluster_gold_file(args),
        "benchmark_data_prefix": benchmark_data_prefix,
        "query_mode": args.query_mode,
        "agent_model": args.agent_model,
        "user_sim_model": args.user_sim_model,
        "instance_ids": instance_ids,
        "patience": args.patience,
        "strict": bool(args.strict),
        "use_audited_gold_sql": bool(args.use_audited_gold_sql),
        "max_depth": args.max_depth,
        "prompt_cache": bool(args.prompt_cache),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "slayer_setup": getattr(args, "slayer_setup", "pre-encoded"),
        "slayer_storage_root": getattr(
            args, "slayer_storage_root", "/data/slayer_models"
        ),
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
# SLayer setup delivery (DEV-1468)
# ---------------------------------------------------------------------------


def _dbs_for_instances(instance_ids, benchmark: str = "mini_interact") -> list[str]:
    """Map the selected instance_ids to their distinct ``selected_database``
    via the benchmark's tasks file (never string-split the id — DB names contain
    underscores, e.g. ``california_schools``). Returns a sorted, de-duplicated
    db list."""
    import json as _json

    wanted = set(instance_ids)
    dbs: set[str] = set()
    with paths.benchmark_data_file(benchmark).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            td = _json.loads(line)
            if td.get("instance_id") in wanted:
                db = td.get("selected_database")
                if db:
                    dbs.add(db)
    return sorted(dbs)


def _benchmark_for_dataset(dataset: str | None) -> str:
    """Map a run's ``dataset`` token (canonical name, CLI alias, or marker) to
    its canonical benchmark name for the OTF path roots + container data dir.

    An absent dataset (e.g. a pre-de-bake manifest being resubmitted) falls
    back to ``"mini_interact"``; otherwise the token is resolved through the
    registry so a third benchmark never silently aliases to mini-interact.
    """
    if not dataset:
        return "mini_interact"
    return get_benchmark(dataset).name


def _submit_benchmark(args) -> str:
    """Benchmark for a submit, derived from ``args.dataset`` (defaults to
    mini-interact when the CLI didn't set one)."""
    return _benchmark_for_dataset(getattr(args, "dataset", None))


def _validate_gold_under_data_root(args) -> None:
    """The gated gold sidecar rides along inside the GCS dataset upload (it
    lives in the benchmark data root), so it MUST physically sit under that
    root. Fail fast at submit if ``--gold-file`` points elsewhere — otherwise
    it would be silently absent in-cluster and every task would fail to
    score. No-op when no gold file is given."""
    gold_file = getattr(args, "gold_file", None)
    if not gold_file:
        return
    data_root = paths.benchmark_data_root(_submit_benchmark(args)).resolve()
    gp = Path(gold_file).expanduser().resolve()
    if not gp.is_file():
        raise FileNotFoundError(f"--gold-file not found: {gp}")
    if not gp.is_relative_to(data_root):
        raise ValueError(
            f"--gold-file {gp} must live under the benchmark data root "
            f"{data_root} so it rides along in the GCS dataset upload; "
            f"move the gold sidecar into the data dir."
        )


def _in_cluster_gold_file(args) -> str | None:
    """Translate the local ``--gold-file`` to its in-cluster path.

    The gold rides along in the GCS dataset upload and the actor downloads the
    dataset into the benchmark's ``container_data_dir``, so the gold lands at
    ``container_data_dir/<path-relative-to-data-root>``. Falls back to the
    basename when the gold isn't under the data root (defensive — submit's
    :func:`_validate_gold_under_data_root` enforces the contract for real
    submits; direct ``build_manifest`` callers in tests may pass an arbitrary
    path)."""
    gold_file = getattr(args, "gold_file", None)
    if not gold_file:
        return None
    benchmark = _submit_benchmark(args)
    data_root = paths.benchmark_data_root(benchmark).resolve()
    gp = Path(gold_file).expanduser()
    try:
        rel = gp.resolve().relative_to(data_root)
    except ValueError:
        rel = Path(gp.name)
    return str(Path(get_benchmark(benchmark).container_data_dir) / rel)


def _slayer_uploads_for(args) -> list[tuple[Path, str, bool]]:
    """Return the local-dir → GCS-artifact-name uploads to ship for this
    combo, each tagged ``required`` (must exist + non-empty) or optional
    (uploaded only when present, as a "skip this encode" seed for the cloud).

    DEV-1470: for ``pydantic_ai_otf_encode + on-the-fly`` the deterministic
    cache (``slayer_otf_cache``) is REQUIRED (it's the input to the LLM-driven
    setup encoder, kept local because it's free to build); the per-DB
    reference (``slayer_models_otf``) is OPTIONAL — if present, uploaded as a
    seed so the cloud skips re-encoding that DB; if absent, the cloud encodes
    lazily on first task.
    """
    setup = args.slayer_setup
    fw = args.framework
    benchmark = _submit_benchmark(args)
    if setup == "pre-encoded":
        return [(submitter_repo_root() / "slayer_models", "slayer_models", True)]
    if fw == "pydantic_ai_otf_encode":
        return [
            (paths.slayer_otf_cache_root(benchmark=benchmark),
             "slayer_otf_cache", True),
            (paths.slayer_models_otf_root(benchmark=benchmark),
             "slayer_models_otf", False),
        ]
    # pydantic_ai_recursive + on-the-fly — cache only, no LLM-encoded reference.
    return [
        (paths.slayer_otf_cache_root(benchmark=benchmark),
         "slayer_otf_cache", True),
    ]


def _artifact_present(root: Path, db: str, artifact: str) -> bool:
    """Presence semantics per artifact: ``slayer_models`` = a NON-EMPTY
    committed dir (no marker); OTF layers = their completeness marker file is
    present."""
    db_dir = root / db
    if artifact == "slayer_models":
        return db_dir.is_dir() and any(db_dir.iterdir())
    marker = "_cache_fp.txt" if artifact == "slayer_otf_cache" else "_reference_fp.txt"
    return (db_dir / marker).is_file()


def _build_missing_otf_caches(
    cache_root: Path, dbs: list[str], benchmark: str = "mini_interact",
) -> None:
    """Build the deterministic OTF ingest cache locally for each DB in `dbs`.

    The cache is free to build (no LLMs) and fully deterministic, so a missing
    ``slayer_otf_cache`` is auto-created here — laptop-side, before upload —
    rather than aborting the submit. The cloud runner never builds REQUIRED
    setup in-cluster (it consumes the uploaded cache), so this is the only
    place it can be produced for the submit to proceed."""
    data_root = paths.benchmark_data_root(benchmark)
    for db in dbs:
        logger.info(
            "cloud slayer: deterministic cache missing for %s — building it "
            "locally (no LLMs) before upload", db,
        )
        asyncio.run(
            ensure_db_cache(
                db, cache_root=cache_root, mini_interact_root=data_root,
                force=False,
            )
        )


def _check_slayer_setup_present(args) -> list[str]:
    """Validate (BEFORE build/push/cluster) that every REQUIRED artifact is
    present locally for every selected DB. Optional artifacts (e.g. the
    ``slayer_models_otf`` seed under ``otf_encode + on-the-fly``) are not
    required — the cloud will encode missing references on the fly.

    A missing ``slayer_otf_cache`` (the deterministic ingest cache) is NOT a
    hard error: it's free to build locally, so we build it here instead of
    aborting. Other REQUIRED artifacts (e.g. the hand-authored ``slayer_models``
    reference) can't be auto-built and still fail fast.

    Returns the db list to upload."""
    benchmark = _submit_benchmark(args)
    dbs = _dbs_for_instances(args.instance_ids, benchmark)
    uploads = _slayer_uploads_for(args)
    for root, artifact, required in uploads:
        if not required:
            continue
        missing = [db for db in dbs if not _artifact_present(root, db, artifact)]
        if not missing:
            continue
        if artifact == "slayer_otf_cache":
            # Deterministic + LLM-free → build it rather than erroring.
            _build_missing_otf_caches(root, missing, benchmark)
            continue
        marker_hint = {
            "slayer_models": "non-empty <db>/ dir",
            "slayer_otf_cache": "_cache_fp.txt",
            "slayer_models_otf": "_reference_fp.txt",
        }.get(artifact, artifact)
        raise FileNotFoundError(
            f"cloud slayer: required local setup missing for {missing} "
            f"under {root} (combo {args.slayer_setup}/{args.framework}, "
            f"artifact {artifact} expects {marker_hint}); build it "
            f"locally first — the cloud runner never builds REQUIRED "
            f"setup in-cluster."
        )
    return dbs


def _upload_slayer_setup(args, run_id: str, dbs: list[str]) -> None:
    """Upload every artifact's local dir PER selected DB to
    ``runs/<run_id>/slayer_setup/<artifact>/<db>/``. Optional artifacts whose
    local ``<db>/`` is absent are skipped — the cloud will encode the missing
    reference on the fly."""
    uploads = _slayer_uploads_for(args)
    client = default_gcs_client()
    for root, artifact, required in uploads:
        for db in dbs:
            if not required and not _artifact_present(root, db, artifact):
                # Optional seed absent for this db — cloud will encode.
                continue
            prefix = f"runs/{run_id}/slayer_setup/{artifact}/{db}"
            gcs.upload_dir_prefix(root / db, prefix, client=client)


def _instance_ids_sorted_by_db(
    instance_ids, benchmark: str = "mini_interact",
) -> list[str]:
    """DEV-1470: sort iids by ``(selected_database, instance_id)`` so same-db
    tasks are adjacent in the ActorPool dispatch order — a single actor then
    typically does all encoding for a given DB and the cross-actor encode
    race is rare. Unknown iids (absent from the dataset) sort to the end
    grouped by their iid string so they still appear deterministically."""
    import json as _json
    wanted = list(instance_ids)
    if not wanted:
        return wanted
    data_file = paths.benchmark_data_file(benchmark)
    # De-bake: the dataset is GCS-delivered, so `resubmit` may run on a machine
    # that has no local copy. DB-grouping is only a dispatch optimization (it
    # makes same-db tasks adjacent so one actor tends to do all of a DB's
    # encoding), never a correctness gate — so a missing local data file falls
    # back to the input order rather than failing the resubmit (Codex).
    if not data_file.is_file():
        logger.info(
            "instance DB-grouping skipped: local data file %s absent "
            "(e.g. resubmit without the local dataset) — preserving input order",
            data_file,
        )
        return wanted
    db_by_iid: dict[str, str] = {}
    with data_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            td = _json.loads(line)
            iid = td.get("instance_id")
            if iid in wanted:
                db = td.get("selected_database") or ""
                db_by_iid[iid] = db
    # Stable, deterministic sort by (db, iid). Unknown iids → ("", iid).
    return sorted(wanted, key=lambda iid: (db_by_iid.get(iid, ""), iid))


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
    # Gated gold must ride along in the GCS dataset upload — fail fast at
    # submit if it's outside the data root (before any build/upload/cluster).
    _validate_gold_under_data_root(args)
    repo_root = submitter_repo_root()
    # DEV-1468: slayer fail-fast — verify the local setup is present BEFORE
    # building/pushing the image or bringing up a cluster (no in-cloud builds).
    slayer_dbs: list[str] = []
    if args.query_mode == "slayer":
        slayer_dbs = _check_slayer_setup_present(args)
    tag = image.image_tag(
        repo_root,
        paths.audited_gold_root(),
        allow_dirty=args.allow_dirty,
    )
    image_uri = image.build_and_push(
        tag, repo_root,
        audited_gold_root=paths.audited_gold_root(),
        force=False,
    )
    # De-bake: upload the benchmark dataset ONCE to its content-hashed GCS
    # prefix (skipped when the hash already exists). The actor downloads it
    # per node — the dataset is no longer baked into the image.
    benchmark_data_prefix = benchmark_data.ensure_uploaded(_submit_benchmark(args))
    run_id = args.run_id or mint_run_id(args.framework, args.query_mode)
    manifest = build_manifest(
        args, image_uri=image_uri, run_id=run_id,
        benchmark_data_prefix=benchmark_data_prefix,
    )
    gcs.write_manifest(run_id, manifest)
    # Upload the slayer setup under the run prefix (per selected DB) so the
    # actor downloads it in-cluster. After the manifest so kill/resubmit work.
    if args.query_mode == "slayer":
        _upload_slayer_setup(args, run_id, slayer_dbs)
    yaml_path = cluster.render_from_manifest(manifest, cache_dir=yaml_cache_dir())
    h = install_signal_handlers(run_id=run_id, yaml_path=yaml_path)
    submit_succeeded = False
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        env_vars = read_api_keys_from_local_env(
            args.agent_model, args.user_sim_model, query_mode=args.query_mode,
        )
        job_args = _build_job_args(
            args, run_id, attempt=1,
            benchmark_data_prefix=benchmark_data_prefix,
        )
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


def _build_job_args(
    args, run_id: str, *, attempt: int, benchmark_data_prefix: str | None = None,
) -> list[str]:
    benchmark = _submit_benchmark(args)
    job_args = [
        "--run-id", run_id,
        "--attempt", str(attempt),
        "--framework", args.framework,
        "--query-mode", args.query_mode,
        "--mode", args.mode,
        "--dataset", benchmark,
        "--agent-model", args.agent_model,
        "--user-sim-model", args.user_sim_model,
        "--patience", str(args.patience),
        "--max-depth", str(args.max_depth),
        "--num-actors", str(args.workers * args.actors_per_worker),
        # DEV-1470: group same-db iids adjacently so a single actor typically
        # does all encoding for a given DB.
        "--instance-ids",
        ",".join(_instance_ids_sorted_by_db(args.instance_ids, benchmark)),
    ]
    if benchmark_data_prefix:
        job_args += ["--benchmark-data-prefix", benchmark_data_prefix]
    gold_file = _in_cluster_gold_file(args)
    if gold_file:
        job_args += ["--gold-file", gold_file]
    if args.strict:
        job_args.append("--strict")
    if args.use_audited_gold_sql:
        job_args.append("--use-audited-gold-sql")
    if args.prompt_cache:
        job_args.append("--prompt-cache")
    else:
        job_args.append("--no-prompt-cache")
    if getattr(args, "reasoning_effort", None):
        job_args += ["--reasoning-effort", args.reasoning_effort]
    job_args += [
        "--slayer-setup", getattr(args, "slayer_setup", "pre-encoded"),
        "--slayer-storage-root",
        getattr(args, "slayer_storage_root", "/data/slayer_models"),
    ]
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
    # DEV-1470: promote per-DB cloud-encoded OTF references from
    # <run_dir>/post_run/slayer_models_otf/<shard>/<db>/ into the global
    # warm cache at paths.slayer_models_otf_root()/<db>/ (newest-mtime-wins
    # per file, with the marker invariant preserved under a per-DB flock).
    # No-op for runs without any post_run/ shards (raw, pre-encoded, recursive).
    merge_report = _post_run_merge.merge_post_run_into_warm_cache(
        run_dir=dest,
        reference_root=paths.slayer_models_otf_root(
            benchmark=_benchmark_for_dataset(manifest.get("dataset")),
        ),
    )
    metrics["merge_report"] = merge_report
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
            query_mode=manifest.get("query_mode", "raw"),
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
    benchmark = _benchmark_for_dataset(manifest.get("dataset"))
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
        # DEV-1470: db-grouped retries — same rule as submit.
        "--instance-ids", ",".join(_instance_ids_sorted_by_db(missing, benchmark)),
    ]
    # Pass --dataset / --benchmark-data-prefix ONLY when the manifest carries
    # them. A manifest with a `dataset` key was written by a driver new enough
    # to also bake `--dataset` support into its image; a pre-`--dataset`
    # manifest (no key) pins an OLDER image whose `ray_app` argparse rejects
    # `--dataset`, so resubmitting it MUST omit the flag and let the old image
    # use its baked mini-interact data (Codex). The prefix is independently
    # conditional for the same reason (de-bake came later than `--dataset`).
    if manifest.get("dataset"):
        job_args += ["--dataset", benchmark]
    prefix = manifest.get("benchmark_data_prefix")
    if prefix:
        job_args += ["--benchmark-data-prefix", prefix]
    # Manifest gold_file is ALREADY the in-cluster path (build_manifest
    # translated it at submit), so pass it through unchanged.
    gold_file = manifest.get("gold_file")
    if gold_file:
        job_args += ["--gold-file", str(gold_file)]
    if manifest.get("strict"):
        job_args.append("--strict")
    if manifest.get("use_audited_gold_sql"):
        job_args.append("--use-audited-gold-sql")
    if manifest.get("prompt_cache", True):
        job_args.append("--prompt-cache")
    else:
        job_args.append("--no-prompt-cache")
    if manifest.get("reasoning_effort"):
        job_args += ["--reasoning-effort", manifest["reasoning_effort"]]
    job_args += [
        "--slayer-setup", manifest.get("slayer_setup", "pre-encoded"),
        "--slayer-storage-root",
        manifest.get("slayer_storage_root", "/data/slayer_models"),
    ]
    return job_args
