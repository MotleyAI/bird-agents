"""In-cluster Ray driver. Invoked via `ray job submit -- python ray_app.py
<args>` from the laptop-side driver.

Holds the WorkerActor + ActorPool dispatch loop, the once-per-worker SLayer
setup download (DEV-1468 — mirrors local; no per-task ephemeral server), the
fd-level log capture, and the heartbeat writer.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from bird_interact_agents.cloud import gcs as _gcs
from bird_interact_agents.cloud import upload_back as _upload_back


# ---------------------------------------------------------------------------
# GCS client default (overridable in tests)
# ---------------------------------------------------------------------------


def default_gcs_client():
    return _gcs.default_gcs_client()


# ---------------------------------------------------------------------------
# SLayer setup download (DEV-1468 — once per worker process; mirrors local)
# ---------------------------------------------------------------------------


_ARTIFACT_ROOT_FN_NAME = {
    "slayer_models": "slayer_models_root",
    "slayer_otf_cache": "slayer_otf_cache_root",
    "slayer_models_otf": "slayer_models_otf_root",
}


def _slayer_artifacts_for(cfg: dict[str, Any]) -> list[tuple[str, Path, bool]]:
    """Return ``[(artifact, dest_root, required), ...]`` for the run's combo.

    DEV-1470: ``pydantic_ai_otf_encode + on-the-fly`` requires the
    deterministic cache (``slayer_otf_cache``, input to the LLM encoder) and
    has an OPTIONAL ``slayer_models_otf`` seed (skipped if absent in GCS;
    merged into the existing root file-by-file if present, never atomic-
    replace, so cloud-built references already on disk survive an actor
    restart's re-download attempt — H3 / M3).
    """
    from bird_interact_agents import paths

    setup = cfg.get("slayer_setup")
    fw = cfg.get("framework")
    if setup == "pre-encoded":
        artifacts = [("slayer_models", True)]
    elif fw == "pydantic_ai_otf_encode":
        artifacts = [
            ("slayer_otf_cache", True),
            ("slayer_models_otf", False),
        ]
    else:
        artifacts = [("slayer_otf_cache", True)]
    out: list[tuple[str, Path, bool]] = []
    for artifact, required in artifacts:
        root_fn = getattr(paths, _ARTIFACT_ROOT_FN_NAME[artifact])
        out.append((artifact, root_fn(), required))
    return out


def _slayer_download_target(cfg: dict[str, Any]) -> tuple[str, Path]:
    """Back-compat single-artifact accessor for tests/code that still expects
    one (artifact, root) pair. Returns the first artifact in the combo's
    list — for combos with multiple artifacts (otf_encode + on-the-fly), that
    is the REQUIRED cache."""
    artifact, dest_root, _required = _slayer_artifacts_for(cfg)[0]
    return artifact, dest_root


@contextmanager
def _per_db_build_lock(reference_root: Path, db: str) -> Iterator[None]:
    """Cross-process per-DB lock shared with the merger and
    :func:`bird_interact_agents.slayer_otf.reference_build._build_reference`.
    The optional seed download takes this lock so it cannot land on top of
    an in-flight cloud encoder (H4)."""
    reference_root.mkdir(parents=True, exist_ok=True)
    lock_path = reference_root / f"{db}.build.lock"
    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _download_required_artifact(
    *, run_id: str, artifact: str, dest_root: Path, client,
) -> None:
    """Required-artifact download: atomic via tmp + ``os.rename`` onto an
    ABSENT ``dest_root``. Root-level ``.download_complete`` marker makes
    repeated calls a no-op (idempotent across a VM's worker processes).
    Empty GCS prefix is FATAL — a missing required upload must surface, not
    silently cache as an empty setup."""
    marker = dest_root / ".download_complete"
    if marker.is_file():
        return

    prefix = f"runs/{run_id}/slayer_setup/{artifact}/"
    parent = dest_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest_root.name}.dl-", dir=str(parent)))
    try:
        _gcs.download_prefix(prefix, tmp, client=client)
        if not any(p.is_file() for p in tmp.rglob("*")):
            raise FileNotFoundError(
                f"slayer setup download found no files under gs://"
                f"{_gcs.BUCKET_NAME}/{prefix} for run {run_id} — the submit "
                f"upload is missing/empty; refusing to cache an empty setup"
            )
        (tmp / ".download_complete").write_text("ok")  # marker LAST
        try:
            os.rename(tmp, dest_root)
        except OSError:
            if not marker.is_file():
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


_OPTIONAL_SEED_MARKER = ".optional_seed_download_complete"


def _download_optional_seed(
    *, run_id: str, artifact: str, dest_root: Path, client,
) -> None:
    """Optional-seed download (DEV-1470 for the ``slayer_models_otf`` artifact
    under ``otf_encode + on-the-fly``):

    * If the GCS prefix is empty: NO-OP (no marker, no rmtree). The cloud
      will encode any missing per-DB reference lazily.
    * If non-empty: download into a tmp sibling, then MERGE file-by-file into
      the existing ``dest_root`` using per-file ``newest-source-mtime-wins``
      (so a cloud-built reference already present after an actor restart is
      not clobbered by an older seed). Per-DB cross-process ``fcntl.flock``
      on ``<dest_root>/<db>.build.lock`` (shared with the build lock — H4) so
      an in-flight encoder writing to ``<dest_root>/<db>/`` doesn't race the
      merge.
    * Idempotent across restart via the ``.optional_seed_download_complete``
      marker at the root.
    """
    marker = dest_root / _OPTIONAL_SEED_MARKER
    if marker.is_file():
        return

    prefix = f"runs/{run_id}/slayer_setup/{artifact}/"
    parent = dest_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest_root.name}.seed-", dir=str(parent)))
    try:
        _gcs.download_prefix(prefix, tmp, client=client)
        files = [p for p in tmp.rglob("*") if p.is_file()]
        if not files:
            # Empty prefix → no seed, no marker (cloud will encode). The
            # ABSENCE of the marker is intentional: lets a re-run attempt the
            # download again, harmless when still empty.
            return
        # Group files by their top-level db dir (first path component
        # relative to tmp). Each db's merge is done under its own build-lock.
        dbs: dict[str, list[Path]] = {}
        for p in files:
            rel = p.relative_to(tmp)
            db = rel.parts[0] if rel.parts else ""
            if not db:
                continue
            dbs.setdefault(db, []).append(p)
        dest_root.mkdir(parents=True, exist_ok=True)
        for db, db_files in sorted(dbs.items()):
            with _per_db_build_lock(dest_root, db):
                for src in db_files:
                    rel = src.relative_to(tmp)
                    dst = dest_root / rel
                    src_mtime = src.stat().st_mtime
                    try:
                        local_mtime = dst.stat().st_mtime
                    except FileNotFoundError:
                        local_mtime = 0.0
                    if src_mtime <= local_mtime:
                        continue  # local is newer (e.g. cloud-built); keep
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic per-file replace.
                    tmp_dst = dst.parent / f".{dst.name}.seed-{os.getpid()}"
                    tmp_dst.write_bytes(src.read_bytes())
                    os.replace(tmp_dst, dst)
        marker.write_text("ok")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def download_slayer_setup(run_id: str, cfg: dict[str, Any], *, client) -> None:
    """Download the run's uploaded slayer setup into each artifact's
    env-override root, ONCE per worker process. No-op unless
    ``cfg['query_mode'] == 'slayer'``.

    DEV-1470: a combo can list multiple artifacts via :func:`_slayer_artifacts_for`,
    each tagged REQUIRED or OPTIONAL. REQUIRED uses the atomic
    ``tmp + os.rename`` path with the root-level ``.download_complete`` marker
    (empty prefix is fatal). OPTIONAL uses :func:`_download_optional_seed`
    (empty prefix → no-op; non-empty → per-DB cross-process locked file-merge
    so an actor restart cannot clobber a previously cloud-built reference).
    """
    if cfg.get("query_mode") != "slayer":
        return
    for artifact, dest_root, required in _slayer_artifacts_for(cfg):
        if required:
            _download_required_artifact(
                run_id=run_id, artifact=artifact, dest_root=dest_root,
                client=client,
            )
        else:
            _download_optional_seed(
                run_id=run_id, artifact=artifact, dest_root=dest_root,
                client=client,
            )


# ---------------------------------------------------------------------------
# fd-level log capture (catches subprocess stderr/stdout too)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def fd_capture(log_path: Path):
    """Redirect OS-level fd 1 and 2 to `log_path` for the duration of the
    `with` block. Subprocess inheritance picks this up too."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Flush Python buffers before swapping fds.
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


# ---------------------------------------------------------------------------
# Heartbeat writer
# ---------------------------------------------------------------------------


class HeartbeatWriter:
    """Background thread that writes `runs/<run-id>/status.json` every
    `interval_s` seconds while running, and once more at `stop_and_flush`."""

    def __init__(
        self,
        *,
        run_id: str,
        total: int,
        attempt: int,
        ray_job_id: str,
        client=None,
        interval_s: float = 30.0,
    ) -> None:
        self.run_id = run_id
        self.total = total
        self.attempt = attempt
        self.ray_job_id = ray_job_id
        self.interval_s = interval_s
        self.client = client or default_gcs_client()
        self._done = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick_done(self) -> None:
        with self._lock:
            self._done += 1

    def _write(self, terminal_state: str | None) -> None:
        with self._lock:
            done = self._done
        status = {
            "ray_job_id": self.ray_job_id,
            "last_heartbeat_ts": time.time(),
            "rows_done": done,
            "rows_total": self.total,
            "terminal_state": terminal_state,
            "attempt": self.attempt,
        }
        _gcs.write_status(self.run_id, status, client=self.client)

    def _loop(self) -> None:
        # Emit one heartbeat immediately so consumers see the run is live.
        self._write(terminal_state=None)
        while not self._stop.wait(self.interval_s):
            self._write(terminal_state=None)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_and_flush(self, *, terminal_state: str) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._write(terminal_state=terminal_state)


# ---------------------------------------------------------------------------
# Per-task body — runs inside the actor
# ---------------------------------------------------------------------------


def _build_error_row(iid: str, database: str, message: str) -> dict:
    return {
        "instance_id": iid,
        "database": database,
        "phase1_passed": False,
        "phase2_passed": False,
        "total_reward": 0.0,
        "submitted_sql": None,
        "submitted_query": None,
        "ground_truth_sql": None,
        "error": message,
        "submission_status": "never_submitted",
        "phase1_observation": None,
        "phase2_observation": None,
        "predicted_result_json": None,
        "gold_result_json": None,
        "n_agent_turns": 0,
        "duration_s": 0.0,
    }


async def _run_one_task_async(
    *,
    task_data: dict,
    framework: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    user_sim_model: str,
    patience: int,
    strict: bool,
    use_audited_gold_sql: bool,
    prompt_cache: bool,
    max_depth: int,
    data_dir: str,
    slayer_storage_root: str | None,
    slayer_setup: str = "pre-encoded",
    cached_runner: Any = None,
) -> dict:
    # Defer the import so monkeypatching `bird_interact_agents.run.run_one_task`
    # in tests is honoured (monkeypatch replaces the attribute on the module).
    from bird_interact_agents import run as run_mod

    if cached_runner is not None:
        return await run_mod.run_one_task_with_runner(
            cached_runner, task_data,
            data_dir=data_dir, patience=patience, user_sim_model=user_sim_model,
        )
    return await run_mod.run_one_task(
        task_data=task_data,
        data_dir=data_dir,
        framework=framework,
        query_mode=query_mode,
        mode=mode,
        agent_model=agent_model,
        user_sim_model=user_sim_model,
        patience=patience,
        strict=strict,
        use_audited_gold_sql=use_audited_gold_sql,
        prompt_cache=prompt_cache,
        max_depth=max_depth,
        slayer_storage_root=slayer_storage_root,
        slayer_setup=slayer_setup,
    )


def _run_one_in_actor(
    *,
    task_data: dict,
    cfg: dict[str, Any],
    run_id: str,
    attempt: int,
    gcs_client,
    cached_runner: Any = None,
    uploaded_dbs: set[str] | None = None,
    initial_seed_fp_by_db: dict[str, str] | None = None,
) -> str:
    """The per-task body that runs INSIDE the actor. Captures logs, invokes
    run_one_task (which resolves per-task SLayer storage from the downloaded
    setup, exactly as the local path does), writes the row + log to GCS, then
    fires the DEV-1470 upload-back triple (best-effort), and returns the iid.

    DEV-1468: there is no per-task ephemeral SLayer server anymore — the setup
    is downloaded once per worker (``download_slayer_setup`` in the actor
    ``__init__``) and ``run_one_task`` builds the per-task variant storage from
    it (via ``cfg['slayer_storage_root']`` / ``slayer_setup``).

    DEV-1470: after the row/log writes (and BEFORE wiping ``log_dir``), invoke
    in order: ``upload_per_task_debug`` → ``upload_per_task_setup_sessions``
    → ``upload_otf_reference_delta``. Any upload-back exception is swallowed
    and logged to stderr — the per-task row already landed, and a logging
    failure must never poison the actor."""
    iid = str(task_data.get("instance_id") or "")
    database = str(task_data.get("selected_database") or "")
    log_dir = Path(tempfile.mkdtemp(prefix="cloud_log_"))
    log_tmp = log_dir / "task.log"
    task_start_ts = time.time()

    try:
        with fd_capture(log_tmp):
            try:
                data_dir = os.environ.get(
                    "BIRD_DB_PATH", cfg.get("data_dir", "/data/mini-interact")
                )

                row = asyncio.run(
                    _run_one_task_async(
                        task_data=task_data,
                        framework=cfg["framework"],
                        query_mode=cfg["query_mode"],
                        mode=cfg["mode"],
                        agent_model=cfg["agent_model"],
                        user_sim_model=cfg["user_sim_model"],
                        patience=cfg["patience"],
                        strict=cfg["strict"],
                        use_audited_gold_sql=cfg["use_audited_gold_sql"],
                        prompt_cache=cfg["prompt_cache"],
                        max_depth=cfg["max_depth"],
                        data_dir=data_dir,
                        slayer_storage_root=cfg.get("slayer_storage_root"),
                        slayer_setup=cfg.get("slayer_setup", "pre-encoded"),
                        cached_runner=cached_runner,
                    )
                )
            except Exception as e:  # noqa: BLE001
                row = _build_error_row(iid, database, str(e))
    finally:
        pass

    _gcs.write_row(run_id, iid, attempt, row, client=gcs_client)
    try:
        log_bytes = log_tmp.read_bytes() if log_tmp.exists() else b""
    except OSError:
        log_bytes = b""
    if log_bytes:
        _gcs.write_log(run_id, iid, attempt, log_bytes, client=gcs_client)

    # DEV-1470: best-effort upload-back triple. Each helper swallows its own
    # exceptions, but we also wrap the whole block so a programming bug here
    # never prevents the log tmp-dir cleanup.
    try:
        work_root = Path(tempfile.gettempdir()) / "bird_interact_slayer_otf"
        _upload_back.upload_per_task_debug(
            run_id=run_id, iid=iid, attempt=attempt,
            work_root=work_root, client=gcs_client,
        )
        _upload_back.upload_per_task_setup_sessions(
            run_id=run_id, iid=iid, attempt=attempt,
            setup_sessions_root=work_root / "_setup_sessions",
            task_start_ts=task_start_ts, client=gcs_client,
        )
        _upload_back.upload_otf_reference_delta(
            run_id=run_id, cfg=cfg,
            shard=f"{socket.gethostname()}-{os.getpid()}",
            uploaded_dbs=uploaded_dbs if uploaded_dbs is not None else set(),
            initial_seed_fp_by_db=initial_seed_fp_by_db or {},
            client=gcs_client,
        )
    except Exception:  # noqa: BLE001
        sys.stderr.write(
            f"[upload_back] outer failure for {iid}: {traceback.format_exc()}\n"
        )

    # Now safe to drop the log tmp dir.
    shutil.rmtree(log_dir, ignore_errors=True)
    return iid


def _snapshot_initial_seed_fps(cfg: dict[str, Any]) -> dict[str, str]:
    """DEV-1470: after ``download_slayer_setup`` has run, snapshot the
    per-DB ``_reference_fp.txt`` content for every db present under
    ``paths.slayer_models_otf_root()``. Drives
    :func:`upload_back.upload_otf_reference_delta`'s skip-or-upload decision —
    a db whose on-disk fp matches the seed snapshot wasn't rebuilt by this
    actor and must NOT be uploaded back. Empty dict when the root is absent
    (no seed → every cloud-built reference is genuinely new).
    """
    if cfg.get("query_mode") != "slayer":
        return {}
    if cfg.get("framework") != "pydantic_ai_otf_encode":
        return {}
    from bird_interact_agents import paths
    ref_root = paths.slayer_models_otf_root()
    if not ref_root.is_dir():
        return {}
    out: dict[str, str] = {}
    for db_dir in sorted(p for p in ref_root.iterdir() if p.is_dir()):
        marker = db_dir / "_reference_fp.txt"
        if marker.is_file():
            try:
                out[db_dir.name] = marker.read_text().strip()
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------
# WorkerActor — minimal wrapper so tests can mock the actor surface
# ---------------------------------------------------------------------------


class _LocalActor:
    """Local Python actor stand-in used when Ray's actor model isn't
    available (e.g. local_mode tests). Same `.run_one` interface.

    Defaults to building its own GCS client in __init__ — the real Ray
    `WorkerActor` MUST self-build because `google.cloud.storage.Client` is
    unpicklable and can't cross the actor constructor boundary. As an
    in-process stand-in, `_LocalActor` additionally accepts an injected
    `gcs_client`, which lets tests pass a fake directly instead of having
    to monkeypatch `default_gcs_client` (no pickling involved locally)."""

    def __init__(
        self,
        cfg: dict[str, Any],
        run_id: str,
        attempt: int,
        gcs_client=None,
    ):
        self.cfg = cfg
        self.run_id = run_id
        self.attempt = attempt
        self.gcs_client = gcs_client or default_gcs_client()
        # DEV-1468: download the uploaded SLayer setup ONCE per process (gated
        # on slayer mode) before any task runs, so run_one_task finds the
        # artifacts at the env-override roots — exactly like local.
        if cfg.get("query_mode") == "slayer":
            download_slayer_setup(run_id, cfg, client=self.gcs_client)
        # DEV-1470: per-actor state for upload_otf_reference_delta. Snapshot
        # the seed fingerprints AFTER download (so we observe what download
        # just landed); uploaded_dbs starts empty and grows on successful
        # upload, so failed uploads remain eligible for retry on later tasks.
        self.initial_seed_fp_by_db = _snapshot_initial_seed_fps(cfg)
        self.uploaded_dbs: set[str] = set()
        # CR#14: cache the framework runner across tasks for raw mode.
        # Slayer mode keeps per-task reconstruction because the storage
        # root rotates per task (per §4#5 isolation).
        self._cached_runner = _maybe_build_cached_runner(cfg)

    def run_one(self, task_data: dict) -> str:
        return _run_one_in_actor(
            task_data=task_data,
            cfg=self.cfg,
            run_id=self.run_id,
            attempt=self.attempt,
            gcs_client=self.gcs_client,
            cached_runner=self._cached_runner,
            uploaded_dbs=self.uploaded_dbs,
            initial_seed_fp_by_db=self.initial_seed_fp_by_db,
        )


def _maybe_build_cached_runner(cfg: dict[str, Any]):
    """Return a cacheable runner for raw mode; None otherwise (slayer
    mode rotates `slayer_storage_root` per task, oracle mode doesn't
    benefit)."""
    if cfg["query_mode"] != "raw" or cfg["mode"] == "oracle":
        return None
    from bird_interact_agents import run as run_mod
    return run_mod.make_runner(
        framework=cfg["framework"],
        query_mode=cfg["query_mode"],
        mode=cfg["mode"],
        agent_model=cfg["agent_model"],
        strict=cfg["strict"],
        prompt_cache=cfg["prompt_cache"],
        max_depth=cfg["max_depth"],
        slayer_storage_root=None,
    )


def _build_actor_class():
    """Return a Ray-remote actor class with the same `.run_one` interface
    as `_LocalActor`. Lazy import so test environments without Ray can
    use _LocalActor."""
    import ray

    @ray.remote(max_restarts=3, max_task_retries=0)
    class WorkerActor:
        def __init__(self, cfg: dict[str, Any], run_id: str, attempt: int):
            self.cfg = cfg
            self.run_id = run_id
            self.attempt = attempt
            # Build the GCS client INSIDE the actor process — it's
            # unpicklable, so it can't be a constructor arg shipped from
            # the driver (that raised PicklingError). Each actor builds
            # its own from the VM service-account metadata creds.
            self.gcs_client = default_gcs_client()
            # DEV-1468: download the uploaded SLayer setup ONCE per worker
            # process (gated on slayer mode) before any task runs. Mirrors
            # `_LocalActor`; the root-level marker makes concurrent actors on
            # one VM converge.
            if cfg.get("query_mode") == "slayer":
                download_slayer_setup(run_id, cfg, client=self.gcs_client)
            # DEV-1470 — per-actor upload-back state. AFTER the download.
            self.initial_seed_fp_by_db = _snapshot_initial_seed_fps(cfg)
            self.uploaded_dbs = set()
            # CR#14 — cache the framework runner across tasks for raw
            # mode. `_LocalActor` does the same in its __init__; without
            # mirroring it here, the real Ray production path was paying
            # per-task agent reconstruction.
            self.cached_runner = _maybe_build_cached_runner(cfg)

        def run_one(self, task_data: dict) -> str:
            return _run_one_in_actor(
                task_data=task_data,
                cfg=self.cfg,
                run_id=self.run_id,
                attempt=self.attempt,
                gcs_client=self.gcs_client,
                cached_runner=self.cached_runner,
                uploaded_dbs=self.uploaded_dbs,
                initial_seed_fp_by_db=self.initial_seed_fp_by_db,
            )

    return WorkerActor


def _with_actor_env(actor_cls: Any, actor_env_vars: dict[str, str] | None) -> Any:
    """Bind `actor_env_vars` (e.g. API keys) onto a Ray actor class as a
    PER-ACTOR runtime_env. This ships the secrets to worker actors WITHOUT
    putting them in the job's runtime_env — which `ray job list`/the
    dashboard echo back. No-op (returns the class unchanged) when there are
    no env vars, so test actor classes without `.options()` are untouched."""
    if actor_env_vars:
        return actor_cls.options(runtime_env={"env_vars": actor_env_vars})
    return actor_cls


# ---------------------------------------------------------------------------
# Drain loop — handles RayActorError without aborting the run
# ---------------------------------------------------------------------------


def drain_pool(
    *,
    pool: Any,
    run_id: str,
    attempt: int,
    gcs_client,
    leftover_iids: Iterable[str] | None = None,
    dispatch: Callable[[Any, str], Any] | None = None,
    heartbeat: HeartbeatWriter | None = None,
) -> None:
    """Drain a pool that exposes `has_next`, `get_next_unordered`, `submit`.

    On `RayActorError`, recovers the in-flight iid via `pool.last_failed_iid`
    (whose presence is the contract real Ray callers fulfil by tracking
    actor→task themselves), writes a synthetic `actor-lost` row, and
    continues.
    """
    try:
        from ray.exceptions import RayActorError  # type: ignore[import-not-found]
    except ImportError:
        class RayActorError(Exception):
            pass

    leftover = list(leftover_iids or [])
    while pool.has_next():
        try:
            # Consume one completed result (advances the pool). The iid isn't
            # needed here — the actor already wrote its per-task row; drain_pool
            # only emits synthetic actor-lost rows on failure.
            pool.get_next_unordered()
            if heartbeat is not None:
                heartbeat.tick_done()
            if leftover and dispatch is not None:
                next_iid = leftover.pop(0)
                pool.submit(dispatch, next_iid)
        except RayActorError:
            lost_iid = getattr(pool, "last_failed_iid", None)
            if lost_iid:
                err_row = _build_error_row(
                    lost_iid, "", "actor-lost"
                )
                _gcs.write_row(run_id, lost_iid, attempt, err_row,
                                client=gcs_client)
                if heartbeat is not None:
                    heartbeat.tick_done()
            # Carry on with the next leftover (if any).
            if leftover and dispatch is not None:
                next_iid = leftover.pop(0)
                try:
                    pool.submit(dispatch, next_iid)
                except Exception:  # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# Public entry point: run_pool
# ---------------------------------------------------------------------------


def run_pool(
    *,
    run_id: str,
    instance_ids: list[str],
    framework: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    num_actors: int,
    attempt: int,
    task_data_by_id: dict[str, dict],
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
    patience: int = 3,
    strict: bool = False,
    use_audited_gold_sql: bool = False,
    prompt_cache: bool = True,
    max_depth: int = 3,
    slayer_setup: str = "pre-encoded",
    slayer_storage_root: str | None = None,
    ray_job_id: str = "local",
    gcs_client=None,
    heartbeat_interval_s: float = 30.0,
    local_only: bool = False,
    actor_cls: Any = None,
    actor_env_vars: dict[str, str] | None = None,
) -> None:
    """Construct actors, dispatch all `instance_ids` via Ray's ActorPool,
    handle actor death, write heartbeat + rows.

    `actor_env_vars` (e.g. API keys) are applied as a PER-ACTOR runtime_env
    so they reach the worker actors without ever entering the *job's*
    runtime_env — which `ray job list`/the dashboard echo back. They're
    delivered to the head out-of-band (a secrets file rsync'd in, never on
    a command line) and threaded here by `main`."""
    client = gcs_client or default_gcs_client()
    cfg: dict[str, Any] = {
        "framework": framework,
        "query_mode": query_mode,
        "mode": mode,
        "agent_model": agent_model,
        "user_sim_model": user_sim_model,
        "patience": patience,
        "strict": strict,
        "use_audited_gold_sql": use_audited_gold_sql,
        "prompt_cache": prompt_cache,
        "max_depth": max_depth,
        "slayer_setup": slayer_setup,
        "slayer_storage_root": slayer_storage_root,
        "data_dir": os.environ.get("BIRD_DB_PATH", "/data/mini-interact"),
    }

    heartbeat = HeartbeatWriter(
        run_id=run_id, total=len(instance_ids), attempt=attempt,
        ray_job_id=ray_job_id, client=client,
        interval_s=heartbeat_interval_s,
    )

    try:
        import ray  # type: ignore[import-not-found]
        ray.util  # type: ignore[attr-defined]
        ray_available = True
    except ImportError:
        ray_available = False

    use_local = local_only or not ray_available

    heartbeat.start()
    try:
        if use_local:
            # In-process actors share this process's os.environ, so apply
            # the secrets here (the per-actor runtime_env path below only
            # works for real, separate Ray worker processes).
            if actor_env_vars:
                os.environ.update(actor_env_vars)
            if actor_cls is None:
                # Our own _LocalActor takes the client at construction, so it
                # never calls `default_gcs_client()` — which can fail without
                # real creds (Codex). Keeps local-mode writes on run_pool's
                # client.
                actors = [
                    _LocalActor(cfg, run_id, attempt, gcs_client=client)
                    for _ in range(num_actors)
                ]
            else:
                # A custom actor_cls (tests) may not accept `gcs_client`;
                # construct it plainly, then keep its writes on run_pool's
                # client if it exposes the attribute (the heartbeat already
                # uses `client`, so otherwise an injected client would be
                # honoured by the heartbeat but ignored by the actors,
                # splitting a run's artifacts across backends). The real Ray
                # path can't do this — the client is unpicklable, so remote
                # actors must self-build; see `_build_actor_class`.
                actors = [
                    actor_cls(cfg, run_id, attempt)
                    for _ in range(num_actors)
                ]
                for actor in actors:
                    if hasattr(actor, "gcs_client"):
                        actor.gcs_client = client
            for iid in instance_ids:
                actor = actors[hash(iid) % num_actors]
                actor.run_one(task_data_by_id[iid])
                heartbeat.tick_done()
        else:
            ActorCls = _with_actor_env(
                actor_cls or _build_actor_class(), actor_env_vars
            )
            actors = [
                ActorCls.remote(cfg, run_id, attempt)
                for _ in range(num_actors)
            ]
            _run_with_actors(
                actors=actors,
                instance_ids=instance_ids,
                task_data_by_id=task_data_by_id,
                run_id=run_id,
                attempt=attempt,
                gcs_client=client,
                heartbeat=heartbeat,
                actor_factory=lambda: ActorCls.remote(cfg, run_id, attempt),
            )
        heartbeat.stop_and_flush(terminal_state="done")
    except Exception:
        heartbeat.stop_and_flush(terminal_state="error")
        raise


def _run_with_actors(
    *,
    actors: list,
    instance_ids: list[str],
    task_data_by_id: dict[str, dict],
    run_id: str,
    attempt: int,
    gcs_client,
    heartbeat: HeartbeatWriter,
    actor_factory: Callable[[], Any],
) -> None:
    """Drive a pool of Ray actors with precise actor→iid bookkeeping.

    Replaces `ray.util.ActorPool`, which doesn't expose which task was on
    a dying actor when `RayActorError` fires. We maintain
    `in_flight: future → (actor, iid)` ourselves, so an actor death
    produces an `actor-lost` row keyed to the exact iid that was on the
    dying actor.

    Replacement actors are minted via `actor_factory` so a partially-dead
    cluster doesn't bleed throughput as actors die. Dispatch-side errors
    are surfaced via a logged error row rather than silently swallowed
    (CR#13).
    """
    import ray  # type: ignore[import-not-found]
    from ray.exceptions import RayActorError  # type: ignore[import-not-found]

    in_flight: dict[Any, tuple[Any, str]] = {}
    free_actors: list[Any] = list(actors)
    pending = list(instance_ids)

    def _dispatch_to(actor: Any, iid: str) -> None:
        try:
            future = actor.run_one.remote(task_data_by_id[iid])
            in_flight[future] = (actor, iid)
        except Exception as e:  # noqa: BLE001
            # Could not dispatch — log + write an `error` row so the iid
            # is visible in `eval.json` rather than silently missing.
            err_row = _build_error_row(iid, "", f"dispatch-failure: {e}")
            try:
                _gcs.write_row(run_id, iid, attempt, err_row, client=gcs_client)
            except Exception:  # noqa: BLE001
                pass
            heartbeat.tick_done()
            # Do NOT recycle the failing actor — if its handle is dead /
            # unreachable, the same exception fires for every remaining
            # iid and burns the run. Mint a replacement instead so a
            # single bad actor doesn't take down the cluster.
            try:
                free_actors.append(actor_factory())
            except Exception:  # noqa: BLE001
                pass

    def _fill_free() -> None:
        while free_actors and pending:
            actor = free_actors.pop(0)
            iid = pending.pop(0)
            _dispatch_to(actor, iid)

    _fill_free()
    while in_flight:
        ready_futures, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        future = ready_futures[0]
        actor, iid = in_flight.pop(future)
        try:
            ray.get(future)
            heartbeat.tick_done()
            free_actors.append(actor)
        except RayActorError:
            err_row = _build_error_row(iid, "", "actor-lost")
            try:
                _gcs.write_row(run_id, iid, attempt, err_row,
                                client=gcs_client)
            except Exception:  # noqa: BLE001 — best effort; log is enough
                pass
            heartbeat.tick_done()
            # Mint a replacement so we don't lose throughput.
            try:
                free_actors.append(actor_factory())
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            # The task escaped the actor's *internal* try/except — i.e. a
            # failure OUTSIDE run_one_task (most likely the final
            # `write_row`/`write_log` GCS calls). The actor did NOT persist
            # a row, so we must write one here; otherwise the run reports
            # `done` with the row silently missing (exactly the bug this
            # branch used to cause). Surface the traceback to `ray job
            # logs` so infra failures (GCS perms, connectivity) are
            # diagnosable.
            import traceback
            sys.stderr.write(
                f"[bird-interact-cloud] task {iid} raised out of actor:\n"
                f"{traceback.format_exc()}\n"
            )
            err_row = _build_error_row(iid, "", f"actor-task-error: {e}")
            try:
                _gcs.write_row(run_id, iid, attempt, err_row,
                                client=gcs_client)
            except Exception as we:  # noqa: BLE001
                sys.stderr.write(
                    f"[bird-interact-cloud] write_row ALSO failed for "
                    f"{iid}: {we!r}\n"
                )
            heartbeat.tick_done()
            free_actors.append(actor)
        _fill_free()

    # Drain any pending iids that never got dispatched (e.g. all actors
    # died and `actor_factory()` kept failing). Without this, the run
    # would terminate as "done" with those iids silently missing from
    # GCS — the spec promises record-and-move-on, not record-and-drop.
    if pending:
        for iid in pending:
            err_row = _build_error_row(iid, "", "undispatched: no live actors")
            try:
                _gcs.write_row(run_id, iid, attempt, err_row, client=gcs_client)
            except Exception:  # noqa: BLE001
                pass
            heartbeat.tick_done()
        pending.clear()


# ---------------------------------------------------------------------------
# CLI entry — `python ray_app.py --run-id ... --gcs-bucket ... --attempt N`
# ---------------------------------------------------------------------------


def _load_task_data(
    instance_ids: list[str], *, use_audited_gold_sql: bool = False,
) -> dict[str, dict]:
    """Load per-task dicts from the dataset on disk via paths.py.

    When `use_audited_gold_sql=True`, applies the audited-gold overlay
    (same helper `run_evaluation` uses) so the cloud actor evaluates
    against the audited gold-SQL — not the raw unaudited dataset (Cx3).
    """
    from bird_interact_agents import paths

    data_path = paths.mini_interact_data_file()
    wanted = set(instance_ids)
    rows: list[dict] = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            td = json.loads(line)
            iid = td.get("instance_id")
            if iid in wanted:
                rows.append(td)
    if use_audited_gold_sql:
        from bird_interact_agents.harness import apply_audited_gold_overlay
        apply_audited_gold_overlay(rows, paths.audited_gold_root())
    return {td["instance_id"]: td for td in rows}


def _load_secrets_file(path: str | None) -> dict[str, str] | None:
    """Load the out-of-band secrets file (JSON dict of env vars) and delete
    it immediately to minimise secret-at-rest on the head. Returns None when
    no path is given. Missing/garbage files raise — a silently-empty env
    would make every actor's LLM call fail with an opaque auth error."""
    if not path:
        return None
    p = Path(path)
    try:
        secrets = json.loads(p.read_text())
    finally:
        # Best-effort delete even if parsing failed; the file holds secrets.
        try:
            p.unlink()
        except OSError:
            pass
    if not isinstance(secrets, dict):
        raise ValueError(f"secrets file {path} must be a JSON object")
    return {str(k): str(v) for k, v in secrets.items()}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--attempt", required=True, type=int)
    p.add_argument("--ray-job-id", default="unknown")
    p.add_argument("--framework", required=True)
    p.add_argument("--query-mode", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--agent-model", required=True)
    p.add_argument("--user-sim-model", required=True)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--use-audited-gold-sql", action="store_true")
    p.add_argument("--prompt-cache", dest="prompt_cache", action="store_true",
                   default=True)
    p.add_argument("--no-prompt-cache", dest="prompt_cache",
                   action="store_false")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--num-actors", type=int, default=4)
    p.add_argument("--slayer-setup", default="pre-encoded",
                   choices=("pre-encoded", "on-the-fly"))
    p.add_argument("--slayer-storage-root", default="/data/slayer_models")
    p.add_argument("--instance-ids", required=True,
                   help="comma-separated list")
    p.add_argument(
        "--secrets-file", default=None,
        help="path (on the head, inside the container) to a JSON file of "
             "env vars (e.g. API keys) to apply as a per-actor runtime_env. "
             "Delivered out-of-band so secrets never enter the job's "
             "runtime_env (which `ray job list` echoes).",
    )
    args = p.parse_args(argv)

    actor_env_vars = _load_secrets_file(args.secrets_file)

    instance_ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    task_data_by_id = _load_task_data(
        instance_ids, use_audited_gold_sql=args.use_audited_gold_sql,
    )

    run_pool(
        run_id=args.run_id,
        instance_ids=instance_ids,
        framework=args.framework,
        query_mode=args.query_mode,
        mode=args.mode,
        agent_model=args.agent_model,
        num_actors=args.num_actors,
        attempt=args.attempt,
        task_data_by_id=task_data_by_id,
        user_sim_model=args.user_sim_model,
        patience=args.patience,
        strict=args.strict,
        use_audited_gold_sql=args.use_audited_gold_sql,
        prompt_cache=args.prompt_cache,
        max_depth=args.max_depth,
        slayer_setup=args.slayer_setup,
        slayer_storage_root=args.slayer_storage_root,
        ray_job_id=args.ray_job_id,
        actor_env_vars=actor_env_vars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
