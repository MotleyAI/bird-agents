"""GCS sink for cloud runs. SDK-only — no `gsutil` shell-outs.

Object layout (see SPEC §6.4):

    runs/<run-id>/
        manifest.json
        status.json
        rows/<instance_id>/attempt-<n>.json
        logs/<instance_id>/attempt-<n>.log
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from google.api_core.exceptions import NotFound as _GcsNotFound
except ModuleNotFoundError:  # pragma: no cover - exercised only on cloud-free installs
    # DEV-1640: the local process-pool worker imports this module (via
    # ray_app / persistence) but only ever uses LocalFsStore — it never calls
    # a real GCS function. Guarding this single top-level google import keeps
    # `gcs` importable under a cloud-free local install (README
    # `.[claude-sdk,dev]`, which lacks the `cloud` extra's google-cloud-storage)
    # so local runs don't crash every spawned worker at import time. Any REAL
    # GCS call still fails clearly at `default_gcs_client()`
    # (`from google.cloud import storage`).
    class _GcsNotFound(Exception):
        """Sentinel stand-in for google.api_core NotFound when
        google-cloud-storage is not installed."""

from bird_interact_agents.cloud import config
from bird_interact_agents.frameworks import is_otf_encode_framework


logger = logging.getLogger(__name__)


BUCKET_NAME = config.BUCKET_NAME
BUCKET_REGION = config.REGION


# ---------------------------------------------------------------------------
# DEV-1653: resumable large-artifact upload tuning.
#
# The client-default per-blob timeout (~120s, single-shot) can't finish a
# 355 MB pg_dump on a slow/flaky link — the whole upload aborts. These generous
# defaults (per-blob request timeout + an overall retry deadline) give a large
# blob room to stream, and `upload_dir_prefix` skips already-present blobs so a
# retried submit resumes at file granularity instead of restarting the tree.
# ---------------------------------------------------------------------------
_UPLOAD_TIMEOUT_S = 900.0          # per-blob request timeout (streamed upload)
_UPLOAD_RETRY_DEADLINE_S = 1800.0  # overall retry deadline per blob

try:  # pragma: no cover - trivial guard, mirrors the NotFound import above
    from google.cloud.storage.retry import DEFAULT_RETRY as _DEFAULT_RETRY

    _UPLOAD_RETRY = _DEFAULT_RETRY.with_timeout(_UPLOAD_RETRY_DEADLINE_S)
except ImportError:
    # Cloud-free local install (no google-cloud-storage). Real uploads never
    # run here; passing retry=None is harmless because no upload is issued.
    _UPLOAD_RETRY = None


def default_gcs_client():
    """Construct a real google-cloud-storage Client. Tests inject a fake
    via the `client` kwarg on every helper below."""
    from google.cloud import storage  # type: ignore[import-not-found]

    return storage.Client()


# ---------------------------------------------------------------------------
# Object-name builders
# ---------------------------------------------------------------------------


def manifest_blob(run_id: str) -> str:
    return f"runs/{run_id}/manifest.json"


def status_blob(run_id: str) -> str:
    return f"runs/{run_id}/status.json"


def row_blob(run_id: str, instance_id: str, attempt: int) -> str:
    return f"runs/{run_id}/rows/{instance_id}/attempt-{attempt}.json"


def log_blob(run_id: str, instance_id: str, attempt: int) -> str:
    return f"runs/{run_id}/logs/{instance_id}/attempt-{attempt}.log"


def submission_annotation_blob(run_id: str, instance_id: str) -> str:
    """DEV-1515: per-task submission annotation blob path. One file per
    (run, instance); the in-cloud grader writes it once per task; the
    fetch path downloads + merges to ``<main_checkout>/annotations/``."""
    return f"runs/{run_id}/rows/{instance_id}/submission_annotation.json"


def _normalise_benchmark(benchmark: str) -> str:
    return benchmark.replace("-", "_")


def partial_transcript_blob(run_id: str, instance_id: str) -> str:
    """In-flight claude_sdk transcript for a task, uploaded on a throttle WHILE
    the task runs (not just at completion) so a hung/slow task is inspectable
    from the laptop. Co-located under ``rows/<iid>/`` so the fetch path picks it
    up with everything else."""
    return f"runs/{run_id}/rows/{instance_id}/partial_transcript.jsonl"


def diagnostics_blob(run_id: str) -> str:
    """Head-node diagnostics dump captured when a non-detached run ends in a
    non-clean terminal state (stalled / timed-out). Lives at the run root so
    the fetch path downloads it alongside ``status.json`` / ``manifest.json``,
    making the head-node evidence inspectable after teardown destroys the VM."""
    return f"runs/{run_id}/diagnostics.txt"


def task_annotation_blob(run_id: str, instance_id: str) -> str:
    """DEV-1518: run-specific task annotation blob path."""
    return f"runs/{run_id}/rows/{instance_id}/task_annotation.json"


def audited_gold_variants_blob(run_id: str, instance_id: str) -> str:
    """DEV-1518: run-specific audited gold variants blob path (JSONL)."""
    return f"runs/{run_id}/rows/{instance_id}/audited_gold_variants.jsonl"


def stable_task_annotation_blob(benchmark: str, db: str, instance_id: str) -> str:
    """DEV-1518: stable (cross-run) task annotation blob path."""
    bm = _normalise_benchmark(benchmark)
    return f"annotations/{bm}/{db}/{instance_id}.task.json"


def stable_audited_gold_variants_blob(benchmark: str, db: str, instance_id: str) -> str:
    """DEV-1518: stable (cross-run) audited gold variants blob path."""
    bm = _normalise_benchmark(benchmark)
    return f"audited_gold/{bm}/{db}/{instance_id}.variants.jsonl"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_row(
    run_id: str,
    instance_id: str,
    attempt: int,
    row: dict,
    *,
    client=None,
) -> None:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(row_blob(run_id, instance_id, attempt))
    blob.upload_from_string(
        json.dumps(row).encode(), content_type="application/json"
    )


def write_log(
    run_id: str,
    instance_id: str,
    attempt: int,
    text: bytes,
    *,
    client=None,
) -> None:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(log_blob(run_id, instance_id, attempt))
    blob.upload_from_string(text, content_type="text/plain")


def write_partial_transcript(
    run_id: str,
    instance_id: str,
    text: str,
    *,
    client=None,
) -> None:
    """Upload the (growing) in-flight transcript JSONL for one task. Called on
    a throttle while the task runs, so it gets overwritten with a fuller
    version each time and the final upload is the complete in-flight view."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        partial_transcript_blob(run_id, instance_id)
    )
    blob.upload_from_string(text, content_type="application/x-ndjson")


def read_partial_transcript(
    run_id: str, instance_id: str, *, client=None,
) -> "str | None":
    """Return the uploaded in-flight transcript JSONL for one task, or None if
    it was never written (task finished before the first throttled upload, or
    never started)."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        partial_transcript_blob(run_id, instance_id)
    )
    try:
        if not blob.exists():
            return None
        return blob.download_as_bytes().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — best-effort diagnostics read
        return None


def partial_transcript_updated_ts(
    run_id: str, instance_id: str, *, client=None,
) -> "float | None":
    """Epoch seconds of the last write to a task's in-flight partial transcript,
    or None if it was never written. A streaming task refreshes this every
    ~throttle seconds, so it is the forward-progress signal `wait_until_done`
    uses to tell a slow-but-healthy task from a wedged one."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        partial_transcript_blob(run_id, instance_id)
    )
    try:
        blob.reload()
        updated = blob.updated
        return updated.timestamp() if updated is not None else None
    except Exception:  # noqa: BLE001 — best-effort progress probe
        return None


def write_submission_annotation(
    run_id: str,
    instance_id: str,
    annotation: dict,
    *,
    client=None,
) -> None:
    """DEV-1515: upload a per-task SubmissionAnnotation. The cloud
    worker calls this once per task right after ``grade_and_write``."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        submission_annotation_blob(run_id, instance_id),
    )
    blob.upload_from_string(
        json.dumps(annotation, indent=2).encode(),
        content_type="application/json",
    )


def read_submission_annotation(
    run_id: str,
    instance_id: str,
    *,
    client=None,
) -> dict:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        submission_annotation_blob(run_id, instance_id),
    )
    return json.loads(blob.download_as_bytes())


def write_manifest(run_id: str, manifest: dict, *, client=None) -> None:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(manifest_blob(run_id))
    blob.upload_from_string(
        json.dumps(manifest, indent=2).encode(), content_type="application/json"
    )


def write_status(run_id: str, status: dict, *, client=None) -> None:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(status_blob(run_id))
    blob.upload_from_string(
        json.dumps(status).encode(), content_type="application/json"
    )


def write_diagnostics(run_id: str, text: str, *, client=None) -> None:
    """Persist a head-node diagnostics dump for `run_id` (see diagnostics_blob)."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(diagnostics_blob(run_id))
    blob.upload_from_string(text.encode(), content_type="text/plain")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _attempt_re() -> re.Pattern[str]:
    return re.compile(r"^runs/[^/]+/rows/(?P<iid>[^/]+)/attempt-(?P<n>\d+)\.json$")


def list_attempts(run_id: str, *, client=None) -> dict[str, list[int]]:
    """Return `{iid: [attempt_n, ...]}` (sorted ascending per iid)."""
    client = client or default_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"runs/{run_id}/rows/"
    result: dict[str, list[int]] = {}
    pat = _attempt_re()
    for blob in bucket.list_blobs(prefix=prefix):
        m = pat.match(blob.name)
        if not m:
            continue
        result.setdefault(m.group("iid"), []).append(int(m.group("n")))
    for iid in result:
        result[iid].sort()
    return result


def latest_attempt(run_id: str, instance_id: str, *, client=None) -> int | None:
    attempts = list_attempts(run_id, client=client)
    lst = attempts.get(instance_id)
    return lst[-1] if lst else None


def read_row(
    run_id: str,
    instance_id: str,
    attempt: int,
    *,
    client=None,
) -> dict:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(row_blob(run_id, instance_id, attempt))
    return json.loads(blob.download_as_bytes())


def read_manifest(run_id: str, *, client=None) -> dict:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(manifest_blob(run_id))
    return json.loads(blob.download_as_bytes())


def read_status(run_id: str, *, client=None) -> dict | None:
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(status_blob(run_id))
    try:
        return json.loads(blob.download_as_bytes())
    except Exception:  # noqa: BLE001 — missing object is the common "no status yet"
        return None


# ---------------------------------------------------------------------------
# Bulk download (replaces `gsutil -m rsync`)
# ---------------------------------------------------------------------------


def download_prefix(
    prefix: str,
    dest: Path,
    *,
    max_workers: int = 32,
    client=None,
    skip_missing_blobs: bool = False,
) -> None:
    """Download every blob under `prefix` to `dest/<path-after-prefix>`.

    SDK-only; no `gsutil`. Idempotent — overwrites existing files with the
    latest bytes from GCS. Generalises `concurrent_download_prefix`: the
    `prefix` is stripped from each blob name to form the local relative path,
    so a blob `runs/<id>/slayer_setup/slayer_otf_cache/<db>/x.yaml` downloaded
    with `prefix='runs/<id>/slayer_setup/slayer_otf_cache/'` lands at
    `dest/<db>/x.yaml`.

    ``skip_missing_blobs`` (default ``False``): when True, ``NotFound``
    (404) errors raised mid-download are logged and skipped instead of
    propagating. Only the ``fetch``-from-live-run path needs this — the
    cluster constantly rewrites hot blobs (``status.json``, heartbeat),
    and the SDK pins the generation seen at ``list_blobs`` into the
    download URL, so a concurrent write between listing and downloading
    raises ``NotFound`` on the old generation. The swallow is
    deliberately narrowed to ``NotFound``: transient network / auth /
    5xx errors still propagate even when the flag is on, so the caller
    doesn't tear down the cluster on a partial download. Strict callers
    (slayer-setup, benchmark-data) leave the flag ``False`` so even
    expected-missing blobs surface instead of silently landing a
    partial cache.
    """
    client = client or default_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    dest = Path(dest)

    def _one(blob: Any) -> None:
        # Strip the prefix AND any leading slash, so a prefix passed without a
        # trailing slash (e.g. ".../db") doesn't yield an absolute "/x.yaml"
        # that would escape `dest`.
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            return
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = blob.download_as_bytes()
        except _GcsNotFound as exc:
            # Generation-pin 404: the blob was rewritten between
            # ``list_blobs`` and the per-blob download. Only swallow this
            # specific class — let auth / 5xx / network errors propagate
            # so the caller doesn't tear down the cluster on a partial
            # fetch (would lose the only copy of the missing data).
            if not skip_missing_blobs:
                raise
            logger.warning(
                "[download_prefix] skipping %s: %s: %s",
                blob.name, type(exc).__name__, exc,
            )
            return
        target.write_bytes(data)

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return
    dest.mkdir(parents=True, exist_ok=True)
    # Cap concurrency at the actual blob count.
    workers = max(1, min(max_workers, len(blobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, blobs))


def concurrent_download_prefix(
    run_id: str,
    dest: Path,
    *,
    max_workers: int = 32,
    client=None,
    skip_missing_blobs: bool = True,
) -> None:
    """Download every blob under `runs/<run_id>/` to `dest/<relative-path>`.

    Thin wrapper over `download_prefix` (one shared code path); the semantics
    `fetch` needs. ``skip_missing_blobs`` defaults to ``True`` because the
    canonical caller (``driver.fetch``) races with a live cluster that's
    constantly rewriting hot blobs — see ``download_prefix`` for the full
    rationale.
    """
    download_prefix(
        f"runs/{run_id}/", dest, max_workers=max_workers, client=client,
        skip_missing_blobs=skip_missing_blobs,
    )


def upload_dir_prefix(
    local_dir: Path,
    prefix: str,
    *,
    max_workers: int = 32,
    client=None,
    exclude=None,
    timeout: float = _UPLOAD_TIMEOUT_S,
    retry=_UPLOAD_RETRY,
) -> None:
    """Upload every file under `local_dir` to `<prefix>/<relpath>`.

    SDK-only; no `gsutil`. Mirrors `download_prefix`: a file
    `local_dir/<db>/models/x.yaml` uploaded with
    `prefix='runs/<id>/slayer_setup/slayer_models/<db>'` becomes the blob
    `runs/<id>/slayer_setup/slayer_models/<db>/models/x.yaml`. Binary files
    (e.g. `embeddings.db`) and `_`-prefixed marker files are shipped verbatim.

    `exclude`, when given, is a predicate `(rel_path: Path) -> bool` applied to
    each file's path RELATIVE to `local_dir`; matching files are skipped. Used
    by the benchmark-data upload to drop `.git/` so the GCS tree matches the
    content hash (both exclude VCS metadata).

    DEV-1653 — resumable + generous timeout for large/flaky uploads:

    * **Skip-existing.** The prefix is listed ONCE up front and any blob already
      present with a matching byte size is skipped, so a retried upload resumes
      at file granularity instead of re-sending the whole tree. Size-only is
      exact here because both callers use immutable prefixes — benchmark-data is
      content-hashed (`benchmark-data/<b>/<hash>/`) and slayer-setup is under a
      unique `runs/<run-id>/`; same name+size ⟺ same content. The check is
      best-effort (assumes an immutable prefix and a single writer); a
      concurrent second writer would at worst re-send a byte-identical blob.
    * **Streamed, tunable upload.** Files stream via `upload_from_filename`
      (resumable for large blobs) with a generous per-blob `timeout` and a
      `retry` carrying a long deadline — a 355 MB pg_dump can't finish inside
      the client-default ~120s single-shot PUT on a slow link.

    Content-type is left to the SDK's extension guess (objects here are only
    ever consumed via `download_as_bytes`, so it is irrelevant); this is if
    anything an improvement over the old single default that mislabeled binary
    blobs.
    """
    client = client or default_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    local_dir = Path(local_dir)
    base = prefix.rstrip("/")

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    if exclude is not None:
        files = [p for p in files if not exclude(p.relative_to(local_dir))]
    if not files:
        return

    # List the destination ONCE (before any upload): {blob_name: size} already
    # present under this prefix. A list failure propagates here, aborting the
    # upload before a single blob is sent (so a caller's completeness marker is
    # never written on a failed attempt). The trailing slash scopes the listing
    # to THIS base, not a sibling prefix sharing a name stem.
    existing = {b.name: b.size for b in bucket.list_blobs(prefix=base + "/")}

    def _one(path: Path) -> None:
        rel = path.relative_to(local_dir).as_posix()
        name = f"{base}/{rel}"
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            # The file vanished between the dir-walk and the stat — e.g. a
            # transient SQLite WAL sidecar removed when its DB closed. Not real
            # content; skip rather than abort the whole upload.
            return
        if existing.get(name) == size:
            # Already uploaded with a matching size — resumable skip.
            return
        try:
            bucket.blob(name).upload_from_filename(
                str(path), timeout=timeout, retry=retry,
            )
        except FileNotFoundError:
            # Vanished between the stat and the streamed upload — same belt as
            # above for anything that disappears under us mid-transfer.
            return

    workers = max(1, min(max_workers, len(files)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, files))


# ---------------------------------------------------------------------------
# SLayer setup delivery (DEV-1468)
# ---------------------------------------------------------------------------

# The per-combo GCS artifact-dir name under runs/<id>/slayer_setup/. Single
# source of truth shared by the driver (upload) and the in-cluster actor
# (download).
_ARTIFACT_PRE_ENCODED = "slayer_models"
_ARTIFACT_OTF_CACHE = "slayer_otf_cache"
_ARTIFACT_OTF_REFERENCE = "slayer_models_otf"


def slayer_artifact_name(slayer_setup: str, framework: str) -> str:
    """Map a (slayer_setup, framework) combo to its GCS artifact-dir name.

    - pre-encoded (any framework) -> ``slayer_models``
    - on-the-fly + pydantic_ai_recursive -> ``slayer_otf_cache``
    - on-the-fly + an OTF encode framework -> ``slayer_models_otf``
      (``claude_sdk_otf_encode`` / legacy ``pydantic_ai_otf_encode``)
    """
    if slayer_setup == "pre-encoded":
        return _ARTIFACT_PRE_ENCODED
    if is_otf_encode_framework(framework):
        return _ARTIFACT_OTF_REFERENCE
    return _ARTIFACT_OTF_CACHE


def slayer_setup_prefix(run_id: str, artifact: str, db: str | None = None) -> str:
    """GCS prefix for an uploaded slayer-setup artifact dir (optionally for one
    db): ``runs/<run_id>/slayer_setup/<artifact>[/<db>]``."""
    base = f"runs/{run_id}/slayer_setup/{artifact}"
    return f"{base}/{db}" if db else base


# ---------------------------------------------------------------------------
# Bucket / lifecycle
# ---------------------------------------------------------------------------


def ensure_bucket(
    name: str = BUCKET_NAME,
    region: str = BUCKET_REGION,
    worker_sa_email: str | None = None,
    *,
    client=None,
) -> None:
    """Create the bucket (if absent) with a 30-day lifecycle on `runs/`
    and the worker SA's `roles/storage.objectUser` bound. Idempotent.

    The lifecycle and IAM binding are enforced EVERY call — not just on
    create — so an existing bucket without the rule converges to the
    intended contract on next submit (CR#6)."""
    client = client or default_gcs_client()
    try:
        bucket = client.get_bucket(name)
    except Exception:  # noqa: BLE001 — NotFound from SDK
        bucket = client.create_bucket(name, location=region)

    desired_rule = {
        "action": {"type": "Delete"},
        "condition": {"age": 30, "matchesPrefix": ["runs/"]},
    }
    current_rules = list(bucket.lifecycle_rules or [])
    if desired_rule not in current_rules:
        bucket.lifecycle_rules = current_rules + [desired_rule]
        bucket.patch()

    if worker_sa_email:
        policy = bucket.get_iam_policy(requested_policy_version=3)
        member = f"serviceAccount:{worker_sa_email}"
        wanted = "roles/storage.objectUser"
        already_bound = any(
            b["role"] == wanted and member in b["members"]
            for b in policy.bindings
        )
        if not already_bound:
            policy.bindings.append({"role": wanted, "members": {member}})
            bucket.set_iam_policy(policy)


# ---------------------------------------------------------------------------
# DEV-1518: annotator GCS helpers
# ---------------------------------------------------------------------------


def blob_exists(blob_name: str, *, client=None) -> bool:
    """Return True iff the blob exists in the bucket."""
    client = client or default_gcs_client()
    return client.bucket(BUCKET_NAME).blob(blob_name).exists()


def write_task_annotation(
    run_id: str,
    instance_id: str,
    annotation,
    *,
    client=None,
) -> None:
    """Write a TaskAnnotation to the run-specific blob path."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(task_annotation_blob(run_id, instance_id))
    blob.upload_from_string(
        annotation.model_dump_json(indent=2).encode(),
        content_type="application/json",
    )


def write_stable_task_annotation(
    benchmark: str,
    db: str,
    instance_id: str,
    annotation,
    *,
    client=None,
) -> None:
    """Write a TaskAnnotation to the stable (cross-run) blob path."""
    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        stable_task_annotation_blob(benchmark, db, instance_id)
    )
    blob.upload_from_string(
        annotation.model_dump_json(indent=2).encode(),
        content_type="application/json",
    )


def write_audited_gold_variants(
    run_id: str,
    instance_id: str,
    variants: list[dict],
    *,
    benchmark: str,
    selected_database: str,
    client=None,
) -> None:
    """Write audited gold variants as JSONL to the run-specific blob path.

    Writes a single ``AuditedGoldRow`` JSON line when variants is non-empty,
    or an empty file when variants is empty (original_gold_is_correct=True).
    """
    from bird_interact_agents.eval.annotation_schema import AuditedGoldRow

    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        audited_gold_variants_blob(run_id, instance_id)
    )
    if variants:
        row = AuditedGoldRow(
            instance_id=instance_id,
            selected_database=selected_database,
            benchmark=benchmark,
            variants=variants,  # AuditedGoldRow accepts dicts via model_validate coercion
        )
        content = row.model_dump_json() + "\n"
    else:
        content = ""
    blob.upload_from_string(content.encode(), content_type="application/jsonl")


def write_stable_audited_gold_variants(
    benchmark: str,
    db: str,
    instance_id: str,
    variants: list[dict],
    *,
    client=None,
) -> None:
    """Write audited gold variants as JSONL to the stable (cross-run) blob path.

    Writes a single ``AuditedGoldRow`` JSON line when variants is non-empty,
    or an empty file when variants is empty (original_gold_is_correct=True).
    """
    from bird_interact_agents.eval.annotation_schema import AuditedGoldRow

    client = client or default_gcs_client()
    blob = client.bucket(BUCKET_NAME).blob(
        stable_audited_gold_variants_blob(benchmark, db, instance_id)
    )
    if variants:
        row = AuditedGoldRow(
            instance_id=instance_id,
            selected_database=db,
            benchmark=benchmark,
            variants=variants,
        )
        content = row.model_dump_json() + "\n"
    else:
        content = ""
    blob.upload_from_string(content.encode(), content_type="application/jsonl")
