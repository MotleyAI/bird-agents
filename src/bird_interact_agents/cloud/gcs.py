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


from bird_interact_agents.cloud import config


logger = logging.getLogger(__name__)


BUCKET_NAME = config.BUCKET_NAME
BUCKET_REGION = config.REGION


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

    ``skip_missing_blobs`` (default ``False``): when True, per-blob download
    failures are logged and skipped instead of propagating. Only the
    ``fetch``-from-live-run path needs this — the cluster constantly
    rewrites hot blobs (``status.json``, heartbeat), and the SDK pins the
    generation seen at ``list_blobs`` into the download URL, so a
    concurrent write between listing and downloading raises 404 on the
    old generation. Strict callers (slayer-setup, benchmark-data) must
    leave it ``False`` so a transient failure surfaces instead of
    silently landing a partial cache.
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
        except Exception as exc:  # noqa: BLE001
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

    def _one(path: Path) -> None:
        rel = path.relative_to(local_dir).as_posix()
        blob = bucket.blob(f"{base}/{rel}")
        blob.upload_from_string(path.read_bytes())

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
    - on-the-fly + pydantic_ai_otf_encode -> ``slayer_models_otf``
    """
    if slayer_setup == "pre-encoded":
        return _ARTIFACT_PRE_ENCODED
    if framework == "pydantic_ai_otf_encode":
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
