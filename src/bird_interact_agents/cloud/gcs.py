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
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


from bird_interact_agents.cloud import config


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


def concurrent_download_prefix(
    run_id: str,
    dest: Path,
    *,
    max_workers: int = 32,
    client=None,
) -> None:
    """Download every blob under `runs/<run_id>/` to `dest/<same-relative-path>`.

    SDK-only; no `gsutil`. Idempotent — overwrites existing files with the
    latest bytes from GCS, which is the semantics `fetch` needs.
    """
    client = client or default_gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"runs/{run_id}/"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def _one(blob: Any) -> None:
        rel = blob.name[len(prefix):]
        if not rel:
            return
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = blob.download_as_bytes()
        target.write_bytes(data)

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        return
    # Cap concurrency at the actual blob count.
    workers = max(1, min(max_workers, len(blobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, blobs))


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
