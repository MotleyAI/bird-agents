"""Integration probes I1–I7 for SPEC_DEV-1453.

These are GATED — they're skipped unless `BIRD_INTERACT_CLOUD_SMOKE=1` is
exported. They hit real GCP (project `motley-team-475011`) and spin up real
VMs.  Cost-bearing; never run in CI.

Run with:

    BIRD_INTERACT_CLOUD_SMOKE=1 pytest tests/integration/test_cloud_smoke.py -x

Each test cleans up after itself via `bird-interact-cloud kill <run-id>`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

GATE = os.environ.get("BIRD_INTERACT_CLOUD_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not GATE, reason="Set BIRD_INTERACT_CLOUD_SMOKE=1 to enable cloud probes",
)


# ---------------------------------------------------------------------------
# I1 — --detach returns before the Ray job completes.
# ---------------------------------------------------------------------------


def test_i1_detached_returns_fast(tmp_path) -> None:
    t0 = time.perf_counter()
    # Bound the submit subprocess so a hung CLI doesn't stall the probe
    # indefinitely. 5 min is the upper bound on bring-up + image push.
    proc = subprocess.run(
        [
            "bird-interact-cloud", "submit", "--detach",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "households_5",
            "--mode", "c-interact",
            "--workers", "1", "--actors-per-worker", "1",
        ],
        capture_output=True, text=True, timeout=300, check=True,
    )
    out = proc.stdout or ""
    elapsed = time.perf_counter() - t0

    # Resilient run_id extraction. The CLI prints `submitted: <run-id>`;
    # parse explicitly so an unexpected stdout shape doesn't crash before
    # the try/finally cleanup gets a chance to fire.
    run_id = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("submitted:"):
            parts = line.split(":", 1)[1].split()
            if parts:
                run_id = parts[0]
                break
    if not run_id:
        # Fall back to "last token" but still inside a try so cleanup
        # runs if even that fails. Empty run_id => skip cleanup gracefully.
        try:
            run_id = out.strip().split()[-1]
        except IndexError:
            pytest.fail(f"submit produced no parseable run_id; stdout={out!r}")

    try:
        assert elapsed < 300, f"submit --detach took {elapsed:.0f}s"
        # status.json may take a few seconds to land after detach returns.
        # Poll up to 60s for it to appear, then assert it's mid-run.
        status = None
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                status_raw = subprocess.check_output(
                    [
                        "gcloud", "storage", "cat",
                        f"gs://motley-team-birdbench/runs/{run_id}/status.json",
                    ],
                    text=True, timeout=10,
                )
                status = json.loads(status_raw)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                time.sleep(5)
        assert status is not None, "status.json never appeared within 60s"
        assert status.get("terminal_state") is None
        assert status.get("ray_job_id")
    finally:
        subprocess.run(
            ["bird-interact-cloud", "kill", run_id],
            check=False, timeout=300,
        )


# ---------------------------------------------------------------------------
# I2 — API key visible inside the worker via runtime_env.
# ---------------------------------------------------------------------------


def test_i2_api_key_visible_in_worker(tmp_path) -> None:
    """Submit a tiny `--instance-ids households_5` run with attach, then
    inspect the captured per-task log for evidence that ANTHROPIC_API_KEY
    was visible. The log is uploaded by the actor at task end."""
    # Implementation note: the test asserts a sentinel that the actor
    # emits when the key is present, NOT the key value itself.
    pytest.skip("Implement once driver supports `--probe-env` sentinel emission")


# ---------------------------------------------------------------------------
# I3 — GCS write via metadata-server credentials (no key files).
# ---------------------------------------------------------------------------


def test_i3_worker_writes_gcs_via_metadata_creds() -> None:
    pytest.skip("Implement once `bird-interact-cloud probe gcs` is available")


# ---------------------------------------------------------------------------
# I4 — Image pull uses the worker SA, not the user's docker login.
# ---------------------------------------------------------------------------


def test_i4_image_pull_uses_worker_sa() -> None:
    pytest.skip("Implement via Artifact Registry audit-log inspection")


# ---------------------------------------------------------------------------
# I5 — kill works from a machine without ~/.bird-interact-cloud/<id>.yaml.
# ---------------------------------------------------------------------------


def test_i5_kill_without_cached_yaml(tmp_path) -> None:
    pytest.skip(
        "Set HOME to an empty tmp dir; submit + kill; assert kill succeeds"
    )


# ---------------------------------------------------------------------------
# I6 — Self-delete deletes VM AND boot disk after the safety timer.
# ---------------------------------------------------------------------------


def test_i6_self_delete_removes_disks() -> None:
    pytest.skip(
        "Submit with --max-runtime-hours 0.05; wait 5 minutes; list instances + disks"
    )


# ---------------------------------------------------------------------------
# I7 — End-to-end smoke on one task.
# ---------------------------------------------------------------------------


def test_i7_e2e_one_task() -> None:
    pytest.skip(
        "End-to-end smoke: submit --instance-ids households_5; tear down; "
        "assert eval.json matches local `bird-interact run` on the same iid"
    )
