"""T32–T33: resubmit semantics — missing-iid diff + error-row counts as missing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bird_interact_agents.cloud import driver  # noqa: E402


RUN_ID = "20260521T1422-pydanticai-raw-a1b2c3"


def _build_manifest(image_uri: str, ids: list[str]) -> dict:
    return {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ids,
        "render_inputs": {
            "workers": 2,
            "actors_per_worker": 2,
            "worker_type": "e2-standard-4",
            "zone": "us-central1-a",
            "worker_sa": "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com",
            "max_runtime_hours": 4,
            "image_uri": image_uri,
        },
    }


def _patch(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mocks: dict[str, MagicMock] = {}
    for attr in ("gcs", "cluster"):
        m = MagicMock(name=attr)
        mocks[attr] = m
        monkeypatch.setattr(f"bird_interact_agents.cloud.driver.{attr}", m)
    mocks["cluster"].head_address.return_value = "ray://10.0.0.1:10001"
    mocks["cluster"].submit_job.return_value = "raysubmit_resub_001"
    mocks["cluster"].render_from_manifest.return_value = Path("/tmp/cluster.yaml")
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())
    return mocks


# ---------------------------------------------------------------------------
# T32 — resubmit runs only the missing ids with attempt = max + 1.
# ---------------------------------------------------------------------------


def test_resubmit_runs_only_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage map by iid (attempt → row):
        db_a_1  1=ok                     → done
        db_a_2  no attempts               → missing
        db_a_3  1=err, 2=ok               → done (latest wins)
        db_a_4  1=err                     → missing (T33 — error only)
    """
    image_uri = "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
    manifest = _build_manifest(image_uri, ["db_a_1", "db_a_2", "db_a_3", "db_a_4"])
    mocks = _patch(monkeypatch)
    mocks["gcs"].read_manifest.return_value = manifest
    mocks["gcs"].list_attempts.return_value = {
        "db_a_1": [1],
        "db_a_3": [1, 2],
        "db_a_4": [1],
    }

    def fake_read_row(_run_id, iid, attempt, **_kw):
        table = {
            ("db_a_1", 1): {"instance_id": "db_a_1", "error": None},
            ("db_a_3", 1): {"instance_id": "db_a_3", "error": "boom"},
            ("db_a_3", 2): {"instance_id": "db_a_3", "error": None},
            ("db_a_4", 1): {"instance_id": "db_a_4", "error": "boom"},
        }
        try:
            return table[(iid, attempt)]
        except KeyError:
            raise KeyError(f"{iid}/attempt-{attempt}")

    mocks["gcs"].read_row.side_effect = fake_read_row

    driver.resubmit(RUN_ID)

    mocks["cluster"].submit_job.assert_called_once()
    call = mocks["cluster"].submit_job.call_args
    job_args = (
        call.kwargs.get("args")
        if "args" in call.kwargs
        else (call.args[1] if len(call.args) > 1 else [])
    ) or []
    flat = " ".join(job_args)
    # Missing = {db_a_2, db_a_4}; both must appear in --instance-ids.
    assert "db_a_2" in flat
    assert "db_a_4" in flat
    # Already-done ids must NOT be in the dispatch.
    assert "db_a_1" not in flat
    assert "db_a_3" not in flat
    # New attempt = max(existing attempts) + 1 = 3 (because db_a_3 has [1,2]).
    assert "--attempt" in flat
    assert " 3" in flat or flat.endswith("3")
    # Pinned to the original image URI (via cluster.render_from_manifest).
    mocks["cluster"].render_from_manifest.assert_called()
    rendered_manifest = (
        mocks["cluster"].render_from_manifest.call_args.args[0]
        if mocks["cluster"].render_from_manifest.call_args.args
        else mocks["cluster"].render_from_manifest.call_args.kwargs.get("manifest", {})
    )
    assert rendered_manifest["render_inputs"]["image_uri"] == image_uri


# ---------------------------------------------------------------------------
# T33 — an iid whose only attempt is an error row is treated as missing.
# ---------------------------------------------------------------------------


def test_resubmit_treats_error_only_iid_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_uri = "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
    manifest = _build_manifest(image_uri, ["db_a_1"])
    mocks = _patch(monkeypatch)
    mocks["gcs"].read_manifest.return_value = manifest
    mocks["gcs"].list_attempts.return_value = {"db_a_1": [1]}
    mocks["gcs"].read_row.return_value = {"instance_id": "db_a_1", "error": "boom"}

    driver.resubmit(RUN_ID)
    # The submission happens — the error-only row did NOT short-circuit.
    mocks["cluster"].submit_job.assert_called_once()
    call = mocks["cluster"].submit_job.call_args
    job_args = (
        call.kwargs.get("args")
        if "args" in call.kwargs
        else (call.args[1] if len(call.args) > 1 else [])
    ) or []
    flat = " ".join(job_args)
    assert "db_a_1" in flat


def test_resubmit_short_circuits_when_nothing_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_uri = "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
    manifest = _build_manifest(image_uri, ["db_a_1"])
    mocks = _patch(monkeypatch)
    mocks["gcs"].read_manifest.return_value = manifest
    mocks["gcs"].list_attempts.return_value = {"db_a_1": [1]}
    mocks["gcs"].read_row.return_value = {"instance_id": "db_a_1", "error": None}

    driver.resubmit(RUN_ID)
    mocks["cluster"].up.assert_not_called()
    mocks["cluster"].submit_job.assert_not_called()


def test_resubmit_treats_latest_error_after_earlier_success_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An iid where attempt 1 succeeded but attempt 2 (the latest) errored
    is treated as missing — the latest attempt is canonical."""
    image_uri = "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
    manifest = _build_manifest(image_uri, ["db_a_1"])
    mocks = _patch(monkeypatch)
    mocks["gcs"].read_manifest.return_value = manifest
    mocks["gcs"].list_attempts.return_value = {"db_a_1": [1, 2]}

    def fake_read_row(_run_id, iid, attempt, **_kw):
        if attempt == 1:
            return {"instance_id": iid, "error": None}
        return {"instance_id": iid, "error": "regressed"}

    mocks["gcs"].read_row.side_effect = fake_read_row

    driver.resubmit(RUN_ID)
    mocks["cluster"].submit_job.assert_called_once()
    call = mocks["cluster"].submit_job.call_args
    job_args = (
        call.kwargs.get("args")
        if "args" in call.kwargs
        else (call.args[1] if len(call.args) > 1 else [])
    ) or []
    flat = " ".join(job_args)
    assert "db_a_1" in flat
    # New attempt = max(existing) + 1 = 3
    assert "--attempt 3" in flat or " 3" in flat
