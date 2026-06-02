"""Tests for ray_app_annotator worker logic (DEV-1518).

Contract:
* Skip check requires BOTH stable task annotation AND stable audited-gold
  variants blobs; if only one is present the task must run.
* --override bypasses the skip check.
* Every outcome (skip / success / error) writes attempt-1.json so the
  existing list_attempts() / wait_until_done() completion tracking works.
* On success: run-specific task_annotation.json, audited_gold_variants.jsonl,
  AND both stable blobs are written.
* On error: no annotation or stable blobs are written.
* cluster.submit_job accepts a ray_app_path kwarg so annotator can pass
  ray_app_annotator.py instead of ray_app.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_data(instance_id: str = "shop_1", db: str = "shop") -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": db,
        "amb_user_query": "How many orders?",
        "sol_sql": ["SELECT COUNT(*) FROM orders;"],
        "user_query_ambiguity": {
            "critical_ambiguity": [],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
        "external_knowledge": [],
    }


def _minimal_annotator_result(instance_id: str = "shop_1", db: str = "shop"):
    from bird_interact_agents.agents.annotator.agent import AnnotatorResult
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "annotated_by": "annotator-agent/test",
        "annotated_at": "2026-06-02",
        "amb_user_query": "How many orders?",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 1.",
            "evidence_sources_consulted": ["kb:1"],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": instance_id,
        },
    })
    return AnnotatorResult(
        instance_id=instance_id,
        task_annotation=ann,
        audited_gold_variants=[],
        usage={},
        duration_s=1.0,
    )


def _error_result(instance_id: str = "shop_1"):
    from bird_interact_agents.agents.annotator.agent import AnnotatorResult
    return AnnotatorResult(
        instance_id=instance_id,
        task_annotation=None,
        audited_gold_variants=[],
        usage={},
        duration_s=0.3,
        error="agent timed out after max turns",
    )


def _cfg(override: bool = False) -> dict:
    return {
        "benchmark": "mini_interact",
        "model": "anthropic/claude-opus-4-7",
        "effort": "medium",
        "override": override,
    }


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def test_skip_when_both_stable_blobs_exist(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    store[gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1")] = b"{}"
    store[gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1")] = b""

    agent_calls = []
    monkeypatch.setattr(
        ray_app_annotator, "_run_agent",
        lambda *a, **kw: (agent_calls.append(1), _minimal_annotator_result())[1],
    )

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert agent_calls == []  # was skipped


def test_skip_writes_attempt_row_with_skipped_status(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    store[gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1")] = b"{}"
    store[gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1")] = b""
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _minimal_annotator_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    attempt_blob = "runs/r-1/rows/shop_1/attempt-1.json"
    assert attempt_blob in store
    row = json.loads(store[attempt_blob])
    assert row["status"] == "skipped"
    assert row["instance_id"] == "shop_1"


def test_skip_requires_both_blobs_missing_variants_blob_runs_agent(fake_gcs_bucket, monkeypatch):
    """If only task annotation exists but not variants blob, run the agent."""
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    # Only task annotation blob present; no variants blob
    store[gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1")] = b"{}"

    agent_calls = []
    monkeypatch.setattr(
        ray_app_annotator, "_run_agent",
        lambda *a, **kw: (agent_calls.append(1), _minimal_annotator_result())[1],
    )

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert agent_calls == [1]


def test_skip_requires_both_blobs_missing_annotation_blob_runs_agent(fake_gcs_bucket, monkeypatch):
    """If only variants blob exists but not annotation, run the agent."""
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    store[gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1")] = b""

    agent_calls = []
    monkeypatch.setattr(
        ray_app_annotator, "_run_agent",
        lambda *a, **kw: (agent_calls.append(1), _minimal_annotator_result())[1],
    )

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert agent_calls == [1]


def test_override_bypasses_skip(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    store[gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1")] = b"{}"
    store[gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1")] = b""

    agent_calls = []
    monkeypatch.setattr(
        ray_app_annotator, "_run_agent",
        lambda *a, **kw: (agent_calls.append(1), _minimal_annotator_result())[1],
    )

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(override=True), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert agent_calls == [1]


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_success_writes_run_specific_annotation_blob(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _minimal_annotator_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert gcs.task_annotation_blob("r-1", "shop_1") in store


def test_success_writes_run_specific_variants_blob(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _minimal_annotator_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert gcs.audited_gold_variants_blob("r-1", "shop_1") in store


def test_success_writes_both_stable_blobs(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _minimal_annotator_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1") in store
    assert gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1") in store


def test_success_writes_attempt_row_with_annotated_status(fake_gcs_bucket, monkeypatch):
    """Completion tracking requires attempt-1.json for every outcome."""
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _minimal_annotator_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    attempt_blob = "runs/r-1/rows/shop_1/attempt-1.json"
    assert attempt_blob in store
    row = json.loads(store[attempt_blob])
    assert row["status"] == "annotated"
    assert row["instance_id"] == "shop_1"
    assert "duration_s" in row


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_error_writes_attempt_row_with_error_status(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _error_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    attempt_blob = "runs/r-1/rows/shop_1/attempt-1.json"
    assert attempt_blob in store
    row = json.loads(store[attempt_blob])
    assert row["status"] == "error"
    assert "timed out" in row["error"]


def test_error_does_not_write_annotation_blobs(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _error_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert gcs.task_annotation_blob("r-1", "shop_1") not in store
    assert gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1") not in store
    assert gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1") not in store


# ---------------------------------------------------------------------------
# cluster.submit_job ray_app_path kwarg
# ---------------------------------------------------------------------------

def test_cluster_submit_job_accepts_ray_app_path_kwarg(monkeypatch):
    """submit_job must accept ray_app_path so annotator can point to ray_app_annotator.py."""
    import subprocess
    from bird_interact_agents.cloud import cluster

    captured = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **_: (
            captured.append(argv),
            SimpleNamespace(returncode=0, stdout="raysubmit_abc\n", stderr=""),
        )[1],
    )

    cluster.submit_job(
        head_address="http://localhost:8265",
        args=["--benchmark", "mini_interact"],
        env_vars={},
        ray_app_path=(
            "/app/bird-interact-agents/src/"
            "bird_interact_agents/cloud/ray_app_annotator.py"
        ),
    )

    # The submitted argv must reference the annotator app, not ray_app.py
    full_cmd = " ".join(str(a) for a in captured[-1])
    assert "ray_app_annotator.py" in full_cmd
    assert "ray_app.py" not in full_cmd.replace("ray_app_annotator.py", "")
