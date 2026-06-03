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


def test_skip_copies_stable_blobs_to_run_scoped_paths(fake_gcs_bucket, monkeypatch):
    """On skip, both stable blobs must be copied to the run-scoped row paths
    so that `fetch` can download annotation data for skipped tasks."""
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    ann_content = b'{"instance_id":"shop_1"}'
    var_content = b'{"variant_id":"primary"}\n'
    store[gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1")] = ann_content
    store[gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1")] = var_content
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _minimal_annotator_result())

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    assert store.get(gcs.task_annotation_blob("r-1", "shop_1")) == ann_content
    assert store.get(gcs.audited_gold_variants_blob("r-1", "shop_1")) == var_content


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
# run_annotator_pool → _run_with_actors kwarg contract
# ---------------------------------------------------------------------------

def test_run_annotator_pool_passes_benchmark_kwarg_to_run_with_actors(
    fake_gcs_bucket, monkeypatch
):
    """_run_with_actors must receive benchmark= so it doesn't TypeError in
    cloud actor mode (ray_app._run_with_actors requires benchmark: str)."""
    from types import SimpleNamespace
    from bird_interact_agents.cloud import ray_app
    from bird_interact_agents.cloud import ray_app_annotator

    captured_kwargs: dict = {}

    class FakeHeartbeat:
        def __init__(self, *a, **kw): pass
        def start(self): pass
        def tick_done(self): pass
        def stop_and_flush(self, terminal_state): pass

    def fake_run_with_actors(**kwargs):
        captured_kwargs.update(kwargs)

    fake_actor = SimpleNamespace()
    fake_actor_cls = SimpleNamespace(remote=lambda *a, **kw: fake_actor)

    monkeypatch.setattr(ray_app, "HeartbeatWriter", FakeHeartbeat)
    monkeypatch.setattr(ray_app, "_run_with_actors", fake_run_with_actors)
    monkeypatch.setattr(ray_app, "_with_actor_env", lambda cls, env: cls)
    monkeypatch.setattr(ray_app_annotator, "_build_annotator_actor_class",
                        lambda: fake_actor_cls)

    client, _ = fake_gcs_bucket
    ray_app_annotator.run_annotator_pool(
        run_id="r-1",
        instance_ids=[],
        task_data_by_id={},
        cfg=_cfg(),
        data_path_base="/tmp",
        num_actors=1,
        gcs_client=client,
        local_only=False,
    )

    assert "benchmark" in captured_kwargs, "_run_with_actors not called with benchmark kwarg"
    assert captured_kwargs["benchmark"] == "mini_interact"


def test_annotator_actor_init_calls_download_benchmark_data(monkeypatch):
    """AnnotatorActor.__init__ must call download_benchmark_data so cloud
    workers download the benchmark dataset before running any tasks."""
    import sys
    from types import SimpleNamespace
    from bird_interact_agents.cloud import ray_app_annotator

    fake_ray = SimpleNamespace(remote=lambda cls: cls)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    download_calls: list = []
    monkeypatch.setattr(ray_app_annotator, "download_benchmark_data",
                        lambda cfg, client=None: download_calls.append(cfg))
    monkeypatch.setattr(ray_app_annotator, "default_gcs_client", lambda: None)

    ActorCls = ray_app_annotator._build_annotator_actor_class()
    cfg = {**_cfg(), "dataset": "mini_interact", "benchmark_data_prefix": "gs://b/prefix"}
    ActorCls(cfg=cfg, run_id="r-1", data_path_base="/tmp")

    assert len(download_calls) == 1, "download_benchmark_data must be called in __init__"
    assert download_calls[0].get("benchmark_data_prefix") == "gs://b/prefix"


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
