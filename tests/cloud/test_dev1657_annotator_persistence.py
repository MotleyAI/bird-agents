"""DEV-1657: the annotator worker must persist the trajectory + usage for
EVERY outcome (annotated / error), not just the terminal status row.

Before this, a never-submitted GLM-5.2 run left only a 5-field summary row and
an all-zero usage block — impossible to diagnose. Now:

* `runs/<run_id>/rows/<iid>/attempt-N.trajectory.json` carries the serialized
  SDK message stream (auto-downloaded by `fetch`'s whole-prefix pull).
* The attempt row carries `usage` so `collation._build_metrics` aggregates it
  into `eval.json.total_usage`.
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_data(instance_id: str = "shop_1", db: str = "shop") -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": db,
        "amb_user_query": "How many orders?",
        "sol_sql": ["SELECT COUNT(*) FROM orders;"],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
        "knowledge_ambiguity": [],
        "external_knowledge": [],
    }


def _cfg() -> dict:
    return {
        "benchmark": "mini-interact",
        "model": "anthropic/claude-opus-4-7",
        "effort": "medium",
        "override": False,
    }


def _usage_blob() -> dict:
    from bird_interact_agents.usage import TokenUsage
    u = TokenUsage()
    u.add_call(scope="agent", model="anthropic/claude-opus-4-7", prompt=100, completion=20)
    return u.model_dump()


_TRAJECTORY = [
    {"type": "AssistantMessage", "data": {"content": "exploring the schema"}},
    {"type": "AssistantMessage", "data": {"content": "still exploring"}},
]


def _annotated_result():
    from bird_interact_agents.agents.annotator.agent import AnnotatorResult
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "shop",
        "annotated_by": "annotator-agent/test",
        "annotated_at": "2026-07-08",
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
            "task_jsonl_instance_id": "shop_1",
        },
    })
    return AnnotatorResult(
        instance_id="shop_1",
        task_annotation=ann,
        audited_gold_variants=[],
        usage=_usage_blob(),
        trajectory=list(_TRAJECTORY),
        duration_s=1.0,
    )


def _error_result(trajectory=None):
    from bird_interact_agents.agents.annotator.agent import AnnotatorResult
    return AnnotatorResult(
        instance_id="shop_1",
        task_annotation=None,
        audited_gold_variants=[],
        usage=_usage_blob(),
        trajectory=list(_TRAJECTORY) if trajectory is None else trajectory,
        duration_s=0.5,
        error="Agent did not submit an annotation after 60 turns.",
    )


_TRAJ_BLOB = "runs/r-1/rows/shop_1/attempt-1.trajectory.json"
_ROW_BLOB = "runs/r-1/rows/shop_1/attempt-1.json"


def _run_one(client, monkeypatch, result):
    from bird_interact_agents.cloud import ray_app_annotator
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: result)
    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_error_writes_trajectory_blob(fake_gcs_bucket, monkeypatch):
    client, store = fake_gcs_bucket
    _run_one(client, monkeypatch, _error_result())

    assert _TRAJ_BLOB in store
    assert json.loads(store[_TRAJ_BLOB]) == _TRAJECTORY


def test_error_attempt_row_includes_usage(fake_gcs_bucket, monkeypatch):
    client, store = fake_gcs_bucket
    _run_one(client, monkeypatch, _error_result())

    row = json.loads(store[_ROW_BLOB])
    assert row["status"] == "error"
    assert row["usage"]["prompt_tokens"] == 100
    assert row["usage"]["completion_tokens"] == 20


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_annotated_writes_trajectory_blob(fake_gcs_bucket, monkeypatch):
    client, store = fake_gcs_bucket
    _run_one(client, monkeypatch, _annotated_result())

    assert _TRAJ_BLOB in store
    assert json.loads(store[_TRAJ_BLOB]) == _TRAJECTORY


def test_annotated_attempt_row_includes_usage(fake_gcs_bucket, monkeypatch):
    client, store = fake_gcs_bucket
    _run_one(client, monkeypatch, _annotated_result())

    row = json.loads(store[_ROW_BLOB])
    assert row["status"] == "annotated"
    assert row["usage"]["prompt_tokens"] == 100


# ---------------------------------------------------------------------------
# GCS-write failure after a successful annotation still carries usage + traj
# ---------------------------------------------------------------------------

def test_gcs_write_failure_error_row_keeps_usage_and_trajectory(fake_gcs_bucket, monkeypatch):
    from bird_interact_agents.cloud import ray_app_annotator

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app_annotator, "_run_agent", lambda *a, **kw: _annotated_result())

    def _boom(*a, **kw):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(ray_app_annotator._gcs, "write_task_annotation", _boom)

    ray_app_annotator._run_one_task(
        task_data=_task_data(), cfg=_cfg(), run_id="r-1",
        data_path_base="/tmp/data", gcs_client=client,
    )

    row = json.loads(store[_ROW_BLOB])
    assert row["status"] == "error"
    assert "GCS write failed" in row["error"]
    assert row["usage"]["prompt_tokens"] == 100
    assert _TRAJ_BLOB in store


# ---------------------------------------------------------------------------
# Empty trajectory → no blob (best-effort, don't write empties)
# ---------------------------------------------------------------------------

def test_empty_trajectory_not_written(fake_gcs_bucket, monkeypatch):
    client, store = fake_gcs_bucket
    _run_one(client, monkeypatch, _error_result(trajectory=[]))

    assert _TRAJ_BLOB not in store
    # The attempt row is still written for completion tracking.
    assert _ROW_BLOB in store
