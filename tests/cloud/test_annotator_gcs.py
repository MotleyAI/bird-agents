"""GCS blob-path helpers and write/read functions for the annotator agent (DEV-1518).

Contract:
* blob path helpers return canonical strings (run-specific and stable)
* write helpers serialise correctly into the fake bucket
* blob_exists works on presence/absence
* benchmark name normalisation (dash ↔ underscore) resolves to the same stable path
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_task_annotation_dict(instance_id: str = "shop_1", db: str = "shop") -> dict:
    return {
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "annotated_by": "annotator-agent/test",
        "annotated_at": "2026-06-02",
        "amb_user_query": "How many orders?",
        "external_knowledge": [],
        "masked_terms": [],
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 1 directly answers the question.",
            "evidence_sources_consulted": ["kb:1"],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": instance_id,
        },
    }


# ---------------------------------------------------------------------------
# Blob path helpers — run-specific
# ---------------------------------------------------------------------------

def test_task_annotation_blob_path():
    from bird_interact_agents.cloud import gcs
    assert gcs.task_annotation_blob("run-1", "shop_1") == \
        "runs/run-1/rows/shop_1/task_annotation.json"


def test_audited_gold_variants_blob_path():
    from bird_interact_agents.cloud import gcs
    assert gcs.audited_gold_variants_blob("run-1", "shop_1") == \
        "runs/run-1/rows/shop_1/audited_gold_variants.jsonl"


# ---------------------------------------------------------------------------
# Blob path helpers — stable
# ---------------------------------------------------------------------------

def test_stable_task_annotation_blob_path():
    from bird_interact_agents.cloud import gcs
    assert gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1") == \
        "annotations/mini_interact/shop/shop_1.task.json"


def test_stable_task_annotation_blob_normalises_dash_benchmark():
    from bird_interact_agents.cloud import gcs
    assert gcs.stable_task_annotation_blob("mini-interact", "shop", "shop_1") == \
        gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1")


def test_stable_audited_gold_variants_blob_path():
    from bird_interact_agents.cloud import gcs
    assert gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1") == \
        "audited_gold/mini_interact/shop/shop_1.variants.jsonl"


def test_stable_audited_gold_variants_blob_normalises_dash_benchmark():
    from bird_interact_agents.cloud import gcs
    assert gcs.stable_audited_gold_variants_blob("mini-interact", "shop", "shop_1") == \
        gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1")


# ---------------------------------------------------------------------------
# write_task_annotation — run-specific path
# ---------------------------------------------------------------------------

def test_write_task_annotation_lands_at_correct_blob(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    client, store = fake_gcs_bucket
    ann = TaskAnnotation.model_validate(_minimal_task_annotation_dict())

    gcs.write_task_annotation("run-1", "shop_1", ann, client=client)

    assert gcs.task_annotation_blob("run-1", "shop_1") in store


def test_write_task_annotation_serialises_valid_json(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    client, store = fake_gcs_bucket
    ann = TaskAnnotation.model_validate(_minimal_task_annotation_dict("shop_2", "shop"))

    gcs.write_task_annotation("run-1", "shop_2", ann, client=client)

    raw = store[gcs.task_annotation_blob("run-1", "shop_2")]
    data = json.loads(raw)
    assert data["instance_id"] == "shop_2"
    assert data["kind"] == "task_annotation"


# ---------------------------------------------------------------------------
# write_stable_task_annotation
# ---------------------------------------------------------------------------

def test_write_stable_task_annotation_lands_at_stable_blob(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    client, store = fake_gcs_bucket
    ann = TaskAnnotation.model_validate(_minimal_task_annotation_dict())

    gcs.write_stable_task_annotation("mini_interact", "shop", "shop_1", ann, client=client)

    assert gcs.stable_task_annotation_blob("mini_interact", "shop", "shop_1") in store


# ---------------------------------------------------------------------------
# write_audited_gold_variants
# ---------------------------------------------------------------------------

def test_write_audited_gold_variants_empty_creates_blob(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs

    client, store = fake_gcs_bucket
    gcs.write_audited_gold_variants("run-1", "shop_1", [], client=client)

    blob = gcs.audited_gold_variants_blob("run-1", "shop_1")
    assert blob in store
    # Empty JSONL: no non-blank lines
    lines = [l for l in store[blob].decode().splitlines() if l.strip()]
    assert lines == []


def test_write_audited_gold_variants_two_entries_two_lines(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs

    client, store = fake_gcs_bucket
    variants = [
        {"instance_id": "shop_1", "variant_id": "primary", "audit_status": "clean"},
        {"instance_id": "shop_1", "variant_id": "alt", "audit_status": "edited"},
    ]
    gcs.write_audited_gold_variants("run-1", "shop_1", variants, client=client)

    blob = gcs.audited_gold_variants_blob("run-1", "shop_1")
    lines = [l for l in store[blob].decode().splitlines() if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["variant_id"] == "primary"
    assert json.loads(lines[1])["variant_id"] == "alt"


# ---------------------------------------------------------------------------
# write_stable_audited_gold_variants
# ---------------------------------------------------------------------------

def test_write_stable_audited_gold_variants_lands_at_stable_blob(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs

    client, store = fake_gcs_bucket
    gcs.write_stable_audited_gold_variants(
        "mini_interact", "shop", "shop_1", [], client=client
    )

    assert gcs.stable_audited_gold_variants_blob("mini_interact", "shop", "shop_1") in store


# ---------------------------------------------------------------------------
# blob_exists
# ---------------------------------------------------------------------------

def test_blob_exists_true(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs

    client, store = fake_gcs_bucket
    store["some/path.json"] = b"{}"
    assert gcs.blob_exists("some/path.json", client=client) is True


def test_blob_exists_false(fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs

    client, store = fake_gcs_bucket
    assert gcs.blob_exists("nonexistent/path.json", client=client) is False
