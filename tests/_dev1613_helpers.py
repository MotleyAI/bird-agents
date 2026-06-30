"""Shared fixtures for the DEV-1613 test modules (NOT a test module).

Kept out of ``conftest.py`` so it can be imported explicitly as
``from tests._dev1613_helpers import _make_task_annotation``.
"""
from __future__ import annotations

from typing import Optional


def _make_task_annotation(
    *,
    verdict: str = "insufficient",
    evaluator_prompt: Optional[str] = "rules",
    instance_id: str = "alien_1",
):
    """Build a TaskAnnotation mirroring the orchestration-test fixtures."""
    from bird_interact_agents.eval import (
        AuditedGoldRef,
        GoldVariantRef,
        MetadataSufficiency,
        TaskAnnotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance

    return TaskAnnotation(
        instance_id=instance_id,
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-05-31",
        amb_user_query="x",
        metadata_sufficiency=MetadataSufficiency(verdict=verdict, rationale="r"),
        gold_variants=[
            GoldVariantRef(
                variant_id="primary",
                interpretation="x",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id=instance_id,
                    variant_id="primary",
                ),
            ),
        ],
        evaluator_prompt=evaluator_prompt,
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=instance_id,
        ),
    )
