"""DEV-1515: implicit (auto-synthesized) ``TaskAnnotation``.

For benchmark instances without a human-authored task annotation file on
disk, the grader needs a schema-valid placeholder so the cascade can
still run. The implicit default treats the original gold as the sole
correct answer (``verdict=sufficient`` + ``original_gold_is_correct=True``);
the cascade collapses to N1 and the LLM judge never fires.

The factory is a pure builder — it never writes to disk and never reads
from ``paths.annotations_root()``. Callers are responsible for keeping
implicit annotations in memory only.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.eval.annotation_schema import (
    MetadataSufficiency,
    Provenance,
    TaskAnnotation,
)

IMPLICIT_ANNOTATED_BY = "auto-implicit"
"""Sentinel value for ``TaskAnnotation.annotated_by`` that downstream
consumers can use to distinguish synthesized-on-the-fly annotations from
human-authored ones."""


def _benchmark_task_jsonl_name(benchmark: str) -> str:
    """Return the basename of the benchmark's task JSONL for provenance.

    Falls back to a generic ``"<benchmark>.jsonl"`` placeholder when the
    benchmark token doesn't match a registered ``Benchmark`` descriptor
    (which is fine for tests / forks that haven't registered theirs)."""
    try:
        return get_benchmark(benchmark).data_file
    except ValueError:
        return f"{benchmark}.jsonl"


def implicit_task_annotation(
    *,
    instance_id: str,
    selected_database: str,
    benchmark: str,
    amb_user_query: str = "",
    annotated_at: Optional[str] = None,
) -> TaskAnnotation:
    """Build a schema-valid placeholder ``TaskAnnotation`` for an
    instance that has no on-disk annotation yet.

    Used by the tolerant grader when it cannot find
    ``annotations/<benchmark>/<db>/<instance_id>.task.json``. Returns
    immediately — never touches disk.
    """
    if annotated_at is None:
        annotated_at = (
            _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
        )
    return TaskAnnotation(
        instance_id=instance_id,
        selected_database=selected_database,
        annotated_by=IMPLICIT_ANNOTATED_BY,
        annotated_at=annotated_at,
        amb_user_query=amb_user_query,
        external_knowledge=[],
        masked_terms=[],
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient",
            rationale="implicit default — no annotation file on disk",
            evidence_sources_consulted=[],
        ),
        original_gold_is_correct=True,
        gold_variants=[],
        evaluator_prompt=None,
        provenance=Provenance(
            task_jsonl_path=_benchmark_task_jsonl_name(benchmark),
            task_jsonl_instance_id=instance_id,
            audited_gold_legacy_path=None,
        ),
    )
