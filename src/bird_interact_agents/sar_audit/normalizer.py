"""Project SAR-Agent verdicts into the audited_gold superset schema row."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel


class SARVerdict(BaseModel):
    correctness_flag: bool
    ambiguity_flag: bool
    revised_sql: str | None = None
    revised_question: str | None = None
    reasoning: str = ""


class SampleRowResult(BaseModel):
    row: list | None
    status: Literal["ok", "empty", "error"]
    error: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_normalized_row(
    *,
    task: dict,
    verdict: SARVerdict,
    sample_row_result: SampleRowResult,
    audit_model_requested: str,
    audit_model_actual: str | None,
    step_count: int,
    cost_usd: float | None,
    skill_version: str,
    raw_trajectory: list | None = None,
) -> dict:
    """Map a SAR-Agent verdict to a JSONL row matching the audited_gold
    superset schema."""

    audit_status, audited_sol_sql, changes = _map_status(
        verdict=verdict,
        original_sol_sql=task["sol_sql"],
    )

    revised_question = verdict.revised_question if verdict.ambiguity_flag else None

    return {
        "instance_id": task["instance_id"],
        "selected_database": task["selected_database"],
        "audit_status": audit_status,
        "original_sol_sql": list(task["sol_sql"]),
        "audited_sol_sql": audited_sol_sql,
        "audited_sample_row": sample_row_result.row,
        "audited_sample_row_status": sample_row_result.status,
        "audited_sample_row_error": sample_row_result.error,
        "changes": changes,
        "reasoning_summary": verdict.reasoning,
        "skill_version": skill_version,
        "audited_at": _now_iso(),
        "sar_correctness_flag": verdict.correctness_flag,
        "sar_ambiguity_flag": verdict.ambiguity_flag,
        "revised_question": revised_question,
        "step_count": step_count,
        "cost_usd": cost_usd,
        "audit_model_requested": audit_model_requested,
        "audit_model_actual": audit_model_actual,
        "raw_trajectory": raw_trajectory,
    }


def _map_status(
    *,
    verdict: SARVerdict,
    original_sol_sql: list[str],
) -> tuple[str, list[str], list[dict]]:
    orig = original_sol_sql[0]
    has_revision = verdict.revised_sql is not None and verdict.revised_sql.strip() != ""

    if verdict.ambiguity_flag:
        if has_revision:
            return (
                "ambiguous",
                [verdict.revised_sql],  # type: ignore[list-item]
                [
                    _change(
                        clause_kind="sar_ambiguous_revision",
                        original=orig,
                        replacement=verdict.revised_sql,  # type: ignore[arg-type]
                        why=verdict.reasoning,
                    )
                ],
            )
        return (
            "ambiguous",
            list(original_sol_sql),
            [_change(clause_kind="sar_ambiguous", original=orig, replacement="", why=verdict.reasoning)],
        )

    if verdict.correctness_flag:
        return "clean", list(original_sol_sql), []

    # correctness_flag = False, ambiguity_flag = False
    if has_revision:
        return (
            "edited",
            [verdict.revised_sql],  # type: ignore[list-item]
            [
                _change(
                    clause_kind="sar_revision",
                    original=orig,
                    replacement=verdict.revised_sql,  # type: ignore[arg-type]
                    why=verdict.reasoning,
                )
            ],
        )
    return (
        "unrecoverable",
        list(original_sol_sql),
        [_change(clause_kind="sar_unrecoverable", original=orig, replacement="", why=verdict.reasoning)],
    )


def _change(*, clause_kind: str, original: str, replacement: str, why: str) -> dict:
    return {
        "clause_kind": clause_kind,
        "source": "sar_agent",
        "original": original,
        "replacement": replacement,
        "why_unjustified": why,
        "justified_by": [],
    }
