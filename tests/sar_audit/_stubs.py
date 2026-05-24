"""Shared stub types for SAR-audit tests — importable from any test module."""

from __future__ import annotations

from pydantic import BaseModel


class StubSARVerdict(BaseModel):
    correctness_flag: bool
    ambiguity_flag: bool
    revised_sql: str | None = None
    revised_question: str | None = None
    reasoning: str = ""


class StubSARRunResult(BaseModel):
    verdict: StubSARVerdict
    step_count: int = 1
    cost_usd: float | None = 0.0
    audit_model_actual: str | None = None
    raw_trajectory: list | None = None
