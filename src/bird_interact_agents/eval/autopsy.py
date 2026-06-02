"""DEV-1521: LLM-powered autopsy agent for genuine cascade misses.

``run_autopsy`` is called inline from ``claude_sdk_otf*`` ``run_task``
after the agent session ends, when all N1–N9 cascade tiers fail. It
produces a structured ``AutopsyResult`` that ``grade_and_write`` embeds
into the ``SubmissionAnnotation``.

``_is_genuine_miss`` is the single canonical check for the trigger
condition: since ``grade_submission`` enforces a monotone cascade,
``n9_case_fold=False`` implies every tier failed.

``_read_kb_text`` reads the SLayer memory store's ``memories.yaml``
(at ``{slayer_storage_dir}/memories.yaml``) and extracts the ``learning``
body of all entries whose ``id`` starts with ``{db_name}_kb_``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import yaml

from bird_interact_agents.eval.annotation_schema import (
    AutopsyAnalysis,
    AutopsyResult,
    MissDiagnostics,
    TaskAnnotation,
    TrajectoryDecisionPoint,
    UserSimInteraction,
    UserSimResponseSummary,
)
from bird_interact_agents.model_string import native_model_id

if TYPE_CHECKING:
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM output schema (local to this module — not exposed in annotation_schema)
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class _KeyAsk(BaseModel):
    trajectory_idx: int
    summary: str


class AutopsyLLMOutput(BaseModel):
    """Structured output produced by the autopsy LLM call.

    LLM output type only — never persisted directly; mapped to
    ``AutopsyResult`` before storage.
    """
    pattern: str
    other_details: Optional[str] = None
    narrative: str
    remediation: str
    decision_point_trajectory_index: Optional[int] = None
    decision_point_description: Optional[str] = None
    n_asks: int = 0
    key_asks: List[_KeyAsk]
    disclosed_resolutions: List[str]
    undisclosed_resolutions: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_genuine_miss(cascade: "CascadeVerdict") -> bool:
    """Return True iff ALL cascade tiers failed.

    The cascade returned by ``grade_submission`` is monotone (enforced by
    ``enforce_monotone_cascade``), so ``n9_case_fold=False`` is the single
    sufficient check.
    """
    return not cascade.n9_case_fold


def _read_kb_text(slayer_storage_dir: str, db_name: str) -> str:
    """Read ``{db_name}_kb_*`` SLayer memories from ``memories.yaml``.

    Returns a string of newline-separated ``learning`` paragraphs for all
    matching entries. Returns ``""`` if the file is absent or the directory
    doesn't exist.
    """
    memories_path = Path(slayer_storage_dir) / "memories.yaml"
    if not memories_path.exists():
        return ""
    try:
        entries = yaml.safe_load(memories_path.read_text()) or []
    except Exception:  # noqa: BLE001
        logger.warning("[autopsy] failed to parse %s", memories_path)
        return ""
    prefix = f"{db_name}_kb_"
    paragraphs = []
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id", "")).startswith(prefix):
            learning = entry.get("learning") or ""
            if learning:
                paragraphs.append(learning)
    return "\n\n".join(paragraphs)


def _build_prompt(
    *,
    task_annotation: TaskAnnotation,
    trajectory: list[dict],
    kb_text: str,
    miss_diagnostics: Optional[MissDiagnostics],
) -> str:
    masked_terms = [
        f"  - {mt.term} ({mt.type})"
        for mt in (task_annotation.masked_terms or [])
    ]
    gold_interpretations = [
        f"  - [{gv.variant_id}] {gv.interpretation}"
        for gv in (task_annotation.gold_variants or [])
    ]
    diag_json = (
        miss_diagnostics.model_dump_json(indent=2)
        if miss_diagnostics is not None
        else "null"
    )
    traj_json = json.dumps(
        [{"index": i, **item} for i, item in enumerate(trajectory)],
        indent=None,
    )
    return f"""\
You are analyzing a failed data-analysis task. The task agent produced a \
submission that failed every evaluation tier (genuine cascade miss). Your job \
is to determine the root cause and suggest a remediation.

## Task context
instance_id: {task_annotation.instance_id}
database: {task_annotation.selected_database}
user_query: {task_annotation.amb_user_query}
metadata_sufficiency: {task_annotation.metadata_sufficiency.verdict}
masked_terms:
{chr(10).join(masked_terms) if masked_terms else "  (none)"}
gold_variant_interpretations:
{chr(10).join(gold_interpretations) if gold_interpretations else "  (none)"}

## Knowledge-base items
{kb_text if kb_text else "(no KB items found)"}

## Grader miss diagnostics
{diag_json}

## Agent trajectory
{traj_json}

## Failure pattern definitions
- never_asked_key_question: agent never surfaced a critical clarification \
recoverable via ask_user
- asked_but_ignored_answer: agent asked the right question but disregarded \
the answer
- user_sim_misleading: user-sim gave incorrect/misleading answer
- late_mutation_corrupted_result: correct intermediate; a LOWER/TRIM/ROUND/ \
CAST/schema change corrupted the final output
- wrong_join_path: wrong or missing join path, or wrong host model for encoding
- output_schema_misread: wrong columns, wrong aggregation shape, or wrong row \
structure
- slayer_generation_artifact: SLayer emitted buggy SQL (integer division, \
broken namespace) unrelated to encoding choices
- exhausted_budget_guessing: agent used all turns on exploratory attempts \
without converging
- other: doesn't fit the above (describe in other_details)

Call the `autopsy_output` tool with your analysis.
"""


def _map_output(output: AutopsyLLMOutput) -> AutopsyResult:
    decision_point = None
    if (
        output.decision_point_trajectory_index is not None
        and output.decision_point_description
    ):
        decision_point = TrajectoryDecisionPoint(
            trajectory_item_index=output.decision_point_trajectory_index,
            description=output.decision_point_description,
        )
    user_sim = UserSimInteraction(
        n_asks=output.n_asks,
        key_responses=[
            UserSimResponseSummary(
                trajectory_idx=ka.trajectory_idx,
                summary=ka.summary,
            )
            for ka in output.key_asks
        ],
        disclosed_resolutions=list(output.disclosed_resolutions),
        undisclosed_resolutions=list(output.undisclosed_resolutions),
    )
    return AutopsyResult(
        analysis=AutopsyAnalysis(
            pattern=output.pattern,  # type: ignore[arg-type]
            other_details=output.other_details,
            narrative=output.narrative,
            remediation=output.remediation,
        ),
        decision_point=decision_point,
        user_sim_interaction=user_sim,
    )


# JSON schema for the autopsy_output tool (derived from AutopsyLLMOutput)
_AUTOPSY_TOOL_SCHEMA = {
    "name": "autopsy_output",
    "description": "Report the root-cause analysis of the agent failure.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "enum": [
                    "never_asked_key_question",
                    "asked_but_ignored_answer",
                    "user_sim_misleading",
                    "late_mutation_corrupted_result",
                    "wrong_join_path",
                    "output_schema_misread",
                    "slayer_generation_artifact",
                    "exhausted_budget_guessing",
                    "other",
                ],
            },
            "other_details": {"type": ["string", "null"]},
            "narrative": {"type": "string"},
            "remediation": {"type": "string"},
            "decision_point_trajectory_index": {"type": ["integer", "null"]},
            "decision_point_description": {"type": ["string", "null"]},
            "n_asks": {"type": "integer", "default": 0},
            "key_asks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "trajectory_idx": {"type": "integer"},
                        "summary": {"type": "string"},
                    },
                    "required": ["trajectory_idx", "summary"],
                },
            },
            "disclosed_resolutions": {"type": "array", "items": {"type": "string"}},
            "undisclosed_resolutions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "pattern", "narrative", "remediation",
            "key_asks", "disclosed_resolutions", "undisclosed_resolutions",
        ],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_autopsy(
    *,
    task_annotation: TaskAnnotation,
    trajectory: list[dict],
    slayer_storage_dir: str,
    miss_diagnostics: Optional[MissDiagnostics],
    model: str,
) -> Optional[AutopsyResult]:
    """Run an LLM autopsy on a genuine cascade miss and return the result.

    Returns ``None`` if the prompt exceeds the context window or if any
    error occurs — autopsy failures must never propagate.
    """
    import anthropic

    kb_text = _read_kb_text(slayer_storage_dir, task_annotation.selected_database)
    prompt = _build_prompt(
        task_annotation=task_annotation,
        trajectory=trajectory,
        kb_text=kb_text,
        miss_diagnostics=miss_diagnostics,
    )
    try:
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=native_model_id(model),
            max_tokens=2048,
            tools=[_AUTOPSY_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "autopsy_output"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.BadRequestError as exc:
        msg = str(exc).lower()
        if "too long" in msg or "context window" in msg or "context_window" in msg:
            logger.warning(
                "[autopsy] context overflow on %s: trajectory has %d items; "
                "skipping autopsy",
                task_annotation.instance_id,
                len(trajectory),
            )
        else:
            logger.error(
                "[autopsy] BadRequestError on %s: %s",
                task_annotation.instance_id,
                exc,
                exc_info=True,
            )
        return None
    except Exception:  # noqa: BLE001
        logger.error(
            "[autopsy] unexpected error on %s",
            task_annotation.instance_id,
            exc_info=True,
        )
        return None

    try:
        tool_use = next(
            b for b in response.content if getattr(b, "type", None) == "tool_use"
        )
        llm_output = AutopsyLLMOutput.model_validate(tool_use.input)
        return _map_output(llm_output)
    except Exception:  # noqa: BLE001
        logger.error(
            "[autopsy] failed to parse LLM output on %s",
            task_annotation.instance_id,
            exc_info=True,
        )
        return None
