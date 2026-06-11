"""DEV-1521 + DEV-1541: LLM-powered autopsy agent for genuine cascade misses.

``run_autopsy`` is called inline from ``claude_sdk_otf*`` ``run_task``
after the agent session ends, when all N1–N9 cascade tiers fail. It
produces a structured ``AutopsyResult`` that ``grade_and_write`` embeds
into the ``SubmissionAnnotation``.

DEV-1541 extends the original DEV-1521 design with three orthogonal
changes:

1. **One-shot vs. a-interact split.** A second LLM-output schema
   (``AutopsyLLMOutputOneShot``) and tool descriptor
   (``_AUTOPSY_TOOL_SCHEMA_ONE_SHOT``) drop the four ``ask_user``-shaped
   fields and the three ``ask_user``-related pattern enum values, which
   the LLM has no way to populate correctly on a one-shot benchmark
   (livesqlbench has no user-sim). ``_build_prompt`` and ``_map_output``
   branch on ``is_one_shot``.

2. **Typed exception capture.** ``run_autopsy`` no longer returns
   ``None`` on failure; every error path returns
   ``AutopsyResult(error=AutopsyError(...))`` so the silent-fail mode
   from the production incident (7 of 67 livesqlbench failures with
   ``autopsy=None`` indistinguishable from "autopsy didn't run") cannot
   recur.

3. **Backfill-friendly metadata.** The persisted ``AutopsyError``
   carries the fully-qualified exception class, a capped message and
   traceback excerpt, and prompt/KB/trajectory size stats so reviewers
   can triage failures without re-running the autopsy.

``_is_genuine_miss`` is the single canonical check for the trigger
condition: since ``grade_submission`` enforces a monotone cascade,
``n9_case_fold=False`` implies every tier failed.

``_read_kb_text`` reads the SLayer memory store's ``memories.yaml``
(at ``{slayer_storage_dir}/memories.yaml``) and extracts the ``learning``
body of all entries whose ``id`` starts with ``{db_name}_kb_``.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import os
import traceback as _tb
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import anthropic
import pydantic
import yaml
from pydantic import BaseModel

from bird_interact_agents.eval.annotation_schema import (
    AutopsyAnalysis,
    AutopsyAnalysisOneShot,
    AutopsyError,
    AutopsyPattern,
    AutopsyPatternOneShot,
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


def _build_anthropic_client() -> "anthropic.AsyncAnthropic":
    """Construct the Anthropic SDK client for the autopsy LLM call.

    DEV-1535 follow-up: on cloud workers running under the OAuth
    subscription path, ``ANTHROPIC_API_KEY`` is deliberately removed from
    the actor env (see ``cloud.ray_app._apply_actor_env_local``) — the
    agent uses the Claude.ai OAuth token instead. Before this helper the
    autopsy stage did ``AsyncAnthropic()`` and the SDK then walked
    ``ANTHROPIC_API_KEY`` → ``ANTHROPIC_AUTH_TOKEN``, found neither, and
    raised ``TypeError("Could not resolve authentication method...")``,
    crashing 18/28 autopsies in the last big mini-interact run. The
    remaining 3 autopsies that ran on the legacy API-key path hit
    ``BadRequestError: "Credit balance is too low"`` — same root cause
    class.

    Resolution order:
      1. ``CLAUDE_CODE_OAUTH_TOKEN`` → Bearer auth (same path as the
         agent, free under the user's subscription).
      2. ``ANTHROPIC_API_KEY`` → x-api-key auth (legacy path; only used
         when subscription auth was opted out at submit).
      3. Neither → ``RuntimeError`` so ``run_autopsy``'s outer
         ``except Exception`` records a meaningful FQN rather than the
         SDK's cryptic ``TypeError``.
    """
    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if oauth:
        return anthropic.AsyncAnthropic(auth_token=oauth, api_key=None)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        return anthropic.AsyncAnthropic(api_key=api_key)
    raise RuntimeError(
        "no Anthropic auth available for autopsy: set "
        "CLAUDE_CODE_OAUTH_TOKEN (subscription) or ANTHROPIC_API_KEY"
    )


# Truncation suffix kept in sync with annotation_schema._cap_excerpt.
_TRUNCATION_SUFFIX = "...[truncated]"


def _truncate(value: str, cap: int) -> str:
    """Hard-cap an excerpt with a marker suffix.

    Mirrors ``annotation_schema._cap_excerpt`` so callers can pre-cap
    explicitly (the AutopsyError validator also caps as a defensive
    backstop)."""
    if not isinstance(value, str) or len(value) <= cap:
        return value
    keep = cap - len(_TRUNCATION_SUFFIX)
    if keep <= 0:
        return _TRUNCATION_SUFFIX[:cap]
    return value[:keep] + _TRUNCATION_SUFFIX


def _strip_thinking_inplace(obj: object) -> None:
    """Recursively strip ThinkingBlock.thinking from a nested dict/list structure.

    Detects ThinkingBlock dicts by the co-presence of "thinking" and "signature"
    keys (the two fields of ``claude_agent_sdk.ThinkingBlock``). Replaces the
    thinking text with a char-count indicator so the autopsy prompt stays within
    the context window without losing any other trajectory information.
    """
    if isinstance(obj, list):
        for item in obj:
            _strip_thinking_inplace(item)
    elif isinstance(obj, dict):
        if (
            "thinking" in obj
            and "signature" in obj
            and isinstance(obj.get("thinking"), str)
        ):
            obj["thinking"] = f"[thinking: {len(obj['thinking'])} chars]"
        for v in obj.values():
            _strip_thinking_inplace(v)


def _compress_trajectory_for_autopsy(trajectory: list[dict]) -> list[dict]:
    """Return a copy of trajectory with ThinkingBlock.thinking content stripped.

    Items whose ``data`` value is a dict (new structured format from
    ``dataclasses.asdict``) are deep-copied and compressed. Items whose
    ``data`` is a plain string (legacy Python repr format) are passed through
    unchanged — the repr is already opaque and can't be selectively compressed.
    """
    result = []
    for item in trajectory:
        data = item.get("data")
        if isinstance(data, dict):
            data = copy.deepcopy(data)
            _strip_thinking_inplace(data)
            result.append({**item, "data": data})
        else:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# LLM output schemas (local to this module — never persisted directly;
# mapped to annotation_schema.AutopsyAnalysis* before storage).
# ---------------------------------------------------------------------------


class _KeyAsk(BaseModel):
    trajectory_idx: int
    summary: str


class AutopsyLLMOutput(BaseModel):
    """Structured output produced by the autopsy LLM call on a-interact runs."""
    pattern: AutopsyPattern
    other_details: Optional[str] = None
    narrative: str
    remediation: str
    decision_point_trajectory_index: Optional[int] = None
    decision_point_description: Optional[str] = None
    n_asks: int = 0
    key_asks: List[_KeyAsk]
    disclosed_resolutions: List[str]
    undisclosed_resolutions: List[str]


class AutopsyLLMOutputOneShot(BaseModel):
    """Structured output produced by the autopsy LLM call on one-shot runs.

    Drops the four ``ask_user``-shaped fields (``n_asks``, ``key_asks``,
    ``disclosed_resolutions``, ``undisclosed_resolutions``) and constrains
    ``pattern`` to the six one-shot-valid values.
    """
    pattern: AutopsyPatternOneShot
    other_details: Optional[str] = None
    narrative: str
    remediation: str
    decision_point_trajectory_index: Optional[int] = None
    decision_point_description: Optional[str] = None


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


def _read_kb_text(
    slayer_storage_dir: str,
    db_name: str,
    external_knowledge: list,
) -> str:
    """Read KB SLayer memories relevant to this task from ``memories.yaml``.

    Only entries whose ID matches ``{db_name}_kb_{n}`` for ``n`` in
    ``external_knowledge`` are included. Returns ``""`` if the file is absent,
    the directory doesn't exist, or ``external_knowledge`` is empty.
    """
    if not external_knowledge:
        return ""
    memories_path = Path(slayer_storage_dir) / "memories.yaml"
    if not memories_path.exists():
        return ""
    try:
        entries = yaml.safe_load(memories_path.read_text()) or []
    except Exception:  # noqa: BLE001
        logger.warning("[autopsy] failed to parse %s", memories_path)
        return ""
    allowed_ids = {f"{db_name}_kb_{n}" for n in external_knowledge if isinstance(n, int)}
    paragraphs = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") in allowed_ids:
            learning = entry.get("learning") or ""
            if learning:
                paragraphs.append(learning)
    return "\n\n".join(paragraphs)


_PATTERN_DEFINITIONS_A_INTERACT = """\
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
- slayer_overaggregation: SLayer wrapped the SELECT in a top-level GROUP BY \
of every projected dimension with NO aggregate functions (or only trivial \
MAX/MIN over the grouped key), silently deduplicating raw rows. Signature: \
submitted_sql has GROUP BY but no real aggregates, the gold has no GROUP BY, \
and predicted row count is at or below gold's. Remediation: agent should \
have used mcp__slayer__query_nested (or disabled SLayer's default \
dimension-dedup) when raw per-record rows were required
- exhausted_budget_guessing: agent used all turns on exploratory attempts \
without converging
- other: doesn't fit the above (describe in other_details)"""


_PATTERN_DEFINITIONS_ONE_SHOT = """\
- late_mutation_corrupted_result: correct intermediate; a LOWER/TRIM/ROUND/ \
CAST/schema change corrupted the final output
- wrong_join_path: wrong or missing join path, or wrong host model for encoding
- output_schema_misread: wrong columns, wrong aggregation shape, or wrong row \
structure
- slayer_generation_artifact: SLayer emitted buggy SQL (integer division, \
broken namespace) unrelated to encoding choices
- slayer_overaggregation: SLayer wrapped the SELECT in a top-level GROUP BY \
of every projected dimension with NO aggregate functions (or only trivial \
MAX/MIN over the grouped key), silently deduplicating raw rows. Signature: \
submitted_sql has GROUP BY but no real aggregates, the gold has no GROUP BY, \
and predicted row count is at or below gold's. Remediation: agent should \
have used mcp__slayer__query_nested (or disabled SLayer's default \
dimension-dedup) when raw per-record rows were required
- exhausted_budget_guessing: agent used all turns on exploratory attempts \
without converging
- other: doesn't fit the above (describe in other_details)"""


def _build_prompt(
    *,
    task_annotation: TaskAnnotation,
    trajectory: list[dict],
    kb_text: str,
    miss_diagnostics: Optional[MissDiagnostics],
    is_one_shot: bool,
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
    compressed = _compress_trajectory_for_autopsy(trajectory)
    traj_json = json.dumps(
        [{"index": i, **item} for i, item in enumerate(compressed)],
        indent=None,
    )

    if is_one_shot:
        # One-shot benchmarks: positively framed, six-pattern menu only.
        # No mention of the dropped patterns — naming them (even in a
        # negative "do NOT use" clause) still encourages the LLM to
        # latch onto the noun (Codex r1 #7).
        context_intro = (
            "This task is a single-turn benchmark submission. The agent "
            "produced one final answer; there is no interactive turn "
            "exchange and no clarification mechanism. Choose the failure "
            "pattern from the seven listed below."
        )
        pattern_definitions = _PATTERN_DEFINITIONS_ONE_SHOT
        ask_fields_instructions = ""
    else:
        context_intro = (
            "This task is an interactive benchmark submission with a "
            "user-sim component. Choose the failure pattern from the "
            "ten listed below."
        )
        pattern_definitions = _PATTERN_DEFINITIONS_A_INTERACT
        ask_fields_instructions = """
## Required ask-related fields (a-interact ONLY)
The `autopsy_output` tool's schema requires `key_asks`,
`disclosed_resolutions`, and `undisclosed_resolutions` on every call.
You MUST include all three — use `[]` when the category is empty rather
than omitting the field (omission causes the call to be rejected by
pydantic validation and the run loses its autopsy).

* `key_asks` (array of `{trajectory_idx, summary}`): the 1-5 most
  load-bearing moments where the agent called `ask_user` (or otherwise
  asked the user-sim a clarifying question). Each item points at the
  ask via its `trajectory_idx` and gives a one-line summary of what was
  asked. If the agent never asked the user-sim, return `[]`.
* `disclosed_resolutions` (array of strings): masked-term resolutions
  that the user-sim ACTUALLY revealed in its responses (e.g. "marginal
  donor = age_diff > 25", "ice time = org_isch_time + exp_time"). One
  short string per resolution. If nothing was disclosed, return `[]`.
* `undisclosed_resolutions` (array of strings): masked-term resolutions
  that the user-sim DECLINED to give or never addressed (often phrased
  by the sim as "out of scope" / "I don't know" / silently ignored).
  If everything asked was answered, return `[]`.

`n_asks` is an integer count of ask_user calls; default 0. Always
sensible to fill (the trajectory has it explicit).
"""

    return f"""\
You are analyzing a failed data-analysis task. The task agent produced a \
submission that failed every evaluation tier (genuine cascade miss). Your job \
is to determine the root cause and suggest a remediation.

{context_intro}

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

## Column-shape guidance (read before scoring miss diagnostics)
Column-header NAMES in the agent's output are irrelevant to correctness — \
the grader compares value tuples positionally. Do not diagnose "column \
naming mismatch" or "namespaced column names" as a root cause. Column-tuple \
COUNT and positional ORDER (column_count_mismatch, column_order_mismatch) \
ARE genuine root causes; focus there, along with row count, row values, \
value types, and formula choices.

## Grader miss diagnostics
{diag_json}

## Agent trajectory
{traj_json}

## Failure pattern definitions
{pattern_definitions}
{ask_fields_instructions}
Call the `autopsy_output` tool with your analysis.
"""


def _map_output(
    output, is_one_shot: bool,
) -> AutopsyResult:
    """Map LLM-output Pydantic instance to a persisted ``AutopsyResult``.

    On one-shot benchmarks the result has no user-sim component
    (``user_sim_interaction=None``) by construction.
    """
    decision_point = None
    if (
        output.decision_point_trajectory_index is not None
        and output.decision_point_description
    ):
        decision_point = TrajectoryDecisionPoint(
            trajectory_item_index=output.decision_point_trajectory_index,
            description=output.decision_point_description,
        )

    if is_one_shot:
        assert isinstance(output, AutopsyLLMOutputOneShot)
        analysis = AutopsyAnalysisOneShot(
            pattern=output.pattern,
            other_details=output.other_details,
            narrative=output.narrative,
            remediation=output.remediation,
        )
        return AutopsyResult(
            analysis=analysis,
            decision_point=decision_point,
            user_sim_interaction=None,
        )

    assert isinstance(output, AutopsyLLMOutput)
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
    analysis = AutopsyAnalysis(
        pattern=output.pattern,
        other_details=output.other_details,
        narrative=output.narrative,
        remediation=output.remediation,
    )
    return AutopsyResult(
        analysis=analysis,
        decision_point=decision_point,
        user_sim_interaction=user_sim,
    )


def _inline_refs(schema: dict) -> dict:
    """Resolve and inline ``$ref``/``$defs`` in a JSON Schema.

    Pydantic's ``model_json_schema()`` emits ``$ref``s for nested models.
    Anthropic's tool ``input_schema`` accepts standard JSON Schema, but
    inlining keeps the schema flat — fewer moving parts, easier to read
    in error logs, and no chance of a future SDK version stumbling on a
    discovery step."""
    defs = schema.pop("$defs", {})

    def resolve(node: object) -> object:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                return resolve(defs[key])
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)  # type: ignore[return-value]


def _pydantic_to_tool_schema(
    model_cls: type[BaseModel],
    *,
    name: str,
    description: str,
) -> dict:
    """Generate an Anthropic tool descriptor from a Pydantic model.

    Hand-mirroring drifts: a new field or pattern-enum value on the
    Pydantic model silently diverges from the hand-written tool schema;
    the LLM returns output the (looser) tool schema accepts but Pydantic
    rejects, and the autopsy lands as ``validation_error`` (the failure
    that motivated this helper). Generating from the model keeps a single
    source of truth."""
    return {
        "name": name,
        "description": description,
        "input_schema": _inline_refs(model_cls.model_json_schema()),
    }


_AUTOPSY_TOOL_SCHEMA = _pydantic_to_tool_schema(
    AutopsyLLMOutput,
    name="autopsy_output",
    description="Report the root-cause analysis of the agent failure.",
)


# DEV-1541: one-shot variant drops the four ``ask_user``-shaped fields
# (``n_asks``, ``key_asks``, ``disclosed_resolutions``,
# ``undisclosed_resolutions``) and restricts ``pattern`` to the six
# one-shot-valid values. Both the field set and the enum live on
# ``AutopsyLLMOutputOneShot`` — the tool schema follows automatically.
_AUTOPSY_TOOL_SCHEMA_ONE_SHOT = _pydantic_to_tool_schema(
    AutopsyLLMOutputOneShot,
    name="autopsy_output",
    description="Report the root-cause analysis of the agent failure.",
)


# ---------------------------------------------------------------------------
# DEV-1541: error-result construction + classification
# ---------------------------------------------------------------------------

def _looks_like_context_overflow(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "too long" in msg
        or "context window" in msg
        or "context_window" in msg
    )


def _fqn(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _autopsy_error_result(
    *,
    kind: str,
    exc: BaseException,
    prompt: str,
    kb_text: str,
    trajectory: list,
    model: str,
) -> AutopsyResult:
    """Build an ``AutopsyResult(error=AutopsyError(...))`` for any failure
    path. Centralises the truncation + FQN + stats capture so every
    exception clause produces the same shape."""
    return AutopsyResult(
        error=AutopsyError(
            kind=kind,  # type: ignore[arg-type]
            exception_class=_fqn(exc),
            message_excerpt=_truncate(str(exc), 500),
            traceback_excerpt=_truncate(_tb.format_exc(), 2000),
            prompt_chars=len(prompt),
            kb_chars=len(kb_text),
            trajectory_items=len(trajectory),
            model=model,
            timestamp=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0),
        )
    )


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
    is_one_shot: bool,
) -> AutopsyResult:
    """Run an LLM autopsy on a genuine cascade miss and return the result.

    DEV-1541: never returns ``None`` on failure. Any error during the
    LLM call or its output parsing results in an
    ``AutopsyResult(error=AutopsyError(...))`` carrying the failure
    metadata (kind, FQN, message + traceback excerpts, prompt/KB/traj
    stats). Returning ``None`` was the silent-failure bug that DEV-1541
    fixes.

    ``is_one_shot`` selects between two LLM-output schemas and tool
    descriptors. The one-shot path drops ``ask_user``-related patterns
    and fields entirely; ``user_sim_interaction`` resolves to ``None``.
    """
    # DEV-1541 r2 (CodeRabbit outside-diff): the KB read + prompt build +
    # schema selection must live INSIDE the error boundary too. If
    # ``_read_kb_text`` or ``_build_prompt`` raises (corrupt
    # ``memories.yaml``, oversized task fields, …), the agent caller's
    # top-level ``except Exception:`` would still swallow the exception
    # and re-introduce the silent ``autopsy=None`` regression DEV-1541
    # exists to kill. Pre-initialise to empty strings so the error path
    # can still record ``prompt_chars`` / ``kb_chars`` even when prep
    # never completed.
    kb_text = ""
    prompt = ""
    tool_schema: dict = {}
    schema_cls: type[BaseModel] = AutopsyLLMOutput
    client: Optional["anthropic.AsyncAnthropic"] = None
    try:
        kb_text = _read_kb_text(
            slayer_storage_dir,
            task_annotation.selected_database,
            task_annotation.external_knowledge,
        )
        prompt = _build_prompt(
            task_annotation=task_annotation,
            trajectory=trajectory,
            kb_text=kb_text,
            miss_diagnostics=miss_diagnostics,
            is_one_shot=is_one_shot,
        )
        tool_schema = (
            _AUTOPSY_TOOL_SCHEMA_ONE_SHOT if is_one_shot else _AUTOPSY_TOOL_SCHEMA
        )
        schema_cls = (
            AutopsyLLMOutputOneShot if is_one_shot else AutopsyLLMOutput
        )
        client = _build_anthropic_client()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[autopsy] prep failed on %s",
            task_annotation.instance_id, exc_info=True,
        )
        return _autopsy_error_result(
            kind="unknown", exc=exc, prompt=prompt, kb_text=kb_text,
            trajectory=trajectory, model=model,
        )

    assert client is not None  # the prep try/except returned otherwise.

    # One LLM call + one corrective retry on Pydantic validation failure.
    # The retry sends the validation error back via a ``tool_result`` block
    # with ``is_error=True`` so the model sees exactly which fields it
    # dropped. Anthropic's ``required`` enforcement on tool input is
    # best-effort; with long prompts the model occasionally omits a
    # leading field (this was the ``archeology_10`` regression — 353k
    # chars, ``pattern`` missing). Other failure kinds (API errors,
    # missing ``tool_use`` block, BadRequest) do NOT retry — they are
    # not model self-correctable.
    messages: list = [{"role": "user", "content": prompt}]
    last_validation_exc: Optional[pydantic.ValidationError] = None
    for attempt in range(2):
        try:
            response = await client.messages.create(
                model=native_model_id(model),
                max_tokens=2048,
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": "autopsy_output"},
                messages=messages,
            )
        except anthropic.BadRequestError as exc:
            kind = "context_overflow" if _looks_like_context_overflow(exc) else "api_error"
            logger.error(
                "[autopsy] BadRequestError on %s (kind=%s): %s",
                task_annotation.instance_id, kind, exc,
                exc_info=True,
            )
            return _autopsy_error_result(
                kind=kind, exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )
        except anthropic.APIConnectionError as exc:
            # Codex r1 #5: must be ordered BEFORE APIError, since
            # APIConnectionError is a sibling (not subclass) of
            # APIStatusError in the anthropic SDK; APITimeoutError is a
            # subclass of APIConnectionError and resolves here too.
            logger.error(
                "[autopsy] APIConnectionError on %s: %s",
                task_annotation.instance_id, exc, exc_info=True,
            )
            return _autopsy_error_result(
                kind="network_error", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )
        except anthropic.APIError as exc:
            logger.error(
                "[autopsy] APIError on %s: %s",
                task_annotation.instance_id, exc, exc_info=True,
            )
            return _autopsy_error_result(
                kind="api_error", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[autopsy] unexpected error on %s",
                task_annotation.instance_id, exc_info=True,
            )
            return _autopsy_error_result(
                kind="unknown", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )

        try:
            tool_use = next(
                b for b in response.content if getattr(b, "type", None) == "tool_use"
            )
        except StopIteration as exc:
            logger.error(
                "[autopsy] no tool_use block in response on %s",
                task_annotation.instance_id, exc_info=True,
            )
            return _autopsy_error_result(
                kind="missing_tool_use", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[autopsy] iterating response.content failed on %s",
                task_annotation.instance_id, exc_info=True,
            )
            return _autopsy_error_result(
                kind="unknown", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )

        try:
            llm_output = schema_cls.model_validate(tool_use.input)
            return _map_output(llm_output, is_one_shot=is_one_shot)
        except pydantic.ValidationError as exc:
            last_validation_exc = exc
            logger.warning(
                "[autopsy] LLM output failed schema validation on %s "
                "(attempt %d/2): %s",
                task_annotation.instance_id, attempt + 1, exc,
            )
            if attempt == 0:
                # Append the model's failed tool_use turn, then a user
                # turn carrying the validation error as a tool_result.
                # ``is_error=True`` signals the model that the previous
                # call was rejected — Anthropic's tool-use docs recommend
                # exactly this shape for corrective retries.
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": (
                            "Your previous autopsy_output failed Pydantic "
                            "schema validation:\n\n"
                            f"{exc}\n\n"
                            "Return a corrected autopsy_output that "
                            "satisfies the schema. Every field listed in "
                            "the tool's `required` array MUST be present."
                        ),
                        "is_error": True,
                    }],
                })
                continue
            logger.error(
                "[autopsy] LLM output failed schema validation on %s "
                "after retry: %s",
                task_annotation.instance_id, exc,
                exc_info=True,
            )
            return _autopsy_error_result(
                kind="validation_error", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[autopsy] mapping LLM output failed on %s",
                task_annotation.instance_id, exc_info=True,
            )
            return _autopsy_error_result(
                kind="unknown", exc=exc, prompt=prompt, kb_text=kb_text,
                trajectory=trajectory, model=model,
            )

    # Unreachable in practice (the loop returns on every branch), but
    # falls through here if the retry-loop bound is ever raised without
    # re-checking the validation-error return path.
    assert last_validation_exc is not None
    return _autopsy_error_result(
        kind="validation_error", exc=last_validation_exc,
        prompt=prompt, kb_text=kb_text,
        trajectory=trajectory, model=model,
    )
