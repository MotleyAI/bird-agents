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
import math
import os
import re
import traceback as _tb
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import anthropic
import pydantic
import yaml
from pydantic import BaseModel

from bird_interact_agents.agents.claude_sdk.context_budget import (
    context_window_for,
)
from bird_interact_agents.provider_registry import (
    get_provider,
    provider_api_key,
    requires_thinking,
    resolve_base_url,
)

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


# The autopsy is a best-effort, post-hoc explainer that runs inline in the
# actor right after a miss — exactly when a serial run is under the most
# rate-limit pressure. The Anthropic SDK retries 408/409/429/5xx/overloaded
# with exponential backoff + jitter and honors the `Retry-After` header; its
# default of 2 retries proved too few (whole batches of autopsies came back as
# `rate_limit_error` 429 → eval_failed). Raise it so a transient 429 is ridden
# out rather than recorded as an autopsy failure.
_AUTOPSY_MAX_RETRIES = 6


def _build_anthropic_client(model: str = "") -> "anthropic.AsyncAnthropic":
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

    DEV-1555 Stage 2: registry open-weight models (e.g.
    ``moonshot/kimi-k2.7-code``) route to the provider's
    Anthropic-compatible endpoint instead — ambient Anthropic credentials
    are deliberately ignored for those.

    DEV-1604: this covers BOTH anthropic-format providers (Moonshot, z.ai
    coding-plan) AND openai-format providers reached through the bridge
    (Doubleword, z.ai per-token). ``run_autopsy`` runs inline in the actor
    after a miss, so ``resolve_base_url(spec)`` returns the loopback bridge URL
    the actor already set on ``base_url_env`` for a bridged provider (and its
    fail-fast guard raises clearly if the bridge wasn't set). Gating on
    ``api_format == "anthropic"`` used to drop the openai-format providers onto
    the ambient-Anthropic path — which is stripped on a registry run, so a
    Doubleword autopsy lost all autopsy capability.
    """
    spec = get_provider(model)
    if spec is not None:
        # Codex r5: registry Anthropic-compatible endpoints (Moonshot's
        # `/anthropic` base) expect Bearer auth, not the legacy
        # `x-api-key` header. The Anthropic SDK routes `api_key=` to
        # `x-api-key` and `auth_token=` to the Bearer header. The main
        # SDK agent already uses ANTHROPIC_AUTH_TOKEN for the same
        # reason via `sdk_session_env`; autopsy must match or the
        # request 401s. (Bridged providers point at the loopback proxy,
        # which ignores the inbound bearer and upstream-auths itself.)
        # Pass api_key="" (NOT None) so the Anthropic SDK does NOT
        # silently fall back to the ambient ANTHROPIC_API_KEY env var
        # — otherwise a developer running the autopsy locally with
        # ambient Anthropic creds would have both x-api-key and Bearer
        # headers in the request and the provider endpoint could
        # prefer the wrong one.
        return anthropic.AsyncAnthropic(
            base_url=resolve_base_url(spec),
            auth_token=provider_api_key(spec),
            api_key="",
            max_retries=_AUTOPSY_MAX_RETRIES,
        )
    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if oauth:
        return anthropic.AsyncAnthropic(
            auth_token=oauth, api_key=None, max_retries=_AUTOPSY_MAX_RETRIES,
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        return anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=_AUTOPSY_MAX_RETRIES,
        )
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
    """Return a copy of trajectory with ThinkingBlock.thinking content stripped
    and the ``tool_use_result`` SDK echo replaced by a size marker.

    The SDK dump stores every tool result TWICE: the ``content`` block the
    model actually saw plus a (typically larger) ``tool_use_result`` echo.
    The echo carries no information the model acted on, yet roughly doubles
    the serialized trajectory — stripping it is the cheapest autopsy-prompt
    halving available (DEV-1555).

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
            echo = data.get("tool_use_result")
            if echo is not None and not (
                isinstance(echo, str) and echo.startswith("[tool_use_result:")
            ):
                data["tool_use_result"] = (
                    f"[tool_use_result: {len(json.dumps(echo))} chars]"
                )
            result.append({**item, "data": data})
        else:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# DEV-1555: trajectory squeeze so the autopsy prompt fits the model window
# ---------------------------------------------------------------------------

# chars-per-token divisor for dense JSON trajectories (measured ~3-3.5 on
# real sessions; 3.5 with the 0.75 window fraction leaves headroom).
_CHARS_PER_TOKEN = 3.5

# Reserved for the autopsy completion + tool schema on top of the prompt.
_OUTPUT_RESERVE_TOKENS = 4096

# Window fraction usable by the prompt.
_WINDOW_FRACTION = 0.75


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _elide_tool_result_blocks(item: dict) -> int:
    """Replace tool-result block bodies in one trajectory item with size
    markers. Returns the number of serialized chars saved. Assistant text
    and tool inputs are never touched (they lack the ``tool_use_id`` key)."""
    data = item.get("data")
    if not isinstance(data, dict):
        return 0
    content = data.get("content")
    if not isinstance(content, list):
        return 0
    saved = 0
    for block in content:
        if not isinstance(block, dict) or "tool_use_id" not in block:
            continue
        body = block.get("content")
        if body is None or (
            isinstance(body, str) and body.startswith("[tool result elided:")
        ):
            continue
        serialized = json.dumps(body)
        marker = f"[tool result elided: {len(serialized)} chars]"
        block["content"] = marker
        saved += len(serialized) - len(json.dumps(marker))
    return saved


def fit_trajectory_for_autopsy(
    trajectory: list[dict],
    *,
    budget_tokens: int,
    keep_last: int = 20,
    keep_head: int = 5,
) -> list[dict]:
    """Deterministically shrink a (compressed) trajectory under a token budget.

    Phase A elides tool-result block bodies oldest-first, never touching the
    last ``keep_last`` items. If that is not enough, phase B drops whole
    middle items behind a single ``ElidedItems`` marker, preserving the first
    ``keep_head`` and last ``keep_last`` items. Pure function: the input is
    never mutated and equal inputs yield equal outputs.
    """
    items = copy.deepcopy(trajectory)
    chars = len(json.dumps(items))
    if _estimate_tokens(json.dumps(trajectory)) <= budget_tokens:
        return items

    budget_chars = int(budget_tokens * _CHARS_PER_TOKEN)

    # CR r1: shrink the protected tail when the trajectory is shorter than
    # ``keep_head + keep_last``, so both phases still have something to
    # work on under a tight budget. Without this, both phases no-op when
    # ``len(items) <= keep_last`` and ``fit_trajectory_for_autopsy``
    # silently returns the oversized input.
    protected_tail = min(keep_last, max(len(items) - keep_head, 0))

    # Phase A: elide tool-result bodies oldest-first, tail protected.
    for idx in range(max(0, len(items) - protected_tail)):
        if chars <= budget_chars:
            break
        chars -= _elide_tool_result_blocks(items[idx])

    if chars <= budget_chars:
        return items

    # Phase B: drop middle items behind one marker.
    marker = {"type": "ElidedItems", "data": "[elided 0 trajectory items]"}
    dropped = 0
    while True:
        chars = len(json.dumps(items))
        if chars <= budget_chars:
            break
        droppable = [
            i
            for i in range(keep_head, len(items) - protected_tail)
            if items[i] is not marker
        ]
        if not droppable:
            break
        mid = droppable[len(droppable) // 2]
        if dropped == 0:
            items[mid] = marker
        else:
            del items[mid]
        dropped += 1
        marker["data"] = f"[elided {dropped} trajectory items]"
    return items


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

_FENCED_JSON_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)


def _extract_json_candidate(text: str) -> Optional[dict]:
    """Deterministic JSON extraction for the no-tool_use fallback (DEV-1555).

    Preference order: the first fenced ```json block that parses to a
    dict, else the first balanced ``{...}`` region that parses to a dict.
    Returns None when no candidate parses — the caller maps that to
    ``missing_tool_use`` (schema validation happens exactly once,
    downstream)."""
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                except ValueError:
                    start = None
                    continue
                if isinstance(obj, dict):
                    return obj
                start = None
    return None


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
    precompressed: bool = False,
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
    compressed = (
        trajectory if precompressed else _compress_trajectory_for_autopsy(trajectory)
    )
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
                if key not in defs:
                    raise ValueError(
                        f"Unresolved $ref: {ref}; "
                        f"$defs keys = {sorted(defs)}"
                    )
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
        compressed = _compress_trajectory_for_autopsy(trajectory)
        prompt = _build_prompt(
            task_annotation=task_annotation,
            trajectory=compressed,
            kb_text=kb_text,
            miss_diagnostics=miss_diagnostics,
            is_one_shot=is_one_shot,
            precompressed=True,
        )
        # DEV-1555: fit the prompt inside the autopsy model's context
        # window. The squeeze only ever shrinks the trajectory portion;
        # 64 tokens of slack absorb ceil-rounding in the estimates.
        budget = (
            int(context_window_for(model) * _WINDOW_FRACTION)
            - _OUTPUT_RESERVE_TOKENS
        )
        if _estimate_tokens(prompt) > budget:
            overhead = _estimate_tokens(prompt) - _estimate_tokens(
                json.dumps(compressed)
            )
            fitted = fit_trajectory_for_autopsy(
                compressed,
                budget_tokens=max(budget - overhead - 64, 1),
            )
            prompt = _build_prompt(
                task_annotation=task_annotation,
                trajectory=fitted,
                kb_text=kb_text,
                miss_diagnostics=miss_diagnostics,
                is_one_shot=is_one_shot,
                precompressed=True,
            )
            # CR r1: post-squeeze size check. Without this, a pathological
            # trajectory whose head/tail alone exceeds the budget would
            # silently hit the API with an oversized prompt and fail at
            # request time (400 / context_length_exceeded). Surface the
            # failure here with a diagnostic message so the caller can
            # downgrade gracefully.
            if _estimate_tokens(prompt) > budget:
                raise RuntimeError(
                    "autopsy prompt still exceeds the model's context "
                    f"budget after trajectory squeeze "
                    f"(estimated {_estimate_tokens(prompt)} tokens > "
                    f"{budget} budget; keep_head/keep_last contents likely "
                    "exceed budget alone)."
                )
        tool_schema = (
            _AUTOPSY_TOOL_SCHEMA_ONE_SHOT if is_one_shot else _AUTOPSY_TOOL_SCHEMA
        )
        schema_cls = (
            AutopsyLLMOutputOneShot if is_one_shot else AutopsyLLMOutput
        )
        client = _build_anthropic_client(model)
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

    # One LLM call + one corrective retry on Pydantic validation failure
    # (origin/main, DEV-1545 follow-up: archeology_10 regression — the
    # model occasionally drops a leading field on long prompts; the retry
    # ships the validation error back as a ``tool_result`` so the model
    # sees what to fix). DEV-1555 Stage 2 layers on a model-aware client
    # + thinking/auto tool_choice path for Moonshot/Kimi, and a JSON-
    # text fallback when third-party Anthropic-compatible endpoints don't
    # honor forced tool_choice.
    messages: list = [{"role": "user", "content": prompt}]
    last_validation_exc: Optional[pydantic.ValidationError] = None
    for attempt in range(2):
        create_kwargs: dict = {
            "model": native_model_id(model),
            "max_tokens": 2048,
            "tools": [tool_schema],
            "tool_choice": {"type": "tool", "name": "autopsy_output"},
            "messages": messages,
        }
        if requires_thinking(model):
            # Probed live (kimi-k2.7-code): rejects requests without
            # thinking enabled, AND forced tool_choice is incompatible
            # with thinking — switch to auto and lean on the JSON-text
            # fallback when the model answers without the tool.
            create_kwargs["thinking"] = {
                "type": "enabled", "budget_tokens": 1024,
            }
            create_kwargs["tool_choice"] = {"type": "auto"}
            create_kwargs["max_tokens"] = 4096
        try:
            response = await client.messages.create(**create_kwargs)
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

        # Extract the payload — prefer the tool_use block, fall back to
        # JSON inside text blocks (DEV-1555 Stage 2: third-party
        # Anthropic-compatible endpoints may not honor forced
        # tool_choice).
        tool_use = None
        payload: object
        try:
            tool_use = next(
                b for b in response.content if getattr(b, "type", None) == "tool_use"
            )
            payload = tool_use.input
        except StopIteration as exc:
            text = "\n".join(
                t
                for b in response.content
                if getattr(b, "type", None) == "text"
                and isinstance(t := getattr(b, "text", None), str)
            )
            candidate = _extract_json_candidate(text)
            if candidate is None:
                logger.error(
                    "[autopsy] no tool_use block and no parseable JSON in "
                    "response on %s",
                    task_annotation.instance_id, exc_info=True,
                )
                return _autopsy_error_result(
                    kind="missing_tool_use", exc=exc, prompt=prompt,
                    kb_text=kb_text, trajectory=trajectory, model=model,
                )
            payload = candidate
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
            llm_output = schema_cls.model_validate(payload)
            return _map_output(llm_output, is_one_shot=is_one_shot)
        except pydantic.ValidationError as exc:
            last_validation_exc = exc
            logger.warning(
                "[autopsy] LLM output failed schema validation on %s "
                "(attempt %d/2): %s",
                task_annotation.instance_id, attempt + 1, exc,
            )
            if attempt == 0 and tool_use is not None:
                # Corrective retry — only applicable when the model
                # used the tool (we have a `tool_use_id` to bind the
                # validation error to). JSON-text-fallback path has no
                # tool_use_id, so a retry there doesn't have a clean
                # tool_result shape — skip retry and surface the
                # validation error.
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
                "(no further retries): %s",
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
