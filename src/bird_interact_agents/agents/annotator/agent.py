"""Annotator agent — full agentic audit of a single benchmark task (DEV-1518).

Uses the Claude Agent SDK.  The agent has access to all DB-exploration tools
the regular agent has plus two annotation-specific tools:

* ``get_ambiguity_resolutions`` — surfaces critical_ambiguity snippets and
  knowledge_ambiguity definitions from the task data.
* ``submit_annotation`` — validates the TaskAnnotation JSON the agent
  produces and signals completion.

The harness calls ``run_task()`` which drives the SDK loop and returns an
``AnnotatorResult``.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
from typing import Optional

from pydantic import BaseModel, ValidationError

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

from bird_interact_agents.agents._tool_specs import (
    BIRD_INTERACT_TOOLS,
    render_action,
)
from bird_interact_agents.eval.annotation_schema import MaskedTerm, TaskAnnotation
from bird_interact_agents.eval.implicit_annotation import _benchmark_task_jsonl_name
from bird_interact_agents.harness import (
    MAX_MODEL_TURNS,
    SampleStatus,
    execute_env_action,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.model_string import is_anthropic, native_model_id
from bird_interact_agents.agents.annotator.prompts import build_system_prompt

_BY_NAME = {t.name: t for t in BIRD_INTERACT_TOOLS}
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AnnotatorResult
# ---------------------------------------------------------------------------

class AnnotatorResult(BaseModel):
    """Return value of run_task()."""

    instance_id: str
    task_annotation: Optional[TaskAnnotation] = None
    audited_gold_variants: list[dict] = []
    usage: dict = {}
    duration_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-task context (contextvar)
# ---------------------------------------------------------------------------

_ctx_var: contextvars.ContextVar[dict] = contextvars.ContextVar("_annotator_ctx_var")


class _CtxProxy:
    def __getitem__(self, key):
        return _ctx_var.get()[key]

    def __setitem__(self, key, value):
        _ctx_var.get()[key] = value

    def __contains__(self, key):
        try:
            return key in _ctx_var.get()
        except LookupError:
            return False

    def get(self, key, default=None):
        try:
            return _ctx_var.get().get(key, default)
        except LookupError:
            return default

    def update(self, *args, **kwargs):
        try:
            current = _ctx_var.get()
        except LookupError:
            current = {}
            _ctx_var.set(current)
        current.update(*args, **kwargs)


_ctx = _CtxProxy()


def _text(msg: str) -> dict:
    return {"content": [{"type": "text", "text": str(msg)}]}


# ---------------------------------------------------------------------------
# Annotation-specific tool implementations
# (module-level names are raw async callables so tests can call them directly;
#  the @tool-decorated wrappers are stored as _*_tool variables for the SDK)
# ---------------------------------------------------------------------------

async def get_ambiguity_resolutions(args: dict) -> dict:
    task_data = _ctx.get("task_data", {})
    uqa = task_data.get("user_query_ambiguity", {})
    critical = uqa.get("critical_ambiguity", [])
    knowledge = task_data.get("knowledge_ambiguity", [])

    lines: list[str] = []

    if critical:
        lines.append("=== Critical Ambiguity Resolutions ===")
        for item in critical:
            term = item.get("term", "")
            sql = item.get("sql_snippet", "")
            evidence = item.get("metadata_evidence", "")
            lines.append(f"  term: {term}")
            if sql:
                lines.append(f"    sql_snippet: {sql}")
            if evidence:
                lines.append(f"    metadata_evidence: {evidence}")
    else:
        lines.append("No critical ambiguity entries for this task.")

    if knowledge:
        lines.append("")
        lines.append("=== Knowledge Ambiguity Definitions ===")
        for item in knowledge:
            term = item.get("term", "")
            defn = item.get("definition", "")
            lines.append(f"  term: {term}")
            if defn:
                lines.append(f"    definition: {defn}")

    return _text("\n".join(lines) if lines else "No ambiguity data available.")


async def submit_annotation(args: dict) -> dict:
    ta_json = args.get("task_annotation_json", "")
    av_json = args.get("audited_gold_variants_json", "[]")

    try:
        ta_dict = json.loads(ta_json)
    except json.JSONDecodeError as e:
        return _text(f"Error: invalid JSON in task_annotation_json: {e}")

    try:
        task_annotation = TaskAnnotation.model_validate(ta_dict)
    except ValidationError as e:
        return _text(f"Validation error in task_annotation_json: {e}")
    except Exception as e:
        return _text(f"Error parsing task_annotation_json: {e}")

    task_data = _ctx.get("task_data") or {}
    expected_iid = task_data.get("instance_id")
    expected_db = task_data.get("selected_database")
    expected_benchmark = _ctx.get("benchmark")
    if expected_iid and task_annotation.instance_id != expected_iid:
        return _text(
            f"Validation error: task_annotation instance_id must be "
            f"{expected_iid!r}, got {task_annotation.instance_id!r}"
        )
    if expected_db and task_annotation.selected_database != expected_db:
        return _text(
            f"Validation error: task_annotation selected_database must be "
            f"{expected_db!r}, got {task_annotation.selected_database!r}"
        )

    try:
        audited_gold_variants: list[dict] = json.loads(av_json)
        if not isinstance(audited_gold_variants, list):
            raise TypeError("Expected a JSON array")
    except (json.JSONDecodeError, TypeError) as e:
        return _text(f"Error: invalid JSON in audited_gold_variants_json: {e}")

    _REQUIRED_VARIANT_FIELDS = {
        "instance_id", "selected_database", "benchmark",
        "audit_status", "audited_sol_sql", "variant_id",
    }
    for i, variant in enumerate(audited_gold_variants):
        if not isinstance(variant, dict):
            return _text(
                f"Error: audited_gold_variants[{i}] is not a dict (got {type(variant).__name__})"
            )
        missing = _REQUIRED_VARIANT_FIELDS - variant.keys()
        if missing:
            return _text(
                f"Error: audited_gold_variants[{i}] is missing required fields: "
                f"{sorted(missing)}. Add them and retry."
            )
        if expected_iid and variant.get("instance_id") != expected_iid:
            return _text(
                f"Error: audited_gold_variants[{i}].instance_id must be "
                f"{expected_iid!r}, got {variant.get('instance_id')!r}"
            )
        if expected_db and variant.get("selected_database") != expected_db:
            return _text(
                f"Error: audited_gold_variants[{i}].selected_database must be "
                f"{expected_db!r}, got {variant.get('selected_database')!r}"
            )
        if expected_benchmark and variant.get("benchmark") != expected_benchmark:
            return _text(
                f"Error: audited_gold_variants[{i}].benchmark must be "
                f"{expected_benchmark!r}, got {variant.get('benchmark')!r}"
            )
        if not isinstance(variant.get("audited_sol_sql"), list):
            return _text(
                f"Error: audited_gold_variants[{i}].audited_sol_sql must be a list "
                f"of SQL strings, got {type(variant.get('audited_sol_sql')).__name__!r}. "
                f"Wrap single SQL in a list: [\"SELECT ...\"]"
            )
        _VALID_AUDIT_STATUSES = {"clean", "edited", "unrecoverable", "original_correct"}
        _status = variant.get("audit_status")
        if _status != "unrecoverable" and not variant.get("audited_sol_sql"):
            return _text(
                f"Error: audited_gold_variants[{i}].audited_sol_sql must contain at least "
                f"one SQL string when audit_status={_status!r}. "
                f"Only audit_status='unrecoverable' permits an empty list."
            )
        if variant.get("audit_status") not in _VALID_AUDIT_STATUSES:
            return _text(
                f"Error: audited_gold_variants[{i}].audit_status must be one of "
                f"{sorted(_VALID_AUDIT_STATUSES)}, got {variant.get('audit_status')!r}"
            )
        _primary_val = variant.get("primary")
        if _primary_val is not None and not isinstance(_primary_val, bool):
            return _text(
                f"Error: audited_gold_variants[{i}].primary must be a boolean "
                f"(true or false), got {type(_primary_val).__name__!r}. "
                f"Use JSON true/false, not 1/0 or strings."
            )

    if task_annotation.original_gold_is_correct and audited_gold_variants:
        return _text(
            "Validation error: original_gold_is_correct=True requires an empty "
            "audited_gold_variants array. Set audited_gold_variants_json to '[]'."
        )

    if not task_annotation.original_gold_is_correct and task_annotation.gold_variants:
        submitted_variant_ids = {v.get("variant_id") for v in audited_gold_variants}
        for gvr in task_annotation.gold_variants:
            ref_vid = gvr.audited_gold_ref.variant_id
            if ref_vid not in submitted_variant_ids:
                return _text(
                    f"Validation error: gold_variants entry {gvr.variant_id!r} references "
                    f"audited_gold_ref.variant_id={ref_vid!r} but no matching entry "
                    f"found in audited_gold_variants_json. Add the audited row or "
                    f"set original_gold_is_correct=True."
                )
        variant_primary_map = {v.get("variant_id"): v.get("primary", False)
                               for v in audited_gold_variants}
        for gvr in task_annotation.gold_variants:
            if gvr.primary and not variant_primary_map.get(gvr.audited_gold_ref.variant_id, False):
                return _text(
                    f"Validation error: gold_variants entry {gvr.variant_id!r} is marked "
                    f"primary=True but the matching audited_gold_variants row has primary=False. "
                    f"Set primary=True in the audited variant row."
                )

    # Reverse cross-check: if audited variants submitted, gold_variants must reference each one.
    if not task_annotation.original_gold_is_correct and audited_gold_variants:
        if not task_annotation.gold_variants:
            return _text(
                "Validation error: audited_gold_variants is non-empty but gold_variants is "
                "empty. Add a GoldVariantRef in gold_variants for each submitted audited variant."
            )
        gold_ref_ids = {gvr.audited_gold_ref.variant_id for gvr in task_annotation.gold_variants}
        for v in audited_gold_variants:
            vid = v.get("variant_id")
            if vid not in gold_ref_ids:
                return _text(
                    f"Validation error: audited variant {vid!r} has no matching GoldVariantRef "
                    f"in gold_variants. Add a gold_variants entry with "
                    f"audited_gold_ref.variant_id={vid!r}."
                )

    primary_count = sum(1 for v in audited_gold_variants if v.get("primary", False))
    if primary_count > 1:
        return _text(
            f"Validation error: at most one audited_gold_variants entry may have "
            f"primary=True; got {primary_count}. Mark exactly the primary variant."
        )
    if audited_gold_variants and primary_count == 0:
        return _text(
            "Validation error: at least one audited_gold_variants entry must have primary=True."
        )

    _ctx["annotation_result"] = {
        "task_annotation": task_annotation,
        "audited_gold_variants": audited_gold_variants,
    }
    _ctx["_submission_done"] = True
    return _text("Annotation submitted successfully.")


# Tool-decorated wrappers for the SDK.
_get_ambiguity_resolutions_tool = tool(
    "get_ambiguity_resolutions",
    (
        "Return the critical ambiguity resolutions (masked terms, "
        "SQL snippets, KB evidence) and knowledge_ambiguity definitions "
        "pre-computed for this task."
    ),
    {},
)(get_ambiguity_resolutions)

_submit_annotation_tool = tool(
    "submit_annotation",
    (
        "Submit the completed TaskAnnotation. "
        "``task_annotation_json``: the full TaskAnnotation as a JSON string. "
        "``audited_gold_variants_json``: a JSON array of audited-gold-variant "
        "dicts (may be [] for clean-gold tasks). "
        "Returns an error message if validation fails so you can retry."
    ),
    {"task_annotation_json": str, "audited_gold_variants_json": str},
)(submit_annotation)


# ---------------------------------------------------------------------------
# DB-exploration tool implementations
# ---------------------------------------------------------------------------

def _run_env_sync(action_str: str) -> dict:
    data_path_base = _ctx.get("data_path_base", "")
    task_data = _ctx.get("task_data", {})
    status = SampleStatus(
        idx=0, original_data=task_data, remaining_budget=999, total_budget=999
    )
    observation, _ = execute_env_action(action_str, status, data_path_base)
    return _text(str(observation))


async def execute_sql(args: dict) -> dict:
    return _run_env_sync(render_action(_BY_NAME["execute_sql"], sql=args["sql"]))


async def get_schema(args: dict) -> dict:
    return _run_env_sync(render_action(_BY_NAME["get_schema"]))


async def get_all_column_meanings(args: dict) -> dict:
    return _run_env_sync(render_action(_BY_NAME["get_all_column_meanings"]))


async def get_column_meaning(args: dict) -> dict:
    return _run_env_sync(
        render_action(
            _BY_NAME["get_column_meaning"],
            table_name=args["table_name"],
            column_name=args["column_name"],
        )
    )


async def get_all_external_knowledge_names(args: dict) -> dict:
    return _run_env_sync(render_action(_BY_NAME["get_all_external_knowledge_names"]))


async def get_knowledge_definition(args: dict) -> dict:
    return _run_env_sync(
        render_action(
            _BY_NAME["get_knowledge_definition"],
            knowledge_name=args["knowledge_name"],
        )
    )


async def get_all_knowledge_definitions(args: dict) -> dict:
    return _run_env_sync(render_action(_BY_NAME["get_all_knowledge_definitions"]))


_execute_sql_tool = tool(
    "execute_sql", _BY_NAME["execute_sql"].description, {"sql": str}
)(execute_sql)

_get_schema_tool = tool("get_schema", _BY_NAME["get_schema"].description, {})(get_schema)

_get_all_column_meanings_tool = tool(
    "get_all_column_meanings", _BY_NAME["get_all_column_meanings"].description, {}
)(get_all_column_meanings)

_get_column_meaning_tool = tool(
    "get_column_meaning",
    _BY_NAME["get_column_meaning"].description,
    {"table_name": str, "column_name": str},
)(get_column_meaning)

_get_all_external_knowledge_names_tool = tool(
    "get_all_external_knowledge_names",
    _BY_NAME["get_all_external_knowledge_names"].description,
    {},
)(get_all_external_knowledge_names)

_get_knowledge_definition_tool = tool(
    "get_knowledge_definition",
    _BY_NAME["get_knowledge_definition"].description,
    {"knowledge_name": str},
)(get_knowledge_definition)

_get_all_knowledge_definitions_tool = tool(
    "get_all_knowledge_definitions",
    _BY_NAME["get_all_knowledge_definitions"].description,
    {},
)(get_all_knowledge_definitions)


_EXPLORATION_SDK_TOOLS = [
    _execute_sql_tool,
    _get_schema_tool,
    _get_all_column_meanings_tool,
    _get_column_meaning_tool,
    _get_all_external_knowledge_names_tool,
    _get_knowledge_definition_tool,
    _get_all_knowledge_definitions_tool,
]

_MINI_INTERACT_BENCHMARK = "mini_interact"


def _select_annotation_sdk_tools(benchmark: str) -> list:
    if benchmark == _MINI_INTERACT_BENCHMARK:
        return [_get_ambiguity_resolutions_tool, _submit_annotation_tool]
    return [_submit_annotation_tool]


# ---------------------------------------------------------------------------
# Harness helper: fill AuditedGoldRef.file
# ---------------------------------------------------------------------------

_AUDITED_GOLD_FILE: dict[str, str] = {
    "mini_interact": "audited_gold/mini_interact_audited.jsonl",
    "livesqlbench": "audited_gold/livesqlbench_audited.jsonl",
}


def _fill_audited_gold_ref_files(ann: TaskAnnotation, *, benchmark: str) -> TaskAnnotation:
    """Replace __HARNESS_FILLS__ sentinel in AuditedGoldRef.file."""
    if not ann.gold_variants:
        return ann
    canonical_file = _AUDITED_GOLD_FILE.get(
        benchmark, f"audited_gold/{benchmark}_audited.jsonl"
    )
    updated = ann.model_copy(deep=True)
    for gvr in updated.gold_variants:
        if gvr.audited_gold_ref and gvr.audited_gold_ref.file == "__HARNESS_FILLS__":
            gvr.audited_gold_ref.file = canonical_file
    return updated


def _fill_deterministic_fields(
    ann: TaskAnnotation,
    *,
    task_data: dict,
    benchmark: str,
) -> TaskAnnotation:
    """Overwrite fields that are deterministically derivable from task data.

    These fields should never depend on the agent guessing: provenance paths,
    external KB IDs, and (for mini_interact) the is_mask=True masked terms that
    are pre-computed in critical_ambiguity.  Agent-supplied is_mask=False entries
    (schema-linking ambiguities) are preserved.
    """
    updated = ann.model_copy(deep=True)

    # Provenance — always authoritative from benchmark registry and instance_id.
    updated.provenance.task_jsonl_path = _benchmark_task_jsonl_name(benchmark)
    instance_id = task_data.get("instance_id")
    if instance_id:
        updated.provenance.task_jsonl_instance_id = instance_id

    # External knowledge — verbatim from task data.
    ext_kb = task_data.get("external_knowledge")
    if ext_kb is not None:
        updated.external_knowledge = list(ext_kb)

    # Masked terms: merge critical_ambiguity (is_mask=True) without duplicating.
    if benchmark == _MINI_INTERACT_BENCHMARK:
        uqa = task_data.get("user_query_ambiguity", {})
        critical = uqa.get("critical_ambiguity", []) if isinstance(uqa, dict) else []
        if critical:
            existing_terms = {mt.term for mt in updated.masked_terms}
            for item in critical:
                term = item.get("term", "")
                if not term or term in existing_terms:
                    continue
                raw_evidence = item.get("metadata_evidence")
                if isinstance(raw_evidence, list):
                    evidence: list = raw_evidence
                elif isinstance(raw_evidence, str) and raw_evidence:
                    evidence = [raw_evidence]
                else:
                    evidence = []
                updated.masked_terms = list(updated.masked_terms) + [
                    MaskedTerm(
                        term=term,
                        type=item.get("type", "knowledge_linking_ambiguity"),
                        is_mask=True,
                        metadata_evidence=evidence,
                    )
                ]

    return updated


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------

async def run_task(
    task_data: dict,
    data_path_base: str,
    benchmark: str,
    model: str = "anthropic/claude-opus-4-7",
    effort: str = "medium",
    max_turns: int | None = None,
) -> AnnotatorResult:
    """Drive a single annotator task using the Claude Agent SDK."""
    instance_id = task_data["instance_id"]
    db_name = task_data["selected_database"]
    t0 = time.monotonic()

    if not is_anthropic(model):
        msg = (
            f"annotator agent requires an Anthropic model; got {model!r}. "
            "Use an anthropic/* model string."
        )
        logger.warning("[%s] %s", instance_id, msg)
        return AnnotatorResult(
            instance_id=instance_id,
            error=msg,
            duration_s=time.monotonic() - t0,
        )

    load_db_data_if_needed(db_name, data_path_base)
    if benchmark != _MINI_INTERACT_BENCHMARK:
        materialize_task_db(task_data, data_path_base)

    ctx_dict: dict = {
        "task_data": task_data,
        "data_path_base": data_path_base,
        "benchmark": benchmark,
        "annotation_result": None,
        "_submission_done": False,
    }
    _ctx_var.set(ctx_dict)

    annotation_sdk_tools = _select_annotation_sdk_tools(benchmark)
    all_sdk_tools = _EXPLORATION_SDK_TOOLS + annotation_sdk_tools
    tool_names = [t.name for t in all_sdk_tools]
    tool_names_prefixed = [f"mcp__bird-annotator-tools__{n}" for n in tool_names]

    server = create_sdk_mcp_server(
        name="bird-annotator-tools", version="1.0.0", tools=all_sdk_tools
    )

    system_prompt = build_system_prompt(task_data=task_data, benchmark=benchmark)
    cap = max_turns or MAX_MODEL_TURNS

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"bird-annotator-tools": server},
        allowed_tools=tool_names_prefixed,
        tools=[],
        setting_sources=[],
        effort=effort,
        max_turns=cap,
        model=native_model_id(model),
    )

    turns = 0
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(task_data["amb_user_query"])
            async for msg in client.receive_response():
                if ctx_dict.get("_submission_done"):
                    break
                if type(msg).__name__ == "AssistantMessage":
                    turns += 1
                    if turns >= cap:
                        logger.warning(
                            "Max turns (%d) reached for %s; stopping.", cap, instance_id
                        )
                        break
    except Exception as e:
        logger.error("Annotator error on %s: %s", instance_id, e)
        return AnnotatorResult(
            instance_id=instance_id,
            error=str(e),
            duration_s=time.monotonic() - t0,
        )

    if not ctx_dict.get("_submission_done"):
        return AnnotatorResult(
            instance_id=instance_id,
            error=f"Agent did not submit an annotation after {turns} turns.",
            duration_s=time.monotonic() - t0,
        )

    ann_result = ctx_dict.get("annotation_result")
    if ann_result is None:
        return AnnotatorResult(
            instance_id=instance_id,
            error="Submission flag set but no annotation_result stored.",
            duration_s=time.monotonic() - t0,
        )

    task_annotation: TaskAnnotation = ann_result["task_annotation"]
    task_annotation = _fill_audited_gold_ref_files(task_annotation, benchmark=benchmark)
    task_annotation = _fill_deterministic_fields(
        task_annotation, task_data=task_data, benchmark=benchmark
    )
    audited_gold_variants: list[dict] = ann_result["audited_gold_variants"]

    return AnnotatorResult(
        instance_id=instance_id,
        task_annotation=task_annotation,
        audited_gold_variants=audited_gold_variants,
        duration_s=time.monotonic() - t0,
    )
