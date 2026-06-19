"""Analyze SLayer query-syntax rejections in mini-interact slayer trajectories.

In slayer mode the agent answers each task by composing a **SlayerQuery DSL**
payload (JSON: ``source_model`` / ``measures`` / ``dimensions`` / ``filters`` /
``order`` / …) and running it through query tools. When that payload is
malformed the tool *rejects* it and the agent must re-author it. This module
quantifies that friction:

* which DSL mistakes the agent makes (the failure-mode taxonomy),
* how many retries it burns before getting a valid query (retry-to-recovery),
* whether that friction is ever a **major contributor to a task failing**.

It walks the LATEST slayer ``*.trajectory.json`` per ``(db, instance_id)`` under
``runs/<benchmark>/``, pairs each query tool-call with its result (by
``tool_use_id``), and classifies every query result into exactly one bucket:

* ``dsl``      — SlayerQuery DSL validation error (the headline bucket).
* ``semantic`` — valid DSL but a bad *reference* (``no such column/table``,
  generated-SQL syntax error). Tallied for context, excluded from headline.
* ``gate``     — ``submit_query`` refused because the agent didn't preview the
  exact query through ``query`` first. Workflow violation, not syntax.
* ``other``    — an ``is_error`` result that matched no known pattern.
* ``ok``       — the query executed.

The five query-syntax tools (slayer MCP + bird-interact-tools wrapper):
``mcp__slayer__query``, ``mcp__slayer__query_nested``,
``mcp__bird-interact-tools__query``, ``mcp__bird-interact-tools__query_nested``,
``mcp__bird-interact-tools__submit_query``.

This file is BOTH an importable library (used by the companion notebook
``notebooks/query_syntax_rejections.ipynb``) and a CLI:

    uv run python scripts/analyze_query_syntax_rejections.py \\
        --benchmark mini-interact --mode slayer            # human text summary
    uv run python scripts/analyze_query_syntax_rejections.py --json  # payload

The notebook is the rich human-readable output (tables + reproducible plots);
the CLI text render is a quick terminal sanity-check; ``--json`` feeds both the
notebook and the tests.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bird_interact_agents import paths

# ---------------------------------------------------------------------------
# The five tools that take a SlayerQuery DSL payload (both the slayer MCP and
# the bird-interact-tools wrapper expose them).
# ---------------------------------------------------------------------------
QUERY_TOOLS = frozenset(
    {
        "mcp__slayer__query",
        "mcp__slayer__query_nested",
        "mcp__bird-interact-tools__query",
        "mcp__bird-interact-tools__query_nested",
        "mcp__bird-interact-tools__submit_query",
    }
)

#: how many distinct example error texts to retain per DSL sub_tag.
_MAX_EXAMPLES_PER_SUBTAG = 3

_FILE_MODE_RE = re.compile(r"-(raw|slayer)-")
_TS_RE = re.compile(r"(\d{8}t\d{4})")
_BUDGET_NOTE_RE = re.compile(r"\[Remaining budget:\s*([0-9.]+)\s*/\s*([0-9.]+)\]")


# ---------------------------------------------------------------------------
# Rejection taxonomy. ONE ordered list is the single source of truth shared by
# the report and the tests. First match wins, so the order matters:
#   1. ``gate`` (very specific submit-gate wording),
#   2. ``dsl`` sub-tags (specific DSL-validation wording),
#   3. ``semantic`` (broad "no such …" / generated-SQL "syntax error").
# Matching ``semantic`` last keeps a generic "syntax error" from stealing a
# more specific DSL or gate match.
# ---------------------------------------------------------------------------
# (sub_tag, bucket, regex) — regexes are matched case-insensitively.
ERROR_TAXONOMY: list[tuple[str, str, re.Pattern[str]]] = [
    # --- infra / not-the-agent's-syntax (excluded from headline) ---
    # Permission-mode denials ("Claude requested permissions to use X, but you
    # haven't…") are a harness/approval artefact, NOT a query the agent got
    # wrong. Schema-drift is a stale-cache state error after editing models.
    (
        "permission_denied",
        "infra",
        re.compile(r"requested permissions to use", re.IGNORECASE),
    ),
    (
        "schema_drift",
        "infra",
        re.compile(r"schema drift detected", re.IGNORECASE),
    ),
    # --- submit-gate (excluded from headline) ---
    (
        "submit_without_preview",
        "gate",
        re.compile(
            r"before submitting you must run the exact query"
            r"|your last tool call was .* not [`']?query",
            re.IGNORECASE,
        ),
    ),
    # --- DSL validation (headline bucket) ---
    (
        "order_shape",
        "dsl",
        re.compile(
            r"order\.\d+\.column|order\.\d+\.direction"
            r"|validation error for slayerquery",
            re.IGNORECASE,
        ),
    ),
    (
        "aggregation_rule",
        "dsl",
        re.compile(
            r"aggregation '[^']*' not allowed"
            r"|aggregation '[^']*' is not (?:allowed|applicable|a built-in)"
            r"|aggregation '[^']*' on measure '[^']*' requires"
            r"|use '\*:count' for count\(\*\)"
            r"|not allowed with measure"
            r"|not applicable to .* column",
            re.IGNORECASE,
        ),
    ),
    (
        "unknown_transform",
        "dsl",
        re.compile(r"unknown transform function", re.IGNORECASE),
    ),
    (
        "formula_syntax",
        "dsl",
        re.compile(
            r"unsupported formula syntax"
            r"|invalid formula syntax"
            r"|expecting \)"  # formula-expression parse error
            r"|required keyword: '[^']*' missing",  # sqlglot parse failure
            re.IGNORECASE,
        ),
    ),
    (
        "bare_measure",
        "dsl",
        re.compile(
            r"bare measure name '[^']*' is not valid|use colon syntax",
            re.IGNORECASE,
        ),
    ),
    (
        "transform_usage",
        "dsl",
        re.compile(
            r"\((?:lag|lead|rank|cumsum|first|last)\) requires"
            r"|transform '[^']*': partition_by",
            re.IGNORECASE,
        ),
    ),
    (
        # SLayer-layer filter-construction rejections: referencing a derived
        # column / a non-reachable model / an undefined name inside a filter,
        # an unknown filter function, a subquery, or otherwise invalid syntax.
        "filter_construction",
        "dsl",
        re.compile(
            r"filter '[^']*' references (?:column|model)"
            r"|filter references unknown name"
            r"|unknown filter function"
            r"|subqueries are not allowed in dsl filters"
            r"|invalid filter syntax",
            re.IGNORECASE,
        ),
    ),
    (
        "model_not_found",
        "dsl",
        re.compile(
            r"model '[^']*' not found|invalid model name", re.IGNORECASE
        ),
    ),
    (
        "cross_model_measure",
        "dsl",
        re.compile(r"cross-model measure", re.IGNORECASE),
    ),
    (
        "measure_collision",
        "dsl",
        re.compile(
            r"canonicalises to the same aggregation"
            r"|that shadows the ca"
            r"|shadows the canonical"
            r"|measure name '[^']*' collid"
            r"|collides with",
            re.IGNORECASE,
        ),
    ),
    (
        "window_in_filter",
        "dsl",
        re.compile(
            r"window function.*not allowed in filters"
            r"|use a rank-family transform"
            r"|whose sql contains a window function",
            re.IGNORECASE,
        ),
    ),
    (
        "top_level_shape",
        "dsl",
        re.compile(
            r"invalid query json: expected either"
            r"|contains unsupported top-level field"
            r"|missing the required [`']?source_model"
            r"|use the [`']?query_nested[`']? tool"
            r"|validation errors? for queryarguments",
            re.IGNORECASE,
        ),
    ),
    (
        "json_parse",
        "dsl",
        re.compile(
            r"invalid json .*submission aborted"
            r"|invalid json"
            r"|could not be parsed as json",
            re.IGNORECASE,
        ),
    ),
    # SLayer-layer reference rejections (no declared join path / SLayer can't
    # resolve a referenced name / SLayer can't compile to SQL). These are the
    # query compiler refusing the spec — DSL, not a DB-execution failure.
    (
        "no_join_path",
        "dsl",
        re.compile(r"has no join to", re.IGNORECASE),
    ),
    (
        "column_not_found",
        "dsl",
        re.compile(r"column '[^']*' not found|not found in model", re.IGNORECASE),
    ),
    (
        "could_not_generate_sql",
        "dsl",
        re.compile(r"could not generate sql", re.IGNORECASE),
    ),
    # generic Pydantic / SLayer validation fallthrough (still a DSL-shape error)
    (
        "validation_other",
        "dsl",
        re.compile(r"\bvalidation error\b|field required", re.IGNORECASE),
    ),
    # --- semantic / DB-EXECUTION (valid spec, the database rejected it at
    #     runtime — excluded from the headline per the analysis scope) ---
    (
        "no_such_column",
        "semantic",
        re.compile(r"no such column", re.IGNORECASE),
    ),
    (
        "no_such_table",
        "semantic",
        re.compile(r"no such table", re.IGNORECASE),
    ),
    (
        "generated_sql_syntax",
        "semantic",
        re.compile(r"syntax error", re.IGNORECASE),
    ),
    (
        "database_error_other",
        "semantic",
        re.compile(r"database error", re.IGNORECASE),
    ),
]

#: DSL sub-tags in a stable display order (for tables / plots).
DSL_SUBTAGS = [tag for tag, bucket, _ in ERROR_TAXONOMY if bucket == "dsl"]

#: One-line plain-English description of each DSL sub_tag — what the agent got
#: wrong. Shared by the CLI and the notebook so the taxonomy is documented in
#: exactly one place.
SUBTAG_DESCRIPTIONS: dict[str, str] = {
    "order_shape": (
        "`order` written as `{\"col\": \"desc\"}` instead of the required "
        "`{\"column\": \"col\", \"direction\": \"desc\"}` object."
    ),
    "filter_construction": (
        "A `filters` entry SLayer can't build: references a derived/aggregated "
        "column or an undefined name, uses an unknown filter function, or "
        "embeds a subquery (not allowed in DSL filters)."
    ),
    "aggregation_rule": (
        "An aggregation that isn't allowed for the column/measure (e.g. "
        "`countd` on a primary key, `count_distinct` on `*`, or a non-built-in "
        "aggregation)."
    ),
    "unknown_transform": (
        "Referenced a transform function that doesn't exist (e.g. `ROUND` as a "
        "transform instead of `change`/`cumsum`/`rank`/…)."
    ),
    "measure_collision": (
        "Two measures in the query canonicalise to the same aggregation, or a "
        "measure name shadows/collides with a source column."
    ),
    "column_not_found": (
        "Referenced a column/field that the named model doesn't expose "
        "(a SLayer-layer name resolution failure, not a DB error)."
    ),
    "top_level_shape": (
        "The top-level query JSON is the wrong shape: missing `source_model`, "
        "an unsupported top-level field, an array passed to a single-stage "
        "tool, or a `queryArguments` schema-validation error."
    ),
    "formula_syntax": (
        "A `measures` formula that won't parse — invalid/unsupported formula "
        "syntax or a raw-SQL expression SLayer can't compile."
    ),
    "no_join_path": (
        "The query crosses two models with no declared join between them "
        "(`Model X has no join to Y`)."
    ),
    "transform_usage": (
        "A transform used incorrectly — e.g. a `partition_by`/ordering column "
        "missing for `rank`/`lag`/`dense_rank`."
    ),
    "bare_measure": (
        "A bare measure name with no aggregation — needs colon syntax "
        "(e.g. `revenue:sum`, `*:count`)."
    ),
    "model_not_found": (
        "`source_model` (or a referenced stage) names a model that doesn't "
        "exist — often a typo'd nested-stage name."
    ),
    "window_in_filter": (
        "A window function used inside a `filters` entry — use a rank-family "
        "transform (`rank(...) <= N`, `percent_rank(...)`) instead."
    ),
    "json_parse": "The `query_json` argument was not valid JSON.",
    "could_not_generate_sql": (
        "SLayer accepted the shape but could not compile the spec to SQL "
        "(a catch-all translation failure)."
    ),
    "cross_model_measure": (
        "A cross-model measure referenced in a context where SLayer can't "
        "resolve it."
    ),
    "validation_other": (
        "A SLayer/Pydantic validation error not captured by a more specific "
        "sub_tag (safety-net bucket)."
    ),
}


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------
def mode_from_filename(name: str) -> Optional[str]:
    """Return ``"raw"`` / ``"slayer"`` / ``None`` from the cloud-run filename's
    mode token (``<ts>-<framework>-<mode>-<hash>.trajectory.json``)."""
    m = _FILE_MODE_RE.search(name)
    return m.group(1) if m else None


def _timestamp(name: str) -> Optional[datetime]:
    m = _TS_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dt%H%M")
    except ValueError:
        return None


def latest_trajectory_per_iid(
    *,
    benchmark: str = "mini-interact",
    mode: str = "slayer",
    runs_root: Optional[Path] = None,
) -> dict[tuple[str, str], Path]:
    """Pick the chronologically-latest ``*.trajectory.json`` per ``(db, iid)``
    whose filename mode token matches ``mode``.

    We deliberately do NOT apply ``cascade_for_combo``'s manifest-join
    "latest non-eval_failed" rule: that filter is about grader infrastructure
    and is irrelevant here — we want the literal latest trajectory the agent
    produced, so the recorded tool calls reflect the current prompt + tooling.
    """
    runs_root = runs_root or (paths.runs_root() / benchmark)
    chosen: dict[tuple[str, str], tuple[datetime, Path]] = {}
    if not runs_root.exists():
        return {}
    for path in runs_root.rglob("*.trajectory.json"):
        if mode_from_filename(path.name) != mode:
            continue
        ts = _timestamp(path.name)
        if ts is None:
            continue
        try:
            rel = path.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        db, iid, _ = parts
        key = (db, iid)
        if key not in chosen or ts > chosen[key][0]:
            chosen[key] = (ts, path)
    return {k: v[1] for k, v in chosen.items()}


# ---------------------------------------------------------------------------
# Tool-call / result pairing + classification
# ---------------------------------------------------------------------------
def _result_text(content: Any) -> str:
    """Flatten a tool-result ``content`` to text. It is either a plain string
    (``{"result": "…"}`` / ``Error executing tool …``) or a list of
    ``{"type": "text", "text": "…"}`` blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return "" if content is None else str(content)


def classify_result(text: str, *, is_error: bool) -> tuple[str, Optional[str]]:
    """Classify one query-tool result into ``(bucket, sub_tag)``.

    Buckets: ``dsl`` / ``semantic`` / ``gate`` / ``infra`` (matched via the
    taxonomy), ``other`` (an ``is_error`` result matching no pattern), or
    ``ok``.
    """
    for sub_tag, bucket, rx in ERROR_TAXONOMY:
        if rx.search(text):
            return bucket, sub_tag
    if is_error:
        return "other", None
    return "ok", None


def _build_id_to_name(trajectory: list[Any]) -> dict[str, str]:
    """Map ``tool_use_id → tool_name`` from AssistantMessage content blocks.

    Mirrors the first pass of
    ``_run_capture.extract_tool_stats_from_claude_sdk_trajectory`` but keys on
    ``id``/``name`` directly (block ``type`` is sometimes absent in the stored
    trajectory shape)."""
    id_to_name: dict[str, str] = {}
    for item in trajectory:
        if not isinstance(item, dict) or item.get("type") != "AssistantMessage":
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            continue
        for block in data.get("content") or []:
            if isinstance(block, dict) and isinstance(block.get("id"), str):
                id_to_name[block["id"]] = block.get("name") or "<unknown>"
    return id_to_name


def query_call_sequence(
    trajectory: list[Any],
) -> list[dict[str, Any]]:
    """Return the chronological sequence of query-tool RESULTS in the
    trajectory, each as ``{"tool", "bucket", "sub_tag", "is_error", "text"}``.

    Only results for the five :data:`QUERY_TOOLS` are kept. Order follows the
    UserMessage (tool-result) order in the trajectory, which is the order the
    agent observed them.
    """
    id_to_name = _build_id_to_name(trajectory)
    seq: list[dict[str, Any]] = []
    for item in trajectory:
        if not isinstance(item, dict) or item.get("type") != "UserMessage":
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            continue
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            tid = block.get("tool_use_id")
            name = id_to_name.get(tid) if isinstance(tid, str) else None
            if name not in QUERY_TOOLS:
                continue
            text = _result_text(block.get("content"))
            is_error = block.get("is_error") is True
            bucket, sub_tag = classify_result(text, is_error=is_error)
            seq.append(
                {
                    "tool": name,
                    "bucket": bucket,
                    "sub_tag": sub_tag,
                    "is_error": is_error,
                    "text": text[:400],
                }
            )
    return seq


def _dsl_chains(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find maximal runs of consecutive ``dsl`` results in ``seq``.

    A chain is ``recovered`` iff a non-``dsl`` query call follows it (the agent
    stopped making that DSL mistake); ``unrecovered`` iff the chain runs to the
    last query call. ``length`` = number of DSL rejections = retries the agent
    spent on that mistake before moving on.
    """
    chains: list[dict[str, Any]] = []
    i = 0
    n = len(seq)
    while i < n:
        if seq[i]["bucket"] != "dsl":
            i += 1
            continue
        j = i
        sub_tags: list[str] = []
        while j < n and seq[j]["bucket"] == "dsl":
            sub_tags.append(seq[j]["sub_tag"] or "validation_other")
            j += 1
        recovered = j < n  # a non-dsl query call follows the chain
        chains.append(
            {
                "length": j - i,
                "recovered": recovered,
                "sub_tags": sub_tags,
            }
        )
        i = j
    return chains


def _budget_signal(trajectory: list[Any]) -> dict[str, Any]:
    """Scan all tool results for the ``[Remaining budget: X / Y]`` note and the
    ``Budget exhausted`` gate message to reconstruct a budget signal.

    Returns ``{"final_remaining", "total", "exhausted_gate_hits"}``. ``final_*``
    are from the LAST budget note seen (None if the trajectory carries none —
    e.g. MCP-only results that don't append the note)."""
    final_remaining: Optional[float] = None
    total: Optional[float] = None
    exhausted = 0
    for item in trajectory:
        if not isinstance(item, dict) or item.get("type") != "UserMessage":
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            continue
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            text = _result_text(block.get("content"))
            if "Budget exhausted" in text:
                exhausted += 1
            m = _BUDGET_NOTE_RE.search(text)
            if m:
                final_remaining = float(m.group(1))
                total = float(m.group(2))
    return {
        "final_remaining": final_remaining,
        "total": total,
        "exhausted_gate_hits": exhausted,
    }


# Cost of one query preview / submit, for the budget-burn estimate. Mirrors
# ``harness.ACTION_COSTS`` (kept as literals to avoid importing the harness for
# a read-only analysis script).
_QUERY_COST = 1.0
_SUBMIT_COST = 3.0


def analyze_trajectory(data: dict[str, Any], *, db: str, iid: str) -> dict[str, Any]:
    """Produce the per-task analysis record from a loaded trajectory dict."""
    traj = data.get("trajectory") or []
    seq = query_call_sequence(traj) if isinstance(traj, list) else []

    bucket_counts = Counter(c["bucket"] for c in seq)
    dsl_subtag_counts = Counter(
        c["sub_tag"] or "validation_other" for c in seq if c["bucket"] == "dsl"
    )
    per_tool_dsl = Counter(c["tool"] for c in seq if c["bucket"] == "dsl")
    chains = _dsl_chains(seq)

    # Representative example error text per DSL sub_tag (first occurrence in
    # this task), so the report can show what each failure mode looks like.
    dsl_examples: dict[str, dict[str, str]] = {}
    for c in seq:
        if c["bucket"] != "dsl":
            continue
        tag = c["sub_tag"] or "validation_other"
        if tag not in dsl_examples:
            dsl_examples[tag] = {
                "iid": iid,
                "tool": c["tool"],
                "text": " ".join(c["text"].split()),  # collapse whitespace
            }

    n_query_calls = len(seq)
    n_ok = bucket_counts.get("ok", 0)
    n_dsl = bucket_counts.get("dsl", 0)
    max_consecutive = max((ch["length"] for ch in chains), default=0)

    phase1_passed = bool(data.get("phase1_passed"))
    submission_status = data.get("submission_status")
    total_reward = data.get("total_reward")
    fatal_error = data.get("error")

    budget = _budget_signal(traj) if isinstance(traj, list) else {}

    # --- major-contributor flags (only meaningful for FAILED tasks) ---
    failed = not phase1_passed
    direct = failed and submission_status in {"json_error", "translation_error"}
    never_clean = failed and n_query_calls > 0 and n_ok == 0
    # Budget-burn: the task hit force_submit / budget-exhaustion AND a material
    # share of the budget plausibly went to DSL-rejected query attempts.
    est_dsl_coins = n_dsl * _QUERY_COST
    exhausted = bool(budget.get("exhausted_gate_hits")) or (
        budget.get("final_remaining") is not None
        and budget["final_remaining"] <= _SUBMIT_COST
    )
    budget_burn = (
        failed
        and exhausted
        and budget.get("total") not in (None, 0)
        and est_dsl_coins >= 0.25 * float(budget["total"] or 0)
    )
    major_contributor = bool(direct or never_clean or budget_burn)

    return {
        "db": db,
        "iid": iid,
        "phase1_passed": phase1_passed,
        "failed": failed,
        "submission_status": submission_status,
        "total_reward": total_reward,
        "fatal_error": fatal_error,
        "n_query_calls": n_query_calls,
        "n_ok": n_ok,
        "n_dsl": n_dsl,
        "n_semantic": bucket_counts.get("semantic", 0),
        "n_gate": bucket_counts.get("gate", 0),
        "n_infra": bucket_counts.get("infra", 0),
        "n_other_error": bucket_counts.get("other", 0),
        "dsl_subtag_counts": dict(dsl_subtag_counts),
        "dsl_examples": dsl_examples,
        "per_tool_dsl": dict(per_tool_dsl),
        "max_consecutive_dsl": max_consecutive,
        "chains": chains,
        "budget": budget,
        "est_dsl_coins": est_dsl_coins,
        "flags": {
            "direct": direct,
            "never_clean": never_clean,
            "budget_burn": budget_burn,
            "major_contributor": major_contributor,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-task records into headline aggregates for the report."""
    n_tasks = len(per_task)
    n_failed = sum(1 for t in per_task if t["failed"])

    dsl_subtags: Counter[str] = Counter()
    per_tool_dsl: Counter[str] = Counter()
    bucket_totals: Counter[str] = Counter()
    retry_hist: Counter[int] = Counter()  # chain length -> count
    recovered = unrecovered = 0
    total_query_calls = 0
    # up to _MAX_EXAMPLES_PER_SUBTAG distinct example texts per DSL sub_tag
    examples: dict[str, list[dict[str, str]]] = {}
    seen_example_text: dict[str, set[str]] = {}

    tasks_with_dsl = 0
    for t in per_task:
        for k, v in t["dsl_subtag_counts"].items():
            dsl_subtags[k] += v
        for tag, ex in t.get("dsl_examples", {}).items():
            bucket = examples.setdefault(tag, [])
            seen = seen_example_text.setdefault(tag, set())
            if len(bucket) < _MAX_EXAMPLES_PER_SUBTAG and ex["text"] not in seen:
                seen.add(ex["text"])
                bucket.append(ex)
        for k, v in t["per_tool_dsl"].items():
            per_tool_dsl[k] += v
        bucket_totals["dsl"] += t["n_dsl"]
        bucket_totals["semantic"] += t["n_semantic"]
        bucket_totals["gate"] += t["n_gate"]
        bucket_totals["infra"] += t["n_infra"]
        bucket_totals["other"] += t["n_other_error"]
        bucket_totals["ok"] += t["n_ok"]
        total_query_calls += t["n_query_calls"]
        if t["n_dsl"] > 0:
            tasks_with_dsl += 1
        for ch in t["chains"]:
            retry_hist[ch["length"]] += 1
            if ch["recovered"]:
                recovered += 1
            else:
                unrecovered += 1

    flag_counts = {
        "direct": sum(1 for t in per_task if t["flags"]["direct"]),
        "never_clean": sum(1 for t in per_task if t["flags"]["never_clean"]),
        "budget_burn": sum(1 for t in per_task if t["flags"]["budget_burn"]),
        "major_contributor": sum(
            1 for t in per_task if t["flags"]["major_contributor"]
        ),
    }
    major_tasks = [
        {
            "iid": t["iid"],
            "submission_status": t["submission_status"],
            "n_dsl": t["n_dsl"],
            "max_consecutive_dsl": t["max_consecutive_dsl"],
            "est_dsl_coins": t["est_dsl_coins"],
            "budget": t["budget"],
            "flags": {k: v for k, v in t["flags"].items() if k != "major_contributor"},
            "dsl_subtag_counts": t["dsl_subtag_counts"],
        }
        for t in per_task
        if t["flags"]["major_contributor"]
    ]

    return {
        "n_tasks": n_tasks,
        "n_failed": n_failed,
        "n_tasks_with_dsl": tasks_with_dsl,
        "total_query_calls": total_query_calls,
        "bucket_totals": dict(bucket_totals),
        "dsl_subtag_counts": dict(dsl_subtags.most_common()),
        "dsl_examples": examples,
        "per_tool_dsl": dict(per_tool_dsl.most_common()),
        "retry_histogram": {str(k): retry_hist[k] for k in sorted(retry_hist)},
        "chains_recovered": recovered,
        "chains_unrecovered": unrecovered,
        "flag_counts": flag_counts,
        "major_contributor_tasks": sorted(
            major_tasks, key=lambda x: -x["n_dsl"]
        ),
    }


def build_payload(
    *,
    benchmark: str = "mini-interact",
    mode: str = "slayer",
    runs_root: Optional[Path] = None,
) -> dict[str, Any]:
    """End-to-end: select latest trajectories, analyze each, aggregate.

    Returns ``{"benchmark", "mode", "per_task", "aggregate"}`` — the payload the
    notebook and the CLI both consume.
    """
    chosen = latest_trajectory_per_iid(
        benchmark=benchmark, mode=mode, runs_root=runs_root
    )
    per_task: list[dict[str, Any]] = []
    for (db, iid), path in sorted(chosen.items()):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        per_task.append(analyze_trajectory(data, db=db, iid=iid))
    return {
        "benchmark": benchmark,
        "mode": mode,
        "per_task": per_task,
        "aggregate": aggregate(per_task),
    }


# ---------------------------------------------------------------------------
# CLI text render (quick terminal sanity-check; the notebook is the rich view)
# ---------------------------------------------------------------------------
def render_text(payload: dict[str, Any]) -> str:
    agg = payload["aggregate"]
    n = agg["n_tasks"]
    lines: list[str] = []
    bar = "=" * 78
    lines.append(bar)
    lines.append(
        f"Query-syntax rejection summary  benchmark={payload['benchmark']}  "
        f"mode={payload['mode']}  n_tasks={n}  (latest trajectory per iid)"
    )
    lines.append(bar)
    if not n:
        lines.append("No matching trajectories.")
        return "\n".join(lines)

    bt = agg["bucket_totals"]
    lines.append("")
    lines.append(
        f"Query-tool calls: {agg['total_query_calls']}   "
        f"ok={bt.get('ok', 0)}  dsl={bt.get('dsl', 0)}  "
        f"semantic={bt.get('semantic', 0)} (excl)  "
        f"gate={bt.get('gate', 0)} (excl)  "
        f"infra={bt.get('infra', 0)} (excl)  other={bt.get('other', 0)}"
    )
    lines.append(
        f"Tasks failed: {agg['n_failed']}/{n}   "
        f"tasks with >=1 DSL rejection: {agg['n_tasks_with_dsl']}/{n}"
    )

    lines.append("")
    lines.append("DSL failure modes (headline):")
    lines.append(f"  {'sub_tag':<24}{'count':>8}{'share':>9}")
    total_dsl = bt.get("dsl", 0) or 1
    for tag, c in agg["dsl_subtag_counts"].items():
        lines.append(f"  {tag:<24}{c:>8}{c / total_dsl * 100:>8.1f}%")

    lines.append("")
    lines.append("DSL rejections by tool:")
    for tool, c in agg["per_tool_dsl"].items():
        lines.append(f"  {c:>6}  {tool}")

    lines.append("")
    lines.append(
        f"Retry-to-recovery: {agg['chains_recovered']} recovered chains, "
        f"{agg['chains_unrecovered']} unrecovered. Chain-length histogram:"
    )
    for length, c in agg["retry_histogram"].items():
        lines.append(f"  {length} consecutive DSL retries: {c} chains")

    lines.append("")
    fc = agg["flag_counts"]
    lines.append(
        f"Major-contributor-to-failure: {fc['major_contributor']} tasks "
        f"(direct={fc['direct']}  never_clean={fc['never_clean']}  "
        f"budget_burn={fc['budget_burn']})"
    )
    for t in agg["major_contributor_tasks"][:20]:
        on = ",".join(k for k, v in t["flags"].items() if v)
        lines.append(
            f"  {t['iid']:<28} status={t['submission_status']!s:<16} "
            f"n_dsl={t['n_dsl']:<3} maxrun={t['max_consecutive_dsl']:<2} [{on}]"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", default="mini-interact")
    parser.add_argument(
        "--mode", default="slayer", choices=("raw", "slayer")
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the machine-readable payload instead of the human table.",
    )
    args = parser.parse_args(argv)

    payload = build_payload(benchmark=args.benchmark, mode=args.mode)
    if args.json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0
    print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
