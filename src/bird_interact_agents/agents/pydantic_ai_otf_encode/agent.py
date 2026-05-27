"""Public class + run_task entry point for the on-the-fly KB-encode adapter.

`PydanticAIOtfEncodeAgent` is the SLayer-only / a-interact-only adapter
selectable via ``--framework pydantic_ai_otf_encode`` in the benchmark
runner. Mirrors the structure of ``pydantic_ai_recursive.agent`` —
copy-not-subclass per the plan — with these DEV-1454 specifics:

* Forces ``slayer_setup='on-the-fly'`` at __init__; raises ValueError
  on any other value.
* Uses this module's local ``factories`` / ``deps`` / ``prompts`` so
  the sub-clarifier registers the new ``kb_to_slayer`` tool and the
  ``AgentRecord`` Literal accepts ``"kb_encoder"``.
* Result row gains a top-level ``kb_encoded`` field carrying the
  per-task dedup registry.

The constructor reserve, MCP server setup, OTF cache + scratch dir
helpers, and trajectory aggregation behave identically to the
recursive adapter so the parity tests pass.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import expressions as sqlglot_exp
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
from sqlalchemy.exc import SQLAlchemyError

from slayer.core.query import SlayerQuery
from slayer.engine.query_engine import SlayerQueryEngine
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.agents._run_capture import (
    _count_turns,
    _extract_tool_stats,
    _serialize_messages,
)
from bird_interact_agents.agents._session_log import write_index, write_session
from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
    AgentRecord,
    SharedTaskState,
    TaskDeps,
)
from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
    _build_projection_resolver,
    _build_query_constructor,
    _build_root_clarifier,
)
from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
    PROJECTION_RESOLVER_PROMPT,
    QUERY_CONSTRUCTOR_PROMPT,
    ROOT_CLARIFIER_PROMPT,
)
from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
    _aggregate_runs,
    _fold_run_usage_into_deps,
    _ResolverResult,
)
from bird_interact_agents.harness import (
    ACTION_COSTS,
    MAX_MODEL_TURNS,
    SampleStatus,
    finalize_result_row,
    load_db_data_if_needed,
    slayer_mcp_stdio_config,
)
from bird_interact_agents.hard8_preprocessor import (
    build_task_variant_storage,
    extract_deleted_kb_ids,
)
from bird_interact_agents.usage import TokenUsage
from bird_interact_agents.slayer_otf import ensure_db_reference
from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
    make_setup_build_encoder,
)
from bird_interact_agents import paths as _paths

logger = logging.getLogger(__name__)


# Per-task cap on the number of error-sample blobs retained — matches
# the recursive adapter so trajectories aren't trivially comparable.
_TOOL_ERROR_SAMPLES_PER_TASK = 10

# DEV-1454: the projection resolver no longer uses structured output (so it can
# reason in text). PROJECTION_RESOLVER_PROMPT is imported from the recursive
# adapter (shared); rather than fork it, append the submit instruction here at
# the otf call site so the recursive adapter is unaffected.
_PROJECTION_SUBMIT_SUFFIX = (
    "\n\nDELIVER your confirmed output columns by calling "
    "`submit_projection(columns_json=...)` with a JSON array of column-name "
    'strings (e.g. ["region", "revenue"]); an empty array [] is allowed and '
    "triggers a recovery pass. Reason in text first, then call it once and "
    "reply briefly to finish — the result is only recorded via the tool."
)


def _constructor_reserve() -> float:
    """Bird-coin reserve held back from the clarifier phase. Identical
    formula to the recursive adapter (`2 * ask_user + submit_query`);
    the parity test pins them equal so a future change to one MUST be
    mirrored to the other."""
    return (
        2 * ACTION_COSTS["ask_user"]
        + ACTION_COSTS["submit_query"]
    )


# Tools whose writes are validated BEFORE they persist (DEV-1454).
_WRITE_TOOLS = frozenset({"edit_model", "create_model"})
# Expected failure classes from a validation query — a genuinely bad expression
# (dialect error, unknown column/function) or a malformed query. Anything else
# is treated as a validation-harness bug and fails OPEN (never blocks a write).
_VALIDATION_FAILURES = (SQLAlchemyError, ValueError)
# A model source change in the SAME edit_model call would make the inline copy
# validate new columns against the OLD source — skip rather than mis-validate.
_SOURCE_CHANGE_KEYS = ("sql_table", "sql", "source_queries")


def _inline_validation_queries(name: str, args: dict) -> list[tuple[str, dict]]:
    """Build ``(label, query_dict)`` that validate the agent's PROPOSED entities
    against an in-memory copy WITHOUT persisting anything.

    Each query selects the proposed entity with ``limit=0`` so its SQL compiles +
    executes through the full semantic layer (joins, cross-model refs) but returns
    no rows. New/changed columns and measures are aliased to one-off names so an
    inline ``ModelExtension`` (which *appends*, never replaces) validates the
    agent's NEW expression instead of resolving a stale same-named entity.

    Covers columns + measures (edit_model) and create_model. Aggregations (inline
    ModelExtension has no aggregations field), scalar-only edits, and same-call
    source swaps emit no query — documented limitations, left to the final
    presence+tagging verify.
    """
    args = args or {}
    if name == "edit_model":
        model = args.get("model_name")
        if not model:
            return []
        # A same-call source swap invalidates the inline-against-old-source copy.
        if any(args.get(k) for k in _SOURCE_CHANGE_KEYS):
            return []
        cols = [
            c for c in (args.get("columns") or [])
            if isinstance(c, dict) and c.get("name") and c.get("sql")
        ]
        real_cols = [
            {"name": c["name"], "sql": c["sql"], "type": c.get("type", "string")}
            for c in cols
        ]
        out: list[tuple[str, dict]] = []
        for i, c in enumerate(cols):
            alias = f"__v_col_{i}"
            ext = {"source_name": model,
                   "columns": [{"name": alias, "sql": c["sql"],
                                "type": c.get("type", "string")}]}
            out.append((f"column {model}.{c['name']}",
                        {"source_model": ext, "dimensions": [alias], "limit": 0}))
        measures = [
            m for m in (args.get("measures") or [])
            if isinstance(m, dict) and m.get("name") and m.get("formula")
        ]
        # NB known limitation: a measure that references a column being MODIFIED
        # in this same edit_model call resolves to the OLD column — ModelExtension
        # appends (it can't replace), and SLayer takes the first same-named match,
        # which is the base model's existing column. So such a measure is validated
        # against the pre-edit column. New columns (added this call) don't collide
        # and validate correctly. Faithfully fixing the modify-in-same-call case
        # would need a full replacement-copy of the model (heavier); the one-off
        # alias was chosen for simplicity. This is narrow (the encoder usually
        # adds a column and its measures in separate calls).
        for j, m in enumerate(measures):
            alias = f"__v_meas_{j}"
            ext = {"source_name": model, "columns": real_cols,
                   "measures": [{"name": alias, "formula": m["formula"]}]}
            out.append((f"measure {model}.{m['name']}",
                        {"source_model": ext, "measures": [{"formula": alias}],
                         "limit": 0}))
        return out
    if name == "create_model":
        nm = args.get("name")
        backing = args.get("query")
        if backing is not None:
            # Backing-query model: validate the query the agent supplied.
            if isinstance(backing, dict):
                q = dict(backing)
                q["limit"] = 0
                return [(f"model {nm} (backing query)", q)]
            return []  # multi-stage list — not inline-validated (limitation)
        cols = [
            c for c in (args.get("columns") or [])
            if isinstance(c, dict) and c.get("name")
        ]
        if not (nm and cols and (args.get("sql_table") or args.get("sql"))):
            return []
        inline: dict[str, Any] = {
            "name": f"__v_{nm}", "data_source": args.get("data_source"),
            "columns": [{"name": c["name"], "sql": c.get("sql", c["name"]),
                         "type": c.get("type", "string")} for c in cols],
        }
        if args.get("sql_table"):
            inline["sql_table"] = args["sql_table"]
        else:
            inline["sql"] = args["sql"]
        out = []
        # Validate EVERY proposed column (selecting one as a dimension compiles
        # only that column's SQL) — not just the first — so a bad later column
        # can't slip past the gate and persist.
        for c in cols:
            out.append((f"model {nm}.{c['name']}",
                        {"source_model": inline, "dimensions": [c["name"]],
                         "limit": 0}))
        # …and every proposed measure (formula resolves against the columns above).
        for m in (args.get("measures") or []):
            if isinstance(m, dict) and m.get("name") and m.get("formula"):
                out.append((f"model {nm}.{m['name']}",
                            {"source_model": inline,
                             "measures": [{"formula": m["formula"]}], "limit": 0}))
        return out
    return []


def _health_query(datasource: str) -> dict:
    """A trivial ``SELECT 1`` against ``datasource`` — used to tell a genuine bad
    expression from an engine/connection failure."""
    return {
        "source_model": {
            "name": "__healthcheck__", "sql": "SELECT 1 AS one",
            "data_source": datasource,
            "columns": [{"name": "one", "sql": "one", "type": "number"}],
        },
        "dimensions": ["one"], "limit": 0,
    }


async def _engine_healthy(engine: Any, datasource: Any) -> bool:
    """True if the validation engine can run ``SELECT 1`` against ``datasource``
    (so a query failure is the agent's SQL, not infra). When the datasource is
    unknown we can't probe — assume healthy so we fail CLOSED (block the suspect
    write) rather than silently letting it through."""
    if not datasource:
        return True
    try:
        await engine.execute(SlayerQuery.model_validate(_health_query(datasource)))
        return True
    except Exception:  # noqa: BLE001 — any probe failure ⇒ engine/DB unreachable
        return False


async def _resolve_validation_datasource(storage: Any, name: str, args: dict) -> Any:
    """The datasource for the proposed write — explicit ``data_source`` arg, else
    (edit_model) the host model's datasource read from storage."""
    ds = (args or {}).get("data_source")
    if ds:
        return ds
    if name == "edit_model" and (args or {}).get("model_name"):
        try:
            model = await storage.get_model(args["model_name"])
            return getattr(model, "data_source", None)
        except Exception:  # noqa: BLE001 — best-effort
            return None
    return None


def _validation_feedback(tool_name: str, problems: list[str]) -> str:
    """Tool result returned to the agent when a proposed write fails validation.
    The change was NOT persisted — this string IS the only effect."""
    return (
        "⚠️ VALIDATION FAILED — the change was NOT saved (nothing was "
        "persisted). The proposed SQL does not execute against the live "
        "database:\n"
        + "\n".join(f"  - {p}" for p in problems)
        + "\n\nThis datasource is SQLite — use LIKE/LOWER (not ILIKE), REAL "
        f"(not DOUBLE PRECISION), JSON_EXTRACT, etc. Fix the SQL and call "
        f"`{tool_name}` again with the corrected expression(s)."
    )


async def _collect_validation_problems(
    name: str, tool_args: dict, *, storage: Any, engine: Any,
) -> list[str]:
    """Run each inline validation query; on an expected failure, health-probe to
    rule out infra, recording a problem only when the engine itself is healthy."""
    problems: list[str] = []
    datasource: Any = None
    resolved = False
    for label, qd in _inline_validation_queries(name, tool_args):
        try:
            await engine.execute(SlayerQuery.model_validate(qd))
        except _VALIDATION_FAILURES as exc:
            if not resolved:
                datasource = await _resolve_validation_datasource(
                    storage, name, tool_args,
                )
                resolved = True
            if await _engine_healthy(engine, datasource):
                problems.append(f"{label}: {type(exc).__name__}: {exc}")
            else:
                logger.warning(
                    "validate-before-persist: engine unhealthy "
                    "(datasource=%s) — skipping gate for %s: %s",
                    datasource, label, exc,
                )
    return problems


# ---------------------------------------------------------------------------
# DEV-1478: literal-existence validation. Catches the broken-predicate
# failure mode where the agent emits `<col> = '<literal>'` / `<col> IN (...)`
# whose literals never occur in the column's stored distinct values
# (Column.sampled_values, falling back to Column.sampled text).
# ---------------------------------------------------------------------------


# Wrapper functions whose argument is the "bare column" for literal-check
# purposes (the encoder STYLE_GUIDE forces LOWER(TRIM(...))). Add new
# wrappers conservatively — wrappers that ALTER the value (e.g. SUBSTR,
# CAST to a different type) MUST NOT be peeled.
_PEELABLE_WRAPPERS = (
    sqlglot_exp.Lower, sqlglot_exp.Upper, sqlglot_exp.Trim,
)

def _peel_wrapper(node: Any) -> tuple[Any, bool]:
    """Peel safe text-normalising wrappers (LOWER/UPPER/TRIM, possibly
    nested) until a non-wrapper expression is reached.

    Returns ``(inner_node, was_wrapped)`` — the bare inner expression and
    a boolean recording whether ANY wrapper was stripped. The boolean
    feeds the literal-existence check (Codex DEV-1478 follow-up): when
    the LHS was un-normalized at the SQL level, the right-hand literal
    must NOT be silently casefolded by the validator — otherwise the
    validator accepts a broken predicate like ``col = 'Yes'`` against a
    column whose sampled distinct values are ``['yes', 'no']`` (the
    runtime returns zero rows even though the validator would pass).
    """
    cur = node
    wrapped = False
    while isinstance(cur, _PEELABLE_WRAPPERS):
        cur = cur.this
        wrapped = True
    return cur, wrapped


def _column_ref(node: Any) -> tuple[str | None, str] | None:
    """If ``node`` is a `Column` (post-peeling), return
    ``(alias_or_none, column_name)``. Else None.

    sqlglot's `Column` has `.this` (the column name) and `.table` (the
    alias / table-qualifier, or None for a bare column reference).
    """
    if not isinstance(node, sqlglot_exp.Column):
        return None
    name = node.this.name if hasattr(node.this, "name") else str(node.this)
    table = node.table or None
    return (table or None, name)


async def _resolve_column_via_host(
    host: Any, storage: Any, host_model_name: str,
    alias: str | None, col_name: str, *, db: str | None = None,
) -> Any:
    """Resolve a column reference to a SLayer Column object (best-effort):

      a. Bare column name → look up on the pre-fetched host model.
      b. ``<alias>.<col>`` where ``<alias>`` matches the host's own name →
         same as (a).
      c. ``<alias>.<col>`` where ``<alias>`` matches one of the host's
         ``joins[*].target_model`` → load the target via storage (scoped
         to ``db`` when known so a same-named model in another datasource
         doesn't shadow the right one) and look up ``<col>`` there.
      d. Multi-hop ``<path>__<alias>.<col>`` is OUT of scope; returns None.

    Returns None on any failure; caller treats None as "skip silently"
    (no false-positive). ``host`` is the pre-resolved host SlayerModel —
    the caller fetches it once at the top of the walk so cross-table
    references don't trigger redundant ``storage.get_model`` calls (which
    inflate test counts on agent-side validation tests).
    """
    if host is None:
        return None
    if alias is None or alias == host_model_name:
        for c in getattr(host, "columns", None) or []:
            if c.name == col_name:
                return c
        return None
    for join in getattr(host, "joins", None) or []:
        if getattr(join, "target_model", None) == alias:
            if storage is None:
                return None
            try:
                target = await _get_model_scoped(storage, alias, db)
            except Exception:  # noqa: BLE001
                return None
            if target is None:
                return None
            for c in getattr(target, "columns", None) or []:
                if c.name == col_name:
                    return c
            return None
    return None


async def _get_model_scoped(storage: Any, name: str, db: str | None) -> Any:
    """Load a model by name, scoped to ``db`` when known so a same-named
    model in a different datasource (YAMLStorage's priority resolution)
    doesn't shadow the intended one. When ``db`` is empty/None, falls
    back to the unscoped lookup the existing
    `test_write_gate_resolves_datasource_via_storage_fallback` pins."""
    if db:
        try:
            return await storage.get_model(name, data_source=db)
        except TypeError:
            # Older storage signatures without `data_source` kwarg.
            return await storage.get_model(name)
    return await storage.get_model(name)


def _sampled_set(col: Any) -> list[str] | None:
    """Return the set of known distinct values for ``col`` as a list, or
    None if the literal-existence check should be skipped.

    Source of truth is ``Column.sampled_values`` (DEV-1480's structured
    field). When that's present (including ``[]`` for the "zero distinct"
    case), use it directly.

    When it's absent (older SLayer / not-yet-profiled columns), DO NOT
    naive-split the human-readable ``Column.sampled`` text — fragments
    of values that themselves contain commas (e.g. ``"R$ 1,000-3,000"``
    or ``"Springfield, IL"``) would mis-fragment and produce false
    positives, violating the validator's no-false-positive contract.
    Skip the check entirely instead; DEV-1480 ships ``sampled_values``
    and makes the check effective everywhere. (Codex PR #4 finding L.)
    """
    structured = getattr(col, "sampled_values", None)
    if structured is not None:
        return list(structured)
    return None


def _string_literals_under(eq_or_in: Any) -> list[str]:
    """Extract string literals from an EQ or In expression's RHS. For EQ:
    one literal (or none if not a string). For In: every Literal in
    ``expressions`` that is a string. Non-string literals (numeric, etc.)
    are out of scope — only string literals are checked."""
    out: list[str] = []
    if isinstance(eq_or_in, sqlglot_exp.EQ):
        rhs = eq_or_in.expression
        if isinstance(rhs, sqlglot_exp.Literal) and rhs.is_string:
            out.append(rhs.this)
    elif isinstance(eq_or_in, sqlglot_exp.In):
        for e in eq_or_in.expressions or []:
            if isinstance(e, sqlglot_exp.Literal) and e.is_string:
                out.append(e.this)
    return out


def _iter_target_sqls(name: str, tool_args: dict) -> list[str]:
    """Every SQL string the literal walker should scan from a proposed
    edit_model / create_model payload."""
    out: list[str] = []
    if name == "edit_model":
        for c in tool_args.get("columns") or []:
            if isinstance(c, dict):
                if c.get("sql"):
                    out.append(str(c["sql"]))
                if c.get("filter"):
                    out.append(str(c["filter"]))
        for m in tool_args.get("measures") or []:
            if isinstance(m, dict) and m.get("formula"):
                out.append(str(m["formula"]))
    elif name == "create_model":
        for c in tool_args.get("columns") or []:
            if isinstance(c, dict):
                if c.get("sql"):
                    out.append(str(c["sql"]))
                if c.get("filter"):
                    out.append(str(c["filter"]))
        for m in tool_args.get("measures") or []:
            if isinstance(m, dict) and m.get("formula"):
                out.append(str(m["formula"]))
        # NB: `create_model(query=[...])` backing-query stages are NOT
        # collected here — their column references resolve against each
        # stage's own `source_model`, not the outer create_model's name.
        # Those are scanned separately via `_iter_query_stage_targets`
        # below, which pairs each query SQL with its stage's source_model
        # for proper host resolution.
    return out


def _iter_query_stage_targets(
    name: str, tool_args: dict,
) -> list[tuple[str, str]]:
    """Codex finding (PR #4 follow-up): walk `create_model(query=[stage1,
    stage2, ...])` backing-query stages and yield ``(sql, source_model)``
    pairs. Each stage's literals resolve against ITS source_model — not
    the outer create_model's name — because the agent's SQL inside a
    query-stage filter (e.g. ``"col = 'High Income'"``) references
    columns of the stage's source table.

    Only the string-form `source_model` (a model name lookup) is in
    scope; inline ModelExtension `source_model` dicts are silently
    skipped (they don't sit in storage and the synthetic-host trick is
    overkill for stage-internal references).

    Returns an empty list for anything other than `create_model` so the
    caller can flatten unconditionally.
    """
    if name != "create_model":
        return []
    query = tool_args.get("query")
    if query is None:
        return []
    stages = query if isinstance(query, list) else [query]
    out: list[tuple[str, str]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        src = stage.get("source_model")
        if not isinstance(src, str) or not src:
            continue
        for f in stage.get("filters") or []:
            if isinstance(f, str) and f:
                out.append((f, src))
        for m in stage.get("measures") or []:
            # SlayerQuery measures accept BOTH dict-form
            # `{"formula": "..."}` AND bare-string form `"col:sum"` or
            # `"CASE WHEN col = 'x' THEN 1 ELSE 0 END:sum"`. Walk both.
            if isinstance(m, dict) and m.get("formula"):
                out.append((str(m["formula"]), src))
            elif isinstance(m, str) and m:
                out.append((m, src))
        for d in stage.get("dimensions") or []:
            if isinstance(d, dict) and d.get("filter"):
                out.append((str(d["filter"]), src))
    return out


def _host_model_name(name: str, tool_args: dict) -> str | None:
    """The host model the proposed entity will live on. For edit_model
    it's ``model_name``; for create_model it's the new model's ``name``
    (so cross-table refs in its columns resolve via joins on that name
    if it exists)."""
    if name == "edit_model":
        return tool_args.get("model_name")
    if name == "create_model":
        return tool_args.get("name")
    return None


class _SyntheticColumn:
    """A minimal Column stand-in for ``_resolve_column_via_host`` to find
    when the host model doesn't exist in storage yet (``create_model``).
    Carries only the fields the literal walker reads:
    ``name``, ``sampled``, ``sampled_values``."""

    __slots__ = ("name", "sampled", "sampled_values")

    def __init__(
        self, name: str, sampled: Any = None, sampled_values: Any = None,
    ) -> None:
        self.name = name
        self.sampled = sampled
        self.sampled_values = sampled_values


class _SyntheticHost:
    """Minimal host model stand-in for ``create_model`` payloads. Carries
    only what ``_resolve_column_via_host`` reads: ``columns``, ``joins``.
    Joins are empty because a not-yet-persisted model can't have edges
    in the join graph; cross-table refs from a create_model payload
    therefore silently skip (consistent with the best-effort policy)."""

    __slots__ = ("columns", "joins")

    def __init__(self, columns: list, joins: list | None = None) -> None:
        self.columns = columns
        self.joins = joins or []


def _build_synthetic_host_from_create_args(tool_args: dict) -> Any:
    """CodeRabbit PR #4 thread 1: when ``create_model`` fires for a model
    that doesn't exist in storage yet, build a stand-in host from the
    proposal's own columns so the bare-column walker can still resolve
    references and check literal membership.

    The synthetic columns carry whatever ``sampled`` / ``sampled_values``
    the proposal happens to populate — usually nothing. In that case the
    walker correctly skips (the column has no known set), preserving the
    best-effort no-false-positive contract. The fix's value lies in the
    rare cases where the agent DOES copy upstream sampled metadata into
    the create_model payload (R-MULTISTAGE stages, R-EXISTS per-parent
    stages, etc.), AND it lets the unparseable-SQL gate still fire.

    Returns None when the payload has no columns at all (nothing to
    resolve against)."""
    cols_arg = tool_args.get("columns") or []
    columns: list = []
    for c in cols_arg:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        columns.append(_SyntheticColumn(
            name=c["name"],
            sampled=c.get("sampled"),
            sampled_values=c.get("sampled_values"),
        ))
    if not columns:
        return None
    return _SyntheticHost(columns=columns)


async def _collect_literal_problems(
    name: str, tool_args: dict, *, storage: Any, db: str = "",
) -> list[str]:
    """DEV-1478: walk every SQL string in the proposed write, find
    ``<col> = 'literal'`` and ``<col> IN ('a', 'b', ...)`` patterns
    (peeling LOWER/UPPER/TRIM wrappers), and check membership against
    the column's stored distinct values. Returns one problem string per
    column-with-misses, listing the bad literals and the full known set.

    Best-effort: parse failures, unresolvable column references, and
    sampled-set absence all yield NO problem (skip silently). This
    matches plan §C: never false-positive.

    Codex follow-up: ``db`` is now used. Storage lookups are scoped to
    the proposed write's ``data_source`` (when in args) so a same-named
    model in a different datasource doesn't shadow the right one under
    YAMLStorage's priority resolution. When ``data_source`` is absent
    from args (the corner case the existing
    ``test_write_gate_resolves_datasource_via_storage_fallback`` pins),
    falls back to an unscoped lookup."""
    problems: list[str] = []
    host_name = _host_model_name(name, tool_args)
    if not host_name or storage is None:
        return problems

    # Prefer the proposed write's data_source over the caller-supplied
    # `db` arg — the agent's contract is to pass data_source on every
    # write, and that's the authoritative scope for this lookup.
    args_db = (tool_args or {}).get("data_source")
    scope_db = str(args_db) if args_db else (db or None)

    sqls = _iter_target_sqls(name, tool_args)
    stage_targets = _iter_query_stage_targets(name, tool_args)
    if not sqls and not stage_targets:
        return problems

    host: Any = None
    if sqls:
        try:
            host = await _get_model_scoped(storage, host_name, scope_db)
        except Exception:  # noqa: BLE001 — best-effort
            host = None
        if host is None and name == "create_model":
            host = _build_synthetic_host_from_create_args(tool_args)
        if host is not None:
            for sql in sqls:
                problems.extend(
                    await _check_sql_literals(
                        sql, host, storage, host_name, db=scope_db,
                    )
                )

    # Codex follow-up: query-backed model creation
    # (`create_model(query=[stage1, stage2, ...])`) — each stage's
    # literals reference columns on the STAGE's `source_model`, not on
    # the outer create_model's name. Resolve per-stage and scope to
    # ``scope_db`` so the right datasource's model wins.
    for sql, src_model in stage_targets:
        stage_host: Any
        try:
            stage_host = await _get_model_scoped(storage, src_model, scope_db)
        except Exception:  # noqa: BLE001 — best-effort
            stage_host = None
        if stage_host is None:
            continue
        problems.extend(
            await _check_sql_literals(
                sql, stage_host, storage, src_model, db=scope_db,
            )
        )
    return problems


async def _check_sql_literals(
    sql: str, host: Any, storage: Any, host_name: str,
    *, db: str | None = None,
) -> list[str]:
    """Walk one SQL string for EQ / IN string-literal predicates and
    return one problem per column whose literals miss the host's sampled
    set. Pure helper extracted from `_collect_literal_problems` so the
    same logic applies to both the outer-entity host AND per-stage hosts
    in `create_model(query=[...])` payloads. ``db`` scopes joined-target
    storage lookups so a same-named model in another datasource doesn't
    shadow the right one.

    Best-effort: parse failures, unresolvable column refs, and missing
    sampled metadata all yield NO problem (skip silently)."""
    out: list[str] = []
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:  # noqa: BLE001 — unparseable is the existing SQL gate's job
        return out
    if tree is None:
        return out
    for node in tree.walk():
        if not isinstance(node, (sqlglot_exp.EQ, sqlglot_exp.In)):
            continue
        lhs_peeled, lhs_wrapped = _peel_wrapper(node.this)
        ref = _column_ref(lhs_peeled)
        if ref is None:
            continue
        alias, col_name = ref
        literals = _string_literals_under(node)
        if not literals:
            continue
        col = await _resolve_column_via_host(
            host, storage, host_name, alias, col_name, db=db,
        )
        if col is None:
            continue
        known = _sampled_set(col)
        if known is None:
            continue
        if not known:
            literals_str = ", ".join(f"'{lit}'" for lit in literals)
            qualified = f"{alias}.{col_name}" if alias else col_name
            out.append(
                f"column {qualified}: literal(s) {literals_str} cannot "
                f"match because the column has no values (all NULL / "
                f"empty). DEFER this KB and cite the empty column."
            )
            continue
        # Codex DEV-1478 follow-up: only strip + casefold both sides
        # when the LHS was actually wrapped by LOWER/UPPER/TRIM at the
        # SQL level. The STYLE_GUIDE *encourages* `LOWER(TRIM(<col>))`
        # but it is an LLM-level convention, not a runtime guarantee —
        # silently normalising the validator's comparison would accept
        # an unwrapped `col = 'Yes'` against sampled `['yes', 'no']`
        # even though the SQLite runtime returns zero rows for that
        # predicate. When the LHS was unwrapped, compare literals
        # against `known` exactly.
        #
        # Codex PR #4 finding M (the original whitespace-tolerance
        # motivation) still holds: when the encoder did wrap with
        # LOWER+TRIM, both sides get normalised the same way so a
        # sampled value `' yes '` matches a literal `'yes'`.
        if lhs_wrapped:
            known_compare = {k.strip().casefold() for k in known}
            misses = [
                lit for lit in literals
                if lit.strip().casefold() not in known_compare
            ]
        else:
            known_compare = set(known)
            misses = [lit for lit in literals if lit not in known_compare]
        if not misses:
            continue
        qualified = f"{alias}.{col_name}" if alias else col_name
        misses_str = ", ".join(f"'{m}'" for m in misses)
        known_str = ", ".join(f"'{v}'" for v in known)
        out.append(
            f"column {qualified}: literal(s) {misses_str} do not occur "
            f"in this column's sampled distinct values. Known values: "
            f"{known_str}. Pick one that exists, or DEFER this KB with "
            f"notes citing the failing literal(s) and the known set."
        )
    return out


async def _validate_and_call(
    call_tool, name: str, tool_args: dict, *, storage: Any, engine: Any,
):
    """``process_tool_call`` core (DEV-1454): for ``edit_model``/``create_model``,
    validate the agent's PROPOSED entities against an in-memory copy via a
    client-side ``SlayerQueryEngine`` (a ``query(limit=0)`` that RAISES on bad
    SQL) BEFORE persisting. On failure the real write is NOT called — the error
    is returned so the agent self-corrects on its next turn; nothing invalid is
    ever saved. An unexpected harness error fails OPEN (the write proceeds) so a
    bug here never blocks valid work.

    DEV-1478: also runs the literal-existence check via
    ``_collect_literal_problems`` — string literals in ``=`` / ``IN``
    predicates that don't occur in the column's sampled distinct values
    are rejected with a feedback string that lists the valid values. SQL
    and literal feedback are combined into ONE tool result so the agent
    sees the full picture in a single turn."""
    if name not in _WRITE_TOOLS:
        return await call_tool(name, tool_args, None)
    args = tool_args or {}
    try:
        sql_problems = await _collect_validation_problems(
            name, args, storage=storage, engine=engine,
        )
    except Exception:  # noqa: BLE001 — harness bug must never block a write
        logger.exception(
            "validate-before-persist harness error; allowing write %s", name,
        )
        return await call_tool(name, tool_args, None)
    try:
        literal_problems = await _collect_literal_problems(
            name, args, storage=storage,
        )
    except Exception:  # noqa: BLE001 — literal-collector bug must fail OPEN
        # CodeRabbit PR #4 thread 2 + Codex: do NOT short-circuit to
        # `call_tool` here — `sql_problems` from the existing SQL gate may
        # already be non-empty, and skipping straight to the write would
        # turn a literal-collector harness bug into a bypass for known-
        # invalid SQL. Drop only the literal gate's results; the
        # `if problems:` check below still blocks the write when the SQL
        # gate flagged anything.
        logger.exception(
            "literal-existence validation harness error; allowing write %s "
            "(SQL-gate problems still enforced)", name,
        )
        literal_problems = []
    problems = list(sql_problems) + list(literal_problems)
    if problems:
        return _validation_feedback(name, problems)
    return await call_tool(name, tool_args, None)


def _build_shared_slayer_server(slayer_storage_dir: str) -> MCPServerStdio:
    """One MCPServerStdio per task, shared across the spawn tree.
    Mirrors the recursive adapter, plus a ``process_tool_call`` hook that
    validates each ``edit_model``/``create_model`` write BEFORE it persists and
    feeds any SQL error back to the agent (DEV-1454). The hook runs a client-side
    ``SlayerQueryEngine`` over the SAME storage dir (``YAMLStorage.get_model``
    reads disk uncached, so it sees the subprocess's latest writes)."""
    cfg = slayer_mcp_stdio_config(slayer_storage_dir)
    storage = YAMLStorage(base_dir=slayer_storage_dir)
    engine = SlayerQueryEngine(storage=storage)

    async def _process_tool_call(ctx, call_tool, name, tool_args):
        return await _validate_and_call(
            call_tool, name, tool_args, storage=storage, engine=engine,
        )

    return MCPServerStdio(
        command=cfg["command"], args=cfg["args"], env=cfg["env"],
        max_retries=100, timeout=300,
        process_tool_call=_process_tool_call,
    )


def _otf_work_dir(instance_id: str) -> Path:
    p = (
        Path(tempfile.gettempdir())
        / "bird_interact_slayer_otf"
        / instance_id
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _resolve_otf_task_storage_dir(
    *,
    db_name: str,
    task_data: dict,
    data_path_base: str,
    build_encoder,
) -> tuple[str, list[int]]:
    """DEV-1454: lazily build the durable per-DB reference (setup encode of the
    full KB), then materialise a per-task HARD-8 variant copy of it — entities
    whose ``meta.kb_id`` is in this task's ``deleted_knowledge`` are dropped.
    The reference encodes the FULL KB; per-task masking happens only here."""
    deleted = sorted(extract_deleted_kb_ids(task_data))
    instance_id = task_data["instance_id"]
    reference_root = _paths.slayer_models_otf_root()
    await ensure_db_reference(
        db_name,
        reference_root=reference_root,
        cache_root=_paths.slayer_otf_cache_root(),
        mini_interact_root=Path(data_path_base),
        build_encoder=build_encoder,
    )
    scratch = await build_task_variant_storage(
        canonical_storage_root=reference_root,
        db_name=db_name,
        deleted_kb_ids=set(deleted),
        work_dir=_otf_work_dir(instance_id),
        # Honour the run's --db-path: the reference was built against this root
        # (mini_interact_root=data_path_base above), so the per-task datasource
        # must resolve its portable connection string against the SAME root,
        # not the default sibling mini-interact/ ($BIRD_DB_PATH still wins).
        mini_interact_root=Path(data_path_base),
    )
    return str(scratch), deleted


def _merge_tool_stats(parts: list[dict | None]) -> dict | None:
    """Merge per-agent tool_call_stats. Identical to recursive adapter."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    per_tool_map: dict[str, dict[str, Any]] = {}
    error_samples: list[dict[str, str]] = []
    total_calls = 0
    total_errors = 0
    for p in parts:
        for entry in p.get("per_tool", []):
            name = entry["tool"]
            agg = per_tool_map.setdefault(
                name, {"tool": name, "n_calls": 0, "n_errors": 0},
            )
            agg["n_calls"] += entry.get("n_calls", 0)
            agg["n_errors"] += entry.get("n_errors", 0)
        total_calls += p.get("total_calls", 0)
        total_errors += p.get("total_errors", 0)
        for sample in p.get("error_samples", []):
            if len(error_samples) >= _TOOL_ERROR_SAMPLES_PER_TASK:
                break
            error_samples.append(sample)
    per_tool = sorted(
        per_tool_map.values(),
        key=lambda x: (-x["n_calls"], x["tool"]),
    )
    return {
        "per_tool": per_tool,
        "total_calls": total_calls,
        "total_errors": total_errors,
        "error_samples": error_samples,
    }


class PydanticAIOtfEncodeAgent:
    """SLayer a-interact-only adapter with a recursive clarifier tree
    and a runtime KB → SLayer entity encoder."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        max_depth: int = 3,
        prompt_cache: bool = True,
        slayer_setup: str = "on-the-fly",
    ) -> None:
        from bird_interact_agents.agents.pydantic_ai.agent import (
            _anthropic_cache_settings,
            _build_anthropic_model_with_retries,
        )
        from bird_interact_agents.model_string import (
            build_pydantic_ai_model,
            is_anthropic,
            native_model_id,
        )

        if slayer_setup != "on-the-fly":
            raise ValueError(
                "pydantic_ai_otf_encode requires slayer_setup='on-the-fly'; "
                f"got {slayer_setup!r}"
            )

        self.slayer_storage_root = slayer_storage_root
        self.model_id = model
        self.slayer_setup = slayer_setup
        anthropic_model = (
            _build_anthropic_model_with_retries(native_model_id(model))
            if is_anthropic(model) else None
        )
        self.model = anthropic_model or build_pydantic_ai_model(model)
        self.max_depth = max_depth
        self._model_settings = (
            _anthropic_cache_settings()
            if (prompt_cache and is_anthropic(model)) else None
        )

    async def run_task(
        self,
        task_data: dict,
        data_path_base: str,
        budget: float,
        query_mode: str,
        eval_mode: str = "a-interact",
        user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version: str = "v2",
    ) -> dict:
        if query_mode != "slayer":
            raise ValueError(
                "pydantic_ai_otf_encode supports only --query-mode slayer; "
                f"got {query_mode!r}"
            )
        if eval_mode != "a-interact":
            raise ValueError(
                "pydantic_ai_otf_encode supports only --mode a-interact; "
                f"got {eval_mode!r}"
            )

        db_name = task_data["selected_database"]
        instance_id = task_data["instance_id"]

        from slayer.storage.yaml_storage import YAMLStorage

        # Pre-init for the except path so a failure during the (potentially
        # expensive, failure-prone) pre-run setup — notably the lazy per-DB
        # reference build inside `_resolve_otf_task_storage_dir` — returns a
        # finalized error row instead of raising and aborting the whole batch
        # (CodeRabbit).
        deleted_kb_ids: list[int] = []
        slayer_storage_dir = ""
        shared: SharedTaskState | None = None
        current_record: AgentRecord | None = None
        current_deps: TaskDeps | None = None

        try:
            # Cheap, can't-fail state first, so the except path always has a
            # `shared` + root record to finalize against.
            status = SampleStatus(
                idx=0,
                original_data=task_data,
                remaining_budget=budget,
                total_budget=budget,
            )
            shared = SharedTaskState(
                status=status,
                data_path_base=data_path_base,
                db_name=db_name,
                amb_user_query=task_data["amb_user_query"],
                slayer_storage_dir="",
                user_sim_model=user_sim_model,
                user_sim_prompt_version=user_sim_prompt_version,
            )
            root_record = AgentRecord(
                role="root_clarifier",
                depth=0,
                parent_idx=None,
                instruction=task_data["amb_user_query"],
                started_at=time.monotonic(),
            )
            shared.agent_records.append(root_record)
            root_idx = len(shared.agent_records) - 1
            root_deps = TaskDeps(
                shared=shared, depth=0, max_depth=self.max_depth,
                self_record_idx=root_idx,
            )
            current_record = root_record
            current_deps = root_deps

            # --- pre-run setup (failure here → finalized error row) ---
            load_db_data_if_needed(db_name, data_path_base)
            # Always on-the-fly for this adapter (init enforced). The per-DB
            # reference is built once (lazily) by a setup encoder that shares
            # this agent's model; the per-task copy is a HARD-8 variant of it.
            build_encoder = make_setup_build_encoder(
                model=self.model,
                model_settings=self._model_settings,
                self_model_id=self.model_id,
                build_shared_slayer_server=_build_shared_slayer_server,
            )
            slayer_storage_dir, deleted_kb_ids = (
                await _resolve_otf_task_storage_dir(
                    db_name=db_name,
                    task_data=task_data,
                    data_path_base=data_path_base,
                    build_encoder=build_encoder,
                )
            )
            shared.slayer_storage_dir = slayer_storage_dir
            # Eagerly initialise the YAMLStorage on the shared state so the
            # KB loader (called inside sub_clarifier tools) doesn't have to
            # build it itself.
            shared._slayer_storage = YAMLStorage(base_dir=slayer_storage_dir)

            reserve = _constructor_reserve()
            total_budget = status.remaining_budget
            status.remaining_budget = max(
                0.0, status.remaining_budget - reserve,
            )

            slayer_server = (
                _build_shared_slayer_server(slayer_storage_dir)
                if slayer_storage_dir else None
            )

            async with (slayer_server if slayer_server is not None
                        else _null_async_context()):
                # ----- ROOT PHASE -----
                root_agent = _build_root_clarifier(
                    model=self.model,
                    model_settings=self._model_settings,
                    shared_slayer_server=slayer_server,
                    max_depth=self.max_depth,
                    self_model_id=self.model_id,
                )
                root_prompt = ROOT_CLARIFIER_PROMPT.format(
                    budget=shared.status.remaining_budget,
                    db_name=db_name,
                    user_query=task_data["amb_user_query"],
                )
                root_run = await root_agent.run(
                    user_prompt=task_data["amb_user_query"],
                    instructions=root_prompt,
                    deps=root_deps,
                    usage_limits=UsageLimits(
                        request_limit=MAX_MODEL_TURNS * 2,
                    ),
                )
                _fill_record_from_run(
                    root_record, root_run, root_deps, self.model_id,
                )
                spec = str(root_run.output) or task_data["amb_user_query"]

                shared.status.remaining_budget = min(
                    total_budget,
                    shared.status.remaining_budget + reserve,
                )
                if shared.status.remaining_budget > ACTION_COSTS["submit_query"]:
                    shared.status.force_submit = False

                # ----- PROJECTION-RESOLVER PHASE -----
                resolver_record = AgentRecord(
                    role="projection_resolver",
                    depth=0,
                    parent_idx=None,
                    instruction="resolve projection",
                    started_at=time.monotonic(),
                )
                shared.agent_records.append(resolver_record)
                resolver_idx = len(shared.agent_records) - 1
                resolver_deps = TaskDeps(
                    shared=shared, depth=0, max_depth=0,
                    self_record_idx=resolver_idx,
                )
                current_record = resolver_record
                current_deps = resolver_deps
                resolver_agent = _build_projection_resolver(
                    model=self.model,
                    model_settings=self._model_settings,
                    self_model_id=self.model_id,
                )
                resolver_prompt = PROJECTION_RESOLVER_PROMPT.format(
                    amb_user_query=task_data["amb_user_query"],
                    spec=spec,
                    budget=shared.status.remaining_budget,
                    db_name=db_name,
                ) + _PROJECTION_SUBMIT_SUFFIX
                resolver_result = await _run_projection_resolver(
                    resolver_agent=resolver_agent,
                    instructions=resolver_prompt,
                    user_prompt=task_data["amb_user_query"],
                    deps=resolver_deps,
                    model_id=self.model_id,
                )
                resolver_record.output = repr(resolver_result.projection)
                resolver_record.user_sim_transcript = list(
                    resolver_deps.user_sim_transcript,
                )
                resolver_record.usage = resolver_deps.usage
                resolver_record.messages = resolver_result.messages
                resolver_record.tool_call_stats = resolver_result.tool_call_stats
                resolver_record.n_agent_turns = resolver_result.n_agent_turns
                resolver_record.ended_at = time.monotonic()

                if resolver_result.status == "empty_after_guard":
                    shared.submitter_result = {
                        "phase1_passed": False,
                        "phase2_passed": False,
                        "total_reward": 0.0,
                        "finished": False,
                        "submitted_sql": None,
                        "submitted_query": None,
                        "submission_status": "never_submitted",
                    }
                    return _finalize(
                        shared=shared,
                        instance_id=instance_id,
                        db_name=db_name,
                        deleted_kb_ids=deleted_kb_ids,
                        slayer_storage_dir=slayer_storage_dir,
                        final_output_excerpt="",
                        error=None,
                        projection_resolver_status="empty_after_guard",
                    )

                # ----- CONSTRUCTOR PHASE -----
                constructor_record = AgentRecord(
                    role="query_constructor",
                    depth=0,
                    parent_idx=None,
                    instruction="assemble + submit",
                    started_at=time.monotonic(),
                )
                shared.agent_records.append(constructor_record)
                constructor_idx = len(shared.agent_records) - 1
                constructor_deps = TaskDeps(
                    shared=shared, depth=0, max_depth=0,
                    self_record_idx=constructor_idx,
                )
                current_record = constructor_record
                current_deps = constructor_deps
                confirmed_projection_tuple = tuple(resolver_result.projection)
                constructor_agent = _build_query_constructor(
                    model=self.model,
                    model_settings=self._model_settings,
                    shared_slayer_server=slayer_server,
                    confirmed_projection=confirmed_projection_tuple,
                    self_model_id=self.model_id,
                )
                confirmed_projection_block = "\n".join(
                    f"  {i + 1}. {name}"
                    for i, name in enumerate(confirmed_projection_tuple)
                )
                constructor_prompt = QUERY_CONSTRUCTOR_PROMPT.format(
                    amb_user_query=task_data["amb_user_query"],
                    spec=spec,
                    confirmed_projection=confirmed_projection_block,
                    budget=shared.status.remaining_budget,
                    db_name=db_name,
                )
                constructor_run = await constructor_agent.run(
                    user_prompt=task_data["amb_user_query"],
                    instructions=constructor_prompt,
                    deps=constructor_deps,
                    usage_limits=UsageLimits(
                        request_limit=MAX_MODEL_TURNS * 2,
                    ),
                )
                _fill_record_from_run(
                    constructor_record, constructor_run, constructor_deps,
                    self.model_id,
                )
                constructor_output = str(constructor_run.output)
        except Exception as e:
            logger.exception(
                "OTF-encode agent error on %s: %s", instance_id, e,
            )
            subs = getattr(e, "exceptions", None)
            if subs:
                for i, sub in enumerate(subs):
                    logger.error("  sub-exception %d: %r", i, sub)
            if current_record is not None and current_record.error is None:
                current_record.error = f"{type(e).__name__}: {e}"
                if current_deps is not None:
                    current_record.usage = current_deps.usage
                    current_record.user_sim_transcript = list(
                        current_deps.user_sim_transcript,
                    )
                current_record.ended_at = time.monotonic()
            if shared is None:
                # Setup failed before `shared` was built (should be unreachable
                # — shared is created first inside the try — but keeps the batch
                # alive with an error row instead of propagating).
                return _minimal_error_row(
                    instance_id=instance_id, db_name=db_name,
                    deleted_kb_ids=deleted_kb_ids,
                    slayer_storage_dir=slayer_storage_dir, error=str(e),
                )
            return _finalize(
                shared=shared,
                instance_id=instance_id,
                db_name=db_name,
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
                final_output_excerpt="",
                error=str(e),
            )

        return _finalize(
            shared=shared,
            instance_id=instance_id,
            db_name=db_name,
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
            final_output_excerpt=constructor_output[:500],
            error=None,
        )


class _null_async_context:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


async def _run_projection_resolver(
    *,
    resolver_agent: Any,
    instructions: str,
    user_prompt: str,
    deps: Any,
    model_id: str,
) -> _ResolverResult:
    """Stage 2 with empty-list guard — copy of the recursive adapter's
    helper bound to this module's TaskDeps."""
    # DEV-1454: the resolver delivers its columns via submit_projection into
    # per-run deps (not structured output), so it can reason in text. Reset the
    # slot before each attempt; read it after. Wrap each run so a valid
    # submission survives a post-submit failure (e.g. cap hit chasing the final
    # text) — mirrors the encoders' "submission survives an exception".
    deps.projection_submission = None
    first_run = None
    try:
        first_run = await resolver_agent.run(
            user_prompt=user_prompt,
            instructions=instructions,
            deps=deps,
            usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
        )
        _fold_run_usage_into_deps(first_run, deps, model_id)
    except Exception:  # noqa: BLE001 — keep any submission captured before the raise
        logger.exception("projection resolver first run raised")
    projection = list(deps.projection_submission or [])
    runs = [r for r in (first_run,) if r is not None]
    if projection:
        return _aggregate_runs(runs, "confirmed", projection)

    deps.projection_submission = None
    recovery_run = None
    try:
        recovery_run = await resolver_agent.run(
            user_prompt=(
                "Your previous output was an empty list. Propose at least one "
                "output column you derive from the user's question and the "
                "specification, then ask the user to confirm or refine."
            ),
            instructions=instructions,
            deps=deps,
            message_history=(
                first_run.all_messages() if first_run is not None else None
            ),
            usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
        )
        _fold_run_usage_into_deps(recovery_run, deps, model_id)
    except Exception:  # noqa: BLE001 — keep any submission captured before the raise
        logger.exception("projection resolver recovery run raised")
    projection = list(deps.projection_submission or [])
    runs = [r for r in (first_run, recovery_run) if r is not None]
    if projection:
        return _aggregate_runs(runs, "confirmed", projection)
    return _aggregate_runs(runs, "empty_after_guard", [])


def _fill_record_from_run(
    record: AgentRecord, run: Any, deps: TaskDeps, self_model_id: str,
) -> None:
    """Identical to the recursive adapter's same-named helper."""
    run_usage = run.usage()
    deps.usage.add_call(
        scope="agent",
        model=self_model_id,
        prompt=getattr(run_usage, "input_tokens", 0) or 0,
        completion=getattr(run_usage, "output_tokens", 0) or 0,
        cache_read=getattr(run_usage, "cache_read_tokens", 0) or 0,
        cache_write=getattr(run_usage, "cache_write_tokens", 0) or 0,
    )
    record.output = str(run.output)
    record.user_sim_transcript = list(deps.user_sim_transcript)
    record.usage = deps.usage
    record.messages = _serialize_messages(run)
    record.tool_call_stats = _extract_tool_stats(run)
    record.n_agent_turns = _count_turns(run)
    record.ended_at = time.monotonic()


def _emit_task_sessions(shared: SharedTaskState, instance_id: str) -> None:
    """Write one session file per spawned task agent into the task's /tmp work
    dir (``_otf_work_dir(instance_id)/sessions/``) so any sub-agent's session is
    trivially isolatable — open one file — instead of digging through
    eval.json's nested ``trajectory.agents``. Mirrors the setup encoder's
    per-kb session logging. Best-effort: never raises."""
    try:
        sessions_dir = _otf_work_dir(instance_id) / "sessions"
        rows: list[dict] = []
        for i, rec in enumerate(shared.agent_records):
            sid = f"{i:02d}__{rec.role}" + (
                f"__kb{rec.kb_id}" if rec.kb_id is not None else ""
            )
            duration = (
                rec.ended_at - rec.started_at
                if rec.ended_at and rec.started_at else None
            )
            rows.append(write_session(
                sessions_dir, sid,
                messages=rec.messages,
                tool_call_stats=rec.tool_call_stats,
                n_turns=rec.n_agent_turns,
                role=rec.role,
                meta={
                    "kb_id": rec.kb_id, "focus": rec.focus,
                    "depth": rec.depth, "parent_idx": rec.parent_idx,
                },
                status=("error" if rec.error else "ok"),
                output=rec.output, error=rec.error,
                usage=(rec.usage.model_dump() if rec.usage else None),
                duration_s=duration,
            ))
        write_index(sessions_dir, rows)
    except Exception:  # noqa: BLE001 — session logging must not break the task
        logger.exception("failed to emit task sessions for %s", instance_id)


def _finalize(
    *,
    shared: SharedTaskState,
    instance_id: str,
    db_name: str,
    deleted_kb_ids: list[int],
    slayer_storage_dir: str,
    final_output_excerpt: str,
    error: str | None,
    projection_resolver_status: str | None = None,
) -> dict:
    """Build the result row from the shared state. Adds the
    `kb_encoded` field on top of the recursive adapter's row shape."""
    _emit_task_sessions(shared, instance_id)
    submitter = shared.submitter_result or {}

    total_usage = TokenUsage()
    n_turns_total: int | None = None
    for rec in shared.agent_records:
        total_usage.merge(rec.usage)
        if rec.n_agent_turns is not None:
            n_turns_total = (n_turns_total or 0) + rec.n_agent_turns
    tool_stats = _merge_tool_stats(
        [r.tool_call_stats for r in shared.agent_records],
    )

    trajectory = {
        "final_output_excerpt": final_output_excerpt,
        "agents": [r.model_dump() for r in shared.agent_records],
    }

    row = {
        "task_id": instance_id,
        "instance_id": instance_id,
        "database": db_name,
        "phase1_passed": submitter.get("phase1_passed", False),
        "phase2_passed": submitter.get("phase2_passed", False),
        "total_reward": submitter.get("total_reward", 0.0),
        "submitted_sql": submitter.get("submitted_sql"),
        "submitted_query": submitter.get("submitted_query"),
        "trajectory": trajectory,
        "error": error,
        "usage": total_usage.model_dump(),
        "submission_status": submitter.get(
            "submission_status", "never_submitted",
        ),
        "phase1_observation": submitter.get("phase1_observation"),
        "phase2_observation": submitter.get("phase2_observation"),
        "predicted_result_json": submitter.get("predicted_result_json"),
        "gold_result_json": submitter.get("gold_result_json"),
        "n_agent_turns": n_turns_total,
        "tool_call_stats": tool_stats,
        # Dual-eval fields — parity with the recursive adapter (merged from
        # master); populated only under --use-audited-gold-sql, NULL otherwise.
        "phase1_passed_audited": submitter.get("phase1_passed_audited"),
        "phase1_passed_original": submitter.get("phase1_passed_original"),
        "phase1_observation_audited": submitter.get("phase1_observation_audited"),
        "phase1_observation_original": submitter.get("phase1_observation_original"),
        # DEV-1454-specific: per-task KB encode registry.
        "kb_encoded": [r.model_dump() for r in shared.kb_encoded],
    }
    if projection_resolver_status is not None:
        row["projection_resolver_status"] = projection_resolver_status
    return finalize_result_row(
        row,
        deleted_kb_ids=deleted_kb_ids,
        slayer_storage_dir=slayer_storage_dir,
    )


def _minimal_error_row(
    *,
    instance_id: str,
    db_name: str,
    deleted_kb_ids: list[int],
    slayer_storage_dir: str,
    error: str,
) -> dict:
    """Finalized error row for a setup failure that happened before a
    ``SharedTaskState`` existed. Mirrors ``_finalize``'s key set (parity) with
    empty trajectory/usage so batch evaluation records the error instead of
    aborting."""
    row = {
        "task_id": instance_id,
        "instance_id": instance_id,
        "database": db_name,
        "phase1_passed": False,
        "phase2_passed": False,
        "total_reward": 0.0,
        "submitted_sql": None,
        "submitted_query": None,
        "trajectory": {"final_output_excerpt": "", "agents": []},
        "error": error,
        "usage": TokenUsage().model_dump(),
        "submission_status": "never_submitted",
        "phase1_observation": None,
        "phase2_observation": None,
        "predicted_result_json": None,
        "gold_result_json": None,
        "n_agent_turns": None,
        "tool_call_stats": None,
        "phase1_passed_audited": None,
        "phase1_passed_original": None,
        "phase1_observation_audited": None,
        "phase1_observation_original": None,
        "kb_encoded": [],
    }
    return finalize_result_row(
        row,
        deleted_kb_ids=deleted_kb_ids,
        slayer_storage_dir=slayer_storage_dir,
    )
