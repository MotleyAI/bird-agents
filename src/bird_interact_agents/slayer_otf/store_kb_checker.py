"""DEV-1671: deterministic checker for saved edited-model stores.

A NUDGE tool. Given a saved store's SLayer models, the OTF-cache baseline, the KB
rows, the task's advisory ``external_knowledge`` anchors, and the final (winning)
``SlayerQuery``, it reports whether the store is in a PASSING STATE:

  1. every surviving agent-added entity is USED by the winning query,
  2. relevant KB items are encoded as entities,
  3. entities refer to each other (a clean DAG — KB defs referenced, not inlined).

It is PURELY DETERMINISTIC and makes NO concept-vs-answer judgment — it only flags
agent-created entities that lack a ``[kb=N]`` provenance tag (``NON_KB_ENTITY``);
the concept-vs-answer call + ``[concept]`` tagging + demotion is an EXTERNAL step.

All lineage is resolved through SLayer's OWN helpers — ``core.formula`` for
filter/measure references, and ``engine.column_dependency`` internals for
column-SQL references (composed with the trivial-base filter dropped so
scaffolding referenced by a used derived column counts as used). No hand-rolled
SQL/formula parser (memory: ``feedback_reuse_slayer_not_reinvent``).
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

import sqlglot
from pydantic import BaseModel
from sqlglot import exp

from slayer.core.formula import parse_filter as _formula_parse_filter
from slayer.core.formula import parse_formula as _parse_formula
from slayer.core.models import Column, ModelMeasure, SlayerModel
from slayer.engine.column_dependency import (
    _DEPENDENCY_DIALECT,
    _is_trivial_base,
    _resolve_target_for_ref,
    _root_scope_column_ids,
)

_KB_RE = re.compile(r"\[kb=(\d+)\]")
_CONCEPT_RE = re.compile(r"\[concept\]")

FindingCategory = Literal[
    "UNUSED_AGENT_ENTITY",
    "INLINE_QUERY_WORK",
    "NON_KB_ENTITY",
    "INLINED_KB_DEF",
    "DEFERRED_RELEVANT_KB",
    "EXPECTED_KB_NOT_MATERIALIZED",
    "ORPHAN_KB_ENTITY",
]

# Advisory categories don't count against a store being "clean" — they're soft
# hints, not actionable defects (external_knowledge is unreliable, orphan tags are
# cosmetic). Everything is a recommendation; nothing is a hard error.
_ADVISORY = {"DEFERRED_RELEVANT_KB", "EXPECTED_KB_NOT_MATERIALIZED", "ORPHAN_KB_ENTITY"}


class Finding(BaseModel):
    category: FindingCategory
    level: Literal["error", "flag"]
    model: Optional[str] = None
    entity: Optional[str] = None
    kb_id: Optional[int] = None
    detail: str


class StoreCheckReport(BaseModel):
    instance_id: Optional[str] = None
    db: Optional[str] = None
    benchmark: Optional[str] = None
    reward: Optional[float] = None
    baseline_available: bool = True
    findings: list[Finding] = []
    ok: bool = True

    @classmethod
    def from_findings(
        cls,
        findings: list[Finding],
        *,
        instance_id: str | None = None,
        db: str | None = None,
        benchmark: str | None = None,
        reward: float | None = None,
        baseline_available: bool = True,
    ) -> "StoreCheckReport":
        # "clean" = baseline available AND no substantive (non-advisory) recommendations.
        ok = baseline_available and not any(
            f.category not in _ADVISORY for f in findings
        )
        return cls(
            instance_id=instance_id,
            db=db,
            benchmark=benchmark,
            reward=reward,
            baseline_available=baseline_available,
            findings=list(findings),
            ok=ok,
        )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _kb_tag(text: str | None) -> int | None:
    if not text:
        return None
    m = _KB_RE.search(text)
    return int(m.group(1)) if m else None


def _has_concept(text: str | None) -> bool:
    return bool(text) and bool(_CONCEPT_RE.search(text))


def _children_ids(row: dict) -> list[int]:
    """children_knowledge normalised: the ``-1`` "no children" sentinel (int or
    inside a list) collapses to no children."""
    ck = row.get("children_knowledge")
    if ck is None:
        return []
    if isinstance(ck, int):
        return [] if ck < 0 else [ck]
    return [c for c in ck if isinstance(c, int) and c >= 0]


def relevant_kb_closure(kb_rows: list[dict], anchors: list[int]) -> set[int]:
    """Transitive closure of ``anchors`` over ``children_knowledge`` (normalising
    the ``-1`` sentinel). Pure integer-graph walk over our own KB data."""
    children = {int(r["id"]): _children_ids(r) for r in kb_rows if "id" in r}
    seen: set[int] = set()
    stack = [a for a in anchors]
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        stack.extend(children.get(k, []))
    return seen


def _measures_by_name(model: SlayerModel) -> dict[str, ModelMeasure]:
    return {m.name: m for m in (model.measures or [])}


def _all_column_refs(
    column: Column, host: SlayerModel, reachable: dict[str, SlayerModel]
) -> list[tuple[str, str]]:
    """Every (model, column) the column's SQL references — INCLUDING trivial-base
    passthroughs (unlike ``_column_dependencies``, which drops them). Reuses
    SLayer's own alias/scope resolution helpers, only omitting the trivial-base
    filter, so scaffolding referenced by a used derived column counts as used."""
    if column.sql is None:
        return []
    try:
        parsed = sqlglot.parse_one(column.sql, dialect=_DEPENDENCY_DIALECT)
    except Exception:
        return []
    root_ids = _root_scope_column_ids(parsed=parsed)
    refs: list[tuple[str, str]] = []
    for node in parsed.find_all(exp.Column):
        if id(node) not in root_ids:
            continue
        if node.args.get("db") or node.args.get("catalog"):
            continue
        table_id = node.args.get("table")
        alias = table_id.name if table_id is not None else None
        target = _resolve_target_for_ref(table_alias=alias, host=host, reachable=reachable)
        if target is None:
            continue
        tcol = target.get_column(node.name)
        if tcol is None:
            continue
        refs.append((target.name, tcol.name))
    return refs


_JOIN_PAIR_SRC_ATTRS = ("source_column", "source_field", "from_column", "left", "source", "column")
_JOIN_PAIR_TGT_ATTRS = ("target_column", "target_field", "to_column", "right", "target", "column")


def _bare_column(v: Any) -> str | None:
    """Strip a ``model.column`` / ``alias.column`` qualifier down to the bare column
    name (SLayer supports dotted join_pairs, e.g. ``["a.local_col", "b.remote_col"]``);
    join keys are matched against unqualified ``Column.name``s."""
    if not isinstance(v, str):
        return None
    return v.rsplit(".", 1)[-1]


def _join_pair_names(pair: Any) -> tuple[str | None, str | None]:
    """Source (near) and target (far) column names of one ``join_pairs`` entry —
    list-form ``[src, tgt]`` (Codex #4) or an object exposing source/target-ish
    attributes. Qualified names are reduced to the bare column. Either side may be
    ``None`` when not resolvable."""
    if isinstance(pair, (list, tuple)):
        src = pair[0] if pair and isinstance(pair[0], str) else None
        tgt = pair[1] if len(pair) >= 2 and isinstance(pair[1], str) else None
        return _bare_column(src), _bare_column(tgt)
    src = next((getattr(pair, a, None) for a in _JOIN_PAIR_SRC_ATTRS
                if isinstance(getattr(pair, a, None), str)), None)
    tgt = next((getattr(pair, a, None) for a in _JOIN_PAIR_TGT_ATTRS
                if isinstance(getattr(pair, a, None), str)), None)
    return _bare_column(src), _bare_column(tgt)


def _join_key_columns(model: SlayerModel) -> set[str]:
    """Column names that are JOIN keys of ``model`` — from declared ``joins``
    (``join_pairs``) and, for a ``sql``/``backing_query_sql``-backed model, from
    the JOIN ... ON conditions (parsed with SLayer's dependency dialect). These are
    structural scaffolding and are exempt from the unused check even when the
    winning query doesn't project them (the join lives in the model, not the query)."""
    names: set[str] = set()
    for j in model.joins or []:
        for pair in getattr(j, "join_pairs", []) or []:
            src, _ = _join_pair_names(pair)
            if src is not None:
                names.add(src)
    for sql in (getattr(model, "sql", None), getattr(model, "backing_query_sql", None)):
        if not sql:
            continue
        try:
            parsed = sqlglot.parse_one(sql, dialect=_DEPENDENCY_DIALECT)
        except Exception:
            continue
        for jn in parsed.find_all(exp.Join):
            for c in jn.find_all(exp.Column):
                names.add(c.name)
    existing = {c.name for c in (model.columns or [])}
    return names & existing


def _measure_ref_names(parsed: Any) -> list[str]:
    """Column/measure names referenced by a parsed formula/aggregation, recursing into
    window/scalar TransformFields (e.g. ``rank(snap_ts:max, partition_by=…)`` wraps its
    aggregation in ``.inner``) so a column used only inside a transform still counts."""
    names: list[str] = []
    mn = getattr(parsed, "measure_name", None)
    if isinstance(mn, str):
        names.append(mn)
    agg_refs = getattr(parsed, "agg_refs", None)
    if isinstance(agg_refs, dict):
        for ref in agg_refs.values():
            rn = getattr(ref, "measure_name", None)
            if isinstance(rn, str):
                names.append(rn)
    extra = getattr(parsed, "measure_names", None)
    if extra:
        names.extend([n for n in extra if isinstance(n, str)])
    inner = getattr(parsed, "inner", None)  # TransformField wraps its operand
    if inner is not None and inner is not parsed:
        names.extend(_measure_ref_names(inner))
    subs = getattr(parsed, "sub_transforms", None)
    if isinstance(subs, dict):
        for s in subs.values():
            names.extend(_measure_ref_names(s))
    return [n for n in names if n and n != "*"]


# ---------------------------------------------------------------------------
# query entity closure
# ---------------------------------------------------------------------------


def _dim_name(d: Any) -> str | None:
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        return d.get("name")
    return None


def _stage_reference_names(stage: dict, model: SlayerModel) -> set[str]:
    """Raw entity names a single stage references, resolved with SLayer's own
    parsers (dimensions/order = bare names; filters = ``core.formula.parse_filter``;
    measures = ``parse_formula`` / stored-measure-name match)."""
    names: set[str] = set()
    for d in stage.get("dimensions", []) or []:
        n = _dim_name(d)
        if n:
            names.add(n)
    for o in stage.get("order", []) or []:
        col = o.get("column") if isinstance(o, dict) else (o if isinstance(o, str) else None)
        if col:
            names.add(col.split(":", 1)[0])  # strip an agg suffix, e.g. "risk_score:avg" (Codex #6)
    for td in stage.get("time_dimensions", []) or []:  # Codex #3
        if isinstance(td, dict):
            n = td.get("dimension") or td.get("name")
            if n:
                names.add(n)
        elif isinstance(td, str):
            names.add(td)
    for f in stage.get("filters", []) or []:
        if not isinstance(f, str):
            continue
        try:
            names.update(_formula_parse_filter(f).columns or [])
        except Exception:
            continue
    measures_by_name = _measures_by_name(model)
    named = {n: m.formula for n, m in measures_by_name.items()}  # Codex #5: bare measure refs
    for meas in stage.get("measures", []) or []:
        ref = meas.get("formula") or meas.get("name") if isinstance(meas, dict) else meas
        if not isinstance(ref, str):
            continue
        if ref in measures_by_name:  # a stored measure referenced by name
            names.add(ref)
            continue
        try:
            names.update(_measure_ref_names(_parse_formula(ref, named_measures=named)))
        except Exception:
            continue
    return names


_CONST_SQL_RE = re.compile(r"^\s*('([^']|'')*'|[+-]?[\d.]+|true|false|null)\s*$", re.I)


def _inline_model_does_work(sm: dict) -> bool:
    """True if an inline ModelExtension source_model does real computation worth encoding —
    it has measures, or a column whose sql is not a pure constant literal. A constant-only
    inline column (e.g. sql="'sprint'") is not encodable work, so it is not flagged."""
    if sm.get("measures"):
        return True
    for key in ("columns", "extra_columns"):  # source_name/columns AND base_model/extra_columns forms
        for c in sm.get(key) or []:
            if isinstance(c, dict):
                s = c.get("sql")
                if s is not None and not _CONST_SQL_RE.match(str(s)):
                    return True
    return False


_JSON_RE = re.compile(r"jsonb?_extract_path_text|->>|#>>", re.I)
_CASE_RE = re.compile(r"\bcase\b", re.I)
_ARITH_RE = re.compile(r"[+\-*/]")


def _inline_query_work(winning_query: dict | list) -> list["Finding"]:
    """Recommend (soft) encoding when the winning query does work INLINE that should
    be a stored column/concept/model. STRONG signals only (Egor): an inline model /
    column defined inside the query, and JSON extraction / CASE / multi-column
    arithmetic in a filter. It cannot statically match an inline expr to a specific
    KB formula, so it's a recommendation to CONSIDER encoding, not an assertion."""
    out: list[Finding] = []
    stages = winning_query if isinstance(winning_query, list) else [winning_query]
    listed = isinstance(winning_query, list)
    for si, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        loc = f"stage {si}" if listed else "query"
        sm = stage.get("source_model")
        if isinstance(sm, dict) and _inline_model_does_work(sm):
            out.append(Finding(
                category="INLINE_QUERY_WORK", level="flag",
                detail=f"{loc}: source_model is an inline model defined in the query — "
                f"consider encoding it as a stored model and referencing it."))
        for d in stage.get("dimensions", []) or []:
            if isinstance(d, dict) and d.get("sql"):
                out.append(Finding(
                    category="INLINE_QUERY_WORK", level="flag", entity=d.get("name"),
                    detail=f"{loc}: dimension '{d.get('name')}' is defined inline (sql=…) — "
                    f"consider encoding it as a stored column/concept and referencing it."))
        for f in stage.get("filters", []) or []:
            if not isinstance(f, str):
                continue
            sig = None
            if _JSON_RE.search(f):
                sig = "JSON extraction"
            elif _CASE_RE.search(f):
                sig = "a CASE expression"
            else:
                try:
                    cols = _formula_parse_filter(f).columns or []
                except Exception:
                    cols = []
                if len(set(cols)) >= 2 and _ARITH_RE.search(f):
                    sig = "arithmetic over multiple columns"
            if sig:
                out.append(Finding(
                    category="INLINE_QUERY_WORK", level="flag",
                    detail=f"{loc}: filter does {sig} inline ({f!r}) — if this computes a KB "
                    f"item or reusable concept, consider encoding it as a column/concept and "
                    f"referencing it."))
    return out


def _reachable_models(
    root: SlayerModel, models_by_name: dict[str, SlayerModel]
) -> dict[str, SlayerModel]:
    """Root + every model transitively reachable through declared joins (BFS over
    ``model.joins[*].target_model``). Used to resolve cross-model query references
    (a filter/dimension may name a column on a JOINED model, not the root)."""
    out: dict[str, SlayerModel] = {root.name: root}
    stack = [root]
    while stack:
        m = stack.pop()
        for j in m.joins or []:
            tgt = getattr(j, "target_model", None)
            if isinstance(tgt, str) and tgt not in out and tgt in models_by_name:
                out[tgt] = models_by_name[tgt]
                stack.append(models_by_name[tgt])
    return out


def _resolve_ref(
    name: str, root: SlayerModel, reachable: dict[str, SlayerModel]
) -> tuple[str, str] | None:
    """Resolve a query reference NAME to a (model, entity) that exists. Handles a
    qualified ``Model.col`` / join-path ``a__b.col`` / multi-hop dotted join-path
    ``a.b.col`` (target = the deepest joined model, i.e. the last path segment) and a
    bare ``col`` (root first, then any reachable/joined model — matching SLayer's own
    bare-name resolution). Returns None if it resolves to nothing in the store.

    SLayer expresses a multi-hop join path with DOTS (e.g.
    ``RiskManagement.DataFlow.transfer_route`` from a ``Compliance`` root reachable via
    ``Compliance -> RiskManagement -> DataFlow``); the intermediate hops are structural
    and the entity lives on the LAST model segment. So the target is the final path
    segment regardless of whether the path uses ``.`` or ``__`` separators."""
    def _has(m: SlayerModel, ent: str) -> bool:
        return m.get_column(ent) is not None or ent in _measures_by_name(m)

    if "." in name:
        prefix, col = name.rsplit(".", 1)
        # The deepest model in the path is the last segment under either separator: a
        # dotted multi-hop path (``a.b.col`` → ``a.b`` → ``b``) or the ``a__b`` form.
        target = prefix.split("__")[-1].split(".")[-1]
        m = reachable.get(target)
        if m is not None and _has(m, col):
            return (m.name, col)
        return None
    if _has(root, name):
        return (root.name, name)
    for m in reachable.values():
        if m.name != root.name and _has(m, name):
            return (m.name, name)
    return None


def _query_closure(
    winning_query: dict | list, models_by_name: dict[str, SlayerModel]
) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (used_model_names, used_entities) where used_entities is the
    transitive set of (model, column|measure) the winning query references.
    Handles nested multi-stage queries whose later ``source_model`` is a prior
    stage NAME by carrying each stage's underlying store-model root."""
    stages = winning_query if isinstance(winning_query, list) else [winning_query]
    stage_root: dict[str, str] = {}
    used_models: set[str] = set()
    seed: set[tuple[str, str]] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        src = stage.get("source_model")
        # source_model may be a bare model name (str), a prior-stage NAME (str), or an
        # inline ModelExtension `{"source_name": <base model>, "columns": [...]}` (dict).
        # For the extension, the BASE model (source_name) is still a real store model the
        # query filters/dimensions can reference (Codex #2) — resolve it as the root.
        if isinstance(src, str):
            root = stage_root.get(src, src)  # prior-stage name → its root; else a store model
        elif isinstance(src, dict) and isinstance(src.get("source_name"), str):
            base = src["source_name"]
            root = stage_root.get(base, base)
        else:
            root = None
        if stage.get("name") and root is not None:
            stage_root[stage["name"]] = root
        model = models_by_name.get(root) if root is not None else None
        if model is None:
            continue
        used_models.add(root)
        reachable = _reachable_models(model, models_by_name)
        for name in _stage_reference_names(stage, model):
            resolved = _resolve_ref(name, model, reachable)
            if resolved is not None:
                seed.add(resolved)
                used_models.add(resolved[0])  # a joined model referenced by the query is used
    return used_models, _expand_closure(seed, models_by_name)


def _source_query_refs(model: SlayerModel) -> list[tuple[str, str]]:
    """The (model, column/measure) refs a ``source_queries``-backed (nested-query) model
    consumes on OTHER models. These are invisible to ``Column.sql`` traversal — a
    grain-bridge model projects its output from a nested query, so its inputs (often the
    base KB definitions) are load-bearing when the bridge's output is used, but the closure
    would otherwise miss them and flag them UNUSED."""
    refs: list[tuple[str, str]] = []
    for sq in getattr(model, "source_queries", None) or []:
        get = sq.get if isinstance(sq, dict) else (lambda k, _o=sq: getattr(_o, k, None))
        src = get("source_model")
        for d in get("dimensions") or []:
            dget = d.get if isinstance(d, dict) else (lambda k, _o=d: getattr(_o, k, None))
            dm, dn = dget("model") or src, dget("name")
            if isinstance(dm, str) and isinstance(dn, str):
                refs.append((dm, dn))
        for meas in get("measures") or []:
            mget = meas.get if isinstance(meas, dict) else (lambda k, _o=meas: getattr(_o, k, None))
            mm, f = mget("model") or src, mget("formula")
            if not isinstance(mm, str) or not isinstance(f, str):
                continue
            names: list[str] = []
            try:
                names = _measure_ref_names(_parse_formula(f))
            except Exception:
                names = []
            if not names and re.match(r"^\w+$", f.strip()):
                names = [f.strip()]  # a bare column/measure ref, e.g. "pfis"
            refs.extend((mm, rn) for rn in names)
    return refs


def _expand_closure(
    seed: set[tuple[str, str]], models_by_name: dict[str, SlayerModel]
) -> set[tuple[str, str]]:
    used: set[tuple[str, str]] = set()
    sq_expanded: set[str] = set()
    stack = list(seed)
    while stack:
        key = stack.pop()
        if key in used:
            continue
        used.add(key)
        model_name, name = key
        model = models_by_name.get(model_name)
        if model is None:
            continue
        # A source_queries-backed model that is USED pulls in its nested-query inputs
        # (load-bearing base defs the Column.sql walk cannot see).
        if getattr(model, "source_queries", None) and model_name not in sq_expanded:
            sq_expanded.add(model_name)
            stack.extend(_source_query_refs(model))
        measures = _measures_by_name(model)
        if name in measures:
            named = {n: m.formula for n, m in measures.items()}  # Codex #5: measure-of-measure
            try:
                for rn in _measure_ref_names(
                    _parse_formula(measures[name].formula, named_measures=named)
                ):
                    if model.get_column(rn) is not None or rn in measures:
                        stack.append((model_name, rn))
            except Exception:
                pass
            continue
        col = model.get_column(name)
        if col is not None:
            for dep in _all_column_refs(col, model, models_by_name):
                stack.append(dep)
    return used


# ---------------------------------------------------------------------------
# agent-added diff
# ---------------------------------------------------------------------------


def _agent_added(
    store_models: list[SlayerModel], baseline_models: list[SlayerModel]
) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    """Return (new_model_names, agent_columns, agent_measures) — entities present
    in the store but not the baseline, diffed PER HOST MODEL."""
    base_by_name = {m.name: m for m in baseline_models}
    new_models: set[str] = set()
    cols: set[tuple[str, str]] = set()
    measures: set[tuple[str, str]] = set()
    for m in store_models:
        base = base_by_name.get(m.name)
        if base is None:
            new_models.add(m.name)
            for c in m.columns or []:
                cols.add((m.name, c.name))
            for mm in m.measures or []:
                measures.add((m.name, mm.name))
            continue
        base_cols = {c.name for c in base.columns or []}
        base_meas = {mm.name for mm in base.measures or []}
        for c in m.columns or []:
            if c.name not in base_cols:
                cols.add((m.name, c.name))
        for mm in m.measures or []:
            if mm.name not in base_meas:
                measures.add((m.name, mm.name))
    return new_models, cols, measures


# ---------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------


async def check_models(
    *,
    store_models: list[SlayerModel],
    baseline_models: list[SlayerModel] | None,
    kb_rows: list[dict],
    relevant_kb_ids: list[int],
    winning_query: dict | list,
    memories: list[dict] | None = None,
) -> list[Finding]:
    """Deterministic passing-state check. Returns findings (empty list when
    ``baseline_models`` is None — the caller marks the report baseline-unavailable
    and skips, rather than passing)."""
    if baseline_models is None:
        return []

    store_by_name = {m.name: m for m in store_models}
    col_by_key: dict[tuple[str, str], Column] = {}
    for m in store_models:
        for c in m.columns or []:
            col_by_key[(m.name, c.name)] = c

    new_models, agent_cols, agent_measures = _agent_added(store_models, baseline_models)
    used_models, used_entities = _query_closure(winning_query, store_by_name)
    relevant = relevant_kb_closure(kb_rows, relevant_kb_ids)
    join_keys_by_model = {m.name: _join_key_columns(m) for m in store_models}
    # Far-side join keys: a declared join ``A.joins[*] -> target_model B`` on
    # ``join_pairs [[a_col, b_col]]`` makes ``B.b_col`` a structural join key of B
    # even though the join is declared on A (and B.joins is empty). Exempt it on B
    # too — a source_queries bridge carries the fac_id/fac_key join column that the
    # parent joins on but no column SQL references, so it would otherwise read as
    # UNUSED scratch.
    for m in store_models:
        for j in m.joins or []:
            tgt = getattr(j, "target_model", None)
            tm = store_by_name.get(tgt) if isinstance(tgt, str) else None
            if tm is None:
                continue
            tcols = {c.name for c in tm.columns or []}
            for pair in getattr(j, "join_pairs", []) or []:
                _, far = _join_pair_names(pair)
                if far and far in tcols:
                    join_keys_by_model.setdefault(tgt, set()).add(far)

    # KBs carried by USED entities → their children_knowledge closure defines the
    # "support" KBs that a used entity legitimately depends on (for the inlined rule).
    used_kb_ids: set[int] = set()
    for (mname, ename) in used_entities:
        col = col_by_key.get((mname, ename))
        if col is not None:
            tag = _kb_tag(col.description)
        else:
            model = store_by_name.get(mname)
            meas = _measures_by_name(model).get(ename) if model is not None else None
            tag = _kb_tag(meas.description) if meas is not None else None
        if tag is not None:
            used_kb_ids.add(tag)
    # "support" KBs = STRICT descendants of used KBs (a used entity may inline a
    # child KB's def). Excluding used_kb_ids themselves is what distinguishes an
    # inlined component (KB 0 under used KB 4 → INLINED) from a same-KB unused
    # DUPLICATE (a second [kb=4] column → UNUSED, delete the dup).
    support_kb_ids = relevant_kb_closure(kb_rows, list(used_kb_ids)) - used_kb_ids

    findings: list[Finding] = []

    def _tag_and_desc(mname: str, ename: str) -> tuple[int | None, bool, str | None]:
        col = col_by_key.get((mname, ename))
        if col is not None:
            return _kb_tag(col.description), _has_concept(col.description), col.description
        meas = _measures_by_name(store_by_name[mname]).get(ename) if mname in store_by_name else None
        d = meas.description if meas is not None else None
        return _kb_tag(d), _has_concept(d), d

    # --- (1) wholly-unused new agent models: one model-level finding, skip entities
    handled_entities: set[tuple[str, str]] = set()
    for mname in new_models:
        model_used = mname in used_models or any(
            k[0] == mname for k in used_entities
        )
        if not model_used:
            m = store_by_name[mname]
            has_prov = any(
                _kb_tag(c.description) is not None or _has_concept(c.description)
                for c in (m.columns or [])
            ) or any(_kb_tag(mm.description) is not None for mm in (m.measures or []))
            if has_prov:
                detail = (f"agent-added model '{mname}' (carries [kb]/[concept] entities) is not "
                          f"referenced by the winning query — consider whether a cleaner query "
                          f"should reference it; remove it only if it is not useful.")
            else:
                detail = (f"agent-added model '{mname}' is not referenced by the winning query — "
                          f"likely one-off scratch; consider deleting.")
            findings.append(
                Finding(
                    category="UNUSED_AGENT_ENTITY",
                    level="flag",
                    model=mname,
                    entity=mname,
                    detail=detail,
                )
            )
            for c in store_by_name[mname].columns or []:
                handled_entities.add((mname, c.name))
            for mm in store_by_name[mname].measures or []:
                handled_entities.add((mname, mm.name))

    # --- (2) per-entity checks for the remaining agent-added columns + measures
    for (mname, ename) in sorted(agent_cols | agent_measures):
        if (mname, ename) in handled_entities:
            continue
        col = col_by_key.get((mname, ename))
        is_measure = col is None
        tag, is_concept, _desc = _tag_and_desc(mname, ename)
        is_pk = bool(col.primary_key) if col is not None else False
        is_trivial = (not is_measure) and _is_trivial_base(column=col)
        is_derived = is_measure or (not is_trivial)
        used = (mname, ename) in used_entities

        # ORPHAN: a [kb=N] tag for a KB outside the advisory relevant set (any usage).
        if tag is not None and relevant and tag not in relevant:
            findings.append(
                Finding(
                    category="ORPHAN_KB_ENTITY",
                    level="flag",
                    model=mname,
                    entity=ename,
                    kb_id=tag,
                    detail=f"'{ename}' is tagged [kb={tag}] but KB {tag} is not in the "
                    f"advisory relevant set {sorted(relevant)}.",
                )
            )

        if used:
            # NON_KB: a used DERIVED entity with no [kb] tag and no [concept] marker.
            if is_derived and tag is None and not is_concept:
                findings.append(
                    Finding(
                        category="NON_KB_ENTITY",
                        level="flag",
                        model=mname,
                        entity=ename,
                        detail=f"agent-created derived {'measure' if is_measure else 'column'} "
                        f"'{ename}' is used by the query but carries no [kb=N] provenance and no "
                        f"[concept] marker — classify it (tag [concept] or demote into the query).",
                    )
                )
            continue

        # unused entity below
        if is_pk or ename in join_keys_by_model.get(mname, set()):
            continue  # PK / join key is structural scaffolding, never "unused"
        # A trivial-base (passthrough) column of a source_queries-backed bridge is a
        # compiler-generated projection of the backing query, NOT independently
        # removable scratch — you would have to rewrite the source_query, and a
        # ratio-of-aggregates measure (e.g. ``mar = miss_appt:sum / pat_ref:count_distinct``)
        # legitimately surfaces its component columns in the projection. Structural.
        owner = store_by_name.get(mname)
        if is_trivial and owner is not None and getattr(owner, "source_queries", None):
            continue
        # inlined KB-support precedence: a derived [kb] entity whose KB supports a
        # used entity but is not referenced by name → reference it, don't delete.
        if is_derived and tag is not None and tag in support_kb_ids:
            findings.append(
                Finding(
                    category="INLINED_KB_DEF",
                    level="flag",
                    model=mname,
                    entity=ename,
                    kb_id=tag,
                    detail=f"[kb={tag}] '{ename}' is not referenced by name, yet a used entity "
                    f"inlines its definition — rewrite the consumer to reference '{ename}' "
                    f"(clean DAG). Gate the rewrite on the identical-result check.",
                )
            )
            continue
        kind = "measure" if is_measure else "column"
        if tag is not None or is_concept:
            prov = f"[kb={tag}]" if tag is not None else "[concept]"
            detail = (f"{prov} {kind} '{ename}' is not referenced by the winning query — "
                      f"consider whether a cleaner query should reference it (it may be a useful "
                      f"reusable definition), or remove it if it is not useful.")
        else:
            detail = (f"agent-added {kind} '{ename}' is not referenced by the winning query — "
                      f"likely one-off scratch; consider deleting.")
        findings.append(
            Finding(
                category="UNUSED_AGENT_ENTITY",
                level="flag",
                model=mname,
                entity=ename,
                kb_id=tag,
                detail=detail,
            )
        )

    # --- (3) KB-materialization cross-checks (advisory / soft)
    materialized_kb: set[int] = set()
    for m in store_models:
        for c in m.columns or []:
            t = _kb_tag(c.description)
            if t is not None:
                materialized_kb.add(t)
        for mm in m.measures or []:
            t = _kb_tag(mm.description)
            if t is not None:
                materialized_kb.add(t)
    memory_kb: set[int] = set()
    for mem in memories or []:
        t = _kb_tag(mem.get("text")) if isinstance(mem, dict) else None
        if t is None and isinstance(mem, dict) and isinstance(mem.get("id"), int):
            t = mem["id"]
        if t is not None:
            memory_kb.add(t)

    not_materialized: list[int] = []
    for kb_id in sorted(relevant):
        if kb_id in materialized_kb:
            continue
        if kb_id in memory_kb:
            findings.append(
                Finding(
                    category="DEFERRED_RELEVANT_KB",
                    level="flag",
                    kb_id=kb_id,
                    detail=f"relevant KB {kb_id} is present only as a deferred memory, not an "
                    f"entity — suspicious for a KB needed to answer the query.",
                )
            )
        else:
            not_materialized.append(kb_id)
    # summarised into ONE advisory flag (external_knowledge is unreliable, and its
    # transitive closure can be large — one line per store, not one per KB).
    if not_materialized:
        findings.append(
            Finding(
                category="EXPECTED_KB_NOT_MATERIALIZED",
                level="flag",
                kb_id=not_materialized[0] if len(not_materialized) == 1 else None,
                detail=f"{len(not_materialized)} advisory-relevant KB(s) not materialised as "
                f"entities: {not_materialized}. external_knowledge is unreliable — verify "
                f"relevance before encoding; waive if the anchors are mislabeled for this task.",
            )
        )

    findings.extend(_inline_query_work(winning_query))
    return findings


# ---------------------------------------------------------------------------
# YAMLStorage adapters — load real stores through a live SLayer instance
# ---------------------------------------------------------------------------


async def load_store_models(storage: Any, db: str) -> list[SlayerModel]:
    """Load every model of datasource ``db`` from a live ``YAMLStorage`` — the
    SLayer instance is the only path to the models (never re-parse YAML text)."""
    names = await storage.list_models(data_source=db)
    models: list[SlayerModel] = []
    for name in names:
        model = await storage.get_model(name, data_source=db)
        if model is not None:
            models.append(model)
    return models


async def check_store(
    *,
    store_storage: Any,
    baseline_storage: Any | None,
    db: str,
    kb_rows: list[dict],
    relevant_kb_ids: list[int],
    winning_query: dict | list,
    memories: list[dict] | None = None,
    instance_id: str | None = None,
    benchmark: str | None = None,
    reward: float | None = None,
) -> StoreCheckReport:
    """Load store + baseline models via ``YAMLStorage`` and run :func:`check_models`.
    ``baseline_storage=None`` (cache absent / ``cache_fp`` mismatch) → skipped with
    ``baseline_available=False`` (never silently passed)."""
    if baseline_storage is None:
        return StoreCheckReport.from_findings(
            [], instance_id=instance_id, db=db, benchmark=benchmark,
            reward=reward, baseline_available=False,
        )
    store_models = await load_store_models(store_storage, db)
    baseline_models = await load_store_models(baseline_storage, db)
    findings = await check_models(
        store_models=store_models,
        baseline_models=baseline_models,
        kb_rows=kb_rows,
        relevant_kb_ids=relevant_kb_ids,
        winning_query=winning_query,
        memories=memories,
    )
    return StoreCheckReport.from_findings(
        findings, instance_id=instance_id, db=db, benchmark=benchmark,
        reward=reward, baseline_available=True,
    )
