"""DEV-1515: tolerant grader producing the 8-row cascading verdict.

Each ``grade_submission`` call returns a ``CascadeVerdict`` carrying
N1..N8 booleans + diagnostic informational fields. The cascade is
monotone by construction: passing at level N implies passing at level
N+1. ``enforce_monotone_cascade`` is the single place that enforces
this property; both the inline grader and the aggregator pipe their
raw bools through it.

LLM-judge tests stay mechanical per the project memory rule: never
assert on prompt content; cache key + timeout + persisted
SubmissionEvaluation fields only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import sqlglot
import sqlglot.expressions as sg_expr
from pydantic import BaseModel, ConfigDict, Field

from bird_interact_agents.eval.annotation_schema import (
    PhaseVerdict,
    RowsetRelation,
    TaskAnnotation,
    VariantInformational,
    VariantMatch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORDER BY parser
# ---------------------------------------------------------------------------


class OrderByKey(BaseModel):
    """One parsed ORDER BY term.

    ``column_index`` is the 0-based index into the SELECT list. ``None``
    means the term references an expression that is NOT a select-list
    column (e.g. ``ORDER BY a + b`` where the select list has ``a, b``
    individually). The grader falls back to N3-strict for the variant
    in that case rather than guessing the bucket key.
    """
    model_config = ConfigDict(extra="forbid")

    column_index: Optional[int]
    desc: bool = False


def _select_list_aliases(select: sg_expr.Select) -> List[Tuple[str, sg_expr.Expression]]:
    """Return ``[(name, expr)]`` for each item in the SELECT list.

    ``name`` is the explicit alias if present, else the column name when
    the expression is a bare Column, else the raw SQL of the expression
    (useful for substring matching on bare expressions like ``a + b``).
    """
    out: List[Tuple[str, sg_expr.Expression]] = []
    for proj in select.expressions or []:
        if isinstance(proj, sg_expr.Alias):
            alias = proj.alias_or_name
            inner = proj.this
            out.append((alias.lower() if alias else "", inner))
        elif isinstance(proj, sg_expr.Column):
            out.append((proj.name.lower(), proj))
        else:
            out.append((proj.sql().lower(), proj))
    return out


def parse_orderby_keys(sql: str) -> List[OrderByKey]:
    """Parse ``sql`` and return the ORDER BY column indices into the
    SELECT list.

    Returns ``[]`` when no ORDER BY clause is present. For a term that
    can't be mapped (an expression not in the SELECT list), the
    corresponding ``OrderByKey.column_index`` is ``None``.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError:
        return []
    select = parsed.find(sg_expr.Select)
    if select is None:
        return []
    order = select.args.get("order")
    if order is None:
        return []
    aliases = _select_list_aliases(select)
    keys: List[OrderByKey] = []
    for term in order.expressions:
        desc = bool(term.args.get("desc"))
        expr = term.this
        idx: Optional[int] = None
        # Bare integer: ORDER BY 1, 2, …
        if isinstance(expr, sg_expr.Literal) and expr.is_int:
            try:
                pos = int(expr.this)
                if 1 <= pos <= len(aliases):
                    idx = pos - 1
            except (ValueError, TypeError):
                idx = None
        else:
            name = expr.name.lower() if isinstance(expr, sg_expr.Column) else None
            if name is None:
                name = expr.sql().lower()
            for i, (alias_or_col, _expr) in enumerate(aliases):
                if alias_or_col and alias_or_col == name:
                    idx = i
                    break
        keys.append(OrderByKey(column_index=idx, desc=desc))
    return keys


# ---------------------------------------------------------------------------
# Cell-level relaxations
# ---------------------------------------------------------------------------


def _row_count_match(pred: Sequence[Sequence], gold: Sequence[Sequence]) -> bool:
    return len(pred) == len(gold)


def compare_tie_order(
    pred: Sequence[Sequence],
    gold: Sequence[Sequence],
    *,
    orderby_indices: Sequence[int],
) -> bool:
    """Bucket each row by the values at ``orderby_indices``, then check
    set-equality within each bucket.

    Empty ``orderby_indices`` (no ORDER BY in the source SQL) → fall back
    to set-equality (which is what N3 does)."""
    if not _row_count_match(pred, gold):
        return False
    if not orderby_indices:
        return _set_equal(pred, gold)

    def _key(row: Sequence) -> Tuple:
        return tuple(row[i] for i in orderby_indices)

    pred_buckets: dict[Tuple, list[tuple]] = {}
    gold_buckets: dict[Tuple, list[tuple]] = {}
    for r in pred:
        pred_buckets.setdefault(_key(r), []).append(tuple(r))
    for r in gold:
        gold_buckets.setdefault(_key(r), []).append(tuple(r))
    if pred_buckets.keys() != gold_buckets.keys():
        return False
    # Cross-bucket order: the keys must appear in the same sequence in pred
    # and gold (we already required _row_count_match, so the bucket-position
    # check uses the FIRST occurrence in each side).
    pred_key_order = [_key(r) for r in pred]
    gold_key_order = [_key(r) for r in gold]
    seen_pred: list[Tuple] = []
    seen_gold: list[Tuple] = []
    for k in pred_key_order:
        if k not in seen_pred:
            seen_pred.append(k)
    for k in gold_key_order:
        if k not in seen_gold:
            seen_gold.append(k)
    if seen_pred != seen_gold:
        return False
    for k in pred_buckets:
        # Bag-equality within bucket so duplicates count.
        if sorted(map(_canonical_repr, pred_buckets[k])) != sorted(
            map(_canonical_repr, gold_buckets[k])
        ):
            return False
    return True


def _set_equal(pred: Sequence[Sequence], gold: Sequence[Sequence]) -> bool:
    """Bag-equality (multiset) on the canonical-repr of each row."""
    return sorted(map(_canonical_repr, pred)) == sorted(
        map(_canonical_repr, gold)
    )


def _canonical_repr(row: Sequence) -> str:
    """Stable string repr for a row so heterogeneous tuples sort."""
    return "|".join(repr(c) for c in row)


def _numeric_cell_equal(a: Any, b: Any, *, epsilon: float) -> bool:
    """Per-cell numeric tolerance. Non-numeric cells fall back to ==."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= epsilon
    return a == b


def compare_numeric_epsilon(
    pred: Sequence[Sequence],
    gold: Sequence[Sequence],
    *,
    epsilon: float,
) -> bool:
    if not _row_count_match(pred, gold):
        return False
    # Per-row, per-cell tolerance, but we still need bag semantics.
    used = [False] * len(gold)
    for pr in pred:
        match = -1
        for j, gr in enumerate(gold):
            if used[j] or len(pr) != len(gr):
                continue
            if all(
                _numeric_cell_equal(a, b, epsilon=epsilon)
                for a, b in zip(pr, gr)
            ):
                match = j
                break
        if match < 0:
            return False
        used[match] = True
    return all(used)


def _strip_trailing(cell: Any) -> Any:
    return cell.rstrip() if isinstance(cell, str) else cell


def compare_trailing_whitespace(
    pred: Sequence[Sequence],
    gold: Sequence[Sequence],
) -> bool:
    if not _row_count_match(pred, gold):
        return False
    pred_n = [tuple(_strip_trailing(c) for c in r) for r in pred]
    gold_n = [tuple(_strip_trailing(c) for c in r) for r in gold]
    return _set_equal(pred_n, gold_n)


def compare_column_order(
    pred: Sequence[Sequence],
    gold: Sequence[Sequence],
    *,
    pred_cols: Sequence[str],
    gold_cols: Sequence[str],
) -> bool:
    """Align columns by case-insensitive name, then set-equal the rows.

    Returns False when column counts differ, or when the column-name
    sets differ (modulo case)."""
    if len(pred_cols) != len(gold_cols):
        return False
    pred_l = [c.lower() for c in pred_cols]
    gold_l = [c.lower() for c in gold_cols]
    if set(pred_l) != set(gold_l):
        return False
    # Permutation: position in pred for each gold column.
    perm = [pred_l.index(c) for c in gold_l]
    aligned = [tuple(r[i] for i in perm) for r in pred]
    return _set_equal(aligned, gold)


# ---------------------------------------------------------------------------
# Tier 2 informational helpers
# ---------------------------------------------------------------------------


def classify_rowset_relation(
    *,
    pred: Sequence[Sequence],
    gold: Sequence[Sequence],
) -> RowsetRelation:
    """Set-relation between pred and gold rowsets (canonical-repr keys)."""
    p = set(_canonical_repr(r) for r in pred)
    g = set(_canonical_repr(r) for r in gold)
    if not p and not g:
        return "equal_rowset"
    if p == g:
        return "equal_rowset"
    if p < g:
        return "strict_subset_of"
    if p > g:
        return "strict_superset_of"
    if p & g:
        return "overlapping"
    return "disjoint"


def _first_divergent_row(
    *,
    pred: Sequence[Sequence],
    gold: Sequence[Sequence],
) -> Tuple[Optional[int], Optional[str]]:
    """For positional row-wise comparison, return the first index where
    the two sides differ + a one-line cell-diff string.

    When rowcounts differ, returns ``(min(len_pred, len_gold), "<len mismatch>")``."""
    min_len = min(len(pred), len(gold))
    for i in range(min_len):
        if list(pred[i]) != list(gold[i]):
            diff_cells = []
            for j in range(min(len(pred[i]), len(gold[i]))):
                if pred[i][j] != gold[i][j]:
                    diff_cells.append(
                        f"col {j}: {pred[i][j]!r} vs {gold[i][j]!r}"
                    )
            return i, f"row {i}: " + "; ".join(diff_cells)
    if len(pred) != len(gold):
        return min_len, (
            f"<row count: pred={len(pred)} vs gold={len(gold)}>"
        )
    return None, None


def _column_diff(
    *,
    pred_cols: Sequence[str], gold_cols: Sequence[str],
) -> Tuple[bool, bool, bool]:
    """(count_match, name_match_case_insensitive, order_match)."""
    count = len(pred_cols) == len(gold_cols)
    nm_set = (
        count
        and set(c.lower() for c in pred_cols) == set(c.lower() for c in gold_cols)
    )
    order = count and [c.lower() for c in pred_cols] == [c.lower() for c in gold_cols]
    return count, nm_set, order


# ---------------------------------------------------------------------------
# Monotone enforcement
# ---------------------------------------------------------------------------


_CASCADE_ORDER = [
    "n1_original_gold",
    "n2_audited_primary",
    "n3_any_audited_variant",
    "n4_tie_order",
    "n5_llm_judge",
    "n6_numeric_epsilon",
    "n7_trailing_whitespace",
    "n8_column_order",
]


def enforce_monotone_cascade(raw: dict[str, bool]) -> dict[str, bool]:
    """Given raw N1..N8 bools, return a monotone-enforced version: once
    True, every subsequent level stays True (a pass at level N implies
    a pass at level N+1)."""
    out: dict[str, bool] = {}
    seen_true = False
    for f in _CASCADE_ORDER:
        v = bool(raw.get(f, False))
        if seen_true or v:
            out[f] = True
            seen_true = True
        else:
            out[f] = False
    return out


# ---------------------------------------------------------------------------
# LLM judge protocol + cached wrapper
# ---------------------------------------------------------------------------


class LLMJudgeProtocol(Protocol):
    """Minimal contract: ``judge`` returns True (accept), False (reject),
    or None (timeout / transient error → fall through)."""
    def judge(self, **kwargs: Any) -> Optional[bool]:  # pragma: no cover
        ...


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _cache_key(payload: dict[str, Any]) -> str:
    keys = (
        "model_name",
        "annotation_content_hash",
        "gold_variants_content_hash",
        "submitted_sql_normalized",
    )
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode())
        h.update(b"=")
        h.update(_stable_json(payload.get(k, "")).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _normalize_sql(sql: str) -> str:
    # Collapse whitespace; case-insensitive normalization is too risky
    # (literal identifiers), so we keep case.
    return " ".join(sql.split())


class CachedLLMJudge:
    """Wraps any LLM judge with a JSON-on-disk cache keyed by content.

    Key dimensions: model name, annotation content hash, gold-variants
    content hash, normalized submitted SQL. Run-id is deliberately NOT
    included so offline re-grade reuses worker-side decisions when
    nothing meaningful changed.

    The cache value is ``{"verdict": True|False, "instance_id": ..., …}``
    so ``clear_llm_judge_cache(instance_ids=[...])`` can filter by
    instance.
    """
    def __init__(self, *, inner: Any, cache_path: Path):
        self._inner = inner
        self._cache_path = Path(cache_path)
        if self._cache_path.exists():
            self._cache: dict[str, dict] = json.loads(self._cache_path.read_text())
        else:
            self._cache = {}

    @property
    def model_name(self) -> str:
        return getattr(self._inner, "model_name", "unknown")

    def _persist(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(_stable_json(self._cache))

    def judge(self, **kwargs: Any) -> Optional[bool]:
        payload = dict(kwargs)
        payload["model_name"] = self.model_name
        payload["submitted_sql_normalized"] = _normalize_sql(
            payload.get("submitted_sql", "")
        )
        key = _cache_key(payload)
        if key in self._cache:
            return self._cache[key].get("verdict")
        result = self._inner.judge(**kwargs)
        # Store ONLY definitive verdicts; None = timeout, retry next time.
        if result is not None:
            entry: dict[str, Any] = {"verdict": bool(result)}
            if "instance_id" in kwargs:
                entry["instance_id"] = kwargs["instance_id"]
            self._cache[key] = entry
            self._persist()
        return result


# ---------------------------------------------------------------------------
# Default executor (real SQLite path) + grade_submission orchestrator
# ---------------------------------------------------------------------------


ExecutorResult = Tuple[Sequence[Sequence], Sequence[str]]
ExecutorProtocol = Callable[..., ExecutorResult]


def default_executor(
    sql: str,
    *,
    db_path: Path,
    conn: Optional[sqlite3.Connection] = None,
) -> ExecutorResult:
    """Execute ``sql`` against the SQLite DB at ``db_path`` and return
    ``(rows, column_names)``. Caches the connection when reused; the
    caller is responsible for closing if it owns one."""
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = list(cur.fetchall())
        cols = [d[0] for d in cur.description] if cur.description else []
        return rows, cols
    finally:
        if own_conn:
            conn.close()


# ---------------------------------------------------------------------------
# CascadeVerdict (in-memory grader output)
# ---------------------------------------------------------------------------


class CascadeVerdict(BaseModel):
    """Per-submission cascade output. Persisted via the ``SubmissionEvaluation``
    fields in the SubmissionAnnotation. The grader returns the verdict
    pre-monotone-enforced (callers can re-enforce defensively)."""
    model_config = ConfigDict(extra="forbid")

    n1_original_gold: bool
    n2_audited_primary: bool
    n3_any_audited_variant: bool
    n4_tie_order: bool
    n5_llm_judge: bool
    n6_numeric_epsilon: bool
    n7_trailing_whitespace: bool
    n8_column_order: bool
    matched_variant_id: Optional[str] = None
    novel_reading_judgment: Optional[PhaseVerdict] = None
    variant_matches: List[VariantMatch] = Field(default_factory=list)
    rowset_relations: List[VariantMatch] = Field(default_factory=list)


def _multi_sql_execute(
    sqls: List[str],
    *,
    db_path: Path,
    conn: Any,
    executor: ExecutorProtocol,
) -> ExecutorResult:
    """Execute a list of SQL strings and return the rows+cols of the
    LAST one. Mirrors BIRD-Interact's evaluator semantics: prior items
    in the list set up state (CREATE TEMP …); only the last returns
    rows for comparison.

    Connection-scoped state (TEMP tables, PRAGMAs, attached DBs) only
    survives across statements inside the SAME sqlite connection. When
    the caller passes ``conn=None`` and lets the default SQLite executor
    open one, the executor would otherwise open+close a fresh connection
    per statement and the setup statements would silently lose their
    TEMP state before the final comparison statement runs. Open one
    shared connection here (closed in a finally) so the whole list runs
    against the same connection.
    """
    own_conn: Optional[sqlite3.Connection] = None
    if conn is None and executor is default_executor and len(sqls) > 1:
        try:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, timeout=30,
            )
            own_conn = conn
        except sqlite3.Error:
            # db_path unreachable / not a sqlite file — let the
            # downstream executor call raise with the original error.
            conn = None
    try:
        rows: Sequence[Sequence] = []
        cols: Sequence[str] = []
        for sql in sqls:
            rows, cols = executor(sql, db_path=db_path, conn=conn)
        return rows, cols
    finally:
        if own_conn is not None:
            own_conn.close()


def grade_submission(
    *,
    task_annotation: TaskAnnotation,
    audited_gold_rows: List[dict],
    original_sol_sql: List[str],
    submitted_sql: str,
    db_path: Path,
    conn: Any = None,
    executor: Optional[ExecutorProtocol] = None,
    llm_judge: Optional[Any] = None,
    epsilon: float = 1e-6,
) -> CascadeVerdict:
    """Compute the 8-row cascade for a single submission.

    * ``audited_gold_rows`` may be empty (missing-annotation graceful
      default — see ``implicit_annotation`` factory). The cascade then
      collapses to N1; N2 and N3 mirror N1.
    * The cascade is monotone by construction: ``enforce_monotone_cascade``
      is applied before returning.
    * N4 uses the ORIGINAL gold's ORDER BY (locked simplification).
    * N5 fires ONLY when ``task_annotation.metadata_sufficiency.verdict``
      is ``"insufficient"`` AND N4 didn't already pass.
    """
    if executor is None:
        executor = default_executor  # type: ignore[assignment]
    assert executor is not None  # narrowing

    # 1) Run predicted + original gold + each variant.
    pred_rows, pred_cols = executor(submitted_sql, db_path=db_path, conn=conn)
    orig_rows, orig_cols = _multi_sql_execute(
        list(original_sol_sql), db_path=db_path, conn=conn, executor=executor,
    )

    variant_results: list[tuple[dict, Sequence[Sequence], Sequence[str]]] = []
    for v in audited_gold_rows:
        sqls = list(v.get("audited_sol_sql") or [])
        if not sqls:
            continue
        v_rows, v_cols = _multi_sql_execute(
            sqls, db_path=db_path, conn=conn, executor=executor,
        )
        variant_results.append((v, v_rows, v_cols))

    # 2) N1 — original gold strict.
    n1 = _set_equal(pred_rows, orig_rows)

    # 3) N2/N3 — audited primary / any variant strict.
    primary = next(
        (vr for vr in variant_results if vr[0].get("primary")), None,
    )
    if primary is None and not variant_results:
        # No audited variants at all → N2 == N3 == N1.
        n2 = n1
        n3 = n1
        matched_variant: Optional[str] = None
    else:
        n2 = bool(primary and _set_equal(pred_rows, primary[1]))
        matched_variant = primary[0].get("variant_id") if n2 else None
        n3 = n2
        if not n3:
            for v, v_rows, _v_cols in variant_results:
                if _set_equal(pred_rows, v_rows):
                    n3 = True
                    matched_variant = v.get("variant_id")
                    break

    # 4) N4 — tie-order against the (primary or original) variant. The
    # bucket spec is sourced from the ORIGINAL gold's ORDER BY.
    n4 = n3
    if not n4:
        orderby = parse_orderby_keys(
            original_sol_sql[-1] if original_sol_sql else ""
        )
        indices = [
            k.column_index for k in orderby if k.column_index is not None
        ]
        # If ANY key didn't resolve, we leave it out — bucketing on
        # partial keys is still safer than collapsing to N3.
        candidates = (
            [(primary[0], primary[1])] if primary else []
        ) + [(v[0], v[1]) for v in variant_results if not v[0].get("primary")]
        if not candidates:
            # No variants → fall back to original gold itself.
            candidates = [({}, orig_rows)]
        for v_meta, v_rows in candidates:
            if compare_tie_order(pred_rows, v_rows, orderby_indices=indices):
                n4 = True
                if v_meta.get("variant_id"):
                    matched_variant = v_meta["variant_id"]
                break

    # 5) N5 — LLM judge, gated on insufficient verdict.
    novel_judgment: Optional[PhaseVerdict] = None
    n5 = n4
    if not n5 and (
        task_annotation.metadata_sufficiency.verdict == "insufficient"
        and llm_judge is not None
        and task_annotation.evaluator_prompt is not None
    ):
        try:
            judged = llm_judge.judge(
                evaluator_prompt=task_annotation.evaluator_prompt,
                gold_variants_summary=[
                    {
                        "variant_id": v.get("variant_id"),
                        "interpretation": next(
                            (gv.interpretation for gv in task_annotation.gold_variants
                             if gv.variant_id == v.get("variant_id")), "",
                        ),
                    }
                    for v in audited_gold_rows
                ],
                metadata_anchors=[m.term for m in task_annotation.masked_terms],
                submitted_sql=submitted_sql,
                predicted_rows_head=list(pred_rows[:20]),
                annotation_content_hash=_annotation_hash(task_annotation),
                gold_variants_content_hash=_gold_hash(audited_gold_rows),
                instance_id=task_annotation.instance_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception("LLM judge raised on instance=%s",
                             task_annotation.instance_id)
            judged = None
        if judged is True:
            n5 = True
            novel_judgment = "pass"
        elif judged is False:
            novel_judgment = "fail"
        else:
            novel_judgment = None

    # 6) N6/N7/N8 — cell-level relaxations applied across all variants.
    n6, n7, n8 = n5, n5, n5
    if not n6:
        for _v_meta, v_rows, _v_cols in variant_results + [
            ({"variant_id": "__original__"}, orig_rows, orig_cols),  # type: ignore[list-item]
        ]:
            if compare_numeric_epsilon(pred_rows, v_rows, epsilon=epsilon):
                n6 = True
                break
    if not n7:
        n7 = n6
        if not n7:
            for _v_meta, v_rows, _v_cols in variant_results + [
                ({"variant_id": "__original__"}, orig_rows, orig_cols),  # type: ignore[list-item]
            ]:
                if compare_trailing_whitespace(pred_rows, v_rows):
                    n7 = True
                    break
    if not n8:
        n8 = n7
        if not n8:
            for _v_meta, v_rows, v_cols in variant_results + [
                ({"variant_id": "__original__"}, orig_rows, orig_cols),  # type: ignore[list-item]
            ]:
                if compare_column_order(
                    pred_rows, v_rows,
                    pred_cols=list(pred_cols), gold_cols=list(v_cols),
                ):
                    n8 = True
                    break

    # 7) Tier 2 informational per variant.
    info_matches: list[VariantMatch] = []
    for v_meta, v_rows, v_cols in variant_results:
        rel = classify_rowset_relation(pred=pred_rows, gold=v_rows)
        cm, nm, om = _column_diff(pred_cols=list(pred_cols), gold_cols=list(v_cols))
        fdri, fdcd = _first_divergent_row(pred=pred_rows, gold=v_rows)
        info_matches.append(VariantMatch(
            variant_id=v_meta.get("variant_id", "primary"),
            match=rel,
            informational=VariantInformational(
                rowset_relation=rel,
                column_count_match=cm,
                column_name_match_case_insensitive=nm,
                column_order_match=om,
                first_divergent_row_index=fdri,
                first_divergent_cell_diff=fdcd,
            ),
        ))

    raw = {
        "n1_original_gold": n1, "n2_audited_primary": n2,
        "n3_any_audited_variant": n3, "n4_tie_order": n4,
        "n5_llm_judge": n5, "n6_numeric_epsilon": n6,
        "n7_trailing_whitespace": n7, "n8_column_order": n8,
    }
    enforced = enforce_monotone_cascade(raw)
    return CascadeVerdict(
        **enforced,
        matched_variant_id=matched_variant,
        novel_reading_judgment=novel_judgment,
        variant_matches=info_matches,
    )


def _annotation_hash(ann: TaskAnnotation) -> str:
    return hashlib.sha256(ann.model_dump_json().encode()).hexdigest()


def _gold_hash(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(_stable_json(r).encode())
        h.update(b"\x00")
    return h.hexdigest()
