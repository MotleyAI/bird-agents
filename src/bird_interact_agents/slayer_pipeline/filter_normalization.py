"""Deterministic case/whitespace normalization of SLayer Mode-B query filters.

DEV-1478 follow-up. The benchmark's NL questions + KB carry no information about
how the data spells its values, so a text-equality filter like
``category == 'Gadgets'`` semantically means *all* case/whitespace variants
(``'gadgets'``, ``'GADGETS'``, ``' Gadgets '``). Relying on the LLM
(encoder/agent) to remember to wrap text comparisons in ``LOWER(TRIM(...))`` is
unreliable. This module does it deterministically on the structured
``SlayerQuery.filters`` (Mode-B DSL) wherever a query flows through this repo.

Scope:
- **Mode-B filter strings only** (the Python-expression DSL,
  ``"status == 'completed'"``). Applied to FILTER clauses — never projections,
  dimensions, GROUP BY, or joins (those are structurally separate args).
- Operators ``==`` / ``!=`` / ``in`` / ``not in`` against string literals.
  ``LIKE`` is left untouched in v1 (already fuzzy; a separate pattern-lowering
  pass would be needed).

Mode-B is Python-expression syntax, so we parse + rewrite + re-emit with stdlib
``ast`` (NOT sqlglot — ``==`` isn't SQL). SQL-style inputs (``=``, ``<>``,
uppercase ``IN``/``AND``/``OR``/``NOT``/``IS``/``NULL``) are canonicalized first
by reusing SLayer's own ``_preprocess_sql_operators`` — the canonical
preprocessing ``parse_filter`` itself runs — so we accept exactly what SLayer
accepts and never diverge. ``lower``/``trim`` are first-class Mode-B scalars
(SLayer ``STRING_HYGIENE_OPS``), so ``lower(trim(col)) == 'x'`` round-trips
through SLayer's translator.
"""

from __future__ import annotations

import ast
import copy
from typing import Any

# Reuse SLayer's canonical SQL-operator preprocessing (user owns SLayer; reuse
# over reinvent). It is string-literal-aware: `=`→`==`, `<>`→`!=`,
# IN/AND/OR/NOT/IS lowercased, NULL→None — exactly what `parse_filter` runs
# before its own `ast.parse`.
from slayer.core.formula import _preprocess_sql_operators

# Hygiene calls we peel off the LHS before re-wrapping, so an already- or
# partially-normalized predicate canonicalizes instead of double-wrapping.
_PEELABLE_FUNCS = frozenset({"lower", "trim", "upper"})

# Comparison operators we normalize. Membership (`in`/`not in`) + equality.
# Ordering ops (`<`, `>`, …) and identity (`is`) are left alone.
_IN_SCOPE_OPS = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)


def _is_placeholder(value: str) -> bool:
    """A SLayer query-variable placeholder like ``'{Status}'``. These are
    substituted BEFORE filter parsing, so lowercasing them would break
    substitution — treat as out of scope."""
    return "{" in value and "}" in value


def _peel_to_column(node: ast.expr) -> ast.expr | None:
    """If ``node`` is a column reference — a bare ``Name``/``Attribute`` or a
    ``lower``/``trim``/``upper`` call (possibly nested) wrapping one — return
    the bare column-ref node. Else ``None`` (the LHS isn't a plain column, so
    we don't touch the comparison)."""
    cur = node
    while (
        isinstance(cur, ast.Call)
        and isinstance(cur.func, ast.Name)
        and cur.func.id in _PEELABLE_FUNCS
        and len(cur.args) == 1
        and not cur.keywords
    ):
        cur = cur.args[0]
    return cur if isinstance(cur, (ast.Name, ast.Attribute)) else None


def _wrap_lower_trim(col: ast.expr) -> ast.expr:
    """Build the canonical ``lower(trim(<col>))`` call node."""
    return ast.Call(
        func=ast.Name(id="lower", ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Name(id="trim", ctx=ast.Load()),
                args=[col],
                keywords=[],
            )
        ],
        keywords=[],
    )


def _string_literal_nodes(comparator: ast.expr) -> list[ast.Constant] | None:
    """Return the str-``Constant`` node(s) iff ``comparator`` is a string
    literal OR a non-empty ``Tuple``/``List`` of string literals, none of which
    is a ``{...}`` placeholder. Else ``None`` (out of scope: numeric / column /
    placeholder / mixed)."""
    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
        return None if _is_placeholder(comparator.value) else [comparator]
    if isinstance(comparator, (ast.Tuple, ast.List)) and comparator.elts:
        elts = comparator.elts
        if all(
            isinstance(e, ast.Constant)
            and isinstance(e.value, str)
            and not _is_placeholder(e.value)
            for e in elts
        ):
            return [e for e in elts if isinstance(e, ast.Constant)]
    return None


class _FilterNormalizer(ast.NodeTransformer):
    """Wraps the column side of in-scope text comparisons in ``lower(trim(...))``
    and lowercases the string literal(s). Records whether anything changed."""

    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        # Only simple (non-chained) comparisons.
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        if not isinstance(node.ops[0], _IN_SCOPE_OPS):
            return node
        literals = _string_literal_nodes(node.comparators[0])
        if literals is None:
            return node
        base_col = _peel_to_column(node.left)
        if base_col is None:
            return node
        # In scope: canonicalize the LHS to lower(trim(col)) AND lowercase the
        # literal(s) — both, always. (Canonicalizing the LHS without lowering
        # the literal, or vice-versa, would still mismatch at runtime.)
        node.left = _wrap_lower_trim(base_col)
        for lit in literals:
            if isinstance(lit.value, str):  # always true per _string_literal_nodes
                lit.value = lit.value.lower()
        self.changed = True
        return node


def normalize_mode_b_filter(filter_str: str) -> str:
    """Normalize one Mode-B filter expression.

    Returns the input UNCHANGED on any failure (unparseable, chained compare,
    LIKE/concat/colon-aggregation, out-of-scope operator, non-string or
    placeholder RHS, non-column LHS) — fail-safe. Idempotent: the canonical
    ``lower(trim(col)) == '<lower>'`` form is a fixed point.

    Boundary (v1): we apply only ``_preprocess_sql_operators``, the one SLayer
    preprocessing step that round-trips cleanly back to Mode-B. The others
    (func-style/colon aggregations, ``||`` concat, ``LIKE``) rewrite into forms
    that don't unparse back to Mode-B, so a filter string that COMBINES a text
    predicate with an aggregation in ONE string (e.g.
    ``"category == 'X' and amount:sum > 5"``) no-ops entirely. The idiomatic
    fix is separate ``filters`` list entries — each normalizes independently.
    """
    if not isinstance(filter_str, str) or not filter_str.strip():
        return filter_str
    try:
        processed = _preprocess_sql_operators(filter_str)
        tree = ast.parse(processed, mode="eval")
        normalizer = _FilterNormalizer()
        normalizer.visit(tree)
        if not normalizer.changed:
            # Nothing in scope — return the ORIGINAL verbatim (avoid cosmetic
            # ast.unparse reformatting of untouched filters).
            return filter_str
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001 — any failure → leave the filter as-is
        return filter_str


def _normalize_filters_list(filters: Any) -> Any:
    if not isinstance(filters, list):
        return filters
    return [
        normalize_mode_b_filter(f) if isinstance(f, str) else f
        for f in filters
    ]


def normalize_filters_list(
    filters: Any, *, normalize: bool = True,
) -> Any:
    """Public helper: deep-copy + (optionally) normalize a bare filters list.

    The ``mcp__bird-interact-tools__query`` wrapper (DEV-1534 Fix C) uses
    this to pre-process its ``filters`` arg before forwarding to SLayer's
    MCP ``query``. Returns ``None`` unchanged so the wrapper can forward
    ``filters: Optional[List[str]] = None`` without special-casing.

    With ``normalize=False`` (the opt-out path): returns a deep-copy of
    the input list, leaving each filter string verbatim. The agent gets
    case-sensitive equality semantics. With ``normalize=True`` (default):
    each string filter is wrapped in ``lower(trim(...))`` and the literal
    is lowercased, per ``normalize_mode_b_filter``.
    """
    if filters is None:
        return None
    if not isinstance(filters, list):
        return copy.deepcopy(filters)
    if not normalize:
        return copy.deepcopy(filters)
    return _normalize_filters_list(filters)


def _normalize_stage(stage: Any) -> None:
    """In-place (on an already-copied stage dict): normalize its ``filters``."""
    if isinstance(stage, dict) and isinstance(stage.get("filters"), list):
        stage["filters"] = _normalize_filters_list(stage["filters"])


def normalize_query_payload(parsed: Any, *, normalize: bool = True) -> Any:
    """``submit_slayer_query`` variant: a single SlayerQuery dict OR a list of
    stage dicts (the two shapes ``_compile_sql`` accepts). Deep-copies; never
    mutates the input.

    DEV-1534 Fix C: with ``normalize=False`` (the opt-out path), returns
    a deep-copy of ``parsed`` with every filter string VERBATIM (no
    ``lower(trim(...))`` wrap, no literal lowercasing). The agent gets
    case-sensitive equality semantics. This is wired into
    ``submit_slayer_query`` via a separate ``normalize_filters`` tool
    parameter — the flag lives OUTSIDE the JSON DSL so the recorded
    ``submitted_query`` stays the agent's original payload.
    """
    parsed = copy.deepcopy(parsed)
    if not normalize:
        return parsed
    if isinstance(parsed, dict):
        _normalize_stage(parsed)
    elif isinstance(parsed, list):
        for stage in parsed:
            _normalize_stage(stage)
    return parsed


def normalize_tool_filters(tool_name: str, tool_args: Any) -> Any:
    """Facade entry: deep-copy ``tool_args`` and normalize the Mode-B filter
    strings in the arg that carries them, dispatched by MCP tool name:

    - ``query`` → ``filters`` (top-level list)
    - ``query_nested`` → each stage in ``queries`` → its ``filters``
    - ``create_model`` → ``query`` (a stage dict or list of stage dicts)
    - ``edit_model`` → each stage in ``source_queries`` → its ``filters``
      (NOT ``add_filters`` / ``remove_filters`` — those are Mode-A SQL)

    Leaves ``dimensions`` / ``measures`` / ``order`` / joins untouched. Returns
    the (possibly unchanged) deep copy; never mutates the input.
    """
    if not isinstance(tool_args, dict):
        return tool_args
    args = copy.deepcopy(tool_args)
    if tool_name == "query":
        if isinstance(args.get("filters"), list):
            args["filters"] = _normalize_filters_list(args["filters"])
    elif tool_name == "query_nested":
        if isinstance(args.get("queries"), list):
            for stage in args["queries"]:
                _normalize_stage(stage)
    elif tool_name == "create_model":
        q = args.get("query")
        if isinstance(q, dict):
            _normalize_stage(q)
        elif isinstance(q, list):
            for stage in q:
                _normalize_stage(stage)
    elif tool_name == "edit_model":
        if isinstance(args.get("source_queries"), list):
            for stage in args["source_queries"]:
                _normalize_stage(stage)
    return args
