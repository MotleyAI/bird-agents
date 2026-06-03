"""DEV-1515 session 4: failure-mode diagnostics at grading time.

Tests the ``MissDiagnostics`` model + the diagnostics population in
``grade_submission`` for cascade-fail submissions. Per the v3 spec
(see ``plans/read-all-the-comments-peaceful-karp.md``):

* every cascade-fail populates ``cascade.miss_diagnostics``
* comparison reference is the BEST-OVERLAP audited variant
  (multiset cardinality; tie-break primary > alphabetical variant_id)
* SQL parse failure → nullable signals + parse_ok=False
* SQL execution failure → ``pred_rows=[]`` falls through cascade
  + ``sql_execution_error`` flag + error excerpt
* multiple flags can fire simultaneously (independent rules)
* one-shot benchmarks (livesqlbench) skip the ``never_asked_user``
  signal entirely
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers — sqlite DB fixture, task annotation, audited-row builder
# ---------------------------------------------------------------------------


def _build_db(tmp_path: Path) -> Path:
    """Build a small sqlite DB with two tables for the diagnostics fixtures."""
    db = tmp_path / "diag.sqlite"
    con = sqlite3.connect(str(db))
    try:
        con.execute("CREATE TABLE t1 (id INTEGER, val TEXT)")
        con.execute("CREATE TABLE t2 (id INTEGER, name TEXT)")
        con.executemany(
            "INSERT INTO t1 (id, val) VALUES (?, ?)",
            [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")],
        )
        con.executemany(
            "INSERT INTO t2 (id, name) VALUES (?, ?)",
            [(1, "alpha"), (2, "beta"), (3, "gamma")],
        )
        con.commit()
    finally:
        con.close()
    return db


def _task_annotation(
    *,
    instance_id: str = "alien_1",
    verdict: str = "sufficient",
    variant_ids: Optional[list[tuple[str, bool]]] = None,
):
    """Build a TaskAnnotation. ``variant_ids`` is a list of (variant_id,
    primary) pairs that mirrors the audited rows used by the test;
    defaults to a single ``primary`` variant. Required so the
    task annotation's `gold_variants` declares the same variants that
    the test passes via `audited_gold_rows` — otherwise downstream
    validation (or future grader changes that cross-check the two
    sources) would fail."""
    from bird_interact_agents.eval.annotation_schema import (
        AuditedGoldRef,
        GoldVariantRef,
        MetadataSufficiency,
        Provenance,
        TaskAnnotation,
    )
    if variant_ids is None:
        variant_ids = [("primary", True)]
    return TaskAnnotation(
        instance_id=instance_id,
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-05-31",
        amb_user_query="x",
        metadata_sufficiency=MetadataSufficiency(
            verdict=verdict, rationale="r",
        ),
        gold_variants=[
            GoldVariantRef(
                variant_id=vid,
                interpretation="x",
                primary=p,
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id=instance_id,
                    variant_id=vid,
                ),
            )
            for (vid, p) in variant_ids
        ],
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=instance_id,
        ),
    )


def _audited_row(
    *,
    variant_id: str,
    primary: bool,
    audited_sol_sql: list[str],
    instance_id: str = "alien_1",
) -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": "alien",
        "benchmark": "mini-interact",
        "audit_status": "edited",
        "original_sol_sql": ["SELECT id FROM t1 WHERE id < 0"],
        "audited_sol_sql": audited_sol_sql,
        "variant_id": variant_id,
        "primary": primary,
        "changes": [],
        "reasoning_summary": "",
        "skill_version": "audit-gold-sql/1.0",
        "audited_at": "2026-05-30T00:00:00+00:00",
    }


def _grade(
    *,
    db: Path,
    submitted_sql: str,
    audited_sol_sql_per_variant: list[tuple[str, bool, list[str]]],
    original_sol_sql: Optional[list[str]] = None,
    verdict: str = "sufficient",
    user_sim_n_asks: Optional[int] = None,
):
    """Wrapper that builds the inputs and calls grade_submission. The
    test asserts on the returned CascadeVerdict."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    audited = [
        _audited_row(
            variant_id=vid, primary=p, audited_sol_sql=sqls,
        )
        for (vid, p, sqls) in audited_sol_sql_per_variant
    ]
    if original_sol_sql is None:
        # Sentinel rowset that no test's agent SQL accidentally matches —
        # otherwise an agent returning empty rows would spuriously
        # pass N1 against an empty original gold and miss_diagnostics
        # would stay None.
        original_sol_sql = ["SELECT 'sentinel' AS marker FROM t1 WHERE id = 1"]
    # Derive gold_variants from the audited list so task annotation +
    # audited rows agree on variant identity (Codex minor #10).
    variant_ids = [(vid, p) for (vid, p, _) in audited_sol_sql_per_variant]
    return grade_submission(
        task_annotation=_task_annotation(verdict=verdict, variant_ids=variant_ids),
        audited_gold_rows=audited,
        original_sol_sql=original_sol_sql,
        submitted_sql=submitted_sql,
        db_path=db,
        conn=None,
        user_sim_n_asks=user_sim_n_asks,
    )


# ---------------------------------------------------------------------------
# Cascade-pass leaves diagnostics None
# ---------------------------------------------------------------------------


def test_cascade_pass_at_n3_leaves_miss_diagnostics_none(tmp_path: Path):
    """Diagnostics are computed ONLY on a strict miss. When the cascade
    passes at N3 (strict set equality), miss_diagnostics is None."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id < 3 ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2) ORDER BY id"]),
        ],
    )
    assert verdict.n3_any_audited_variant is True
    assert verdict.miss_diagnostics is None


def test_cascade_pass_at_n6_leaves_miss_diagnostics_none(tmp_path: Path):
    """Diagnostics stay None for any cascade tier pass, not just N3.
    Construct an instance where strict equality misses but N6 numeric
    epsilon flips the cascade to pass — and assert miss_diagnostics is
    still None. The happy-path skip-diagnostics contract is per-cascade
    not per-strict-tier."""
    db = _build_db(tmp_path)
    # Agent: 12.345600001 (within 1e-6 of 12.3456); Gold: 12.3456
    verdict = _grade(
        db=db,
        submitted_sql="SELECT 12.345600001 AS v",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT 12.3456 AS v"]),
        ],
    )
    # N3 strict misses, N6 numeric-epsilon passes
    assert verdict.n3_any_audited_variant is False
    assert verdict.n6_numeric_epsilon is True
    assert verdict.miss_diagnostics is None


# ---------------------------------------------------------------------------
# Per-flag fixtures — each test engineers a scenario where ONLY the
# named flag should fire (modulo flags that are entailed by the scenario).
# ---------------------------------------------------------------------------


def test_flag_sql_execution_error(tmp_path: Path):
    """Agent SQL raises at execution time. Cascade MUST complete with
    all N-tiers False, pred_rows captured as empty, the error excerpt
    captured, and the diagnostics populated with both
    sql_execution_error AND empty_agent_result (since pred_rows=[]
    while the best variant returns ≥1 row).

    Codex major #5: pin the full cascade-fall-through contract, not
    just the executed_ok boolean."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT * FROM does_not_exist",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 ORDER BY id"]),
        ],
    )
    # Cascade fall-through: every tier False.
    assert verdict.n1_original_gold is False
    assert verdict.n2_audited_primary is False
    assert verdict.n3_any_audited_variant is False
    assert verdict.n4_tie_order is False
    assert verdict.n6_numeric_epsilon is False
    assert verdict.n7_trailing_whitespace is False
    assert verdict.n8_column_order is False
    assert verdict.n9_case_fold is False

    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_sql_executed_ok is False
    assert md.agent_sql_error_excerpt is not None
    assert "no such table" in md.agent_sql_error_excerpt.lower()
    assert md.agent_row_count == 0
    assert md.best_variant_row_count > 0
    assert "sql_execution_error" in md.miss_patterns
    assert "empty_agent_result" in md.miss_patterns


def test_flag_sql_parse_error_agent_side(tmp_path: Path):
    """Agent SQL is unparseable garbage; sqlglot fails. parse_ok flag
    flips False; sql_parse_error flag fires; AGENT-side SQL-derived
    fields are None (Optional sentinel) rather than False/0.

    The best-variant side parses fine — its fields are concrete."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="THIS IS NOT SQL AT ALL ;;",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_sql_parse_ok is False
    assert md.agent_sql_parse_error is not None
    assert md.agent_tables_referenced is None
    assert md.agent_has_group_by is None
    assert md.agent_has_aggregate is None
    assert md.agent_join_count is None
    assert md.agent_where_conjunct_count is None
    assert md.agent_has_having is None
    assert md.agent_has_limit is None
    assert md.table_set_match is None
    # Best-variant side parsed fine.
    assert md.best_variant_sql_parse_ok is True
    assert md.best_variant_sql_parse_error is None
    assert md.best_variant_tables_referenced == ["t1"]
    assert md.best_variant_has_group_by is False
    assert "sql_parse_error" in md.miss_patterns


def test_flag_sql_parse_error_best_variant_side(tmp_path: Path):
    """Mixed-variants case: one variant executes fine, one is malformed.
    The malformed variant is SKIPPED from variant_results (not coerced
    to ``([], [])`` — that would risk false N2/N3 passes per
    CodeRabbit r3336709435). The surviving good variant becomes the
    best-overlap reference; if its SQL parses fine, sql_parse_error
    does NOT fire because the broken variant was excluded from the
    sqlglot pass.

    Audit-quality issues (broken variants in the gold set) are out of
    scope for MissDiagnostics — they belong to a separate audit-side
    annotation. This test pins the new contract: broken variants are
    silently skipped and the diagnostics path proceeds against the
    surviving ones."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["NOT A VALID SQL AT ALL ;;"]),
            ("alt", False, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    # The broken "primary" was skipped from variant_results; only "alt"
    # survives, so best_variant_id must be "alt".
    assert md.best_variant_id == "alt"
    assert md.agent_sql_parse_ok is True
    assert md.best_variant_sql_parse_ok is True
    assert "sql_parse_error" not in md.miss_patterns


def test_all_variants_failing_execution_leaves_diagnostics_none(tmp_path: Path):
    """When EVERY audited variant fails to execute, there is no
    canonical gold to diagnose against — miss_diagnostics stays None
    (we never coerce to empty, which would risk a false N2/N3 pass).
    The cascade's `phase1_against_*` fields still record the misses;
    callers can detect the unevaluable-grading state by observing
    miss_diagnostics is None on a cascade-fail."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["NOT A VALID SQL AT ALL ;;"]),
        ],
    )
    # Cascade should fail across the board.
    assert verdict.n2_audited_primary is False
    assert verdict.n3_any_audited_variant is False
    # No surviving variant → no canonical reference → no diagnostics.
    assert verdict.miss_diagnostics is None


def test_flag_empty_agent_result(tmp_path: Path):
    """Agent returns zero rows; best variant returns non-empty."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id > 999",  # 0 rows
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id <= 3 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_row_count == 0
    assert md.best_variant_row_count > 0
    assert "empty_agent_result" in md.miss_patterns


def test_flag_wrong_table_set(tmp_path: Path):
    """Agent references t2; gold references t1. Same column shape so
    column_projection_mismatch doesn't fire, but table_set_match=False
    so wrong_table_set does."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t2 ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_sql_parse_ok is True
    assert md.best_variant_sql_parse_ok is True
    assert md.agent_tables_referenced == ["t2"]
    assert md.best_variant_tables_referenced == ["t1"]
    assert md.table_set_match is False
    assert "wrong_table_set" in md.miss_patterns


def test_flag_aggregation_shape_mismatch(tmp_path: Path):
    """Agent has no GROUP BY / no aggregate; gold has both. Flag fires."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT COUNT(*) AS n FROM t1 GROUP BY val"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_has_group_by is False
    assert md.best_variant_has_group_by is True
    assert md.agent_has_aggregate is False
    assert md.best_variant_has_aggregate is True
    assert "aggregation_shape_mismatch" in md.miss_patterns


def test_flag_column_count_mismatch(tmp_path: Path):
    """Agent projects 1 column; gold projects 2. Different arity →
    bag equality on canonical row reprs cannot hold (same for
    BIRD-Interact's ex_base). column_count_mismatch is the
    load-bearing column-shape flag for this case."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id, val FROM t1 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    # Informational fields are populated.
    assert md.column_count_match is False
    assert md.agent_column_count == 1
    assert md.best_variant_column_count == 2
    flags = set(md.miss_patterns)
    assert "column_count_mismatch" in flags
    # Mutually exclusive — order can't be checked when counts differ.
    assert "column_order_mismatch" not in flags


def test_flag_column_order_mismatch(tmp_path: Path):
    """Agent projects the right COLUMNS (after stripping slayer's
    dot-prefix + lowercasing) but in a different ORDER from gold.
    Same column count, normalised name lists match as SETS but differ
    as LISTS → column_order_mismatch fires; column_count_mismatch
    does not. This is the near-miss pattern where N8 column-order
    tolerance would have rescued the cascade if slayer's namespacing
    hadn't tripped its column-name set check."""
    db = _build_db(tmp_path)
    # Agent's column names use slayer-namespacing; gold's are bare;
    # order is reversed. Force rowset divergence so the cascade
    # actually reaches diagnostics (values differ in row 0 vs row 0
    # because the columns are swapped in the SELECT list).
    verdict = _grade(
        db=db,
        submitted_sql=(
            'SELECT val AS "t1.val", id AS "t1.id" FROM t1 WHERE id <= 2 ORDER BY id'
        ),
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT id, val FROM t1 WHERE id <= 2 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.column_count_match is True
    assert md.agent_column_count == 2
    assert md.best_variant_column_count == 2
    flags = set(md.miss_patterns)
    assert "column_order_mismatch" in flags
    assert "column_count_mismatch" not in flags


def test_column_name_only_divergence_no_column_flag(tmp_path: Path):
    """Same column count + names differ in a NON-RECOVERABLE way
    (normalised name sets are different — agent picked actually
    different columns, not just renamed/namespaced). Neither
    column_count_mismatch nor column_order_mismatch fires —
    column-NAME-only divergence is stylistic / actually-different
    projection and we don't surface it as a column-shape flag
    (whatever else caused the cascade fail will surface elsewhere)."""
    db = _build_db(tmp_path)
    # Agent projects `id`; gold projects `val`. Same arity (1), but
    # the normalised name sets are {'id'} vs {'val'} — disjoint.
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id <= 2 ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT val FROM t1 WHERE id <= 2 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.column_count_match is True
    # column-name signal is populated as informational.
    assert md.column_name_match_case_insensitive is False
    flags = set(md.miss_patterns)
    # No column-shape flag fires; the rowset flags do the talking.
    assert "column_count_mismatch" not in flags
    assert "column_order_mismatch" not in flags


def test_flag_predicate_count_mismatch(tmp_path: Path):
    """Agent has 1 WHERE conjunct; gold has 2. predicate_count_mismatch
    fires. Rowsets MUST differ for the cascade to reach the diagnostics
    path — engineer a fixture where the extra conjunct on the gold side
    actually filters rows differently."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id > 0",  # 5 rows, 1 conjunct
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT id FROM t1 WHERE id > 0 AND id < 3"]),  # 2 rows, 2 conjuncts
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_where_conjunct_count == 1
    assert md.best_variant_where_conjunct_count == 2
    assert "predicate_count_mismatch" in md.miss_patterns


def test_flag_having_presence_mismatch(tmp_path: Path):
    """Agent has no HAVING; gold has HAVING that actually filters rows.
    Rowsets must differ so the cascade reaches diagnostics."""
    db = _build_db(tmp_path)
    # Agent: all 5 val groups, no filter. Gold: same shape but HAVING
    # restricts to groups with COUNT > 99 (none qualify → 0 rows).
    verdict = _grade(
        db=db,
        submitted_sql="SELECT val, COUNT(*) FROM t1 GROUP BY val",
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT val, COUNT(*) FROM t1 GROUP BY val "
              "HAVING COUNT(*) > 99"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_has_having is False
    assert md.best_variant_has_having is True
    assert "having_presence_mismatch" in md.miss_patterns


def test_flag_limit_presence_mismatch(tmp_path: Path):
    """Agent has LIMIT; gold has no LIMIT. limit_presence_mismatch fires."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 ORDER BY id LIMIT 2",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_has_limit is True
    assert md.best_variant_has_limit is False
    assert "limit_presence_mismatch" in md.miss_patterns


def test_flag_disjoint_rowset(tmp_path: Path):
    """Agent and gold rowsets are disjoint. Negative-coverage: the
    three other rowset_relation flags must NOT also fire (mutually
    exclusive — Codex major #6)."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (4, 5)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.rowset_relation_to_best == "disjoint"
    assert md.overlap_with_best == 0
    flags = set(md.miss_patterns)
    assert "disjoint_rowset" in flags
    for absent in ("partial_match_overlap", "agent_undercount", "agent_overcount"):
        assert absent not in flags, (
            f"{absent!r} contradicts disjoint_rowset; saw {flags}"
        )


def test_flag_partial_match_overlap(tmp_path: Path):
    """Agent and gold rowsets overlap but neither is subset of the
    other. Other rowset flags MUST NOT fire."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2, 3)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (2, 3, 4)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.rowset_relation_to_best == "overlapping"
    assert md.overlap_with_best > 0
    flags = set(md.miss_patterns)
    assert "partial_match_overlap" in flags
    for absent in ("disjoint_rowset", "agent_undercount", "agent_overcount"):
        assert absent not in flags, (
            f"{absent!r} contradicts partial_match_overlap; saw {flags}"
        )


def test_flag_agent_undercount(tmp_path: Path):
    """Agent is a strict subset of gold (over-restrictive filter).
    Other rowset flags MUST NOT fire."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2)",
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT id FROM t1 WHERE id IN (1, 2, 3, 4)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.rowset_relation_to_best == "strict_subset_of"
    flags = set(md.miss_patterns)
    assert "agent_undercount" in flags
    for absent in ("disjoint_rowset", "partial_match_overlap", "agent_overcount"):
        assert absent not in flags, (
            f"{absent!r} contradicts agent_undercount; saw {flags}"
        )


def test_flag_agent_overcount(tmp_path: Path):
    """Agent is a strict superset of gold (under-restrictive filter).
    Other rowset flags MUST NOT fire."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2, 3, 4)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.rowset_relation_to_best == "strict_superset_of"
    flags = set(md.miss_patterns)
    assert "agent_overcount" in flags
    for absent in ("disjoint_rowset", "partial_match_overlap", "agent_undercount"):
        assert absent not in flags, (
            f"{absent!r} contradicts agent_overcount; saw {flags}"
        )


def test_flag_never_asked_user_interactive_zero(tmp_path: Path):
    """Interactive benchmark, agent didn't ask. user_sim_n_asks=0 flags
    never_asked_user."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
        user_sim_n_asks=0,
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.user_sim_n_asks == 0
    assert "never_asked_user" in md.miss_patterns


def test_interactive_with_asks_skips_never_asked_user(tmp_path: Path):
    """Interactive benchmark, agent DID ask (n_asks > 0). never_asked_user
    is NOT in the flag list; the field is set to the non-zero count."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
        user_sim_n_asks=3,
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.user_sim_n_asks == 3
    assert "never_asked_user" not in md.miss_patterns


def test_one_shot_benchmark_skips_never_asked_user(tmp_path: Path):
    """One-shot benchmark (no user-sim): user_sim_n_asks kwarg = None.
    Field stored as None; never_asked_user NOT in flags."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
        user_sim_n_asks=None,
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.user_sim_n_asks is None
    assert "never_asked_user" not in md.miss_patterns


# ---------------------------------------------------------------------------
# Multi-flag fixture
# ---------------------------------------------------------------------------


def test_multi_flag_fixture_fires_every_applicable_flag(tmp_path: Path):
    """Single submission that simultaneously trips multiple rules:
    column projection differs, table set differs, agent has GROUP BY
    that gold doesn't, agent has LIMIT gold doesn't, user_sim_n_asks=0.
    Assert ALL five flags appear in miss_patterns (flags are NOT
    first-match-wins)."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql=(
            "SELECT id, name FROM t2 GROUP BY id, name LIMIT 2"
        ),
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 ORDER BY id"]),
        ],
        user_sim_n_asks=0,
    )
    md = verdict.miss_diagnostics
    assert md is not None
    flags = set(md.miss_patterns)
    for required in (
        "wrong_table_set",
        "aggregation_shape_mismatch",
        "limit_presence_mismatch",
        "never_asked_user",
        # Agent projects 2 cols (id, name) vs gold's 1 col (id) —
        # arity mismatch fires the load-bearing column-count flag.
        "column_count_mismatch",
    ):
        assert required in flags, (
            f"expected {required!r} in {flags} (multi-flag scenario)"
        )
    # When counts differ, the order check is short-circuited.
    assert "column_order_mismatch" not in flags


def test_multi_flag_fixture_with_sql_parse_error(tmp_path: Path):
    """Codex minor #9 — second multi-flag fixture covering the planned
    combination: empty_agent_result + wrong_table_set + never_asked_user
    + sql_parse_error. Agent SQL is unparseable garbage; the cascade
    falls through to all-False; multiple flags fire independently."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="NOT A VALID SQL AT ALL ;;",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 ORDER BY id"]),
        ],
        user_sim_n_asks=0,
    )
    md = verdict.miss_diagnostics
    assert md is not None
    flags = set(md.miss_patterns)
    for required in (
        "sql_parse_error",
        "empty_agent_result",
        "never_asked_user",
        "sql_execution_error",  # unparseable SQL also fails to execute
    ):
        assert required in flags, (
            f"expected {required!r} in {flags} "
            f"(multi-flag parse-error scenario)"
        )


# ---------------------------------------------------------------------------
# Best-overlap variant selection
# ---------------------------------------------------------------------------


def test_best_overlap_picks_alt_variant_over_primary(tmp_path: Path):
    """Same-column-shape rowsets; primary has 0 overlap with agent,
    alt has strictly higher (non-zero) overlap. Cascade fails because
    neither variant matches strictly. best_variant_id MUST be 'alt' —
    a buggy implementation that always picks the primary would fail
    this exact equality check (Codex critical)."""
    db = _build_db(tmp_path)
    # Agent: ids {1, 2, 3} (1-col)
    # Primary: ids {7, 8, 9} (1-col) — overlap 0
    # Alt:     ids {3, 4, 5} (1-col) — overlap 1 (the row (3,))
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2, 3) ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (7, 8, 9)"]),
            ("alt", False, ["SELECT id FROM t1 WHERE id IN (3, 4, 5)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.best_variant_id == "alt"
    assert md.overlap_with_best == 1
    assert md.best_variant_row_count == 3


def test_best_overlap_tie_break_prefers_primary(tmp_path: Path):
    """Both variants have IDENTICAL non-zero overlap with the agent.
    Tie-break MUST pick the primary. Zero-overlap fixtures (Codex
    major #2) would pass even if the implementation never computes
    overlap, so the agent here must actually overlap both variants
    by the same non-zero amount."""
    db = _build_db(tmp_path)
    # Agent rows {1, 2, 3}; both variants share exactly row (1,) with agent.
    # Overlap counts: primary=1, alt=1.
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2, 3) ORDER BY id",
        audited_sol_sql_per_variant=[
            ("alt", False, ["SELECT id FROM t1 WHERE id IN (1, 4)"]),
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 5)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.overlap_with_best == 1
    assert md.best_variant_id == "primary"


def test_best_overlap_tie_break_alphabetical_when_no_primary(tmp_path: Path):
    """No variant has primary=True; both have identical NON-ZERO
    overlap. Lexicographically smallest variant_id wins (Codex
    major #2: use non-zero overlap so the test pins tie-break
    behaviour, not just default-ordering on disjoint sets)."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (1, 2, 3) ORDER BY id",
        audited_sol_sql_per_variant=[
            ("zeta", False, ["SELECT id FROM t1 WHERE id IN (1, 4)"]),
            ("alpha", False, ["SELECT id FROM t1 WHERE id IN (1, 5)"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.overlap_with_best == 1
    assert md.best_variant_id == "alpha"


# ---------------------------------------------------------------------------
# Multiset (bag) semantics
# ---------------------------------------------------------------------------


def test_multiset_overlap_counts_duplicates(tmp_path: Path):
    """Agent returns ('x',) three times; gold returns ('x',) five times.
    Multiset overlap = min(3, 5) = 3. Set overlap would be 1; the
    implementation must use bag semantics."""
    db = _build_db(tmp_path)
    # Use UNION ALL to produce duplicate rows deterministically.
    verdict = _grade(
        db=db,
        submitted_sql=(
            "SELECT 'x' AS v FROM t1 WHERE id <= 3"  # 3 rows of ('x',)
        ),
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT 'x' AS v FROM t1 WHERE id <= 5"]),  # 5 rows of ('x',)
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_row_count == 3
    assert md.best_variant_row_count == 5
    # bag intersection = min(3,5) per cell
    assert md.overlap_with_best == 3


def test_multiset_subset_detection_with_duplicates(tmp_path: Path):
    """Agent rows ⊂ gold rows as MULTISETS (every count in agent ≤
    count in gold, at least one strict)."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT 'x' AS v FROM t1 WHERE id <= 2",  # ('x',)×2
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT 'x' AS v FROM t1 WHERE id <= 4"]),  # ('x',)×4
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.rowset_relation_to_best == "strict_subset_of"


def test_multiset_superset_detection_with_duplicates(tmp_path: Path):
    """Symmetric of the subset test (Codex major #3): agent has MORE
    copies of the duplicate row than gold. Bag semantics MUST flag
    strict_superset_of, not 'overlapping' or 'equal_rowset' that a
    plain-set implementation would emit."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT 'x' AS v FROM t1 WHERE id <= 5",  # ('x',)×5
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT 'x' AS v FROM t1 WHERE id <= 3"]),  # ('x',)×3
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_row_count == 5
    assert md.best_variant_row_count == 3
    assert md.rowset_relation_to_best == "strict_superset_of"
    assert "agent_overcount" in md.miss_patterns
    assert "agent_undercount" not in md.miss_patterns


# ---------------------------------------------------------------------------
# CTE alias exclusion
# ---------------------------------------------------------------------------


def test_cte_aliases_excluded_from_tables(tmp_path: Path):
    """SQL with a WITH clause: the CTE name should NOT appear in
    agent_tables_referenced. Only the underlying base table should."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql=(
            "WITH high AS (SELECT id FROM t1 WHERE id > 3) "
            "SELECT id FROM high ORDER BY id"
        ),
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT id FROM t2 ORDER BY id"]),  # different table → flags wrong_table_set
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_sql_parse_ok is True
    assert md.agent_tables_referenced == ["t1"]
    assert "high" not in (md.agent_tables_referenced or [])


def test_derived_table_aliases_excluded_from_tables(tmp_path: Path):
    """`SELECT … FROM (SELECT id FROM t1) AS sub` — `sub` is a derived-
    table alias and must NOT appear in agent_tables_referenced."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql=(
            "SELECT id FROM (SELECT id FROM t1 WHERE id > 3) AS sub "
            "ORDER BY id"
        ),
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t2 ORDER BY id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_tables_referenced == ["t1"]
    assert "sub" not in (md.agent_tables_referenced or [])


def test_cte_aliases_excluded_from_best_variant_tables(tmp_path: Path):
    """Codex major #8 — the alias-exclusion rule must apply to BOTH
    sides. Best-variant SQL with a CTE must also report only base
    tables in `best_variant_tables_referenced`."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t2 ORDER BY id",  # different base table
        audited_sol_sql_per_variant=[
            ("primary", True, [
                "WITH high AS (SELECT id FROM t1 WHERE id > 3) "
                "SELECT id FROM high ORDER BY id",
            ]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.best_variant_sql_parse_ok is True
    assert md.best_variant_tables_referenced == ["t1"]
    assert "high" not in (md.best_variant_tables_referenced or [])


def test_derived_table_aliases_excluded_from_best_variant_tables(tmp_path: Path):
    """Mirror for derived-table aliases on the best-variant side."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t2 ORDER BY id",
        audited_sol_sql_per_variant=[
            ("primary", True, [
                "SELECT id FROM (SELECT id FROM t1 WHERE id > 3) AS sub "
                "ORDER BY id",
            ]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.best_variant_tables_referenced == ["t1"]
    assert "sub" not in (md.best_variant_tables_referenced or [])


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------


def test_tables_referenced_lists_sorted_alphabetically(tmp_path: Path):
    """Both agent_tables_referenced and best_variant_tables_referenced
    are alphabetically sorted before persist, regardless of FROM/JOIN
    order in the source SQL. Force rowset divergence with a filter on
    the gold side so the cascade actually reaches diagnostics."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql=(
            "SELECT t2.id FROM t2 JOIN t1 ON t1.id = t2.id ORDER BY t2.id"
        ),
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT t1.id FROM t1 JOIN t2 ON t1.id = t2.id "
              "WHERE t1.id > 1 ORDER BY t1.id"]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.agent_tables_referenced == sorted(md.agent_tables_referenced or [])
    assert md.best_variant_tables_referenced == sorted(
        md.best_variant_tables_referenced or [],
    )
    assert md.agent_tables_referenced == ["t1", "t2"]
    assert md.best_variant_tables_referenced == ["t1", "t2"]
    assert md.table_set_match is True


def test_miss_patterns_sorted_alphabetically(tmp_path: Path):
    """miss_patterns is sorted alphabetically before persist."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t2 WHERE id IN (1, 2) LIMIT 1",
        audited_sol_sql_per_variant=[
            ("primary", True,
             ["SELECT id FROM t1 WHERE id > 999"]),  # empty
        ],
        user_sim_n_asks=0,
    )
    md = verdict.miss_diagnostics
    assert md is not None
    assert md.miss_patterns == sorted(md.miss_patterns)
    assert len(md.miss_patterns) >= 2  # multiple flags should fire


# ---------------------------------------------------------------------------
# FailureClassification integration
# ---------------------------------------------------------------------------


def test_failure_classification_primary_is_agent_miss(tmp_path: Path):
    """End-to-end: when grade_in_place builds a SubmissionAnnotation from
    a cascade-fail verdict, FailureClassification.primary is
    'agent_miss' (not 'other'); agent_at_fault=True; remediation='agent'."""
    from bird_interact_agents.eval.grade_in_place import _build_submission_annotation
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
        user_sim_n_asks=0,
    )
    ann = _build_submission_annotation(
        task_annotation=_task_annotation(),
        cascade=verdict,
        benchmark="mini-interact",
        run_id="test-run",
        trajectory_path="rows/alien_1/attempt-1.json",
        predicted_row_count=None,
        duration_s=None,
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        n_agent_turns=None,
        n_ask_user_calls=None,
    )
    assert ann.failure_classification.primary == "agent_miss"
    assert ann.failure_classification.agent_at_fault is True
    assert ann.failure_classification.remediation_target == "agent"


def test_failure_classification_details_mentions_strict_miss(tmp_path: Path):
    """The details string is free-form for humans; downstream consumers
    must use miss_diagnostics.miss_patterns for structured signals.
    Assert only that 'strict miss' appears (weak content check)."""
    from bird_interact_agents.eval.grade_in_place import _build_submission_annotation
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
    )
    ann = _build_submission_annotation(
        task_annotation=_task_annotation(),
        cascade=verdict,
        benchmark="mini-interact",
        run_id="test-run",
        trajectory_path="rows/alien_1/attempt-1.json",
        predicted_row_count=None,
        duration_s=None,
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        n_agent_turns=None,
        n_ask_user_calls=None,
    )
    assert "strict miss" in ann.failure_classification.details.lower()


def test_submission_evaluation_carries_miss_diagnostics(tmp_path: Path):
    """grade_in_place plumbs cascade.miss_diagnostics → ev.miss_diagnostics."""
    from bird_interact_agents.eval.grade_in_place import _build_submission_annotation
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id IN (4, 5)",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
    )
    ann = _build_submission_annotation(
        task_annotation=_task_annotation(),
        cascade=verdict,
        benchmark="mini-interact",
        run_id="test-run",
        trajectory_path="rows/alien_1/attempt-1.json",
        predicted_row_count=None,
        duration_s=None,
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        n_agent_turns=None,
        n_ask_user_calls=None,
    )
    assert ann.evaluation.miss_diagnostics is not None
    assert ann.evaluation.miss_diagnostics.best_variant_id == "primary"


# ---------------------------------------------------------------------------
# Back-compat: old SubmissionEvaluation JSON without miss_diagnostics
# ---------------------------------------------------------------------------


def test_back_compat_old_submission_evaluation_validates():
    """Existing SubmissionAnnotation JSON files (53 on disk) lack the
    new miss_diagnostics field. Optional[MissDiagnostics] = None must
    let them validate; extra='forbid' does NOT reject missing optional
    fields with defaults."""
    from bird_interact_agents.eval.annotation_schema import SubmissionEvaluation
    legacy = {
        "phase1_against_original_gold": "pass",
        "phase1_against_audited_primary": "pass",
        "phase1_against_any_audited_variant": "pass",
        "phase1_against_variants": [],
        "correct_up_to_tie_order": True,
        "novel_reading_judgment": None,
        "correct_under_numeric_epsilon": True,
        "correct_under_trailing_whitespace": True,
        "correct_under_column_order": True,
        "correct_under_case_fold": True,
        "numeric_epsilon": 1e-6,
        "verdict": "correct",
        "matched_variant_id": "primary",
        "rationale": "",
    }
    ev = SubmissionEvaluation.model_validate(legacy)
    assert ev.miss_diagnostics is None
    # Round-trip the new shape too — must serialize miss_diagnostics: None
    dumped = ev.model_dump()
    assert "miss_diagnostics" in dumped
    assert dumped["miss_diagnostics"] is None


# ---------------------------------------------------------------------------
# Codex r8: multi-statement gold (CREATE TEMP + SELECT, DDL prelude, etc.)
# MUST NOT crash diagnostics. The pre-fix assertion bubbled an
# AssertionError out of ``_compute_miss_diagnostics`` and the cloud +
# local fallbacks dropped the structured ``miss_patterns`` for the
# entire row. The fix uses the LAST statement of the gold's
# ``audited_sol_sql`` list for sqlglot parsing — that's the SELECT
# query under diagnosis; the setup statements don't constrain miss
# patterns.
# ---------------------------------------------------------------------------


def test_multi_statement_audited_gold_uses_last_for_sql_signals(
    tmp_path: Path,
):
    """A 2-statement audited gold (DDL setup + SELECT) must compute
    diagnostics successfully; the sqlglot-derived signals come from
    the SELECT statement, NOT the DDL."""
    db = _build_db(tmp_path)
    # Agent reads from t1; gold's "real" reading is the SELECT against
    # a temp table built from t1. Both reference t1 via the SELECT.
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id = 9999",  # disjoint -> miss
        audited_sol_sql_per_variant=[
            ("primary", True, [
                "CREATE TEMP TABLE tmp AS SELECT id FROM t1",
                "SELECT id FROM tmp",
            ]),
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None, (
        "multi-statement audited gold must NOT crash diagnostics; "
        "got miss_diagnostics=None"
    )
    # The best-variant SQL parse used the SELECT (the LAST statement),
    # not the CREATE TEMP TABLE (which sqlglot would also parse but
    # would yield empty / wrong table extraction).
    assert md.best_variant_sql_parse_ok is True
    assert md.best_variant_tables_referenced == ["tmp"], (
        f"best_variant_tables_referenced must be parsed from the LAST "
        f"statement (SELECT id FROM tmp), got "
        f"{md.best_variant_tables_referenced!r}"
    )


def test_multi_statement_original_gold_does_not_crash_diagnostics(
    tmp_path: Path,
):
    """Original gold can also carry a multi-statement list (the
    same DDL + SELECT shape). Diagnostics must still compute — the
    original gold is only used for the ``original_gold_row_count``
    field, not sqlglot parsing — so any number of statements is OK
    as long as the cascade ran with them."""
    db = _build_db(tmp_path)
    verdict = _grade(
        db=db,
        submitted_sql="SELECT id FROM t1 WHERE id = 9999",
        audited_sol_sql_per_variant=[
            ("primary", True, ["SELECT id FROM t1 WHERE id IN (1, 2)"]),
        ],
        original_sol_sql=[
            "CREATE TEMP TABLE tmp AS SELECT id FROM t1",
            "SELECT id FROM tmp",
        ],
    )
    md = verdict.miss_diagnostics
    assert md is not None, (
        "multi-statement original gold must not crash diagnostics"
    )
