"""DEV-1515: end-to-end cascade orchestration tests for tolerant_grader.

These exercise ``grade_submission`` end-to-end against a fake executor
(no real SQLite) and verify the 8-row cascade (N1..N8), monotonicity,
LLM-judge gating + cache, and the missing-annotation graceful default.

LLM-judge tests stay mechanical per the project convention: we never
assert on prompt content, only on cache key, timeout fall-through, and
SubmissionEvaluation fields the grader produces.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers — fake executor + audited-gold rows + task annotation fixtures
# ---------------------------------------------------------------------------


def _audited_row(
    *, instance_id: str, variant_id: str, primary: bool,
    audited_sol_sql: List[str], db: str = "alien",
) -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": db,
        "benchmark": "mini_interact",
        "audit_status": "edited",
        "original_sol_sql": ["SELECT original FROM t"],
        "audited_sol_sql": audited_sol_sql,
        "variant_id": variant_id,
        "primary": primary,
        "changes": [],
        "reasoning_summary": "",
        "skill_version": "audit-gold-sql/1.0",
        "audited_at": "2026-05-30T00:00:00+00:00",
    }


def _make_task_annotation(
    *,
    verdict: str = "sufficient",
    evaluator_prompt: Optional[str] = None,
    instance_id: str = "alien_1",
):
    from bird_interact_agents.eval import (
        AuditedGoldRef,
        GoldVariantRef,
        MetadataSufficiency,
        TaskAnnotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance

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
                variant_id="primary",
                interpretation="x",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id=instance_id,
                    variant_id="primary",
                ),
            ),
        ],
        evaluator_prompt=evaluator_prompt,
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=instance_id,
        ),
    )


class FakeExecutor:
    """Returns canned (rows, column_names) for each SQL string, in order.

    `responses` is a dict SQL→(rows, column_names). Any SQL not in the
    dict raises — tests must enumerate every expected SQL.
    """
    def __init__(self, responses):
        self.responses = responses
        self.calls: List[str] = []

    def __call__(self, sql, *, db_path, conn):
        self.calls.append(sql)
        if sql not in self.responses:
            raise AssertionError(f"unexpected SQL: {sql!r}")
        return self.responses[sql]


# ---------------------------------------------------------------------------
# N1..N3 strict cascade
# ---------------------------------------------------------------------------


def test_n1_passes_when_predicted_matches_original_gold():
    """Predicted matches original gold but NOT audited primary. The
    cascade is monotone (N1 pass ⇒ N2+ pass via enforce_monotone_cascade),
    so per-task verdict is True from N1 onward. The aggregate delta
    D2/D3 stays 0 for this row (audited didn't add a new pass)."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "SELECT 1"
    original_gold = "SELECT original FROM t"
    audited = "SELECT audited FROM t"
    executor = FakeExecutor({
        submitted: ([(1,)], ["a"]),
        original_gold: ([(1,)], ["a"]),
        audited: ([(99,)], ["a"]),  # different — only original matches
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    assert verdict.n1_original_gold is True
    # N2..N8 all True via monotone enforcement.
    assert verdict.n2_audited_primary is True
    # The variant_matches diagnostic block still surfaces the audited
    # mismatch (it's per-variant Tier 2 info, not gated by the cascade).
    assert verdict.variant_matches[0].informational is not None
    assert verdict.variant_matches[0].informational.rowset_relation == "disjoint"


def test_n3_passes_for_non_primary_variant_match():
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "SELECT predicted"
    original_gold = "SELECT original"
    primary_sql = "SELECT primary_audit"
    alt_sql = "SELECT alt_audit"
    executor = FakeExecutor({
        submitted: ([(42,)], ["x"]),
        original_gold: ([(1,)], ["x"]),
        primary_sql: ([(2,)], ["x"]),
        alt_sql: ([(42,)], ["x"]),
    })
    ann = _make_task_annotation()
    gold_rows = [
        _audited_row(
            instance_id="alien_1", variant_id="primary", primary=True,
            audited_sol_sql=[primary_sql],
        ),
        _audited_row(
            instance_id="alien_1", variant_id="alt", primary=False,
            audited_sol_sql=[alt_sql],
        ),
    ]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    assert verdict.n1_original_gold is False
    assert verdict.n2_audited_primary is False
    assert verdict.n3_any_audited_variant is True
    assert verdict.matched_variant_id == "alt"


# ---------------------------------------------------------------------------
# N4 — tie-order (uses original_gold's ORDER BY per spec)
# ---------------------------------------------------------------------------


def test_n4_passes_when_only_tie_order_differs():
    """Predicted has ties at a='A' reordered relative to gold. With the
    set-equality N3 comparator (back-compat with today's ex_base default),
    set-eq passes for in-bucket reorders so N3 already True. N4
    (bucket-by-ORDER-BY) also passes — this is the spec-mandated cascade
    row, even though in the set-eq world it coincides with N3 in the
    no-duplicate case.

    The dedicated comparator-level tie-order test in
    ``test_tolerant_grader_comparators.py::test_n4_reordered_within_bucket_passes``
    exercises the bucketing logic directly.
    """
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "SELECT * FROM t"
    original_gold = "SELECT a, b FROM t ORDER BY a"
    audited = "SELECT a, b FROM t ORDER BY a"
    executor = FakeExecutor({
        submitted: ([("A", 2), ("A", 1), ("B", 9)], ["a", "b"]),
        original_gold: ([("A", 1), ("A", 2), ("B", 9)], ["a", "b"]),
        audited: ([("A", 1), ("A", 2), ("B", 9)], ["a", "b"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    # In the set-eq world N3 already passes for in-bucket reorders.
    assert verdict.n3_any_audited_variant is True
    assert verdict.n4_tie_order is True


def test_n4_uses_original_gold_orderby_not_variant_orderby():
    """N4 bucketing is sourced from the ORIGINAL gold's ORDER BY clause
    (locked simplification), not each variant's. Construct a case where
    the original gold has ORDER BY but the variant's audited SQL doesn't
    — bucketing must STILL follow the original gold."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    # Original gold has ORDER BY on column 0; audited variant doesn't.
    original_gold = "SELECT a, b FROM t ORDER BY a"
    audited = "SELECT a, b FROM t"
    executor = FakeExecutor({
        # Predicted: ties on a='A' reordered; cross-bucket order preserved.
        submitted: ([("A", 2), ("A", 1), ("B", 9)], ["a", "b"]),
        original_gold: ([("A", 1), ("A", 2), ("B", 9)], ["a", "b"]),
        audited: ([("A", 1), ("A", 2), ("B", 9)], ["a", "b"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    # Using ORIGINAL gold's ORDER BY: bucket by column 0 → all rows
    # share the bucket-equivalence at ('A', _) → set-equality passes.
    assert verdict.n4_tie_order is True


def test_n4_collapses_to_n3_when_no_orderby():
    """No ORDER BY in original gold → N4 == N3."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "SELECT a FROM t"  # no ORDER BY
    audited = "SELECT a FROM t"
    executor = FakeExecutor({
        submitted: ([(2,), (1,)], ["a"]),
        original_gold: ([(1,), (2,)], ["a"]),
        audited: ([(1,), (2,)], ["a"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    # No ORDER BY → ordering doesn't matter; sets are equal so N3 passes.
    assert verdict.n3_any_audited_variant is True
    assert verdict.n4_tie_order is True


# ---------------------------------------------------------------------------
# N5 — LLM judge (only fires for verdict=='insufficient' AND no N4 pass)
# ---------------------------------------------------------------------------


class FakeLLMJudge:
    def __init__(self, accept: Optional[bool]):
        self.accept = accept
        self.calls = 0

    def judge(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        return self.accept


def test_n5_does_not_fire_for_sufficient_verdict():
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([(1,)], ["a"]),
        original_gold: ([(2,)], ["a"]),
        audited: ([(2,)], ["a"]),
    })
    ann = _make_task_annotation(
        verdict="sufficient", evaluator_prompt="rules",
    )
    judge = FakeLLMJudge(accept=True)
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
        llm_judge=judge,
    )
    assert verdict.n5_llm_judge is False
    assert judge.calls == 0  # the gate held — no call paid


def test_n5_fires_only_for_insufficient_verdict_when_no_variant_matched():
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([(1,)], ["a"]),
        original_gold: ([(2,)], ["a"]),
        audited: ([(2,)], ["a"]),
    })
    ann = _make_task_annotation(
        verdict="insufficient", evaluator_prompt="rules",
    )
    judge = FakeLLMJudge(accept=True)
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
        llm_judge=judge,
    )
    assert verdict.n5_llm_judge is True
    assert judge.calls == 1


def test_n5_does_not_invoke_judge_when_n4_already_passes_via_tie_order():
    """Gate semantics: N5 fires ONLY when verdict=='insufficient' AND
    NO variant matched at N4. If N4 passes via tie-order on a sufficient
    task, the judge must not be called even with a non-None judge."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "SELECT a FROM t ORDER BY a"
    audited = "SELECT a FROM t ORDER BY a"
    executor = FakeExecutor({
        # N3 strict fails (different in-bucket order); N4 tie-order passes.
        submitted: ([("A", 2), ("A", 1)], ["a", "b"]),
        original_gold: ([("A", 1), ("A", 2)], ["a", "b"]),
        audited: ([("A", 1), ("A", 2)], ["a", "b"]),
    })
    ann = _make_task_annotation(
        verdict="insufficient", evaluator_prompt="rules",
    )
    judge = FakeLLMJudge(accept=False)
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
        llm_judge=judge,
    )
    assert verdict.n4_tie_order is True
    assert verdict.n5_llm_judge is True
    assert judge.calls == 0, (
        "N5 must not invoke the judge when N4 already passes"
    )


def test_n5_monotone_never_takes_away_n4_pass():
    """If N4 already passes, N5 must remain True even if the LLM judge
    would have rejected — the cascade is monotone by construction."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([(1,)], ["a"]),
        original_gold: ([(1,)], ["a"]),
        audited: ([(1,)], ["a"]),
    })
    ann = _make_task_annotation(
        verdict="insufficient", evaluator_prompt="rules",
    )
    judge = FakeLLMJudge(accept=False)  # would say no, but cascade monotone
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
        llm_judge=judge,
    )
    # Deterministic tiers all pass → N5 must stay True regardless of judge.
    assert verdict.n4_tie_order is True
    assert verdict.n5_llm_judge is True


def test_n5_timeout_falls_through_to_none():
    """LLMJudge returning None (timeout/error) leaves N5 as the previous
    cascade value (no novel reading accepted)."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([(1,)], ["a"]),
        original_gold: ([(2,)], ["a"]),
        audited: ([(2,)], ["a"]),
    })
    ann = _make_task_annotation(
        verdict="insufficient", evaluator_prompt="rules",
    )
    judge = FakeLLMJudge(accept=None)  # timeout
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
        llm_judge=judge,
    )
    assert verdict.n4_tie_order is False
    assert verdict.n5_llm_judge is False
    assert verdict.novel_reading_judgment is None


# ---------------------------------------------------------------------------
# N6/N7/N8 — cell-level relaxations stacked on N5
# ---------------------------------------------------------------------------


def test_n6_numeric_epsilon_lifts_failing_n5_to_pass():
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([(1.0000001,)], ["v"]),
        original_gold: ([(1.0,)], ["v"]),
        audited: ([(1.0,)], ["v"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
        epsilon=1e-6,
    )
    assert verdict.n5_llm_judge is False  # sufficient + no llm_judge
    assert verdict.n6_numeric_epsilon is True
    assert verdict.n7_trailing_whitespace is True  # monotone
    assert verdict.n8_column_order is True  # monotone


def test_n7_trailing_whitespace_lifts_failing_n6_to_pass():
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([("High Income ",)], ["bracket"]),
        original_gold: ([("High Income",)], ["bracket"]),
        audited: ([("High Income",)], ["bracket"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    assert verdict.n6_numeric_epsilon is False
    assert verdict.n7_trailing_whitespace is True


def test_n9_case_fold_lifts_failing_n8_to_pass():
    """A case-only mismatch in cell strings fails N7/N8 but lifts at N9."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([("HIGH",), ("Low",)], ["bracket"]),
        original_gold: ([("high",), ("low",)], ["bracket"]),
        audited: ([("high",), ("low",)], ["bracket"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    assert verdict.n7_trailing_whitespace is False
    assert verdict.n8_column_order is False
    assert verdict.n9_case_fold is True


def test_n8_column_order_uses_column_metadata():
    """N8 needs cursor.description-style column names; ensure the
    executor's returned `column_names` parameter is the path used."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    executor = FakeExecutor({
        submitted: ([("foo", 1)], ["name", "id"]),
        original_gold: ([(1, "foo")], ["id", "name"]),
        audited: ([(1, "foo")], ["id", "name"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    # Without name-alignment, tuples don't match (positional).
    assert verdict.n7_trailing_whitespace is False
    # With name-alignment, columns align → True.
    assert verdict.n8_column_order is True


# ---------------------------------------------------------------------------
# Cascade monotonicity — synthetic matrix
# ---------------------------------------------------------------------------


def test_cascade_is_monotone_for_every_possible_pass_pattern():
    """Whatever the comparators return, the persisted CascadeVerdict must
    satisfy N1 ≤ N2 ≤ N3 ≤ N4 ≤ N5 ≤ N6 ≤ N7 ≤ N8 (each step admits a
    superset of passes from the previous).

    Drive the property through a synthetic matrix: any task that passes
    at level N must also report `pass` at level N+1.
    """
    from bird_interact_agents.eval.tolerant_grader import (
        enforce_monotone_cascade,
    )

    fields = [
        "n1_original_gold", "n2_audited_primary", "n3_any_audited_variant",
        "n4_tie_order", "n5_llm_judge",
        "n6_numeric_epsilon", "n7_trailing_whitespace", "n8_column_order",
        "n9_case_fold",
    ]
    n = len(fields)
    for mask in range(2 ** n):
        raw = {fields[i]: bool((mask >> i) & 1) for i in range(n)}
        enforced = enforce_monotone_cascade(raw)
        prev = False
        for f in fields:
            cur = enforced[f]
            # Monotone: once True, every subsequent level stays True.
            if prev:
                assert cur is True, (
                    f"monotone broken at {f}: prev True, got {cur} "
                    f"with raw={raw}"
                )
            prev = cur


# ---------------------------------------------------------------------------
# Missing-annotation graceful default — no write, cascade collapses to N1
# ---------------------------------------------------------------------------


def test_missing_annotation_collapses_cascade_to_n1(monkeypatch, tmp_path):
    from bird_interact_agents.eval.tolerant_grader import grade_submission
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )

    submitted = "S"
    original_gold = "G"
    executor = FakeExecutor({
        submitted: ([(1,)], ["a"]),
        original_gold: ([(1,)], ["a"]),
    })
    ann = implicit_task_annotation(
        instance_id="alien_99",
        selected_database="alien",
        benchmark="mini-interact",
        amb_user_query="x",
    )
    # No audited-gold rows on disk for this instance — the implicit
    # contract is "single variant == original gold".
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=[],
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    assert verdict.n1_original_gold is True
    # Cascade collapses: N2 == N3 == N1 (no separate audited primary).
    assert verdict.n2_audited_primary is True
    assert verdict.n3_any_audited_variant is True


# ---------------------------------------------------------------------------
# LLM-judge cache — keyed by content hash, not run-id
# ---------------------------------------------------------------------------


def test_llm_judge_cache_hit_avoids_second_call(tmp_path):
    from bird_interact_agents.eval.tolerant_grader import CachedLLMJudge

    class CountingInner:
        def __init__(self):
            self.calls = 0
        def judge(self, **kwargs):
            self.calls += 1
            return True

    inner = CountingInner()
    cache_path = tmp_path / "llm_judge_cache.json"
    cached = CachedLLMJudge(inner=inner, cache_path=cache_path)

    payload = dict(
        evaluator_prompt="rules",
        gold_variants_summary=[{"variant_id": "primary"}],
        metadata_anchors=[],
        submitted_sql="SELECT 1",
        predicted_rows_head=[(1,)],
        annotation_content_hash="abc",
        gold_variants_content_hash="def",
    )

    out1 = cached.judge(**payload)
    out2 = cached.judge(**payload)
    assert out1 is True and out2 is True
    assert inner.calls == 1  # second call short-circuits via cache


def test_llm_judge_cache_key_changes_when_annotation_hash_changes(tmp_path):
    from bird_interact_agents.eval.tolerant_grader import CachedLLMJudge

    class CountingInner:
        def __init__(self):
            self.calls = 0
        def judge(self, **kwargs):
            self.calls += 1
            return True

    inner = CountingInner()
    cache_path = tmp_path / "llm_judge_cache.json"
    cached = CachedLLMJudge(inner=inner, cache_path=cache_path)

    base = dict(
        evaluator_prompt="rules",
        gold_variants_summary=[],
        metadata_anchors=[],
        submitted_sql="SELECT 1",
        predicted_rows_head=[(1,)],
        gold_variants_content_hash="def",
    )
    cached.judge(annotation_content_hash="abc", **base)
    cached.judge(annotation_content_hash="abc2", **base)  # changed annotation
    assert inner.calls == 2  # different keys; no cache hit


def test_llm_judge_cache_key_changes_when_gold_variants_hash_changes(tmp_path):
    """Different gold-variants content ⇒ different cache key, even if
    everything else is identical. New variants must NOT acquit via stale
    cache."""
    from bird_interact_agents.eval.tolerant_grader import CachedLLMJudge

    class CountingInner:
        calls = 0
        def judge(self, **kwargs):
            self.calls += 1
            return True

    inner = CountingInner()
    cached = CachedLLMJudge(
        inner=inner, cache_path=tmp_path / "c.json",
    )
    base = dict(
        evaluator_prompt="rules", gold_variants_summary=[],
        metadata_anchors=[], submitted_sql="X",
        predicted_rows_head=[], annotation_content_hash="a",
    )
    cached.judge(gold_variants_content_hash="g1", **base)
    cached.judge(gold_variants_content_hash="g2", **base)
    assert inner.calls == 2


def test_llm_judge_cache_key_changes_when_model_changes(tmp_path):
    """Same content, different model ⇒ different cache key. Caches must
    survive a model bump (Opus 4.7 → 4.8) without giving stale verdicts."""
    from bird_interact_agents.eval.tolerant_grader import CachedLLMJudge

    class CountingInner:
        def __init__(self, model_name):
            self.calls = 0
            self.model_name = model_name
        def judge(self, **kwargs):
            self.calls += 1
            return True

    cache_path = tmp_path / "c.json"
    inner1 = CountingInner(model_name="claude-opus-4-7")
    CachedLLMJudge(inner=inner1, cache_path=cache_path).judge(
        evaluator_prompt="r", gold_variants_summary=[], metadata_anchors=[],
        submitted_sql="X", predicted_rows_head=[],
        annotation_content_hash="a", gold_variants_content_hash="g",
    )
    inner2 = CountingInner(model_name="claude-opus-4-8")  # different
    CachedLLMJudge(inner=inner2, cache_path=cache_path).judge(
        evaluator_prompt="r", gold_variants_summary=[], metadata_anchors=[],
        submitted_sql="X", predicted_rows_head=[],
        annotation_content_hash="a", gold_variants_content_hash="g",
    )
    # The second call must hit the inner (different model in cache key).
    assert inner2.calls == 1


def test_llm_judge_cache_key_does_not_include_run_id(tmp_path):
    """Offline re-grade across runs MUST reuse cached verdicts when the
    content (annotation, gold, sql, model) is unchanged. Run-id is
    therefore NOT part of the key."""
    from bird_interact_agents.eval.tolerant_grader import CachedLLMJudge

    class CountingInner:
        def __init__(self):
            self.calls = 0
        def judge(self, **kwargs):
            self.calls += 1
            return True

    inner = CountingInner()
    cache_path = tmp_path / "c.json"
    # Two calls in different "run contexts" — content identical.
    for _ in range(2):
        CachedLLMJudge(inner=inner, cache_path=cache_path).judge(
            evaluator_prompt="r", gold_variants_summary=[],
            metadata_anchors=[], submitted_sql="X", predicted_rows_head=[],
            annotation_content_hash="a", gold_variants_content_hash="g",
        )
    assert inner.calls == 1  # second call is a cache hit


def test_llm_judge_cache_persists_across_process(tmp_path):
    """Caches live on disk so an offline re-grade can reuse cloud-side
    decisions."""
    from bird_interact_agents.eval.tolerant_grader import CachedLLMJudge

    class CountingInner:
        def __init__(self, val=True):
            self.calls = 0
            self.val = val
        def judge(self, **kwargs):
            self.calls += 1
            return self.val

    cache_path = tmp_path / "llm_judge_cache.json"
    # First process — populate cache.
    inner1 = CountingInner(val=True)
    CachedLLMJudge(inner=inner1, cache_path=cache_path).judge(
        evaluator_prompt="rules",
        gold_variants_summary=[],
        metadata_anchors=[],
        submitted_sql="X",
        predicted_rows_head=[],
        annotation_content_hash="a",
        gold_variants_content_hash="b",
    )
    # Second process — cache should be re-used; inner must not be hit.
    inner2 = CountingInner(val=False)  # different val to prove cache used
    out = CachedLLMJudge(inner=inner2, cache_path=cache_path).judge(
        evaluator_prompt="rules",
        gold_variants_summary=[],
        metadata_anchors=[],
        submitted_sql="X",
        predicted_rows_head=[],
        annotation_content_hash="a",
        gold_variants_content_hash="b",
    )
    assert out is True  # the cached `True`, not the new inner's `False`
    assert inner2.calls == 0


# ---------------------------------------------------------------------------
# Tier 2 informational — rowset relation, column diff, first divergent row
# ---------------------------------------------------------------------------


def test_tier2_rowset_relation_equal():
    from bird_interact_agents.eval.tolerant_grader import classify_rowset_relation

    assert classify_rowset_relation(
        pred=[(1,), (2,)], gold=[(2,), (1,)],
    ) == "equal_rowset"


def test_tier2_rowset_relation_strict_subset():
    from bird_interact_agents.eval.tolerant_grader import classify_rowset_relation

    assert classify_rowset_relation(
        pred=[(1,)], gold=[(1,), (2,)],
    ) == "strict_subset_of"


def test_tier2_rowset_relation_strict_superset():
    from bird_interact_agents.eval.tolerant_grader import classify_rowset_relation

    assert classify_rowset_relation(
        pred=[(1,), (2,)], gold=[(1,)],
    ) == "strict_superset_of"


def test_tier2_rowset_relation_overlapping():
    from bird_interact_agents.eval.tolerant_grader import classify_rowset_relation

    assert classify_rowset_relation(
        pred=[(1,), (2,)], gold=[(2,), (3,)],
    ) == "overlapping"


def test_tier2_rowset_relation_disjoint():
    from bird_interact_agents.eval.tolerant_grader import classify_rowset_relation

    assert classify_rowset_relation(
        pred=[(1,)], gold=[(2,)],
    ) == "disjoint"


def test_tier2_populated_on_grader_output():
    """Each variant in CascadeVerdict.variant_matches carries an
    informational sub-block with exact field values pinned."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    submitted = "S"
    original_gold = "G"
    audited = "A"
    # Predicted = (1, "x"); gold = (2, "x"). Same column count + names.
    # First divergent row index = 0; cell diff at column 0.
    executor = FakeExecutor({
        submitted: ([(1, "x")], ["id", "label"]),
        original_gold: ([(2, "x")], ["id", "label"]),
        audited: ([(2, "x")], ["id", "label"]),
    })
    ann = _make_task_annotation()
    gold_rows = [_audited_row(
        instance_id="alien_1", variant_id="primary", primary=True,
        audited_sol_sql=[audited],
    )]
    verdict = grade_submission(
        task_annotation=ann,
        audited_gold_rows=gold_rows,
        original_sol_sql=[original_gold],
        submitted_sql=submitted,
        db_path=Path("/dev/null"),
        conn=None,
        executor=executor,
    )
    assert verdict.variant_matches
    info = verdict.variant_matches[0].informational
    assert info is not None
    # Same column count + names + order ⇒ all three structural flags True.
    assert info.column_count_match is True
    assert info.column_name_match_case_insensitive is True
    assert info.column_order_match is True
    # Disjoint rowsets (pred=[(1,"x")], gold=[(2,"x")]).
    assert info.rowset_relation == "disjoint"
    # First (and only) divergent row is at index 0.
    assert info.first_divergent_row_index == 0
    # The cell-diff string must reference both values for human review.
    assert info.first_divergent_cell_diff is not None
    assert "1" in info.first_divergent_cell_diff
    assert "2" in info.first_divergent_cell_diff
