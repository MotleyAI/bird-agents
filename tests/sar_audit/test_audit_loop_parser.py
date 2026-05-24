"""Verdict-text parser: `Correctness:/Is_ambiguous:/Explanation:/Revised:` format."""

from __future__ import annotations

from bird_interact_agents.sar_audit.audit_loop import parse_verdict


def test_clean_verdict():
    text = """
Correctness: Yes, the SQL accurately answers the question.
Is_ambiguous: No, the question is unambiguous.
Explanation: All clauses are correct.
Revised:
"""
    v = parse_verdict(text)
    assert v.correctness_flag is True
    assert v.ambiguity_flag is False
    assert v.revised_sql is None
    assert v.revised_question is None
    assert "All clauses" in v.reasoning


def test_edited_verdict_with_revised_sql():
    text = """
Correctness: No, the query is missing an ORDER BY ASC clause.
Is_ambiguous: No
Explanation: The annotator omitted an explicit sort order.
Revised:
```sql
SELECT x FROM t ORDER BY x ASC LIMIT 1
```
"""
    v = parse_verdict(text)
    assert v.correctness_flag is False
    assert v.ambiguity_flag is False
    assert v.revised_sql is not None
    assert "SELECT x FROM t" in v.revised_sql
    assert "ORDER BY x ASC" in v.revised_sql


def test_ambiguous_with_revised_question():
    text = """
Correctness: Yes
Is_ambiguous: Yes, "smallest" could mean by value or by id.
Explanation: The question is ambiguous in ordering.
Revised: Did you mean smallest by value, or by id?
"""
    v = parse_verdict(text)
    assert v.ambiguity_flag is True
    assert v.revised_question is not None
    assert "smallest" in v.revised_question.lower()


def test_unrecoverable_no_revised():
    text = """
Correctness: No, but the intent is unclear.
Is_ambiguous: No
Explanation: Cannot determine annotator's intent without more info.
Revised:
"""
    v = parse_verdict(text)
    assert v.correctness_flag is False
    assert v.ambiguity_flag is False
    assert v.revised_sql is None
    assert v.revised_question is None


def test_revised_with_fenced_sql_and_question():
    text = """
Correctness: No
Is_ambiguous: Yes
Explanation: both wrong
Revised:
The question should clarify whether to include nulls.
```sql
SELECT x FROM t WHERE x IS NOT NULL ORDER BY x LIMIT 1
```
"""
    v = parse_verdict(text)
    assert v.correctness_flag is False
    assert v.ambiguity_flag is True
    assert v.revised_sql is not None
    assert "WHERE x IS NOT NULL" in v.revised_sql
    assert v.revised_question is not None
    assert "include nulls" in v.revised_question.lower()


def test_missing_sections_fail_closed():
    """Fail-closed default: empty / headerless text yields correctness=False
    so the row lands as `unrecoverable` instead of silently looking clean."""
    v = parse_verdict("nonsense text with no headers")
    assert v.correctness_flag is False  # fail-closed
    assert v.ambiguity_flag is False
    assert v.revised_sql is None
    # No SELECT/WITH in the body → "nonsense" is treated as the (absent)
    # revised section, which has no header, so revised_question is None.
    assert v.revised_question is None


def test_revised_only_question_no_fence():
    """A non-SQL-looking Revised section is treated as a question revision."""
    text = """
Correctness: Yes
Is_ambiguous: Yes
Explanation: x
Revised: Could you clarify what "wealthy" means in this context?
"""
    v = parse_verdict(text)
    assert v.revised_sql is None
    assert v.revised_question is not None
    assert "wealthy" in v.revised_question.lower()


def test_revised_only_sql_no_fence():
    """A Revised section that starts with SELECT/WITH is treated as SQL."""
    text = """
Correctness: No
Is_ambiguous: No
Explanation: missing distinct
Revised: SELECT DISTINCT x FROM t
"""
    v = parse_verdict(text)
    assert v.revised_sql is not None
    assert v.revised_sql.upper().startswith("SELECT DISTINCT")
    assert v.revised_question is None


def test_revised_with_label_prefix_no_fence():
    """`Revised SQL:` label prefix → strip and treat remainder as SQL.

    The bare label can appear because the model writes the analyse_result
    inside the upstream `Revised:` section as `Revised SQL:\\nWITH ...`.
    The old parser missed this and treated the whole block as a question.
    """
    text = """
Correctness: No
Is_ambiguous: No
Explanation: missing CTE
Revised: Revised SQL:
WITH x AS (SELECT 1) SELECT * FROM x
"""
    v = parse_verdict(text)
    assert v.revised_sql is not None
    assert v.revised_sql.upper().startswith("WITH")
    assert v.revised_question is None


def test_revised_with_sql_label_uppercase():
    """`SQL:` label (no `Revised` word) — strip too."""
    text = """
Correctness: No
Is_ambiguous: No
Explanation: x
Revised:
SQL: SELECT 1
"""
    v = parse_verdict(text)
    assert v.revised_sql is not None
    assert v.revised_sql.upper().startswith("SELECT")
    assert v.revised_question is None


def test_revised_prose_then_inline_select():
    """Prose followed by SELECT in the same paragraph → split into question + SQL."""
    text = """
Correctness: No
Is_ambiguous: Yes
Explanation: x
Revised: The question should specify the sort order. SELECT x FROM t ORDER BY x
"""
    v = parse_verdict(text)
    assert v.revised_sql is not None
    assert "SELECT x FROM t" in v.revised_sql
    assert v.revised_question is not None
    assert "sort order" in v.revised_question.lower()


def test_revised_prose_with_with_in_words_not_classified_as_sql():
    """`with` inside prose like 'with the hint' must NOT be parsed as SQL."""
    text = """
Correctness: Yes
Is_ambiguous: Yes
Explanation: x
Revised: Please clarify what you mean — perhaps align it with the hint about ordering.
"""
    v = parse_verdict(text)
    assert v.revised_sql is None  # prose-only, no SQL clause boundary
    assert v.revised_question is not None
    assert "align" in v.revised_question.lower()


def test_revised_prose_with_select_in_words_not_classified_as_sql():
    """`select` as an English verb in prose must NOT trigger SQL extraction."""
    text = """
Correctness: Yes
Is_ambiguous: Yes
Explanation: x
Revised: Please select the appropriate granularity for your analysis.
"""
    v = parse_verdict(text)
    assert v.revised_sql is None
    assert v.revised_question is not None
    assert "granularity" in v.revised_question.lower()


def test_revised_prose_with_inline_sql_after_colon_is_picked_up():
    """A real `SELECT` after a `:` boundary IS recognised even when prose precedes."""
    text = """
Correctness: No
Is_ambiguous: No
Explanation: missing distinct
Revised: An alternative phrasing would be: SELECT DISTINCT x FROM t
"""
    v = parse_verdict(text)
    assert v.revised_sql is not None
    assert v.revised_sql.upper().startswith("SELECT DISTINCT")
    # The pre-colon prose becomes the question revision.
    assert v.revised_question is not None
    assert "alternative phrasing" in v.revised_question.lower()
