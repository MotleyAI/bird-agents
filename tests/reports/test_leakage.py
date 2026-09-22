"""Tests for ``--check-leakage`` diagnostic (Codex finding #2 fold-in).

Spec (DEV-1553):
* The leakage check scans every prompt_flow entry's ``prompt`` field for
  case-insensitive substring matches against the instance's
  ``ground_truth_sql``. The leakage counter for the instance is the
  number of entries that contain ANY non-trivial substring of the gold
  SQL.
* Does NOT redact — only reports a count into ``manifest.leakage_check``.
* Below a minimum substring length (default 12 chars) matches are
  ignored, so short fragments like ``SELECT`` or ``FROM x`` do not
  produce false positives.
"""

from __future__ import annotations


def test_leakage_check_zero_when_clean():
    """Observation contains schema text only — gold SQL not present."""
    from bird_interact_agents.reports.leakage import count_leakage

    n = count_leakage(
        prompts=["table_a:\n  col_x int\n", "Phase 1 SQL Correct!"],
        ground_truth_sql="SELECT trader.id FROM trader JOIN compliancecase ON x = y",
        min_substring=12,
    )
    assert n == 0


def test_leakage_check_flags_full_gold_substring():
    """An observation containing the gold SQL verbatim flags as 1."""
    from bird_interact_agents.reports.leakage import count_leakage

    gold = "SELECT trader.id FROM trader JOIN compliancecase ON x = y"
    n = count_leakage(
        prompts=[f"Hint from user: try `{gold}`."],
        ground_truth_sql=gold,
        min_substring=12,
    )
    assert n == 1


def test_leakage_check_counts_per_prompt():
    from bird_interact_agents.reports.leakage import count_leakage

    gold = "SELECT trader.id FROM trader JOIN compliancecase ON x = y"
    n = count_leakage(
        prompts=[
            "schema info, no leak",
            f"first leak: {gold}",
            "more schema",
            f"second leak: {gold[:30]}",  # partial but ≥ min_substring
        ],
        ground_truth_sql=gold,
        min_substring=12,
    )
    assert n == 2


def test_leakage_check_ignores_short_fragments():
    """Common SQL fragments (under min_substring) must not flag."""
    from bird_interact_agents.reports.leakage import count_leakage

    n = count_leakage(
        prompts=["SELECT", "FROM x", "JOIN y", "WHERE z = 1"],
        ground_truth_sql="SELECT * FROM x JOIN y ON x.k = y.k WHERE z = 1",
        min_substring=12,
    )
    assert n == 0


def test_leakage_check_handles_missing_gold():
    """No ground_truth_sql → 0, no error."""
    from bird_interact_agents.reports.leakage import count_leakage

    assert (
        count_leakage(
            prompts=["anything"], ground_truth_sql=None, min_substring=12
        )
        == 0
    )
    assert (
        count_leakage(
            prompts=["anything"], ground_truth_sql="", min_substring=12
        )
        == 0
    )
