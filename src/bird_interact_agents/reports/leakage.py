"""Optional ``--check-leakage`` diagnostic (Codex finding #2 fold-in).

Scans every prompt for case-insensitive substrings of the
``ground_truth_sql``. Reports a count into ``manifest.leakage_check``;
NEVER redacts. The leaderboard's 15-expert user-simulator review is the
authoritative leakage validation channel; this is just a quick sanity
check we can run before mailing the submission.
"""

from __future__ import annotations


def _gold_substrings(gold: str, min_substring: int) -> list[str]:
    """Sliding-window substrings of length ``min_substring`` over ``gold``.

    A prompt that contains any of them is flagged as 1 leak. We dedupe to
    avoid double-counting overlapping windows when the gold has repeated
    spans.
    """
    if not gold or len(gold) < min_substring:
        return []
    seen: set[str] = set()
    for i in range(len(gold) - min_substring + 1):
        seen.add(gold[i : i + min_substring].lower())
    return list(seen)


def count_leakage(
    *,
    prompts: list[str],
    ground_truth_sql: str | None,
    min_substring: int = 12,
) -> int:
    """Return the number of prompts that contain ANY substring of the gold
    SQL ≥ ``min_substring`` chars long.

    Empty or short golds → 0 (the threshold itself filters them out).
    """
    if not ground_truth_sql:
        return 0
    needles = _gold_substrings(ground_truth_sql, min_substring)
    if not needles:
        return 0
    n = 0
    for prompt in prompts:
        haystack = (prompt or "").lower()
        if any(needle in haystack for needle in needles):
            n += 1
    return n
