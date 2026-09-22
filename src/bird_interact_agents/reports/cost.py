"""Section VI Universal Cost Scheme.

This is the LEADERBOARD contract for a-Interact custom agents — NOT our
internal ``harness.ACTION_COSTS`` table. The harness assigns its own
costs at run-time (per ``ACTION_COSTS`` in ``harness.py``) but for the
report we MUST replay using Section VI rules:

* ``ask = 2``, ``submit = 3``, ``execute = 1`` (fixed; token counts
  ignored for these).
* Every other action is token-aware: ``input < 250 AND output < 1000``
  → ``0.5``; else ``1.0`` (Section VI prose example: a getter that
  produces a 400-token observation off a 4-token call → 0.5; same
  getter that produces a 2500-token observation → 1.0).

Canonical names ``ask`` / ``submit`` / ``execute`` come from upstream's
``eval_react_bird_interact.py``; the Section VI prose uses
``ask_user`` / ``submit_sql`` / ``execute_sql``. ``action_canonicalize``
defines the mapping; we accept the short forms here.
"""

from __future__ import annotations


FIXED_COSTS: dict[str, int] = {"ask": 2, "submit": 3, "execute": 1}

SECTION_VI_THRESHOLDS: dict[str, float] = {
    "input_tokens_lt": 250,
    "output_tokens_lt": 1000,
    "cheap_cost": 0.5,
    "expensive_cost": 1.0,
}


def compute_action_cost(
    canonical_action: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return the Section VI cost for one action call.

    ``canonical_action`` is the upstream-canonical form returned by
    ``action_canonicalize.canonicalize_action`` — i.e. ``ask`` /
    ``submit`` / ``execute`` for the fixed-cost trio, anything else for
    token-aware actions.
    """
    if canonical_action in FIXED_COSTS:
        return float(FIXED_COSTS[canonical_action])
    if (
        input_tokens < SECTION_VI_THRESHOLDS["input_tokens_lt"]
        and output_tokens < SECTION_VI_THRESHOLDS["output_tokens_lt"]
    ):
        return SECTION_VI_THRESHOLDS["cheap_cost"]
    return SECTION_VI_THRESHOLDS["expensive_cost"]
