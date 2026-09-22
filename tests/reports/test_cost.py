"""Tests for the Section VI Universal Cost Scheme.

Spec (DEV-1553):
* Fixed-cost actions: ``ask = 2``, ``submit = 3``, ``execute = 1``.
* Token-aware actions: ``input < 250 AND output < 1000 → 0.5``; else ``1.0``.
  AND-semantics MUST be preserved at the boundary (Codex finding #4 demands
  contract-exact behavior).
* Cost classification depends on the *canonical* action name, not the raw
  MCP tool name. Anything not in the fixed set is token-aware.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixed-cost actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canonical,expected",
    [("ask", 2), ("submit", 3), ("execute", 1)],
)
def test_fixed_cost_actions(canonical, expected):
    from bird_interact_agents.reports.cost import compute_action_cost

    # Token counts must be IGNORED for fixed actions.
    assert compute_action_cost(canonical, input_tokens=9999, output_tokens=9999) == expected
    assert compute_action_cost(canonical, input_tokens=0, output_tokens=0) == expected


# ---------------------------------------------------------------------------
# Token-aware actions: AND-semantics at the threshold boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_toks,out_toks,expected",
    [
        # Cheap quadrant: both strictly below.
        (0, 0, 0.5),
        (249, 999, 0.5),
        # Crossing EITHER threshold → expensive.
        (250, 999, 1.0),  # input at threshold
        (249, 1000, 1.0),  # output at threshold
        (250, 1000, 1.0),  # both at threshold
        # Far above thresholds.
        (5000, 5000, 1.0),
    ],
)
def test_token_aware_threshold_boundaries(in_toks, out_toks, expected):
    from bird_interact_agents.reports.cost import compute_action_cost

    assert (
        compute_action_cost(
            "mcp__slayer__search",
            input_tokens=in_toks,
            output_tokens=out_toks,
        )
        == expected
    )


def test_unknown_canonical_action_is_token_aware_not_fixed():
    """Anything outside {ask, submit, execute} routes through the token rule."""
    from bird_interact_agents.reports.cost import compute_action_cost

    cheap = compute_action_cost("get_schema", input_tokens=0, output_tokens=10)
    assert cheap == 0.5
    expensive = compute_action_cost(
        "get_schema", input_tokens=0, output_tokens=5000
    )
    assert expensive == 1.0


# ---------------------------------------------------------------------------
# Section VI prose example reproduction (paper Appendix J).
# ---------------------------------------------------------------------------


def test_section_vi_paper_example_cheap():
    """``get_the_first_n_table_schema(3)`` with in=4, out=400 → 0.5."""
    from bird_interact_agents.reports.cost import compute_action_cost

    assert (
        compute_action_cost(
            "get_the_first_n_table_schema",
            input_tokens=4,
            output_tokens=400,
        )
        == 0.5
    )


def test_section_vi_paper_example_expensive():
    """``get_the_first_n_table_schema(10)`` with in=4, out=2500 → 1.0."""
    from bird_interact_agents.reports.cost import compute_action_cost

    assert (
        compute_action_cost(
            "get_the_first_n_table_schema",
            input_tokens=4,
            output_tokens=2500,
        )
        == 1.0
    )


# ---------------------------------------------------------------------------
# Exposed cost-table constants (used in manifest.fixed_costs).
# ---------------------------------------------------------------------------


def test_fixed_costs_constant_matches_section_vi():
    from bird_interact_agents.reports.cost import FIXED_COSTS

    assert FIXED_COSTS == {"ask": 2, "submit": 3, "execute": 1}


def test_section_vi_thresholds_constant():
    from bird_interact_agents.reports.cost import SECTION_VI_THRESHOLDS

    assert SECTION_VI_THRESHOLDS == {
        "input_tokens_lt": 250,
        "output_tokens_lt": 1000,
        "cheap_cost": 0.5,
        "expensive_cost": 1.0,
    }
