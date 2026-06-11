"""Tests for the harness-faithful budget calculation + Section VI cost replay.

Spec (DEV-1553):
* ``total_budget`` for a-Interact = ``harness.calculate_budget(task_data,
  patience, mode="a-interact") = 6 + 2*ambiguity_count + 2*patience``.
* Section VI costs are summed cumulatively over the trajectory in submit
  order; ``remaining_budget[k] = max(0, total_budget - sum_{i<=k}
  action_cost[i])``.
* The maximum is clipped at 0 (matches harness runtime ``update_budget``).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# total_budget matches harness.calculate_budget exactly
# ---------------------------------------------------------------------------


def test_total_budget_zero_ambiguity_patience3():
    from bird_interact_agents.reports.budget import calculate_total_budget

    task_data = {"user_query_ambiguity": {}, "knowledge_ambiguity": []}
    # 6 + 2*0 + 2*3
    assert calculate_total_budget(task_data, patience=3) == 12.0


def test_total_budget_critical_ambiguity_counted():
    from bird_interact_agents.reports.budget import calculate_total_budget

    task_data = {
        "user_query_ambiguity": {"critical_ambiguity": ["a", "b"]},
        "knowledge_ambiguity": [],
    }
    # 6 + 2*2 + 2*3
    assert calculate_total_budget(task_data, patience=3) == 16.0


def test_total_budget_knowledge_ambiguity_counted():
    from bird_interact_agents.reports.budget import calculate_total_budget

    task_data = {
        "user_query_ambiguity": {},
        "knowledge_ambiguity": [{"id": 1}, {"id": 2}, {"id": 3}],
    }
    # 6 + 2*3 + 2*3
    assert calculate_total_budget(task_data, patience=3) == 18.0


def test_total_budget_both_kinds_of_ambiguity():
    from bird_interact_agents.reports.budget import calculate_total_budget

    task_data = {
        "user_query_ambiguity": {"critical_ambiguity": ["a"]},
        "knowledge_ambiguity": [{"id": 1}, {"id": 2}, {"id": 3}],
    }
    # 6 + 2*4 + 2*500
    assert calculate_total_budget(task_data, patience=500) == 1014.0


def test_total_budget_matches_harness_calculate_budget():
    """Pin to the harness implementation so a future change is intentional."""
    from bird_interact_agents.harness import calculate_budget as harness_calc
    from bird_interact_agents.reports.budget import calculate_total_budget

    for patience in (3, 500):
        for crit in ([], ["a"], ["a", "b", "c", "d"]):
            for kb_amb_n in (0, 1, 5):
                task_data = {
                    "user_query_ambiguity": {"critical_ambiguity": crit},
                    "knowledge_ambiguity": [{"id": i} for i in range(kb_amb_n)],
                }
                expected = harness_calc(
                    task_data, patience=patience, mode="a-interact"
                )
                got = calculate_total_budget(task_data, patience=patience)
                assert got == expected, (
                    patience,
                    len(crit),
                    kb_amb_n,
                    expected,
                    got,
                )


# ---------------------------------------------------------------------------
# Cumulative Section VI replay over a turn list
# ---------------------------------------------------------------------------


def test_replay_monotone_decreasing_clipped_at_zero():
    from bird_interact_agents.reports.budget import replay_remaining_budget

    action_costs = [3.0, 2.0, 1.0, 0.5, 0.5, 0.5]
    remaining = replay_remaining_budget(
        total_budget=6.0, action_costs=action_costs
    )
    # Cumulative: 3, 5, 6, 6.5, 7, 7.5 → remaining: 3, 1, 0, 0, 0, 0
    assert remaining == [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # Monotonically non-increasing.
    for a, b in zip(remaining, remaining[1:]):
        assert a >= b


def test_replay_zero_actions_returns_empty():
    from bird_interact_agents.reports.budget import replay_remaining_budget

    assert replay_remaining_budget(total_budget=12.0, action_costs=[]) == []


def test_replay_never_negative():
    from bird_interact_agents.reports.budget import replay_remaining_budget

    remaining = replay_remaining_budget(
        total_budget=4.0, action_costs=[3.0, 3.0, 3.0]
    )
    assert all(r >= 0.0 for r in remaining)
    assert remaining[-1] == 0.0
