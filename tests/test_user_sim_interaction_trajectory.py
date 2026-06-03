"""DEV-1515 round 11: ``_user_sim_interaction_from_trajectory`` must
handle every trajectory shape agents actually emit, not just the
canonical list-of-turn-steps shape.

Codex r10: ``pydantic_ai_otf_encode/agent.py:1537`` (and the recursive
flavor's ``agent.py:892``) emit ``trajectory`` as a DICT
(``{"final_output_excerpt": ..., "agents": [...]}``). The pre-fix
helper's call sites did ``list(attempt_data.get("trajectory") or [])``
which on a dict yields the dict's KEYS (strings). Then the helper
iterated and called ``item.get("role")``, hitting ``AttributeError``
on ``str`` and crashing the grader-fallback / skeleton-build paths
AFTER the cascade had already computed cleanly. These tests pin the
defensive contract: any non-list-of-dicts shape degrades to a 0-asks
``UserSimInteraction()`` default.
"""

from __future__ import annotations

from bird_interact_agents.eval.annotate import (
    _user_sim_interaction_from_trajectory,
)


def test_handles_list_of_dicts_canonical_shape():
    """The original shape still works: list of turn-step dicts with
    role markers. Both the ask count and the recorded response land."""
    traj = [
        {"role": "tool_call", "name": "ask_user", "args": {"q": "?"}},
        {"role": "user_sim", "content": "yes please"},
        {"role": "tool_call", "name": "submit_query"},
    ]
    interaction = _user_sim_interaction_from_trajectory(traj)
    assert interaction.n_asks == 1
    assert len(interaction.key_responses) == 1
    assert interaction.key_responses[0].summary == "yes please"


def test_handles_dict_trajectory_from_pydantic_ai_otf_encode():
    """Codex r10 — load-bearing case. Trajectory emitted as a dict
    (``pydantic_ai_otf_encode.agent``) must NOT crash the helper."""
    dict_traj = {
        "final_output_excerpt": "the agent said something",
        "agents": [
            {"agent_id": "encoder", "messages": ["hi"]},
        ],
    }
    # No exception → contract satisfied. Result is a default 0-asks.
    interaction = _user_sim_interaction_from_trajectory(dict_traj)
    assert interaction.n_asks == 0
    assert interaction.key_responses == []


def test_handles_list_of_strings_from_list_coercion_of_dict():
    """Mirrors the pre-fix call-site behaviour where
    ``list(some_dict)`` returned the dict's keys (strings). Even if a
    caller still wraps with ``list(...)`` and lands a list of strings,
    the helper must NOT raise — non-dict items are skipped."""
    interaction = _user_sim_interaction_from_trajectory(
        ["final_output_excerpt", "agents"]
    )
    assert interaction.n_asks == 0


def test_handles_none_trajectory():
    """Safety floor: ``None`` from ``r.get("trajectory")`` → default."""
    interaction = _user_sim_interaction_from_trajectory(None)
    assert interaction.n_asks == 0


def test_handles_empty_list_trajectory():
    interaction = _user_sim_interaction_from_trajectory([])
    assert interaction.n_asks == 0


def test_handles_mixed_list_with_some_non_dict_items():
    """A list that contains BOTH dicts (legitimate turn steps) AND
    stray non-dict entries (a stringification slip-up) must count the
    dicts correctly and skip the rest."""
    traj = [
        {"role": "tool_call", "name": "ask_user"},
        {"role": "user_sim", "content": "first answer"},
        "stray_string_should_be_skipped",
        {"role": "tool_call", "name": "ask_user"},
        42,  # not a dict
        {"role": "tool_call", "name": "submit_query"},
    ]
    interaction = _user_sim_interaction_from_trajectory(traj)
    # 2 ask_user calls counted; the first user_sim follows an ask_user
    # directly so its response is recorded.
    assert interaction.n_asks == 2
    assert len(interaction.key_responses) == 1
    assert interaction.key_responses[0].summary == "first answer"


def test_does_not_crash_when_previous_step_is_non_dict():
    """The recently-followed-an-ask detection looks at ``traj[i-1]``.
    A non-dict previous step must be guarded against the same
    ``.get()`` -> AttributeError landmine."""
    traj = [
        "not_a_dict_first_step",
        {"role": "user_sim", "content": "something"},
    ]
    # Must NOT raise. The user_sim step at i=1 has a non-dict prev,
    # so no response gets recorded — but no exception either.
    interaction = _user_sim_interaction_from_trajectory(traj)
    assert interaction.n_asks == 0
    assert interaction.key_responses == []
