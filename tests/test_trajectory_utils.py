"""Shape-agnostic accessors over the per-task ``trajectory`` dict.

The repo has two trajectory shapes in flight:

* **Old** (pydantic_ai adapter): ``{messages, user_sim_transcript, final_output_excerpt}``
* **New** (pydantic_ai_recursive): ``{agents: [{messages, user_sim_transcript, ...}], final_output_excerpt}``

``trajectory_utils`` wraps both behind three small accessors so any future
analysis script works on either without branching. These tests pin the
contract down: same return shape regardless of which input shape is fed in.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# iter_messages — flatten across agents
# ---------------------------------------------------------------------------


def test_iter_messages_old_shape():
    from bird_interact_agents.trajectory_utils import iter_messages

    traj = {
        "final_output_excerpt": "...",
        "messages": [{"kind": "request"}, {"kind": "response"}],
        "user_sim_transcript": [],
    }
    out = iter_messages(traj)
    assert out == [{"kind": "request"}, {"kind": "response"}]
    # Returns a NEW list — mutation must not leak back into traj.
    out.append({"kind": "spy"})
    assert len(traj["messages"]) == 2


def test_iter_messages_new_shape_preserves_order():
    from bird_interact_agents.trajectory_utils import iter_messages

    traj = {
        "final_output_excerpt": "...",
        "agents": [
            {"role": "root_clarifier", "messages": [{"id": "m1"}]},
            {"role": "sub_clarifier", "messages": [{"id": "m2"}, {"id": "m3"}]},
            {"role": "query_constructor", "messages": [{"id": "m4"}]},
        ],
    }
    assert iter_messages(traj) == [
        {"id": "m1"}, {"id": "m2"}, {"id": "m3"}, {"id": "m4"},
    ]


def test_iter_messages_new_shape_missing_messages_field():
    """An agent record whose ``messages`` is absent (e.g. a failed subagent
    whose error was recorded before any model turns) must not crash."""
    from bird_interact_agents.trajectory_utils import iter_messages

    traj = {
        "agents": [
            {"role": "root_clarifier"},                       # no messages
            {"role": "sub_clarifier", "messages": [{"id": "m1"}]},
        ],
    }
    assert iter_messages(traj) == [{"id": "m1"}]


def test_iter_messages_empty():
    from bird_interact_agents.trajectory_utils import iter_messages

    assert iter_messages({}) == []
    assert iter_messages({"agents": []}) == []
    assert iter_messages({"messages": []}) == []


# ---------------------------------------------------------------------------
# iter_user_sim_transcript — flatten across agents
# ---------------------------------------------------------------------------


def test_iter_user_sim_transcript_old_shape():
    from bird_interact_agents.trajectory_utils import iter_user_sim_transcript

    traj = {
        "messages": [],
        "user_sim_transcript": [
            {"phase": "encoder", "agent_question": "q1"},
            {"phase": "decoder", "agent_question": "q1"},
        ],
    }
    out = iter_user_sim_transcript(traj)
    assert out == [
        {"phase": "encoder", "agent_question": "q1"},
        {"phase": "decoder", "agent_question": "q1"},
    ]


def test_iter_user_sim_transcript_new_shape_preserves_order():
    from bird_interact_agents.trajectory_utils import iter_user_sim_transcript

    traj = {
        "agents": [
            {"role": "sub_clarifier",
             "user_sim_transcript": [{"phase": "encoder", "agent_question": "A"}]},
            {"role": "sub_clarifier",
             "user_sim_transcript": [{"phase": "encoder", "agent_question": "B"}]},
            {"role": "query_constructor",
             "user_sim_transcript": [{"phase": "encoder", "agent_question": "C"}]},
        ],
    }
    out = iter_user_sim_transcript(traj)
    assert [t["agent_question"] for t in out] == ["A", "B", "C"]


def test_iter_user_sim_transcript_new_shape_missing_field():
    """Root clarifier never calls ask_user, so its agent record carries no
    ``user_sim_transcript`` field. The accessor must tolerate that."""
    from bird_interact_agents.trajectory_utils import iter_user_sim_transcript

    traj = {
        "agents": [
            {"role": "root_clarifier"},
            {"role": "sub_clarifier", "user_sim_transcript": [{"phase": "encoder"}]},
        ],
    }
    assert iter_user_sim_transcript(traj) == [{"phase": "encoder"}]


# ---------------------------------------------------------------------------
# iter_agent_records — return per-agent list, synthesise for old shape
# ---------------------------------------------------------------------------


def test_iter_agent_records_old_shape_synthesises_single_record():
    """For the old single-agent shape, return ONE synthetic record with
    role ``legacy_single_agent`` carrying the flat messages and
    user_sim_transcript, so downstream code can iterate uniformly."""
    from bird_interact_agents.trajectory_utils import iter_agent_records

    traj = {
        "final_output_excerpt": "ok",
        "messages": [{"id": "m1"}, {"id": "m2"}],
        "user_sim_transcript": [{"phase": "encoder"}],
    }
    out = iter_agent_records(traj)
    assert len(out) == 1
    rec = out[0]
    assert rec["role"] == "legacy_single_agent"
    assert rec["depth"] == 0
    assert rec["parent_idx"] is None
    assert rec["messages"] == [{"id": "m1"}, {"id": "m2"}]
    assert rec["user_sim_transcript"] == [{"phase": "encoder"}]


def test_iter_agent_records_new_shape_passthrough():
    from bird_interact_agents.trajectory_utils import iter_agent_records

    agents = [
        {"role": "root_clarifier", "depth": 0, "parent_idx": None,
         "messages": [], "user_sim_transcript": []},
        {"role": "sub_clarifier", "depth": 1, "parent_idx": 0,
         "messages": [], "user_sim_transcript": []},
        {"role": "query_constructor", "depth": 0, "parent_idx": None,
         "messages": [], "user_sim_transcript": []},
    ]
    out = iter_agent_records({"agents": agents})
    assert out == agents
    # Returns a NEW list — mutation must not leak back.
    out.pop()
    assert len({a["role"] for a in agents}) == 3


def test_iter_agent_records_old_shape_no_messages():
    """An error-path old-shape trajectory may have empty / missing
    messages/user_sim_transcript. Synthesised record fills with empty lists."""
    from bird_interact_agents.trajectory_utils import iter_agent_records

    traj = {"final_output_excerpt": "", "messages": [], "user_sim_transcript": []}
    out = iter_agent_records(traj)
    assert len(out) == 1
    assert out[0]["messages"] == []
    assert out[0]["user_sim_transcript"] == []
    assert out[0]["role"] == "legacy_single_agent"


def test_accessors_tolerate_non_dict_trajectory():
    """`run.py`'s outer catastrophic-error path writes ``"trajectory": []``
    for tasks that died before any per-task data was captured. The
    accessors must return [] rather than raising — otherwise loading
    such a result row crashes analysis scripts."""
    from bird_interact_agents.trajectory_utils import (
        iter_agent_records,
        iter_messages,
        iter_user_sim_transcript,
    )

    for bad in ([], "oops", None, 0, 3.14):
        assert iter_messages(bad) == []
        assert iter_user_sim_transcript(bad) == []
        assert iter_agent_records(bad) == []


def test_accessors_tolerate_non_list_agents():
    """Equally defensive: ``"agents": "x"`` (or any non-list) should
    fall through to the old-shape path rather than blow up."""
    from bird_interact_agents.trajectory_utils import (
        iter_agent_records,
        iter_messages,
    )

    traj = {"agents": "not-a-list", "messages": [{"id": "m1"}]}
    # Non-list agents → fall through to old-shape behaviour.
    assert iter_messages(traj) == [{"id": "m1"}]
    # iter_agent_records synthesises the legacy single-record entry.
    out = iter_agent_records(traj)
    assert out and out[0]["role"] == "legacy_single_agent"


def test_iter_agent_records_old_shape_missing_keys():
    """A truly minimal old-shape trajectory — only `final_output_excerpt`,
    no `messages` or `user_sim_transcript` at all (some error paths in
    the existing pydantic_ai adapter omit them). The synthesised record
    must still be produced with empty lists, not raise KeyError."""
    from bird_interact_agents.trajectory_utils import iter_agent_records

    traj = {"final_output_excerpt": ""}
    out = iter_agent_records(traj)
    assert len(out) == 1
    assert out[0]["role"] == "legacy_single_agent"
    assert out[0]["messages"] == []
    assert out[0]["user_sim_transcript"] == []
    # Same for the flat accessors.
    from bird_interact_agents.trajectory_utils import (
        iter_messages,
        iter_user_sim_transcript,
    )
    assert iter_messages(traj) == []
    assert iter_user_sim_transcript(traj) == []
