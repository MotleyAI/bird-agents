"""DEV-1613: the in-task agent state/deps must carry ``agent_model`` so
the shared submit helpers can build the N5 judge. ``_submit`` reads it via
``getattr(state, "agent_model", None)`` — these tests pin the two
representative seams (pydantic_ai ``TaskDeps`` field + claude_sdk
``_state_view`` adapter); all other ``_submit``-sharing adapters thread the
same field in parallel.
"""
from __future__ import annotations


def test_pydantic_ai_taskdeps_has_agent_model_field():
    from bird_interact_agents.agents.pydantic_ai.agent import TaskDeps

    assert "agent_model" in TaskDeps.model_fields


def test_claude_sdk_state_view_exposes_agent_model():
    import bird_interact_agents.agents.claude_sdk.agent as csdk

    token = csdk._ctx_var.set({
        "status": object(),
        "data_path_base": "/dev/null",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "agent_model": "anthropic/claude-opus-4-7",
    })
    try:
        view = csdk._state_view()
    finally:
        csdk._ctx_var.reset(token)

    assert getattr(view, "agent_model", None) == "anthropic/claude-opus-4-7"
