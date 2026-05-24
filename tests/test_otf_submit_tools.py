"""Tests for the ``submit_encoding`` / ``submit_projection`` tools + the
"must submit before finishing" output-validator gate (DEV-1454).

Why these exist: the encoders (`setup_encoder`, `kb_encoder`) and the
`projection_resolver` used a structured ``output_type`` (`EncoderResult` /
`list[str]`), which makes pydantic-ai send Anthropic ``tool_choice='any'`` —
forbidding any assistant text, so the model can never reason between tool calls
(it thrashed: 100+ tool calls, zero reasoning). The fix drops ``output_type``
(→ text output → ``tool_choice='auto'`` → ReAct on ANY provider) and captures
the final result through an explicit ``submit_*`` tool into per-run deps, mirroring
the existing ``submit_query`` pattern. A per-run output-validator forces the model
to actually submit before a bare-text response can end the run.

These are framework-only unit tests (no LLM/MCP): they call the registered tool's
function directly with a fake ``RunContext`` and a duck-typed deps holder.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext

from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
    EncodedEntity, EncoderResult, TaskDeps,
)


def _ctx(deps):
    """A minimal RunContext for calling a tool's function directly."""
    return RunContext(deps=deps, model=None, usage=None, prompt="", run_step=0)


def _get_tool(agent: Agent, name: str):
    return dict(agent._function_toolset.tools)[name].function


# ---------------------------------------------------------------------------
# submit_encoding
# ---------------------------------------------------------------------------


async def test_submit_encoding_captures_result_into_deps():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _register_submit_encoding,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    _register_submit_encoding(agent)
    fn = _get_tool(agent, "submit_encoding")

    deps = SimpleNamespace(encoder_submission=None)
    result_json = (
        '{"kb_id": 7, "status": "encoded", '
        '"entities": [{"kind": "column", "host_model": "properties", '
        '"name": "bath_ratio", "entity_ref": "households.properties.bath_ratio"}], '
        '"notes": "ok"}'
    )
    out = await fn(_ctx(deps), result_json=result_json)
    assert isinstance(deps.encoder_submission, EncoderResult)
    assert deps.encoder_submission.kb_id == 7
    assert deps.encoder_submission.status == "encoded"
    assert deps.encoder_submission.entities[0].name == "bath_ratio"
    assert isinstance(out, str)  # confirmation back to the agent


async def test_submit_encoding_bad_json_raises_model_retry_and_does_not_capture():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _register_submit_encoding,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    _register_submit_encoding(agent)
    fn = _get_tool(agent, "submit_encoding")

    deps = SimpleNamespace(encoder_submission=None)
    with pytest.raises(ModelRetry):
        await fn(_ctx(deps), result_json="not json{{")
    assert deps.encoder_submission is None  # nothing captured on failure


async def test_submit_encoding_invalid_shape_raises_model_retry():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _register_submit_encoding,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    _register_submit_encoding(agent)
    fn = _get_tool(agent, "submit_encoding")

    deps = SimpleNamespace(encoder_submission=None)
    # valid json, but not an EncoderResult (missing required status)
    with pytest.raises(ModelRetry):
        await fn(_ctx(deps), result_json='{"kb_id": 1}')
    assert deps.encoder_submission is None


# ---------------------------------------------------------------------------
# submit_projection
# ---------------------------------------------------------------------------


async def test_submit_projection_captures_list():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _register_submit_projection,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    _register_submit_projection(agent)
    fn = _get_tool(agent, "submit_projection")

    deps = SimpleNamespace(projection_submission=None)
    out = await fn(_ctx(deps), columns_json='["region", "revenue"]')
    assert deps.projection_submission == ["region", "revenue"]
    assert isinstance(out, str)


async def test_submit_projection_empty_list_is_accepted():
    """An empty projection is a VALID submission (it triggers the resolver's
    recovery pass downstream) — the gate must let it through, not reject it."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _register_submit_projection,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    _register_submit_projection(agent)
    fn = _get_tool(agent, "submit_projection")

    deps = SimpleNamespace(projection_submission=None)
    await fn(_ctx(deps), columns_json="[]")
    assert deps.projection_submission == []  # captured, not None


async def test_submit_projection_non_list_raises_model_retry():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _register_submit_projection,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    _register_submit_projection(agent)
    fn = _get_tool(agent, "submit_projection")

    deps = SimpleNamespace(projection_submission=None)
    with pytest.raises(ModelRetry):
        await fn(_ctx(deps), columns_json='{"not": "a list"}')
    assert deps.projection_submission is None


# ---------------------------------------------------------------------------
# the "must submit before finishing" gate
# ---------------------------------------------------------------------------


async def test_require_submission_gate_blocks_until_submitted():
    """The output-validator must raise ModelRetry while the submission slot is
    None (so a bare-text response can't end the run before submit), and pass the
    output through once a submission exists — including an empty-list projection
    (not None)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _require_submission,
    )

    gate = _require_submission("encoder_submission")
    # not submitted yet → block
    with pytest.raises(ModelRetry):
        await gate(_ctx(SimpleNamespace(encoder_submission=None)), "done")
    # submitted → pass the output through unchanged
    sentinel = EncoderResult(kb_id=1, status="deferred")
    out = await gate(_ctx(SimpleNamespace(encoder_submission=sentinel)), "done")
    assert out == "done"

    # empty-list projection counts as "submitted" (not None)
    gate_proj = _require_submission("projection_submission")
    out2 = await gate_proj(_ctx(SimpleNamespace(projection_submission=[])), "done")
    assert out2 == "done"
    with pytest.raises(ModelRetry):
        await gate_proj(_ctx(SimpleNamespace(projection_submission=None)), "done")


# ---------------------------------------------------------------------------
# builders: tool surface (no structured output_type; submit tool present)
# ---------------------------------------------------------------------------


def _assert_text_output_with_gate(agent):
    """The whole point of the refactor: NO structured output_type (so the model
    can reason in text → tool_choice='auto') AND a must-submit output-validator
    gate registered (so a bare-text reply can't end the run before submit). A
    regression that re-adds output_type or drops the gate must fail here."""
    assert agent.output_type is str           # structured output_type removed
    assert len(agent._output_validators) >= 1  # the must-submit gate is wired


def test_setup_encoder_has_submit_encoding_and_no_ask_user():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        _build_setup_encoder,
    )

    agent = _build_setup_encoder(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    tools = set(dict(agent._function_toolset.tools).keys())
    assert "submit_encoding" in tools
    assert "ask_user" not in tools          # setup encoder still has no clarify
    assert "submit_query" not in tools
    _assert_text_output_with_gate(agent)


def test_kb_encoder_has_submit_encoding_and_ask_user():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _build_kb_encoder,
    )

    agent = _build_kb_encoder(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    tools = set(dict(agent._function_toolset.tools).keys())
    assert "submit_encoding" in tools
    assert "ask_user" in tools              # task-time encoder keeps clarify
    _assert_text_output_with_gate(agent)


def test_projection_resolver_has_submit_projection():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _build_projection_resolver,
    )

    agent = _build_projection_resolver(
        model="test", model_settings=None, self_model_id="test",
    )
    tools = set(dict(agent._function_toolset.tools).keys())
    assert "submit_projection" in tools
    _assert_text_output_with_gate(agent)
