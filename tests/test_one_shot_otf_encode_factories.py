"""DEV-1462 — one-shot factories + prompts for `pydantic_ai_otf_encode`.

Mirrors `test_one_shot_recursive_factories.py` for the with-encoding flavor,
which has TWO extra ask_user surfaces vs the recursive flavor:

  * `_build_kb_encoder` (spawned by `kb_to_slayer`) uses ask_user.
  * `_register_kb_to_slayer` dispatches that encoder — must build the
    one-shot variant in one-shot mode.

Plus the otf_encode-specific delivery tools (`submit_projection`,
`submit_encoding`) are kept; only `ask_user` is removed.
"""

from __future__ import annotations

import re
from typing import Any

import pytest


def _tools(agent: Any) -> dict:
    return dict(agent._function_toolset.tools)


# ---------------------------------------------------------------------------
# Factory presence
# ---------------------------------------------------------------------------


def test_one_shot_factories_exist():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    for name in (
        "_build_sub_explorer",
        "_build_projection_resolver_oneshot",
        "_build_query_constructor_oneshot",
        "_build_kb_encoder_oneshot",
    ):
        assert hasattr(factories, name), f"factories.{name} missing"


# ---------------------------------------------------------------------------
# Tool surfaces — no ask_user anywhere.
# ---------------------------------------------------------------------------


def test_sub_explorer_has_kb_to_slayer_but_no_ask_user():
    """otf_encode's explorer keeps `kb_to_slayer` (the KB→model encoder
    dispatch is the whole point of this flavor) but drops `ask_user`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_sub_explorer(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "kb_to_slayer" in names, (
        "otf_encode sub-explorer MUST keep kb_to_slayer in one-shot."
    )
    assert "spawn_subagent" in names
    assert "submit_query" not in names
    assert "submit_encoding" not in names  # that's on the kb_encoder, not the explorer


def test_kb_encoder_oneshot_has_submit_encoding_but_no_ask_user():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_kb_encoder_oneshot(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "submit_encoding" in names
    assert "submit_query" not in names
    assert "spawn_subagent" not in names
    assert "kb_to_slayer" not in names


def test_projection_resolver_oneshot_has_submit_projection_but_no_ask_user():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_projection_resolver_oneshot(
        model="test", model_settings=None,
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "submit_projection" in names, (
        "otf_encode resolver keeps `submit_projection` (it reasons in text "
        "and delivers via a tool); only ask_user is removed."
    )
    assert "submit_query" not in names
    assert "spawn_subagent" not in names


def test_query_constructor_oneshot_no_ask_user_keeps_writes():
    """otf_encode's constructor keeps `create_model`/`edit_model` (the
    write-validation hook is set at the MCPServerStdio layer, not the
    function-toolset) but `ask_user` is gone, `submit_query` stays."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_query_constructor_oneshot(
        model="test", model_settings=None,
        shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b"),
        self_model_id="test",
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "submit_query" in names
    assert "spawn_subagent" not in names


# ---------------------------------------------------------------------------
# kb_to_slayer dispatch — in one-shot, must build the one-shot kb_encoder.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_in_one_shot_dispatches_oneshot_kb_encoder(monkeypatch):
    """`_register_kb_to_slayer` is parametrized by `eval_mode`/child-builder.
    In one-shot, the kb_to_slayer tool MUST construct the one-shot
    kb_encoder (no ask_user), not the a-interact one."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories
    from bird_interact_agents import usage as usage_mod

    # Short-circuit litellm cost-of-token lookup — the stub uses model="test"
    # which litellm can't price; without this the kb_encoder usage record-keeping
    # raises before we reach the eval_mode dispatch assertion.
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    oneshot_calls: list = []
    classic_calls: list = []

    monkeypatch.setattr(
        factories, "_build_kb_encoder_oneshot",
        lambda **kw: oneshot_calls.append(kw) or _StubKbEncoder(),
    )
    monkeypatch.setattr(
        factories, "_build_kb_encoder",
        lambda **kw: classic_calls.append(kw) or _StubKbEncoder(),
    )

    # Build a sub-explorer in one-shot mode; its kb_to_slayer wrapper
    # closes over the one-shot encoder selection.
    agent = factories._build_sub_explorer(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    kb_tool = _tools(agent)["kb_to_slayer"]
    ctx = _make_ctx_with_loaded_kb()
    # The kb_to_slayer tool walks kb deps + dispatches the encoder per kb_id.
    # We only assert the BRANCH chosen: oneshot vs classic.
    await kb_tool.function(ctx, kb_ids=[1])
    assert oneshot_calls, (
        f"one-shot kb_to_slayer MUST dispatch _build_kb_encoder_oneshot; "
        f"classic_calls={classic_calls} oneshot_calls={oneshot_calls}"
    )
    assert not classic_calls, (
        "one-shot kb_to_slayer MUST NOT build the ask_user-carrying encoder."
    )


class _StubKbEncoder:
    """Minimal pydantic-ai-like stub the kb_to_slayer wrapper can call."""
    async def run(self, *a, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(
            output="done",
            usage=lambda: SimpleNamespace(
                input_tokens=0, output_tokens=0,
                cache_read_tokens=0, cache_write_tokens=0,
            ),
            all_messages=lambda: [],
        )


def _make_ctx_with_loaded_kb():
    """Build a TaskDeps with a single pre-loaded KB row so kb_to_slayer
    has something to walk without hitting real storage."""
    from pydantic_ai import RunContext

    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult, SharedTaskState, TaskDeps,
    )
    from bird_interact_agents.harness import SampleStatus

    status = SampleStatus(
        idx=0, original_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench-base-lite-sqlite",
        },
        remaining_budget=100.0, total_budget=100.0,
    )
    shared = SharedTaskState(
        status=status, data_path_base="/tmp/ignored", db_name="fake_db",
        amb_user_query="x", slayer_storage_dir="",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    shared._kb_rows_by_id = {
        1: {"id": 1, "_memory_entities": [], "_learning": "kb body"},
    }
    deps = TaskDeps(shared=shared, depth=0, max_depth=3, self_record_idx=None)
    return RunContext(deps=deps, model=None, usage=None, prompt="", run_step=0)


# ---------------------------------------------------------------------------
# Prompts — placeholders + absence-only content checks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,placeholders",
    [
        ("ROOT_EXPLORER_PROMPT", ("{budget}", "{db_name}", "{user_query}")),
        ("SUB_EXPLORER_PROMPT", ("{budget}", "{db_name}",
                                 "{focus}", "{instruction}")),
        ("PROJECTION_RESOLVER_ONESHOT_PROMPT",
         ("{amb_user_query}", "{spec}", "{budget}", "{db_name}")),
        ("QUERY_CONSTRUCTOR_ONESHOT_PROMPT",
         ("{amb_user_query}", "{spec}",
          "{confirmed_projection}", "{budget}", "{db_name}")),
        ("KB_ENCODER_ONESHOT_PROMPT",
         ("{db_name}", "{kb_id}", "{kb_row_yaml}",
          "{deps_block}", "{budget}")),
    ],
)
def test_one_shot_prompts_exist_with_required_placeholders(name, placeholders):
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    body = getattr(prompts, name, None)
    assert isinstance(body, str) and body.strip()
    for p in placeholders:
        assert p in body, f"{name} missing {p!r}"


_USER_SIM_FORBIDDEN_RE = re.compile(
    r"\b("
    r"ask[_ ]user|user[- ]sim|user-sim|ask the user|the user-sim"
    r")\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "name",
    [
        "ROOT_EXPLORER_PROMPT", "SUB_EXPLORER_PROMPT",
        "PROJECTION_RESOLVER_ONESHOT_PROMPT",
        "QUERY_CONSTRUCTOR_ONESHOT_PROMPT",
        "KB_ENCODER_ONESHOT_PROMPT",
    ],
)
def test_one_shot_prompts_have_no_ask_user_or_user_sim_language(name):
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    body = getattr(prompts, name)
    m = _USER_SIM_FORBIDDEN_RE.search(body)
    assert m is None, (
        f"{name} contains forbidden ask_user/user-sim text: "
        f"{m.group(0)!r} at offset {m.start()}"
    )


def test_one_shot_projection_submit_suffix_has_no_user_language():
    """The existing `_PROJECTION_SUBMIT_SUFFIX` says 'ask the user to confirm'
    — one-shot needs a parallel suffix that drops that wording."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent

    suffix_name = "_PROJECTION_SUBMIT_SUFFIX_ONESHOT"
    assert hasattr(agent, suffix_name), (
        f"{suffix_name} must exist so the one-shot run_task can replace "
        f"the a-interact `_PROJECTION_SUBMIT_SUFFIX` (which says 'ask the "
        f"user to confirm')."
    )
    suffix = getattr(agent, suffix_name)
    m = _USER_SIM_FORBIDDEN_RE.search(suffix)
    assert m is None, (
        f"{suffix_name} must not contain ask_user/user-sim text; "
        f"got hit: {m.group(0)!r}" if m else ""
    )


def test_one_shot_tool_docstrings_have_no_ask_user_language():
    """Codex #7 — scan EVERY native tool's docstring across all one-shot
    factories (incl. the kb_encoder) for forbidden `ask_user`/`user-sim`
    text. otf_encode's parallel of the same check on the recursive side."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agents = {
        "sub_explorer": factories._build_sub_explorer(
            model="test", model_settings=None,
            shared_slayer_server=None, self_model_id="test",
        ),
        "projection_resolver_oneshot": factories._build_projection_resolver_oneshot(
            model="test", model_settings=None,
        ),
        "query_constructor_oneshot": factories._build_query_constructor_oneshot(
            model="test", model_settings=None,
            shared_slayer_server=None,
            confirmed_projection=("c",),
            self_model_id="test",
        ),
        "kb_encoder_oneshot": factories._build_kb_encoder_oneshot(
            model="test", model_settings=None,
            shared_slayer_server=None, self_model_id="test",
        ),
    }
    offenders: list[tuple[str, str, str]] = []
    for role, agent in agents.items():
        for tool_name, tool in _tools(agent).items():
            doc = (
                getattr(tool, "description", None)
                or getattr(getattr(tool, "function", None), "__doc__", None)
                or ""
            )
            m = _USER_SIM_FORBIDDEN_RE.search(doc)
            if m is not None:
                offenders.append((role, tool_name, m.group(0)))
    assert not offenders, (
        "otf_encode one-shot tool docstrings must not reference ask_user / "
        f"user-sim. Offenders: {offenders!r}"
    )
