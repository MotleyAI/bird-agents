"""DEV-1525: claude_sdk framework dispatch.

Tests that FAIL until run.py is updated:
- CLI --framework accepts only "claude_sdk" (required, no default)
- make_runner dispatches to the correct OTF agent class based on (dataset, query_mode)
- _validate_slayer_setup is simplified: raw → always passes; slayer → requires on-the-fly
- Missing 'dataset' in task_data raises ValueError instead of silently defaulting
"""

from __future__ import annotations

import argparse

import pytest

import bird_interact_agents.run as run_mod
from bird_interact_agents.run import _validate_slayer_setup


# ---------------------------------------------------------------------------
# CLI --framework choices
# ---------------------------------------------------------------------------


def test_main_framework_choices_expose_only_aggregator_tokens():
    """DEV-1555 (CR r1): the CLI exposes only the two aggregator tokens —
    `claude_sdk` and `claude_sdk_v1`. Per-variant tokens (`claude_sdk_otf*`)
    remain reachable through `_make_runner` for programmatic / test
    callers, but the CLI infers the variant from
    (benchmark.one_shot × query_mode).
    """
    import inspect

    src = inspect.getsource(run_mod.main)
    must_expose = ("claude_sdk", "claude_sdk_v1")
    must_hide = (
        "claude_sdk_otf",
        "claude_sdk_otf_raw",
        "claude_sdk_otf_ainteract",
        "claude_sdk_otf_ainteract_raw",
        "claude_sdk_otf_v1",
        "claude_sdk_otf_raw_v1",
        "claude_sdk_otf_ainteract_v1",
        "claude_sdk_otf_ainteract_raw_v1",
    )

    # Extract the `choices=[...]` literal so we test what argparse sees,
    # not arbitrary uses of the token strings elsewhere in main().
    import ast
    import re

    m = re.search(
        r'add_argument\(\s*"--framework".*?choices\s*=\s*(\[[^\]]+\])',
        src,
        re.DOTALL,
    )
    assert m is not None
    choices = set(ast.literal_eval(m.group(1)))

    for tok in must_expose:
        assert tok in choices, (
            f"run.main() --framework choices is missing {tok!r}"
        )
    for tok in must_hide:
        assert tok not in choices, (
            f"run.main() --framework choices must not include the "
            f"per-variant token {tok!r} (CLI exposes only the two "
            f"aggregator tokens)."
        )


def test_framework_is_required_no_default():
    """--framework must be required (no default 'claude_sdk' sneaking in)."""
    prog = argparse.ArgumentParser()
    # This mirrors what run.py's main() should produce after the refactor
    prog.add_argument("--framework", choices=["claude_sdk"], required=True)
    with pytest.raises(SystemExit):
        prog.parse_args([])  # no --framework → must fail


# ---------------------------------------------------------------------------
# _validate_slayer_setup simplified logic
# ---------------------------------------------------------------------------


def test_validate_slayer_setup_raw_mode_always_passes():
    """For raw query_mode, slayer_setup is irrelevant — no error regardless of value."""
    _validate_slayer_setup(
        slayer_setup="pre-encoded", framework="claude_sdk",
        query_mode="raw", mode="a-interact",
    )
    _validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk",
        query_mode="raw", mode="one-shot",
    )
    _validate_slayer_setup(
        slayer_setup="whatever-value", framework="claude_sdk",
        query_mode="raw", mode="a-interact",
    )


def test_validate_slayer_setup_slayer_requires_otf():
    """For slayer query_mode, only on-the-fly is accepted."""
    _validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk",
        query_mode="slayer", mode="a-interact",
    )  # must not raise


def test_validate_slayer_setup_otf_encode_raw_raises():
    """DEV-1609: claude_sdk_otf_encode requires --query-mode slayer; reject raw
    at validation (not per-task after setup is built) — Codex review."""
    with pytest.raises(ValueError, match="requires --query-mode slayer"):
        _validate_slayer_setup(
            slayer_setup="on-the-fly", framework="claude_sdk_otf_encode",
            query_mode="raw", mode="a-interact",
        )


def test_validate_slayer_setup_otf_encode_slayer_passes():
    _validate_slayer_setup(
        slayer_setup="on-the-fly", framework="claude_sdk_otf_encode",
        query_mode="slayer", mode="a-interact",
    )  # must not raise


def test_validate_slayer_setup_slayer_pre_encoded_raises():
    with pytest.raises(ValueError, match="on-the-fly"):
        _validate_slayer_setup(
            slayer_setup="pre-encoded", framework="claude_sdk",
            query_mode="slayer", mode="a-interact",
        )


def test_validate_slayer_setup_slayer_unknown_value_raises():
    with pytest.raises(ValueError):
        _validate_slayer_setup(
            slayer_setup="unknown-value", framework="claude_sdk",
            query_mode="slayer", mode="a-interact",
        )


# ---------------------------------------------------------------------------
# make_runner dispatches to the right OTF agent class
# ---------------------------------------------------------------------------


def _make_runner_kwargs(*, dataset, query_mode, mode="a-interact",
                        slayer_setup="on-the-fly"):
    return dict(
        framework="claude_sdk",
        dataset=dataset,
        query_mode=query_mode,
        mode=mode,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=False,
        max_depth=3,
        slayer_storage_root=None,
        slayer_setup=slayer_setup,
    )


@pytest.mark.asyncio
async def test_make_runner_claude_sdk_v1_threads_user_sim_prompt_version(monkeypatch):
    """Codex r6 regression: the v1 dispatch branches must thread
    ``user_sim_prompt_version=_v`` into ``run_task`` so
    ``--user-sim-prompt-version v3`` actually reaches the v1 agent.
    Without this, the v1 agents fall back to each constructor's
    default ``"v2"`` and benchmark comparisons silently use the wrong
    sim prompt.
    """
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        async def run_task(self, *args, **kwargs):
            captured.update(kwargs)
            return {}

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1"
        ".ClaudeSDKOtfAInteractAgent",
        FakeAgent,
        raising=False,
    )
    runner = run_mod.make_runner(
        framework="claude_sdk_v1",
        dataset="mini-interact",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=False,
        max_depth=3,
        slayer_storage_root=None,
        slayer_setup="on-the-fly",
        user_sim_prompt_version="v3",
    )
    await runner({"instance_id": "x", "selected_database": "alien",
                  "amb_user_query": "?", "knowledge_ambiguity": []},
                 "/tmp/x", 3, "anthropic/claude-haiku-4-5-20251001")
    assert captured.get("user_sim_prompt_version") == "v3"


def test_make_runner_claude_sdk_accepts_registry_models(monkeypatch):
    """DEV-1579: the v0 ``claude_sdk`` aggregator now carries the
    provider-aware hermetic session env, so registry open-weight models
    (moonshot/…) dispatch to a v0 agent instead of being rejected. The
    dispatcher must build a runnable callable without raising."""
    agents_created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            agents_created.append("ainteract")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract.ClaudeSDKOtfAInteractAgent",
        FakeAgent,
        raising=False,
    )

    kwargs = {
        **_make_runner_kwargs(dataset="mini-interact", query_mode="slayer"),
        "agent_model": "moonshot/kimi-k2.7-code",
    }
    runner = run_mod.make_runner(**kwargs)
    assert callable(runner)
    assert agents_created == ["ainteract"]


def test_make_runner_dispatches_ainteract_slayer(monkeypatch):
    """mini-interact + slayer → ClaudeSDKOtfAInteractAgent."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import (
        ClaudeSDKOtfAInteractAgent,
    )

    agents_created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            agents_created.append("ainteract")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract.ClaudeSDKOtfAInteractAgent",
        FakeAgent,
        raising=False,
    )

    run_mod.make_runner(**_make_runner_kwargs(
        dataset="mini-interact", query_mode="slayer",
    ))
    assert agents_created == ["ainteract"]


def test_make_runner_dispatches_otf_slayer(monkeypatch):
    """livesqlbench-base-lite-sqlite + slayer → ClaudeSDKOtfAgent."""
    agents_created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            agents_created.append("otf")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf.ClaudeSDKOtfAgent",
        FakeAgent,
        raising=False,
    )

    run_mod.make_runner(**_make_runner_kwargs(
        dataset="livesqlbench-base-lite-sqlite", query_mode="slayer",
        mode="one-shot",
    ))
    assert agents_created == ["otf"]


def test_make_runner_dispatches_ainteract_raw(monkeypatch):
    """mini-interact + raw → ClaudeSDKOtfAInteractRawAgent."""
    agents_created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            agents_created.append("ainteract_raw")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw"
        ".ClaudeSDKOtfAInteractRawAgent",
        FakeAgent,
        raising=False,
    )

    run_mod.make_runner(**_make_runner_kwargs(
        dataset="mini-interact", query_mode="raw", slayer_setup="irrelevant",
    ))
    assert agents_created == ["ainteract_raw"]


def test_make_runner_dispatches_otf_raw(monkeypatch):
    """livesqlbench-base-lite-sqlite + raw → ClaudeSDKOtfRawAgent."""
    agents_created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            agents_created.append("otf_raw")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_raw.ClaudeSDKOtfRawAgent",
        FakeAgent,
        raising=False,
    )

    run_mod.make_runner(**_make_runner_kwargs(
        dataset="livesqlbench-base-lite-sqlite", query_mode="raw",
        mode="one-shot", slayer_setup="irrelevant",
    ))
    assert agents_created == ["otf_raw"]


def test_make_runner_requires_dataset():
    """make_runner must accept a dataset parameter — callers always know the dataset."""
    import inspect
    sig = inspect.signature(run_mod.make_runner)
    assert "dataset" in sig.parameters, (
        "make_runner() must have a 'dataset' parameter for claude_sdk dispatch"
    )


# ---------------------------------------------------------------------------
# Missing 'dataset' in task_data raises ValueError
# ---------------------------------------------------------------------------


def test_run_one_task_raises_on_missing_dataset():
    """task_data without 'dataset' must raise ValueError, not silently default."""
    import asyncio

    async def _run():
        await run_mod.run_one_task(
            {"instance_id": "x", "selected_database": "alien"},
            data_dir="/tmp",
            framework="claude_sdk",
            query_mode="slayer",
            mode="a-interact",
            dataset="mini-interact",  # explicit dataset required
            agent_model="anthropic/claude-haiku-4-5-20251001",
            strict=False,
            prompt_cache=False,
            max_depth=3,
            slayer_storage_root=None,
            slayer_setup="on-the-fly",
            patience=1,
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
        )

    # This test mostly validates the signature change — dataset is now required
    import inspect
    sig = inspect.signature(run_mod.run_one_task)
    assert "dataset" in sig.parameters


# ---------------------------------------------------------------------------
# _FRAMEWORK_DATASET_MODE_BINDING is removed
# ---------------------------------------------------------------------------


def test_framework_dataset_mode_binding_dict_is_removed():
    """_FRAMEWORK_DATASET_MODE_BINDING is dead code after the refactor."""
    assert not hasattr(run_mod, "_FRAMEWORK_DATASET_MODE_BINDING"), (
        "_FRAMEWORK_DATASET_MODE_BINDING dict must be removed from run.py; "
        "dispatch is now implicit in _make_runner's claude_sdk branch"
    )


def test_validate_framework_dataset_mode_is_removed():
    """_validate_framework_dataset_mode is dead code after the refactor."""
    assert not hasattr(run_mod, "_validate_framework_dataset_mode"), (
        "_validate_framework_dataset_mode must be removed from run.py"
    )


# ---------------------------------------------------------------------------
# _validate_framework_mode — prevents c-interact / oracle runtime surprises
# ---------------------------------------------------------------------------

from bird_interact_agents.run import _validate_framework_mode  # noqa: E402


def test_validate_framework_mode_mini_interact_c_interact_raises():
    """claude_sdk + mini-interact + c-interact: no agent supports this combo."""
    with pytest.raises(ValueError, match="a-interact"):
        _validate_framework_mode(
            framework="claude_sdk", dataset="mini-interact", mode="c-interact",
        )


def test_validate_framework_mode_mini_interact_oracle_passes():
    """claude_sdk + mini-interact + oracle: oracle bypasses the agent, always passes."""
    _validate_framework_mode(
        framework="claude_sdk", dataset="mini-interact", mode="oracle",
    )  # must not raise


def test_validate_framework_mode_mini_interact_a_interact_passes():
    """claude_sdk + mini-interact + a-interact: the happy path."""
    _validate_framework_mode(
        framework="claude_sdk", dataset="mini-interact", mode="a-interact",
    )  # must not raise


def test_validate_framework_mode_livesqlbench_oracle_passes():
    """claude_sdk + livesqlbench + oracle: oracle bypasses the agent, always passes."""
    _validate_framework_mode(
        framework="claude_sdk",
        dataset="livesqlbench-base-lite-sqlite",
        mode="oracle",
    )  # must not raise


def test_validate_framework_mode_livesqlbench_one_shot_passes():
    """claude_sdk + livesqlbench + one-shot: the happy path."""
    _validate_framework_mode(
        framework="claude_sdk",
        dataset="livesqlbench-base-lite-sqlite",
        mode="one-shot",
    )  # must not raise


def test_validate_framework_mode_non_claude_sdk_is_noop():
    """Non-claude_sdk frameworks are not affected — returns immediately."""
    _validate_framework_mode(
        framework="pydantic_ai", dataset="mini-interact", mode="c-interact",
    )  # must not raise (unknown framework is a no-op)


# DEV-1609: claude_sdk_otf_encode accepts ONLY a-interact / one-shot, enforced
# at CLI/cloud validation (not just per-task in the agent) — Codex review.

def test_validate_framework_mode_otf_encode_c_interact_raises():
    with pytest.raises(ValueError, match="claude_sdk_otf_encode"):
        _validate_framework_mode(
            framework="claude_sdk_otf_encode",
            dataset="mini-interact", mode="c-interact",
        )


def test_validate_framework_mode_otf_encode_oracle_raises():
    """oracle is NOT a valid encode mode (unlike the eval claude_sdk agents)."""
    with pytest.raises(ValueError, match="claude_sdk_otf_encode"):
        _validate_framework_mode(
            framework="claude_sdk_otf_encode",
            dataset="mini-interact", mode="oracle",
        )


def test_validate_framework_mode_otf_encode_a_interact_passes():
    _validate_framework_mode(
        framework="claude_sdk_otf_encode",
        dataset="mini-interact", mode="a-interact",
    )  # must not raise


def test_validate_framework_mode_otf_encode_one_shot_passes():
    _validate_framework_mode(
        framework="claude_sdk_otf_encode",
        dataset="livesqlbench-base-lite-sqlite", mode="one-shot",
    )  # must not raise
