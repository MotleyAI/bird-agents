"""Tests for `run.py`'s integration of the new framework.

Two things to lock down:

* `--framework pydantic_ai_otf_encode` is accepted by the CLI.
* The framework selection branch instantiates `PydanticAIOtfEncodeAgent`.
* The setup-mode guard requires `--slayer-setup on-the-fly` with this
  framework (mirrors the existing guard for `pydantic_ai_recursive`).
"""

from __future__ import annotations

import pytest

# DEV-1640: these tests pin the LOCAL in-process per-task wiring / grading by
# monkeypatching agents + graders + loaders, which a spawned worker process
# cannot see. The process pool is now the default, so route run_evaluation
# through the retained legacy single-loop path (identical per-task wiring).
@pytest.fixture(autouse=True)
def _dev1640_force_legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


def _framework_choices_from_parser():
    """Introspect `run.py`'s argparse parser without invoking it. We
    monkey-walk `main` up to the `parse_args` boundary by capturing
    the parser instance via a sys.argv with `--help`.

    The cheap alternative — `run_mod._parser()` — doesn't exist in
    `run.py` today and the plan does NOT require a new public helper
    (per Codex test-review finding 10). Instead we read the inline
    `choices=[...]` argument from the source AST.
    """
    import ast
    import inspect

    from bird_interact_agents import run as run_mod

    src = inspect.getsource(run_mod.main)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(getattr(node.func, "attr", None), "lower", lambda: "")() == "add_argument"
        ):
            # First positional arg is the flag name; check for "--framework".
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--framework":
                for kw in node.keywords:
                    if kw.arg == "choices" and isinstance(kw.value, ast.List):
                        return {
                            elt.value for elt in kw.value.elts
                            if isinstance(elt, ast.Constant)
                        }
    raise AssertionError("could not find --framework choices in run.main")


def test_framework_choice_accepted_by_arg_parser():
    """`claude_sdk` is in the argparse `choices` list."""
    choices = _framework_choices_from_parser()
    assert "claude_sdk" in choices


def test_framework_choices_still_include_existing_frameworks():
    choices = _framework_choices_from_parser()
    assert {"claude_sdk"}.issubset(choices)


@pytest.mark.asyncio
async def test_run_evaluation_branches_to_new_agent(monkeypatch, tmp_path):
    """When `framework='claude_sdk'` with a non-one-shot dataset and slayer query
    mode, `run_evaluation` instantiates `ClaudeSDKOtfAInteractAgent`.
    We intercept the constructor to confirm and short-circuit the rest."""
    from bird_interact_agents import run as run_mod

    constructed = []

    class _Sentinel(Exception):
        pass

    class _FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise _Sentinel("stop here")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract."
        "ClaudeSDKOtfAInteractAgent",
        _FakeAgent,
        raising=False,
    )
    # Also stub the loader so we don't need a real data file.
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **kw: [])

    with pytest.raises(_Sentinel):
        await run_mod.run_evaluation(
            data_path="/tmp/x.jsonl", data_dir="/tmp",
            output_path=str(tmp_path / "eval.json"),
            mode="a-interact", query_mode="slayer",
            framework="claude_sdk",
            dataset="mini-interact",
            slayer_setup="on-the-fly",
        )
    assert len(constructed) == 1
    # The agent was built with the on-the-fly setup mode.
    assert constructed[0].get("slayer_setup") == "on-the-fly"


def test_cli_rejects_pre_encoded_for_otf_encode_framework(monkeypatch):
    """DEV-1586: --pre-encoded-models is for the claude_sdk consumers only;
    passing it with pydantic_ai_otf_encode (the encoder) is rejected."""
    from bird_interact_agents import run as run_mod
    import sys

    argv = [
        "prog",
        "--dataset", "mini-interact",
        "--agent-model", "anthropic/claude-sonnet-4-5",
        "--framework", "pydantic_ai_otf_encode",
        "--pre-encoded-models", "otf",
        "--query-mode", "slayer",
        "--mode", "a-interact",
        "--data", "/tmp/x.jsonl",
        "--db-path", "/tmp",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_mod.main()


def test_cli_defaults_to_on_the_fly(
    monkeypatch, tmp_path,
):
    """Parser accepts slayer mode with no --pre-encoded-models; slayer_setup
    derives to on-the-fly. We monkeypatch run_evaluation to a noop."""
    from bird_interact_agents import run as run_mod
    import sys

    argv = [
        "prog",
        "--dataset", "mini-interact",
        "--agent-model", "anthropic/claude-sonnet-4-5",
        "--no-subscription-auth",
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
        "--data", "/tmp/x.jsonl",
        "--db-path", "/tmp",
        "--output", str(tmp_path / "eval.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    called = {}

    async def fake_run(**kw):
        called.update(kw)
        return {}

    monkeypatch.setattr(run_mod, "run_evaluation", fake_run)
    run_mod.main()
    assert called["framework"] == "claude_sdk"
    assert called["slayer_setup"] == "on-the-fly"
    assert called["pre_encoded_source"] is None
