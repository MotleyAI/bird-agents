"""DEV-1649: --save-edited-models / --apply-edited-models CLI flags —
argparse plumbing, validation guards, and threading to the agent ctor.

Mirrors test_slayer_setup_flag.py conventions (drive run.main with synth argv,
capture run_evaluation kwargs; validation via make_runner).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bird_interact_agents import run as run_module


@pytest.fixture(autouse=True)
def _dev1640_force_legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


def _argv_base(tmp_path: Path) -> list[str]:
    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "mini-interact"
    db_path.mkdir()
    return [
        "bird-interact",
        "--dataset", "mini-interact",
        "--agent-model", "anthropic/claude-sonnet-4-5",
        "--no-subscription-auth",
        "--data", str(data),
        "--db-path", str(db_path),
        "--output", str(tmp_path / "out.json"),
        "--limit", "0",
    ]


def _drive_main(monkeypatch, argv: list[str]) -> dict:
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {"metrics": "fake"}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", argv)
    run_module.main()
    return captured


def _slayer_argv(tmp_path):
    # --framework claude_sdk dispatches to the on-the-fly OTF agents by
    # benchmark/mode; the otf agent names are not user-facing CLI choices.
    return _argv_base(tmp_path) + [
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
    ]


# --------------------------------------------------------------------------
# argparse plumbing
# --------------------------------------------------------------------------


def test_flags_default_false(monkeypatch, tmp_path):
    kwargs = _drive_main(monkeypatch, _slayer_argv(tmp_path))
    assert kwargs.get("save_edited_models") is False
    assert kwargs.get("apply_edited_models") is False


def test_save_flag_parse_and_plumb(monkeypatch, tmp_path):
    argv = _slayer_argv(tmp_path) + ["--save-edited-models"]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("save_edited_models") is True
    assert kwargs.get("apply_edited_models") is False


def test_apply_flag_parse_and_plumb(monkeypatch, tmp_path):
    argv = _slayer_argv(tmp_path) + ["--apply-edited-models"]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("apply_edited_models") is True
    assert kwargs.get("save_edited_models") is False


def test_save_and_apply_flags_are_mutually_exclusive_cli(monkeypatch, tmp_path):
    # argparse mutually-exclusive group → SystemExit (exit code 2) with both.
    argv = _slayer_argv(tmp_path) + [
        "--save-edited-models", "--apply-edited-models",
    ]
    with pytest.raises(SystemExit):
        _drive_main(monkeypatch, argv)


def test_flags_threaded_to_agent_constructor(monkeypatch, tmp_path):
    captured_init: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured_init.update(kwargs)

        async def run_task(self, *a, **k):  # pragma: no cover
            return {"task_id": "noop"}

    import bird_interact_agents.agents.claude_sdk_otf_ainteract as pkg

    monkeypatch.setattr(pkg, "ClaudeSDKOtfAInteractAgent", FakeAgent)

    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "mini-interact"
    db_path.mkdir()

    import asyncio

    asyncio.run(run_module.run_evaluation(
        data_path=str(data),
        data_dir=str(db_path),
        output_path=str(tmp_path / "out.json"),
        mode="a-interact",
        query_mode="slayer",
        framework="claude_sdk_otf_ainteract",
        slayer_setup="on-the-fly",
        save_edited_models=True,
        apply_edited_models=False,
        dataset="mini-interact",
        limit=0,
    ))

    assert captured_init.get("save_edited_models") is True
    assert captured_init.get("apply_edited_models") is False


# --------------------------------------------------------------------------
# validation guards
# --------------------------------------------------------------------------


def _make_runner(**over):
    kwargs = dict(
        framework="claude_sdk_otf_ainteract",
        dataset="mini-interact",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        strict=False,
        prompt_cache=True,
        max_depth=3,
        slayer_storage_root=None,
        slayer_setup="on-the-fly",
    )
    kwargs.update(over)
    return run_module.make_runner(**kwargs)


def test_make_runner_accepts_slayer_on_the_fly_with_save():
    _make_runner(save_edited_models=True)


def test_make_runner_accepts_slayer_on_the_fly_with_apply():
    _make_runner(apply_edited_models=True)


def test_make_runner_rejects_both_flags():
    # Defensive guard in _validate_slayer_setup — catches programmatic / cloud
    # callers even though the CLI already blocks it via the argparse group.
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_runner(save_edited_models=True, apply_edited_models=True)


def test_make_runner_rejects_flags_with_raw():
    with pytest.raises(ValueError):
        _make_runner(
            framework="claude_sdk", query_mode="raw", slayer_setup="on-the-fly",
            save_edited_models=True,
        )


def test_make_runner_rejects_flags_with_pre_encoded():
    with pytest.raises(ValueError):
        _make_runner(
            framework="claude_sdk", slayer_setup="pre-encoded",
            pre_encoded_source="otf", apply_edited_models=True,
        )


def test_make_runner_rejects_flags_with_otf_encode():
    with pytest.raises(ValueError):
        _make_runner(
            framework="claude_sdk_otf_encode", save_edited_models=True,
        )


def test_cli_fast_fails_on_bad_flag_combo(monkeypatch, tmp_path):
    """The main() fail-fast validation (before postgres bootstrap) must reject
    a bad flag combo, not let it slip to run_evaluation (process-reviews
    CodeRabbit major). argparse's parser.error exits with code 2."""
    argv = _argv_base(tmp_path) + [
        "--framework", "claude_sdk",
        "--query-mode", "raw",
        "--mode", "a-interact",
        "--save-edited-models",
    ]

    async def fail_run_evaluation(**_kwargs):  # pragma: no cover - must not run
        pytest.fail("run_evaluation reached — fail-fast validation did not fire")

    monkeypatch.setattr(run_module, "run_evaluation", fail_run_evaluation)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        run_module.main()
    # argparse's parser.error() exits with code 2.
    assert exc_info.value.code == 2
