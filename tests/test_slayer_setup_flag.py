"""Tests for the ``--slayer-setup`` CLI flag and its fail-fast guards.

The flag is orthogonal to ``--framework`` / ``--query-mode`` / ``--mode``
but ``slayer`` query_mode REQUIRES ``on-the-fly``; using ``pre-encoded``
with slayer must fail fast BEFORE any task runs.

These tests drive ``run.main`` directly with synthesised argv so they
exercise both argparse and the validation hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bird_interact_agents import run as run_module


def _argv_base(tmp_path: Path) -> list[str]:
    """Minimum required CLI flags so argparse doesn't error on missing
    required arguments (``--data``, ``--db-path``)."""
    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "mini-interact"
    db_path.mkdir()
    return [
        "bird-interact",
        "--dataset", "mini-interact",
        "--data", str(data),
        "--db-path", str(db_path),
        "--output", str(tmp_path / "out.json"),
        "--limit", "0",
    ]


def _drive_main(monkeypatch, argv: list[str]) -> dict:
    """Run ``run.main`` with the given argv. Captures the call into
    ``run_evaluation`` so tests can assert the plumbed kwargs without
    actually executing the evaluation loop. Returns the captured kwargs.
    """
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {"metrics": "fake"}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", argv)
    run_module.main()
    return captured


# ---------------------------------------------------------------------------
# Default behaviour: flag omitted → pre-encoded
# ---------------------------------------------------------------------------


def _assert_failed_validation(
    capsys, exc: BaseException, *, must_contain: list[str],
) -> None:
    """Assert that a fail-fast validation triggered SystemExit (argparse
    path) or ValueError (plain raise), AND that the user-facing error
    message contains all the ``must_contain`` substrings.

    For ``SystemExit``, ``argparse.ArgumentParser.error`` writes the
    message to stderr and exits with code 2; ``str(SystemExit)`` is
    only the integer code, so we read stderr via ``capsys`` instead.
    For ``ValueError``, the message is in ``str(exc)``. Either path is
    acceptable as long as the user sees all required substrings.
    """
    if isinstance(exc, SystemExit):
        captured = capsys.readouterr()
        msg = (captured.err or "") + (captured.out or "")
    else:
        msg = str(exc)
    for substr in must_contain:
        assert substr in msg, (
            f"validation error must mention {substr!r}; got: {msg!r}"
        )


def test_default_is_pre_encoded(monkeypatch, tmp_path: Path):
    """When ``--slayer-setup`` is omitted, ``run_evaluation`` receives
    ``slayer_setup='pre-encoded'`` so the existing pre-encoded path
    runs unchanged. Use --query-mode raw so pre-encoded is valid."""
    argv = _argv_base(tmp_path) + [
        "--framework", "claude_sdk",
        "--query-mode", "raw",
        "--mode", "a-interact",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("slayer_setup") == "pre-encoded"


# ---------------------------------------------------------------------------
# Happy path: on-the-fly + claude_sdk + slayer + a-interact
# ---------------------------------------------------------------------------


def test_on_the_fly_with_valid_combo_is_accepted(monkeypatch, tmp_path: Path):
    argv = _argv_base(tmp_path) + [
        "--slayer-setup", "on-the-fly",
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("slayer_setup") == "on-the-fly"
    assert kwargs.get("framework") == "claude_sdk"
    assert kwargs.get("query_mode") == "slayer"
    assert kwargs.get("mode") == "a-interact"


def test_on_the_fly_is_plumbed_to_agent_constructor(monkeypatch, tmp_path: Path):
    """End-to-end plumbing: ``--slayer-setup on-the-fly`` must reach the
    ``PydanticAIRecursiveAgent`` constructor (or its run-time entry point),
    not just stop at ``run_evaluation``. We monkeypatch the agent class so
    we can capture the kwarg without spinning up a real model client."""
    captured_init: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured_init.update(kwargs)
            self.slayer_setup = kwargs.get("slayer_setup")

        async def run_task(self, *args, **kwargs):  # pragma: no cover
            return {"task_id": "noop"}

    from bird_interact_agents.agents.pydantic_ai_recursive import (
        agent as par_agent_mod,
    )
    monkeypatch.setattr(
        par_agent_mod, "PydanticAIRecursiveAgent", FakeAgent,
    )

    # Patch the symbol where run.py actually imports it (the alias inside
    # run_evaluation), via the module-level import path used at call time.
    import bird_interact_agents.agents.pydantic_ai_recursive as par_pkg
    monkeypatch.setattr(par_pkg, "PydanticAIRecursiveAgent", FakeAgent)

    # Real evaluation drives load_tasks; we feed it an empty jsonl so the
    # task loop is a no-op but the agent constructor is still hit.
    # Use pydantic_ai_recursive directly via run_evaluation (not CLI) to
    # avoid the CLI's framework choices restriction.
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
        framework="pydantic_ai_recursive",
        slayer_setup="on-the-fly",
        dataset="mini-interact",
        limit=0,
    ))

    assert captured_init.get("slayer_setup") == "on-the-fly", (
        f"slayer_setup must reach the agent constructor; "
        f"captured init kwargs: {captured_init}"
    )


# ---------------------------------------------------------------------------
# Fail-fast guards — validate before any task runs
# ---------------------------------------------------------------------------


def test_pre_encoded_with_slayer_is_rejected_by_run_evaluation(
    tmp_path: Path,
):
    """``pre-encoded`` + ``slayer`` query mode must fail fast — the only
    valid setup for slayer is on-the-fly."""
    import asyncio

    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "mini-interact"
    db_path.mkdir()

    with pytest.raises(ValueError) as exc_info:
        asyncio.run(run_module.run_evaluation(
            data_path=str(data),
            data_dir=str(db_path),
            output_path=str(tmp_path / "out.json"),
            mode="a-interact",
            query_mode="slayer",
            framework="claude_sdk",
            slayer_setup="pre-encoded",
            dataset="mini-interact",
            limit=0,
        ))
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "on-the-fly" in msg


def test_make_runner_accepts_on_the_fly_for_any_framework():
    """With the simplified validator, on-the-fly is always accepted for
    slayer query mode (framework is no longer checked)."""
    # Should not raise
    run_module.make_runner(
        framework="claude_sdk",
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


def test_make_runner_rejects_pre_encoded_with_slayer():
    """``pre-encoded`` + ``slayer`` is always rejected by make_runner."""
    with pytest.raises(ValueError) as exc_info:
        run_module.make_runner(
            framework="claude_sdk",
            dataset="mini-interact",
            query_mode="slayer",
            mode="a-interact",
            agent_model="anthropic/claude-sonnet-4-5",
            strict=False,
            prompt_cache=True,
            max_depth=3,
            slayer_storage_root=None,
            slayer_setup="pre-encoded",
        )
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "on-the-fly" in msg


async def test_run_one_task_rejects_pre_encoded_with_slayer(
    tmp_path: Path,
):
    """``run_one_task`` must validate ``slayer_setup`` against query_mode
    before constructing the runner."""
    data_dir = tmp_path / "mini-interact"
    data_dir.mkdir()
    with pytest.raises(ValueError) as exc_info:
        await run_module.run_one_task(
            task_data={"instance_id": "x", "selected_database": "x"},
            data_dir=str(data_dir),
            dataset="mini-interact",
            framework="claude_sdk",
            query_mode="slayer",
            mode="a-interact",
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            patience=3,
            strict=False,
            use_audited_gold_sql=False,
            prompt_cache=True,
            max_depth=3,
            slayer_storage_root=None,
            slayer_setup="pre-encoded",
        )
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "on-the-fly" in msg


async def test_run_evaluation_rejects_pre_encoded_with_slayer(
    tmp_path: Path,
):
    """A programmatic caller that bypasses the CLI parser and calls
    ``run_evaluation(...)`` directly with ``pre-encoded`` + ``slayer``
    must get a fail-fast error."""
    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "mini-interact"
    db_path.mkdir()
    output = tmp_path / "out.json"

    with pytest.raises(ValueError) as exc_info:
        await run_module.run_evaluation(
            data_path=str(data),
            data_dir=str(db_path),
            output_path=str(output),
            mode="a-interact",
            query_mode="slayer",
            framework="claude_sdk",
            slayer_setup="pre-encoded",
            dataset="mini-interact",
            limit=0,
        )
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "on-the-fly" in msg


# ---------------------------------------------------------------------------
# DEV-1468 — --otf-rebuild (renamed from --otf-rebuild-reference, old name
# kept as a hidden alias) is plumbed to run_evaluation as `otf_rebuild`.
# ---------------------------------------------------------------------------


def _otf_argv(tmp_path: Path, *extra: str) -> list[str]:
    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "mini-interact"
    db_path.mkdir()
    return [
        "bird-interact",
        "--dataset", "mini-interact",
        "--data", str(data),
        "--db-path", str(db_path),
        "--output", str(tmp_path / "out.json"),
        "--limit", "0",
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
        "--slayer-setup", "on-the-fly",
        *extra,
    ]


def test_otf_rebuild_flag_plumbed(monkeypatch, tmp_path: Path):
    kwargs = _drive_main(monkeypatch, _otf_argv(tmp_path, "--otf-rebuild"))
    assert kwargs.get("otf_rebuild") is True


def test_otf_rebuild_reference_alias_still_works(monkeypatch, tmp_path: Path):
    """The old --otf-rebuild-reference name stays as a hidden alias for
    git/script continuity; it sets the same `otf_rebuild` flag."""
    kwargs = _drive_main(monkeypatch, _otf_argv(tmp_path, "--otf-rebuild-reference"))
    assert kwargs.get("otf_rebuild") is True


def test_otf_rebuild_default_false(monkeypatch, tmp_path: Path):
    kwargs = _drive_main(monkeypatch, _otf_argv(tmp_path))
    assert kwargs.get("otf_rebuild") is False
