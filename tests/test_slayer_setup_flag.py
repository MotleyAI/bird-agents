"""Tests for the ``--slayer-setup`` CLI flag and its fail-fast guards.

The flag is orthogonal to ``--framework`` / ``--query-mode`` / ``--mode``
but only ``pydantic_ai_recursive + slayer + a-interact`` accepts the
``on-the-fly`` value. Invalid combinations must fail BEFORE any task
runs (Codex finding: today's per-task error path stamps a bogus result
row instead of failing fast).

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
    runs unchanged."""
    argv = _argv_base(tmp_path) + [
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
        "--mode", "a-interact",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("slayer_setup") == "pre-encoded"


# ---------------------------------------------------------------------------
# Happy path: on-the-fly + pydantic_ai_recursive + slayer + a-interact
# ---------------------------------------------------------------------------


def test_on_the_fly_with_valid_combo_is_accepted(monkeypatch, tmp_path: Path):
    argv = _argv_base(tmp_path) + [
        "--slayer-setup", "on-the-fly",
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
        "--mode", "a-interact",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("slayer_setup") == "on-the-fly"
    assert kwargs.get("framework") == "pydantic_ai_recursive"
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
    argv = _argv_base(tmp_path) + [
        "--slayer-setup", "on-the-fly",
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
        "--mode", "a-interact",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run_module.main()

    assert captured_init.get("slayer_setup") == "on-the-fly", (
        f"slayer_setup must reach the agent constructor; "
        f"captured init kwargs: {captured_init}"
    )


# ---------------------------------------------------------------------------
# Fail-fast guards — Codex finding: validate before any task runs
# ---------------------------------------------------------------------------


def test_on_the_fly_rejects_wrong_framework(monkeypatch, tmp_path: Path, capsys):
    """``on-the-fly`` only makes sense for ``pydantic_ai_recursive`` —
    any other framework must fail fast with a clear error naming both
    flags."""
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):  # pragma: no cover - should not run
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        sys, "argv",
        _argv_base(tmp_path) + [
            "--slayer-setup", "on-the-fly",
            "--framework", "pydantic_ai",
            "--query-mode", "slayer",
            "--mode", "a-interact",
        ],
    )
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        run_module.main()
    assert not captured, "run_evaluation must not be called"
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["slayer-setup", "framework"],
    )


def test_on_the_fly_rejects_raw_query_mode(monkeypatch, tmp_path: Path, capsys):
    """``on-the-fly`` needs SLayer storage; ``--query-mode raw`` makes
    no sense and must fail fast."""
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):  # pragma: no cover
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        sys, "argv",
        _argv_base(tmp_path) + [
            "--slayer-setup", "on-the-fly",
            "--framework", "pydantic_ai_recursive",
            "--query-mode", "raw",
            "--mode", "a-interact",
        ],
    )
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        run_module.main()
    assert not captured
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["slayer-setup", "query-mode"],
    )


def test_make_runner_rejects_invalid_slayer_setup_programmatically():
    """Codex finding (round-4): ``make_runner`` is the public factory
    used by the cloud actor and other throughput-sensitive callers
    that bypass the CLI. A caller passing on-the-fly with an
    unsupported framework must hit a ValueError, not silently get a
    runner that ignores the setting."""
    with pytest.raises(ValueError) as exc_info:
        run_module.make_runner(
            framework="pydantic_ai",      # wrong framework
            query_mode="slayer",
            mode="a-interact",
            agent_model="anthropic/claude-sonnet-4-5",
            strict=False,
            prompt_cache=True,
            max_depth=3,
            slayer_storage_root=None,
            slayer_setup="on-the-fly",
        )
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "slayer_setup" in msg
    assert "framework" in msg


async def test_run_one_task_rejects_invalid_slayer_setup_programmatically(
    tmp_path: Path,
):
    """Codex finding (round-5): ``run_one_task`` is the public one-task
    API the cloud actor uses. It must validate ``slayer_setup`` against
    framework/query_mode/mode before constructing the runner; otherwise
    a cloud caller passing on-the-fly with the wrong framework would
    silently get a pre-encoded runner."""
    data_dir = tmp_path / "mini-interact"
    data_dir.mkdir()
    with pytest.raises(ValueError) as exc_info:
        await run_module.run_one_task(
            task_data={"instance_id": "x", "selected_database": "x"},
            data_dir=str(data_dir),
            framework="pydantic_ai",      # wrong framework
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
            slayer_setup="on-the-fly",
        )
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "slayer_setup" in msg
    assert "framework" in msg


async def test_run_evaluation_rejects_invalid_slayer_setup_programmatically(
    tmp_path: Path,
):
    """Codex finding: a programmatic caller that bypasses the CLI parser
    and calls ``run_evaluation(...)`` directly with an unsupported
    combination must also get a fail-fast error — not silently see
    the option ignored or fail later per task.
    """
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
            framework="pydantic_ai",      # wrong framework
            limit=0,
            slayer_setup="on-the-fly",
        )
    msg = str(exc_info.value)
    assert "slayer-setup" in msg or "slayer_setup" in msg
    assert "framework" in msg


# ---------------------------------------------------------------------------
# DEV-1468 — --otf-rebuild (renamed from --otf-rebuild-reference, old name
# kept as a hidden alias) is plumbed to run_evaluation as `otf_rebuild`.
# ---------------------------------------------------------------------------


def _otf_argv(tmp_path: Path, *extra: str) -> list[str]:
    return _argv_base(tmp_path) + [
        "--framework", "pydantic_ai_otf_encode",
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


def test_on_the_fly_rejects_non_a_interact_mode(
    monkeypatch, tmp_path: Path, capsys,
):
    """The PydanticAI recursive agent only supports a-interact today;
    on-the-fly inherits that constraint and must fail fast on c-interact
    or oracle."""
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):  # pragma: no cover
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        sys, "argv",
        _argv_base(tmp_path) + [
            "--slayer-setup", "on-the-fly",
            "--framework", "pydantic_ai_recursive",
            "--query-mode", "slayer",
            "--mode", "c-interact",
        ],
    )
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        run_module.main()
    assert not captured
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["slayer-setup", "mode"],
    )
