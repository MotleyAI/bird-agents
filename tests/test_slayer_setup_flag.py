"""Tests for the ``--pre-encoded-models`` CLI flag and its derivation /
fail-fast guards (DEV-1586).

The retired ``--slayer-setup`` flag is replaced by the user-facing
``--pre-encoded-models {otf,custom}``. The internal ``slayer_setup`` is
DERIVED from it (``"pre-encoded"`` when set, else ``"on-the-fly"``) and is
still threaded into the runner factory because cloud routing consumes it.

These tests drive ``run.main`` directly with synthesised argv so they
exercise both argparse and the validation hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bird_interact_agents import run as run_module

# DEV-1640: these tests pin the LOCAL in-process per-task wiring / grading by
# monkeypatching agents + graders + loaders, which a spawned worker process
# cannot see. The process pool is now the default, so route run_evaluation
# through the retained legacy single-loop path (identical per-task wiring).
@pytest.fixture(autouse=True)
def _dev1640_force_legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


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
        # --agent-model is REQUIRED since the cloud-alignment change.
        "--agent-model", "anthropic/claude-sonnet-4-5",
        # claude_sdk* + Anthropic requires an explicit subscription-auth choice.
        "--no-subscription-auth",
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
# Default behaviour: flag omitted → on-the-fly (derived)
# ---------------------------------------------------------------------------


def test_default_is_on_the_fly(monkeypatch, tmp_path: Path):
    """When ``--pre-encoded-models`` is omitted, ``run_evaluation`` receives
    ``slayer_setup='on-the-fly'`` and ``pre_encoded_source=None``."""
    argv = _argv_base(tmp_path) + [
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("slayer_setup") == "on-the-fly"
    assert kwargs.get("pre_encoded_source") is None


@pytest.mark.parametrize("source", ["otf", "custom"])
def test_pre_encoded_models_derives_and_plumbs(monkeypatch, tmp_path: Path, source):
    """``--pre-encoded-models <src>`` derives slayer_setup='pre-encoded' and
    threads both the derived value and the source into ``run_evaluation``."""
    argv = _argv_base(tmp_path) + [
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
        "--pre-encoded-models", source,
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("slayer_setup") == "pre-encoded"
    assert kwargs.get("pre_encoded_source") == source


def test_slayer_setup_flag_retired(monkeypatch, tmp_path: Path):
    """``--slayer-setup`` is retired from the user-facing local CLI."""
    argv = _argv_base(tmp_path) + [
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
        "--slayer-setup", "on-the-fly",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        run_module.main()


# ---------------------------------------------------------------------------
# End-to-end plumbing into the agent constructor
# ---------------------------------------------------------------------------


def test_pre_encoded_source_is_plumbed_to_agent_constructor(
    monkeypatch, tmp_path: Path,
):
    """``--pre-encoded-models custom`` must reach the claude_sdk OTF agent
    constructor as ``pre_encoded_source``."""
    captured_init: dict = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured_init.update(kwargs)

        async def run_task(self, *args, **kwargs):  # pragma: no cover
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
        slayer_setup="pre-encoded",
        pre_encoded_source="custom",
        dataset="mini-interact",
        limit=0,
    ))

    assert captured_init.get("pre_encoded_source") == "custom", (
        f"pre_encoded_source must reach the agent constructor; "
        f"captured init kwargs: {captured_init}"
    )


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_make_runner_accepts_pre_encoded_otf():
    """pre-encoded + otf is accepted (was rejected pre-DEV-1586)."""
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
        pre_encoded_source="otf",
    )


def test_make_runner_accepts_on_the_fly_default():
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


def test_make_runner_rejects_inconsistent_setup_and_source():
    """A derived/source mismatch (pre-encoded slayer_setup without a source,
    or vice-versa) is rejected."""
    with pytest.raises(ValueError):
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
            pre_encoded_source=None,
        )


def test_make_runner_rejects_pre_encoded_for_otf_encode_framework():
    """pre-encoded mode is for the claude_sdk consumers only."""
    with pytest.raises(ValueError):
        run_module.make_runner(
            framework="pydantic_ai_otf_encode",
            dataset="mini-interact",
            query_mode="slayer",
            mode="a-interact",
            agent_model="anthropic/claude-sonnet-4-5",
            strict=False,
            prompt_cache=True,
            max_depth=3,
            slayer_storage_root=None,
            slayer_setup="pre-encoded",
            pre_encoded_source="otf",
        )


async def test_run_one_task_threads_pre_encoded_source(tmp_path: Path):
    """``run_one_task`` accepts and validates ``pre_encoded_source``."""
    data_dir = tmp_path / "mini-interact"
    data_dir.mkdir()
    # Mismatch (pre-encoded source on an unsupported framework) → fail fast.
    with pytest.raises(ValueError):
        await run_module.run_one_task(
            task_data={"instance_id": "x", "selected_database": "x"},
            data_dir=str(data_dir),
            dataset="mini-interact",
            framework="pydantic_ai_otf_encode",
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
            pre_encoded_source="otf",
        )


# ---------------------------------------------------------------------------
# --otf-rebuild is plumbed to run_evaluation as `otf_rebuild` (unchanged by
# DEV-1586; on-the-fly is now the default so no --slayer-setup needed).
# ---------------------------------------------------------------------------


def _otf_argv(tmp_path: Path, *extra: str) -> list[str]:
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
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--mode", "a-interact",
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
