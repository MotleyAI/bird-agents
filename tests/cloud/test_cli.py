"""T35: CLI argument parsing.

Asserts that `bird-interact-cloud submit` exposes all new flags, that the
mode values match the local `bird-interact` CLI, and that mutually exclusive
combinations are rejected.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import cli  # noqa: E402


# ---------------------------------------------------------------------------
# Mode names must match the local CLI (Codex MAJOR #10).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["a-interact", "c-interact", "oracle"])
def test_mode_values_accepted(mode: str) -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", mode,
        ]
    )
    assert ns.mode == mode


def test_unknown_mode_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "interactive",  # not a real local mode
            ]
        )


# ---------------------------------------------------------------------------
# All pass-through flags parse.
# ---------------------------------------------------------------------------


def test_pass_through_flags_parse() -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--framework", "pydantic_ai_recursive",
            "--query-mode", "raw",  # slayer is guarded for cloud (see below)
            "--agent-model", "cerebras/zai-glm-4.7",
            "--instance-ids", "db_a_1,db_a_2,db_a_3",
            "--mode", "c-interact",
            "--use-audited-gold-sql",
            "--max-depth", "5",
            "--no-prompt-cache",
            "--workers", "8",
            "--actors-per-worker", "2",
            "--worker-type", "e2-standard-8",
            "--max-runtime-hours", "12",
            "--patience", "4",
            "--strict",
            "--detach",
        ]
    )
    assert ns.use_audited_gold_sql is True
    assert ns.max_depth == 5
    assert ns.prompt_cache is False
    assert ns.workers == 8
    assert ns.actors_per_worker == 2
    assert ns.worker_type == "e2-standard-8"
    assert ns.max_runtime_hours == 12
    assert ns.patience == 4
    assert ns.strict is True
    assert ns.detach is True
    assert ns.instance_ids == ["db_a_1", "db_a_2", "db_a_3"]


def test_prompt_cache_default_on() -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
        ]
    )
    assert ns.prompt_cache is True


# ---------------------------------------------------------------------------
# --detach --allow-dirty is mutually exclusive.
# ---------------------------------------------------------------------------


def test_cloud_slayer_mode_rejected_at_submit() -> None:
    """Cloud slayer mode isn't implemented yet — submit must reject it fast
    (before image build / cluster bring-up) with a clear message, rather than
    failing per-task mid-run. Tracked for a follow-up PR."""
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai_recursive",
                "--query-mode", "slayer",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
            ]
        )


def test_detach_and_allow_dirty_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
                "--detach",
                "--allow-dirty",
            ]
        )


# ---------------------------------------------------------------------------
# CR#2 — empty `--instance-ids` rejected at argparse time.
# ---------------------------------------------------------------------------


def test_empty_instance_ids_string_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "",
                "--mode", "c-interact",
            ]
        )


def test_empty_instance_ids_file_rejected(tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n  \n")  # whitespace only
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids-file", str(empty),
                "--mode", "c-interact",
            ]
        )


# ---------------------------------------------------------------------------
# Sub-commands present.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sub", ["submit", "fetch", "kill", "list", "build", "resubmit"])
def test_subcommand_registered(sub: str) -> None:
    # All sub-commands at least parse to a known namespace. `fetch` / `kill`
    # / `resubmit` take a run-id positional; `list` / `build` are flagless.
    if sub == "list":
        ns = cli.parse_args(["list"])
    elif sub == "build":
        ns = cli.parse_args(["build"])
    elif sub == "submit":
        ns = cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
            ]
        )
    else:
        ns = cli.parse_args([sub, "some-run-id"])
    assert ns.subcommand == sub
