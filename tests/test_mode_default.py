"""`--mode` defaults per benchmark: one-shot benchmarks default to `one-shot`,
interactive ones to `a-interact`. `--mode` is only needed to select a
non-default mode (`oracle`, or `c-interact` if/when it is wired to an agent).
"""

from __future__ import annotations

import sys

import pytest

from bird_interact_agents.benchmark import all_benchmarks, get_benchmark
from bird_interact_agents.cloud import cli


# ---------------------------------------------------------------------------
# Source of truth: Benchmark.default_mode
# ---------------------------------------------------------------------------


def test_default_mode_one_shot_vs_interactive():
    assert get_benchmark("mini-interact").default_mode == "a-interact"
    assert get_benchmark("livesqlbench-base-lite-sqlite").default_mode == "one-shot"


def test_default_mode_always_supported():
    """Invariant: every benchmark's default_mode is one of its supported_modes,
    so an omitted --mode can never produce an unsupported combo."""
    for b in all_benchmarks():
        assert b.default_mode in b.supported_modes, b.name
        assert b.default_mode == ("one-shot" if b.one_shot else "a-interact")


# ---------------------------------------------------------------------------
# Cloud CLI: --mode optional, derived per benchmark
# ---------------------------------------------------------------------------


def _submit_argv(dataset, mode_args, extra=()):
    return [
        "submit", "--framework", "claude_sdk_v1", "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--user-sim-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1", "--dataset", dataset,
        "--no-subscription-auth", *mode_args, *extra,
    ]


def test_cloud_submit_mode_defaults_a_interact_for_mini():
    ns = cli.parse_args(_submit_argv("mini-interact", []))
    assert ns.mode == "a-interact"


def test_cloud_submit_mode_defaults_one_shot_for_livesqlbench():
    ns = cli.parse_args(_submit_argv(
        "livesqlbench-base-lite-sqlite", [], ("--no-require-annotation",)
    ))
    assert ns.mode == "one-shot"


def test_cloud_submit_explicit_mode_still_honored():
    ns = cli.parse_args(_submit_argv("mini-interact", ["--mode", "oracle"]))
    assert ns.mode == "oracle"


def test_cloud_submit_explicit_unsupported_mode_rejected():
    # one-shot is not a supported mode for mini-interact → fail fast.
    with pytest.raises(SystemExit):
        cli.parse_args(_submit_argv("mini-interact", ["--mode", "one-shot"]))


# ---------------------------------------------------------------------------
# Local runner (run.main): --mode optional, derived per benchmark
# ---------------------------------------------------------------------------


def _run_main_capture_mode(monkeypatch, dataset, mode_args, tmp_path):
    """Invoke run.main() with a minimal argv, stubbing run_evaluation to capture
    the derived mode without executing a real run."""
    from bird_interact_agents import run

    captured = {}

    async def _fake_run_evaluation(*, mode, **kwargs):
        captured["mode"] = mode
        return {"results": [], "total_usage": {}}

    monkeypatch.setattr(run, "run_evaluation", _fake_run_evaluation)
    data = tmp_path / "data.jsonl"
    data.write_text("")
    argv = [
        "bird-interact", "--framework", "claude_sdk", "--query-mode", "raw",
        "--dataset", dataset, "--agent-model",
        "anthropic/claude-haiku-4-5-20251001", "--no-subscription-auth",
        "--data", str(data), "--db-path", str(tmp_path),
        "--output", str(tmp_path / "eval.json"), *mode_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()
    return captured["mode"]


def test_local_run_mode_defaults_a_interact_for_mini(monkeypatch, tmp_path):
    assert _run_main_capture_mode(monkeypatch, "mini-interact", [], tmp_path) == (
        "a-interact"
    )


def test_local_run_mode_defaults_one_shot_for_livesqlbench(monkeypatch, tmp_path):
    mode = _run_main_capture_mode(
        monkeypatch, "livesqlbench-base-lite-sqlite", [], tmp_path
    )
    assert mode == "one-shot"


def test_local_run_explicit_mode_still_honored(monkeypatch, tmp_path):
    assert _run_main_capture_mode(
        monkeypatch, "mini-interact", ["--mode", "oracle"], tmp_path
    ) == "oracle"
