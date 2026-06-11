"""Tests for the ``bird-interact-cloud submission`` subcommand.

Spec (DEV-1553) tests #13 (CLI argparse + dispatch) + #19 (selection
coverage), plus an end-to-end smoke that exercises the converter +
output writer through the CLI entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.reports._fixtures import (
    stage_run,
    trajectory_one_phase_pass,
    trajectory_two_phase_pass,
)


# ---------------------------------------------------------------------------
# Argparse contract
# ---------------------------------------------------------------------------


def test_read_patience_returns_default_on_non_numeric_value(tmp_path: Path):
    """CodeRabbit #1: ``int(pat)`` must not crash on a non-numeric
    patience field — it falls back to ``(None, "default")`` consistent
    with the function's existing JSON / file-IO defensiveness."""
    from bird_interact_agents.reports.cli import _read_patience_for_instance

    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    (inst_dir / "r1.json").write_text(json.dumps({"patience": "not-a-number"}))
    assert _read_patience_for_instance(inst_dir, "r1") == (None, "default")


def test_read_patience_returns_default_on_malformed_json(tmp_path: Path):
    from bird_interact_agents.reports.cli import _read_patience_for_instance

    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    (inst_dir / "r1.json").write_text("not-json{{{")
    assert _read_patience_for_instance(inst_dir, "r1") == (None, "default")


def test_read_patience_reads_numeric_value(tmp_path: Path):
    from bird_interact_agents.reports.cli import _read_patience_for_instance

    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    (inst_dir / "r1.json").write_text(json.dumps({"patience": 7}))
    patience, source = _read_patience_for_instance(inst_dir, "r1")
    assert patience == 7
    assert source == "runs:r1.json"


def test_submission_requires_run_id_or_selection():
    from bird_interact_agents.cloud.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "submission",
                "--team-name",
                "Motley",
                "--method-name",
                "SLayer",
                "--benchmark",
                "bird-interact-lite-exp",
            ]
        )


def test_submission_rejects_both_run_id_and_selection(tmp_path):
    from bird_interact_agents.cloud.cli import main

    sel = tmp_path / "sel.jsonl"
    sel.write_text("")
    with pytest.raises(SystemExit):
        main(
            [
                "submission",
                "--team-name",
                "Motley",
                "--method-name",
                "SLayer",
                "--benchmark",
                "bird-interact-lite-exp",
                "--run-id",
                "r1",
                "--selection",
                str(sel),
            ]
        )


def test_submission_requires_team_and_method(tmp_path):
    from bird_interact_agents.cloud.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "submission",
                "--benchmark",
                "bird-interact-lite-exp",
                "--run-id",
                "r1",
            ]
        )


def test_submission_benchmark_rejects_out_of_scope(tmp_path):
    """Only the three a-Interact benchmarks are accepted."""
    from bird_interact_agents.cloud.cli import main

    with pytest.raises(SystemExit):
        main(
            [
                "submission",
                "--team-name",
                "Motley",
                "--method-name",
                "SLayer",
                "--benchmark",
                "livesqlbench-base-lite-sqlite",
                "--run-id",
                "r1",
            ]
        )


# ---------------------------------------------------------------------------
# End-to-end: --run-id with full coverage produces a usable submission dir
# ---------------------------------------------------------------------------


def _stub_split(monkeypatch, instance_ids):
    from bird_interact_agents.reports import coverage as _cov

    monkeypatch.setattr(
        _cov, "load_benchmark_instance_ids", lambda benchmark: set(instance_ids)
    )


def _stub_fake_tokenizer(monkeypatch):
    from bird_interact_agents.reports import tokens as _tokens

    def _fake(s, *, model="claude-haiku-4-5-20251001"):
        return max(1, len(s) // 4)

    monkeypatch.setattr(_tokens, "count_tokens", _fake)


def test_submission_end_to_end_run_id_full_coverage(
    tmp_path: Path, monkeypatch
):
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
            ("alien", "alien_2", trajectory_two_phase_pass(instance_id="alien_2")),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1", "alien_2"})
    _stub_fake_tokenizer(monkeypatch)

    # Stub task_data lookup (zero ambiguities → budget = 12).
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    rc = main(
        [
            "submission",
            "--team-name",
            "Motley",
            "--method-name",
            "SLayer-Agent",
            "--benchmark",
            "bird-interact-lite-exp",
            "--run-id",
            "run-xyz",
        ]
    )
    assert rc == 0

    # Verify artefacts.
    out_root = tmp_path / "reports" / "bird-interact-lite-exp" / "a-Interact"
    [sub_dir] = list(out_root.iterdir())
    assert (sub_dir / "submission.jsonl").exists()
    assert (sub_dir / "email_title.txt").exists()
    assert (sub_dir / "manifest.json").exists()

    title = (sub_dir / "email_title.txt").read_text()
    assert title == "[BIRD-INTERACT-1.0-lite][a-Interact][Motley][SLayer-Agent]"

    lines = (sub_dir / "submission.jsonl").read_text().splitlines()
    assert len(lines) == 2
    ids = {json.loads(line)["instance_id"] for line in lines}
    assert ids == {"alien_1", "alien_2"}


# ---------------------------------------------------------------------------
# --selection coverage check (Codex finding #1)
# ---------------------------------------------------------------------------


def test_submission_selection_partial_aborts_without_allow_partial(
    tmp_path: Path, monkeypatch
):
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1", "alien_2", "alien_3"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    sel_path = tmp_path / "sel.jsonl"
    sel_path.write_text(json.dumps({"instance_id": "alien_1", "run_id": "run-xyz"}) + "\n")

    with pytest.raises(SystemExit):
        main(
            [
                "submission",
                "--team-name",
                "Motley",
                "--method-name",
                "SLayer-Agent",
                "--benchmark",
                "bird-interact-lite-exp",
                "--selection",
                str(sel_path),
            ]
        )


def test_submission_selection_partial_with_allow_partial_succeeds(
    tmp_path: Path, monkeypatch
):
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1", "alien_2", "alien_3"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    sel_path = tmp_path / "sel.jsonl"
    sel_path.write_text(json.dumps({"instance_id": "alien_1", "run_id": "run-xyz"}) + "\n")

    rc = main(
        [
            "submission",
            "--team-name",
            "Motley",
            "--method-name",
            "SLayer-Agent",
            "--benchmark",
            "bird-interact-lite-exp",
            "--selection",
            str(sel_path),
            "--allow-partial",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# --patience flag
# ---------------------------------------------------------------------------


def test_submission_run_id_partial_coverage_aborts_without_allow_partial(
    tmp_path: Path, monkeypatch
):
    """The --run-id path also enforces full coverage by default."""
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1", "alien_2"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    with pytest.raises(SystemExit):
        main(
            [
                "submission",
                "--team-name",
                "Motley",
                "--method-name",
                "SLayer-Agent",
                "--benchmark",
                "bird-interact-lite-exp",
                "--run-id",
                "run-xyz",
            ]
        )


def test_submission_run_id_partial_coverage_with_allow_partial_succeeds(
    tmp_path: Path, monkeypatch
):
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1", "alien_2"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    rc = main(
        [
            "submission",
            "--team-name",
            "Motley",
            "--method-name",
            "SLayer-Agent",
            "--benchmark",
            "bird-interact-lite-exp",
            "--run-id",
            "run-xyz",
            "--allow-partial",
        ]
    )
    assert rc == 0


def test_submission_no_thinking_flag_strips_thinking(
    tmp_path: Path, monkeypatch
):
    """--no-thinking CLI flag flips the converter's include_thinking to False."""
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main
    from tests.reports._fixtures import (
        assistant_msg,
        build_trajectory,
        system_msg,
        tool_result_msg,
        tool_use_block,
        user_text_msg,
    )

    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        assistant_msg(
            thinking="thinking content",
            text="text content",
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": "SELECT 1"},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="Phase 1 SQL Correct!"),
    ]
    traj = build_trajectory(
        instance_id="alien_1", trajectory_steps=steps, submitted_sql="SELECT 1"
    )

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[("alien", "alien_1", traj)],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    rc = main(
        [
            "submission",
            "--team-name",
            "Motley",
            "--method-name",
            "SLayer-Agent",
            "--benchmark",
            "bird-interact-lite-exp",
            "--run-id",
            "run-xyz",
            "--no-thinking",
        ]
    )
    assert rc == 0

    out_root = tmp_path / "reports" / "bird-interact-lite-exp" / "a-Interact"
    [sub_dir] = list(out_root.iterdir())
    row = json.loads((sub_dir / "submission.jsonl").read_text().splitlines()[0])
    assert "thinking" not in row["prompt_flow"][0]["response"]


def test_submission_check_leakage_flag_writes_manifest_counts(
    tmp_path: Path, monkeypatch
):
    """--check-leakage scans each instance's prompts for gold-SQL substrings
    and records per-instance counts in manifest.leakage_check."""
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main
    from tests.reports._fixtures import (
        assistant_msg,
        build_trajectory,
        system_msg,
        tool_result_msg,
        tool_use_block,
        user_text_msg,
    )

    gold = "SELECT trader.id FROM trader JOIN compliancecase ON x = y"
    steps = [
        system_msg(),
        user_text_msg(text=f"Hint: try `{gold}`."),  # gold leaked in initial prompt
        assistant_msg(
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": gold},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="Phase 1 SQL Correct!"),
    ]
    traj = build_trajectory(
        instance_id="alien_1",
        trajectory_steps=steps,
        submitted_sql=gold,
        ground_truth_sql=gold,
    )
    clean_traj = trajectory_one_phase_pass(instance_id="alien_2", sql="SELECT 1")

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", traj),
            ("alien", "alien_2", clean_traj),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1", "alien_2"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    rc = main(
        [
            "submission",
            "--team-name",
            "Motley",
            "--method-name",
            "SLayer-Agent",
            "--benchmark",
            "bird-interact-lite-exp",
            "--run-id",
            "run-xyz",
            "--check-leakage",
        ]
    )
    assert rc == 0

    out_root = tmp_path / "reports" / "bird-interact-lite-exp" / "a-Interact"
    [sub_dir] = list(out_root.iterdir())
    mf = json.loads((sub_dir / "manifest.json").read_text())
    leak = mf["leakage_check"]
    assert leak is not None
    counts_by_id = {
        e["instance_id"]: e["leak_count"] for e in leak["per_instance"]
    }
    assert counts_by_id["alien_1"] >= 1
    assert counts_by_id["alien_2"] == 0
    # The submission rows themselves are unchanged — no redaction.
    rows = [
        json.loads(line)
        for line in (sub_dir / "submission.jsonl").read_text().splitlines()
    ]
    leaky_row = next(r for r in rows if r["instance_id"] == "alien_1")
    assert gold in leaky_row["prompt_flow"][0]["prompt"]


def test_submission_patience_flag_changes_total_budget(
    tmp_path: Path, monkeypatch
):
    """Bumping --patience changes the replayed remaining_budget headroom."""
    from bird_interact_agents import paths
    from bird_interact_agents.cloud.cli import main

    runs_root, results_root = stage_run(
        tmp_path,
        benchmark="bird-interact-lite-exp",
        run_id="run-xyz",
        instances=[
            ("alien", "alien_1", trajectory_one_phase_pass(instance_id="alien_1")),
        ],
    )
    monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
    monkeypatch.setattr(paths, "results_root", lambda: results_root)
    monkeypatch.setattr(paths, "reports_root", lambda: tmp_path / "reports")
    _stub_split(monkeypatch, {"alien_1"})
    _stub_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "bird_interact_agents.reports.budget.lookup_task_data",
        lambda benchmark, instance_id: {
            "user_query_ambiguity": {},
            "knowledge_ambiguity": [],
        },
    )

    rc = main(
        [
            "submission",
            "--team-name",
            "Motley",
            "--method-name",
            "SLayer-Agent",
            "--benchmark",
            "bird-interact-lite-exp",
            "--run-id",
            "run-xyz",
            "--patience",
            "500",
        ]
    )
    assert rc == 0

    out_root = tmp_path / "reports" / "bird-interact-lite-exp" / "a-Interact"
    [sub_dir] = list(out_root.iterdir())
    row = json.loads((sub_dir / "submission.jsonl").read_text().splitlines()[0])
    # total_budget with patience=500, amb=0: 6 + 0 + 2*500 = 1006. Submit
    # cost = 3 → remaining_budget = 1003.
    assert row["prompt_flow"][0]["remaining_budget"] == 1003.0
