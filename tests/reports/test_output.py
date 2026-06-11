"""Tests for the submission output writer.

Spec (DEV-1553) tests #10 (email title), #11 (manifest), #15 (JSONL
schema validity).
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Email title (Section I)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "benchmark,expected_split",
    [
        ("bird-interact-lite-exp", "lite"),
        ("bird-interact-full", "full"),
        ("mini-interact", "mini-interact"),
    ],
)
def test_email_title_string(benchmark, expected_split):
    from bird_interact_agents.reports.output import build_email_title

    title = build_email_title(
        benchmark=benchmark,
        setting="a-Interact",
        team="Motley",
        method="SLayer-Agent",
    )
    assert (
        title
        == f"[BIRD-INTERACT-1.0-{expected_split}][a-Interact][Motley][SLayer-Agent]"
    )


def test_email_title_unsupported_benchmark_raises():
    from bird_interact_agents.reports.output import build_email_title

    with pytest.raises(ValueError):
        build_email_title(
            benchmark="livesqlbench-base-lite-sqlite",
            setting="a-Interact",
            team="Motley",
            method="SLayer-Agent",
        )


# ---------------------------------------------------------------------------
# Write submission directory
# ---------------------------------------------------------------------------


def _stub_row(instance_id: str, phase1_sql: str = "SELECT 1"):
    from bird_interact_agents.reports.schema import PromptFlowEntry, SubmissionRow

    return SubmissionRow(
        instance_id=instance_id,
        subtask_1_predicted_sql=[phase1_sql],
        subtask_2_predicted_sql=[],
        prompt_flow=[
            PromptFlowEntry(
                model="anthropic/claude-opus-4-7",
                user_simulator="anthropic/claude-sonnet-4-6",
                prompt="Task.",
                response="Submitting.",
                action=f"submit({phase1_sql})",
                remaining_budget=9.0,
                action_input_tokens=2,
                action_output_tokens=8,
                action_cost=3,
            )
        ],
    )


def test_write_submission_creates_three_artefacts(tmp_path):
    from bird_interact_agents.reports.output import (
        ManifestPlan,
        write_submission,
    )

    rows = [_stub_row("alien_1"), _stub_row("alien_2", phase1_sql="SELECT 2")]
    plan = ManifestPlan(
        benchmark="bird-interact-lite-exp",
        setting="a-Interact",
        split="lite",
        team="Motley",
        method="SLayer-Agent",
        tag="run-xyz",
        selection_mode="run-id",
        source_run_ids=["run-xyz"],
        generated_at="2026-06-10T16:42:13+00:00",
        instances=[
            {
                "instance_id": "alien_1",
                "run_id": "run-xyz",
                "framework": "claude_sdk_otf",
                "agent_model": "anthropic/claude-opus-4-7",
                "user_sim_model": "anthropic/claude-sonnet-4-6",
                "trajectory_path": "/dev/null",
                "results_db_path": "/dev/null",
            },
            {
                "instance_id": "alien_2",
                "run_id": "run-xyz",
                "framework": "claude_sdk_otf",
                "agent_model": "anthropic/claude-opus-4-7",
                "user_sim_model": "anthropic/claude-sonnet-4-6",
                "trajectory_path": "/dev/null",
                "results_db_path": "/dev/null",
            },
        ],
        patience_resolution=[
            {"instance_id": "alien_1", "patience": 3, "source": "default"},
            {"instance_id": "alien_2", "patience": 3, "source": "default"},
        ],
        leakage_check=None,
        warnings_by_instance=[],
    )
    out_dir = write_submission(rows=rows, plan=plan, out_dir=tmp_path)
    assert (out_dir / "submission.jsonl").exists()
    assert (out_dir / "email_title.txt").exists()
    assert (out_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# JSONL row schema validity (Codex finding #8)
# ---------------------------------------------------------------------------


def test_submission_jsonl_row_has_required_fields_and_no_debug_step(tmp_path):
    from bird_interact_agents.reports.output import (
        ManifestPlan,
        write_submission,
    )

    rows = [_stub_row("alien_1")]
    plan = ManifestPlan(
        benchmark="bird-interact-lite-exp",
        setting="a-Interact",
        split="lite",
        team="Motley",
        method="SLayer-Agent",
        tag="run-xyz",
        selection_mode="run-id",
        source_run_ids=["run-xyz"],
        generated_at="2026-06-10T16:42:13+00:00",
        instances=[],
        patience_resolution=[],
        leakage_check=None,
        warnings_by_instance=[],
    )
    out_dir = write_submission(rows=rows, plan=plan, out_dir=tmp_path)

    lines = (out_dir / "submission.jsonl").read_text().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])

    # Required a-Interact custom-agent fields (Section II).
    assert set(obj.keys()) == {
        "instance_id",
        "subtask_1_predicted_sql",
        "subtask_2_predicted_sql",
        "prompt_flow",
    }
    # No c-Interact-only keys.
    assert "debug_step_1" not in obj
    assert "debug_step_2" not in obj
    # List-of-strings type (literal-spec interpretation).
    assert isinstance(obj["subtask_1_predicted_sql"], list)
    assert all(isinstance(s, str) for s in obj["subtask_1_predicted_sql"])
    assert isinstance(obj["subtask_2_predicted_sql"], list)
    # prompt_flow entry shape.
    entry = obj["prompt_flow"][0]
    required = {
        "model",
        "user_simulator",
        "prompt",
        "response",
        "action",
        "remaining_budget",
        "action_input_tokens",
        "action_output_tokens",
        "action_cost",
    }
    assert required.issubset(entry.keys())
    # No c-Interact `debug_step` smuggled in.
    assert not any(k.startswith("debug_step") for k in entry.keys())


# ---------------------------------------------------------------------------
# Manifest provenance + constants
# ---------------------------------------------------------------------------


def test_manifest_records_provenance_and_constants(tmp_path):
    from bird_interact_agents.reports.output import (
        ManifestPlan,
        write_submission,
    )

    plan = ManifestPlan(
        benchmark="bird-interact-lite-exp",
        setting="a-Interact",
        split="lite",
        team="Motley",
        method="SLayer-Agent",
        tag="run-xyz",
        selection_mode="selection-file",
        source_run_ids=["run-a", "run-b"],
        generated_at="2026-06-10T16:42:13+00:00",
        instances=[
            {
                "instance_id": "alien_1",
                "run_id": "run-a",
                "framework": "claude_sdk_otf",
                "agent_model": "anthropic/claude-opus-4-7",
                "user_sim_model": "anthropic/claude-sonnet-4-6",
                "trajectory_path": "/dev/null",
                "results_db_path": "/dev/null",
            }
        ],
        patience_resolution=[
            {"instance_id": "alien_1", "patience": 3, "source": "default"}
        ],
        leakage_check=None,
        warnings_by_instance=[],
    )
    rows = [_stub_row("alien_1")]
    out_dir = write_submission(rows=rows, plan=plan, out_dir=tmp_path)

    mf = json.loads((out_dir / "manifest.json").read_text())
    assert mf["schema_version"] == 1
    assert mf["kind"] == "bird_interact_submission_manifest"
    assert mf["benchmark"] == "bird-interact-lite-exp"
    assert mf["split"] == "lite"
    assert mf["setting"] == "a-Interact"
    assert mf["team"] == "Motley"
    assert mf["method"] == "SLayer-Agent"
    assert mf["n_instances"] == 1
    assert mf["selection_mode"] == "selection-file"
    assert mf["source_run_ids"] == ["run-a", "run-b"]
    assert mf["instances"][0]["instance_id"] == "alien_1"
    assert mf["patience_resolution"][0]["patience"] == 3
    assert mf["section_vi_threshold"] == {
        "input_tokens_lt": 250,
        "output_tokens_lt": 1000,
        "cheap_cost": 0.5,
        "expensive_cost": 1.0,
    }
    assert mf["fixed_costs"] == {"ask": 2, "submit": 3, "execute": 1}
    assert "anthropic" in mf["tokenizer"].lower()
