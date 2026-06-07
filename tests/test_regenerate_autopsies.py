"""Tests for DEV-1541 backfill script `scripts/regenerate_autopsies.py`.

The script scans the per-(instance, run) annotation store for entries that
either lack an autopsy or carry an AutopsyError, and re-runs `run_autopsy`
to overwrite them in place. It is designed to be re-runnable as new
silent-fail incidents surface (Codex r1 #9 — autopsy errors should not
loop forever, but the work-list intentionally includes them so an updated
autopsy implementation can fix them on the next pass).
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_autopsies.py"


def _write_annotation(
    *,
    runs_root: Path,
    benchmark: str,
    db_name: str,
    instance_id: str,
    run_id: str,
    autopsy: dict | None,
    verdict: str = "agent_miss",
) -> Path:
    """Write a minimal SubmissionAnnotation JSON to the canonical path."""
    payload = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": db_name,
        "task_annotation_ref": (
            f"annotations/{benchmark}/{db_name}/{instance_id}.task.json"
        ),
        "annotated_by": "test",
        "annotated_at": "2026-01-01",
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{instance_id}/attempt-1.json",
        },
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "verdict": verdict,
        },
        "failure_classification": {
            "primary": "agent_miss",
            "agent_at_fault": True,
            "remediation_target": "agent",
        },
        "autopsy": autopsy,
    }
    dest = runs_root / benchmark / db_name / instance_id / f"{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest


def _write_trajectory_sidecar(
    *,
    runs_root: Path,
    benchmark: str,
    db_name: str,
    instance_id: str,
    run_id: str,
) -> Path:
    """Write a minimal trajectory sidecar so the backfill can find it."""
    dest = runs_root / benchmark / db_name / instance_id / f"{run_id}.trajectory.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"trajectory": [{"role": "user", "content": "q"}]}) + "\n")
    return dest


def _autopsy_error_dict(kind: str = "validation_error") -> dict:
    return {
        "analysis": None,
        "error": {
            "kind": kind,
            "exception_class": "pydantic_core._pydantic_core.ValidationError",
            "message_excerpt": "fields missing",
            "traceback_excerpt": "Traceback ...",
            "prompt_chars": 100,
            "kb_chars": 10,
            "trajectory_items": 1,
            "model": "anthropic/claude-sonnet-4-5",
            "timestamp": "2026-06-07T00:00:00+00:00",
        },
        "decision_point": None,
        "user_sim_interaction": None,
    }


def _autopsy_analysis_dict(pattern: str = "wrong_join_path") -> dict:
    return {
        "analysis": {
            "kind": "a_interact",
            "pattern": pattern,
            "narrative": "n",
            "remediation": "r",
        },
        "error": None,
        "decision_point": None,
        "user_sim_interaction": {
            "n_asks": 0,
            "key_responses": [],
            "disclosed_resolutions": [],
            "undisclosed_resolutions": [],
        },
    }


# ---------------------------------------------------------------------------
# §A. Work-finding (scan logic)
# ---------------------------------------------------------------------------

def test_scan_finds_autopsy_none(tmp_path):
    """Annotation with autopsy=None AND verdict=agent_miss → work item."""
    from bird_interact_agents.eval.regenerate_autopsies import scan_work_list

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )

    items = scan_work_list(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
    )
    assert len(items) == 1
    assert items[0].instance_id == "robot_10"
    assert items[0].run_id == "r1"


def test_scan_finds_autopsy_error(tmp_path):
    """Annotation with autopsy.error populated → also a work item."""
    from bird_interact_agents.eval.regenerate_autopsies import scan_work_list

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="museum",
        instance_id="museum_2",
        run_id="r1",
        autopsy=_autopsy_error_dict(),
    )
    items = scan_work_list(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
    )
    assert len(items) == 1


def test_scan_skips_already_analysed(tmp_path):
    """autopsy with populated analysis (no error) → NOT in work-list."""
    from bird_interact_agents.eval.regenerate_autopsies import scan_work_list

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_5",
        run_id="r1",
        autopsy=_autopsy_analysis_dict(),
    )
    items = scan_work_list(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
    )
    assert items == []


def test_scan_skips_non_agent_miss(tmp_path):
    """Passing submissions (verdict != agent_miss) are never work items."""
    from bird_interact_agents.eval.regenerate_autopsies import scan_work_list

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_5",
        run_id="r1",
        autopsy=None,
        verdict="correct",
    )
    items = scan_work_list(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
    )
    assert items == []


def test_scan_respects_run_id_filter(tmp_path):
    """--run-id intersects with the auto-scan."""
    from bird_interact_agents.eval.regenerate_autopsies import scan_work_list

    runs = tmp_path / "runs"
    for run_id in ("r1", "r2"):
        _write_annotation(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            db_name="robot",
            instance_id="robot_10",
            run_id=run_id,
            autopsy=None,
        )
    items = scan_work_list(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        run_id="r1",
    )
    assert len(items) == 1
    assert items[0].run_id == "r1"


def test_scan_respects_instance_ids_filter(tmp_path):
    from bird_interact_agents.eval.regenerate_autopsies import scan_work_list

    runs = tmp_path / "runs"
    for inst in ("robot_10", "museum_2", "museum_4"):
        _write_annotation(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            db_name=inst.split("_")[0],
            instance_id=inst,
            run_id="r1",
            autopsy=None,
        )
    items = scan_work_list(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        instance_ids={"museum_2", "museum_4"},
    )
    assert {i.instance_id for i in items} == {"museum_2", "museum_4"}


# ---------------------------------------------------------------------------
# §B. Dry-run does not instantiate the Anthropic client
# ---------------------------------------------------------------------------

def test_dry_run_does_not_instantiate_anthropic_client(tmp_path, monkeypatch):
    """--dry-run prints the work-list and exits without calling the LLM."""
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )

    # Snapshot the on-disk annotation BEFORE dry-run.
    dest = runs / "livesqlbench-base-lite-sqlite" / "robot" / "robot_10" / "r1.json"
    before = dest.read_text()

    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        report = regenerate(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            model="anthropic/claude-haiku-4-5-20251001",
            dry_run=True,
        )
    mock_anthropic.assert_not_called()
    assert report.work_items == 1
    assert report.regenerated == 0
    # Codex r2 #14: --dry-run must not rewrite the annotation file.
    assert dest.read_text() == before


# ---------------------------------------------------------------------------
# §C. Full integration — backfill overwrites the annotation
# ---------------------------------------------------------------------------

def test_regenerate_overwrites_annotation_with_analysis(tmp_path, monkeypatch):
    """Successful autopsy backfill writes analysis back to the annotation
    on disk. Verifies the full happy path end-to-end."""
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    runs = tmp_path / "runs"
    dest = _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )
    _write_trajectory_sidecar(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
    )

    # Stub the autopsy LLM. Use the autopsy.run_autopsy entry point so the
    # script's plumbing is exercised end-to-end except for the actual model
    # call.
    tool_input = {
        "pattern": "exhausted_budget_guessing",
        "other_details": None,
        "narrative": "Agent exhausted budget on KB exploration.",
        "remediation": "Increase budget.",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
    }
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = tool_input
    mock_response = MagicMock()
    mock_response.content = [mock_tool]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        report = regenerate(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            model="anthropic/claude-haiku-4-5-20251001",
            dry_run=False,
        )

    assert report.work_items == 1
    assert report.regenerated == 1
    # Annotation on disk now carries the new analysis.
    data = json.loads(dest.read_text())
    assert data["autopsy"]["analysis"] is not None
    assert data["autopsy"]["analysis"]["pattern"] == "exhausted_budget_guessing"
    assert data["autopsy"]["error"] is None


def test_regenerate_io_failure_missing_trajectory_increments_errored(tmp_path):
    """Codex r1 #9: if the trajectory sidecar is missing AND no fallback
    can be found, the run counts as IO-errored — it should appear in
    `report.io_errors`, not be silently skipped."""
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )
    # NO trajectory sidecar written.

    # The Anthropic client is never instantiated for IO-failed items.
    with patch("anthropic.AsyncAnthropic") as mock_anthropic:
        report = regenerate(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            model="anthropic/claude-haiku-4-5-20251001",
            dry_run=False,
        )
    assert report.io_errors == 1
    assert report.regenerated == 0
    mock_anthropic.assert_not_called()
    # Codex r2 #13: IO failures MUST flip the exit code to non-zero so
    # cron drivers actually surface broken work.
    assert report.exit_code() != 0


def test_regenerate_autopsy_returning_error_is_not_an_io_failure(tmp_path):
    """Codex r1 #9: when autopsy itself returns an AutopsyError (e.g.
    repeated context_overflow on the same trajectory), that is counted in
    `report.autopsy_errors`, NOT `report.io_errors`. The script should
    exit 0 in this case so a cron driver doesn't loop on transient LLM
    errors."""
    import anthropic
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )
    _write_trajectory_sidecar(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
    )

    mock_client = AsyncMock()
    mock_client.messages.create.side_effect = anthropic.BadRequestError(
        message="prompt is too long",
        response=MagicMock(status_code=400),
        body={},
    )

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        report = regenerate(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            model="anthropic/claude-haiku-4-5-20251001",
            dry_run=False,
        )
    assert report.autopsy_errors == 1
    assert report.regenerated == 0
    assert report.io_errors == 0
    # Exit code logic: only IO errors raise the non-zero exit.
    assert report.exit_code() == 0


# ---------------------------------------------------------------------------
# §D. CLI entry-point smoke (no network, no real LLM)
# ---------------------------------------------------------------------------

def test_cli_dry_run_smoke(tmp_path, monkeypatch):
    """`python scripts/regenerate_autopsies.py --benchmark <b> --dry-run`
    runs end-to-end without crashing. Lives at the CLI seam to catch
    argparse / import regressions."""
    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )
    env_overlay = {
        "BIRD_RUNS_ROOT": str(runs),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--benchmark", "livesqlbench-base-lite-sqlite",
            "--dry-run",
        ],
        env={**__import__("os").environ, **env_overlay},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert "robot_10" in result.stdout


# ---------------------------------------------------------------------------
# §E. Trajectory loading — sidecar preference + cloud-results fallback (Codex r2 #12)
# ---------------------------------------------------------------------------

def test_regenerate_falls_back_to_cloud_results_trajectory(tmp_path, monkeypatch):
    """When the per-(instance, run) sidecar is missing, fall back to
    `results/<benchmark>/cloud/<run_id>/rows/<inst>/attempt-1.json`.
    Verify the regenerate run completes successfully against the fallback."""
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )
    # NO trajectory sidecar; instead write the cloud-results-shape file.
    cloud_traj = (
        tmp_path / "results" / "livesqlbench-base-lite-sqlite" / "cloud"
        / "r1" / "rows" / "robot_10" / "attempt-1.json"
    )
    cloud_traj.parent.mkdir(parents=True, exist_ok=True)
    cloud_traj.write_text(json.dumps(
        {"trajectory": [{"role": "user", "content": "from cloud results"}]}
    ))

    tool_input = {
        "pattern": "exhausted_budget_guessing",
        "other_details": None,
        "narrative": "n",
        "remediation": "r",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
    }
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = tool_input
    mock_response = MagicMock()
    mock_response.content = [mock_tool]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        report = regenerate(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            model="anthropic/claude-haiku-4-5-20251001",
            dry_run=False,
            results_root=tmp_path / "results",
        )

    assert report.regenerated == 1
    assert report.io_errors == 0


def test_regenerate_sidecar_takes_precedence_over_cloud_results(tmp_path):
    """If both sidecar AND cloud-results trajectory exist, the sidecar
    wins (the runs/ store is the authoritative one — Codex r2 #12)."""
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    runs = tmp_path / "runs"
    _write_annotation(
        runs_root=runs,
        benchmark="livesqlbench-base-lite-sqlite",
        db_name="robot",
        instance_id="robot_10",
        run_id="r1",
        autopsy=None,
    )
    # Sidecar carries marker "FROM SIDECAR"; cloud-results carries marker
    # "FROM CLOUD". The autopsy stub's prompt-snapshot captures whichever
    # trajectory text actually got rendered into the prompt.
    sidecar = runs / "livesqlbench-base-lite-sqlite" / "robot" / "robot_10" / "r1.trajectory.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(
        {"trajectory": [{"role": "user", "content": "FROM SIDECAR"}]}
    ))
    cloud_traj = (
        tmp_path / "results" / "livesqlbench-base-lite-sqlite" / "cloud"
        / "r1" / "rows" / "robot_10" / "attempt-1.json"
    )
    cloud_traj.parent.mkdir(parents=True, exist_ok=True)
    cloud_traj.write_text(json.dumps(
        {"trajectory": [{"role": "user", "content": "FROM CLOUD"}]}
    ))

    # Capture the prompt the model receives so we can grep it for the
    # marker.
    captured_prompts: list[str] = []

    async def _capture(**kwargs):
        # The prompt is the first user content; tools/etc. are in kwargs.
        for m in kwargs.get("messages", []):
            if m.get("role") == "user":
                captured_prompts.append(m.get("content", ""))
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.name = "autopsy_output"
        mock_tool.input = {
            "pattern": "exhausted_budget_guessing",
            "other_details": None,
            "narrative": "n",
            "remediation": "r",
            "decision_point_trajectory_index": None,
            "decision_point_description": None,
        }
        mock_response = MagicMock()
        mock_response.content = [mock_tool]
        return mock_response

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=_capture)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        regenerate(
            runs_root=runs,
            benchmark="livesqlbench-base-lite-sqlite",
            model="anthropic/claude-haiku-4-5-20251001",
            dry_run=False,
            results_root=tmp_path / "results",
        )

    assert any("FROM SIDECAR" in p for p in captured_prompts), (
        "sidecar trajectory must take precedence"
    )
    assert not any("FROM CLOUD" in p for p in captured_prompts), (
        "cloud-results trajectory should not be read when sidecar present"
    )
