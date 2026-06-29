"""Tests for ``scripts/backfill_run_versions.py`` (DEV-1591 stream 2).

The backfill stamps ``version`` + ``agent_model`` onto every existing
per-task ``runs/`` record. Contract:
* override table (run-id) wins over the framework→version map;
* plain ``claude_sdk`` → v0, ``claude_sdk_v1`` → v1;
* a missing manifest still version-stamps via override/default-v0, leaving
  ``agent_model`` untouched;
* idempotent — a second run changes nothing;
* ``--dry-run`` writes nothing;
* a record whose existing ``agent_model`` disagrees with the manifest is
  flagged (mismatch counter) — Codex Low #7.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts"
    / "backfill_run_versions.py"
)
_spec = importlib.util.spec_from_file_location("backfill_run_versions", SCRIPT)
backfill_run_versions = importlib.util.module_from_spec(_spec)
sys.modules["backfill_run_versions"] = backfill_run_versions
_spec.loader.exec_module(backfill_run_versions)

_BENCH = "mini-interact"
_CEA364 = "20260629t1209-claudes-slayer-cea364"  # override → v2


def _ann_dict(*, iid, db, run_id, **extra) -> dict:
    d = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": iid,
        "selected_database": db,
        "task_annotation_ref": f"annotations/{_BENCH}/{db}/{iid}.task.json",
        "annotated_by": "test",
        "annotated_at": "2026-06-01T10:00:00+00:00",
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{iid}/attempt-1.json",
            "predicted_row_count": 1, "duration_s": 1.0,
            "cost_usd_agent": 0.0, "cost_usd_user_sim": 0.0,
            "n_agent_turns": 1, "n_ask_user_calls": 0,
        },
        "evaluation": {
            "phase1_against_original_gold": "pass",
            "phase1_against_audited_primary": "pass",
            "phase1_against_any_audited_variant": "pass",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False, "novel_reading_judgment": None,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False, "numeric_epsilon": 1e-6,
            "verdict": "correct", "matched_variant_id": "primary",
            "rationale": "",
        },
        "failure_classification": {
            "primary": "no_fail", "secondary": [], "agent_at_fault": False,
            "remediation_target": "other", "remediation_text": "", "details": "",
        },
        "decision_point": None,
        "user_sim_interaction": {
            "n_asks": 0, "key_responses": [],
            "disclosed_resolutions": [], "undisclosed_resolutions": [],
        },
        "original_gold_annotated_correct": True,
    }
    d.update(extra)
    return d


def _write_record(runs: Path, db, iid, run_id, **extra) -> Path:
    dest = runs / _BENCH / db / iid / f"{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_ann_dict(iid=iid, db=db, run_id=run_id, **extra)))
    return dest


def _write_manifest(results: Path, run_id, *, framework, agent_model):
    dest = results / _BENCH / "cloud" / run_id / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "framework": framework, "agent_model": agent_model,
        "query_mode": "slayer",
    }))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(results))
    return runs, results


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def test_backfill_override_and_framework_map(tree):
    runs, results = tree
    # v2 override run (framework token is claude_sdk → would map to v0)
    p_v2 = _write_record(runs, "households", "households_1", _CEA364)
    _write_manifest(results, _CEA364, framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7")
    # plain v0
    p_v0 = _write_record(runs, "alien", "alien_1",
                         "20260601t1000-claudes-slayer-aaaaaa")
    _write_manifest(results, "20260601t1000-claudes-slayer-aaaaaa",
                    framework="claude_sdk", agent_model="anthropic/claude-haiku-4-5")
    # v1
    p_v1 = _write_record(runs, "alien", "alien_2",
                         "20260601t1100-claudes-slayer-bbbbbb")
    _write_manifest(results, "20260601t1100-claudes-slayer-bbbbbb",
                    framework="claude_sdk_v1", agent_model="zai/glm-5.2")

    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False)

    assert _read(p_v2)["version"] == "v2"
    assert _read(p_v2)["agent_model"] == "anthropic/claude-opus-4-7"
    assert _read(p_v0)["version"] == "v0"
    assert _read(p_v1)["version"] == "v1"
    assert counters["updated"] == 3


def test_backfill_missing_manifest_defaults_v0(tree):
    runs, results = tree
    p = _write_record(runs, "alien", "alien_1",
                      "20260601t1000-claudes-slayer-aaaaaa")
    # no manifest written
    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    rec = _read(p)
    assert rec["version"] == "v0"          # default when framework unknown
    assert rec["agent_model"] is None      # nothing to fill from
    assert counters["no_manifest"] == 1


def test_backfill_override_without_manifest_still_v2(tree):
    runs, _ = tree
    p = _write_record(runs, "households", "households_1", _CEA364)
    backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert _read(p)["version"] == "v2"     # override table keyed by run-id


def test_backfill_is_idempotent(tree):
    runs, results = tree
    _write_record(runs, "alien", "alien_1",
                  "20260601t1000-claudes-slayer-aaaaaa")
    _write_manifest(results, "20260601t1000-claudes-slayer-aaaaaa",
                    framework="claude_sdk", agent_model="anthropic/claude-opus-4-7")
    first = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert first["updated"] == 1
    second = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert second["updated"] == 0
    assert second["unchanged"] == 1


def test_backfill_dry_run_writes_nothing(tree):
    runs, results = tree
    p = _write_record(runs, "alien", "alien_1",
                      "20260601t1000-claudes-slayer-aaaaaa")
    _write_manifest(results, "20260601t1000-claudes-slayer-aaaaaa",
                    framework="claude_sdk", agent_model="anthropic/claude-opus-4-7")
    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False,
                                              dry_run=True)
    assert counters["updated"] == 1           # would-update count still reported
    assert _read(p).get("version") is None    # but nothing written
    assert "agent_model" not in _read(p) or _read(p)["agent_model"] is None


def test_backfill_agent_model_mismatch_flagged(tree):
    runs, results = tree
    # record already carries a (stale) agent_model that disagrees with manifest
    _write_record(runs, "alien", "alien_1",
                  "20260601t1000-claudes-slayer-aaaaaa",
                  agent_model="stale/model")
    _write_manifest(results, "20260601t1000-claudes-slayer-aaaaaa",
                    framework="claude_sdk", agent_model="anthropic/claude-opus-4-7")
    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert counters["agent_model_mismatch"] == 1
