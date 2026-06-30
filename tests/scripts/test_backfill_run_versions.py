"""Tests for ``scripts/backfill_run_versions.py`` (DEV-1591 stream 2).

The backfill is the "just in case" re-filler: it COPIES the producer's
``version`` (+ ``agent_model``) from a run's manifest onto any record that is
missing it. It never reconstructs the version from the framework. Contract:
* a record missing ``version`` gets ``manifest["version"]`` copied;
* no manifest version → left as-is (no guessing);
* a record already stamped is preserved (idempotent);
* ``--dry-run`` writes nothing;
* legacy-flat manifest (``results/cloud/<run_id>/``) is read;
* an existing ``agent_model`` that disagrees with the manifest is flagged.
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


def _write_manifest(results: Path, run_id, *, framework, agent_model,
                    version=None, legacy_flat=False):
    if legacy_flat:
        dest = results / "cloud" / run_id / "manifest.json"
    else:
        dest = results / _BENCH / "cloud" / run_id / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": framework, "agent_model": agent_model,
        "query_mode": "slayer",
    }
    if version is not None:
        payload["version"] = version
    dest.write_text(json.dumps(payload))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(results))
    return runs, results


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def test_backfill_copies_manifest_version(tree):
    runs, results = tree
    # v2 producer literal in the manifest.
    p2 = _write_record(runs, "households", "households_1", "rid_v2")
    _write_manifest(results, "rid_v2", framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7", version="v2")
    # clean v0 producer literal.
    p0 = _write_record(runs, "alien", "alien_1", "rid_v0")
    _write_manifest(results, "rid_v0", framework="claude_sdk",
                    agent_model="anthropic/claude-haiku-4-5", version="v0")

    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert _read(p2)["version"] == "v2"
    assert _read(p2)["agent_model"] == "anthropic/claude-opus-4-7"
    assert _read(p0)["version"] == "v0"
    assert counters["updated"] == 2


def test_backfill_no_manifest_version_leaves_none(tree):
    """No version in the manifest → nothing to copy, no guessing."""
    runs, results = tree
    p = _write_record(runs, "alien", "alien_1", "rid_x")
    _write_manifest(results, "rid_x", framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7")  # no version
    backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    rec = _read(p)
    assert rec["version"] is None
    assert rec["agent_model"] == "anthropic/claude-opus-4-7"  # model still copied


def test_backfill_reads_legacy_flat_manifest(tree):
    """Legacy-flat manifest at results/cloud/<run_id>/ is read for the copy."""
    runs, results = tree
    run_id = "rid_legacy"
    p = _write_record(runs, "alien", "alien_2", run_id)
    _write_manifest(results, run_id, framework="claude_sdk_v1",
                    agent_model="zai/glm-5.2", version="v3", legacy_flat=True)
    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert _read(p)["version"] == "v3"
    assert counters["no_manifest"] == 0


def test_backfill_preserves_existing_stamp(tree):
    """A record already carrying a version is never re-derived/clobbered."""
    runs, results = tree
    run_id = "rid_pre"
    p = _write_record(runs, "alien", "alien_3", run_id,
                      version="v2", agent_model="anthropic/claude-opus-4-7")
    # Manifest disagrees (says v0) — must NOT override the producer literal.
    _write_manifest(results, run_id, framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7", version="v0")
    backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert _read(p)["version"] == "v2"


def test_backfill_is_idempotent(tree):
    runs, results = tree
    _write_record(runs, "alien", "alien_1", "rid_i")
    _write_manifest(results, "rid_i", framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7", version="v2")
    first = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert first["updated"] == 1
    second = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert second["updated"] == 0
    assert second["unchanged"] == 1


def test_backfill_dry_run_writes_nothing(tree):
    runs, results = tree
    p = _write_record(runs, "alien", "alien_1", "rid_d")
    _write_manifest(results, "rid_d", framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7", version="v2")
    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False,
                                              dry_run=True)
    assert counters["updated"] == 1
    assert _read(p).get("version") is None


def test_backfill_dry_run_does_not_cache_gcs_manifest(tree, monkeypatch):
    """A --dry-run must not mutate the results tree: when the manifest is only
    available from GCS, the dry-run reports the change but does NOT persist the
    fetched manifest to the local cache."""
    runs, results = tree
    _write_record(runs, "alien", "alien_1", "rid_gcs")  # version missing

    gcs_manifest = {"framework": "claude_sdk", "query_mode": "slayer",
                    "agent_model": "anthropic/claude-opus-4-7", "version": "v2"}
    monkeypatch.setattr(backfill_run_versions._cascade.gcs, "read_manifest",
                        lambda run_id, client=None: gcs_manifest)

    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=True,
                                              dry_run=True)
    assert counters["updated"] == 1
    # The local cache file must NOT have been written by the dry-run.
    cache_fp = results / _BENCH / "cloud" / "rid_gcs" / "manifest.json"
    assert not cache_fp.exists()
    # And the record itself stays unmodified.
    p = runs / _BENCH / "alien" / "alien_1" / "rid_gcs.json"
    assert _read(p).get("version") is None


def test_backfill_real_run_caches_gcs_manifest(tree, monkeypatch):
    """A real (non-dry) run still warms the local manifest cache on a GCS
    fetch — the cache suppression is scoped to dry-run only."""
    runs, results = tree
    _write_record(runs, "alien", "alien_1", "rid_gcs2")

    gcs_manifest = {"framework": "claude_sdk", "query_mode": "slayer",
                    "agent_model": "anthropic/claude-opus-4-7", "version": "v2"}
    monkeypatch.setattr(backfill_run_versions._cascade.gcs, "read_manifest",
                        lambda run_id, client=None: gcs_manifest)

    backfill_run_versions.backfill(_BENCH, allow_gcs=True, dry_run=False)
    cache_fp = results / _BENCH / "cloud" / "rid_gcs2" / "manifest.json"
    assert cache_fp.exists()


def test_backfill_agent_model_mismatch_flagged(tree):
    runs, results = tree
    p = _write_record(runs, "alien", "alien_1", "rid_m",
                      agent_model="stale/model")
    _write_manifest(results, "rid_m", framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7", version="v2")
    counters = backfill_run_versions.backfill(_BENCH, allow_gcs=False)
    assert counters["agent_model_mismatch"] == 1
    # Copy-not-clobber: the mismatch is flagged but NOT treated as fatal — the
    # missing version is still backfilled from the manifest, and the existing
    # (mismatching) agent_model is left untouched.
    rec = _read(p)
    assert rec["version"] == "v2"
    assert rec["agent_model"] == "stale/model"
