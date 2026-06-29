"""DEV-1591 stream 2 — self-tag ``runs/`` records with ``version`` +
``agent_model`` so this branch's modified runs (v2/v3) don't pollute clean
v0/v1 cascade stats.

Covers:
* ``eval.versioning.resolve_version`` precedence (override table → framework
  map → default-v0, with present-but-unmapped frameworks kept separable).
* ``provenance_from_manifest``.
* ``SubmissionAnnotation`` round-trips with/without the new optional fields
  (legacy files parse, ``version is None``).
* Write-time stamping in ``annotation_io.write_run_annotation`` /
  ``write_run_annotation_no_overwrite``: explicit ``benchmark``/``run_id``
  (covers ``repo_root`` writers — Codex High #3), path-inference fallback,
  the override table working with no manifest on disk, the no-clobber rule,
  and a path outside ``runs_root`` stamping nothing without crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_interact_agents.eval import versioning
from bird_interact_agents.eval.annotation_io import (
    write_run_annotation,
    write_run_annotation_no_overwrite,
    run_annotation_path,
)
from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation


# The three run-ids positively attributable to this branch (the only durable
# v2/v3 signal — manifests carry no code-version field). Mirror the override
# table so the test breaks loudly if a run-id is dropped/retyped.
_CEA364 = "20260629t1209-claudes-slayer-cea364"  # modified-v0 → v2
_4DF43F = "20260624t0833-claudes-slayer-4df43f"  # modified-v0 → v2
_4246FD = "20260624t0844-claudes-slayer-4246fd"  # modified-v1 → v3


# --------------------------------------------------------------------------
# resolve_version
# --------------------------------------------------------------------------
def test_resolve_version_override_beats_framework_map():
    # cea364's framework token is claude_sdk (would map to v0) but it is a
    # modified-v0 run → the override table must win.
    assert versioning.resolve_version(_CEA364, "claude_sdk") == "v2"
    assert versioning.resolve_version(_4DF43F, "claude_sdk") == "v2"
    assert versioning.resolve_version(_4246FD, "claude_sdk_v1") == "v3"


def test_resolve_version_framework_map():
    assert versioning.resolve_version("20260601t1000-x-slayer-zzzzzz",
                                      "claude_sdk") == "v0"
    assert versioning.resolve_version("20260601t1000-x-slayer-zzzzzz",
                                      "claude_sdk_v1") == "v1"


def test_resolve_version_missing_framework_defaults_v0():
    # In-flight clean runs are submitted from origin/main; when the manifest
    # is absent the framework is unknown → default to v0 (per the decision).
    assert versioning.resolve_version("anything", None) == "v0"
    assert versioning.resolve_version("anything", "") == "v0"


def test_resolve_version_present_but_unmapped_framework_stays_separable():
    # An otf/encoder framework must NOT be folded into v0 (else --version v0
    # would wrongly include it — Codex Medium #4). Keep it separable by
    # returning the framework token verbatim.
    assert versioning.resolve_version(
        "20260601t1000-x-slayer-zzzzzz", "claude_sdk_otf_v1"
    ) == "claude_sdk_otf_v1"


# --------------------------------------------------------------------------
# provenance_from_manifest
# --------------------------------------------------------------------------
def test_provenance_from_manifest():
    fw, model = versioning.provenance_from_manifest(
        {"framework": "claude_sdk", "agent_model": "anthropic/claude-opus-4-7"}
    )
    assert fw == "claude_sdk"
    assert model == "anthropic/claude-opus-4-7"


def test_provenance_from_manifest_none_and_empty():
    assert versioning.provenance_from_manifest(None) == (None, None)
    assert versioning.provenance_from_manifest({}) == (None, None)


# --------------------------------------------------------------------------
# Schema round-trips
# --------------------------------------------------------------------------
def _ann_dict(*, iid="alien_1", db="alien", run_id="r1", **extra) -> dict:
    d = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": iid,
        "selected_database": db,
        "task_annotation_ref": f"annotations/mini-interact/{db}/{iid}.task.json",
        "annotated_by": "test",
        "annotated_at": "2026-06-01T10:00:00+00:00",
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{iid}/attempt-1.json",
            "predicted_row_count": 1,
            "duration_s": 1.0,
            "cost_usd_agent": 0.0,
            "cost_usd_user_sim": 0.0,
            "n_agent_turns": 1,
            "n_ask_user_calls": 0,
        },
        "evaluation": {
            "phase1_against_original_gold": "pass",
            "phase1_against_audited_primary": "pass",
            "phase1_against_any_audited_variant": "pass",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "novel_reading_judgment": None,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "correct",
            "matched_variant_id": "primary",
            "rationale": "",
        },
        "failure_classification": {
            "primary": "no_fail",
            "secondary": [],
            "agent_at_fault": False,
            "remediation_target": "other",
            "remediation_text": "",
            "details": "",
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


def test_schema_legacy_record_parses_with_version_none():
    """A record written before DEV-1591 (no version/agent_model keys) must
    still parse, with both new fields defaulting to None."""
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    assert ann.version is None
    assert ann.agent_model is None


def test_schema_round_trips_new_fields():
    ann = SubmissionAnnotation.model_validate(
        _ann_dict(version="v2", agent_model="anthropic/claude-opus-4-7")
    )
    assert ann.version == "v2"
    assert ann.agent_model == "anthropic/claude-opus-4-7"
    # survives a JSON round-trip
    again = SubmissionAnnotation.model_validate_json(ann.model_dump_json())
    assert again.version == "v2"
    assert again.agent_model == "anthropic/claude-opus-4-7"


# --------------------------------------------------------------------------
# stamp_provenance
# --------------------------------------------------------------------------
@pytest.fixture
def isolated_roots(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(results))
    return runs, results


def _write_manifest(results: Path, run_id: str, *, benchmark="mini-interact",
                    framework="claude_sdk",
                    agent_model="anthropic/claude-opus-4-7") -> None:
    dest = results / benchmark / "cloud" / run_id / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "framework": framework, "agent_model": agent_model,
        "query_mode": "slayer",
    }))


def test_stamp_from_explicit_manifest(isolated_roots):
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    versioning.stamp_provenance(
        ann, benchmark="mini-interact", run_id="20260601t1000-x-slayer-zzzzzz",
        manifest={"framework": "claude_sdk_v1",
                  "agent_model": "anthropic/claude-sonnet-4-6"},
    )
    assert ann.version == "v1"
    assert ann.agent_model == "anthropic/claude-sonnet-4-6"


def test_stamp_override_run_id_without_manifest(isolated_roots):
    """No manifest anywhere on disk: the override table (keyed by run-id) must
    still tag cea364 as v2; agent_model stays None."""
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=_CEA364))
    versioning.stamp_provenance(ann, benchmark="mini-interact", run_id=_CEA364)
    assert ann.version == "v2"
    assert ann.agent_model is None


def test_stamp_self_loads_local_manifest(isolated_roots):
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id)
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    versioning.stamp_provenance(ann, benchmark="mini-interact", run_id=run_id)
    assert ann.version == "v0"
    assert ann.agent_model == "anthropic/claude-opus-4-7"


def test_stamp_self_loads_legacy_flat_manifest(isolated_roots):
    """Codex (loop r2): the eval.annotate / regrade legacy-flat run dir is
    ``results/cloud/<run_id>/`` (no benchmark segment). The self-load must
    fall back to it, else a clean claude_sdk_v1 run there is mis-stamped v0
    with no agent_model."""
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    # ONLY the legacy-flat manifest exists (no benchmark-scoped one).
    dest = results / "cloud" / run_id / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "framework": "claude_sdk_v1",
        "agent_model": "anthropic/claude-sonnet-4-6", "query_mode": "slayer",
    }))
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    versioning.stamp_provenance(ann, benchmark="mini-interact", run_id=run_id)
    assert ann.version == "v1"   # not the default v0
    assert ann.agent_model == "anthropic/claude-sonnet-4-6"


def test_stamp_never_clobbers_preset_fields(isolated_roots):
    ann = SubmissionAnnotation.model_validate(
        _ann_dict(version="v3", agent_model="preset/model")
    )
    versioning.stamp_provenance(
        ann, benchmark="mini-interact", run_id="20260601t1000-x-slayer-zzzzzz",
        manifest={"framework": "claude_sdk", "agent_model": "other/model"},
    )
    assert ann.version == "v3"
    assert ann.agent_model == "preset/model"


# --------------------------------------------------------------------------
# write_run_annotation stamping
# --------------------------------------------------------------------------
def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_write_run_annotation_stamps_from_local_manifest(isolated_roots):
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id)
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    write_run_annotation(ann, dest, benchmark="mini-interact", run_id=run_id)
    on_disk = _read(dest)
    assert on_disk["version"] == "v0"
    assert on_disk["agent_model"] == "anthropic/claude-opus-4-7"


def test_write_run_annotation_explicit_args_cover_repo_root_writer(tmp_path,
                                                                   monkeypatch):
    """Codex High #3: a writer that passes an explicit out-of-tree path (the
    regrade ``repo_root`` case) still gets stamped because benchmark/run_id
    are passed explicitly — no reliance on path.relative_to(runs_root())."""
    # runs_root points elsewhere; dest is deliberately NOT under it.
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(tmp_path / "results"))
    _write_manifest(tmp_path / "results", _CEA364)
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=_CEA364))
    dest = tmp_path / "custom_repo" / "runs" / "mini-interact" / "alien" / \
        "alien_1" / f"{_CEA364}.json"
    write_run_annotation(ann, dest, benchmark="mini-interact", run_id=_CEA364)
    assert _read(dest)["version"] == "v2"


def test_write_run_annotation_path_inference_fallback(isolated_roots):
    """No explicit benchmark/run_id: infer from the destination path when it
    lives under runs_root."""
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id)
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    write_run_annotation(ann, dest)  # no explicit benchmark/run_id
    assert _read(dest)["version"] == "v0"


def test_write_run_annotation_outside_runs_root_no_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(tmp_path / "results"))
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    dest = tmp_path / "loose" / "file.json"  # not under runs_root, no args
    write_run_annotation(ann, dest)
    assert _read(dest)["version"] is None  # nothing to infer → left unstamped


def test_no_overwrite_stamps_on_write_but_preserves_existing(isolated_roots):
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id)
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    assert write_run_annotation_no_overwrite(
        ann, dest, benchmark="mini-interact", run_id=run_id) is True
    assert _read(dest)["version"] == "v0"
    # A second write of the SAME attempt must be preserved (no overwrite) and
    # leave the existing stamped content intact.
    ann2 = SubmissionAnnotation.model_validate(
        _ann_dict(run_id=run_id, version="vDIFFERENT")
    )
    assert write_run_annotation_no_overwrite(
        ann2, dest, benchmark="mini-interact", run_id=run_id) is False
    assert _read(dest)["version"] == "v0"  # original preserved
