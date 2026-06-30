"""DEV-1591 stream 2 — version written AT THE SOURCE, copied (never
reconstructed) downstream.

* ``version_for_framework`` is this branch's identity map (applied only by
  producers).
* ``_apply_config_provenance`` (the producer write in ``grade_in_place``)
  stamps the record's ``version`` + ``agent_model`` from the agent config.
* ``copy_provenance_from_manifest`` + ``write_run_annotation`` COPY the
  producer's literal from the run manifest — no framework→version mapping, so
  a clean run merged/regraded here stays clean.
* legacy/back-compat: schema round-trips; untagged record reads as None.
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
from bird_interact_agents.eval.annotation_schema import (
    SubmissionAnnotation,
    SubmissionConfig,
)
from bird_interact_agents.eval.grade_in_place import _apply_config_provenance


@pytest.fixture
def pin_branch_map(monkeypatch):
    """Pin this branch's identity map so producer tests assert v2/v3
    deterministically, independent of the checked-in constant (which flips at
    merge)."""
    monkeypatch.setattr(
        versioning, "VERSION_BY_FRAMEWORK",
        {"claude_sdk": "v2", "claude_sdk_v1": "v3"},
    )


# --------------------------------------------------------------------------
# version_for_framework — the only place the map is applied
# --------------------------------------------------------------------------
def test_version_for_framework(pin_branch_map):
    assert versioning.version_for_framework("claude_sdk") == "v2"
    assert versioning.version_for_framework("claude_sdk_v1") == "v3"
    # Frameworks outside the v0–v3 taxonomy (otf/encoder) → None.
    assert versioning.version_for_framework("claude_sdk_otf_v1") is None
    assert versioning.version_for_framework(None) is None
    assert versioning.version_for_framework("") is None


# --------------------------------------------------------------------------
# Producer write (_apply_config_provenance) — the source
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


def test_producer_writes_version_and_model_from_config(pin_branch_map):
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    _apply_config_provenance(
        ann, SubmissionConfig(framework="claude_sdk_v1", agent_model="zai/glm-5.2"),
    )
    assert ann.version == "v3"
    assert ann.agent_model == "zai/glm-5.2"


def test_producer_clean_framework_v2(pin_branch_map):
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    _apply_config_provenance(
        ann, SubmissionConfig(framework="claude_sdk",
                              agent_model="anthropic/claude-opus-4-7"),
    )
    assert ann.version == "v2"
    assert ann.agent_model == "anthropic/claude-opus-4-7"


def test_producer_unmapped_framework_version_none(pin_branch_map):
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    _apply_config_provenance(
        ann, SubmissionConfig(framework="pydantic_ai_otf_encode", agent_model="x"),
    )
    assert ann.version is None      # outside the taxonomy
    assert ann.agent_model == "x"


def test_producer_no_config_is_noop():
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    _apply_config_provenance(ann, None)
    assert ann.version is None
    assert ann.agent_model is None


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
    again = SubmissionAnnotation.model_validate_json(ann.model_dump_json())
    assert again.version == "v2"
    assert again.agent_model == "anthropic/claude-opus-4-7"


# --------------------------------------------------------------------------
# copy_provenance_from_manifest — downstream copy, no reconstruction
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
                    agent_model="anthropic/claude-opus-4-7",
                    version="v2") -> None:
    dest = results / benchmark / "cloud" / run_id / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": framework, "agent_model": agent_model,
        "query_mode": "slayer",
    }
    if version is not None:
        payload["version"] = version
    dest.write_text(json.dumps(payload))


def test_copy_fills_version_and_model_from_manifest():
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    versioning.copy_provenance_from_manifest(
        ann, benchmark="mini-interact", run_id="r",
        manifest={"version": "v3", "agent_model": "zai/glm-5.2"},
    )
    assert ann.version == "v3"
    assert ann.agent_model == "zai/glm-5.2"


def test_copy_does_not_re_map_framework(pin_branch_map):
    """The crux: copy uses the manifest's LITERAL version. A clean run
    (manifest version=v0) stays v0 even though its framework is claude_sdk
    which THIS branch's map points at v2 — no reconstruction on this
    workstation."""
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    versioning.copy_provenance_from_manifest(
        ann, benchmark="mini-interact", run_id="r",
        manifest={"framework": "claude_sdk", "version": "v0",
                  "agent_model": "anthropic/claude-opus-4-7"},
    )
    assert ann.version == "v0"   # NOT v2


def test_copy_no_clobber_preset():
    ann = SubmissionAnnotation.model_validate(
        _ann_dict(version="v3", agent_model="preset/model")
    )
    versioning.copy_provenance_from_manifest(
        ann, benchmark="mini-interact", run_id="r",
        manifest={"version": "v0", "agent_model": "other/model"},
    )
    assert ann.version == "v3"
    assert ann.agent_model == "preset/model"


def test_copy_no_manifest_or_no_version_leaves_none(isolated_roots):
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    versioning.copy_provenance_from_manifest(
        ann, benchmark="mini-interact", run_id="nope")  # no manifest on disk
    assert ann.version is None
    ann2 = SubmissionAnnotation.model_validate(_ann_dict())
    versioning.copy_provenance_from_manifest(
        ann2, benchmark="mini-interact", run_id="r",
        manifest={"framework": "claude_sdk"})  # no version key
    assert ann2.version is None


def test_copy_self_loads_legacy_flat_manifest(isolated_roots):
    """The eval.annotate / regrade legacy-flat run dir is
    ``results/cloud/<run_id>/`` (no benchmark segment). The self-load must
    fall back to it to copy the producer's version."""
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    dest = results / "cloud" / run_id / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"version": "v3",
                                "agent_model": "anthropic/claude-sonnet-4-6"}))
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    versioning.copy_provenance_from_manifest(
        ann, benchmark="mini-interact", run_id=run_id)
    assert ann.version == "v3"
    assert ann.agent_model == "anthropic/claude-sonnet-4-6"


# --------------------------------------------------------------------------
# write_run_annotation — persists, copying from manifest when missing
# --------------------------------------------------------------------------
def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_write_copies_version_from_local_manifest(isolated_roots):
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id, version="v2")
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    write_run_annotation(ann, dest, benchmark="mini-interact", run_id=run_id)
    on_disk = _read(dest)
    assert on_disk["version"] == "v2"
    assert on_disk["agent_model"] == "anthropic/claude-opus-4-7"


def test_write_preserves_producer_version(isolated_roots):
    """A record the producer already stamped is persisted as-is; the manifest
    copy is a no-op even if the manifest disagrees."""
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id, version="v0")  # disagrees on purpose
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id, version="v2"))
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    write_run_annotation(ann, dest, benchmark="mini-interact", run_id=run_id)
    assert _read(dest)["version"] == "v2"   # producer literal preserved


def test_write_path_inference_fallback(isolated_roots):
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id, version="v2")
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    write_run_annotation(ann, dest)  # no explicit benchmark/run_id
    assert _read(dest)["version"] == "v2"


def test_write_outside_runs_root_no_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(tmp_path / "results"))
    ann = SubmissionAnnotation.model_validate(_ann_dict())
    dest = tmp_path / "loose" / "file.json"  # not under runs_root, no args
    write_run_annotation(ann, dest)
    assert _read(dest)["version"] is None


def test_no_overwrite_copies_then_preserves_existing(isolated_roots):
    runs, results = isolated_roots
    run_id = "20260601t1000-x-slayer-zzzzzz"
    _write_manifest(results, run_id, version="v2")
    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id=run_id,
    )
    ann = SubmissionAnnotation.model_validate(_ann_dict(run_id=run_id))
    assert write_run_annotation_no_overwrite(
        ann, dest, benchmark="mini-interact", run_id=run_id) is True
    assert _read(dest)["version"] == "v2"
    ann2 = SubmissionAnnotation.model_validate(
        _ann_dict(run_id=run_id, version="vDIFFERENT")
    )
    assert write_run_annotation_no_overwrite(
        ann2, dest, benchmark="mini-interact", run_id=run_id) is False
    assert _read(dest)["version"] == "v2"  # original preserved
