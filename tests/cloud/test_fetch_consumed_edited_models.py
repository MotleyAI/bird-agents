"""DEV-1778: cloud round-trip for `consumed_edited_models` — the merge
preserves the per-task field, and `driver.fetch` aggregates the per-db list
into the on-disk manifest.json before writing it (Codex #8)."""
from __future__ import annotations

import json
from pathlib import Path

_CONSUMED = {"db": "alien", "instance_id": "alien_1", "store_fp": "beef" * 16}


def _ann_dict(instance_id="alien_1", *, consumed=None) -> dict:
    body = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": "alien",
        "task_annotation_ref": f"annotations/mini-interact/alien/{instance_id}.task.json",
        "annotated_by": "auto",
        "annotated_at": "2026-08-11",
        "submission": {
            "cloud_run_id": "r1",
            "trajectory_path": f"rows/{instance_id}/attempt-1.json",
        },
        "evaluation": {
            "phase1_against_original_gold": "pass",
            "phase1_against_audited_primary": "pass",
            "phase1_against_any_audited_variant": "pass",
            "verdict": "correct",
        },
        "failure_classification": {
            "primary": "no_fail",
            "agent_at_fault": False,
            "remediation_target": "other",
        },
    }
    if consumed is not None:
        body["consumed_edited_models"] = consumed
    return body


def test_merge_preserves_consumed_edited_models(tmp_path, monkeypatch):
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud.post_run_merge import merge_submission_annotations

    main_checkout = tmp_path / "checkout"
    main_checkout.mkdir()
    monkeypatch.setattr(_paths, "main_checkout_root", lambda: main_checkout)
    rows = tmp_path / "downloaded" / "rows" / "alien_1"
    rows.mkdir(parents=True)
    (rows / "submission_annotation.json").write_text(
        json.dumps(_ann_dict("alien_1", consumed=_CONSUMED))
    )

    merge_submission_annotations(
        downloaded_run_dir=tmp_path / "downloaded", run_id="r1",
        benchmark="mini-interact",
    )
    dest = main_checkout / "runs" / "mini-interact" / "alien" / "alien_1" / "r1.json"
    assert json.loads(dest.read_text())["consumed_edited_models"] == _CONSUMED


def test_fetch_stamps_manifest_with_per_db_list(tmp_path, monkeypatch):
    from bird_interact_agents.cloud import driver

    def fake_download(run_id, dest, client=None):  # noqa: ARG001
        for iid, consumed in (
            ("alien_1", _CONSUMED),
            ("alien_2", {"db": "alien", "instance_id": "alien_2", "store_fp": "f2"}),
            ("alien_3", None),  # non-apply task -> no record
        ):
            d = Path(dest) / "rows" / iid
            d.mkdir(parents=True)
            (d / "submission_annotation.json").write_text(
                json.dumps(_ann_dict(iid, consumed=consumed))
            )

    class _Report:
        def model_dump(self):
            return {}

    # collate() runs AFTER driver.fetch writes manifest.json. Assert here that
    # the aggregate is already on BOTH the in-memory manifest and the on-disk
    # file — proving aggregation happened BEFORE the write (Codex #8).
    def fake_collate(dest, manifest):
        assert manifest.get("consumed_edited_models"), "aggregate missing pre-collate"
        on_disk = json.loads((Path(dest) / "manifest.json").read_text())
        assert on_disk.get("consumed_edited_models"), "manifest.json written without aggregate"
        return {}

    monkeypatch.setattr(driver, "default_gcs_client", lambda: None)
    monkeypatch.setattr(driver, "_benchmark_for_dataset", lambda _d: "mini-interact")
    monkeypatch.setattr(driver.gcs, "read_manifest",
                        lambda run_id, client=None: {"run_id": run_id,
                                                     "dataset": "mini-interact",
                                                     "framework": "claude_sdk"})
    monkeypatch.setattr(driver.gcs, "concurrent_download_prefix", fake_download)
    monkeypatch.setattr(driver.paths, "results_root", lambda: tmp_path / "results")
    monkeypatch.setattr(driver.paths, "slayer_models_otf_root",
                        lambda *, benchmark=None: tmp_path / "otf")
    monkeypatch.setattr(driver._collation, "collate", fake_collate)
    monkeypatch.setattr(driver._post_run_merge, "merge_post_run_into_warm_cache",
                        lambda **_k: {})
    monkeypatch.setattr(driver._post_run_merge, "merge_submission_annotations",
                        lambda **_k: _Report())
    monkeypatch.setattr(driver, "_emit_cascading_phase1_on_fetch",
                        lambda *, dest, metrics, benchmark, run_id: metrics)
    monkeypatch.setattr(driver.cluster, "head_is_alive", lambda run_id: False)

    driver.fetch("r1", kill_after_fetch=False)

    manifest_path = tmp_path / "results" / "mini-interact" / "cloud" / "r1" / "manifest.json"
    consumed = json.loads(manifest_path.read_text())["consumed_edited_models"]
    assert {(r["db"], r["instance_id"], r["store_fp"]) for r in consumed} == {
        ("alien", "alien_1", _CONSUMED["store_fp"]),
        ("alien", "alien_2", "f2"),
    }
