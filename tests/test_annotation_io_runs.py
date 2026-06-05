"""DEV-1533: run annotation I/O path helpers + latest_run_per_instance."""
from __future__ import annotations

import json


def _make_ann_dict(instance_id: str, db: str, run_id: str, annotated_at: str) -> dict:
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "task_annotation_ref": f"annotations/mini-interact/{db}/{instance_id}.task.json",
        "annotated_by": "auto",
        "annotated_at": annotated_at,
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{instance_id}/attempt-1.json",
        },
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": False,
            "novel_reading_judgment": None,
            "correct_under_numeric_epsilon": False,
            "correct_under_trailing_whitespace": False,
            "correct_under_column_order": False,
            "correct_under_case_fold": False,
            "numeric_epsilon": 1e-6,
            "verdict": "agent_miss",
            "matched_variant_id": None,
            "rationale": "",
        },
        "failure_classification": {
            "primary": "agent_miss",
            "secondary": [],
            "agent_at_fault": True,
            "remediation_target": "agent",
            "details": "",
        },
        "decision_point": None,
        "user_sim_interaction": {"n_asks": 0, "key_responses": [],
                                 "disclosed_resolutions": [],
                                 "undisclosed_resolutions": []},
        "original_gold_annotated_correct": True,
    }


def test_run_annotation_path_shape(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import run_annotation_path
    p = run_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_1",
        run_id="r1",
    )
    assert p == tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r1.json"


def test_run_trajectory_path_shape(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import run_trajectory_path
    p = run_trajectory_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_1",
        run_id="r1",
    )
    assert p == tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r1.trajectory.json"


def test_run_annotation_path_uses_canonical_hyphenated_benchmark(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import run_annotation_path
    p = run_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_1",
        run_id="r1",
    )
    assert "mini-interact" in p.parts
    assert "mini_interact" not in p.parts


def test_iter_run_annotations_filters_by_run_id(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import iter_run_annotations

    runs = tmp_path / "runs" / "mini-interact" / "alien"
    for iid, run_id in [("alien_1", "r1"), ("alien_2", "r1"), ("alien_3", "r2")]:
        d = runs / iid
        d.mkdir(parents=True)
        (d / f"{run_id}.json").write_text(
            json.dumps(_make_ann_dict(iid, "alien", run_id, "2026-06-01T10:00:00+00:00"))
        )

    r1_anns = iter_run_annotations(benchmark="mini-interact", run_id="r1")
    assert len(r1_anns) == 2
    assert all(ann.submission.cloud_run_id == "r1" for _, ann in r1_anns)

    all_anns = iter_run_annotations(benchmark="mini-interact")
    assert len(all_anns) == 3


def test_iter_run_annotations_skips_trajectory_sidecars(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import iter_run_annotations

    runs = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
    runs.mkdir(parents=True)
    (runs / "r1.json").write_text(
        json.dumps(_make_ann_dict("alien_1", "alien", "r1", "2026-06-01T10:00:00+00:00"))
    )
    (runs / "r1.trajectory.json").write_text('{"trajectory": []}')

    anns = iter_run_annotations(benchmark="mini-interact")
    assert len(anns) == 1
    assert anns[0][0].name == "r1.json"


def test_iter_run_annotations_warns_on_corrupt_file(monkeypatch, tmp_path, caplog):
    """Corrupt run-annotation files MUST be reported via a logger.warning
    so the operator sees the denominator shrink — silent skip lets
    cascade-aggregated rates inflate without anyone noticing."""
    import logging
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import iter_run_annotations

    runs = tmp_path / "runs" / "mini-interact" / "alien"
    (runs / "alien_1").mkdir(parents=True)
    (runs / "alien_1" / "r1.json").write_text(
        json.dumps(_make_ann_dict("alien_1", "alien", "r1", "2026-06-01T10:00:00+00:00"))
    )
    # Corrupt second file — invalid JSON.
    (runs / "alien_2").mkdir(parents=True)
    corrupt = runs / "alien_2" / "r1.json"
    corrupt.write_text("{not valid json")

    with caplog.at_level(logging.WARNING, logger="bird_interact_agents.eval.annotation_io"):
        anns = iter_run_annotations(benchmark="mini-interact", run_id="r1")

    assert len(anns) == 1, "valid file should still be returned"
    assert any(
        "iter_run_annotations" in rec.message and str(corrupt) in rec.message
        for rec in caplog.records
    ), f"expected warning naming {corrupt}; got {[r.message for r in caplog.records]}"


def test_latest_run_per_instance_picks_max_annotated_at(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import latest_run_per_instance

    runs = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
    runs.mkdir(parents=True)
    # Two runs for alien_1: r1 is older, r2 is newer (by annotated_at).
    (runs / "r1.json").write_text(
        json.dumps(_make_ann_dict("alien_1", "alien", "r1", "2026-06-01T08:00:00+00:00"))
    )
    (runs / "r2.json").write_text(
        json.dumps(_make_ann_dict("alien_1", "alien", "r2", "2026-06-01T10:00:00+00:00"))
    )

    latest = latest_run_per_instance(benchmark="mini-interact")
    assert ("alien", "alien_1") in latest
    run_id, ann = latest[("alien", "alien_1")]
    assert run_id == "r2"
    assert ann.submission.cloud_run_id == "r2"


def test_latest_run_per_instance_handles_non_timestamped_run_ids(monkeypatch, tmp_path):
    """Local run_ids may not be timestamps — annotated_at always breaks the tie."""
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import latest_run_per_instance

    runs = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
    runs.mkdir(parents=True)
    (runs / "abc.json").write_text(
        json.dumps(_make_ann_dict("alien_1", "alien", "abc", "2026-06-01T09:00:00+00:00"))
    )
    (runs / "xyz.json").write_text(
        json.dumps(_make_ann_dict("alien_1", "alien", "xyz", "2026-06-01T11:00:00+00:00"))
    )

    latest = latest_run_per_instance(benchmark="mini-interact")
    run_id, _ = latest[("alien", "alien_1")]
    assert run_id == "xyz"


def test_write_run_annotation_no_overwrite_skips_older_attempt(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.annotation_io import (
        run_annotation_path, write_run_annotation_no_overwrite,
    )
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation, SubmissionMetadata, UserSimInteraction,
    )

    def _make_ann(attempt: int) -> SubmissionAnnotation:
        return SubmissionAnnotation(
            instance_id="alien_1", selected_database="alien",
            task_annotation_ref="x", annotated_by="test",
            annotated_at="2026-06-01T10:00:00+00:00",
            submission=SubmissionMetadata(
                cloud_run_id="r1",
                trajectory_path=f"rows/alien_1/attempt-{attempt}.json",
            ),
            evaluation=SubmissionEvaluation(
                phase1_against_original_gold="fail",
                phase1_against_audited_primary="fail",
                phase1_against_any_audited_variant="fail",
                correct_up_to_tie_order=False,
                correct_under_numeric_epsilon=False,
                correct_under_trailing_whitespace=False,
                correct_under_column_order=False,
                correct_under_case_fold=False,
                numeric_epsilon=1e-6,
                verdict="agent_miss",
                rationale=f"attempt-{attempt}",
            ),
            failure_classification=FailureClassification(
                primary="agent_miss", agent_at_fault=True, remediation_target="agent",
            ),
            user_sim_interaction=UserSimInteraction(),
        )

    dest = run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1", run_id="r1",
    )
    ann2 = _make_ann(2)
    write_run_annotation_no_overwrite(ann2, dest)
    assert dest.exists()

    # attempt-1 must NOT overwrite attempt-2
    ann1 = _make_ann(1)
    written = write_run_annotation_no_overwrite(ann1, dest)
    assert not written
    surviving = json.loads(dest.read_text())
    assert surviving["evaluation"]["rationale"] == "attempt-2"
