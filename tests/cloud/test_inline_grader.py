"""DEV-1515: cloud worker inline grading + per-row write to artefacts.

After ``execute_submit_action`` returns, ray_app.py must call
``grade_in_place.grade_and_write`` so each per-row artefact dir contains
a ``submission_annotation.json`` written by tolerant_grader.

This test isolates the wiring contract — the actual grader logic is
covered in ``tests/test_tolerant_grader_*.py``.
"""
from __future__ import annotations

from pathlib import Path


def test_ray_app_writes_submission_annotation_per_task(monkeypatch, tmp_path):
    """The worker MUST invoke grade_and_write for each task it runs."""
    from bird_interact_agents.cloud import ray_app
    from bird_interact_agents.eval import grade_in_place

    calls: list[dict] = []

    def fake_grade(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        rows_dir = kwargs["rows_dir"]
        instance_id = kwargs["instance_id"]
        d = rows_dir / instance_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "submission_annotation.json").write_text("{}")

    # After DEV-1515 round-4, the per-task grader helper lives in
    # ``grade_in_place`` (canonical location, shared with the local
    # runner); ``ray_app._grade_one_submission`` is now a thin alias.
    # Patch BOTH so the test passes regardless of which lookup path
    # the wiring code uses.
    monkeypatch.setattr(
        grade_in_place, "grade_and_write", fake_grade, raising=True,
    )
    monkeypatch.setattr(
        ray_app, "grade_and_write", fake_grade, raising=True,
    )

    # The simulated submit hook — adapter for ray_app's per-task path.
    # ray_app exposes a `_grade_one_submission(task_data, submitted_sql,
    # rows_dir, run_id, benchmark)` helper that is the integration seam.
    ray_app._grade_one_submission(
        task_data={
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT gold"],
            "original_sol_sql": ["SELECT gold"],
        },
        submitted_sql="SELECT predicted",
        rows_dir=tmp_path,
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        conn=None,
    )

    assert len(calls) == 1
    assert (tmp_path / "alien_1" / "submission_annotation.json").exists()


def test_ray_app_does_not_emit_legacy_phase1_passed_fields(monkeypatch, tmp_path):
    """The per-row result dict that ray_app uploads must NOT contain
    the legacy raw bool fields — those have been replaced by the
    submission_annotation path."""
    import inspect
    from bird_interact_agents.cloud import ray_app

    src = inspect.getsource(ray_app)
    assert "phase1_passed_audited" not in src
    assert "phase1_passed_original" not in src


def test_worker_uses_implicit_annotation_when_file_missing(monkeypatch, tmp_path):
    """If no <instance>.task.json exists in the baked annotations dir,
    the worker falls back to implicit_task_annotation IN MEMORY — no
    file gets written under annotations/."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.cloud import ray_app

    # Empty annotations dir.
    annotations_root = tmp_path / "annotations"
    annotations_root.mkdir()
    monkeypatch.setattr(
        paths_mod, "annotations_root", lambda: annotations_root,
    )

    captured: list[dict] = []

    def fake_grade(**kwargs):  # noqa: ANN003
        captured.append(kwargs)
        d = kwargs["rows_dir"] / kwargs["instance_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "submission_annotation.json").write_text("{}")

    from bird_interact_agents.eval import grade_in_place
    monkeypatch.setattr(
        grade_in_place, "grade_and_write", fake_grade, raising=True,
    )
    monkeypatch.setattr(ray_app, "grade_and_write", fake_grade, raising=True)

    ray_app._grade_one_submission(
        task_data={
            "instance_id": "alien_99",
            "selected_database": "alien",
            "sol_sql": ["SELECT gold"],
            "original_sol_sql": ["SELECT gold"],
            "amb_user_query": "x",
        },
        submitted_sql="SELECT predicted",
        rows_dir=tmp_path / "rows",
        run_id="r1",
        benchmark="mini-interact",
        db_path=Path("/dev/null"),
        conn=None,
    )

    assert len(captured) == 1
    # No <instance>.task.json was written.
    assert not list(annotations_root.rglob("*.task.json"))


# ---------------------------------------------------------------------------
# DEV-1515 round 6: when the cloud inline grader raises, the worker MUST
# still upload a fail-everything submission_annotation.json — otherwise
# the post-fetch ``cascading_phase1`` aggregator either skips the block
# entirely (no per-row anns at all) or raises FileNotFoundError as soon
# as one missing-annotation row is encountered, so a single broken task
# wipes the new N1-N9 metrics from ``eval.json``.
# ---------------------------------------------------------------------------


def test_cloud_grader_failure_uploads_fail_everything_annotation(
    monkeypatch, fake_gcs_bucket,
):
    """``_run_one_in_actor`` invokes the inline grader; when it raises,
    the worker writes + uploads a fail-everything
    ``submission_annotation.json`` so every cloud row contributes to
    the cascade denominator."""
    import json
    import pytest
    from bird_interact_agents.cloud import ray_app

    RUN_ID = "20260602T1200-pydanticai-raw-round6"

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"],
            "database": task_data.get("selected_database", "db_a"),
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "duration_s": 0.01, "error": None,
            "submitted_sql": "SELECT 1",
        }
    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task,
    )

    # The load-bearing patch: the cloud-side inline grader RAISES. The
    # worker's except branch must catch + write + upload the
    # fail-everything fallback. Patch the cloud module's reference (the
    # name the worker actually looks up) so we don't need to also poke
    # at the canonical grade_in_place location.
    def _raise_grader(**_kw):
        raise RuntimeError("simulated grader explosion")
    monkeypatch.setattr(
        ray_app, "_grade_one_submission", _raise_grader, raising=True,
    )

    # No-op the rest of the upload-back triple — irrelevant to this
    # test and they hit the wider FS.
    from bird_interact_agents.cloud import upload_back
    monkeypatch.setattr(
        upload_back, "upload_per_task_debug", lambda **kw: None,
    )
    monkeypatch.setattr(
        upload_back, "upload_per_task_setup_sessions", lambda **kw: None,
    )
    monkeypatch.setattr(
        upload_back, "upload_otf_reference_delta", lambda **kw: None,
    )

    actor = ray_app._LocalActor(
        {"framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
         "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
         "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
         "patience": 3, "strict": False, "use_audited_gold_sql": False,
         "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
         "slayer_storage_root": "/data/slayer_models",
         "data_dir": "/data/mini-interact"},
        RUN_ID, 1, gcs_client=client,
    )
    # MUST NOT raise — grader failure is diagnostic, not result-of-record.
    actor.run_one({"instance_id": "db_a_1", "selected_database": "db_a"})

    # Per-row submission_annotation.json blob landed in GCS storage.
    ann_blob_keys = [
        k for k in store
        if k.endswith("/db_a_1/submission_annotation.json")
    ]
    assert ann_blob_keys, (
        f"fail-everything annotation should have been uploaded; "
        f"store keys: {sorted(store)}"
    )
    payload = json.loads(store[ann_blob_keys[0]].decode())
    # Fail-everything shape: every cascade tier is fail/False, verdict
    # is invalid, primary is 'other' (the cascade was never actually run).
    assert payload["evaluation"]["verdict"] == "invalid"
    assert payload["evaluation"]["phase1_against_original_gold"] == "fail"
    assert payload["evaluation"]["phase1_against_any_audited_variant"] == "fail"
    assert payload["evaluation"]["correct_up_to_tie_order"] is False
    assert payload["failure_classification"]["primary"] == "other"
    assert "simulated grader explosion" in (
        payload["failure_classification"]["details"]
    )


def test_cloud_no_submitted_sql_short_circuits_before_real_grader(
    monkeypatch, fake_gcs_bucket,
):
    """When ``run_one_task`` returns a row with no ``submitted_sql``
    (agent crashed before reaching submit), the cloud worker MUST
    short-circuit BEFORE calling the real grader. Pre-fix the cloud
    path passed ``str(row.get("submitted_sql") or "") == ""`` through
    to ``_grade_one_submission``; SQLite may silently return an empty
    rowset for the empty statement, and ``_set_equal([], [])`` falsely
    passes N1/N2/N3 whenever the gold result is also empty. Mirrors
    the local runner's guard in ``run._grade_local_row``."""
    import json
    from bird_interact_agents.cloud import ray_app

    RUN_ID = "20260602T1215-cloud-no-submit-round7"

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    async def fake_run_one_task(task_data, **_kw):
        # Critical: NO ``submitted_sql`` on the result row — simulates
        # an agent crash before the submit step.
        return {
            "instance_id": task_data["instance_id"],
            "database": task_data.get("selected_database", "db_a"),
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "duration_s": 0.01, "error": "boom",
            "submitted_sql": None,
        }
    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task,
    )

    # Spy on _grade_one_submission — it must NEVER be invoked when
    # submitted_sql is missing, otherwise SQLite's empty-statement
    # behaviour would surface in the grader's pred_rows.
    grader_calls: list[dict] = []

    def _spy_grader(**kwargs):
        grader_calls.append(dict(kwargs))
        raise AssertionError(
            "_grade_one_submission must NOT be called when "
            "submitted_sql is missing; the short-circuit should fire"
        )
    monkeypatch.setattr(
        ray_app, "_grade_one_submission", _spy_grader, raising=True,
    )

    from bird_interact_agents.cloud import upload_back
    monkeypatch.setattr(upload_back, "upload_per_task_debug", lambda **kw: None)
    monkeypatch.setattr(
        upload_back, "upload_per_task_setup_sessions", lambda **kw: None,
    )
    monkeypatch.setattr(
        upload_back, "upload_otf_reference_delta", lambda **kw: None,
    )

    actor = ray_app._LocalActor(
        {"framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
         "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
         "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
         "patience": 3, "strict": False, "use_audited_gold_sql": False,
         "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
         "slayer_storage_root": "/data/slayer_models",
         "data_dir": "/data/mini-interact"},
        RUN_ID, 1, gcs_client=client,
    )
    actor.run_one({"instance_id": "db_a_1", "selected_database": "db_a"})

    # _grade_one_submission was not called — the short-circuit fired.
    assert grader_calls == [], (
        f"_grade_one_submission must not be invoked on missing-submit "
        f"path; got calls: {grader_calls}"
    )

    # And the fail-everything annotation still landed in GCS for the
    # cascading_phase1 denominator.
    ann_keys = [
        k for k in store
        if k.endswith("/db_a_1/submission_annotation.json")
    ]
    assert ann_keys, (
        f"fail-everything annotation missing from upload; keys={sorted(store)}"
    )
    payload = json.loads(store[ann_keys[0]].decode())
    assert payload["evaluation"]["verdict"] == "invalid"
    assert payload["failure_classification"]["primary"] == "other"
    assert (
        "no submitted_sql" in payload["failure_classification"]["details"]
        or "task errored before reaching submit"
        in payload["failure_classification"]["details"]
    )


def test_cloud_uploads_annotation_before_attempt_row(
    monkeypatch, fake_gcs_bucket,
):
    """Codex r7 ordering: ``_run_one_in_actor`` MUST upload the per-row
    submission_annotation.json BEFORE the attempt row blob. ``driver.
    wait_until_done`` counts attempt rows to decide ``done``; if the
    row landed first, non-detached ``submit`` + immediate ``fetch``
    could race the annotation upload and the cascade aggregator would
    either drop ``cascading_phase1`` or surface
    ``cascading_phase1_error``.

    This test records the call order of the two ``_gcs.write_*``
    helpers and asserts the annotation write index is strictly less
    than the row write index.
    """
    from bird_interact_agents.cloud import ray_app, upload_back

    RUN_ID = "20260602T1230-cloud-order-round7"

    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"],
            "database": task_data.get("selected_database", "db_a"),
            "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0,
            "duration_s": 0.01, "error": None,
            "submitted_sql": "SELECT 1",
        }
    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task,
    )

    # Stub the grader so it just writes a minimal annotation file the
    # caller (the cloud worker) then uploads. Mirrors the success path.
    def _fake_grade(**kwargs):
        rows_dir = kwargs["rows_dir"]
        instance_id = kwargs["instance_id"]
        d = rows_dir / instance_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "submission_annotation.json").write_text("{}")
    from bird_interact_agents.eval import grade_in_place
    monkeypatch.setattr(grade_in_place, "grade_and_write", _fake_grade, raising=True)
    monkeypatch.setattr(ray_app, "grade_and_write", _fake_grade, raising=True)

    # Record the order of write_row vs write_submission_annotation.
    order: list[str] = []
    monkeypatch.setattr(
        ray_app._gcs, "write_row",
        lambda *a, **kw: order.append("write_row"),
    )
    monkeypatch.setattr(
        ray_app._gcs, "write_submission_annotation",
        lambda *a, **kw: order.append("write_submission_annotation"),
    )
    monkeypatch.setattr(
        ray_app._gcs, "write_log", lambda *a, **kw: None,
    )
    monkeypatch.setattr(upload_back, "upload_per_task_debug", lambda **kw: None)
    monkeypatch.setattr(
        upload_back, "upload_per_task_setup_sessions", lambda **kw: None,
    )
    monkeypatch.setattr(
        upload_back, "upload_otf_reference_delta", lambda **kw: None,
    )

    actor = ray_app._LocalActor(
        {"framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
         "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
         "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
         "patience": 3, "strict": False, "use_audited_gold_sql": False,
         "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
         "slayer_storage_root": "/data/slayer_models",
         "data_dir": "/data/mini-interact"},
        RUN_ID, 1, gcs_client=client,
    )
    actor.run_one({"instance_id": "db_a_1", "selected_database": "db_a"})

    # Both must have fired.
    assert "write_submission_annotation" in order, (
        f"annotation upload missing; calls={order}"
    )
    assert "write_row" in order, f"row upload missing; calls={order}"
    ann_idx = order.index("write_submission_annotation")
    row_idx = order.index("write_row")
    assert ann_idx < row_idx, (
        f"annotation MUST land before row (so wait_until_done can rely "
        f"on the row as 'fully done' marker); got calls={order}"
    )
