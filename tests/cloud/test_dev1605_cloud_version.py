"""DEV-1605: cloud is version-aware.

Design (one version per run; ``runs/<run_id>/`` already isolates versions
ACROSS runs): the GCS layout stays ``<db>``-keyed, and the ``<version>``
segment lives ONLY in the LOCAL root resolution
(``paths.slayer_models_otf_root(benchmark=..., version=...)``). So:

* the manifest carries ``pre_encoded_version`` (consumer-select) and
  ``encode_version`` (encode-label);
* the submit + resubmit job-arg builders forward ``--pre-encoded-version`` /
  ``--version`` when set;
* the actor's ``_slayer_artifacts_for`` and the driver's ``_slayer_uploads_for``
  resolve the OTF root WITH the run's version;
* ``upload_otf_reference_delta`` resolves its ref-root WITH the encode version.

These are mechanical spy contracts — no behaviour assertions.
"""

from __future__ import annotations

import argparse

import pytest

from bird_interact_agents import paths as _paths


def _spy_otf_root(monkeypatch, base):
    """Spy on slayer_models_otf_root capturing the (benchmark, version) it is
    called with; returns paths rooted under ``base`` (a tmp dir) so nothing
    touches the real main checkout. Returns the captured-calls list."""
    seen: list[tuple[str, object]] = []

    def _models(*, benchmark, version=None):
        seen.append((benchmark, version))
        v = version or "_noversion"
        return base / "slayer_models_otf" / benchmark / v

    monkeypatch.setattr(_paths, "slayer_models_otf_root", _models)
    return seen


# ---------------------------------------------------------------------------
# Manifest carries both version scalars
# ---------------------------------------------------------------------------


def test_build_manifest_carries_version_fields():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(
        framework="claude_sdk_otf", mode="a-interact", query_mode="slayer",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        instance_ids=["alien_1"], patience=3, strict=False,
        use_audited_gold_sql=False, max_depth=3, prompt_cache=True,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        max_runtime_hours=1, slayer_setup="pre-encoded",
        pre_encoded_source="otf", pre_encoded_version="opus-4-7",
        encode_version=None,
    )
    m = driver.build_manifest(args, image_uri="img", run_id="rid")
    assert m["pre_encoded_version"] == "opus-4-7"
    assert "encode_version" in m


def test_build_manifest_version_fields_default_none():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(
        framework="pydantic_ai_otf_encode", mode="a-interact", query_mode="slayer",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        instance_ids=["alien_1"], patience=3, strict=False,
        use_audited_gold_sql=False, max_depth=3, prompt_cache=True,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        max_runtime_hours=1, slayer_setup="on-the-fly",
    )
    m = driver.build_manifest(args, image_uri="img", run_id="rid")
    assert m["pre_encoded_version"] is None
    assert m["encode_version"] is None


# ---------------------------------------------------------------------------
# Submit + resubmit job-arg forwarding
# ---------------------------------------------------------------------------


def test_submit_job_args_forward_version_flags():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(
        framework="claude_sdk_otf", query_mode="slayer", mode="a-interact",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        patience=3, max_depth=3, strict=False, use_audited_gold_sql=False,
        prompt_cache=True, reasoning_effort=None, user_sim_prompt_version=None,
        slayer_setup="pre-encoded", slayer_storage_root="/data/slayer_models",
        pre_encoded_source="otf", pre_encoded_version="glm-5.2",
        encode_version=None, instance_ids=["alien_1"], workers=1, actors_per_worker=1,
    )
    ja = driver._build_job_args(args, "rid", attempt=1)
    assert "--pre-encoded-version" in ja
    assert ja[ja.index("--pre-encoded-version") + 1] == "glm-5.2"


def test_submit_job_args_omit_version_when_unset():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        patience=3, max_depth=3, strict=False, use_audited_gold_sql=False,
        prompt_cache=True, reasoning_effort=None, user_sim_prompt_version=None,
        slayer_setup="on-the-fly", slayer_storage_root="/data/slayer_models",
        pre_encoded_source=None, instance_ids=["alien_1"], workers=1, actors_per_worker=1,
    )
    ja = driver._build_job_args(args, "rid", attempt=1)
    assert "--pre-encoded-version" not in ja


def test_submit_job_args_forward_encode_version():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        patience=3, max_depth=3, strict=False, use_audited_gold_sql=False,
        prompt_cache=True, reasoning_effort=None, user_sim_prompt_version=None,
        slayer_setup="on-the-fly", slayer_storage_root="/data/slayer_models",
        pre_encoded_source=None, encode_version="opus-4-7",
        instance_ids=["alien_1"], workers=1, actors_per_worker=1,
    )
    ja = driver._build_job_args(args, "rid", attempt=1)
    assert "--version" in ja
    assert ja[ja.index("--version") + 1] == "opus-4-7"


def test_build_manifest_preserves_non_none_encode_version():
    from bird_interact_agents.cloud import driver

    args = argparse.Namespace(
        framework="pydantic_ai_otf_encode", mode="a-interact", query_mode="slayer",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        instance_ids=["alien_1"], patience=3, strict=False,
        use_audited_gold_sql=False, max_depth=3, prompt_cache=True,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        max_runtime_hours=1, slayer_setup="on-the-fly",
        encode_version="opus-4-7",
    )
    m = driver.build_manifest(args, image_uri="img", run_id="rid")
    assert m["encode_version"] == "opus-4-7"


def test_resubmit_args_forward_encode_version():
    from bird_interact_agents.cloud import driver

    manifest = {
        "dataset": "mini-interact", "framework": "pydantic_ai_otf_encode",
        "query_mode": "slayer", "mode": "a-interact",
        "agent_model": "anthropic/claude-opus-4-7",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "patience": 3, "max_depth": 3,
        "slayer_setup": "on-the-fly", "encode_version": "opus-4-7",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }
    ja = driver._build_resubmit_args(manifest, run_id="rid", missing=["alien_1"], attempt=2)
    assert "--version" in ja
    assert ja[ja.index("--version") + 1] == "opus-4-7"


def test_resubmit_args_forward_pre_encoded_version():
    from bird_interact_agents.cloud import driver

    manifest = {
        "dataset": "mini-interact", "framework": "claude_sdk_otf",
        "query_mode": "slayer", "mode": "a-interact",
        "agent_model": "anthropic/claude-opus-4-7",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "patience": 3, "max_depth": 3,
        "slayer_setup": "pre-encoded", "pre_encoded_source": "otf",
        "pre_encoded_version": "opus-4-7",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }
    ja = driver._build_resubmit_args(manifest, run_id="rid", missing=["alien_1"], attempt=2)
    assert "--pre-encoded-version" in ja
    assert ja[ja.index("--pre-encoded-version") + 1] == "opus-4-7"


# ---------------------------------------------------------------------------
# Local-root resolution threads the version
# ---------------------------------------------------------------------------


def test_actor_artifacts_pass_pre_encoded_version(tmp_path, monkeypatch):
    from bird_interact_agents.cloud import ray_app

    seen = _spy_otf_root(monkeypatch, tmp_path)
    cfg = {
        "framework": "claude_sdk_otf", "slayer_setup": "pre-encoded",
        "pre_encoded_source": "otf", "pre_encoded_version": "glm-5.2",
        "dataset": "mini-interact",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert ("mini-interact", "glm-5.2") in seen


def test_actor_artifacts_pass_encode_version(tmp_path, monkeypatch):
    from bird_interact_agents.cloud import ray_app

    seen = _spy_otf_root(monkeypatch, tmp_path)
    cfg = {
        "framework": "pydantic_ai_otf_encode", "slayer_setup": "on-the-fly",
        "encode_version": "opus-4-7", "dataset": "mini-interact",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert ("mini-interact", "opus-4-7") in seen


def test_driver_uploads_pass_pre_encoded_version(tmp_path, monkeypatch):
    from bird_interact_agents.cloud import driver

    seen = _spy_otf_root(monkeypatch, tmp_path)
    args = argparse.Namespace(
        framework="claude_sdk_otf", slayer_setup="pre-encoded",
        pre_encoded_source="otf", pre_encoded_version="glm-5.2",
        dataset="mini-interact",
    )
    driver._slayer_uploads_for(args)
    assert ("mini-interact", "glm-5.2") in seen


def test_fetch_merge_target_uses_manifest_encode_version(monkeypatch, tmp_path):
    """The laptop fetch merge target must be resolved WITH the manifest's
    encode_version, so cloud-built shards land in the versioned warm-cache
    dir (not a flat/legacy one)."""
    from pathlib import Path

    from bird_interact_agents.cloud import driver
    from bird_interact_agents.cloud import post_run_merge as _prm
    from tests.cloud.test_driver import _patch_collaborators, RUN_ID

    mocks = _patch_collaborators(monkeypatch)
    fake_results = tmp_path / "results"
    monkeypatch.setattr(driver.paths, "results_root", lambda: fake_results)

    seen_versions: list[object] = []

    def _spy_root(*, benchmark, version=None):
        seen_versions.append(version)
        return tmp_path / "warm" / "slayer_models_otf" / (version or "_nov")

    monkeypatch.setattr(driver.paths, "slayer_models_otf_root", _spy_root)
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID, "instance_ids": ["db_a_1"],
        "dataset": "mini-interact", "encode_version": "opus-4-7",
    }

    def fake_download(run_id, dest, **kw):
        Path(dest).mkdir(parents=True, exist_ok=True)

    mocks["gcs"].concurrent_download_prefix.side_effect = fake_download
    monkeypatch.setattr(
        driver._collation, "collate", lambda run_dir, manifest: {"phase_passes": 1}
    )
    monkeypatch.setattr(
        _prm, "merge_post_run_into_warm_cache",
        lambda **kw: {"merged_dbs": [], "ignored_shards": []},
    )
    driver.fetch(RUN_ID)
    assert "opus-4-7" in seen_versions


def test_upload_otf_reference_delta_resolves_versioned_root(tmp_path, monkeypatch, fake_gcs_bucket):
    from bird_interact_agents.cloud import upload_back

    client, store = fake_gcs_bucket
    seen = _spy_otf_root(monkeypatch, tmp_path)

    # Build a versioned local ref so the spy's returned path actually exists.
    root = tmp_path / "slayer_models_otf" / "mini-interact" / "opus-4-7"
    db_dir = root / "alien"
    db_dir.mkdir(parents=True)
    (db_dir / "_reference_fp.txt").write_text("fp-new")
    (db_dir / "model.yaml").write_text("models: []\n")

    cfg = {
        "query_mode": "slayer", "framework": "pydantic_ai_otf_encode",
        "slayer_setup": "on-the-fly", "dataset": "mini-interact",
        "encode_version": "opus-4-7",
    }
    upload_back.upload_otf_reference_delta(
        run_id="rid", cfg=cfg, shard="s0", uploaded_dbs=set(),
        initial_seed_fp_by_db={}, client=client,
    )
    assert ("mini-interact", "opus-4-7") in seen
    # GCS layout stays <db>-keyed (no version segment) — run_id isolates.
    assert any(k.startswith("runs/rid/post_run/slayer_models_otf/s0/alien/") for k in store)
