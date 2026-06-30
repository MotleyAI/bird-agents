"""DEV-1609: the cloud encode-framework artifact keying must treat
`claude_sdk_otf_encode` IDENTICALLY to the legacy `pydantic_ai_otf_encode`
(both produce `slayer_otf_cache` immutable input + `slayer_models_otf`
mutable merge-back). These tests pin parity at every cloud keying site that
used to hardcode the literal `pydantic_ai_otf_encode`:

* `driver._slayer_uploads_for`     — which artifacts to upload as seeds
* `ray_app._slayer_artifacts_for`  — which artifacts an actor downloads
* `ray_app._snapshot_initial_seed_fps` — encode-run-only seed snapshot
* `gcs.slayer_artifact_name`       — (setup, framework) -> artifact-dir
* `upload_back.upload_otf_reference_delta` — encode-run-only merge-back guard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bird_interact_agents.cloud import driver, gcs, ray_app, upload_back


RUN_ID = "20260627t1000-claudesdkotf-slayer-abcdef"


# ---------------------------------------------------------------------------
# driver._slayer_uploads_for
# ---------------------------------------------------------------------------


def _spy_path_helpers(monkeypatch):
    from bird_interact_agents import paths as _paths

    seen: list[tuple[str, str]] = []

    def _cache(*, benchmark):
        seen.append(("cache", benchmark))
        return Path("/data") / f"otf_cache_{benchmark}"

    def _models(*, benchmark):
        seen.append(("models", benchmark))
        return Path("/data") / f"models_otf_{benchmark}"

    monkeypatch.setattr(_paths, "slayer_otf_cache_root", _cache)
    monkeypatch.setattr(_paths, "slayer_models_otf_root", _models)
    return seen


def test_driver_uploads_for_claude_sdk_encode_match_legacy(monkeypatch):
    """claude_sdk_otf_encode uploads BOTH the cache (required) and the
    models_otf reference (optional seed) — parity with pydantic_ai_otf_encode."""
    seen = _spy_path_helpers(monkeypatch)
    args = argparse.Namespace(
        framework="claude_sdk_otf_encode", slayer_setup="on-the-fly",
        dataset="mini-interact",
    )
    uploads = driver._slayer_uploads_for(args)
    names = {name for _root, name, _req in uploads}
    assert names == {"slayer_otf_cache", "slayer_models_otf"}
    # required flags: cache required (True), reference optional (False)
    req = {name: req for _root, name, req in uploads}
    assert req["slayer_otf_cache"] is True
    assert req["slayer_models_otf"] is False
    assert any(b == "mini-interact" for _, b in seen)


# ---------------------------------------------------------------------------
# ray_app._slayer_artifacts_for
# ---------------------------------------------------------------------------


def test_ray_app_artifacts_for_claude_sdk_encode_match_legacy(monkeypatch):
    seen = _spy_path_helpers(monkeypatch)
    cfg = {
        "framework": "claude_sdk_otf_encode", "slayer_setup": "on-the-fly",
        "dataset": "mini-interact",
    }
    artifacts = ray_app._slayer_artifacts_for(cfg)
    names = {name for name, _root, _req in artifacts}
    assert names == {"slayer_otf_cache", "slayer_models_otf"}
    req = {name: req for name, _root, req in artifacts}
    assert req["slayer_otf_cache"] is True
    assert req["slayer_models_otf"] is False
    assert seen and all(b == "mini-interact" for _, b in seen)


# --- benchmark-root parity: the derived benchmark flows into root selection ---


def test_driver_uploads_for_claude_sdk_encode_livesqlbench_roots(monkeypatch):
    seen = _spy_path_helpers(monkeypatch)
    args = argparse.Namespace(
        framework="claude_sdk_otf_encode", slayer_setup="on-the-fly",
        dataset="livesqlbench-base-lite-sqlite",
    )
    driver._slayer_uploads_for(args)
    assert seen and all(
        b == "livesqlbench-base-lite-sqlite" for _, b in seen
    ), seen


def test_ray_app_artifacts_for_claude_sdk_encode_livesqlbench_roots(monkeypatch):
    seen = _spy_path_helpers(monkeypatch)
    cfg = {
        "framework": "claude_sdk_otf_encode", "slayer_setup": "on-the-fly",
        "dataset": "livesqlbench-base-lite-sqlite",
    }
    ray_app._slayer_artifacts_for(cfg)
    assert seen and all(
        b == "livesqlbench-base-lite-sqlite" for _, b in seen
    ), seen


# --- negatives: a NON-encode framework must NOT get the reference artifact ---


def test_driver_uploads_non_encode_framework_has_no_reference(monkeypatch):
    _spy_path_helpers(monkeypatch)
    args = argparse.Namespace(
        framework="claude_sdk", slayer_setup="on-the-fly",
        dataset="mini-interact",
    )
    uploads = driver._slayer_uploads_for(args)
    names = {name for _root, name, _req in uploads}
    assert "slayer_models_otf" not in names


def test_ray_app_artifacts_non_encode_framework_has_no_reference(monkeypatch):
    _spy_path_helpers(monkeypatch)
    cfg = {
        "framework": "claude_sdk", "slayer_setup": "on-the-fly",
        "dataset": "mini-interact",
    }
    artifacts = ray_app._slayer_artifacts_for(cfg)
    names = {name for name, _root, _req in artifacts}
    assert "slayer_models_otf" not in names


# ---------------------------------------------------------------------------
# gcs.slayer_artifact_name
# ---------------------------------------------------------------------------


def test_gcs_artifact_name_claude_sdk_encode_is_reference():
    assert (
        gcs.slayer_artifact_name("on-the-fly", "claude_sdk_otf_encode")
        == gcs.slayer_artifact_name("on-the-fly", "pydantic_ai_otf_encode")
    )
    assert (
        gcs.slayer_artifact_name("on-the-fly", "claude_sdk_otf_encode")
        == "slayer_models_otf"
    )


# ---------------------------------------------------------------------------
# ray_app._snapshot_initial_seed_fps — encode-run-only
# ---------------------------------------------------------------------------


def _seed_one_ref_fp(store: dict, run_id: str, db: str, fp: str) -> None:
    key = f"runs/{run_id}/slayer_setup/slayer_models_otf/{db}/_reference_fp.txt"
    store[key] = fp.encode()


def test_snapshot_seed_fps_runs_for_claude_sdk_encode(fake_gcs_bucket):
    """The seed snapshot must READ the GCS seed for claude_sdk_otf_encode
    (i.e. NOT early-return like a non-encode framework)."""
    client, store = fake_gcs_bucket
    _seed_one_ref_fp(store, RUN_ID, "alien", "fp-seed-1")
    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "claude_sdk_otf_encode", "dataset": "mini-interact",
    }
    out = ray_app._snapshot_initial_seed_fps(RUN_ID, cfg, client=client)
    assert out == {"alien": "fp-seed-1"}


def test_snapshot_seed_fps_noop_for_non_encode_framework(fake_gcs_bucket):
    client, store = fake_gcs_bucket
    _seed_one_ref_fp(store, RUN_ID, "alien", "fp-seed-1")
    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "claude_sdk", "dataset": "mini-interact",
    }
    out = ray_app._snapshot_initial_seed_fps(RUN_ID, cfg, client=client)
    assert out == {}


# ---------------------------------------------------------------------------
# upload_back.upload_otf_reference_delta — encode-run-only merge-back
# ---------------------------------------------------------------------------


def _make_otf_ref_root(monkeypatch, tmp_path, dbs: dict[str, str]) -> Path:
    from bird_interact_agents import paths as _paths

    root = tmp_path / "slayer_models_otf"
    for db, fp in dbs.items():
        d = root / db
        (d / "models").mkdir(parents=True)
        (d / "models" / "x.yaml").write_text(f"name: {db}\n")
        (d / "memories.yaml").write_text("[]\n")
        (d / "_reference_fp.txt").write_text(fp)
    monkeypatch.setattr(
        _paths, "slayer_models_otf_root", lambda *, benchmark=None: root,
    )
    return root


def test_upload_back_ships_for_claude_sdk_encode(
    monkeypatch, tmp_path, fake_gcs_bucket
):
    """The merge-back delta uploader must ship a cloud-built reference for
    claude_sdk_otf_encode — parity with pydantic_ai_otf_encode."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"alien": "fp-cloud-new"})
    uploaded: set[str] = set()
    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "claude_sdk_otf_encode", "dataset": "mini-interact",
    }
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=cfg, shard="host1-1",
        uploaded_dbs=uploaded, initial_seed_fp_by_db={}, client=client,
    )
    base = f"runs/{RUN_ID}/post_run/slayer_models_otf/host1-1/alien"
    assert store[f"{base}/_reference_fp.txt"] == b"fp-cloud-new"
    assert "alien" in uploaded


def test_upload_back_noop_for_encode_when_not_on_the_fly(
    monkeypatch, tmp_path, fake_gcs_bucket
):
    """The merge-back guard is `on-the-fly`-only: claude_sdk_otf_encode with a
    non-on-the-fly setup must early-return (nothing uploaded)."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"alien": "fp-cloud-new"})
    uploaded: set[str] = set()
    cfg = {
        "query_mode": "slayer", "slayer_setup": "pre-encoded",
        "framework": "claude_sdk_otf_encode", "dataset": "mini-interact",
    }
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=cfg, shard="host1-1",
        uploaded_dbs=uploaded, initial_seed_fp_by_db={}, client=client,
    )
    assert uploaded == set()
    assert not any(k.startswith(f"runs/{RUN_ID}/post_run/") for k in store)
