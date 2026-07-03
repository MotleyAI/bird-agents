"""DEV-1638: the GCS task-annotation sync moved into the package.

``scripts/fetch_local_annotations.py``'s logic now lives at
``bird_interact_agents.local_annotations.sync_annotations`` so BOTH the local
``bird-interact`` run AND the cloud submit pre-build call the SAME function —
guaranteeing the annotation set the local grader reads equals the set the cloud
image bakes (parity, DEV-1638 decision 6b).

Contract pinned here (mechanical, no network):
* returns ``{"fetched","already_local","missing_in_gcs"}`` counts;
* computes local-missing FIRST and only builds the GCS client when at least
  one target is missing locally (offline / all-local runs never touch GCS);
* a blob absent in GCS is COUNTED, never raised (so the caller's
  require-annotation gate can report precisely-missing ids).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents import local_annotations


_BENCH = "livesqlbench-large"
_ROWS = [
    {"instance_id": "solar_panel_6", "selected_database": "solar_panel"},
    {"instance_id": "fake_account_15", "selected_database": "fake_account"},
]


@pytest.fixture(autouse=True)
def _isolate_annotations_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BIRD_ANNOTATIONS_ROOT", str(tmp_path / "annotations"))
    # _load_dataset_instance_db_map is the lightweight id->db resolver; stub it
    # so no real dataset file is needed.
    monkeypatch.setattr(
        local_annotations, "_load_dataset_instance_db_map",
        lambda benchmark=None: {r["instance_id"]: r["selected_database"]
                                for r in _ROWS},
    )
    return tmp_path


def _dest(iid: str, db: str) -> Path:
    from bird_interact_agents.eval.annotation_io import task_annotation_path
    return task_annotation_path(
        benchmark=_BENCH, selected_database=db, instance_id=iid,
    )


class _Blob:
    def __init__(self, payload: bytes | None):
        self._payload = payload

    def exists(self) -> bool:
        return self._payload is not None

    def download_as_bytes(self) -> bytes:
        assert self._payload is not None
        return self._payload


class _Bucket:
    def __init__(self, blobs: dict[str, bytes | None]):
        self._blobs = blobs

    def blob(self, name: str) -> _Blob:
        return _Blob(self._blobs.get(name))


class _Client:
    def __init__(self, blobs: dict[str, bytes | None]):
        self._blobs = blobs
        self.built = True

    def bucket(self, _name: str) -> _Bucket:
        return _Bucket(self._blobs)


def test_all_local_never_builds_gcs_client(monkeypatch):
    """Every target already on disk ⇒ no GCS client is constructed."""
    for r in _ROWS:
        d = _dest(r["instance_id"], r["selected_database"])
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text("{}")

    def _boom():
        raise AssertionError("GCS client must not be built when all-local")

    monkeypatch.setattr(local_annotations._gcs, "default_gcs_client", _boom)

    result = local_annotations.sync_annotations(_BENCH, None)
    assert result["already_local"] == 2
    assert result["fetched"] == 0
    assert result["missing_in_gcs"] == 0


def test_missing_target_is_downloaded(monkeypatch):
    """A locally-absent annotation present in GCS is written to the right path."""
    from bird_interact_agents.cloud import gcs as real_gcs
    blob_name = real_gcs.stable_task_annotation_blob(
        "livesqlbench-large", "solar_panel", "solar_panel_6",
    )
    payload = b'{"instance_id": "solar_panel_6"}'
    client = _Client({blob_name: payload})
    monkeypatch.setattr(
        local_annotations._gcs, "default_gcs_client", lambda: client,
    )

    result = local_annotations.sync_annotations(_BENCH, ["solar_panel_6"])
    assert result["fetched"] == 1
    assert result["missing_in_gcs"] == 0
    assert _dest("solar_panel_6", "solar_panel").read_bytes() == payload


def test_overwrite_redownloads_already_local(monkeypatch):
    """overwrite=True re-fetches even when a local copy exists (builds client)."""
    d = _dest("solar_panel_6", "solar_panel")
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text("stale")
    from bird_interact_agents.cloud import gcs as real_gcs
    blob_name = real_gcs.stable_task_annotation_blob(
        "livesqlbench-large", "solar_panel", "solar_panel_6",
    )
    client = _Client({blob_name: b"fresh"})
    monkeypatch.setattr(
        local_annotations._gcs, "default_gcs_client", lambda: client,
    )
    result = local_annotations.sync_annotations(
        _BENCH, ["solar_panel_6"], overwrite=True,
    )
    assert result["fetched"] == 1
    assert d.read_bytes() == b"fresh"


def test_missing_in_gcs_counted_not_raised(monkeypatch):
    """A target absent locally AND in GCS is counted, never raised."""
    client = _Client({})  # no blobs exist
    monkeypatch.setattr(
        local_annotations._gcs, "default_gcs_client", lambda: client,
    )
    result = local_annotations.sync_annotations(_BENCH, ["fake_account_15"])
    assert result["missing_in_gcs"] == 1
    assert result["fetched"] == 0
