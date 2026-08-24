"""Verify the exported per-DB SLayer YAML trees load cleanly.

W6 gate: every ``slayer_models/<db>/`` directory must round-trip through
``YAMLStorage`` — datasource present + at least one model that
Pydantic-validates against the current ``SlayerModel`` schema. Catches
any malformed YAML produced by the export pipeline before we point a
benchmark run at it.

This test is parametrized over the 27 mini-interact DBs we ship.
Discovers them at collection time so adding a new DB folder makes the
suite pick it up automatically (and removing one shrinks the gate).
"""

import shutil
from pathlib import Path

import pytest
import yaml

from slayer.storage.yaml_storage import YAMLStorage


SLAYER_MODELS_ROOT = Path(__file__).resolve().parent.parent / "slayer_models"


def _repoint_datasource_to_tmp(work: Path, db_name: str) -> None:
    """Rewrite the copied datasource's sqlite ``connection_string`` to a path
    inside the tmp ``work`` dir.

    The committed ``datasources/<db>.yaml`` bakes an absolute sqlite path from
    whatever machine exported it (e.g. ``/home/.../Dropbox/SLayer/mini-interact/
    <db>/<db>.sqlite``). ``get_datasource`` never connects, but ``get_model``'s
    on-load type refinement (``refine_dict_with_live_schema``) opens the
    datasource to probe the live schema whenever a model has refineable
    INT/DOUBLE columns (24 of the 27 DBs) and hard-fails when the path is
    unreachable. Repointing at a per-run tmp sqlite keeps that connect
    succeeding on any machine (sqlite creates an empty DB; the probe then
    finds no table and leaves persisted types unchanged).
    """
    ds_yaml = work / "datasources" / f"{db_name}.yaml"
    doc = yaml.safe_load(ds_yaml.read_text(encoding="utf-8"))
    if str(doc.get("type", "")).lower() == "sqlite":
        committed = doc.get("connection_string")
        assert isinstance(committed, str) and committed.startswith(
            "sqlite:///"
        ), (
            f"{db_name}: committed connection_string {committed!r} "
            f"is not a sqlite:/// URI"
        )
        doc["connection_string"] = f"sqlite:///{work / f'{db_name}.sqlite'}"
        ds_yaml.write_text(
            yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
        )


def _discover_dbs() -> list[str]:
    """Every direct subdirectory of ``slayer_models/`` other than the
    ``_notes`` markdown folder."""
    return sorted(
        p.name
        for p in SLAYER_MODELS_ROOT.iterdir()
        if p.is_dir() and p.name != "_notes"
    )


DBS = _discover_dbs()


@pytest.mark.parametrize("db_name", DBS)
async def test_db_storage_loads(db_name: str, tmp_path):
    """The per-DB YAML round-trips through ``YAMLStorage`` cleanly.

    Operates on a tmp COPY of the committed dir, not in-place: opening a
    ``YAMLStorage`` on a committed dir lazily creates an empty ``embeddings.db``
    sidecar there, which would pollute the working tree. The copy round-trips
    the identical YAML.
    """
    src = SLAYER_MODELS_ROOT / db_name
    work = tmp_path / db_name
    shutil.copytree(src, work)
    _repoint_datasource_to_tmp(work, db_name)
    storage = YAMLStorage(base_dir=str(work))

    ds = await storage.get_datasource(db_name)
    assert ds is not None, f"{db_name}: datasource '{db_name}.yaml' missing"
    assert ds.name == db_name

    model_names = await storage.list_models()
    assert model_names, f"{db_name}: no models in YAMLStorage"

    for name in model_names:
        model = await storage.get_model(name)
        assert model is not None, f"{db_name}: get_model('{name}') returned None"
        assert model.name == name
        assert model.data_source == db_name, (
            f"{db_name}: model '{name}' has data_source={model.data_source!r}, "
            f"expected {db_name!r}"
        )


def test_all_27_dbs_present():
    """Sanity gate: we expect 27 mini-interact DBs."""
    assert len(DBS) == 27, (
        f"Expected 27 DB folders under slayer_models/, found {len(DBS)}: {DBS}"
    )
