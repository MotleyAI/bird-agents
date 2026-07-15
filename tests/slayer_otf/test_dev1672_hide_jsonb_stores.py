"""DEV-1672 Cause 2 propagation: the migration + apply-time self-heal that put
``hidden=True`` on already-expanded raw JSON columns in stores the build-time
encoder change does NOT rebuild.

- ``hide_store_dir`` — flip hidden on every model in a YAML store dir
  (committed OTF reference stores, extracted archives). Idempotent.
- ``hide_archive`` — unpack a saved ``edited_models.tar.gz``, hide, repack,
  PRESERVING the stamped ``_edited_models_meta.json`` (``cache_fp``) so
  ``apply_or_none`` still accepts the archive.
- Apply-time self-heal — ``materialize_from_saved_store`` hides the raw JSON
  columns of the materialised store before it is queried, so an UN-migrated
  archive still yields hidden columns at query time (Codex High finding: the
  migration must not be the only protection).
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from slayer.core.models import Column, DataType, DatasourceConfig, SlayerModel
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf import edited_models
from bird_interact_agents.slayer_otf.hide_jsonb_stores import (
    hide_archive,
    hide_store_dir,
)

DB = "households"
TABLE = "households"
JSON_COL = "socioeconomic"
LEAF = "socioeconomic__Income_Bracket"
_STORE_META = "_edited_models_meta.json"


async def _make_store(base_dir: Path, *, with_leaf: bool = True) -> None:
    """A minimal YAML store: datasource + one model carrying a visible raw JSON
    column (meta.jsonb) and (optionally) one derived leaf."""
    storage = YAMLStorage(base_dir=str(base_dir))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(base_dir / "x.sqlite"))
    )
    cols = [
        Column(name="id", type=DataType.INT, primary_key=True),
        Column(name=JSON_COL, type=DataType.TEXT, hidden=False, meta={"jsonb": True}),
    ]
    if with_leaf:
        cols.append(
            Column(
                name=LEAF,
                sql=f"JSON_EXTRACT({JSON_COL}, '$.Income_Bracket')",
                type=DataType.TEXT,
                meta={"derived_from": {"json_col": JSON_COL, "path": ["Income_Bracket"]}},
            )
        )
    await storage.save_model(
        SlayerModel(name=TABLE, sql_table=TABLE, data_source=DB, columns=cols)
    )


async def _raw_hidden(base_dir: Path) -> bool:
    model = await YAMLStorage(base_dir=str(base_dir)).get_model(TABLE, data_source=DB)
    return {c.name: c for c in model.columns}[JSON_COL].hidden


# ---------------------------------------------------------------------------
# hide_store_dir
# ---------------------------------------------------------------------------


async def test_hide_store_dir_flips_hidden(tmp_path: Path) -> None:
    store = tmp_path / "store"
    await _make_store(store, with_leaf=True)
    n = await hide_store_dir(store, data_source=DB)
    assert n == 1
    assert await _raw_hidden(store) is True


async def test_hide_store_dir_is_idempotent(tmp_path: Path) -> None:
    store = tmp_path / "store"
    await _make_store(store, with_leaf=True)
    assert await hide_store_dir(store, data_source=DB) == 1
    assert await hide_store_dir(store, data_source=DB) == 0
    assert await _raw_hidden(store) is True


async def test_hide_store_dir_skips_leafless(tmp_path: Path) -> None:
    store = tmp_path / "store"
    await _make_store(store, with_leaf=False)
    assert await hide_store_dir(store, data_source=DB) == 0
    assert await _raw_hidden(store) is False


async def test_hide_store_dir_scopes_to_data_source(tmp_path: Path) -> None:
    """A store can hold multiple datasources; ``data_source=`` must confine the
    hiding to that one and leave the others untouched."""
    store = tmp_path / "store"
    await _make_store(store, with_leaf=True)  # datasource DB
    # A second datasource + model with its own visible raw JSON col + leaf.
    other = "otherdb"
    storage = YAMLStorage(base_dir=str(store))
    await storage.save_datasource(
        DatasourceConfig(name=other, type="sqlite", database=str(store / "y.sqlite"))
    )
    await storage.save_model(
        SlayerModel(
            name="t", sql_table="t", data_source=other,
            columns=[
                Column(name="j", type=DataType.TEXT, hidden=False, meta={"jsonb": True}),
                Column(
                    name="j__leaf", sql="JSON_EXTRACT(j, '$.leaf')", type=DataType.TEXT,
                    meta={"derived_from": {"json_col": "j", "path": ["leaf"]}},
                ),
            ],
        )
    )

    assert await hide_store_dir(store, data_source=DB) == 1
    assert await _raw_hidden(store) is True  # DB.households.socioeconomic hidden
    other_model = await YAMLStorage(base_dir=str(store)).get_model("t", data_source=other)
    assert {c.name: c for c in other_model.columns}["j"].hidden is False


# ---------------------------------------------------------------------------
# hide_archive — preserve cache_fp meta so apply still accepts it.
# ---------------------------------------------------------------------------


def _pack_archive(store_dir: Path, archive: Path, meta: dict) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(store_dir, arcname=DB)
        meta_bytes = json.dumps(meta, sort_keys=True).encode()
        info = tarfile.TarInfo(name=f"{DB}/{_STORE_META}")
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))


def _read_archive_meta(archive: Path) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile(f"{DB}/{_STORE_META}")
        assert member is not None
        return json.loads(member.read())


async def _archive_raw_hidden(archive: Path, tmp: Path) -> bool:
    dest = tmp / "extract"
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    return await _raw_hidden(dest / DB)


@pytest.fixture
async def saved_archive(tmp_path: Path):
    store = tmp_path / "scratch" / DB
    await _make_store(store, with_leaf=True)
    meta = edited_models.store_meta(
        benchmark="mini_interact", db=DB, instance_id=f"{DB}_1",
        deleted_kb_ids=[], cache_fp="CACHEFP123",
    )
    archive = tmp_path / "edited_models.tar.gz"
    _pack_archive(store, archive, meta)
    return archive


async def test_hide_archive_hides_and_preserves_cache_fp(saved_archive, tmp_path) -> None:
    assert await _archive_raw_hidden(saved_archive, tmp_path / "before") is False
    n = await hide_archive(saved_archive)
    assert n == 1
    assert await _archive_raw_hidden(saved_archive, tmp_path / "after") is True
    # cache_fp (and the rest of the meta) survive the repack unchanged.
    assert _read_archive_meta(saved_archive)["cache_fp"] == "CACHEFP123"


async def test_hide_archive_is_idempotent(saved_archive, tmp_path) -> None:
    assert await hide_archive(saved_archive) == 1
    assert await hide_archive(saved_archive) == 0


# ---------------------------------------------------------------------------
# Apply-time self-heal: an UN-migrated archive still yields hidden cols.
# ---------------------------------------------------------------------------


async def test_materialize_self_heals_hidden(saved_archive, tmp_path) -> None:
    """``materialize_from_saved_store`` on a stale (visible) archive must return
    a scratch whose raw JSON column is hidden — the correctness guarantee that
    does not depend on the migration having been run.

    The real ``rewrite_datasource_connection_string`` runs (it no-ops because
    the fixture's datasource has ``connection_string is None``), so this
    exercises the true apply path, not a monkeypatched shortcut."""
    work_dir = tmp_path / "work"
    scratch = await edited_models.materialize_from_saved_store(
        db=DB,
        archive=saved_archive,
        work_dir=work_dir,
        task_deleted_kb_ids=[],
        current_cache_fp="CACHEFP123",
        mini_interact_root=tmp_path,
    )
    assert scratch is not None, "archive must be accepted (cache_fp matches)"
    assert await _raw_hidden(Path(scratch)) is True
    # Ordering: self-heal must run BEFORE the baseline manifest so the hidden
    # flip is part of the baseline — otherwise --save-edited-models would count
    # it as an agent edit and re-save a no-op store.
    assert edited_models.scratch_changed(work_dir, Path(scratch)) is False
