"""DEV-1907: the committed `slayer_models/` reference validates under SLayer
0.10.2 and round-trips byte-identically (no rewrite on load).

Covers spec `slayer-reference-store`:
  * "Committed reference validates under the pinned SLayer version"
    (every DB loads; the trusted-account extension filter is preserved).
  * "Loading the reference does not rewrite tracked files" (round-trip no-op).

Loading is done against an EMPTY scratch sqlite (datasource repointed) so
on-load type refinement never hard-fails on a machine-specific data path.
Committed models are schema 6; SLayer 0.10.2's current SlayerModel schema is
10, so today every load migrates + writes back — both contracts are RED until
the one-off storage migration lands.

NOTE (plan gap surfaced by these tests): FIVE query-backed models declare a
now-rejected top-level `measures:` block, not the three named in tasks.md 4.2
— crypto/smart_money_accuracy, mental/facility_metrics, robot/program_efficiency,
robot/robot_per_robot — plus fake/trusted_account's ModelExtension.filters.
The impl must fix all five for "every DB loads" to pass.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import shutil
import sqlite3

import pytest

from slayer.storage.yaml_storage import YAMLStorage


def _models_root() -> pathlib.Path:
    from bird_interact_agents import paths

    return paths.slayer_models_root()


def _all_dbs() -> list[str]:
    root = _models_root()
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "models").is_dir()
    )


_ALL_DBS = _all_dbs()

# Query-backed models that currently declare a rejected top-level `measures:`
# block (must be dropped). tasks.md 4.2 lists only the first three.
_MEASURELESS_TARGETS = [
    ("crypto", "smart_money_accuracy"),
    ("mental", "facility_metrics"),
    ("robot", "program_efficiency"),
    ("robot", "robot_per_robot"),
]


def _prepare_db(db: str, tmp_root: pathlib.Path) -> pathlib.Path:
    """Copy a committed DB to tmp and repoint its datasource(s) at an empty
    scratch sqlite so type refinement cannot hard-fail on a missing data file."""
    dst = tmp_root / db
    shutil.copytree(_models_root() / db, dst)
    ds_dir = dst / "datasources"
    if ds_dir.exists():
        scratch = dst / "_scratch.sqlite"
        sqlite3.connect(str(scratch)).close()
        for dsf in ds_dir.glob("*.yaml"):
            txt = dsf.read_text(encoding="utf-8")
            txt = re.sub(r"connection_string:.*", f"connection_string: sqlite:///{scratch}", txt)
            dsf.write_text(txt, encoding="utf-8")
    return dst


def _data_sources(dst: pathlib.Path) -> list[str]:
    return sorted(p.name for p in (dst / "models").iterdir() if p.is_dir())


def _model_yaml_hashes(dst: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted((dst / "models").rglob("*.yaml")):
        out[str(p.relative_to(dst))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


async def _load_all(dst: pathlib.Path) -> dict[str, Exception]:
    """Load every model in every data_source; return {qualified_name: error}."""
    storage = YAMLStorage(base_dir=str(dst))
    errors: dict[str, Exception] = {}
    for ds in _data_sources(dst):
        for name in await storage.list_models(data_source=ds):
            try:
                await storage.get_model(name, data_source=ds)
            except Exception as e:  # noqa: BLE001 — record, don't abort the sweep
                errors[f"{ds}.{name}"] = e
    return errors


@pytest.mark.parametrize("db", _ALL_DBS)
async def test_every_committed_model_validates(db, tmp_path):
    dst = _prepare_db(db, tmp_path)
    errors = await _load_all(dst)
    assert not errors, "models failed to validate under SLayer 0.10.2:\n" + "\n".join(
        f"  {n}: {type(e).__name__}: {str(e).splitlines()[0]}" for n, e in errors.items()
    )


@pytest.mark.parametrize("db", _ALL_DBS)
async def test_loading_does_not_rewrite_committed_yaml(db, tmp_path):
    dst = _prepare_db(db, tmp_path)
    before = _model_yaml_hashes(dst)
    await _load_all(dst)
    after = _model_yaml_hashes(dst)
    assert set(after) == set(before), (
        f"loading added/removed model YAML files: "
        f"{sorted(set(after) ^ set(before))}"
    )
    changed = sorted(k for k in before if before[k] != after.get(k))
    assert not changed, f"loading rewrote committed YAML (schema drift): {changed}"


@pytest.mark.parametrize("db,model", _MEASURELESS_TARGETS)
async def test_query_backed_model_has_no_top_level_measures(db, model, tmp_path):
    dst = _prepare_db(db, tmp_path)
    storage = YAMLStorage(base_dir=str(dst))
    loaded = await storage.get_model(model, data_source=db)
    assert not loaded.measures, (
        f"{db}/{model} still declares top-level measures {[m.name for m in loaded.measures]}"
    )


async def test_trusted_account_window_preserved_without_extension_filters(tmp_path):
    """fake/trusted_account's 180-day recency window survives as an accepted
    entity (a column plus a stage filter), not a rejected ModelExtension.filters."""
    dst = _prepare_db("fake", tmp_path)
    storage = YAMLStorage(base_dir=str(dst))
    loaded = await storage.get_model("trusted_account", data_source="fake")
    dump = loaded.model_dump(mode="json", exclude_none=True)
    source_queries = dump.get("source_queries", [])

    # No source_query source_model may carry an extension `filters` key
    # (that shape is rejected by 0.10.2).
    for sq in source_queries:
        sm = sq.get("source_model")
        if isinstance(sm, dict):
            assert "filters" not in sm, "180-day window still on a rejected ModelExtension.filters"

    # The window must survive as an accepted stage-level filter, not merely a
    # description mention: some source_query carries it in its own `filters`.
    stage_filter_texts = [
        f for sq in source_queries for f in (sq.get("filters") or []) if isinstance(f, str)
    ]
    assert any("180 days" in f for f in stage_filter_texts), (
        f"180-day window not expressed as a stage filter; stage filters were: {stage_filter_texts}"
    )
