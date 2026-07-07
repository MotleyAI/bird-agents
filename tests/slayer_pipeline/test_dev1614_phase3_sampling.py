"""DEV-1614: ``_phase3_jsonb`` must sample the JSONB leaf columns it adds.

The deterministic OTF ingest expands a JSONB column's ``fields_meaning``
into one SLayer ``Column`` per leaf (``socioeconomic__Income_Bracket``, …)
with a ``JSON_EXTRACT(...)`` ``sql`` and persists them via a bare
``storage.save_model``. SLayer only profiles sample values at end-of-ingest
(before phase 3) or lazily on read — so the leaves were born with
``sampled = sampled_values = None`` and ``inspect``/``search`` showed
description-only, never the real enum.

The fix samples exactly the leaf columns right after they are persisted,
mirroring ``slayer.engine.ingestion``'s end-of-ingest refresh. These tests
build a real SQLite-backed datasource + ``sql_table`` model in-process (no
``slayer ingest`` subprocess — same shape as SLayer's own
``tests/test_engine_profiling.py``), so they are NON-integration.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from slayer.core.models import Column, DataType, DatasourceConfig, SlayerModel
from slayer.engine.query_engine import SlayerQueryEngine
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_pipeline import orchestrator
from bird_interact_agents.slayer_pipeline.orchestrator import _phase3_jsonb

DB = "testdb"
TABLE = "households"
JSON_COL = "socioeconomic"

# Leaf column names emitted by ``expand_one_column`` for the fixture below.
LEAF_INCOME = "socioeconomic__Income_Bracket"
LEAF_TENURE = "socioeconomic__Tenure_Type"
LEAF_AUTO = "socioeconomic__vehicle_counts__Auto_Count"
LEAF_FUEL = "socioeconomic__vehicle_counts__Fuel_Type"

# (Income_Bracket, Tenure_Type, Auto_Count, Fuel_Type) per row.
_ROWS = [
    ("Low Income", "OWNED", 2, "Petrol"),
    ("Low Income", "OWNED", 1, "Petrol"),
    ("Low Income", "RENTED", 0, "Diesel"),
    ("High Income", "OWNED", 3, "Electric"),
    ("High Income", "OWNED", 2, "Petrol"),
    ("Middle Income", "RENTED", 1, "Diesel"),
]

# fields_meaning: Income_Bracket has NO leading type token (defaults TEXT,
# categorical); Tenure_Type/Fuel_Type carry the TEXT token; Auto_Count is
# INTEGER (numeric → min..max, no value list). ``vehicle_counts`` nests two
# leaves so we exercise the >1-segment path that dominates real data.
_FIELDS_MEANING = {
    "Income_Bracket": "Income classification level. Ex. Low Income, High Income",
    "Tenure_Type": "TEXT. Household tenure status. Ex. OWNED, RENTED",
    "vehicle_counts": {
        "Auto_Count": "INTEGER. Number of passenger vehicles owned. ex.0",
        "Fuel_Type": "TEXT. Primary fuel type of the household fleet.",
    },
}


def _build_sqlite(sqlite_path: Path) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(
        f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, {JSON_COL} TEXT)"
    )
    for i, (income, tenure, autos, fuel) in enumerate(_ROWS, start=1):
        blob = json.dumps(
            {
                "Income_Bracket": income,
                "Tenure_Type": tenure,
                "vehicle_counts": {"Auto_Count": autos, "Fuel_Type": fuel},
            }
        )
        conn.execute(
            f"INSERT INTO {TABLE} (id, {JSON_COL}) VALUES (?, ?)", (i, blob)
        )
    conn.commit()
    conn.close()


def _write_meanings(meanings_path: Path) -> None:
    meanings_path.write_text(
        json.dumps(
            {
                f"{DB}|{TABLE}|{JSON_COL}": {
                    "column_meaning": "JSONB column. Socioeconomic attributes.",
                    "fields_meaning": _FIELDS_MEANING,
                }
            }
        ),
        encoding="utf-8",
    )


async def _save_base_model(
    storage: YAMLStorage, *, extra_columns: list[Column] | None = None
) -> None:
    """Persist the datasource + a ``sql_table`` model carrying only the raw
    physical columns (``id`` + the JSONB blob). Phase 3 adds the leaves.
    ``extra_columns`` lets a test pre-seed a hand-written column.
    """
    columns = [
        Column(name="id", type=DataType.INT, primary_key=True),
        Column(name=JSON_COL, type=DataType.TEXT),
    ]
    if extra_columns:
        columns.extend(extra_columns)
    await storage.save_model(
        SlayerModel(
            name=TABLE, sql_table=TABLE, data_source=DB, columns=columns
        )
    )


@pytest.fixture
async def jsonb_db(tmp_path: Path):
    """Real SQLite datasource + model + meanings file. Returns the kwargs
    ``_phase3_jsonb`` needs plus the live ``YAMLStorage``."""
    sqlite_path = tmp_path / f"{DB}.sqlite"
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    storage_dir = tmp_path / "storage"
    _build_sqlite(sqlite_path)
    _write_meanings(meanings_path)

    storage = YAMLStorage(base_dir=str(storage_dir))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(sqlite_path))
    )
    await _save_base_model(storage)

    return SimpleNamespace(
        storage=storage,
        meanings_path=meanings_path,
        sqlite_path=sqlite_path,
    )


# ---------------------------------------------------------------------------
# Headline acceptance: leaves are born sampled.
# ---------------------------------------------------------------------------


async def test_leaf_columns_are_born_sampled(jsonb_db) -> None:
    added, typing_warnings, _drift = await _phase3_jsonb(
        jsonb_db.storage,
        DB,
        meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    assert added == 4  # Income_Bracket, Tenure_Type, Auto_Count, Fuel_Type

    model = await jsonb_db.storage.get_model(TABLE, data_source=DB)
    by_name = {c.name: c for c in model.columns}

    income = by_name[LEAF_INCOME]
    assert income.sampled, "categorical leaf must carry a non-empty sampled string"
    assert income.sampled_values is not None
    assert set(income.sampled_values) == {
        "Low Income",
        "High Income",
        "Middle Income",
    }

    tenure = by_name[LEAF_TENURE]
    assert tenure.sampled_values is not None
    assert set(tenure.sampled_values) == {"OWNED", "RENTED"}


async def test_nested_categorical_leaf_is_sampled(jsonb_db) -> None:
    """The >1-segment path (``vehicle_counts.Fuel_Type``) — the dominant real
    shape — still gets a value list."""
    await _phase3_jsonb(
        jsonb_db.storage,
        DB,
        meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    model = await jsonb_db.storage.get_model(TABLE, data_source=DB)
    fuel = {c.name: c for c in model.columns}[LEAF_FUEL]
    assert fuel.sampled_values is not None
    assert set(fuel.sampled_values) == {"Petrol", "Diesel", "Electric"}


async def test_numeric_leaf_gets_min_max_not_values(jsonb_db) -> None:
    await _phase3_jsonb(
        jsonb_db.storage,
        DB,
        meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    model = await jsonb_db.storage.get_model(TABLE, data_source=DB)
    auto = {c.name: c for c in model.columns}[LEAF_AUTO]
    assert auto.type == DataType.INT
    assert auto.sampled_values is None  # numeric → range form, no value list
    assert auto.sampled is not None
    assert "0" in auto.sampled and "3" in auto.sampled  # min .. max


async def test_sampling_persists_to_disk_not_just_memory(jsonb_db) -> None:
    """Sampling goes through ``storage.update_column_sampled`` (a disk patch);
    a fresh ``get_model`` must observe the values."""
    await _phase3_jsonb(
        jsonb_db.storage,
        DB,
        meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    # Re-instantiate storage from the same base dir → no in-memory carry-over.
    fresh = YAMLStorage(base_dir=str(jsonb_db.storage.base_dir))
    model = await fresh.get_model(TABLE, data_source=DB)
    income = {c.name: c for c in model.columns}[LEAF_INCOME]
    assert income.sampled_values
    assert set(income.sampled_values) == {
        "Low Income",
        "High Income",
        "Middle Income",
    }


# ---------------------------------------------------------------------------
# Error surfacing — best-effort, never aborts (folds into typing_warnings).
# ---------------------------------------------------------------------------


async def test_sample_refresh_error_strings_fold_into_typing_warnings(
    jsonb_db, monkeypatch
) -> None:
    async def fake_refresh(*, model, engine, storage, only_columns):
        return ["households.socioeconomic__Income_Bracket: boom"]

    monkeypatch.setattr(
        orchestrator, "refresh_table_backed_model_sampled", fake_refresh,
        raising=False,
    )
    added, typing_warnings, _drift = await _phase3_jsonb(
        jsonb_db.storage,
        DB,
        meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    assert added == 4  # leaves still added; sampling failure must not abort
    assert any("boom" in w for w in typing_warnings)


async def test_sample_refresh_raise_is_caught(jsonb_db, monkeypatch) -> None:
    async def boom_refresh(*, model, engine, storage, only_columns):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        orchestrator, "refresh_table_backed_model_sampled", boom_refresh,
        raising=False,
    )
    # Must not propagate; the build completes and records a warning.
    added, typing_warnings, _drift = await _phase3_jsonb(
        jsonb_db.storage,
        DB,
        meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    assert added == 4
    assert any("kaboom" in w for w in typing_warnings)


# ---------------------------------------------------------------------------
# Only the genuine leaves are sampled — hand-written / native columns are
# left alone (DEV-1614 Codex review, findings 1 & 3).
# ---------------------------------------------------------------------------


async def test_only_genuine_leaf_columns_passed_to_refresh(
    tmp_path: Path, monkeypatch
) -> None:
    sqlite_path = tmp_path / f"{DB}.sqlite"
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    storage_dir = tmp_path / "storage"
    _build_sqlite(sqlite_path)
    _write_meanings(meanings_path)
    storage = YAMLStorage(base_dir=str(storage_dir))

    # A hand-written column that COLLIDES with a leaf name but has no matching
    # ``meta.derived_from`` — the loop leaves it alone; it must NOT be sampled.
    handwritten = Column(
        name=LEAF_TENURE,
        sql=f"JSON_EXTRACT({JSON_COL}, '$.Tenure_Type')",
        type=DataType.TEXT,
        description="hand-authored; do not touch",
        meta={"hand_written": True},
    )

    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(sqlite_path))
    )
    await _save_base_model(storage, extra_columns=[handwritten])

    captured: dict = {}

    async def capture_refresh(*, model, engine, storage, only_columns):
        captured["only_columns"] = set(only_columns)
        captured["engine_is_query_engine"] = isinstance(engine, SlayerQueryEngine)
        return []

    monkeypatch.setattr(
        orchestrator, "refresh_table_backed_model_sampled", capture_refresh,
        raising=False,
    )
    await _phase3_jsonb(
        storage, DB, meanings_path=meanings_path, sqlite_path=sqlite_path,
    )

    assert captured["only_columns"] == {LEAF_INCOME, LEAF_AUTO, LEAF_FUEL}
    # Explicitly: the collision, the base column, and the pk are excluded.
    assert LEAF_TENURE not in captured["only_columns"]
    assert JSON_COL not in captured["only_columns"]
    assert "id" not in captured["only_columns"]
    # The refresh is handed a real engine bound to this storage.
    assert captured["engine_is_query_engine"] is True


async def test_existing_owned_leaf_is_resampled(jsonb_db) -> None:
    """A leaf already present from a prior run (matching ``meta.derived_from``)
    must be re-sampled on rerun — it is OWNED, so it is included in the
    refresh set and a stale ``sampled`` is replaced with fresh values."""
    storage = jsonb_db.storage
    # Pre-seed the Income_Bracket leaf with the exact derived_from shape
    # ``leaf_to_column`` stamps, plus a stale sentinel sample.
    model = await storage.get_model(TABLE, data_source=DB)
    model.columns.append(
        Column(
            name=LEAF_INCOME,
            sql=f"JSON_EXTRACT({JSON_COL}, '$.Income_Bracket')",
            type=DataType.TEXT,
            description="stale prior-run copy",
            meta={"derived_from": {"json_col": JSON_COL, "path": ["Income_Bracket"]}},
            sampled="STALE",
            sampled_values=["stale-sentinel"],
        )
    )
    await storage.save_model(model)

    added, _typing, _drift = await _phase3_jsonb(
        storage, DB, meanings_path=jsonb_db.meanings_path,
        sqlite_path=jsonb_db.sqlite_path,
    )
    # Income_Bracket already existed → only the other three are *added*.
    assert added == 3

    reloaded = await storage.get_model(TABLE, data_source=DB)
    income = {c.name: c for c in reloaded.columns}[LEAF_INCOME]
    assert income.sampled_values is not None
    assert set(income.sampled_values) == {
        "Low Income",
        "High Income",
        "Middle Income",
    }
    assert "stale-sentinel" not in income.sampled_values


async def test_engine_constructed_once_across_jsonb_columns(
    tmp_path: Path, monkeypatch
) -> None:
    """The ``SlayerQueryEngine`` is built ONCE before the loop and reused for
    every JSONB column in the DB (not per-iteration)."""
    sqlite_path = tmp_path / f"{DB}.sqlite"
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    storage_dir = tmp_path / "storage"

    # Two tables, each with its own JSONB column → the loop iterates twice.
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, {JSON_COL} TEXT)")
    conn.execute('CREATE TABLE properties (id INTEGER PRIMARY KEY, dwelling_specs TEXT)')
    for i, (income, tenure, autos, fuel) in enumerate(_ROWS, start=1):
        conn.execute(
            f"INSERT INTO {TABLE} (id, {JSON_COL}) VALUES (?, ?)",
            (i, json.dumps({"Income_Bracket": income, "Tenure_Type": tenure,
                            "vehicle_counts": {"Auto_Count": autos, "Fuel_Type": fuel}})),
        )
        conn.execute(
            "INSERT INTO properties (id, dwelling_specs) VALUES (?, ?)",
            (i, json.dumps({"Dwelling_Class": "Apartment" if i % 2 else "House"})),
        )
    conn.commit()
    conn.close()

    meanings_path.write_text(json.dumps({
        f"{DB}|{TABLE}|{JSON_COL}": {
            "column_meaning": "JSONB column.", "fields_meaning": _FIELDS_MEANING,
        },
        f"{DB}|properties|dwelling_specs": {
            "column_meaning": "JSONB column.",
            "fields_meaning": {"Dwelling_Class": "TEXT. Dwelling category."},
        },
    }), encoding="utf-8")

    storage = YAMLStorage(base_dir=str(storage_dir))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(sqlite_path))
    )
    await storage.save_model(SlayerModel(
        name=TABLE, sql_table=TABLE, data_source=DB,
        columns=[Column(name="id", type=DataType.INT, primary_key=True),
                 Column(name=JSON_COL, type=DataType.TEXT)],
    ))
    await storage.save_model(SlayerModel(
        name="properties", sql_table="properties", data_source=DB,
        columns=[Column(name="id", type=DataType.INT, primary_key=True),
                 Column(name="dwelling_specs", type=DataType.TEXT)],
    ))

    constructions = {"n": 0}
    real_cls = orchestrator.SlayerQueryEngine

    def counting_engine(*args, **kwargs):
        constructions["n"] += 1
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "SlayerQueryEngine", counting_engine)

    added, _typing, _drift = await _phase3_jsonb(
        storage, DB, meanings_path=meanings_path, sqlite_path=sqlite_path,
    )
    assert added == 5  # 4 from households + 1 from properties
    assert constructions["n"] == 1, "engine must be constructed once, before the loop"


# ---------------------------------------------------------------------------
# Postgres backend (DEV-1648) — JSONB expansion NOW runs for postgres,
# emitting PG-native extracts, and the leaves are still sample-refreshed
# (DEV-1614). Only the SQLite-native detect_drift is skipped.
# ---------------------------------------------------------------------------


async def test_postgres_backend_expands_and_refreshes(tmp_path, monkeypatch) -> None:
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    meanings_path.write_text(json.dumps({
        f"{DB}|{TABLE}|{JSON_COL}": {
            "column_meaning": "JSONB column.",
            "fields_meaning": {"Tenure_Type": "TEXT. Ownership."},
        },
    }), encoding="utf-8")

    storage_dir = tmp_path / "storage"
    storage = YAMLStorage(base_dir=str(storage_dir))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(tmp_path / "x.sqlite"))
    )
    await storage.save_model(SlayerModel(
        name=TABLE, sql_table=TABLE, data_source=DB,
        columns=[Column(name="id", type=DataType.INT, primary_key=True),
                 Column(name=JSON_COL, type=DataType.TEXT)],
    ))

    refresh_calls = {"n": 0}

    async def fake_refresh(*, model, engine, storage, only_columns):
        refresh_calls["n"] += 1
        return []

    monkeypatch.setattr(orchestrator, "refresh_table_backed_model_sampled", fake_refresh)
    monkeypatch.setattr(orchestrator, "SlayerQueryEngine", lambda storage=None: object())

    def fail_drift(*a, **k):  # pragma: no cover
        raise AssertionError("SQLite detect_drift must not run for postgres")

    monkeypatch.setattr(orchestrator, "detect_drift", fail_drift)

    added, typing_warnings, drift = await _phase3_jsonb(
        storage, DB, meanings_path=meanings_path, sqlite_path=None,
        backend="postgres",
    )
    assert added >= 1
    assert refresh_calls["n"] >= 1
    assert drift == []
    model = await storage.get_model(TABLE, data_source=DB)
    leaf = next(c for c in model.columns if c.name == f"{JSON_COL}__Tenure_Type")
    assert "jsonb_extract_path_text" in leaf.sql
    assert "JSON_EXTRACT" not in leaf.sql
