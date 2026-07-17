"""DEV-1672 Cause 2: the OTF phase-3 JSONB expansion must HIDE the raw JSON
model column once it has been expanded into flat leaf columns.

The raw JSON column adds no query capability — every documented path is already
a flat leaf whose ``sql`` (``JSON_EXTRACT`` / ``jsonb_extract_path_text``)
resolves to the underlying TABLE column, not the model column. Leaving the raw
column visible only tempts the query agent to re-implement a leaf inline
(``jsonb_extract_path_text(content_metrics, 'posting', 'posts_per_day')``),
which drives the submit-verify retry loop.

Fix: set ``hidden=True`` on the raw JSON model column (preferred over deletion)
so it disappears from ``inspect``/``search``/``models_summary`` while remaining
available as the leaf-extraction source and a by-name fallback. Guarded on
"≥1 leaf emitted": a documented JSON column with no expandable leaves stays
visible so its data remains discoverable.

These tests build a real SQLite-backed datasource + ``sql_table`` model
in-process (no ``slayer ingest`` subprocess), mirroring
``test_dev1614_phase3_sampling.py`` — NON-integration.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from slayer.core.models import Column, DataType, DatasourceConfig, SlayerModel
from slayer.engine.query_engine import SlayerQueryEngine
from slayer.inspect.service import InspectService
from slayer.mcp.server import _model_to_summary
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_pipeline.jsonb import hide_expanded_jsonb_columns
from bird_interact_agents.slayer_pipeline.orchestrator import _phase3_jsonb

DB = "testdb"
TABLE = "households"
JSON_COL = "socioeconomic"

LEAF_INCOME = "socioeconomic__Income_Bracket"
LEAF_TENURE = "socioeconomic__Tenure_Type"
LEAF_AUTO = "socioeconomic__vehicle_counts__Auto_Count"

_ROWS = [
    ("Low Income", "OWNED", 2),
    ("Low Income", "OWNED", 1),
    ("Low Income", "RENTED", 0),
    ("High Income", "OWNED", 3),
    ("High Income", "OWNED", 2),
    ("Middle Income", "RENTED", 1),
]

_FIELDS_MEANING = {
    "Income_Bracket": "Income classification level. Ex. Low Income, High Income",
    "Tenure_Type": "TEXT. Household tenure status. Ex. OWNED, RENTED",
    "vehicle_counts": {
        "Auto_Count": "INTEGER. Number of passenger vehicles owned. ex.0",
    },
}


def _build_sqlite(sqlite_path: Path) -> None:
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, {JSON_COL} TEXT)")
    for i, (income, tenure, autos) in enumerate(_ROWS, start=1):
        blob = json.dumps(
            {
                "Income_Bracket": income,
                "Tenure_Type": tenure,
                "vehicle_counts": {"Auto_Count": autos},
            }
        )
        conn.execute(
            f"INSERT INTO {TABLE} (id, {JSON_COL}) VALUES (?, ?)", (i, blob)
        )
    conn.commit()
    conn.close()


def _write_meanings(meanings_path: Path, fields_meaning: dict) -> None:
    meanings_path.write_text(
        json.dumps(
            {
                f"{DB}|{TABLE}|{JSON_COL}": {
                    "column_meaning": "JSONB column. Socioeconomic attributes.",
                    "fields_meaning": fields_meaning,
                }
            }
        ),
        encoding="utf-8",
    )


async def _save_base_model(storage: YAMLStorage) -> None:
    await storage.save_model(
        SlayerModel(
            name=TABLE,
            sql_table=TABLE,
            data_source=DB,
            columns=[
                Column(name="id", type=DataType.INT, primary_key=True),
                Column(name=JSON_COL, type=DataType.TEXT),
            ],
        )
    )


@pytest.fixture
async def jsonb_db(tmp_path: Path):
    sqlite_path = tmp_path / f"{DB}.sqlite"
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    storage_dir = tmp_path / "storage"
    _build_sqlite(sqlite_path)
    _write_meanings(meanings_path, _FIELDS_MEANING)

    storage = YAMLStorage(base_dir=str(storage_dir))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(sqlite_path))
    )
    await _save_base_model(storage)
    return SimpleNamespace(
        storage=storage, meanings_path=meanings_path, sqlite_path=sqlite_path
    )


# ---------------------------------------------------------------------------
# hide_expanded_jsonb_columns — pure helper unit tests.
# ---------------------------------------------------------------------------


def _model_with(raw_meta: dict, *, leaf: bool, raw_name: str = JSON_COL) -> SlayerModel:
    cols = [
        Column(name="id", type=DataType.INT, primary_key=True),
        Column(name=raw_name, type=DataType.TEXT, meta=dict(raw_meta)),
    ]
    if leaf:
        cols.append(
            Column(
                name=f"{raw_name}__Income_Bracket",
                sql=f"JSON_EXTRACT({raw_name}, '$.Income_Bracket')",
                type=DataType.TEXT,
                meta={"derived_from": {"json_col": JSON_COL, "path": ["Income_Bracket"]}},
            )
        )
    return SlayerModel(name=TABLE, sql_table=TABLE, data_source=DB, columns=cols)


def test_helper_hides_raw_col_when_leaf_present() -> None:
    model = _model_with({"jsonb": True}, leaf=True)
    assert hide_expanded_jsonb_columns(model) == 1
    by_name = {c.name: c for c in model.columns}
    assert by_name[JSON_COL].hidden is True
    assert by_name[LEAF_INCOME].hidden is False


def test_helper_skips_raw_col_without_leaf() -> None:
    """A JSONB-flagged column with NO derived leaf stays visible (the ≥1-leaf
    guard) so its data remains discoverable."""
    model = _model_with({"jsonb": True}, leaf=False)
    assert hide_expanded_jsonb_columns(model) == 0
    assert {c.name: c for c in model.columns}[JSON_COL].hidden is False


def test_helper_is_idempotent() -> None:
    model = _model_with({"jsonb": True}, leaf=True)
    assert hide_expanded_jsonb_columns(model) == 1
    assert hide_expanded_jsonb_columns(model) == 0  # already hidden → no re-flip


def test_helper_matches_json_col_case_insensitively() -> None:
    """Raw column name and ``derived_from.json_col`` can differ in case
    (meanings lowercases; ingest may not) — the match must be case-folded."""
    model = _model_with({"jsonb": True}, leaf=True, raw_name="Socioeconomic")
    assert hide_expanded_jsonb_columns(model) == 1
    assert {c.name: c for c in model.columns}["Socioeconomic"].hidden is True


def test_helper_ignores_non_jsonb_columns() -> None:
    """A column without ``meta.jsonb`` is never hidden, even if some other
    column happens to name it in derived_from."""
    model = _model_with({}, leaf=True)  # raw col has no jsonb flag
    assert hide_expanded_jsonb_columns(model) == 0
    assert {c.name: c for c in model.columns}[JSON_COL].hidden is False


def test_helper_ignores_hand_written_prefixed_column() -> None:
    """A hand-written column that merely shares the ``<raw>__`` NAME prefix but
    carries NO ``meta.derived_from`` is NOT a generated leaf — it must not
    satisfy the ≥1-leaf guard (guards against a naive ``startswith`` impl)."""
    model = SlayerModel(
        name=TABLE, sql_table=TABLE, data_source=DB,
        columns=[
            Column(name=JSON_COL, type=DataType.TEXT, meta={"jsonb": True}),
            # same prefix, but NOT derived (no meta.derived_from):
            Column(name=f"{JSON_COL}__manual", sql="1", type=DataType.INT),
        ],
    )
    assert hide_expanded_jsonb_columns(model) == 0
    assert {c.name: c for c in model.columns}[JSON_COL].hidden is False


def test_helper_hides_only_json_cols_that_have_their_own_leaf() -> None:
    """Two JSONB cols on one model — only the one with a matching derived leaf
    is hidden (guards against 'hide all jsonb cols if any leaf exists')."""
    model = SlayerModel(
        name=TABLE, sql_table=TABLE, data_source=DB,
        columns=[
            Column(name="a", type=DataType.TEXT, meta={"jsonb": True}),
            Column(
                name="a__x", sql="JSON_EXTRACT(a, '$.x')", type=DataType.TEXT,
                meta={"derived_from": {"json_col": "a", "path": ["x"]}},
            ),
            Column(name="b", type=DataType.TEXT, meta={"jsonb": True}),  # no leaf
        ],
    )
    assert hide_expanded_jsonb_columns(model) == 1
    by_name = {c.name: c for c in model.columns}
    assert by_name["a"].hidden is True
    assert by_name["b"].hidden is False


# ---------------------------------------------------------------------------
# _phase3_jsonb — the encoder wiring.
# ---------------------------------------------------------------------------


async def test_phase3_hides_raw_json_column(jsonb_db) -> None:
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    model = await jsonb_db.storage.get_model(TABLE, data_source=DB)
    by_name = {c.name: c for c in model.columns}
    assert by_name[JSON_COL].hidden is True, "raw JSON col must be hidden"
    assert (by_name[JSON_COL].meta or {}).get("jsonb") is True
    # Leaves stay visible.
    assert by_name[LEAF_INCOME].hidden is False
    assert by_name[LEAF_TENURE].hidden is False
    assert by_name[LEAF_AUTO].hidden is False


async def test_phase3_hidden_persists_and_is_idempotent(jsonb_db) -> None:
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    # Re-run: must not error and must leave the raw col hidden.
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    # Fresh storage → no in-memory carry-over; the flip is on disk.
    fresh = YAMLStorage(base_dir=str(jsonb_db.storage.base_dir))
    model = await fresh.get_model(TABLE, data_source=DB)
    assert {c.name: c for c in model.columns}[JSON_COL].hidden is True


async def test_phase3_leafless_json_col_stays_visible(tmp_path: Path) -> None:
    """A documented JSONB column whose ``fields_meaning`` yields NO leaves must
    NOT be hidden (nothing would replace it)."""
    sqlite_path = tmp_path / f"{DB}.sqlite"
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    _build_sqlite(sqlite_path)
    _write_meanings(meanings_path, {})  # empty fields_meaning → zero leaves

    storage = YAMLStorage(base_dir=str(tmp_path / "storage"))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(sqlite_path))
    )
    await _save_base_model(storage)

    await _phase3_jsonb(
        storage, DB, meanings_path=meanings_path, sqlite_path=sqlite_path,
    )
    model = await storage.get_model(TABLE, data_source=DB)
    raw = {c.name: c for c in model.columns}[JSON_COL]
    assert raw.hidden is False, "leaf-less JSON col must stay discoverable"


async def test_phase3_hides_from_persisted_leaf_when_run_emits_none(tmp_path: Path) -> None:
    """The hide decision reads PERSISTED model state, not the set of leaves
    touched THIS run. A raw col with a pre-existing derived leaf must be hidden
    even when the current ``fields_meaning`` emits zero new leaves (guards
    against a ``leaf_names``-only implementation)."""
    sqlite_path = tmp_path / f"{DB}.sqlite"
    meanings_path = tmp_path / f"{DB}_column_meaning_base.json"
    _build_sqlite(sqlite_path)
    _write_meanings(meanings_path, {})  # zero leaves emitted this run

    storage = YAMLStorage(base_dir=str(tmp_path / "storage"))
    await storage.save_datasource(
        DatasourceConfig(name=DB, type="sqlite", database=str(sqlite_path))
    )
    # Pre-seed the raw col already flagged jsonb + a persisted derived leaf.
    await storage.save_model(
        SlayerModel(
            name=TABLE, sql_table=TABLE, data_source=DB,
            columns=[
                Column(name="id", type=DataType.INT, primary_key=True),
                Column(name=JSON_COL, type=DataType.TEXT, meta={"jsonb": True}),
                Column(
                    name=LEAF_INCOME,
                    sql=f"JSON_EXTRACT({JSON_COL}, '$.Income_Bracket')",
                    type=DataType.TEXT,
                    meta={"derived_from": {"json_col": JSON_COL, "path": ["Income_Bracket"]}},
                ),
            ],
        )
    )

    await _phase3_jsonb(
        storage, DB, meanings_path=meanings_path, sqlite_path=sqlite_path,
    )
    model = await storage.get_model(TABLE, data_source=DB)
    assert {c.name: c for c in model.columns}[JSON_COL].hidden is True


# ---------------------------------------------------------------------------
# Semantics: hidden from DISCOVERY, still resolvable at QUERY time.
# ---------------------------------------------------------------------------


async def test_hidden_raw_col_absent_from_model_summary(jsonb_db) -> None:
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    model = await jsonb_db.storage.get_model(TABLE, data_source=DB)
    summary_cols = {c["name"] for c in _model_to_summary(model)["columns"]}
    assert JSON_COL not in summary_cols, "hidden raw col must not appear in inspect/summary"
    assert LEAF_INCOME in summary_cols, "leaves must still appear"


async def test_hidden_raw_col_absent_from_public_inspect(jsonb_db) -> None:
    """The public ``inspect(entity_type='model')`` output (what the agent sees)
    must omit the hidden raw column as a structured column entry while keeping
    the leaves — asserted on the JSON structure, not brittle markdown."""
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    svc = InspectService(
        storage=jsonb_db.storage,
        engine=SlayerQueryEngine(storage=jsonb_db.storage),
    )
    payload = json.loads(
        await svc.inspect(reference=f"{DB}.{TABLE}", entity_type="model",
                          compact=False, format="json")
    )
    col_names = {c["name"] for c in payload["columns"]}
    assert JSON_COL not in col_names, "hidden raw col must not be a listed column"
    assert LEAF_INCOME in col_names, "leaves must still be listed"


async def test_leaf_aggregate_byte_identical_hidden_vs_visible(jsonb_db) -> None:
    """The KEY safety proof: hiding the raw JSON model column must NOT change a
    leaf aggregate (the leaf SQL resolves to the TABLE column, not the model
    column). Mirrors the issue's delete experiment with hidden instead."""
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    engine = SlayerQueryEngine(storage=jsonb_db.storage)
    query = {"source_model": TABLE, "measures": [f"{LEAF_AUTO}:sum"]}

    # raw col is hidden (post-phase3).
    hidden_resp = await engine.execute(query, data_source=DB)

    # Un-hide the raw col and re-run.
    model = await jsonb_db.storage.get_model(TABLE, data_source=DB)
    for c in model.columns:
        if c.name == JSON_COL:
            c.hidden = False
    await jsonb_db.storage.save_model(model)
    visible_resp = await SlayerQueryEngine(storage=jsonb_db.storage).execute(
        query, data_source=DB
    )

    assert hidden_resp.data == visible_resp.data
    # Non-vacuous: Auto_Count summed over _ROWS is 2+1+0+3+2+1 = 9.
    assert visible_resp.data, "sanity: the aggregate actually returned a row"
    row_vals = [v for v in visible_resp.data[0].values() if v is not None]
    assert 9 in [int(v) for v in row_vals], f"expected sum 9, got {visible_resp.data}"


async def test_hidden_raw_col_still_queryable_by_name(jsonb_db) -> None:
    """The issue prefers hidden over deletion precisely so the raw JSON column
    stays a BY-NAME fallback for any un-expanded path. Referencing it by name in
    a query must still succeed (hiding gates DISCOVERY, not query resolution)."""
    await _phase3_jsonb(
        jsonb_db.storage, DB,
        meanings_path=jsonb_db.meanings_path, sqlite_path=jsonb_db.sqlite_path,
    )
    engine = SlayerQueryEngine(storage=jsonb_db.storage)
    resp = await engine.execute(
        {"source_model": TABLE, "dimensions": [JSON_COL], "measures": ["*:count"]},
        data_source=DB,
    )
    assert resp.data, "hidden raw JSON col must remain queryable by name"
