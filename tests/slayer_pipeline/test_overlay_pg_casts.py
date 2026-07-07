"""DEV-1648: phase-2 overlay leaves physically-TEXT top-level columns as TEXT
on Postgres.

A self-referencing cast baked into ``Column.sql`` cycles SLayer's column
expansion (``ColumnCycleError``), and refining ``Column.type`` while ``sql``
stays bare is what trips the mode-C drift check. So on Postgres a
physically-TEXT top-level column is left as TEXT (its semantic type came
from the column-meaning annotation, not the physical DB). JSON leaves ARE
cast — that's tested in ``test_jsonb_pg.py``. The SQLite path is unchanged.
"""

from __future__ import annotations

from slayer.core.models import Column, DataType, SlayerModel

from bird_interact_agents.slayer_pipeline.overlay import (
    _sqlite_reformat_sql,
    apply_overlay,
)

TABLE = "transplant_matching"


def _model(columns: list[Column]) -> SlayerModel:
    return SlayerModel(
        name=TABLE, sql_table=TABLE, data_source="db", columns=columns
    )


def _by_table(**cols: str) -> dict:
    return {TABLE: {k.lower(): v for k, v in cols.items()}}


# ---------------------------------------------------------------------------
# Postgres: physically-TEXT top-level columns are NOT refined (stay TEXT bare)
# ---------------------------------------------------------------------------


def test_pg_real_token_over_text_stays_text() -> None:
    col = Column(name="score_val", sql="score_val", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. Overall score. Example: 0.005."),
                  backend="postgres")
    assert col.type == DataType.TEXT
    assert col.sql == "score_val"
    assert col.description is not None  # description still applied


def test_pg_bigint_token_over_text_stays_text() -> None:
    col = Column(name="dur_sec", sql="dur_sec", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(dur_sec="BIGINT. Duration seconds. Example: 128."),
                  backend="postgres")
    assert col.type == DataType.TEXT
    assert col.sql == "dur_sec"


def test_pg_timestamp_token_over_text_stays_text() -> None:
    col = Column(name="created_ts", sql="created_ts", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(created_ts="TIMESTAMP. Created. Example: 2025-02-19 08:31:22.330375."),
                  backend="postgres")
    assert col.type == DataType.TEXT
    assert col.sql == "created_ts"


def test_pg_date_annotation_over_text_stays_text() -> None:
    col = Column(name="d", sql="d", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(
        m,
        _by_table(d="Date stored as TEXT in '%d/%m/%Y'. Cast at encode time to TIMESTAMP."),
        backend="postgres",
    )
    assert col.type == DataType.TEXT
    assert col.sql == "d"


def test_pg_enum_token_stamps_meta_only() -> None:
    col = Column(name="method", sql="method", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(method="RefundMethod_enum. method."), backend="postgres")
    assert col.type == DataType.TEXT
    assert col.sql == "method"
    assert (col.meta or {}).get("enum_name") == "RefundMethod_enum"


def test_pg_already_typed_column_unchanged() -> None:
    # Physical type already DOUBLE (live PG numeric) -> unchanged.
    col = Column(name="score_val", sql="score_val", type=DataType.DOUBLE)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. Overall score."), backend="postgres")
    assert col.type == DataType.DOUBLE
    assert col.sql == "score_val"


# ---------------------------------------------------------------------------
# PK / join-key guard (still description-only)
# ---------------------------------------------------------------------------


def test_pg_primary_key_described_not_refined() -> None:
    col = Column(name="match_rec_registry", sql="match_rec_registry",
                 type=DataType.TEXT, primary_key=True)
    m = _model([col])
    apply_overlay(
        m, _by_table(match_rec_registry="BIGINT. PK id. Example: 128."),
        backend="postgres",
        key_columns=frozenset({(TABLE, "match_rec_registry")}),
    )
    assert col.type == DataType.TEXT
    assert col.sql == "match_rec_registry"
    assert col.description is not None


# ---------------------------------------------------------------------------
# SQLite path byte-unchanged
# ---------------------------------------------------------------------------


def test_sqlite_numeric_token_sets_type_bare_sql() -> None:
    col = Column(name="score_val", sql="score_val", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. Overall score."), backend="sqlite")
    assert col.type == DataType.DOUBLE
    assert col.sql == "score_val"


def test_sqlite_dev1381_annotation_uses_sqlite_reformat() -> None:
    col = Column(name="d", sql="d", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(
        m,
        _by_table(d="Date stored as TEXT in '%d/%m/%Y'. Cast at encode time to TIMESTAMP."),
        backend="sqlite",
    )
    assert col.type == DataType.TIMESTAMP
    assert col.sql == _sqlite_reformat_sql("d", "%d/%m/%Y")


def test_sqlite_default_backend_matches_prior_behaviour() -> None:
    col = Column(name="score_val", sql="score_val", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. score."))  # no backend kwarg
    assert col.type == DataType.DOUBLE
    assert col.sql == "score_val"
