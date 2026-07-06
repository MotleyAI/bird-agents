"""DEV-1648: phase-2 overlay bakes NULL-safe Postgres casts.

Only when the physical (pre-overlay, ingested) type is TEXT; guarded for
PK/join-key columns; SQLite path byte-unchanged.
"""

from __future__ import annotations

from slayer.core.models import Column, DataType, SlayerModel

from bird_interact_agents.slayer_pipeline.casts import pg_nullsafe_cast, quote_ident
from bird_interact_agents.slayer_pipeline.overlay import (
    _sqlite_reformat_sql,
    apply_overlay,
)

TABLE = "transplant_matching"


def _q(name: str) -> str:
    return quote_ident(name)


def _model(columns: list[Column]) -> SlayerModel:
    return SlayerModel(
        name=TABLE, sql_table=TABLE, data_source="db", columns=columns
    )


def _by_table(**cols: str) -> dict:
    return {TABLE: {k.lower(): v for k, v in cols.items()}}


def _iso_ts_sampler(_table: str, _col: str) -> list[str]:
    return ["2025-02-19 08:31:22.330375", "2024-01-01 00:00:00.000001"]


def _dotted_ts_sampler(_table: str, _col: str) -> list[str]:
    return ["2025.02.19 08:31:22", "2025.03.01 12:00:00"]


def _garbage_sampler(_table: str, _col: str) -> list[str]:
    return ["not a date", "whenever", "soon"]


# ---------------------------------------------------------------------------
# Postgres numeric refinement over physically-TEXT columns
# ---------------------------------------------------------------------------


def test_pg_real_over_text_gets_double_cast() -> None:
    col = Column(name="score_val", sql="score_val", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. Overall score. Example: 0.005."),
                  backend="postgres")
    assert col.type == DataType.DOUBLE
    assert col.sql == pg_nullsafe_cast(_q("score_val"), DataType.DOUBLE)


def test_pg_bigint_over_text_gets_bounded_bigint_cast() -> None:
    col = Column(name="dur_sec", sql="dur_sec", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(dur_sec="BIGINT. Duration seconds. Example: 128."),
                  backend="postgres")
    assert col.type == DataType.INT
    assert col.sql == pg_nullsafe_cast(_q("dur_sec"), DataType.INT)


# ---------------------------------------------------------------------------
# Postgres timestamp refinement — format discovered by live sampling
# ---------------------------------------------------------------------------


def test_pg_timestamp_token_iso_sampled() -> None:
    col = Column(name="created_ts", sql="created_ts", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(created_ts="TIMESTAMP. Created. Example: 2025-02-19 08:31:22.330375."),
                  backend="postgres", pg_sampler=_iso_ts_sampler)
    assert col.type == DataType.TIMESTAMP
    # 6-digit fraction -> US.
    assert col.sql == pg_nullsafe_cast(
        _q("created_ts"), DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f", frac_pg="US"
    )
    assert col.meta["date_source_format"] == "%Y-%m-%d %H:%M:%S.%f"
    assert col.meta["detected_by"] == "pg_sample"


def test_pg_timestamp_token_dotted_sampled() -> None:
    col = Column(name="match_ts", sql="match_ts", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(match_ts="TIMESTAMP. Executed. Possible values: 2025.02.19 08:31:22."),
                  backend="postgres", pg_sampler=_dotted_ts_sampler)
    assert col.type == DataType.TIMESTAMP
    # No fraction present -> base format WITHOUT ".%f".
    assert col.sql == pg_nullsafe_cast(
        _q("match_ts"), DataType.TIMESTAMP, "%Y.%m.%d %H:%M:%S"
    )


def test_pg_timestamp_token_millisecond_width_uses_ms() -> None:
    col = Column(name="ms_ts", sql="ms_ts", type=DataType.TEXT)
    m = _model([col])

    def ms_sampler(_t: str, _c: str) -> list[str]:
        return ["2025-02-19 08:31:22.330", "2025-03-01 00:00:00.001"]

    apply_overlay(m, _by_table(ms_ts="TIMESTAMP. Millis."),
                  backend="postgres", pg_sampler=ms_sampler)
    assert col.sql == pg_nullsafe_cast(
        _q("ms_ts"), DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f", frac_pg="MS"
    )


def test_pg_date_token_sampled_uses_to_date() -> None:
    col = Column(name="event_date", sql="event_date", type=DataType.TEXT)
    m = _model([col])

    def date_sampler(_t: str, _c: str) -> list[str]:
        return ["2025-02-19", "2025-03-01"]

    apply_overlay(m, _by_table(event_date="DATE. Event day."),
                  backend="postgres", pg_sampler=date_sampler)
    assert col.type == DataType.DATE
    assert col.sql == pg_nullsafe_cast(_q("event_date"), DataType.DATE, "%Y-%m-%d")


def test_pg_timestamp_token_undetectable_left_text_with_warning() -> None:
    col = Column(name="weird_ts", sql="weird_ts", type=DataType.TEXT)
    m = _model([col])
    _touched, warnings = apply_overlay(
        m, _by_table(weird_ts="TIMESTAMP. Some ts."),
        backend="postgres", pg_sampler=_garbage_sampler,
    )
    # Never silently NULL a whole column: leave it TEXT + bare, warn loudly.
    assert col.type == DataType.TEXT
    assert col.sql == "weird_ts"
    assert any("weird_ts" in w for w in warnings)


def test_pg_dev1381_annotation_uses_explicit_format_no_sampling() -> None:
    col = Column(name="d", sql="d", type=DataType.TEXT)
    m = _model([col])

    def _boom(_t: str, _c: str) -> list[str]:  # must not be called
        raise AssertionError("annotation path must not sample")

    apply_overlay(
        m,
        _by_table(d="Date stored as TEXT in '%d/%m/%Y'. Cast at encode time to TIMESTAMP."),
        backend="postgres", pg_sampler=_boom,
    )
    assert col.type == DataType.TIMESTAMP
    assert col.sql == pg_nullsafe_cast(_q("d"), DataType.TIMESTAMP, "%d/%m/%Y")
    assert col.meta["date_source_format"] == "%d/%m/%Y"
    assert col.meta["detected_by"] == "column_meaning_annotation"


# ---------------------------------------------------------------------------
# Physical-TEXT gate
# ---------------------------------------------------------------------------


def test_pg_already_numeric_column_not_rewritten() -> None:
    # Physical type is already DOUBLE (live PG numeric) -> no cast, stays bare.
    col = Column(name="score_val", sql="score_val", type=DataType.DOUBLE)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. Overall score."),
                  backend="postgres")
    assert col.type == DataType.DOUBLE
    assert col.sql == "score_val"


def test_pg_already_int_column_not_rewritten() -> None:
    col = Column(name="dur_sec", sql="dur_sec", type=DataType.INT)
    m = _model([col])
    apply_overlay(m, _by_table(dur_sec="BIGINT. Duration."), backend="postgres")
    assert col.type == DataType.INT
    assert col.sql == "dur_sec"


def test_pg_boolean_over_text_left_text_with_warning() -> None:
    # A BOOLEAN token over a physically-TEXT column must NOT be retyped to
    # BOOLEAN (that would drift persisted-BOOLEAN vs live-TEXT); left TEXT.
    col = Column(name="is_active", sql="is_active", type=DataType.TEXT)
    m = _model([col])
    _touched, warnings = apply_overlay(
        m, _by_table(is_active="BOOLEAN. Active flag."), backend="postgres",
    )
    assert col.type == DataType.TEXT
    assert col.sql == "is_active"
    assert any("is_active" in w for w in warnings)


def test_pg_annotation_on_already_timestamp_column_not_rewritten() -> None:
    col = Column(name="d", sql="d", type=DataType.TIMESTAMP)
    m = _model([col])
    apply_overlay(
        m,
        _by_table(d="Date stored as TEXT in '%d/%m/%Y'. Cast at encode time to TIMESTAMP."),
        backend="postgres",
    )
    assert col.type == DataType.TIMESTAMP
    assert col.sql == "d"  # already the right physical type -> no cast


# ---------------------------------------------------------------------------
# PK / join-key guard
# ---------------------------------------------------------------------------


def test_pg_primary_key_not_refined_but_described() -> None:
    col = Column(name="match_rec_registry", sql="match_rec_registry",
                 type=DataType.TEXT, primary_key=True)
    m = _model([col])
    apply_overlay(
        m, _by_table(match_rec_registry="BIGINT. PK id. Example: 128."),
        backend="postgres",
        key_columns=frozenset({(TABLE, "match_rec_registry")}),
    )
    assert col.type == DataType.TEXT          # NOT retyped
    assert col.sql == "match_rec_registry"    # NOT rewritten
    assert col.description is not None         # description still applied


def test_pg_join_key_not_refined() -> None:
    col = Column(name="donor_ref_reg", sql="donor_ref_reg", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(
        m, _by_table(donor_ref_reg="BIGINT. FK to demographics."),
        backend="postgres",
        key_columns=frozenset({(TABLE, "donor_ref_reg")}),
    )
    assert col.type == DataType.TEXT
    assert col.sql == "donor_ref_reg"


def test_pg_referenced_side_join_key_not_refined() -> None:
    # The referenced (target-model) side of a join is guarded on ITS OWN
    # model even though it is not a PK there.
    col = Column(name="donor_registry", sql="donor_registry", type=DataType.TEXT)
    demo = SlayerModel(
        name="demographics", sql_table="demographics", data_source="db",
        columns=[col],
    )
    apply_overlay(
        demo,
        {"demographics": {"donor_registry": "BIGINT. Registry number."}},
        backend="postgres",
        key_columns=frozenset({("demographics", "donor_registry")}),
    )
    assert col.type == DataType.TEXT
    assert col.sql == "donor_registry"


# ---------------------------------------------------------------------------
# SQLite path byte-unchanged
# ---------------------------------------------------------------------------


def test_sqlite_numeric_token_leaves_sql_bare() -> None:
    col = Column(name="score_val", sql="score_val", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. Overall score."), backend="sqlite")
    assert col.type == DataType.DOUBLE
    assert col.sql == "score_val"  # bare, unchanged


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
    # Calling apply_overlay with NO backend kwarg must behave as SQLite
    # (back-compat for the mini-interact orchestrator entrypoint).
    col = Column(name="score_val", sql="score_val", type=DataType.TEXT)
    m = _model([col])
    apply_overlay(m, _by_table(score_val="REAL. score."))
    assert col.type == DataType.DOUBLE
    assert col.sql == "score_val"
