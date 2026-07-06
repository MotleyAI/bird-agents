"""DEV-1648: JSONB-leaf expansion emits Postgres-native extracts + casts.

On postgres the leaf ``sql`` becomes ``jsonb_extract_path_text("col",
'seg', …)`` (identifier-quoted base, escaped string-literal segments,
numeric segment as array index), with numeric/date leaves wrapped in the
same NULL-safe cast. Leaves are always derived, so drift never applies.
The SQLite ``JSON_EXTRACT`` path is byte-unchanged.
"""

from __future__ import annotations

from slayer.core.models import DataType

from bird_interact_agents.slayer_pipeline.casts import pg_nullsafe_cast
from bird_interact_agents.slayer_pipeline.jsonb import expand_one_column

JSON_COL = "dwelling_specs"


def _pg_extract(col: str, *segs: str) -> str:
    inner = ", ".join("'" + s.replace("'", "''") + "'" for s in segs)
    return f'jsonb_extract_path_text("{col}", {inner})'


def _by_name(entry: dict, **kw) -> dict:
    cols, warns = expand_one_column(JSON_COL, entry, **kw)
    return {c.name: c for c in cols}, warns


# ---------------------------------------------------------------------------
# Postgres extraction + casting
# ---------------------------------------------------------------------------


def test_pg_numeric_leaf_double_cast() -> None:
    entry = {"fields_meaning": {"Bath_Count": "REAL. Total bathrooms."}}
    cols, _ = _by_name(entry, backend="postgres", table="households")
    leaf = cols["dwelling_specs__Bath_Count"]
    extract = _pg_extract(JSON_COL, "Bath_Count")
    assert leaf.type == DataType.DOUBLE
    assert leaf.sql == pg_nullsafe_cast(extract, DataType.DOUBLE)


def test_pg_integer_leaf_bigint_cast() -> None:
    entry = {"fields_meaning": {"Auto_Count": "INTEGER. Vehicles."}}
    cols, _ = _by_name(entry, backend="postgres", table="households")
    leaf = cols["dwelling_specs__Auto_Count"]
    extract = _pg_extract(JSON_COL, "Auto_Count")
    assert leaf.type == DataType.INT
    assert leaf.sql == pg_nullsafe_cast(extract, DataType.INT)


def test_pg_text_leaf_is_bare_extract() -> None:
    entry = {"fields_meaning": {"Tenure_Type": "TEXT. Ownership."}}
    cols, _ = _by_name(entry, backend="postgres", table="households")
    leaf = cols["dwelling_specs__Tenure_Type"]
    assert leaf.type == DataType.TEXT
    assert leaf.sql == _pg_extract(JSON_COL, "Tenure_Type")


def test_pg_nested_path_segments() -> None:
    entry = {"fields_meaning": {"vehicle_counts": {"Auto_Count": "INTEGER. n."}}}
    cols, _ = _by_name(entry, backend="postgres", table="households")
    leaf = cols["dwelling_specs__vehicle_counts__Auto_Count"]
    extract = _pg_extract(JSON_COL, "vehicle_counts", "Auto_Count")
    assert leaf.sql == pg_nullsafe_cast(extract, DataType.INT)


def test_pg_array_index_segment_is_text_arg() -> None:
    entry = {"fields_meaning": {"irradiance_types": {"3": "NUMERIC(7,3). poa."}}}
    cols, _ = _by_name(entry, backend="postgres", table="cond")
    leaf = cols["dwelling_specs__irradiance_types__3"]
    extract = _pg_extract(JSON_COL, "irradiance_types", "3")
    assert extract.endswith(", '3')")
    assert leaf.sql == pg_nullsafe_cast(extract, DataType.DOUBLE)


def test_pg_segment_single_quote_escaped() -> None:
    entry = {"fields_meaning": {"O'Brien": "TEXT. name."}}
    cols, _ = _by_name(entry, backend="postgres", table="t")
    leaf = cols["dwelling_specs__O'Brien"]
    assert leaf.sql == 'jsonb_extract_path_text("dwelling_specs", \'O\'\'Brien\')'


# ---------------------------------------------------------------------------
# Postgres date leaf — best-effort detection
# ---------------------------------------------------------------------------


def test_pg_date_leaf_detected_via_extract_sampler() -> None:
    entry = {"fields_meaning": {"event_ts": "TIMESTAMP. When it happened."}}
    samples = ["2025-02-19 08:31:22.330375", "2025-03-01 00:00:00"]

    def sampler(_table: str, _extract: str) -> list[str]:
        return samples

    cols, _ = _by_name(entry, backend="postgres", table="t", pg_extract_sampler=sampler)
    leaf = cols["dwelling_specs__event_ts"]
    extract = _pg_extract(JSON_COL, "event_ts")
    assert leaf.type == DataType.TIMESTAMP
    assert leaf.sql == pg_nullsafe_cast(extract, DataType.TIMESTAMP, "%Y-%m-%d %H:%M:%S.%f")


def test_pg_date_leaf_undetectable_stays_bare_extract_with_warning() -> None:
    entry = {"fields_meaning": {"event_ts": "TIMESTAMP. When."}}

    def sampler(_table: str, _extract: str) -> list[str]:
        return ["whenever", "soon"]

    cols, warns = _by_name(entry, backend="postgres", table="t", pg_extract_sampler=sampler)
    leaf = cols["dwelling_specs__event_ts"]
    extract = _pg_extract(JSON_COL, "event_ts")
    # Still derived (drift-safe), just not cast; warned.
    assert leaf.sql == extract
    assert any("event_ts" in w for w in warns)


# ---------------------------------------------------------------------------
# SQLite path byte-unchanged
# ---------------------------------------------------------------------------


def test_sqlite_leaf_unchanged_json_extract() -> None:
    entry = {"fields_meaning": {"Bath_Count": "REAL. Total bathrooms."}}
    cols, _ = _by_name(entry)  # default backend = sqlite
    leaf = cols["dwelling_specs__Bath_Count"]
    assert leaf.sql == "JSON_EXTRACT(dwelling_specs, '$.Bath_Count')"
    assert leaf.type == DataType.DOUBLE


def test_sqlite_array_index_unchanged() -> None:
    entry = {"fields_meaning": {"irradiance_types": {"3": "NUMERIC(7,3). poa."}}}
    cols, _ = _by_name(entry)
    leaf = cols["dwelling_specs__irradiance_types__3"]
    assert leaf.sql == "JSON_EXTRACT(dwelling_specs, '$.irradiance_types[3]')"
