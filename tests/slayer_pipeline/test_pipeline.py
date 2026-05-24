"""Tests for the deterministic phases of the DB→SLayer-model rework.

Covers:
- ``types.parse_leading_type_token`` — leading-type-token regex and the
  SQL→SLayer DataType mapping.
- ``jsonb.walk_fields_meaning`` + ``expand_one_column`` — full-path
  naming, JSON_EXTRACT sql, meta.derived_from, idempotency, nested
  fields_meaning walking.
- ``overlay`` — column_meaning overlay incl. DEV-1381 date-format
  annotation parsing + SQLite reformat for non-ISO formats.
- ``dates._validate_format_against_samples`` — strptime gate.

LLM-mocked end-to-end is covered in test_dates_llm.py.
"""

from __future__ import annotations

from slayer.core.models import Column, DataType, SlayerModel

from bird_interact_agents.slayer_pipeline.dates import _validate_format_against_samples
from bird_interact_agents.slayer_pipeline.jsonb import (
    expand_one_column,
    jsonb_meaning_entries,
    walk_fields_meaning,
)
from bird_interact_agents.slayer_pipeline.overlay import (
    DATE_ANNOTATION_RE,
    _sqlite_reformat_sql,
    apply_overlay,
)
from bird_interact_agents.slayer_pipeline.types import parse_leading_type_token


# ---------- types ----------


def test_type_token_known_tokens() -> None:
    assert parse_leading_type_token("REAL. fee.")[0] == DataType.DOUBLE
    assert parse_leading_type_token("DECIMAL(12,3). amt.")[0] == DataType.DOUBLE
    assert parse_leading_type_token("DOUBLE PRECISION. x.")[0] == DataType.DOUBLE
    assert parse_leading_type_token("CHAR(10). y.")[0] == DataType.TEXT
    assert parse_leading_type_token("INTEGER. n.")[0] == DataType.INT
    assert parse_leading_type_token("A SERIAL primary key.")[0] == DataType.INT
    assert parse_leading_type_token("BOOLEAN. flag.")[0] == DataType.BOOLEAN
    assert parse_leading_type_token("DATE representing the date.")[0] == DataType.DATE
    assert (
        parse_leading_type_token("TIMESTAMP noting the moment.")[0]
        == DataType.TIMESTAMP
    )


def test_type_token_jsonb_and_enum() -> None:
    dt, meta = parse_leading_type_token("JSONB. nested object.")
    assert dt == DataType.TEXT
    assert meta == {"jsonb": True}

    dt, meta = parse_leading_type_token("RefundMethod_enum. method.")
    assert dt == DataType.TEXT
    assert meta == {"enum_name": "RefundMethod_enum"}


def test_type_token_no_match_returns_none() -> None:
    assert parse_leading_type_token("Total number of bathrooms.") is None
    assert parse_leading_type_token("Foreign key referencing households.") is None
    assert parse_leading_type_token("") is None
    assert parse_leading_type_token(None) is None


def test_type_token_rejects_mixed_case_prose() -> None:
    """Prose like 'Date stored as TEXT in ...' (DEV-1381 annotation
    leader) must NOT infer DATE from the leading 'Date' word — only
    all-uppercase tokens or explicit `_enum` suffixes count."""
    assert parse_leading_type_token(
        "Date stored as TEXT in '%Y-%m-%d'. Cast to TIMESTAMP."
    ) is None
    assert parse_leading_type_token("Time spent on task.") is None
    # All-uppercase DATE / TIMESTAMP still match.
    assert parse_leading_type_token("DATE representing the date.")[0] == DataType.DATE
    assert (
        parse_leading_type_token("TIMESTAMP noting the moment.")[0]
        == DataType.TIMESTAMP
    )


# ---------- jsonb walker ----------


def test_walker_flat() -> None:
    fields = {
        "Bath_Count": "REAL. Total number of bathrooms.",
        "Dwelling_Class": "TEXT. Class label.",
    }
    leaves = list(walk_fields_meaning("dwelling_specs", fields))
    by_name = {leaf.column_name: leaf for leaf in leaves}
    assert "dwelling_specs__Bath_Count" in by_name
    assert "dwelling_specs__Dwelling_Class" in by_name
    bath = by_name["dwelling_specs__Bath_Count"]
    assert bath.sql == "JSON_EXTRACT(dwelling_specs, '$.Bath_Count')"
    assert bath.type == DataType.DOUBLE
    assert bath.label == "Bath Count"
    assert "CAST" not in bath.sql
    assert bath.no_type_token is False


def test_walker_array_index_emits_bracket_path() -> None:
    """Numeric segments must become `[N]` array indexes, not `.N`
    (which would target a JSON object key with that name)."""
    fields = {
        "irradiance_types": {"3": "NUMERIC(7,3) plane-of-array irradiance."}
    }
    leaves = list(walk_fields_meaning("irradiance_conditions", fields))
    assert len(leaves) == 1
    assert (
        leaves[0].sql
        == "JSON_EXTRACT(irradiance_conditions, '$.irradiance_types[3]')"
    )
    assert "." + "3" not in leaves[0].sql


def test_walker_nested_three_levels() -> None:
    fields = {
        "fees": {
            "restocking_fee": "REAL. Restocking fee.",
            "repackaging_cost": "REAL. Repackaging cost.",
        },
        "refund": {
            "refund_amount": "REAL. Refund amount.",
            "method": "RefundMethod_enum. Method used.",
        },
    }
    leaves = list(walk_fields_meaning("cost_breakdown", fields))
    names = {leaf.column_name for leaf in leaves}
    assert "cost_breakdown__fees__restocking_fee" in names
    assert "cost_breakdown__refund__refund_amount" in names
    assert "cost_breakdown__refund__method" in names

    method = next(
        leaf for leaf in leaves if leaf.column_name == "cost_breakdown__refund__method"
    )
    assert method.sql == "JSON_EXTRACT(cost_breakdown, '$.refund.method')"
    assert method.enum_name == "RefundMethod_enum"


def test_walker_records_missing_type_token() -> None:
    fields = {"key": "No type token here."}
    leaves = list(walk_fields_meaning("col", fields))
    assert len(leaves) == 1
    assert leaves[0].no_type_token is True
    assert leaves[0].type == DataType.TEXT


def test_expand_one_column_emits_derived_from() -> None:
    entry = {
        "column_meaning": "JSONB column.",
        "fields_meaning": {"Bath_Count": "REAL. Bathrooms."},
    }
    cols, warnings = expand_one_column("dwelling_specs", entry)
    assert len(cols) == 1
    assert cols[0].meta == {
        "derived_from": {"json_col": "dwelling_specs", "path": ["Bath_Count"]},
    }
    assert warnings == []


def test_expand_one_column_no_token_warns() -> None:
    entry = {
        "column_meaning": "JSONB column.",
        "fields_meaning": {"Bath_Count": "Total number."},
    }
    cols, warnings = expand_one_column("dwelling_specs", entry)
    assert cols[0].type == DataType.TEXT
    assert len(warnings) == 1
    assert "Bath_Count" in warnings[0]


def test_jsonb_meaning_entries_yields_only_dict_entries() -> None:
    meanings = {
        "db|tbl|plain_col": "A plain TEXT.",
        "db|tbl|json_col": {
            "column_meaning": "JSONB.",
            "fields_meaning": {"x": "REAL. x."},
        },
        "db|tbl|dict_without_fields": {"column_meaning": "JSONB but no fields_meaning."},
    }
    entries = list(jsonb_meaning_entries(meanings))
    assert entries == [("tbl", "json_col", meanings["db|tbl|json_col"])]


# ---------- overlay ----------


def _model(cols: list[Column]) -> SlayerModel:
    return SlayerModel(
        name="orders",
        sql_table="orders",
        data_source="demo",
        columns=cols,
    )


def test_overlay_sets_description_and_type_from_leading_token() -> None:
    col = Column(name="amount", sql="amount", type=DataType.TEXT)
    model = _model([col])
    by_table = {"orders": {"amount": "REAL. Order total in USD."}}
    touched, warns = apply_overlay(model, by_table)
    assert touched == 1
    assert col.description == "REAL. Order total in USD."
    assert col.type == DataType.DOUBLE
    assert warns == []


def test_overlay_dev1381_iso_date_annotation_passthrough() -> None:
    col = Column(name="initdate", sql="initdate", type=DataType.TEXT)
    model = _model([col])
    by_table = {
        "orders": {
            "initdate": (
                "Date stored as TEXT in '%Y-%m-%d'. Cast at encode time to TIMESTAMP."
            )
        }
    }
    apply_overlay(model, by_table)
    assert col.type == DataType.TIMESTAMP
    assert col.sql == "initdate"  # passthrough
    assert col.meta["date_source_format"] == "%Y-%m-%d"
    assert col.meta["detected_by"] == "column_meaning_annotation"


def test_overlay_dev1381_dd_mm_yyyy_emits_reformat_sql() -> None:
    col = Column(name="beginday", sql="beginday", type=DataType.TEXT)
    model = _model([col])
    by_table = {
        "orders": {
            "beginday": (
                "Date stored as TEXT in '%d/%m/%Y'. Cast at encode time to TIMESTAMP."
            )
        }
    }
    apply_overlay(model, by_table)
    assert col.type == DataType.TIMESTAMP
    # DD/MM/YYYY → ISO via SUBSTR concat.
    assert "SUBSTR(beginday, 7, 4)" in col.sql
    assert "SUBSTR(beginday, 4, 2)" in col.sql
    assert "SUBSTR(beginday, 1, 2)" in col.sql
    assert col.meta["date_source_format"] == "%d/%m/%Y"


def test_overlay_dev1381_unsupported_format_keeps_text() -> None:
    """Annotation with a format that has no SQLite rewrite path AND
    isn't ISO-passthrough must leave the column TEXT and warn,
    rather than promote to TIMESTAMP with a broken Column.sql."""
    col = Column(name="d", sql="d", type=DataType.TEXT)
    model = _model([col])
    by_table = {
        "orders": {
            "d": (
                "Date stored as TEXT in '%d.%m.%Y'. "
                "Cast at encode time to TIMESTAMP."
            )
        }
    }
    _touched, warns = apply_overlay(model, by_table)
    assert col.type == DataType.TEXT
    assert col.sql == "d"
    assert (col.meta or {}).get("date_source_format") is None
    assert any("unsupported source_format '%d.%m.%Y'" in w for w in warns)


def test_overlay_handles_jsonb_dict_entry() -> None:
    col = Column(name="dwelling_specs", sql="dwelling_specs", type=DataType.TEXT)
    model = _model([col])
    by_table = {
        "orders": {
            "dwelling_specs": {
                "column_meaning": "JSONB column. Combines structural type.",
                "fields_meaning": {"Bath_Count": "REAL. Number of bathrooms."},
            }
        }
    }
    apply_overlay(model, by_table)
    assert col.description.startswith("JSONB column.")
    assert col.type == DataType.TEXT
    assert col.meta == {"jsonb": True}


def test_overlay_idempotent_on_rerun() -> None:
    col = Column(name="amount", sql="amount", type=DataType.TEXT)
    model = _model([col])
    by_table = {"orders": {"amount": "REAL. Order total."}}
    apply_overlay(model, by_table)
    snap = col.model_dump()
    apply_overlay(model, by_table)
    assert col.model_dump() == snap


def test_overlay_skips_unrelated_tables() -> None:
    col = Column(name="amount", sql="amount", type=DataType.TEXT)
    model = _model([col])
    by_table = {"OTHER_TABLE": {"amount": "REAL. Order total."}}
    touched, warns = apply_overlay(model, by_table)
    assert touched == 0
    assert col.type == DataType.TEXT  # untouched


# ---------- DEV-1381 annotation regex ----------


def test_dev1381_regex_matches_both_examples() -> None:
    a = (
        "Date stored as TEXT in '%Y-%m-%d'. Cast at encode time to TIMESTAMP."
    )
    b = (
        "Datetime stored as TEXT in '%Y-%m-%d %H:%M:%S'. "
        "Cast at encode time to TIMESTAMP."
    )
    assert DATE_ANNOTATION_RE.search(a).group(1) == "%Y-%m-%d"
    assert DATE_ANNOTATION_RE.search(b).group(1) == "%Y-%m-%d %H:%M:%S"
    assert DATE_ANNOTATION_RE.search("not a date column.") is None


def test_sqlite_reformat_known_formats() -> None:
    assert _sqlite_reformat_sql("d", "%Y-%m-%d") is None
    assert _sqlite_reformat_sql("d", "%d/%m/%Y").startswith("SUBSTR(d, 7, 4)")
    assert _sqlite_reformat_sql("d", "%Y/%m/%d") == "REPLACE(d, '/', '-')"
    assert _sqlite_reformat_sql("d", "%Q/%v unknown") is None


# ---------- strptime gate ----------


def test_strptime_gate_accepts_consistent_samples() -> None:
    samples = ["2024-01-01", "2024-02-29", "2025-12-31"]
    assert _validate_format_against_samples("%Y-%m-%d", samples) is True


def test_strptime_gate_rejects_mixed_samples() -> None:
    samples = ["2024-01-01", "12/31/2024"]
    assert _validate_format_against_samples("%Y-%m-%d", samples) is False


def test_strptime_gate_empty_returns_false() -> None:
    assert _validate_format_against_samples("%Y-%m-%d", []) is False
