"""Unit tests for AuditedGoldVariant and AuditedGoldRow (DEV-1518)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bird_interact_agents.eval.annotation_schema import AuditedGoldRow, AuditedGoldVariant


# ---------------------------------------------------------------------------
# AuditedGoldVariant — status normalisation
# ---------------------------------------------------------------------------

def test_variant_clean_normalises_to_original():
    v = AuditedGoldVariant(
        variant_id="v1",
        audit_status="clean",  # type: ignore[arg-type]
        audited_sol_sql=["SELECT 1"],
        primary=True,
    )
    assert v.audit_status == "original"


def test_variant_original_correct_normalises_to_original():
    v = AuditedGoldVariant(
        variant_id="v1",
        audit_status="original_correct",  # type: ignore[arg-type]
        audited_sol_sql=["SELECT 1"],
        primary=True,
    )
    assert v.audit_status == "original"


def test_variant_edited_kept():
    v = AuditedGoldVariant(
        variant_id="v1",
        audit_status="edited",
        audited_sol_sql=["SELECT 2"],
        primary=True,
    )
    assert v.audit_status == "edited"


def test_variant_unrecoverable_rejected():
    with pytest.raises(ValidationError, match="unrecoverable"):
        AuditedGoldVariant(
            variant_id="v1",
            audit_status="unrecoverable",  # type: ignore[arg-type]
            audited_sol_sql=[],
            primary=True,
        )


def test_variant_empty_sql_rejected():
    with pytest.raises(ValidationError, match="audited_sol_sql must contain"):
        AuditedGoldVariant(
            variant_id="v1",
            audit_status="edited",
            audited_sol_sql=[],
            primary=True,
        )


def test_variant_extra_fields_ignored():
    v = AuditedGoldVariant.model_validate({
        "variant_id": "v1",
        "audit_status": "edited",
        "audited_sol_sql": ["SELECT 1"],
        "primary": True,
        "is_gold": True,
        "instance_id": "task_1",
    })
    assert v.variant_id == "v1"
    assert not hasattr(v, "is_gold")


def test_variant_notes_stored():
    v = AuditedGoldVariant(
        variant_id="v1",
        audit_status="edited",
        audited_sol_sql=["SELECT 1"],
        primary=False,
        notes="some note",
    )
    assert v.notes == "some note"


# ---------------------------------------------------------------------------
# AuditedGoldRow — structural invariants
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> AuditedGoldRow:
    defaults = {
        "instance_id": "task_1",
        "selected_database": "mydb",
        "benchmark": "test-bench",
        "variants": [
            {"variant_id": "v1", "audit_status": "edited",
             "audited_sol_sql": ["SELECT 1"], "primary": True},
        ],
    }
    defaults.update(kwargs)
    return AuditedGoldRow.model_validate(defaults)


def test_row_primary_variant_returned():
    row = _make_row()
    assert row.primary_variant.variant_id == "v1"


def test_row_no_variants_rejected():
    with pytest.raises(ValidationError, match="variants must be non-empty"):
        AuditedGoldRow(
            instance_id="task_1",
            selected_database="mydb",
            benchmark="bench",
            variants=[],
        )


def test_row_zero_primary_rejected():
    with pytest.raises(ValidationError, match="exactly one variant must have primary=True"):
        AuditedGoldRow.model_validate({
            "instance_id": "task_1",
            "selected_database": "mydb",
            "benchmark": "bench",
            "variants": [
                {"variant_id": "v1", "audit_status": "edited",
                 "audited_sol_sql": ["SELECT 1"], "primary": False},
            ],
        })


def test_row_two_primaries_rejected():
    with pytest.raises(ValidationError, match="exactly one variant must have primary=True"):
        AuditedGoldRow.model_validate({
            "instance_id": "task_1",
            "selected_database": "mydb",
            "benchmark": "bench",
            "variants": [
                {"variant_id": "v1", "audit_status": "edited",
                 "audited_sol_sql": ["SELECT 1"], "primary": True},
                {"variant_id": "v2", "audit_status": "original",
                 "audited_sol_sql": ["SELECT 2"], "primary": True},
            ],
        })


def test_row_multi_variant_one_primary():
    row = AuditedGoldRow.model_validate({
        "instance_id": "task_1",
        "selected_database": "mydb",
        "benchmark": "bench",
        "variants": [
            {"variant_id": "v1", "audit_status": "original",
             "audited_sol_sql": ["SELECT 1"], "primary": False},
            {"variant_id": "v2", "audit_status": "edited",
             "audited_sol_sql": ["SELECT 2"], "primary": True},
        ],
    })
    assert row.primary_variant.variant_id == "v2"
    assert len(row.variants) == 2


# ---------------------------------------------------------------------------
# AuditedGoldRow.from_flat_rows
# ---------------------------------------------------------------------------

def _flat(variant_id="v1", audit_status="edited", primary=True, **extra) -> dict:
    base = {
        "instance_id": "task_1",
        "selected_database": "mydb",
        "benchmark": "bench",
        "variant_id": variant_id,
        "audit_status": audit_status,
        "audited_sol_sql": ["SELECT 1"],
        "primary": primary,
    }
    base.update(extra)
    return base


def test_from_flat_rows_single():
    row = AuditedGoldRow.from_flat_rows([_flat()])
    assert row.instance_id == "task_1"
    assert len(row.variants) == 1
    assert row.primary_variant.variant_id == "v1"


def test_from_flat_rows_skips_unrecoverable():
    rows = [
        _flat("v1", "unrecoverable", primary=False),
        _flat("v2", "edited", primary=True),
    ]
    row = AuditedGoldRow.from_flat_rows(rows)
    assert len(row.variants) == 1
    assert row.variants[0].variant_id == "v2"


def test_from_flat_rows_all_unrecoverable_raises():
    with pytest.raises(ValueError, match="no valid variants"):
        AuditedGoldRow.from_flat_rows([_flat("v1", "unrecoverable", primary=False)])


def test_from_flat_rows_normalises_clean():
    row = AuditedGoldRow.from_flat_rows([_flat("v1", "clean")])
    assert row.variants[0].audit_status == "original"


def test_from_flat_rows_defaults_first_to_primary():
    rows = [_flat("v1", "edited", primary=False), _flat("v2", "edited", primary=False)]
    row = AuditedGoldRow.from_flat_rows(rows)
    assert row.variants[0].primary is True
    assert row.variants[1].primary is False


def test_from_flat_rows_variant_description_to_notes():
    flat = _flat()
    flat["variant_description"] = "some description"
    row = AuditedGoldRow.from_flat_rows([flat])
    assert row.variants[0].notes == "some description"


def test_from_flat_rows_is_gold_dropped():
    flat = _flat()
    flat["is_gold"] = True
    row = AuditedGoldRow.from_flat_rows([flat])
    assert not hasattr(row.variants[0], "is_gold")
