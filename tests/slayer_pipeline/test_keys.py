"""DEV-1648: DB-wide PK + join-key collection for the refinement guard.

``collect_key_columns`` returns a set of ``(table_lower, col_lower)``
tuples — TABLE-SCOPED, never a bare name set, so a common name like
``id`` in one table does not guard an unrelated table's ``id``. A guarded
column keeps its description but is never retyped / rewritten to a derived
cast, so it stays base/TEXT and matches live introspection (no drift).
"""

from __future__ import annotations

from slayer.core.models import Column, DataType, ModelJoin, SlayerModel

from bird_interact_agents.slayer_pipeline.overlay import collect_key_columns


def _col(name: str, *, pk: bool = False, type_: DataType = DataType.TEXT) -> Column:
    return Column(name=name, sql=name, type=type_, primary_key=pk)


def test_collects_primary_keys() -> None:
    m = SlayerModel(
        name="transplant_matching",
        sql_table="transplant_matching",
        data_source="db",
        columns=[_col("match_rec_registry", pk=True), _col("created_ts")],
    )
    keys = collect_key_columns([m])
    assert ("transplant_matching", "match_rec_registry") in keys
    assert ("transplant_matching", "created_ts") not in keys


def test_collects_local_and_referenced_join_sides() -> None:
    src = SlayerModel(
        name="transplant_matching",
        sql_table="transplant_matching",
        data_source="db",
        columns=[_col("donor_ref_reg")],
        joins=[
            ModelJoin(
                target_model="demographics",
                join_pairs=[["donor_ref_reg", "donor_registry"]],
            )
        ],
    )
    keys = collect_key_columns([src])
    # local side lives on the source table
    assert ("transplant_matching", "donor_ref_reg") in keys
    # referenced side lives on the target model (even if not a PK there)
    assert ("demographics", "donor_registry") in keys


def test_resolves_dotted_join_pairs() -> None:
    m = SlayerModel(
        name="a",
        sql_table="a",
        data_source="db",
        columns=[_col("x")],
        joins=[
            ModelJoin(
                target_model="b",
                join_pairs=[["a.local_col", "b.remote_col"]],
            )
        ],
    )
    keys = collect_key_columns([m])
    assert ("a", "local_col") in keys
    assert ("b", "remote_col") in keys


def test_uses_sql_table_not_model_name() -> None:
    m = SlayerModel(
        name="ModelName",
        sql_table="physical_table",
        data_source="db",
        columns=[_col("pk", pk=True)],
    )
    keys = collect_key_columns([m])
    assert ("physical_table", "pk") in keys
    assert ("modelname", "pk") not in keys


def test_same_column_name_in_unrelated_tables_is_table_scoped() -> None:
    a = SlayerModel(
        name="a", sql_table="a", data_source="db",
        columns=[_col("id", pk=True), _col("amount", type_=DataType.TEXT)],
    )
    b = SlayerModel(
        name="b", sql_table="b", data_source="db",
        columns=[_col("id", type_=DataType.TEXT), _col("amount", type_=DataType.TEXT)],
    )
    keys = collect_key_columns([a, b])
    assert ("a", "id") in keys      # a.id is the PK
    assert ("b", "id") not in keys  # b.id is a plain column, NOT guarded


def test_referenced_side_resolves_target_model_name_to_sql_table() -> None:
    # A join's target_model is a model NAME; the guard is keyed on the
    # physical table. When name != sql_table, the referenced-side key must
    # still land on the target's sql_table.
    src = SlayerModel(
        name="a", sql_table="a", data_source="db",
        columns=[_col("fk")],
        joins=[ModelJoin(target_model="DemographicsModel",
                         join_pairs=[["fk", "reg"]])],
    )
    target = SlayerModel(
        name="DemographicsModel", sql_table="demographics_tbl",
        data_source="db", columns=[_col("reg")],
    )
    keys = collect_key_columns([src, target])
    assert ("demographics_tbl", "reg") in keys       # resolved to sql_table
    assert ("demographicsmodel", "reg") not in keys  # NOT the model name


def test_case_insensitive() -> None:
    m = SlayerModel(
        name="T", sql_table="MixedCase", data_source="db",
        columns=[_col("PkCol", pk=True)],
    )
    keys = collect_key_columns([m])
    assert ("mixedcase", "pkcol") in keys
