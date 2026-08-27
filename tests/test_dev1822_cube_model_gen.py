"""DEV-1822: deterministic Cube model generation from introspected schema +
column-meaning JSON (incl. JSON-leaf dimensions and numeric agg measures).

Covers Codex C4 (identifier sanitize/collision), C5 (JSON-leaf null-safe casts
via reused slayer_pipeline.jsonb), C7 (fingerprint stability + change).
"""

from __future__ import annotations

import pytest
import yaml

from bird_interact_agents.cube_local import model_gen as mg


# --- fixtures ---------------------------------------------------------------

def _tables():
    return [
        mg.TableInfo(
            schema_name="public", table_name="customers",
            columns=[
                mg.ColumnInfo(name="id", pg_type="integer", is_pk=True),
                mg.ColumnInfo(name="Full Name", pg_type="text"),
                mg.ColumnInfo(name="region", pg_type="text"),
                mg.ColumnInfo(name="balance", pg_type="numeric"),
                mg.ColumnInfo(name="created_at", pg_type="timestamp without time zone"),
                mg.ColumnInfo(name="signup_date", pg_type="date"),
                mg.ColumnInfo(name="active", pg_type="boolean"),
                mg.ColumnInfo(name="attrs", pg_type="jsonb"),
            ],
        ),
        mg.TableInfo(
            schema_name="public", table_name="orders",
            columns=[
                mg.ColumnInfo(name="id", pg_type="integer", is_pk=True),
                mg.ColumnInfo(name="customer_id", pg_type="integer",
                              fk=mg.FKRef(table="customers", column="id")),
                mg.ColumnInfo(name="amount", pg_type="numeric"),
            ],
        ),
    ]


def _meanings():
    return {
        "db|customers|balance": "the account balance",
        "db|customers|Full Name": "the customer's full name",
        "db|customers|attrs": {
            "fields_meaning": {
                "score": "INTEGER. the credit score",
                "tier": "TEXT. loyalty tier",
            }
        },
    }


def _by_name(items):
    return {i.name: i for i in items}


def _cube(defs, name):
    return next(c for c in defs if c.name == name)


# --- cube / dimension shape -------------------------------------------------

def test_cube_table_and_dimensions():
    defs = mg.build_cube_defs(_tables(), _meanings())
    cust = _cube(defs, "customers")
    assert cust.sql_table == '"public"."customers"'
    dims = _by_name(cust.dimensions)
    # every non-jsonb column becomes a dimension (mixed-case/space sanitized)
    assert set(dims) >= {"id", "full_name", "region", "balance",
                         "attrs__score", "attrs__tier"}
    assert dims["id"].primary_key is True
    assert dims["id"].type == "number"
    assert dims["full_name"].sql == '"Full Name"'
    assert dims["full_name"].type == "string"
    assert dims["balance"].type == "number"
    assert dims["balance"].description == "the account balance"
    # a jsonb column is NOT a raw dimension (only its documented leaves are);
    # a `::text` blob dim is invalid to Cube and near-useless.
    assert "attrs" not in dims


def test_temporal_and_boolean_column_types():
    dims = _by_name(_cube(mg.build_cube_defs(_tables(), _meanings()), "customers").dimensions)
    assert dims["created_at"].type == "time"     # timestamp → time
    assert dims["signup_date"].type == "time"     # date → time
    assert dims["active"].type == "boolean"


def test_json_leaf_dimensions_use_nullsafe_casts():
    defs = mg.build_cube_defs(_tables(), _meanings())
    dims = _by_name(_cube(defs, "customers").dimensions)
    # numeric leaf → number, null-safe CASE cast (malformed rows → NULL, C5)
    assert dims["attrs__score"].type == "number"
    assert "CASE WHEN" in dims["attrs__score"].sql
    assert "jsonb_extract_path_text" in dims["attrs__score"].sql
    # text leaf → string, bare extract
    assert dims["attrs__tier"].type == "string"
    assert dims["attrs__tier"].sql == "jsonb_extract_path_text(\"attrs\", 'tier')"


def test_leaf_sql_is_the_reused_jsonb_helper_output(monkeypatch):
    """C5: leaf SQL/type MUST come from slayer_pipeline.jsonb, not a divergent
    re-implementation — compare against the helper's own output verbatim."""
    from bird_interact_agents.slayer_pipeline.jsonb import walk_fields_meaning
    fm = _meanings()["db|customers|attrs"]["fields_meaning"]
    ref = {leaf.column_name: leaf for leaf in
           walk_fields_meaning("attrs", fm, backend="postgres", table="customers")}
    dims = _by_name(_cube(mg.build_cube_defs(_tables(), _meanings()), "customers").dimensions)
    assert dims["attrs__score"].sql == ref["attrs__score"].sql
    assert dims["attrs__tier"].sql == ref["attrs__tier"].sql


def test_numeric_agg_measures_for_every_number_dim():
    defs = mg.build_cube_defs(_tables(), _meanings())
    cust = _cube(defs, "customers")
    measures = _by_name(cust.measures)
    assert "count" in measures and measures["count"].type == "count"
    for agg in ("sum", "avg", "min", "max"):
        assert f"balance_{agg}" in measures
        assert measures[f"balance_{agg}"].type == agg
        assert measures[f"balance_{agg}"].sql == '"balance"'
        # json-derived numeric leaf also gets agg measures
        assert f"attrs__score_{agg}" in measures
    # string dims get no agg measures
    assert "region_sum" not in measures
    assert "full_name_min" not in measures


def test_fk_join_generated():
    defs = mg.build_cube_defs(_tables(), _meanings())
    orders = _cube(defs, "orders")
    joins = _by_name(orders.joins)
    assert "customers" in joins
    assert joins["customers"].relationship == "many_to_one"
    assert joins["customers"].sql == '{CUBE}."customer_id" = {customers}."id"'


def test_first_fk_wins_and_self_fk_skipped():
    tables = [
        mg.TableInfo(
            schema_name="public", table_name="node",
            columns=[
                mg.ColumnInfo(name="id", pg_type="integer", is_pk=True),
                mg.ColumnInfo(name="parent_id", pg_type="integer",
                              fk=mg.FKRef(table="node", column="id")),  # self-FK → skip
                mg.ColumnInfo(name="a_id", pg_type="integer",
                              fk=mg.FKRef(table="other", column="id")),
                mg.ColumnInfo(name="a_id2", pg_type="integer",
                              fk=mg.FKRef(table="other", column="id")),  # 2nd FK to same → skip
            ],
        ),
        mg.TableInfo(schema_name="public", table_name="other",
                     columns=[mg.ColumnInfo(name="id", pg_type="integer", is_pk=True)]),
    ]
    defs = mg.build_cube_defs(tables, {})
    node = _cube(defs, "node")
    join_names = [j.name for j in node.joins]
    assert join_names == ["other"]  # exactly one join, first FK, no self-join


# --- sanitize / collisions (C4) --------------------------------------------

@pytest.mark.parametrize(
    "raw,taken,expected",
    [
        ("Full Name", set(), "full_name"),
        ("1st_place", set(), "_1st_place"),      # leading digit
        ("a-b", set(), "a_b"),
        ("select", set(), "select"),             # SQL reserved word: fine as a member (SQL is quoted)
        ("a b", {"a_b"}, "a_b_2"),               # collision → suffixed
        ("a b", {"a_b", "a_b_2"}, "a_b_3"),
    ],
)
def test_sanitize_member_name(raw, taken, expected):
    assert mg.sanitize_member_name(raw, set(taken)) == expected


def test_collision_across_columns():
    tables = [mg.TableInfo(
        schema_name="public", table_name="t",
        columns=[
            mg.ColumnInfo(name="id", pg_type="integer", is_pk=True),
            mg.ColumnInfo(name="a b", pg_type="text"),
            mg.ColumnInfo(name="a-b", pg_type="text"),
        ],
    )]
    dims = _by_name(mg.build_cube_defs(tables, {})[0].dimensions)
    assert "a_b" in dims and "a_b_2" in dims


# --- fingerprint (C7) -------------------------------------------------------

def test_fingerprint_stable():
    assert mg.model_fingerprint(_tables(), _meanings()) == \
        mg.model_fingerprint(_tables(), _meanings())


def test_fingerprint_changes_on_schema_change():
    t2 = _tables()
    t2[0].columns.append(mg.ColumnInfo(name="new_col", pg_type="text"))
    assert mg.model_fingerprint(_tables(), _meanings()) != \
        mg.model_fingerprint(t2, _meanings())


def test_fingerprint_changes_on_meaning_change():
    m2 = _meanings()
    m2["db|customers|balance"] = "changed meaning"
    assert mg.model_fingerprint(_tables(), _meanings()) != \
        mg.model_fingerprint(_tables(), m2)


def test_fingerprint_changes_on_version_bump(monkeypatch):
    fp1 = mg.model_fingerprint(_tables(), _meanings())
    monkeypatch.setattr(mg, "MODEL_GEN_VERSION", mg.MODEL_GEN_VERSION + 1)
    assert mg.model_fingerprint(_tables(), _meanings()) != fp1


# --- YAML rendering ---------------------------------------------------------

def test_descriptions_collapsed_to_single_line():
    """Cube's YAML parser reads multi-line/folded descriptions as null (which
    fails the dimension schema), so descriptions must be single-line."""
    tables = [mg.TableInfo(schema_name="public", table_name="t", columns=[
        mg.ColumnInfo(name="id", pg_type="integer", is_pk=True),
        mg.ColumnInfo(name="notes", pg_type="text"),
    ])]
    meanings = {"db|t|notes": "JSONB. A long note.\nExample:\n  {a: b}.  extra   spaces"}
    defs = mg.build_cube_defs(tables, meanings)
    desc = _by_name(defs[0].dimensions)["notes"].description
    assert "\n" not in desc
    # braces neutralised ({/} trip Cube's template compiler); whitespace collapsed
    assert "{" not in desc and "}" not in desc
    assert desc == "JSONB. A long note. Example: (a: b). extra spaces"
    # round-trips through the rendered YAML with no embedded newline (no wrap)
    doc = yaml.safe_load(mg.render_model_files(defs)["t.yml"])
    rendered = next(d for d in doc["cubes"][0]["dimensions"] if d["name"] == "notes")
    assert rendered["description"] == desc


def test_render_model_files_valid_yaml():
    files = mg.render_model_files(mg.build_cube_defs(_tables(), _meanings()))
    assert "customers.yml" in files and "orders.yml" in files
    doc = yaml.safe_load(files["customers.yml"])
    cube = doc["cubes"][0]
    assert cube["name"] == "customers"
    assert cube["sql_table"] == '"public"."customers"'
    assert any(m["name"] == "balance_sum" for m in cube["measures"])
