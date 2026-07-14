"""DEV-1671: unit tests for the deterministic edited-model-store KB checker.

The checker (``bird_interact_agents.slayer_otf.store_kb_checker``) is a NUDGE tool
that reports whether a saved edited-model store is in a "passing state":

  1. every surviving agent-added entity is USED by the final (winning) query,
  2. relevant KB items are encoded as entities,
  3. entities refer to each other (clean DAG — KB defs referenced, not inlined),

and flags anything that isn't. It is purely deterministic — it does NOT make the
concept-vs-answer judgment (that is an external analysis step). All lineage is
resolved through SLayer's OWN helpers (``column_dependency`` / ``enrichment`` /
``core.formula``) against real ``SlayerModel`` objects — never a hand-rolled parser.

These are UNIT tests: ``_column_dependencies`` is pure SQLGlot (no postgres), so
fixtures are real ``SlayerModel`` objects and no DB is touched. The dynamic
identical-result check (Component B) needs postgres and lives elsewhere.

Fixture variety (Codex r2 #6): table-backed models, a query-backed / nested-stage
case, and a persisted ``ModelMeasure`` formula dependency.
"""

from __future__ import annotations

import pytest

from slayer.core.models import Column, ModelMeasure, SlayerModel

from bird_interact_agents.slayer_otf.store_kb_checker import (
    Finding,
    StoreCheckReport,
    check_models,
    relevant_kb_closure,
)

# ---------------------------------------------------------------------------
# fixture builders — real SLayer objects
# ---------------------------------------------------------------------------


def col(name, sql, *, kb=None, concept=False, type="DOUBLE", pk=False):
    """A Column, tagging the description with ``[kb=N]`` / ``[concept]`` the way
    the store convention does (provenance lives in the description, not meta)."""
    desc = None
    if kb is not None:
        desc = f"[kb={kb}] {name}"
    elif concept:
        desc = f"[concept] {name}"
    return Column(name=name, sql=sql, type=type, primary_key=pk, description=desc)


def measure(name, formula, *, kb=None):
    desc = f"[kb={kb}] {name}" if kb is not None else None
    return ModelMeasure(name=name, formula=formula, description=desc)


def tbl_model(name, columns, *, measures=None, table="t", joins=None):
    return SlayerModel(
        name=name,
        data_source="db",
        sql_table=table,
        columns=columns,
        measures=measures or [],
        joins=joins or [],
    )


def kb(id_, children):
    return {"id": id_, "knowledge": f"kb{id_}", "definition": f"def{id_}",
            "type": "calculation_knowledge", "children_knowledge": children}


def cats(findings, category):
    return [f for f in findings if f.category == category]


def entities(findings, category):
    return {f.entity for f in cats(findings, category)}


# ---------------------------------------------------------------------------
# baseline: two plain base models the agent starts from
# ---------------------------------------------------------------------------


def base_models():
    return [
        tbl_model("financial_management", [
            col("casetag", "casetag", type="TEXT"),
            col("shipping_fee", "shipping_fee"),
            col("restocking_fee", "restocking_fee"),
        ], table="financial_management"),
        tbl_model("returns", [col("casenum", "casenum", type="TEXT")], table="returns"),
    ]


# ===========================================================================
# relevant-KB closure (pure int graph over children_knowledge)
# ===========================================================================


def test_relevant_kb_closure_walks_children():
    rows = [kb(4, [0]), kb(0, [10, 11]), kb(10, []), kb(11, [])]
    assert relevant_kb_closure(rows, [4]) == {4, 0, 10, 11}


def test_relevant_kb_closure_normalizes_minus_one_sentinel():
    # Codex r2/#7: children_knowledge uses -1 for "no children" in real stores.
    rows = [kb(27, -1), kb(28, [-1]), kb(29, [])]
    assert relevant_kb_closure(rows, [27, 28, 29]) == {27, 28, 29}


def test_relevant_kb_closure_empty_anchor():
    assert relevant_kb_closure([kb(1, [])], []) == set()


# ===========================================================================
# core finding categories
# ===========================================================================


async def test_clean_store_no_findings():
    """All agent-added entities are [kb]/[concept]-tagged AND used → zero findings."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("casetag", "casetag", type="TEXT", pk=True),
            col("shipping_fee", "shipping_fee", kb=10),
            col("restocking_fee", "restocking_fee", kb=11),
            col("trc", "shipping_fee + restocking_fee", kb=0),
            col("sal", "trc + 1", kb=4),
        ], table="financial_management",
           measures=[measure("avg_sal", "round(sal:avg, 2)", kb=4)]),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [0]), kb(0, [10, 11]), kb(10, []), kb(11, [])],
        relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "measures": [{"formula": "avg_sal"}]},
    )
    assert findings == [], [f.model_dump() for f in findings]


async def test_unused_agent_model_flagged_error():
    """A whole agent-added model the winning query never touches → UNUSED (error)."""
    store = base_models() + [
        tbl_model("return_sal", [col("sal", "shipping_fee + 1", kb=4)],
                  table="financial_management"),
        # broken/abandoned sibling — never referenced by the winning query
        tbl_model("return_sal_native", [col("sal2", "shipping_fee + 2", kb=4)],
                  table="financial_management"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal"]},
    )
    unused = cats(findings, "UNUSED_AGENT_ENTITY")
    native = [f for f in unused if f.model == "return_sal_native"]
    assert native, "return_sal_native must be reported with model=return_sal_native"
    # recommendations, not hard errors (the model carries a [kb] column, so it reads as
    # "consider referencing / remove if not useful" rather than a bare delete)
    assert all(f.level == "flag" for f in native)
    assert any("remove" in f.detail or "reference" in f.detail or "delet" in f.detail
               for f in native)


async def test_untagged_derived_used_column_flagged_non_kb():
    """residential case: a used derived column with no [kb]/[concept] tag → NON_KB_ENTITY flag
    (the checker does NOT decide concept-vs-answer; it only flags missing provenance)."""
    store = base_models() + [
        tbl_model("properties", [
            col("propref", "propref", type="INT", pk=True),
            col("roomy_area_per_person", "bath*10 + room*15", type="DOUBLE"),  # untagged, derived
        ], table="properties"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[],
        winning_query={"source_model": "properties",
                       "filters": ["roomy_area_per_person > 20"]},
    )
    assert "roomy_area_per_person" in entities(findings, "NON_KB_ENTITY")
    # and it is NOT flagged unused (it IS used via the filter)
    assert "roomy_area_per_person" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_concept_tagged_column_not_flagged():
    """Same column with a [concept] provenance marker → no NON_KB_ENTITY finding."""
    store = base_models() + [
        tbl_model("properties", [
            col("propref", "propref", type="INT", pk=True),
            col("roomy_area_per_person", "bath*10 + room*15", concept=True),
        ], table="properties"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[],
        winning_query={"source_model": "properties",
                       "filters": ["roomy_area_per_person > 20"]},
    )
    assert cats(findings, "NON_KB_ENTITY") == []


async def test_kb_tagged_aggregation_measure_not_flagged_non_kb():
    """A [kb]-tagged measure that is really the answer is NOT the checker's job to flag
    (that's the external A.7 judgment). The deterministic checker leaves it alone."""
    store = base_models() + [
        tbl_model("return_sal", [col("sal", "shipping_fee + 1", kb=4)],
                  table="financial_management",
                  measures=[measure("avg_sal", "round(sal:avg, 2)", kb=4)]),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "measures": [{"formula": "avg_sal"}]},
    )
    assert cats(findings, "NON_KB_ENTITY") == []


# ===========================================================================
# clean-DAG: inlined KB def (inbound-edge rule) + UNUSED precedence
# ===========================================================================


async def test_inlined_kb_def_inbound_edge_fires():
    """`sal` inlines trc's formula rather than referencing `trc`; trc's KB (0) is a child of
    the used KB (4). → INLINED_KB_DEF (nudge: reference trc), NOT delete."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("shipping_fee", "shipping_fee", kb=10),
            col("restocking_fee", "restocking_fee", kb=11),
            col("trc", "shipping_fee + restocking_fee", kb=0),
            col("sal", "(shipping_fee + restocking_fee) + 1", kb=4),  # INLINES trc
        ], table="financial_management",
           measures=[measure("avg_sal", "round(sal:avg, 2)", kb=4)]),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [0]), kb(0, [10, 11]), kb(10, []), kb(11, [])],
        relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "measures": [{"formula": "avg_sal"}]},
    )
    assert "trc" in entities(findings, "INLINED_KB_DEF")
    # precedence (Codex r2 #3): trc must NOT be reported as unused/delete
    assert "trc" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_inlined_kb_def_clears_when_referenced():
    """After `sal = trc + 1`, trc has an inbound edge → no INLINED_KB_DEF."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("shipping_fee", "shipping_fee", kb=10),
            col("restocking_fee", "restocking_fee", kb=11),
            col("trc", "shipping_fee + restocking_fee", kb=0),
            col("sal", "trc + 1", kb=4),  # REFERENCES trc
        ], table="financial_management",
           measures=[measure("avg_sal", "round(sal:avg, 2)", kb=4)]),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [0]), kb(0, [10, 11]), kb(10, []), kb(11, [])],
        relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "measures": [{"formula": "avg_sal"}]},
    )
    assert cats(findings, "INLINED_KB_DEF") == []


# ===========================================================================
# scaffolding
# ===========================================================================


async def test_scaffolding_referenced_by_used_entity_ok():
    """A trivial-base passthrough referenced (transitively) by a used derived col → no finding."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("shipping_fee", "shipping_fee"),          # trivial base, untagged scaffolding
            col("sal", "shipping_fee + 1", kb=4),          # used; depends on shipping_fee
        ], table="financial_management"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal"]},
    )
    assert "shipping_fee" not in entities(findings, "UNUSED_AGENT_ENTITY")
    assert "shipping_fee" not in entities(findings, "NON_KB_ENTITY")


async def test_scaffolding_referenced_by_nothing_is_unused():
    store = base_models() + [
        tbl_model("return_sal", [
            col("sal", "restocking_fee + 1", kb=4),
            col("orphan_raw", "shipping_fee"),  # trivial base, referenced by nothing
        ], table="financial_management"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal"]},
    )
    assert "orphan_raw" in entities(findings, "UNUSED_AGENT_ENTITY")


# ===========================================================================
# closure resolution: filters, stored measure formula, nested stages
# ===========================================================================


async def test_filter_string_marks_column_used():
    """A column referenced ONLY in a filter expression string counts as USED (Codex r2 #1).
    Proven by contrast: an otherwise-identical unused sibling IS flagged unused, so the pass
    genuinely depends on resolving the filter, not on a blanket exemption (Codex-tests #4)."""
    store = base_models() + [
        tbl_model("properties", [
            col("propref", "propref", type="INT", pk=True),
            col("area_per_person", "bath*10 + room*15", kb=27),   # used only via the filter
            col("unused_sibling", "bath*10 + room*20", kb=27),     # never referenced
        ], table="properties"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(27, [])], relevant_kb_ids=[27],
        winning_query={"source_model": "properties", "filters": ["area_per_person > 20"]},
    )
    unused = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "area_per_person" not in unused
    assert "unused_sibling" in unused


async def test_unused_concept_entity_is_flagged():
    """Concepts are NOT globally exempt from the unused check: an unreferenced [concept] column
    is still UNUSED_AGENT_ENTITY (Codex-tests #4)."""
    store = base_models() + [
        tbl_model("properties", [
            col("propref", "propref", type="INT", pk=True),
            col("used_concept", "bath*10", concept=True),
            col("dead_concept", "room*15", concept=True),   # referenced by nothing
        ], table="properties"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[],
        winning_query={"source_model": "properties", "dimensions": ["used_concept"]},
    )
    unused = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "dead_concept" in unused
    assert "used_concept" not in unused


async def test_unused_query_backed_model_flagged():
    """A query-backed (source_queries) agent model the winning query never uses → UNUSED.
    Mirrors the real broken `return_sal_native` sibling (Codex-tests #5: query-backed fixture)."""
    store = base_models() + [
        tbl_model("return_sal", [col("sal", "shipping_fee + 1", kb=4)],
                  table="financial_management"),
        SlayerModel(
            name="return_sal_native", data_source="db",
            source_queries=[{"source_model": "financial_management",
                             "dimensions": [{"name": "casetag"}]}],
            backing_query_sql="SELECT casetag FROM financial_management",
            columns=[col("casetag", "casetag", type="TEXT")],
            measures=[measure("avg_sal", "sum(casetag)/ *:count", kb=4)],
        ),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal"]},
    )
    unused = {f.model for f in cats(findings, "UNUSED_AGENT_ENTITY")}
    assert "return_sal_native" in unused


async def test_agent_added_column_and_measure_on_existing_baseline_model():
    """Agent-added entities are diffed PER HOST MODEL: a new column + new measure added to an
    EXISTING baseline model are agent-added; an unused one is flagged (Codex-tests #6, #7)."""
    baseline = base_models()
    # store = same models, but financial_management gains two agent columns + one measure
    store = [
        tbl_model("financial_management", [
            col("casetag", "casetag", type="TEXT"),
            col("shipping_fee", "shipping_fee"),
            col("restocking_fee", "restocking_fee"),
            col("used_new", "shipping_fee + 1", kb=4),        # agent-added, used
            col("unused_new", "restocking_fee + 1", kb=4),    # agent-added, unused
        ], table="financial_management",
           measures=[measure("unused_measure", "used_new:sum", kb=4)]),  # agent-added, unused
        tbl_model("returns", [col("casenum", "casenum", type="TEXT")], table="returns"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=baseline,
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "financial_management", "dimensions": ["used_new"]},
    )
    unused = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "unused_new" in unused
    assert "unused_measure" in unused
    assert "used_new" not in unused
    # baseline columns are NOT agent-added → never flagged
    assert "shipping_fee" not in unused and "casetag" not in unused


async def test_stored_measure_formula_marks_column_used():
    """A stored measure's formula references sal → sal counts as used even if no dimension names it."""
    store = base_models() + [
        tbl_model("return_sal", [col("sal", "shipping_fee + 1", kb=4)],
                  table="financial_management",
                  measures=[measure("avg_sal", "round(sal:avg, 2)", kb=4)]),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "measures": [{"formula": "avg_sal"}]},
    )
    assert "sal" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_two_stage_nested_query_closure():
    """fake_account shape. Forces UNION-over-stages + prior-stage-name resolution:
    - `srs` is referenced ONLY in stage 1 (a naive last-stage-only checker would call it unused);
    - stage 2's `source_model` is the stage-1 NAME `s` (must not be treated as a store model);
    - `abandoned_cis` is referenced by NO stage → unused."""
    store = base_models() + [
        tbl_model("risk_and_moderation", [
            col("acct_risk", "acct_risk", type="TEXT"),
            col("srs", "0.4 * risk_value", kb=4),          # stage-1 only
            col("srs_round3", "round(srs, 3)", kb=4),        # stage-1 + stage-2
            col("abandoned_cis", "0.9 * risk_value", kb=4),  # no stage references it
        ], table="risk_and_moderation"),
    ]
    winning_query = [
        {"name": "s", "source_model": "risk_and_moderation",
         "dimensions": ["acct_risk", "srs", "srs_round3"]},
        {"source_model": "s", "dimensions": ["acct_risk", "srs_round3"],
         "limit": 20, "distinct_dimension_values": False},
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4], winning_query=winning_query,
    )
    used_names = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "srs" not in used_names, "stage-1-only reference must count as used (union over stages)"
    assert "srs_round3" not in used_names
    assert "abandoned_cis" in used_names


async def test_cross_model_filter_reference_marks_joined_column_used():
    """A filter that references a column on a JOINED model (qualified `transactions.is_cross_border`
    or a bare name resolvable through a join) counts as USED — not a false UNUSED (regression:
    the closure previously resolved names against the root model only)."""
    root = SlayerModel(
        name="risk_analytics", data_source="db", sql_table="ra",
        columns=[col("txn_link", "txn_link", type="TEXT")],
        joins=[{"target_model": "transactions", "join_type": "left",
                "join_pairs": [["txn_link", "tid"]]}],
    )
    joined = tbl_model("transactions", [
        col("tid", "tid", type="TEXT", pk=True),
        col("is_cross_border", "flag = 'X'", type="BOOLEAN", concept=True),  # agent-added, joined
        col("unused_joined", "flag = 'Y'", type="BOOLEAN", kb=4),            # agent-added, unused
    ], table="transactions")
    store = base_models() + [root, joined]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "risk_analytics", "measures": ["*:count"],
                       "filters": ["transactions.is_cross_border = 1"]},
    )
    unused = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "is_cross_border" not in unused, "joined-model column used in a filter must not be UNUSED"
    assert "unused_joined" in unused  # a genuinely-unreferenced joined column IS unused


async def test_source_queries_backed_bridge_marks_base_defs_used():
    """A source_queries-backed grain-bridge model consumes base KB defs INSIDE its
    source_queries (invisible to Column.sql traversal). When the bridge's output is used by
    the query, those base defs must count as USED — not false-flagged UNUSED (regression:
    mental_healths_10, where the query joins nested-query bridges over base KB columns)."""
    store = base_models() + [
        tbl_model("assess", [
            col("fac_id", "fac_id", type="INT"),
            col("pfis", "0.5 * raw_score", kb=6),        # base KB def, consumed only via the bridge
        ], table="assess"),
        SlayerModel(
            name="pfis_fac", data_source="db",
            source_queries=[{"source_model": "assess",
                             "measures": [{"name": "pfis", "formula": "pfis"}],
                             "dimensions": [{"name": "fac_id", "model": "assess"}]}],
            backing_query_sql="SELECT fac_id, pfis FROM assess",
            columns=[col("fac_id", "fac_id", type="INT"), col("pfis", "pfis")]),
        SlayerModel(
            name="facilities", data_source="db", sql_table="facilities",
            columns=[col("fkey", "fkey", type="TEXT", pk=True),
                     col("rdd", "pfis_fac.pfis - 1", kb=34)],
            joins=[{"target_model": "pfis_fac", "join_type": "left",
                    "join_pairs": [["fkey", "fac_id"]]}]),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(6, []), kb(34, [6])], relevant_kb_ids=[34],
        winning_query={"source_model": "facilities", "dimensions": ["rdd"]})
    unused = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "pfis" not in unused, "base KB def consumed inside a used bridge's source_queries must be USED"


async def test_inline_extension_source_name_resolves_stored_refs():
    """Codex #2: source_model = ModelExtension {source_name, columns}. A stored column the
    query references (via the base model / filters) must count as USED, not false-UNUSED."""
    store = base_models() + [
        tbl_model("sprint_results", [
            col("winner", "pos = 1", type="BOOLEAN", concept=True),   # used via filter
            col("stray", "x + 1", kb=4),                              # unused
        ], table="t"),
    ]
    q = {"source_model": {"source_name": "sprint_results",
                          "columns": [{"name": "age", "sql": "floor(x)"}]},
         "measures": ["floor(age:avg)"], "filters": ["winner = 1"]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4], winning_query=q)
    unused = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "winner" not in unused
    assert "stray" in unused


async def test_time_dimension_reference_marks_used():
    """Codex #3: a column referenced only via time_dimensions counts as used."""
    store = base_models() + [
        tbl_model("m", [col("event_date_norm", "cast(x as date)", type="DATE", concept=True)],
                  table="t"),
    ]
    q = {"source_model": "m", "measures": ["*:count"],
         "time_dimensions": [{"dimension": "event_date_norm", "granularity": "month"}]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[], winning_query=q)
    assert "event_date_norm" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_order_by_agg_suffix_resolves_column():
    """Codex #6: order by 'col:agg' strips the suffix and marks 'col' used."""
    store = base_models() + [
        tbl_model("m", [col("risk_score", "a + 1", kb=4)], table="t"),
    ]
    q = {"source_model": "m", "measures": ["*:count"],
         "order": [{"column": "risk_score:avg", "direction": "desc"}]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4], winning_query=q)
    assert "risk_score" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_declared_join_pairs_list_form_exempts_join_key():
    """Codex #4: declared join with list-form join_pairs [[a,b]] exempts the source FK column."""
    store = base_models() + [
        SlayerModel(name="orders", data_source="db", sql_table="orders",
                    columns=[col("oid", "oid", type="INT", pk=True),
                             col("customer_id", "customer_id", type="INT")],  # FK, unused by query
                    joins=[{"target_model": "customers", "join_type": "left",
                            "join_pairs": [["customer_id", "id"]]}]),
        tbl_model("customers", [col("id", "id", type="INT", pk=True),
                                col("tier", "tier", kb=4)], table="customers"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "orders", "dimensions": ["customers.tier"]})
    assert "customer_id" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_query_level_measure_agg_suffix_marks_column_used():
    """A query-level measure `"sal:avg"` (agg suffix stripped) marks `sal` used; a `"*:count"`
    marks NO column, so an unrelated unused column is still flagged."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("sal", "shipping_fee + 1", kb=4),
            col("unused_extra", "restocking_fee + 1", kb=4),  # nothing references it
        ], table="financial_management"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "measures": ["sal:avg", "*:count"]},
    )
    used = entities(findings, "UNUSED_AGENT_ENTITY")
    assert "sal" not in used
    assert "unused_extra" in used


async def test_order_only_column_marks_used():
    """A column referenced ONLY via `order` (not dimensions/filters) counts as used."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("sal", "shipping_fee + 1", kb=4),
            col("rank_key", "restocking_fee + 1", kb=4),
        ], table="financial_management"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal"],
                       "order": [{"column": "rank_key", "direction": "desc"}]},
    )
    assert "rank_key" not in entities(findings, "UNUSED_AGENT_ENTITY")


# ===========================================================================
# KB-materialization cross-checks (advisory / soft flags)
# ===========================================================================


async def test_inline_model_extension_source_model_does_not_crash():
    """A winning query whose source_model is an inline dict (ModelExtension), not a
    named store model, must not crash the closure walk (it names no store root)."""
    store = base_models() + [
        tbl_model("return_sal", [col("sal", "shipping_fee + 1", kb=4)],
                  table="financial_management"),
    ]
    winning_query = {
        "source_model": {"base_model": "return_sal", "extra_columns": []},
        "dimensions": ["sal"],
    }
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4], winning_query=winning_query,
    )
    # no crash; deterministic result (sal not resolvable via an inline root → flagged unused)
    assert isinstance(findings, list)


async def test_expected_kb_not_materialized_flag_summarized():
    """Relevant KBs with no [kb=N] entity → a SINGLE summarised EXPECTED_KB flag (not one per
    KB), listing all of them in the detail."""
    store = base_models() + [
        tbl_model("properties", [col("x", "propref + 1", concept=True)], table="properties"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(27, [28]), kb(28, [])], relevant_kb_ids=[27],
        winning_query={"source_model": "properties", "dimensions": ["x"]},
    )
    flags = cats(findings, "EXPECTED_KB_NOT_MATERIALIZED")
    assert len(flags) == 1 and flags[0].level == "flag"
    assert "27" in flags[0].detail and "28" in flags[0].detail


async def test_join_key_column_exempt_from_unused():
    """A trivial-base JOIN-key column of a sql-backed model is NOT flagged unused even when the
    winning query never projects it (the join lives in the model, not the query)."""
    store = base_models() + [
        SlayerModel(
            name="return_sal", data_source="db",
            sql=("SELECT f.creditref AS creditref, f.casetag AS casetag, "
                 "f.shipping_fee + 1 AS sal "
                 "FROM financial_management f JOIN returns r ON f.casetag = r.casenum"),
            columns=[
                col("creditref", "creditref", type="TEXT", pk=True),
                col("casetag", "casetag", type="TEXT"),       # JOIN key, trivial-base, unused
                col("sal", "sal", kb=4),
            ],
        ),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal"]},
    )
    assert "casetag" not in entities(findings, "UNUSED_AGENT_ENTITY")


async def test_deferred_relevant_kb_flag():
    """A relevant KB present ONLY as a deferred [kb=N] memory → DEFERRED_RELEVANT_KB (separate flag)."""
    store = base_models()  # no [kb=5] column/measure anywhere
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(5, [])], relevant_kb_ids=[5],
        winning_query={"source_model": "financial_management", "dimensions": ["shipping_fee"]},
        memories=[{"id": 5, "text": "[kb=5] deferred: cannot operationalise"}],
    )
    deferred = cats(findings, "DEFERRED_RELEVANT_KB")
    assert 5 in {f.kb_id for f in deferred}
    # and it is NOT also reported as not-materialized (memory covers presence)
    assert 5 not in {f.kb_id for f in cats(findings, "EXPECTED_KB_NOT_MATERIALIZED")}


async def test_inline_query_work_json_filter():
    """A filter doing JSON extraction inline → INLINE_QUERY_WORK recommendation (Case B)."""
    store = base_models() + [
        tbl_model("properties", [col("propref", "propref", type="INT", pk=True)],
                  table="properties"),
    ]
    q = {"source_model": "properties", "measures": ["*:count"],
         "filters": ["jsonb_extract_path_text(dwelling_specs,'Bath_Count')::numeric * 10 > 20"]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[], winning_query=q)
    assert cats(findings, "INLINE_QUERY_WORK")


async def test_inline_query_work_multicol_arithmetic_filter():
    store = base_models() + [
        tbl_model("m", [col("a", "a"), col("b", "b"), col("c", "c")], table="t"),
    ]
    q = {"source_model": "m", "measures": ["*:count"], "filters": ["(a + b) > c"]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[], winning_query=q)
    assert cats(findings, "INLINE_QUERY_WORK")


async def test_inline_query_work_inline_source_model():
    store = base_models()
    q = {"source_model": {"base_model": "financial_management", "extra_columns": []},
         "measures": ["*:count"]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[], winning_query=q)
    assert cats(findings, "INLINE_QUERY_WORK")


async def test_simple_stored_column_filter_is_not_inline_work():
    """A filter that references a stored column with a literal comparison is fine — no flag."""
    store = base_models() + [
        tbl_model("properties", [
            col("propref", "propref", type="INT", pk=True),
            col("roomy", "propref + 1", concept=True),
        ], table="properties"),
    ]
    q = {"source_model": "properties", "measures": ["*:count"], "filters": ["roomy > 20"]}
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[], relevant_kb_ids=[], winning_query=q)
    assert cats(findings, "INLINE_QUERY_WORK") == []


async def test_unused_kb_concept_recommends_reference_not_delete():
    """An UNUSED [concept]/[kb] entity is recommended to be referenced-if-useful, NOT deleted;
    an unused UNTAGGED entity reads as scratch-to-delete."""
    store = base_models() + [
        tbl_model("m", [
            col("used", "a1 + 1", kb=4),
            col("concept_unused", "a2 + 1", concept=True),   # unused, tagged concept
            col("scratch_unused", "a3 + 1"),                 # unused, untagged
        ], table="t"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "m", "dimensions": ["used"]})
    unused = {f.entity: f for f in cats(findings, "UNUSED_AGENT_ENTITY")}
    assert "concept_unused" in unused and "reference" in unused["concept_unused"].detail
    assert "delete" not in unused["concept_unused"].detail.lower()
    assert "scratch_unused" in unused and (
        "scratch" in unused["scratch_unused"].detail or "deleting" in unused["scratch_unused"].detail)


async def test_orphan_kb_entity_flag():
    """A [kb=N] entity for a KB outside the relevant closure → ORPHAN_KB_ENTITY (flag),
    independent of whether it is query-used (Codex-tests #8): `bonus` (kb 99) is used,
    `stray` (kb 88) is unused — both must be reported orphan."""
    store = base_models() + [
        tbl_model("return_sal", [
            col("sal", "shipping_fee + 1", kb=4),
            col("bonus", "restocking_fee + 1", kb=99),  # orphan, used
            col("stray", "shipping_fee + 2", kb=88),    # orphan, unused
        ], table="financial_management"),
    ]
    findings = await check_models(
        store_models=store, baseline_models=base_models(),
        kb_rows=[kb(4, []), kb(99, []), kb(88, [])], relevant_kb_ids=[4],
        winning_query={"source_model": "return_sal", "dimensions": ["sal", "bonus"]},
    )
    orphan_kbs = {f.kb_id for f in cats(findings, "ORPHAN_KB_ENTITY")}
    assert 99 in orphan_kbs and 88 in orphan_kbs


# ===========================================================================
# report assembly + baseline availability
# ===========================================================================


def test_report_ok_false_when_substantive_finding():
    findings = [Finding(category="UNUSED_AGENT_ENTITY", level="flag",
                        entity="x", detail="unused")]
    rep = StoreCheckReport.from_findings(findings, instance_id="reverse_logistics_4",
                                         db="reverse_logistics_large",
                                         benchmark="livesqlbench-large", baseline_available=True)
    assert rep.ok is False


def test_report_ok_true_when_only_advisory():
    # advisory-only (EXPECTED_KB / ORPHAN / DEFERRED) does NOT make a store "unclean"
    findings = [Finding(category="EXPECTED_KB_NOT_MATERIALIZED", level="flag",
                        kb_id=27, detail="advisory")]
    rep = StoreCheckReport.from_findings(findings, baseline_available=True)
    assert rep.ok is True


def test_report_ok_false_when_non_kb_flag():
    findings = [Finding(category="NON_KB_ENTITY", level="flag", entity="c", detail="untagged")]
    assert StoreCheckReport.from_findings(findings, baseline_available=True).ok is False


async def test_baseline_missing_skips_not_passes():
    """baseline_models=None (cache absent / cache_fp mismatch) → no diff possible; the checker
    yields no findings and the report is marked baseline_available=False (skipped, not passed)."""
    findings = await check_models(
        store_models=base_models(), baseline_models=None,
        kb_rows=[], relevant_kb_ids=[],
        winning_query={"source_model": "financial_management", "dimensions": ["shipping_fee"]},
    )
    assert findings == []
    rep = StoreCheckReport.from_findings(findings, baseline_available=False)
    assert rep.baseline_available is False
