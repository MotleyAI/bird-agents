"""DEV-1907 (task 2.2): the store KB-checker's filter-ref helper is rehomed to
SLayer 0.10.2 and recurses into aggregate sources + kwarg refs, so a column
referenced ONLY inside an aggregate-over-expression filter still counts as USED.

0.10.2 removed `slayer.core.formula.parse_filter`; the old helper swallowed the
`ValueError` it raises on `sum(<expression>)` and lost the inner columns. The
rehomed helper (parse_filter_expr + walk_parsed_refs into AggCall.source /
kwargs) must recover them. Proven by contrast against an unreferenced sibling.

Imports of `store_kb_checker` are deferred: under 0.10.2 the un-migrated module
fails to import `parse_filter`, so these are RED (feature missing) until task 2.2.
"""

from __future__ import annotations


def _col(name, sql, *, kb=None, type="DOUBLE", pk=False):
    from slayer.core.models import Column

    desc = f"[kb={kb}] {name}" if kb is not None else None
    return Column(name=name, sql=sql, type=type, primary_key=pk, description=desc)


def _tbl(name, columns, *, table="t"):
    from slayer.core.models import SlayerModel

    return SlayerModel(name=name, data_source="db", sql_table=table, columns=columns,
                       measures=[], joins=[])


def _kb(id_, children):
    return {"id": id_, "knowledge": f"kb{id_}", "definition": f"def{id_}",
            "type": "calculation_knowledge", "children_knowledge": children}


def _base():
    return [_tbl("base", [_col("k", "k", type="INT", pk=True)], table="base")]


def _unused(findings):
    return {f.entity for f in findings if f.category == "UNUSED_AGENT_ENTITY"}


async def _run(winning_query):
    from bird_interact_agents.slayer_otf.store_kb_checker import check_models

    store = _base() + [
        _tbl("txns", [
            _col("txid", "txid", type="INT", pk=True),
            _col("amount", "gross_amount", kb=99),
            _col("cost", "gross_cost", kb=99),
            _col("region", "home_region", kb=99),
            _col("unused_sibling", "gross_amount * 2", kb=99),
        ], table="txns"),
    ]
    return await check_models(
        store_models=store, baseline_models=_base(),
        kb_rows=[_kb(99, [])], relevant_kb_ids=[99],
        winning_query=winning_query,
    )


async def test_aggregate_over_expression_filter_marks_inner_columns_used():
    """`sum(amount - cost) > 100` contributes BOTH `amount` and `cost`; an
    otherwise-identical unreferenced sibling is still flagged, so the pass
    genuinely depends on resolving the aggregate source."""
    findings = await _run(
        {"source_model": "txns", "filters": ["sum(amount - cost) > 100"]}
    )
    unused = _unused(findings)
    assert "amount" not in unused
    assert "cost" not in unused
    assert "unused_sibling" in unused


async def test_aggregate_partition_by_kwarg_ref_marks_column_used():
    """A column referenced only in a `partition_by=` kwarg of an aggregate
    filter counts as used."""
    findings = await _run(
        {"source_model": "txns", "filters": ["sum(amount, partition_by=region) > 100"]}
    )
    unused = _unused(findings)
    assert "amount" not in unused
    assert "region" not in unused
    assert "unused_sibling" in unused


async def test_plain_column_filter_still_resolves():
    """Regression: a non-aggregate filter still marks its column used."""
    findings = await _run(
        {"source_model": "txns", "filters": ["amount > 0", "cost > 0", "region > 0"]}
    )
    unused = _unused(findings)
    assert {"amount", "cost", "region"}.isdisjoint(unused)
    assert "unused_sibling" in unused
