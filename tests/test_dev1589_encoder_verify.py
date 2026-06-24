"""DEV-1589: the shared hard-check / auto-wire helpers in
`slayer_otf.encoder_verify`.

These are the framework-neutral verification primitives the claude_sdk build
encoder runs AFTER each per-KB agent session closes (parent storage is then the
single writer). They are deliberately storage-driven and open a FRESH
`YAMLStorage(base_dir=build_dir)` per call (Codex r1 #4 — never a cached handle).
"""

from __future__ import annotations

import pytest
from slayer.core.models import (
    Column,
    DatasourceConfig,
    ModelMeasure,
    SlayerModel,
)
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf import encoder_verify as ev
from bird_interact_agents.slayer_otf.encoder_types import (
    EncodedEntity,
    EncoderResult,
)

DB = "tinydb"


async def _storage(build_dir):
    storage = YAMLStorage(base_dir=str(build_dir))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    return storage


def _ent(name, host="m", kind="column"):
    return EncodedEntity(
        kind=kind, host_model=host, name=name, entity_ref=f"{DB}.{host}.{name}",
    )


# ---------------------------------------------------------------------------
# meta_has_kb_id + scoped presence
# ---------------------------------------------------------------------------


def test_meta_has_kb_id_accepts_int_and_str():
    assert ev.meta_has_kb_id({"kb_id": 5}, 5)
    assert ev.meta_has_kb_id({"kb_id": "5"}, 5)
    assert not ev.meta_has_kb_id({"kb_id": 6}, 5)
    assert not ev.meta_has_kb_id(None, 5)
    assert not ev.meta_has_kb_id({}, 5)


async def test_present_and_tagged_is_datasource_scoped(tmp_path):
    """Codex r1 #5: an identically-named entity under a DIFFERENT datasource
    must NOT satisfy presence for `db`."""
    storage = await _storage(tmp_path)
    # Same model+column name, but a different datasource.
    await storage.save_datasource(DatasourceConfig(
        name="other", type="sqlite", connection_string="sqlite:///y.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="m", data_source="other", sql_table="m",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c", sql="1", meta={"kb_id": 5})],
    ))
    storage2 = YAMLStorage(base_dir=str(tmp_path))
    assert not await ev.entity_present_and_tagged(_ent("c"), storage2, DB, 5)

    # Now add it under DB, tagged → present.
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c", sql="1", meta={"kb_id": 5})],
    ))
    storage3 = YAMLStorage(base_dir=str(tmp_path))
    assert await ev.entity_present_and_tagged(_ent("c"), storage3, DB, 5)


async def test_present_and_tagged_rejects_untagged(tmp_path):
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True), Column(name="c", sql="1")],
    ))
    storage2 = YAMLStorage(base_dir=str(tmp_path))
    assert not await ev.entity_present_and_tagged(_ent("c"), storage2, DB, 5)


# ---------------------------------------------------------------------------
# hard_failures: HC-present
# ---------------------------------------------------------------------------


async def test_hard_failures_flags_absent_entity(tmp_path):
    await _storage(tmp_path)  # datasource only, no model
    failures = await ev.hard_failures(
        tmp_path, DB, 5, [_ent("ghost")], encoded_deps=[],
    )
    assert failures  # non-empty → HC-present failed


async def test_hard_failures_empty_when_present_and_no_deps(tmp_path):
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c", sql="1", meta={"kb_id": 5})],
    ))
    failures = await ev.hard_failures(
        tmp_path, DB, 5, [_ent("c")], encoded_deps=[],
    )
    assert failures == []


# ---------------------------------------------------------------------------
# referenced_identifiers: structured-field inspection (string literals excluded)
# ---------------------------------------------------------------------------


def test_referenced_identifiers_excludes_string_literals():
    toks = ev.referenced_identifiers("CASE WHEN tier = 'premium_revenue' THEN x END")
    assert "tier" in toks
    assert "x" in toks
    # the dep name appears ONLY inside a string literal → must NOT be a token
    assert "premium_revenue" not in toks


def test_referenced_identifiers_picks_up_identifier_position():
    toks = ev.referenced_identifiers("premium_revenue / order_count")
    assert "premium_revenue" in toks
    assert "order_count" in toks


def test_structured_ref_tokens_extracts_values_not_keys():
    """For structured fields (source_queries), a dep reference is a string VALUE
    (e.g. source_model: 'premium_revenue') — it must be extracted (not stripped
    as a literal), and dict KEYS / field names must NOT become tokens (Codex)."""
    toks = ev._structured_ref_tokens(
        {"source_model": "premium_revenue", "measures": ["amount", "qty"]},
    )
    assert "premium_revenue" in toks   # the string VALUE (a real reference)
    assert "amount" in toks and "qty" in toks
    assert "source_model" not in toks  # a dict KEY — not a reference
    assert "measures" not in toks


# ---------------------------------------------------------------------------
# hard_failures: HC-depuse (structured-field)
# ---------------------------------------------------------------------------


async def _storage_with_dep_and_dependent(tmp_path, *, dependent_sql):
    """Dep KB 4 encoded measure `premium_revenue` on model `orders`.
    Dependent KB 7 encodes column `margin` on `orders` with `dependent_sql`."""
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="orders", data_source=DB, sql_table="orders",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="amount", sql="amount"),
            Column(name="margin", sql=dependent_sql, meta={"kb_id": 7}),
        ],
        measures=[ModelMeasure(
            name="premium_revenue", formula="amount:sum", meta={"kb_id": 4},
        )],
    ))
    return storage


def _dep_kb4():
    return EncoderResult(
        kb_id=4, status="encoded",
        entities=[EncodedEntity(kind="measure", host_model="orders",
                                name="premium_revenue",
                                entity_ref=f"{DB}.orders.premium_revenue")],
    )


async def test_hard_failures_depuse_passes_when_dep_referenced(tmp_path):
    await _storage_with_dep_and_dependent(
        tmp_path, dependent_sql="premium_revenue / 2",
    )
    failures = await ev.hard_failures(
        tmp_path, DB, 7, [_ent("margin", host="orders")],
        encoded_deps=[_dep_kb4()],
    )
    assert failures == []


async def test_hard_failures_depuse_fails_when_dep_inlined(tmp_path):
    # Re-derives the logic (SUM(amount)) instead of referencing the dep entity.
    await _storage_with_dep_and_dependent(
        tmp_path, dependent_sql="SUM(amount) / 2",
    )
    failures = await ev.hard_failures(
        tmp_path, DB, 7, [_ent("margin", host="orders")],
        encoded_deps=[_dep_kb4()],
    )
    assert failures  # dep encoded but not referenced → HC-depuse failed


async def test_hard_failures_depuse_string_literal_does_not_count(tmp_path):
    # The dep name appears only inside a string literal — must NOT satisfy.
    await _storage_with_dep_and_dependent(
        tmp_path, dependent_sql="CASE WHEN label='premium_revenue' THEN 1 END",
    )
    failures = await ev.hard_failures(
        tmp_path, DB, 7, [_ent("margin", host="orders")],
        encoded_deps=[_dep_kb4()],
    )
    assert failures


async def test_hard_failures_depuse_via_measure_formula(tmp_path):
    """Structured-field inspection must read a Measure's `formula` (not just a
    Column's `sql`) when the dependent entity is a measure (Codex test #5)."""
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="orders", data_source=DB, sql_table="orders",
        columns=[Column(name="id", primary_key=True), Column(name="amount", sql="amount")],
        measures=[
            ModelMeasure(name="premium_revenue", formula="amount:sum",
                         meta={"kb_id": 4}),
            # dependent measure 'margin_rate' references the dep measure by name
            ModelMeasure(name="margin_rate", formula="premium_revenue / 100",
                         meta={"kb_id": 7}),
        ],
    ))
    ent = EncodedEntity(kind="measure", host_model="orders", name="margin_rate",
                        entity_ref=f"{DB}.orders.margin_rate")
    failures = await ev.hard_failures(
        tmp_path, DB, 7, [ent], encoded_deps=[_dep_kb4()],
    )
    assert failures == []


async def test_hard_failures_depuse_description_only_does_not_count(tmp_path):
    """A dep name appearing only in the dependent's DESCRIPTION (not in a
    structured definition field) does NOT count as a reference (Codex test #5)."""
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="orders", data_source=DB, sql_table="orders",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="amount", sql="amount"),
            Column(name="margin", sql="SUM(amount) / 2",
                   description="builds on premium_revenue conceptually",
                   meta={"kb_id": 7}),
        ],
        measures=[ModelMeasure(name="premium_revenue", formula="amount:sum",
                               meta={"kb_id": 4})],
    ))
    failures = await ev.hard_failures(
        tmp_path, DB, 7, [_ent("margin", host="orders")],
        encoded_deps=[_dep_kb4()],
    )
    assert failures  # description mention is not a real reference


async def test_hard_failures_depuse_via_model_backing_query(tmp_path):
    """For a query-backed MODEL entity, HC-depuse must read structural dep refs
    from backing_query_sql / source_queries, not just model.sql + leaf fields
    (Codex review)."""
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="orders", data_source=DB, sql_table="orders",
        columns=[Column(name="id", primary_key=True), Column(name="amount", sql="amount")],
        measures=[ModelMeasure(name="premium_revenue", formula="amount:sum",
                               meta={"kb_id": 4})],
    ))
    # A dependent query-backed model referencing the dep via its backing query.
    await storage.save_model(SlayerModel(
        name="summary", data_source=DB, sql_table="summary",
        columns=[Column(name="id", primary_key=True)],
        backing_query_sql="SELECT premium_revenue FROM orders",
        meta={"kb_id": 7},
    ))
    ent = EncodedEntity(kind="model", host_model=None, name="summary",
                        entity_ref=f"{DB}.summary")
    failures = await ev.hard_failures(
        tmp_path, DB, 7, [ent], encoded_deps=[_dep_kb4()],
    )
    assert failures == []


async def test_hard_failures_opens_fresh_storage_per_call(tmp_path, monkeypatch):
    """Codex r1 #4 / test #6: verification must construct a FRESH
    YAMLStorage(base_dir=build_dir) each pass — never a cached handle."""
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c", sql="1", meta={"kb_id": 5})],
    ))
    real = ev.YAMLStorage
    count = {"n": 0}

    def _counting(*a, **k):
        count["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ev, "YAMLStorage", _counting)
    await ev.hard_failures(tmp_path, DB, 5, [_ent("c")], encoded_deps=[])
    n_first = count["n"]
    assert n_first >= 1
    # A second call constructs storage AGAIN (no module-level caching).
    await ev.hard_failures(tmp_path, DB, 5, [_ent("c")], encoded_deps=[])
    assert count["n"] > n_first


# ---------------------------------------------------------------------------
# auto-wire: description (HC-desc) + memory backref (HC-mem)
# ---------------------------------------------------------------------------


async def test_autowire_descriptions_injects_and_is_idempotent(tmp_path):
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c", sql="1", description="short", meta={"kb_id": 5})],
    ))
    verbatim = "[kb=5]\nKB item (verbatim):\nid: 5\nknowledge: full text"
    await ev.autowire_descriptions(tmp_path, DB, [_ent("c")], verbatim)

    s2 = YAMLStorage(base_dir=str(tmp_path))
    model = await s2.get_model("m", data_source=DB)
    col = next(c for c in model.columns if c.name == "c")
    assert verbatim in (col.description or "")

    # idempotent: second run does not double-inject
    await ev.autowire_descriptions(tmp_path, DB, [_ent("c")], verbatim)
    s3 = YAMLStorage(base_dir=str(tmp_path))
    model3 = await s3.get_model("m", data_source=DB)
    col3 = next(c for c in model3.columns if c.name == "c")
    assert (col3.description or "").count(verbatim) == 1


async def test_autowire_memory_backrefs_appends_ref(tmp_path):
    storage = await _storage(tmp_path)
    await storage.save_memory(
        learning="KB 5 — x", entities=[DB], query=None,
        id=f"{DB}_kb_5", description="",
    )
    await ev.autowire_memory_backrefs(tmp_path, DB, 5, [_ent("c")])

    s2 = YAMLStorage(base_dir=str(tmp_path))
    mem = await s2.get_memory_row(f"{DB}_kb_5")
    assert f"{DB}.m.c" in mem.entities
    # idempotent
    await ev.autowire_memory_backrefs(tmp_path, DB, 5, [_ent("c")])
    s3 = YAMLStorage(base_dir=str(tmp_path))
    mem3 = await s3.get_memory_row(f"{DB}_kb_5")
    assert mem3.entities.count(f"{DB}.m.c") == 1


# ---------------------------------------------------------------------------
# purge: delete by meta.kb_id + prune backrefs
# ---------------------------------------------------------------------------


async def test_purge_deletes_tagged_entities_and_prunes_memory(tmp_path):
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="keep", sql="1", meta={"kb_id": 99}),   # different kb
            Column(name="drop_me", sql="1", meta={"kb_id": 7}),
        ],
        measures=[ModelMeasure(name="drop_meas", formula="id:count",
                               meta={"kb_id": 7})],
    ))
    await storage.save_memory(
        learning="KB 7 — x", entities=[DB, f"{DB}.m.drop_me", f"{DB}.m.drop_meas"],
        query=None, id=f"{DB}_kb_7", description="",
    )

    await ev.purge_kb_entities_and_backrefs(tmp_path, DB, 7)

    s2 = YAMLStorage(base_dir=str(tmp_path))
    model = await s2.get_model("m", data_source=DB)
    names = {c.name for c in (model.columns or [])}
    meas = {m.name for m in (model.measures or [])}
    assert "drop_me" not in names and "drop_meas" not in meas
    assert "keep" in names  # untouched (different kb_id)
    mem = await s2.get_memory_row(f"{DB}_kb_7")
    assert f"{DB}.m.drop_me" not in mem.entities
    assert f"{DB}.m.drop_meas" not in mem.entities
    assert DB in mem.entities  # the datasource anchor stays


async def test_purge_whole_model_prunes_model_and_leaf_backrefs(tmp_path):
    """When an entire query-backed model is tagged meta.kb_id and deleted, BOTH
    its model-level ref `<db>.<model>` AND any leaf refs `<db>.<model>.<leaf>`
    must be pruned from memory (CodeRabbit) — exact-match removal alone would
    leave the leaf refs dangling."""
    storage = await _storage(tmp_path)
    await storage.save_model(SlayerModel(
        name="qmodel", data_source=DB, sql_table="qmodel",
        columns=[Column(name="id", primary_key=True)],
        meta={"kb_id": 7},   # the WHOLE model is tagged for kb 7
    ))
    await storage.save_memory(
        learning="KB 7 — x",
        entities=[DB, f"{DB}.qmodel", f"{DB}.qmodel.id", f"{DB}.other.keep"],
        query=None, id=f"{DB}_kb_7", description="",
    )

    await ev.purge_kb_entities_and_backrefs(tmp_path, DB, 7)

    s2 = YAMLStorage(base_dir=str(tmp_path))
    assert await s2.get_model("qmodel", data_source=DB) is None  # model deleted
    mem = await s2.get_memory_row(f"{DB}_kb_7")
    assert f"{DB}.qmodel" not in mem.entities       # model-level ref pruned
    assert f"{DB}.qmodel.id" not in mem.entities    # dangling leaf ref pruned
    assert DB in mem.entities                       # datasource anchor stays
    assert f"{DB}.other.keep" in mem.entities        # unrelated ref untouched
