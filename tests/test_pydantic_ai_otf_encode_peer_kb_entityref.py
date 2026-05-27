"""Tests for the peer-KB block's entity_ref + reachable_from_host
enrichment (DEV-1478 follow-up, plan §S2).

The existing ``_format_existing_kb_tagged_entities_block`` already lists
peer KBs by ``host_model.name`` + ``kind`` + ``kb_id``. The encoder has
to reconstruct the qualified ``entity_ref`` ("db.model[.leaf]") on its
own when it wants to reference a peer in cross-model SQL. We enrich the
renderer to surface ``entity_ref`` verbatim AND, for each peer, the set
of candidate host models that can reach the peer via the declared join
graph (hop counts: self = 0, one join = 1, etc.).

Design: the renderer delegates to a typed collector
``_collect_peer_kb_entries`` that returns ``list[PeerKbEntry]``. Tests
exercise the COLLECTOR for structural assertions (entity_ref,
reachable_from_host per peer); a smaller set of tests verify the
renderer's labeled-marker contract so the encoder can find each peer's
fields by literal key (``entity_ref=...``, ``reachable_from_host:
host(hops), host(hops)``).

Per ``feedback_no_prompt_content_tests.md``: tests assert on machine-
readable field markers and structured collector output, not on prose.
"""

from __future__ import annotations

from slayer.core.models import (
    Column,
    DatasourceConfig,
    ModelJoin,
    ModelMeasure,
    SlayerModel,
)
from slayer.storage.yaml_storage import YAMLStorage

DB = "tinydb"


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


async def _ds_with_two_models(tmp_path):
    """Two models linked by a declared join: ``properties -> infrastructure``
    (1 hop), plus a kb-tagged Column on the target."""
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="infrastructure", data_source=DB, sql_table="infrastructure",
        columns=[
            Column(name="infraref", primary_key=True),
            Column(name="infrastructure_quality_score", sql="1",
                   meta={"kb_id": 13}),
        ],
    ))
    await storage.save_model(SlayerModel(
        name="properties", data_source=DB, sql_table="properties",
        columns=[
            Column(name="houselink", primary_key=True),
            Column(name="infralink"),
        ],
        joins=[ModelJoin(
            target_model="infrastructure",
            join_pairs=[["infralink", "infraref"]],
        )],
    ))
    return storage


async def _ds_with_three_hop_chain(tmp_path):
    """Three models in a chain: ``a -> b -> c``. A kb-tagged column on
    ``c`` is 0 hops from ``c``, 1 hop from ``b``, 2 hops from ``a``."""
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="c", data_source=DB, sql_table="c",
        columns=[
            Column(name="c_pk", primary_key=True),
            Column(name="leaf_score", sql="1", meta={"kb_id": 50}),
        ],
    ))
    await storage.save_model(SlayerModel(
        name="b", data_source=DB, sql_table="b",
        columns=[
            Column(name="b_pk", primary_key=True),
            Column(name="b_to_c"),
        ],
        joins=[ModelJoin(
            target_model="c", join_pairs=[["b_to_c", "c_pk"]],
        )],
    ))
    await storage.save_model(SlayerModel(
        name="a", data_source=DB, sql_table="a",
        columns=[
            Column(name="a_pk", primary_key=True),
            Column(name="a_to_b"),
        ],
        joins=[ModelJoin(
            target_model="b", join_pairs=[["a_to_b", "b_pk"]],
        )],
    ))
    return storage


# ---------------------------------------------------------------------------
# Collector — structural data.
# ---------------------------------------------------------------------------


async def test_collector_returns_entity_ref_for_column(tmp_path):
    """The collector's column entry exposes ``entity_ref`` as the
    fully-qualified ``db.model.col`` string, NOT just the parts the
    encoder would have to reassemble."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = await _ds_with_two_models(tmp_path)
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert 13 in by_kb, f"missing kb=13 in collector output: {entries}"
    e = by_kb[13]
    assert e.entity_ref == f"{DB}.infrastructure.infrastructure_quality_score"
    assert e.kind == "column"
    assert e.host_model == "infrastructure"


async def test_collector_returns_entity_ref_for_model_without_leaf(tmp_path):
    """A kb-tagged MODEL → ``entity_ref`` is ``db.model`` (no leaf)
    and ``kind == "model"``."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="mobility_index", data_source=DB,
        sql_table="mobility_index",
        columns=[Column(name="id", primary_key=True)],
        meta={"kb_id": 18},
    ))
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert by_kb[18].entity_ref == f"{DB}.mobility_index"
    assert by_kb[18].kind == "model"


async def test_collector_returns_entity_ref_for_measure(tmp_path):
    """A kb-tagged ModelMeasure → ``entity_ref`` is ``db.model.measure``.
    SLayer treats a measure as a leaf-of-model just like a column."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[Column(name="id", primary_key=True),
                 Column(name="resident", sql="1")],
        measures=[ModelMeasure(
            name="avg_residents", formula="resident:avg",
            meta={"kb_id": 7},
        )],
    ))
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert by_kb[7].entity_ref == f"{DB}.households.avg_residents"
    assert by_kb[7].kind == "measure"


# ---------------------------------------------------------------------------
# Reachability map — per-peer, hop-count-correct.
# ---------------------------------------------------------------------------


async def test_collector_reach_self_host_is_zero_hops(tmp_path):
    """The peer's own host is always reachable at hop 0."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = await _ds_with_two_models(tmp_path)
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    reach = dict(by_kb[13].reachable_from_host)
    assert reach["infrastructure"] == 0


async def test_collector_reach_one_hop_via_declared_join(tmp_path):
    """``properties`` reaches ``infrastructure`` via the declared
    ``properties.infralink = infrastructure.infraref`` join → hop 1.
    Asserts the EXACT reach map so a collector that adds unreachable
    extras would fail."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = await _ds_with_two_models(tmp_path)
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert dict(by_kb[13].reachable_from_host) == {
        "infrastructure": 0,
        "properties": 1,
    }


async def test_collector_reach_multi_hop(tmp_path):
    """A 3-model chain ``a -> b -> c`` with a kb-tagged column on ``c``
    must report exactly ``{c=0, b=1, a=2}`` so the encoder can choose
    any host in the chain knowing the cost — and no other entries leak
    in."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = await _ds_with_three_hop_chain(tmp_path)
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert dict(by_kb[50].reachable_from_host) == {
        "c": 0, "b": 1, "a": 2,
    }


async def test_collector_reach_handles_multi_pair_join(tmp_path):
    """A ``ModelJoin`` with a composite ``join_pairs`` is still one join
    (one edge between two models) and must be reported as hop 1
    EXACTLY ONCE — collapsing via ``dict(...)`` would mask a duplicate
    entry, so assert on the raw list."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="t_target", data_source=DB, sql_table="t_target",
        columns=[
            Column(name="k1", primary_key=True),
            Column(name="k2", primary_key=True),
            Column(name="score", sql="1", meta={"kb_id": 99}),
        ],
    ))
    await storage.save_model(SlayerModel(
        name="t_source", data_source=DB, sql_table="t_source",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="fk1"),
            Column(name="fk2"),
        ],
        joins=[ModelJoin(
            target_model="t_target",
            join_pairs=[["fk1", "k1"], ["fk2", "k2"]],
        )],
    ))
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    reach_list = list(by_kb[99].reachable_from_host)
    # The composite join must produce ONE edge, not two, so the
    # collector's list contains exactly one entry per (host, hops).
    assert reach_list.count(("t_source", 1)) == 1, (
        f"composite join must produce exactly one (t_source, 1) edge; "
        f"got: {reach_list}"
    )
    assert reach_list.count(("t_target", 0)) == 1
    assert len(reach_list) == 2, (
        f"reach list must contain exactly t_source + t_target (no extras "
        f"or duplicates); got: {reach_list}"
    )


async def test_collector_reach_for_model_entity(tmp_path):
    """A kb-tagged MODEL entity also gets a reach map (it's the host
    itself + every model that joins to it). Exact-map assertion so a
    leaky collector would fail."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="centre", data_source=DB, sql_table="centre",
        columns=[Column(name="id", primary_key=True)],
        meta={"kb_id": 30},
    ))
    await storage.save_model(SlayerModel(
        name="leaf", data_source=DB, sql_table="leaf",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="centre_fk"),
        ],
        joins=[ModelJoin(
            target_model="centre", join_pairs=[["centre_fk", "id"]],
        )],
    ))
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert dict(by_kb[30].reachable_from_host) == {"centre": 0, "leaf": 1}


async def test_collector_reach_for_measure_entity(tmp_path):
    """Codex follow-up: measures must carry ``reachable_from_host`` too.
    A measure is a leaf-of-host just like a column, so the host-self
    edge plus declared inbound joins must appear."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[Column(name="housenum", primary_key=True),
                 Column(name="resident", sql="1")],
        measures=[ModelMeasure(
            name="avg_residents", formula="resident:avg",
            meta={"kb_id": 7},
        )],
    ))
    await storage.save_model(SlayerModel(
        name="properties", data_source=DB, sql_table="properties",
        columns=[Column(name="houselink", primary_key=True)],
        joins=[ModelJoin(
            target_model="households",
            join_pairs=[["houselink", "housenum"]],
        )],
    ))
    entries = await _collect_peer_kb_entries(storage, DB)
    by_kb = {e.kb_id: e for e in entries}
    assert dict(by_kb[7].reachable_from_host) == {
        "households": 0, "properties": 1,
    }


# ---------------------------------------------------------------------------
# Renderer — labeled markers.
# ---------------------------------------------------------------------------


async def test_block_includes_labeled_entity_ref_marker(tmp_path):
    """Renderer surfaces ``entity_ref=`` as a labeled field so the
    encoder can find the verbatim ref by key (not by reconstructing
    ``db.model.col`` itself)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = await _ds_with_two_models(tmp_path)
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    expected = (
        f"entity_ref={DB}.infrastructure.infrastructure_quality_score"
    )
    assert expected in block, (
        f"peer-KB block must contain the labeled marker "
        f"`entity_ref=<ref>`; got:\n{block}"
    )


async def test_block_includes_labeled_reachable_from_host_marker(tmp_path):
    """Renderer surfaces the reach map as ``reachable_from_host:`` plus
    a structured ``host(hops)`` list, so the encoder can find each
    peer's reach map by literal key."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = await _ds_with_two_models(tmp_path)
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    assert "reachable_from_host:" in block, (
        f"peer-KB block must include the labeled `reachable_from_host:` "
        f"field; got:\n{block}"
    )
    # Each candidate host must appear with its hop count in the
    # `host(hops)` form within the labeled section.
    assert "infrastructure(0)" in block
    assert "properties(1)" in block


async def test_block_two_peers_each_gets_own_entity_ref_and_reach_map(
    tmp_path,
):
    """Codex follow-up: with TWO peer entities (different kb_ids, on
    different hosts), each peer's section must carry its OWN
    ``entity_ref=`` and its OWN ``reachable_from_host:`` block whose
    contents reflect ITS host (not a copy of the other peer's, not a
    global map for the whole DB). Parse the block into per-entry chunks
    and assert per-peer reach maps."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    # Two-leaf topology: peer kb=11 on `properties`, peer kb=13 on
    # `infrastructure`. `properties` reaches `infrastructure` via the
    # declared join; `infrastructure` does NOT reach `properties`
    # (declared joins are directional).
    await storage.save_model(SlayerModel(
        name="infrastructure", data_source=DB, sql_table="infrastructure",
        columns=[
            Column(name="infraref", primary_key=True),
            Column(name="iqs", sql="1", meta={"kb_id": 13}),
        ],
    ))
    await storage.save_model(SlayerModel(
        name="properties", data_source=DB, sql_table="properties",
        columns=[
            Column(name="houselink", primary_key=True),
            Column(name="infralink"),
            Column(name="density", sql="1", meta={"kb_id": 11}),
        ],
        joins=[ModelJoin(
            target_model="infrastructure",
            join_pairs=[["infralink", "infraref"]],
        )],
    ))
    block = await _format_existing_kb_tagged_entities_block(storage, DB)

    # Both peers' labeled entity_refs must appear.
    density_ref = f"entity_ref={DB}.properties.density"
    iqs_ref = f"entity_ref={DB}.infrastructure.iqs"
    assert density_ref in block
    assert iqs_ref in block

    # The reach-map LABEL must appear at least twice (once per peer) —
    # a one-and-done bug would emit it only once.
    assert block.count("reachable_from_host:") >= 2, (
        f"each peer must carry its own reachable_from_host marker; "
        f"saw only {block.count('reachable_from_host:')} marker(s) in:\n"
        f"{block}"
    )

    # Per-peer association: split the block at each peer's bullet
    # marker so we can inspect the slice that belongs to ONE peer.
    # Renderer contract: peer entries start with `\n  - ` (existing
    # format from the legacy renderer at factories.py:744-821).
    # Split tolerantly so the test survives a renderer that adds a
    # continuation line for reach-map.
    SEP = "\n  - "
    chunks = (SEP + block).split(SEP)
    by_ref: dict[str, str] = {}
    for c in chunks:
        if density_ref in c:
            by_ref[density_ref] = c
        elif iqs_ref in c:
            by_ref[iqs_ref] = c
    assert density_ref in by_ref and iqs_ref in by_ref, (
        f"each peer's entity_ref must be locatable in its own chunk; "
        f"chunks were:\n{chunks}"
    )

    # Joins are directional: `properties.joins=[target=infrastructure]`
    # means properties → infrastructure. So from `properties` you reach
    # `infrastructure`, NOT the reverse.
    #
    # The properties.density peer (host=properties) chunk: ONLY
    # properties(0) belongs here. `infrastructure` does NOT declare a
    # join to properties, so infrastructure cannot reach properties.
    density_chunk = by_ref[density_ref]
    assert "reachable_from_host:" in density_chunk
    assert "properties(0)" in density_chunk
    assert "infrastructure(0)" not in density_chunk, (
        f"density chunk must NOT carry infrastructure(0) — joins are "
        f"directional and infrastructure does not reach properties. "
        f"chunk:\n{density_chunk}"
    )
    assert "infrastructure(1)" not in density_chunk, (
        f"density chunk must NOT carry infrastructure(1) either — "
        f"infrastructure has no declared outgoing join. "
        f"chunk:\n{density_chunk}"
    )
    # The infrastructure.iqs peer (host=infrastructure) chunk: reaches
    # infrastructure(0) AND properties(1) (because properties declares
    # the join to infrastructure).
    iqs_chunk = by_ref[iqs_ref]
    assert "reachable_from_host:" in iqs_chunk
    assert "infrastructure(0)" in iqs_chunk
    assert "properties(1)" in iqs_chunk
    # And cross-check: the iqs chunk must NOT carry density's self-hop
    # (`properties(0)`), which would indicate a global-reach-map bug
    # copying the same data across all peers.
    assert "properties(0)" not in iqs_chunk, (
        f"iqs chunk must NOT carry properties(0); that's density's "
        f"self-hop. Likely a global-reach-map bug. "
        f"chunk:\n{iqs_chunk}"
    )


# ---------------------------------------------------------------------------
# Regression guards — existing contract preserved.
# ---------------------------------------------------------------------------


async def test_block_preserves_kb_id_field(tmp_path):
    """The entity_ref + reach map enrichment must NOT drop the existing
    ``kb_id=N`` rendering — the PEER-KB DEDUP rule matches on it."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = await _ds_with_two_models(tmp_path)
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    assert "kb_id=13" in block


async def test_block_still_filters_by_current_kb_id(tmp_path):
    """Concurrency filter (kb_id < current_kb_id) is unchanged by the
    new fields."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = await _ds_with_two_models(tmp_path)
    block_below = await _format_existing_kb_tagged_entities_block(
        storage, DB, current_kb_id=14,
    )
    assert "kb_id=13" in block_below
    block_above = await _format_existing_kb_tagged_entities_block(
        storage, DB, current_kb_id=10,
    )
    assert "kb_id=13" not in block_above


async def test_collector_filters_by_current_kb_id_strict_less(tmp_path):
    """Codex follow-up: pin the strict-less-than concurrency filter at
    the COLLECTOR level too, including same-id exclusion. Otherwise
    the renderer test alone leaves the contract ambiguous."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="m_lower", data_source=DB, sql_table="m_lower",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c_lower", sql="1", meta={"kb_id": 3})],
    ))
    await storage.save_model(SlayerModel(
        name="m_same", data_source=DB, sql_table="m_same",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c_same", sql="1", meta={"kb_id": 10})],
    ))
    await storage.save_model(SlayerModel(
        name="m_higher", data_source=DB, sql_table="m_higher",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c_higher", sql="1", meta={"kb_id": 15})],
    ))
    entries = await _collect_peer_kb_entries(
        storage, DB, current_kb_id=10,
    )
    kb_ids = {e.kb_id for e in entries}
    assert kb_ids == {3}, (
        f"strict-less-than filter: only kb_id=3 should pass when "
        f"current_kb_id=10; got {kb_ids}"
    )

    # No filter → all three appear.
    all_entries = await _collect_peer_kb_entries(
        storage, DB, current_kb_id=None,
    )
    all_kb_ids = {e.kb_id for e in all_entries}
    assert all_kb_ids == {3, 10, 15}


async def test_collector_preserves_multiple_entries_with_same_kb_id(tmp_path):
    """Codex follow-up: a Column + ModelMeasure sharing the same kb_id
    is a valid R-FILTER recipe output (already covered by the existing
    same-kb_id stability test in
    ``test_pydantic_ai_otf_encode_setup_encoder.py``). The new
    collector must NOT collapse by kb_id — both entries survive with
    distinct ``entity_ref`` + ``kind``."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="active_flag", sql="status='Y'",
                   meta={"kb_id": 5}),
        ],
        measures=[ModelMeasure(
            name="active_share", formula="active_flag:avg",
            meta={"kb_id": 5},
        )],
    ))
    entries = await _collect_peer_kb_entries(storage, DB)
    same_kb = [e for e in entries if e.kb_id == 5]
    assert len(same_kb) == 2, (
        f"both Column + ModelMeasure tagged with kb_id=5 must appear; "
        f"got {len(same_kb)} entries: {same_kb}"
    )
    kinds = sorted(e.kind for e in same_kb)
    assert kinds == ["column", "measure"]
    refs = sorted(e.entity_ref for e in same_kb)
    assert refs == [
        f"{DB}.m.active_flag", f"{DB}.m.active_share",
    ]


async def test_block_empty_when_nothing_tagged(tmp_path):
    """Empty / no-tagged-peers case: renderer returns ``(none)``
    placeholder (matches existing contract)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    assert await _format_existing_kb_tagged_entities_block(storage, DB) == "(none)"


async def test_collector_empty_when_nothing_tagged(tmp_path):
    """Collector returns ``[]`` for a datasource with no kb-tagged
    entities (so callers don't need to special-case `(none)` strings)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _collect_peer_kb_entries,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    assert await _collect_peer_kb_entries(storage, DB) == []
