"""Codex-finding-9 test: storage-verified `kb_to_slayer` smoke.

Sets up a real `YAMLStorage` rooted at a tmp dir, seeds:
- a tiny datasource pointing at a synthetic in-memory model (no
  SQLite — `YAMLStorage` happily holds models without a working
  connection_string for this test's purpose),
- a couple of KB memories via `encode_kb_as_memories`,

then invokes `kb_to_slayer` with a stub encoder that writes via direct
`storage.save_model` calls (mirroring what `mcp__slayer__edit_model`
would do). Asserts that:

* The resulting model carries the newly-added column.
* The post-encode verification step actually finds it (i.e., the
  pipeline ties together).
* When the stub encoder writes a column with a DIFFERENT name than it
  claims in `entity_ref`, the verification flags the mismatch.

This is the test Codex specifically asked for (#9): not just mocked
flow, but real storage round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_ai import RunContext

from slayer.core.models import (
    Aggregation,
    AggregationParam,
    Column,
    DatasourceConfig,
    ModelMeasure,
    SlayerModel,
)
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)


DB = "tinydb"


# ---------------------------------------------------------------------------
# Storage scaffolding
# ---------------------------------------------------------------------------


async def _seed_storage(
    tmp_path: Path, kb_rows: list[dict],
) -> YAMLStorage:
    """Seed a YAMLStorage with: one datasource, one model
    (`households`) with a single primary key column, and the encoded
    KB memories from `kb_rows`."""
    storage = YAMLStorage(base_dir=str(tmp_path))
    ds = DatasourceConfig(
        name=DB, type="sqlite",
        connection_string="sqlite:///not-used.sqlite",
    )
    await storage.save_datasource(ds)

    m = SlayerModel(
        name="households",
        data_source=DB,
        sql_table="households",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="income"),
            Column(name="tier"),
        ],
        measures=[],
        joins=[],
    )
    await storage.save_model(m)

    mems = encode_kb_as_memories(DB, kb_rows, deleted_kb_ids=set())
    (tmp_path / "memories.yaml").write_text(
        yaml.safe_dump(mems, sort_keys=False),
    )
    return storage


def _kb(kb_id: int, **overrides: Any) -> dict:
    row = {
        "id": kb_id,
        "knowledge": f"KB {kb_id} title",
        "description": "d", "definition": "x", "type": "calc",
        "children_knowledge": -1,
    }
    row.update(overrides)
    return row


def _shared_with_storage(tmp_path: Path, storage: YAMLStorage):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        SharedTaskState,
    )
    from bird_interact_agents.harness import SampleStatus

    status = SampleStatus(
        idx=0,
        original_data={
            "selected_database": DB, "instance_id": "t",
            "amb_user_query": "?", "knowledge_ambiguity": [],
        },
        remaining_budget=100.0, total_budget=100.0,
    )
    shared = SharedTaskState(
        status=status, data_path_base="/tmp",
        db_name=DB, amb_user_query="?",
        slayer_storage_dir=str(tmp_path),
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    shared._slayer_storage = storage
    return shared


def _deps(shared):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        TaskDeps,
    )

    return TaskDeps(shared=shared, depth=1, max_depth=3, self_record_idx=0)


def _ctx(deps):
    return RunContext(
        deps=deps, model=None, usage=None, prompt="", run_step=0,
    )


def _build_sub_clarifier_with_stub_encoder(stub_encoder):
    """Bare pydantic-ai Agent with only `kb_to_slayer` registered to a
    stub encoder runner. Mirror of the helper in
    test_pydantic_ai_otf_encode_kb_to_slayer.py."""
    from pydantic_ai import Agent
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        TaskDeps,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    factories._register_kb_to_slayer(
        agent,
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
        _encoder_runner=stub_encoder,
    )
    return agent


def _tool(agent, name):
    return dict(agent._function_toolset.tools)[name]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_storage_round_trip_encoder_writes_column_visible_in_model(
    tmp_path,
):
    """A stub encoder writes a new Column on `households` via the
    storage layer; the post-encode verification step inspects the
    model and confirms the column exists; `kb_to_slayer` returns
    `status='encoded'`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(
        tmp_path,
        kb_rows=[_kb(11, knowledge="High-income flag",
                     definition="income > 50000")],
    )
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        # Mimic what the real encoder would do via mcp__slayer__edit_model:
        # add a new Column to `households` carrying the KB id in meta.
        model = await storage.get_model("households")
        model.columns.append(Column(
            name="is_high_income",
            sql="CASE WHEN income > 50000 THEN 1 ELSE 0 END",
            meta={"kb_id": kb_id},
        ))
        await storage.save_model(model)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="households",
                name="is_high_income",
                entity_ref=f"{DB}.households.is_high_income",
            )],
            notes="encoded as a row-level CASE on households",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[11],
    )
    payload = json.loads(out)

    # Tool returned 'encoded' — the verification step confirmed the
    # entity exists in storage.
    assert payload["11"]["status"] == "encoded"
    assert payload["11"]["entities"][0]["name"] == "is_high_income"

    # Storage actually holds the new column.
    refreshed = await storage.get_model("households")
    col_names = {c.name for c in refreshed.columns}
    assert "is_high_income" in col_names


@pytest.mark.asyncio
async def test_storage_verification_catches_encoder_lying_about_entity_ref(
    tmp_path,
):
    """Codex finding 3 — exercised against real storage. The stub
    encoder writes a column named `foo` but CLAIMS to have written
    `bar`. The verification step inspects the model, sees no `bar`,
    and downgrades to `status='error'`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(tmp_path, kb_rows=[_kb(12)])
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        # Encoder ACTUALLY writes `foo`:
        model = await storage.get_model("households")
        model.columns.append(Column(name="foo", sql="1"))
        await storage.save_model(model)
        # But CLAIMS it wrote `bar`:
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="households",
                name="bar", entity_ref=f"{DB}.households.bar",
            )],
            notes="lied about entity ref",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[12],
    )
    payload = json.loads(out)

    assert payload["12"]["status"] == "error"


@pytest.mark.asyncio
async def test_storage_round_trip_aggregation_kind(tmp_path):
    """Codex test-review finding 5: verification path must work for
    `kind='aggregation'` too — the encoder claims a parameterized
    Aggregation was created on the host model."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(tmp_path, kb_rows=[_kb(15)])
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        model = await storage.get_model("households")
        model.aggregations.append(Aggregation(
            name="weighted_avg",
            formula="SUM({sql} * {w}) / NULLIF(SUM({w}), 0)",
            params=[AggregationParam(name="w", sql="income")],
            meta={"kb_id": kb_id},
        ))
        await storage.save_model(model)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="aggregation", host_model="households",
                name="weighted_avg",
                entity_ref=f"{DB}.households.weighted_avg",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[15],
    )
    payload = json.loads(out)
    assert payload["15"]["status"] == "encoded"


@pytest.mark.asyncio
async def test_storage_verification_catches_missing_aggregation(tmp_path):
    """If the encoder claims an aggregation that wasn't actually
    written, verification downgrades to error."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(tmp_path, kb_rows=[_kb(16)])
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        # Encoder writes NOTHING but claims an aggregation:
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="aggregation", host_model="households",
                name="ghost_agg",
                entity_ref=f"{DB}.households.ghost_agg",
            )],
            notes="lied",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[16],
    )
    payload = json.loads(out)
    assert payload["16"]["status"] == "error"


@pytest.mark.asyncio
async def test_storage_round_trip_query_backed_model_kind(tmp_path):
    """Codex test-review finding 5: verification works for the
    query-backed-model kind. `host_model=None` because the model
    lives at the datasource root."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(tmp_path, kb_rows=[_kb(17)])
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        # Mimic `mcp__slayer__create_model` via storage.save_model
        # with `source_queries=[...]` (one stage).
        from slayer.core.models import SlayerModel
        qb_model = SlayerModel(
            name="rssi", data_source=DB,
            source_queries=[{
                "source_model": "households",
                "dimensions": ["id"],
                "measures": [{"formula": "income:avg", "name": "avg_income"}],
            }],
            columns=[],
            meta={"kb_id": kb_id},
        )
        await storage.save_model(qb_model)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="model", host_model=None,
                name="rssi", entity_ref=f"{DB}.rssi",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[17],
    )
    payload = json.loads(out)
    assert payload["17"]["status"] == "encoded"


@pytest.mark.asyncio
async def test_storage_verification_catches_missing_query_backed_model(
    tmp_path,
):
    """Same shape as the missing-column / missing-measure tests, for
    `kind='model'`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(tmp_path, kb_rows=[_kb(18)])
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="model", host_model=None,
                name="ghost_model", entity_ref=f"{DB}.ghost_model",
            )],
            notes="lied",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[18],
    )
    payload = json.loads(out)
    assert payload["18"]["status"] == "error"


@pytest.mark.asyncio
async def test_storage_round_trip_measure_kind(tmp_path):
    """Same verification path for `kind='measure'` (the encoder claims
    a named ModelMeasure was created)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    storage = await _seed_storage(tmp_path, kb_rows=[_kb(13)])
    shared = _shared_with_storage(tmp_path, storage)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        model = await storage.get_model("households")
        model.measures.append(ModelMeasure(
            name="total_income", formula="income:sum",
            meta={"kb_id": kb_id},
        ))
        await storage.save_model(model)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="measure", host_model="households",
                name="total_income",
                entity_ref=f"{DB}.households.total_income",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[13],
    )
    payload = json.loads(out)
    assert payload["13"]["status"] == "encoded"
    refreshed = await storage.get_model("households")
    assert any(m.name == "total_income" for m in refreshed.measures)
