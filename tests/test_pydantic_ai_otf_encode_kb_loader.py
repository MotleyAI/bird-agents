"""Tests for `kb_to_slayer`'s KB-content source (DEV-1454 re-plan).

The YAML-re-parse hack (`_parse_kb_row_from_memory` / `_ensure_kb_rows_loaded`)
is gone. `kb_to_slayer` now sources each KB through the SLayer storage memory
API — `storage.get_memory(f"{db}_kb_{n}")` for content and `Memory.entities`
for dependency edges — and short-circuits ids the setup pass already encoded.

These tests pin the observable behaviour (not the internal arg names): the
encoder stub accepts `**kwargs` so it's robust to the exact runner seam.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext

from slayer.core.models import Column, DatasourceConfig, SlayerModel
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)
from bird_interact_agents.memory_store_io import write_memories_files


DB = "tinydb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kb(kb_id: int, **ov: Any) -> dict[str, Any]:
    row = {
        "id": kb_id, "knowledge": f"KB {kb_id}", "description": "d",
        "definition": "x", "type": "calc", "children_knowledge": -1,
    }
    row.update(ov)
    return row


async def _seed(
    tmp_path: Path, kb_rows: list[dict], *, models: list[SlayerModel] | None = None,
    extra_memory_entities: dict[int, list[str]] | None = None,
) -> YAMLStorage:
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    for m in (models or [SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[Column(name="id", primary_key=True)],
    )]):
        await storage.save_model(m)

    mems = encode_kb_as_memories(DB, kb_rows, deleted_kb_ids=set())
    # Simulate the setup pass having appended concrete entity refs to a KB's
    # own memory entities list (what `_annotate_memories` does for encoded KBs).
    if extra_memory_entities:
        by_id = {m["id"]: m for m in mems}
        for kb_id, refs in extra_memory_entities.items():
            by_id[f"{DB}_kb_{kb_id}"]["entities"].extend(refs)
    write_memories_files(tmp_path, mems)
    return storage


def _shared(tmp_path: Path, storage: YAMLStorage):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        SharedTaskState,
    )
    from bird_interact_agents.harness import SampleStatus

    status = SampleStatus(
        idx=0,
        original_data={"selected_database": DB, "instance_id": "t",
                       "amb_user_query": "?", "knowledge_ambiguity": []},
        remaining_budget=100.0, total_budget=100.0,
    )
    shared = SharedTaskState(
        status=status, data_path_base="/tmp", db_name=DB, amb_user_query="?",
        slayer_storage_dir=str(tmp_path),
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    shared._slayer_storage = storage
    return shared


def _ctx(shared, idx=0):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import TaskDeps
    deps = TaskDeps(shared=shared, depth=1, max_depth=3, self_record_idx=idx)
    return RunContext(deps=deps, model=None, usage=None, prompt="", run_step=0)


def _agent_with_stub(stub):
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import TaskDeps

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    factories._register_kb_to_slayer(
        agent, model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
        _encoder_runner=stub,
    )
    return dict(agent._function_toolset.tools)["kb_to_slayer"]


def _encoding_stub(record, *, writes=True):
    """Stub runner that (optionally) writes a tagged column for the KB and
    returns `encoded`. Accepts `**kwargs` to be robust to the runner seam."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    async def stub(**kw):
        kb_id = kw["kb_id"]
        record.append(kb_id)
        ctx = kw.get("ctx")
        name = f"kb{kb_id}_col"
        if writes and ctx is not None:
            storage = ctx.deps.shared._slayer_storage
            model = await storage.get_model("households")
            model.columns.append(
                Column(name=name, sql="1", meta={"kb_id": kb_id}),
            )
            await storage.save_model(model)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(kind="column", host_model="households",
                                    name=name, entity_ref=f"{DB}.households.{name}")],
            notes="",
        )
    return stub


# ---------------------------------------------------------------------------
# Memory-API content source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_present_memory_is_encoded(tmp_path):
    """A KB whose memory exists in storage is dispatched to the encoder."""
    storage = await _seed(tmp_path, [_kb(11)])
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    out = json.loads(await tool.function(_ctx(_shared(tmp_path, storage)),
                                         kb_ids=[11]))
    assert record == [11]
    assert out["11"]["status"] == "encoded"


@pytest.mark.asyncio
async def test_absent_deleted_memory_surfaces_per_id_error(tmp_path):
    """HARD-8 deletes a KB by dropping its memory. `kb_to_slayer` must read
    via `get_memory`, find it absent, and surface a per-id error WITHOUT
    crashing or calling the encoder for it."""
    storage = await _seed(tmp_path, [_kb(11)])  # only kb 11 exists
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    out = json.loads(await tool.function(_ctx(_shared(tmp_path, storage)),
                                         kb_ids=[99]))  # 99 was "deleted"
    assert 99 not in record
    assert out["99"]["status"] == "error"


@pytest.mark.asyncio
async def test_dep_edges_come_from_memory_entities(tmp_path):
    """KB 2's memory carries a `memory:tinydb_kb_1` cross-ref (written by
    DEV-1455 from children_knowledge). The dep walk reads `Memory.entities`,
    so requesting [2] encodes 1 before 2."""
    rows = [_kb(1), _kb(2, children_knowledge=[1])]
    storage = await _seed(tmp_path, rows)
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    await tool.function(_ctx(_shared(tmp_path, storage)), kb_ids=[2])
    assert record.index(1) < record.index(2)


# ---------------------------------------------------------------------------
# Short-circuit on setup-encoded ids (Codex #8: only when complete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_circuit_when_setup_entities_present(tmp_path):
    """The setup pass already encoded KB 11 (a tagged column exists AND its
    memory records the entity ref). `kb_to_slayer` must reuse it and NOT
    invoke the task encoder."""
    model = SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="kb11_col", sql="1", meta={"kb_id": 11}),
        ],
    )
    storage = await _seed(
        tmp_path, [_kb(11)], models=[model],
        extra_memory_entities={11: [f"{DB}.households.kb11_col"]},
    )
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    out = json.loads(await tool.function(_ctx(_shared(tmp_path, storage)),
                                         kb_ids=[11]))
    assert record == [], "encoder must NOT run for a setup-encoded id"
    assert out["11"]["status"] == "encoded"
    assert out["11"]["entities"][0]["entity_ref"] == f"{DB}.households.kb11_col"


@pytest.mark.asyncio
async def test_no_short_circuit_when_recorded_entity_missing(tmp_path):
    """Memory records an entity ref, but the entity is absent from storage
    (partial / deleted companion) — `kb_to_slayer` must RE-ENCODE, not
    falsely short-circuit."""
    storage = await _seed(
        tmp_path, [_kb(11)],
        extra_memory_entities={11: [f"{DB}.households.kb11_col"]},  # ref recorded
    )  # ...but no such column exists in the (default) model
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    await tool.function(_ctx(_shared(tmp_path, storage)), kb_ids=[11])
    assert record == [11], "must re-encode when a recorded entity is missing"


@pytest.mark.asyncio
async def test_deferred_memory_without_refs_is_encoded(tmp_path):
    """A deferred KB memory has no recorded entity refs → it is dispatched
    to the (task-time) encoder rather than short-circuited."""
    storage = await _seed(tmp_path, [_kb(12)])  # plain memory, no refs
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    await tool.function(_ctx(_shared(tmp_path, storage)), kb_ids=[12])
    assert record == [12]


@pytest.mark.asyncio
async def test_memory_without_verbatim_block_still_encodes(tmp_path):
    """Codex test-review #3: the content passed to the encoder is the memory
    `learning` text itself — there's no YAML verbatim block to parse — so a
    memory whose body lacks the DEV-1455 `KB item (verbatim ...)` block must
    still encode fine."""
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[Column(name="id", primary_key=True)],
    ))
    # A memory with NO verbatim YAML block — just prose.
    write_memories_files(tmp_path, [{
        "version": 1, "id": f"{DB}_kb_11",
        "learning": "KB 11 — high-income households (prose only, no block)",
        "entities": [DB], "query": None,
    }])

    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    out = json.loads(await tool.function(_ctx(_shared(tmp_path, storage)),
                                         kb_ids=[11]))
    assert record == [11]
    assert out["11"]["status"] == "encoded"


@pytest.mark.asyncio
async def test_no_short_circuit_when_recorded_set_partial(tmp_path):
    """Codex test-review #4: the memory records TWO entity refs but only ONE
    exists in storage → short-circuit must NOT fire (incomplete set), the KB
    is re-encoded."""
    model = SlayerModel(
        name="households", data_source=DB, sql_table="households",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="kb11_a", sql="1", meta={"kb_id": 11}),  # only this exists
        ],
    )
    storage = await _seed(
        tmp_path, [_kb(11)], models=[model],
        extra_memory_entities={11: [f"{DB}.households.kb11_a",
                                    f"{DB}.households.kb11_b"]},  # b is missing
    )
    record: list[int] = []
    tool = _agent_with_stub(_encoding_stub(record))
    await tool.function(_ctx(_shared(tmp_path, storage)), kb_ids=[11])
    assert record == [11], "partial recorded set must re-encode, not short-circuit"
