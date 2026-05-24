"""Tests for `pydantic_ai_otf_encode.deps` data shapes.

Framework-only — no LLM, no MCP server, no I/O. Pins the public Pydantic
shapes that the encoder agent's `output_type` enforces and that the
calling sub-clarifier inspects.

The new shapes vs the recursive-adapter baseline:

* `AgentRecord.role` Literal gains "kb_encoder".
* `AgentRecord` gains an optional `kb_id: int | None` field for
  trajectory grouping by KB.
* `SharedTaskState` gains `kb_encoded: list[EncoderResult]` registry
  plus private `_kb_locks` (per-kb `asyncio.Lock`) and `_kb_rows_by_id`
  (lazy-loaded dict).
* New `EncodedEntity` and `EncoderResult` models — the encoder's
  structured output and the per-kb dedup-registry payload.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# EncodedEntity
# ---------------------------------------------------------------------------


def test_encoded_entity_constructs_with_all_kinds():
    """The Literal must cover column / measure / aggregation / model.
    These are the four entity flavours `mcp__slayer__edit_model` /
    `create_model` write. Anything else is a typo."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity,
    )

    for kind in ("column", "measure", "aggregation", "model"):
        e = EncodedEntity(
            kind=kind,
            host_model="households" if kind != "model" else None,
            name="x_score",
            entity_ref="db.households.x_score",
        )
        assert e.kind == kind
        assert e.name == "x_score"


def test_encoded_entity_rejects_unknown_kind():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity,
    )

    with pytest.raises(ValidationError):
        EncodedEntity(
            kind="dimension",  # not a valid kind
            host_model="households",
            name="x",
            entity_ref="db.households.x",
        )


def test_encoded_entity_query_backed_model_has_no_host():
    """A query-backed `model` lives at the datasource root — `host_model`
    is `None` for kind='model'. The plan doesn't require strict
    enforcement; it just needs the field nullable so the encoder can
    return None for query-backed models."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity,
    )

    e = EncodedEntity(
        kind="model",
        host_model=None,
        name="rssi",
        entity_ref="db.rssi",
    )
    assert e.host_model is None


# ---------------------------------------------------------------------------
# EncoderResult
# ---------------------------------------------------------------------------


def test_encoder_result_happy_path():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    r = EncoderResult(
        kb_id=12,
        status="encoded",
        entities=[
            EncodedEntity(
                kind="column", host_model="m", name="c",
                entity_ref="db.m.c",
            ),
        ],
        notes="encoded as a row-level column on m",
    )
    assert r.status == "encoded"
    assert r.kb_id == 12
    assert r.error is None
    assert len(r.entities) == 1


def test_encoder_result_error_path():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    r = EncoderResult(
        kb_id=12,
        status="error",
        entities=[],
        notes="",
        error="budget exhausted",
    )
    assert r.status == "error"
    assert r.error == "budget exhausted"


def test_encoder_result_rejects_unknown_status():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    with pytest.raises(ValidationError):
        EncoderResult(
            kb_id=12, status="skipped", entities=[], notes="",
        )


def test_encoder_result_deferred_status_with_clarifying_questions():
    """DEV-1454 re-plan: the setup encoder (no ask_user) defers ambiguous
    KB items, returning `status='deferred'` plus the clarifying questions
    a later per-task agent must ask. `clarifying_questions` is a list of
    str (not a Dict — global LLM-output rule) and defaults to empty."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    r = EncoderResult(
        kb_id=12,
        status="deferred",
        entities=[],
        notes="threshold for 'high value' is unspecified",
        clarifying_questions=[
            "What dollar amount counts as 'high value'?",
            "Is the threshold inclusive?",
        ],
    )
    assert r.status == "deferred"
    assert r.entities == []
    assert len(r.clarifying_questions) == 2

    # Defaults to empty for the encoded/error paths.
    enc = EncoderResult(kb_id=1, status="encoded", entities=[], notes="")
    assert enc.clarifying_questions == []


# ---------------------------------------------------------------------------
# AgentRecord — role gains "kb_encoder", new optional kb_id
# ---------------------------------------------------------------------------


def test_agent_record_role_accepts_kb_encoder():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        AgentRecord,
    )

    rec = AgentRecord(
        role="kb_encoder",
        depth=2,
        parent_idx=1,
        instruction="encode kb 5",
        kb_id=5,
    )
    assert rec.role == "kb_encoder"
    assert rec.kb_id == 5


def test_agent_record_kb_id_defaults_to_none_for_non_encoder_roles():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        AgentRecord,
    )

    rec = AgentRecord(
        role="sub_clarifier",
        depth=1,
        parent_idx=0,
        focus="x",
        instruction="...",
    )
    assert rec.kb_id is None


def test_agent_record_role_keeps_recursive_adapter_roles():
    """All four legacy roles still valid."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        AgentRecord,
    )

    for role in (
        "root_clarifier", "sub_clarifier",
        "projection_resolver", "query_constructor",
    ):
        rec = AgentRecord(role=role, depth=0, instruction="x")
        assert rec.role == role


# ---------------------------------------------------------------------------
# SharedTaskState — new fields + lock map + lazy KB cache
# ---------------------------------------------------------------------------


def _make_shared(remaining_budget: float = 100.0):
    import tempfile

    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        SharedTaskState,
    )
    from bird_interact_agents.harness import SampleStatus

    status = SampleStatus(
        idx=0,
        original_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "?",
            "knowledge_ambiguity": [],
        },
        remaining_budget=remaining_budget,
        total_budget=remaining_budget,
    )
    return SharedTaskState(
        status=status,
        # data_path_base is an inert placeholder in these unit tests; use the
        # platform temp dir rather than a hardcoded /tmp (portability + lint).
        data_path_base=tempfile.gettempdir(),
        db_name="fake_db",
        amb_user_query="?",
        slayer_storage_dir="",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )


def test_shared_task_state_starts_with_empty_kb_encoded():
    shared = _make_shared()
    assert shared.kb_encoded == []


def test_shared_task_state_kb_locks_are_lazily_per_kb():
    """`_kb_locks` is a private dict — the lock factory creates one
    `asyncio.Lock` per kb_id on demand, and returns the same lock on
    repeat lookups so two consumers serialise on it."""
    shared = _make_shared()
    # The private attr is created empty:
    assert shared._kb_locks == {}

    # setdefault on the dict is the canonical access pattern:
    lock_a = shared._kb_locks.setdefault(5, asyncio.Lock())
    lock_b = shared._kb_locks.setdefault(5, asyncio.Lock())
    # Same key returns the same instance:
    assert lock_a is lock_b


def test_shared_task_state_kb_rows_by_id_starts_unset():
    """`_kb_rows_by_id` is lazily populated on the first `kb_to_slayer`
    call. None == "not loaded yet"; an empty dict would be ambiguous
    with 'loaded but DB has no KBs'."""
    shared = _make_shared()
    assert shared._kb_rows_by_id is None


def test_shared_task_state_kb_encoded_holds_full_encoder_result():
    """Per Codex finding 8: the registry stores the FULL EncoderResult
    (including notes + error), not just the entities list. This is what
    lets cached failures avoid retries and lets notes survive into
    later kb_to_slayer calls."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    shared = _make_shared()
    shared.kb_encoded.append(EncoderResult(
        kb_id=7, status="encoded",
        entities=[EncodedEntity(
            kind="measure", host_model="orders",
            name="rev", entity_ref="db.orders.rev",
        )],
        notes="encoded as a named SUM measure",
    ))
    shared.kb_encoded.append(EncoderResult(
        kb_id=8, status="error", entities=[], notes="",
        error="budget exhausted",
    ))
    assert len(shared.kb_encoded) == 2
    assert shared.kb_encoded[0].kb_id == 7
    assert shared.kb_encoded[1].error == "budget exhausted"


# ---------------------------------------------------------------------------
# _LegacyAdapter — same routing as the recursive adapter (no new fields)
# ---------------------------------------------------------------------------


def test_legacy_adapter_routes_shared_and_per_agent_attrs():
    """Parity with `test_pydantic_ai_recursive_tools.py`'s adapter test —
    the sibling deps module must keep `_LegacyAdapter` working for
    `_submit.*` duck-typing on the existing helpers
    (`ask_user_impl`, `submit_slayer_query`)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        TaskDeps, _LegacyAdapter,
    )

    shared = _make_shared()
    deps = TaskDeps(shared=shared, depth=0, max_depth=3)
    adapter = _LegacyAdapter(deps)

    # Shared fields:
    assert adapter.status is shared.status
    assert adapter.data_path_base == shared.data_path_base
    assert adapter.user_sim_model == shared.user_sim_model
    assert adapter.user_sim_prompt_version == shared.user_sim_prompt_version
    assert adapter.slayer_storage_dir == shared.slayer_storage_dir

    # Per-agent:
    assert adapter.user_sim_transcript is deps.user_sim_transcript
    assert adapter.usage is deps.usage

    # Write-paths to shared:
    payload = {"phase1_passed": True}
    adapter.result = payload
    assert shared.submitter_result == payload
    sentinel = object()
    adapter._slayer_client = sentinel
    assert shared._slayer_client is sentinel
