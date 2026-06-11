"""Tests for the setup-pass memory annotation
``slayer_otf.reference_build._annotate_memories`` (DEV-1454).

After the setup encoder runs, every KB's pre-loaded memory is re-saved (via
the memory *service* so the embedding hook fires + the ``<db>_kb_<n>`` id and
``created_at`` are preserved) with a per-KB status section appended:

* encoded → ``--- setup-encode: encoded ---`` + the entity refs, AND each
  concrete entity ref added to the memory's ``entities`` list (only on its OWN
  memory);
* deferred → ``--- setup-encode: deferred ---`` + reasons + clarifying
  questions.

``hard8_preprocessor._memory_kb_id`` must still parse the ``KB <n> —`` header
afterwards (so HARD-8 masking keeps working).
"""

from __future__ import annotations

from slayer.core.models import Column, DatasourceConfig, SlayerModel
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)


DB = "tinydb"


def _kb(kb_id, **ov):
    row = {
        "id": kb_id, "knowledge": f"KB {kb_id} title", "description": "d",
        "definition": "x", "type": "calc", "children_knowledge": -1,
    }
    row.update(ov)
    return row


async def _seed(tmp_path, kb_rows, *, with_encoded_col_for=None):
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    cols = [Column(name="id", primary_key=True)]
    if with_encoded_col_for is not None:
        cols.append(Column(
            name=f"kb{with_encoded_col_for}_col", sql="1",
            meta={"kb_id": with_encoded_col_for},
        ))
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households", columns=cols,
    ))
    import yaml
    mems = encode_kb_as_memories(DB, kb_rows, deleted_kb_ids=set())
    (tmp_path / "memories.yaml").write_text(yaml.safe_dump(mems, sort_keys=False))
    return storage


async def test_encoded_memory_gets_entity_ref_in_body_and_entities(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.slayer_otf.reference_build import (
        _annotate_memories,
    )
    from bird_interact_agents.hard8_preprocessor import _memory_kb_id

    rows = [_kb(11, knowledge="High-income flag")]
    storage = await _seed(tmp_path, rows, with_encoded_col_for=11)
    ref = f"{DB}.households.kb11_col"
    results = [EncoderResult(
        kb_id=11, status="encoded",
        entities=[EncodedEntity(
            kind="column", host_model="households", name="kb11_col",
            entity_ref=ref,
        )],
        notes="row-level CASE",
    )]

    await _annotate_memories(storage=storage, db=DB, setup_results=results,
                             kb_rows=rows)

    mem = await storage.get_memory(f"{DB}_kb_11")
    assert "--- setup-encode: encoded ---" in mem.learning
    assert ref in mem.learning
    assert ref in mem.entities, "entity ref must be added to entities list"
    # HARD-8 header still parses.
    assert _memory_kb_id(mem.learning) == 11
    # The original DEV-1455 verbatim block survives ahead of the appended
    # setup section (annotation appends, never rewrites).
    assert "KB item (verbatim" in mem.learning
    assert mem.learning.index("KB item (verbatim") < mem.learning.index(
        "--- setup-encode:"
    )
    # id preserved.
    assert mem.id == f"{DB}_kb_11"


async def test_deferred_memory_gets_reasons_and_clarifying_questions(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.slayer_otf.reference_build import (
        _annotate_memories,
    )

    rows = [_kb(12, knowledge="High value customer")]
    storage = await _seed(tmp_path, rows)
    results = [EncoderResult(
        kb_id=12, status="deferred", entities=[],
        notes="threshold for 'high value' unspecified",
        clarifying_questions=["What amount counts as high value?"],
    )]

    await _annotate_memories(storage=storage, db=DB, setup_results=results,
                             kb_rows=rows)

    mem = await storage.get_memory(f"{DB}_kb_12")
    assert "--- setup-encode: deferred ---" in mem.learning
    assert "high value" in mem.learning.lower()
    assert "What amount counts as high value?" in mem.learning


async def test_created_at_preserved_across_annotation(tmp_path):
    """Re-saving via the service with an explicit id must preserve the
    original ``created_at`` (upsert semantics), not stamp a new one."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.slayer_otf.reference_build import (
        _annotate_memories,
    )

    rows = [_kb(13)]
    storage = await _seed(tmp_path, rows)
    before = await storage.get_memory(f"{DB}_kb_13")
    created_before = before.created_at

    await _annotate_memories(
        storage=storage, db=DB,
        setup_results=[EncoderResult(kb_id=13, status="deferred", entities=[],
                                     notes="x", clarifying_questions=["q"])],
        kb_rows=rows,
    )
    after = await storage.get_memory(f"{DB}_kb_13")
    assert after.created_at == created_before


async def test_annotation_fires_embedding_hook_when_available(tmp_path, monkeypatch):
    """When embeddings ARE available, re-saving the memory must fan through
    SLayer's write-side upsert hook (so ``embeddings.db`` populates as a
    side effect of the write — no explicit refresh pass). We spy on
    ``SearchService.upsert_memory`` so no real OpenAI call is made.

    DEV-1546 / DEV-1550 moved this hook from the (now-removed)
    ``slayer.embeddings.service.EmbeddingService.refresh_memory`` to
    ``slayer.search.service.SearchService.upsert_memory`` (the higher-
    level fan-out hook that ``MemoryService.save_memory`` uses internally)
    to match the SLayer 0.7.3 consolidation."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.slayer_otf.reference_build import (
        _annotate_memories,
    )
    from slayer.embeddings import client as emb_client
    from slayer.search.service import SearchService

    monkeypatch.setattr(emb_client, "is_available", lambda: True)

    calls: list[str] = []

    async def fake_upsert_memory(self, memory):
        calls.append(memory.id)
        return []

    monkeypatch.setattr(
        SearchService, "upsert_memory", fake_upsert_memory,
    )

    rows = [_kb(31)]
    storage = await _seed(tmp_path, rows)
    await _annotate_memories(
        storage=storage, db=DB,
        setup_results=[EncoderResult(kb_id=31, status="deferred", entities=[],
                                     notes="amb", clarifying_questions=["q"])],
        kb_rows=rows,
    )
    assert f"{DB}_kb_31" in calls, "annotation must go through the upsert hook"


async def test_concrete_entity_ref_only_on_its_own_memory(tmp_path):
    """A KB's concrete entity ref must land ONLY on its own memory — never
    on a sibling's — so HARD-8 deletion (which deletes the KB's own memory)
    never leaves a stale concrete ref on a survivor (Codex #6)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.slayer_otf.reference_build import (
        _annotate_memories,
    )

    rows = [_kb(21, knowledge="A"), _kb(22, knowledge="B")]
    storage = await _seed(tmp_path, rows, with_encoded_col_for=21)
    ref = f"{DB}.households.kb21_col"
    results = [
        EncoderResult(
            kb_id=21, status="encoded",
            entities=[EncodedEntity(kind="column", host_model="households",
                                    name="kb21_col", entity_ref=ref)],
            notes="",
        ),
        EncoderResult(kb_id=22, status="deferred", entities=[], notes="amb",
                      clarifying_questions=["q"]),
    ]
    await _annotate_memories(storage=storage, db=DB, setup_results=results,
                             kb_rows=rows)

    own = await storage.get_memory(f"{DB}_kb_21")
    other = await storage.get_memory(f"{DB}_kb_22")
    assert ref in own.entities
    assert ref not in other.entities, "ref must not leak onto a sibling memory"


async def test_annotate_memories_preserves_description(tmp_path):
    """DEV-1550 A2.2 regression gate: ``_annotate_memories`` re-saves the
    memory through ``storage.save_memory(...)`` whose ``description`` kwarg
    defaults to ``None``. If the re-save doesn't explicitly forward
    ``mem.description``, the field A2.1 just populated gets clobbered and
    the compact renderer falls back to first-paragraph-of-``learning`` on
    every annotated memory — defeating the whole point of the upgrade.
    """
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.slayer_otf.reference_build import (
        _annotate_memories,
    )

    probe = "probe-marker-description-summary"
    rows = [_kb(41, knowledge="K", description=probe)]
    storage = await _seed(tmp_path, rows)

    before = await storage.get_memory(f"{DB}_kb_41")
    assert before.description == probe, (
        "precondition: A2.1 plumbing must already populate description"
    )

    await _annotate_memories(
        storage=storage, db=DB,
        setup_results=[EncoderResult(kb_id=41, status="deferred", entities=[],
                                     notes="x", clarifying_questions=["q"])],
        kb_rows=rows,
    )

    after = await storage.get_memory(f"{DB}_kb_41")
    assert after.description == probe, (
        "_annotate_memories must forward description=mem.description to "
        "storage.save_memory; otherwise the compact-mode summary is lost."
    )
