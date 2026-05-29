"""On-the-fly SLayer setup for DEV-1455.

Per-task setup that ingests the relevant SQLite DB into SLayer
(deterministic phases 1-3 of the slayer_pipeline orchestrator) and
encodes the DB's KB items as SLayer memories, preserving the
``children_knowledge`` cross-reference graph as ``memory:<id>``
entity tokens. Replaces the LLM-driven ``kb-to-slayer-models`` skill
at runtime; the trade-off is documented in the plan.

Public surface:

- :func:`kb_memory_encoder.encode_kb_as_memories` — pure function.
- :func:`cache.ensure_db_cache` — runs phases 1-3 into a fingerprinted
  per-DB cache; idempotent and concurrency-safe.
- :func:`runtime.prepare_task_storage` — copies cache into a per-task
  scratch dir and writes ``memories.yaml`` from the encoder output.
"""

from bird_interact_agents.slayer_otf.cache import (
    CacheEntry,
    ensure_db_cache,
    fingerprint_of,
)
from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    EPOCH,
    encode_kb_as_memories,
)
from bird_interact_agents.slayer_otf.reference_build import (
    ReferenceEntry,
    ensure_db_reference,
)
from bird_interact_agents.slayer_otf.runtime import (
    prepare_task_storage,
    resolve_otf_task_storage_dir,
)

__all__ = [
    "CacheEntry",
    "EPOCH",
    "ReferenceEntry",
    "encode_kb_as_memories",
    "ensure_db_cache",
    "ensure_db_reference",
    "fingerprint_of",
    "prepare_task_storage",
    "resolve_otf_task_storage_dir",
]
