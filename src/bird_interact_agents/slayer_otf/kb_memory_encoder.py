"""Encode mini-interact KB items into SLayer memories.

Pure function — no I/O. Given the KB rows for one database and the set
of KB ids the task wants omitted, returns a list of dicts ready to
dump into ``memories.yaml``.

Key design decisions (see plan / DEV-1455):

- Memory id ``f"{db}_kb_{kb_id}"`` (string per DEV-1428; KB ids are
  per-DB so the prefix prevents cross-DB collisions).
- ``entities`` always leads with the bare datasource id ``db`` so
  ``SearchService._filter_memories_by_datasource`` keeps the memory
  eligible under ``search(datasource=db)`` (Codex blocker on the
  original plan).
- Cross-references from ``children_knowledge`` are emitted as
  ``memory:<db>_kb_<child>`` tokens, after filtering out deleted ids
  AND ids absent from the KB row set (with a warning for the latter).
- Body starts with ``KB <id> — <knowledge>\\n\\n`` so
  ``hard8_preprocessor._KB_PREFIX_RE`` keeps recognising the marker.
- ``created_at`` is the fixed EPOCH constant so two runs against the
  same input produce byte-identical YAML.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)


#: Fixed timestamp for every encoded memory — determinism, not history.
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _normalise_children(raw: Any) -> list[int]:
    """Normalise the raw ``children_knowledge`` value into a list of
    non-negative ints.

    Accepts ``None`` / missing / ``-1`` / positive ``int`` / ``list[int]``
    per the BIRD-Interact KB shape. Negative values are dropped (the
    corpus uses ``-1`` as the "no children" sentinel; future negatives
    would carry no meaning either).
    """
    if raw is None or raw == -1:
        return []
    if isinstance(raw, int):
        return [raw] if raw >= 0 else []
    if isinstance(raw, list):
        out: list[int] = []
        for x in raw:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if v >= 0:
                out.append(v)
        return out
    return []


def _resolve_cross_refs(
    *,
    db: str,
    source_kb_id: int,
    raw_children: Any,
    all_kb_ids: set[int],
    deleted_kb_ids: set[int],
) -> list[str]:
    """Turn one row's ``children_knowledge`` into ``memory:<db>_kb_<c>``
    tokens, dropping children that are deleted OR not in the kb row set.

    Logs a single ``WARNING`` for each dropped-because-unknown child so
    the drift is visible. Dropped-because-deleted children are silent —
    that case is the deliberate per-task masking path.
    """
    tokens: list[str] = []
    for c in _normalise_children(raw_children):
        if c in deleted_kb_ids:
            continue
        if c not in all_kb_ids:
            logger.warning(
                "kb_memory_encoder: KB %s in db %r references unknown child "
                "KB id %s — dropping the dangling memory:%s_kb_%s token from "
                "encoded entities. Check the source *_kb.jsonl for schema drift.",
                source_kb_id, db, c, db, c,
            )
            continue
        tokens.append(f"memory:{db}_kb_{c}")
    return tokens


def _verbatim_block(db: str, row: dict) -> str:
    """Header + YAML dump of the original KB row. The header advertises
    the source filename so a human reader (or the agent) can trace the
    memory back to its origin."""
    dump = yaml.safe_dump(row, sort_keys=False).rstrip()
    return f"KB item (verbatim from {db}_kb.jsonl):\n{dump}"


def _build_one(
    *,
    db: str,
    row: dict,
    all_kb_ids: set[int],
    deleted_kb_ids: set[int],
) -> dict:
    """Build one memory dict from a KB row. Caller has already filtered
    deleted rows; this function trusts ``row`` is meant to survive."""
    kb_id = int(row["id"])
    knowledge = row.get("knowledge", "")
    learning = (
        f"KB {kb_id} — {knowledge}\n\n{_verbatim_block(db, row)}"
    )
    entities = [db] + _resolve_cross_refs(
        db=db,
        source_kb_id=kb_id,
        raw_children=row.get("children_knowledge"),
        all_kb_ids=all_kb_ids,
        deleted_kb_ids=deleted_kb_ids,
    )
    return {
        "version": 1,
        "id": f"{db}_kb_{kb_id}",
        "learning": learning,
        "description": row.get("description", ""),
        "entities": entities,
        "query": None,
        "created_at": EPOCH,
    }


def encode_kb_as_memories(
    db: str,
    kb_rows: Iterable[dict],
    deleted_kb_ids: set[int],
) -> list[dict]:
    """Encode the KB rows for one DB into SLayer memory dicts.

    Args:
        db: Datasource id (e.g., ``"cybermarket"``). Used as both the
            memory-id prefix and the first entity on each memory.
        kb_rows: Iterable of parsed ``*_kb.jsonl`` rows. Each row must
            carry an ``id`` (int or int-coercible).
        deleted_kb_ids: KB ids the task wants omitted (from the
            ``knowledge_ambiguity[*].deleted_knowledge`` flattening).
            Rows whose id is in this set are skipped entirely; any
            surviving memory's ``entities`` is filtered to remove
            references to deleted ids.

    Returns:
        A list of memory dicts in the same order as ``kb_rows``, ready
        to be dumped as ``memories.yaml``.
    """
    rows = list(kb_rows)
    all_kb_ids = {int(r["id"]) for r in rows}
    out: list[dict] = []
    for row in rows:
        kb_id = int(row["id"])
        if kb_id in deleted_kb_ids:
            continue
        out.append(_build_one(
            db=db, row=row,
            all_kb_ids=all_kb_ids,
            deleted_kb_ids=deleted_kb_ids,
        ))
    return out
