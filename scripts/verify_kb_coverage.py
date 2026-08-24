#!/usr/bin/env python3
"""Verify every KB entry for a mini-interact DB is either encoded in the
exported SLayer YAML models or documented as a SLayer memory.

Usage:
    python scripts/verify_kb_coverage.py --db <db>
    python scripts/verify_kb_coverage.py --all

Exit 0 only when every KB id for the DB is in **exactly one** of:

  - the *encoded* set: any ``meta.kb_id`` reachable through the YAML
    models in ``bird-interact-agents/slayer_models/<db>/`` (walks
    ``SlayerModel.meta``, every ``Column.meta``, ``ModelMeasure.meta``,
    and ``Aggregation.meta``).
  - the *documented* set: a SLayer memory in the per-DB YAMLStorage at
    ``slayer_models/<db>/`` whose ``learning`` body's first line is
    ``KB <id> — `` (em-dash) and whose ``entities`` (linked_entities)
    list contains at least one ref starting with ``<db>.``.

An id appearing in neither set, or in both, fails the check.

The documented-set check searches via ``SearchService`` (BM25 + tantivy
+ optional dense embeddings) per KB id. DEV-1546 (slayer 0.7.2)
collapsed the per-kind caps into a single ``max_results`` on the
RRF-fused unified ``results`` list; DEV-1549 (slayer 0.7.3) added the
``compact`` flag and the ``cypher_filter`` kind-pin. We use
``cypher_filter='MATCH (n:Memory) RETURN n.id AS id'`` so the unified
``max_results=MAX_MEMORIES_PER_KB`` cap is genuinely a per-kb memory
cap (without the filter, entity hits ranked higher than memories crowd
real memories out and the verifier falsely reports documented KBs as
unaccounted), and ``compact=False`` so ``hit.text`` carries the full
``learning`` body the ``KB_MEMORY_HEAD_RE`` regex parses (compact mode
returns the empty string in ``text`` and surfaces only the one-line
``description``). The per-DB corpus is expected to stay well under
``MAX_MEMORIES_PER_KB``; if a DB's memory corpus ever grows past a few
hundred, bump ``MAX_MEMORIES_PER_KB`` or fall back to reading
``slayer_models/<db>/memories.yaml`` directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from slayer.search.service import SearchService
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents import paths

SLAYER_MODELS_DIR = paths.slayer_models_root()
DEFAULT_MINI_INTERACT_ROOT = paths.benchmark_data_root("mini-interact")

# Compiled at module load: first non-blank line of a deferred-KB memory
# must match this shape exactly. The id capture group is load-bearing.
KB_MEMORY_HEAD_RE = re.compile(r"^KB (\d+) — ")

# Bounded over-fetch ceiling per KB id when searching for documenting
# memories. See module docstring for the rationale.
MAX_MEMORIES_PER_KB = 5


def _kb_path(mini_interact_root: Path, db: str) -> Path:
    return mini_interact_root / db / f"{db}_kb.jsonl"


def load_kb_ids(kb_path: Path) -> set[int]:
    ids: set[int] = set()
    with kb_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(int(json.loads(line)["id"]))
    return ids


def _meta_kb_ids(meta: dict | None) -> set[int]:
    """Pull every KB id stamped on an entity's ``meta``.

    Supports both ``meta.kb_id`` (singular, for a 1:1 entity↔KB mapping)
    and ``meta.kb_ids`` (list, for entities that cover multiple KB
    entries — e.g. a JSON column that aggregates several
    value_illustration descriptions). Both forms can coexist on the
    same entity; ids are unioned.
    """
    if not meta:
        return set()
    out: set[int] = set()
    single = meta.get("kb_id")
    if single is not None:
        out.add(int(single))
    multi = meta.get("kb_ids")
    if multi:
        out.update(int(x) for x in multi)
    return out


async def load_encoded_ids(db_dir: Path) -> set[int]:
    """Load every KB id stamped on any entity in the YAML storage at *db_dir*.

    *db_dir* is a YAMLStorage ``base_dir``: it contains ``models/`` and
    ``datasources/`` sub-directories, populated by the export step in
    the translate-mini-interact-kb skill.
    """
    storage = YAMLStorage(base_dir=str(db_dir))
    ids: set[int] = set()
    for name in await storage.list_models():
        model = await storage.get_model(name)
        if model is None:
            continue
        for owner in (model, *model.columns, *model.measures, *model.aggregations):
            ids |= _meta_kb_ids(getattr(owner, "meta", None))
    return ids


def _first_nonblank_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


async def load_documented_ids(
    *,
    db: str,
    db_dir: Path,
    kb_ids: Iterable[int],
    knowledge: dict[int, str],
) -> set[int]:
    """Return KB ids that a deferred-KB memory in *db_dir* attests to.

    For each KB id, we hit ``SearchService.search(question=...)`` with a
    question built from the KB id + its ``knowledge`` text, then filter
    hits to those whose memory body's first line is exactly
    ``KB <id> — `` AND whose linked entities (``Memory.entities``)
    contain at least one ref starting with ``<db>.``.

    The DB-prefix check is defensive: ``db_dir`` already scopes
    YAMLStorage to one DB's memory store, but the contract on
    deferred-KB memories requires the linked-entities attribution
    independent of where the memory file physically lives.
    """
    # DEV-1668: slayer 0.9.6 stores per-id ``memories/<id>.md`` (no flat
    # ``memories.yaml``); ask the storage layer whether any memory exists rather
    # than probing a specific file layout.
    storage = YAMLStorage(base_dir=str(db_dir))
    if not await storage.list_memories():
        return set()
    service = SearchService(storage=storage)
    documented: set[int] = set()
    db_prefix = f"{db}."
    for kb_id in kb_ids:
        question = f"KB {kb_id} — {knowledge.get(kb_id, '')}"
        # SLayer 0.7.3 (DEV-1549) collapsed the per-kind `max_*` caps into a
        # single `max_results` and made compact-mode the default. We pass
        # `compact=False` so `hit.text` carries the full ``learning`` body
        # the regex below parses, and `cypher_filter='MATCH (n:Memory)
        # RETURN n.id AS id'` so the unified cap is a memory-only cap (the
        # filter is load-bearing: without it, the cap is RRF-fused across
        # kinds and entity hits can crowd memories out, falsely reporting
        # documented KBs as unaccounted). The `kind == "memory"` filter
        # below is defence in depth.
        response = await service.search(
            question=question,
            max_results=MAX_MEMORIES_PER_KB,
            compact=False,
            cypher_filter="MATCH (n:Memory) RETURN n.id AS id",
        )
        for hit in response.results:
            if hit.kind != "memory":
                continue
            head = _first_nonblank_line(hit.text)
            m = KB_MEMORY_HEAD_RE.match(head)
            if not m or int(m.group(1)) != kb_id:
                continue
            try:
                mem = await storage.get_memory(hit.id)
            except Exception:  # noqa: BLE001 — best-effort
                continue
            if any(e.startswith(db_prefix) for e in mem.entities):
                documented.add(kb_id)
                break
    return documented


def _knowledge_text(kb_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with kb_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["id"])] = row.get("knowledge", "")
    return out


async def verify_one(
    db: str, mini_interact_root: Path,
) -> tuple[set[int], set[int], dict[int, str]]:
    """Return ``(unaccounted, overlap, knowledge)`` for the DB.

    ``knowledge`` maps KB id → its ``knowledge`` field, used for human
    readability when reporting offenders.
    """
    kb_path = _kb_path(mini_interact_root, db)
    db_dir = SLAYER_MODELS_DIR / db

    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")
    if not db_dir.is_dir():
        raise FileNotFoundError(f"slayer_models dir not found: {db_dir}")

    all_ids = load_kb_ids(kb_path)
    knowledge = _knowledge_text(kb_path)
    encoded = await load_encoded_ids(db_dir)
    # Only search memories for ids that aren't already encoded; the
    # documented set is unioned with encoded downstream so missing
    # search hits for an encoded id can't cause an unaccounted failure.
    candidates = all_ids - encoded
    documented = await load_documented_ids(
        db=db, db_dir=db_dir, kb_ids=candidates, knowledge=knowledge,
    )

    accounted = encoded | documented
    unaccounted = all_ids - accounted
    overlap = encoded & documented
    return unaccounted, overlap, knowledge


def _list_dbs() -> list[str]:
    if not SLAYER_MODELS_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in SLAYER_MODELS_DIR.iterdir()
        if p.is_dir() and p.name != "_notes"
    )


async def main_async(dbs: Iterable[str], mini_interact_root: Path) -> int:
    bad = False
    for db in dbs:
        unaccounted, overlap, knowledge = await verify_one(db, mini_interact_root)
        if not unaccounted and not overlap:
            print(f"[OK] {db}")
            continue
        bad = True
        if unaccounted:
            print(f"[FAIL] {db}: {len(unaccounted)} unaccounted KB id(s):")
            for kb_id in sorted(unaccounted):
                print(f"  - {kb_id}: {knowledge.get(kb_id, '')}")
        if overlap:
            print(
                f"[FAIL] {db}: {len(overlap)} KB id(s) both encoded and "
                f"documented (must pick one):"
            )
            for kb_id in sorted(overlap):
                print(f"  - {kb_id}: {knowledge.get(kb_id, '')}")
    return 1 if bad else 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Verify mini-interact KB coverage in exported SLayer YAML "
            "models and per-DB SLayer memories."
        ),
        epilog=(
            "After the kb-notes-to-slayer-memories migration, --all will "
            "fail for every DB except `households` until each is re-encoded "
            "under the new translate-mini-interact-kb skill. Use --db "
            "households for the day-to-day gate."
        ),
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--db", help="Verify one DB by name (e.g. 'households').")
    g.add_argument(
        "--all",
        action="store_true",
        help="Verify every DB with a slayer_models/<db>/ folder.",
    )
    p.add_argument(
        "--mini-interact-root",
        default=str(DEFAULT_MINI_INTERACT_ROOT),
        help=(
            "Root of the mini-interact dataset checkout "
            f"(default: {DEFAULT_MINI_INTERACT_ROOT})."
        ),
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    dbs = _list_dbs() if args.all else [args.db]
    if args.all and not dbs:
        print(
            f"[ERROR] No DBs found under {SLAYER_MODELS_DIR}. "
            "Run the translate-mini-interact-kb skill on at least one DB first.",
            file=sys.stderr,
        )
        return 1
    return asyncio.run(main_async(dbs, Path(args.mini_interact_root).resolve()))


if __name__ == "__main__":
    sys.exit(main())
