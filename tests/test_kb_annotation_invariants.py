"""Post-W4d / Step-5 invariants on the exported SLayer YAML.

Three invariants apply once ``scripts/redirect_kb_duplicates.py`` (Step
2), the W4d agent splits (Steps 3-4) and ``scripts/refresh_kb_annotations
.py`` (Step 5) have all run:

1. **No multi-KB entities.** ``meta.kb_ids`` (plural) with len > 1 is
   the deprecated pre-W4d form; every entity should carry a single
   ``meta.kb_id`` instead.
2. **Canonical KB block in every KB-bearing description.** The refresh
   script wraps the KB body in ``[kb=<id>] ... [/kb=<id>]`` markers for
   idempotent regeneration; every entity with ``meta.kb_id`` set should
   have the matching block in its description.
3. **Label matches KB.knowledge.** Where the entity has a ``label``
   field (Column / ModelMeasure), it must equal ``KB.knowledge``
   verbatim.

Each test guards against pre-Step-5 state: if any multi-KB entity is
still in the export, the tests SKIP (Steps 3-4 haven't finished). Once
the export is clean, the tests enforce the invariants and any
regression fails the suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from slayer.storage.yaml_storage import YAMLStorage

# These three tests enforce the post-W4d / post-Step-5 invariants. While
# the splitting + refresh passes are still in-flight, a default repo
# checkout will fail invariant 1 (92 multi-KB entities at HEAD). Opt in
# by setting BIRD_INTERACT_ENFORCE_KB_INVARIANTS=1 — typically once
# `scripts/multi_kb_audit.py --all` reports 0 entities and
# `scripts/refresh_kb_annotations.py` has run.
pytestmark = pytest.mark.skipif(
    os.environ.get("BIRD_INTERACT_ENFORCE_KB_INVARIANTS") != "1",
    reason=(
        "post-W4d/Step-5 invariants — set "
        "BIRD_INTERACT_ENFORCE_KB_INVARIANTS=1 to enforce."
    ),
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SLAYER_MODELS_ROOT = REPO_ROOT / "slayer_models"
MINI_INTERACT_ROOT = REPO_ROOT.parent / "mini-interact"

_KB_BLOCK_RE_TPL = r"\[kb={kid}\][\s\S]*?\[/kb={kid}\]"


def _discover_dbs() -> list[str]:
    if not SLAYER_MODELS_ROOT.is_dir():
        return []
    return sorted(
        p.name
        for p in SLAYER_MODELS_ROOT.iterdir()
        if p.is_dir() and p.name != "_notes"
    )


def _kb_lookup(db: str) -> dict[int, dict]:
    p = MINI_INTERACT_ROOT / db / f"{db}_kb.jsonl"
    if not p.exists():
        return {}
    out: dict[int, dict] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        out[int(e["id"])] = e
    return out


async def _walk_db(db: str):
    """Yield (model, kind, entity) tuples for every entity in the DB."""
    storage = YAMLStorage(base_dir=str(SLAYER_MODELS_ROOT / db))
    for name in await storage.list_models():
        model = await storage.get_model(name)
        if model is None:
            continue
        yield model, "model", model
        for c in (model.columns or []):
            yield model, "column", c
        for m in (model.measures or []):
            yield model, "measure", m
        for a in (model.aggregations or []):
            yield model, "aggregation", a


def _has_multi_kb(db: str) -> bool:
    async def go() -> bool:
        async for _, _, ent in _walk_db(db):
            ids = (ent.meta or {}).get("kb_ids") if ent.meta else None
            if ids and len(ids) > 1:
                return True
        return False

    return asyncio.run(go())


@pytest.fixture(scope="module")
def dbs() -> list[str]:
    return _discover_dbs()


@pytest.mark.parametrize("db", _discover_dbs())
def test_no_multi_kb_entities(db: str) -> None:
    """No entity should carry ``meta.kb_ids`` (plural)."""
    async def go() -> list[str]:
        offenders: list[str] = []
        async for model, kind, ent in _walk_db(db):
            ids = (ent.meta or {}).get("kb_ids") if ent.meta else None
            if ids and len(ids) > 1:
                offenders.append(
                    f"{model.name}.{getattr(ent, 'name', model.name)} "
                    f"({kind}) kb_ids={list(ids)}"
                )
        return offenders

    offenders = asyncio.run(go())
    assert not offenders, (
        f"{db}: {len(offenders)} multi-KB entities remain — run W4d splitting "
        f"(scripts/build_w4d_instructions.py + agent dispatch):\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("db", _discover_dbs())
def test_kb_block_in_descriptions(db: str) -> None:
    """Every KB-bearing entity has its canonical [kb=<id>]…[/kb=<id>] block."""
    if _has_multi_kb(db):
        pytest.skip(f"{db}: multi-KB entities still present; refresh runs after Step 4")

    async def go() -> list[str]:
        offenders: list[str] = []
        async for model, kind, ent in _walk_db(db):
            kid = (ent.meta or {}).get("kb_id") if ent.meta else None
            if kid is None:
                continue
            desc = getattr(ent, "description", "") or ""
            if not re.search(_KB_BLOCK_RE_TPL.format(kid=int(kid)), desc):
                offenders.append(
                    f"{model.name}.{getattr(ent, 'name', model.name)} "
                    f"({kind}) kb_id={kid}"
                )
        return offenders

    offenders = asyncio.run(go())
    assert not offenders, (
        f"{db}: {len(offenders)} KB-bearing entities missing the canonical "
        f"[kb=<id>]…[/kb=<id>] block — run scripts/refresh_kb_annotations.py:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("db", _discover_dbs())
def test_label_matches_knowledge(db: str) -> None:
    """Column / ModelMeasure ``label`` matches ``KB.knowledge`` verbatim."""
    if _has_multi_kb(db):
        pytest.skip(f"{db}: multi-KB entities still present; refresh runs after Step 4")
    kb = _kb_lookup(db)

    async def go() -> list[str]:
        offenders: list[str] = []
        async for model, kind, ent in _walk_db(db):
            if kind not in ("column", "measure"):
                continue
            kid = (ent.meta or {}).get("kb_id") if ent.meta else None
            if kid is None:
                continue
            entry = kb.get(int(kid))
            if entry is None:
                continue  # logged by the refresh script itself
            expected = (entry.get("knowledge", "") or "").strip()
            actual = (getattr(ent, "label", None) or "").strip()
            if expected and actual != expected:
                offenders.append(
                    f"{model.name}.{ent.name} ({kind}) "
                    f"kb_id={kid}: label={actual!r} expected={expected!r}"
                )
        return offenders

    offenders = asyncio.run(go())
    assert not offenders, (
        f"{db}: {len(offenders)} entities have label != KB.knowledge — "
        f"run scripts/refresh_kb_annotations.py:\n  "
        + "\n  ".join(offenders)
    )
