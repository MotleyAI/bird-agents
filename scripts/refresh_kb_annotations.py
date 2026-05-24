#!/usr/bin/env python3
"""Rewrite ``label`` and ``description`` for every KB-bearing entity in
the live SLayer storage from the canonical KB texts.

For each entity carrying a singular ``meta.kb_id``:

- ``entity.label = KB.knowledge`` (verbatim) — applied only on entity
  types that have a ``label`` field (Column, ModelMeasure).
- ``entity.description`` is rewritten to:

      <preserved free-text caveat (if any, above the marker block)>

      [kb=<id>]
      <KB.definition> — <KB.description>
      [/kb=<id>]

  The ``[kb=<id>] ... [/kb=<id>]`` block is regenerable: a re-run
  strip-then-emits it. Free-text *above* the existing block is
  preserved verbatim; everything *inside* the markers is overwritten.

Entities with the legacy ``meta.kb_ids`` plural (not yet split — see
DEV-1362 Step 2-4 / W4d) are skipped: the refresh is single-KB only.
Entities with no ``meta.kb_id`` (utility columns) are left untouched.

Idempotent: a second run is a no-op (same input, same output).

Usage:
    python scripts/refresh_kb_annotations.py
    python scripts/refresh_kb_annotations.py --db <db>
    python scripts/refresh_kb_annotations.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from slayer.storage.sqlite_storage import SQLiteStorage
from slayer.storage.yaml_storage import YAMLStorage

# Reuse the export step so live storage and exported YAML stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_slayer_models import _export_async  # noqa: E402  # pyright: ignore[reportMissingImports]

from bird_interact_agents import paths  # noqa: E402

DEFAULT_SLAYER_STORAGE = Path(os.environ.get(
    "SLAYER_STORAGE",
    str(Path.home() / ".local" / "share" / "slayer"),
))
DEFAULT_MINI_INTERACT_ROOT = paths.mini_interact_root()

# Matches a [kb=<id>]<body>[/kb=<id>] block, allowing surrounding blank
# lines so re-runs collapse them rather than accumulating whitespace.
_BLOCK_RE = re.compile(
    r"\n*\[kb=(\d+)\][\s\S]*?\[/kb=\1\]\n*",
    flags=re.MULTILINE,
)


def _open_source(path: Path):
    if path.is_file() and path.suffix in (".db", ".sqlite"):
        return SQLiteStorage(db_path=str(path))
    return YAMLStorage(base_dir=str(path))


def _kb_lookup(mini_interact_root: Path, db: str) -> dict[int, dict]:
    p = mini_interact_root / db / f"{db}_kb.jsonl"
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


def _canonical_block(kb_id: int, kb_entry: dict) -> str:
    definition = (kb_entry.get("definition", "") or "").strip()
    description = (kb_entry.get("description", "") or "").strip()
    if definition and description:
        body = f"{definition} — {description}"
    else:
        body = definition or description or "(no KB body text)"
    return f"[kb={kb_id}]\n{body}\n[/kb={kb_id}]"


def _compose_description(existing: str | None, kb_id: int, kb_entry: dict) -> str:
    """Strip any prior `[kb=<kb_id>] ... [/kb=<kb_id>]` block from
    *existing* and append a freshly-rendered canonical block.

    Free-text that lives ABOVE / OUTSIDE the marker block is preserved
    verbatim (sans any leading/trailing whitespace introduced by the
    block strip).
    """
    base = (existing or "").strip()
    base = _BLOCK_RE.sub("", base).strip()
    block = _canonical_block(kb_id, kb_entry)
    if base:
        return f"{base}\n\n{block}"
    return block


def _entity_kb_id(meta: dict | None) -> int | None:
    if not meta:
        return None
    val = meta.get("kb_id")
    return int(val) if val is not None else None


def _promote_singleton_legacy_to_singular(entity) -> bool:
    """If entity has only `meta.kb_ids=[X]` (no singular `kb_id`), promote
    the lone id to `meta.kb_id=X` and drop `meta.kb_ids`. Returns True if
    anything was changed.
    """
    meta = entity.meta
    if not meta:
        return False
    legacy = meta.get("kb_ids")
    has_singular = meta.get("kb_id") is not None
    if not legacy or has_singular or len(legacy) != 1:
        return False
    new_meta = {k: v for k, v in meta.items() if k != "kb_ids"}
    new_meta["kb_id"] = int(legacy[0])
    entity.meta = new_meta
    return True


def _refresh_entity(entity, kb_id: int, kb_entry: dict, has_label: bool) -> bool:
    """Mutate *entity* in place. Returns True iff anything changed."""
    new_desc = _compose_description(getattr(entity, "description", None), kb_id, kb_entry)
    changed = False
    if entity.description != new_desc:
        entity.description = new_desc
        changed = True
    if has_label:
        new_label = (kb_entry.get("knowledge", "") or "").strip()
        if not new_label:
            new_label = (kb_entry.get("definition", "") or "").strip()[:120]
        if getattr(entity, "label", None) != new_label:
            entity.label = new_label
            changed = True
    # Drop legacy `meta.kb_ids` plural — splits are done; canonical singular
    # is `kb_id`. Only safe to drop when the kb_ids list adds nothing the
    # singular `kb_id` doesn't already cover (i.e. all elements equal kb_id).
    if entity.meta:
        legacy = entity.meta.get("kb_ids")
        if legacy and all(int(x) == kb_id for x in legacy):
            new_meta = {k: v for k, v in entity.meta.items() if k != "kb_ids"}
            entity.meta = new_meta
            changed = True
    return changed


async def _refresh_one_db(storage, mini_interact_root: Path, db: str, dry_run: bool) -> dict:
    """Refresh every KB-bearing entity in *db*. Returns per-DB stats."""
    kb = _kb_lookup(mini_interact_root, db)
    stats = {
        "models_visited": 0,
        "entities_refreshed": 0,
        "missing_kb_ids": [],
        "skipped_legacy_plural": 0,
    }
    try:
        names = await storage.list_models(data_source=db)
    except Exception as exc:
        print(f"  skip {db}: {exc}", file=sys.stderr)
        return stats

    for name in names:
        try:
            model = await storage.get_model(name, data_source=db)
        except Exception:
            continue
        if model is None or model.data_source != db:
            continue
        stats["models_visited"] += 1
        changed = False

        # Promote singleton legacy `kb_ids=[X]` to singular `kb_id=X` first,
        # then refresh against the canonical KB text.
        if _promote_singleton_legacy_to_singular(model):
            changed = True
        kid = _entity_kb_id(model.meta)
        if kid is not None:
            entry = kb.get(kid)
            if entry is None:
                stats["missing_kb_ids"].append((name, "model", name, kid))
            else:
                if _refresh_entity(model, kid, entry, has_label=False):
                    changed = True
                    stats["entities_refreshed"] += 1
        if model.meta and model.meta.get("kb_ids"):
            stats["skipped_legacy_plural"] += 1

        for c in (model.columns or []):
            if _promote_singleton_legacy_to_singular(c):
                changed = True
            kid = _entity_kb_id(c.meta)
            if kid is not None:
                entry = kb.get(kid)
                if entry is None:
                    stats["missing_kb_ids"].append((name, "column", c.name, kid))
                else:
                    if _refresh_entity(c, kid, entry, has_label=True):
                        changed = True
                        stats["entities_refreshed"] += 1
            if c.meta and c.meta.get("kb_ids"):
                stats["skipped_legacy_plural"] += 1

        for m in (model.measures or []):
            if _promote_singleton_legacy_to_singular(m):
                changed = True
            kid = _entity_kb_id(m.meta)
            if kid is not None:
                entry = kb.get(kid)
                if entry is None:
                    stats["missing_kb_ids"].append((name, "measure", m.name, kid))
                else:
                    if _refresh_entity(m, kid, entry, has_label=True):
                        changed = True
                        stats["entities_refreshed"] += 1
            if m.meta and m.meta.get("kb_ids"):
                stats["skipped_legacy_plural"] += 1

        for a in (model.aggregations or []):
            if _promote_singleton_legacy_to_singular(a):
                changed = True
            kid = _entity_kb_id(a.meta)
            if kid is not None:
                entry = kb.get(kid)
                if entry is None:
                    stats["missing_kb_ids"].append((name, "aggregation", a.name, kid))
                else:
                    # Aggregation has no `label` field.
                    if _refresh_entity(a, kid, entry, has_label=False):
                        changed = True
                        stats["entities_refreshed"] += 1
            if a.meta and a.meta.get("kb_ids"):
                stats["skipped_legacy_plural"] += 1

        if changed and not dry_run:
            await storage.save_model(model)

    return stats


async def main_async(
    source: Path,
    mini_interact_root: Path,
    db_filter: str | None,
    dry_run: bool,
) -> int:
    storage = _open_source(source)
    dbs = [db_filter] if db_filter else sorted(await storage.list_datasources())

    total = {"entities_refreshed": 0, "models_visited": 0, "skipped_legacy_plural": 0}
    affected: list[str] = []
    for db in dbs:
        stats = await _refresh_one_db(storage, mini_interact_root, db, dry_run)
        if stats["entities_refreshed"]:
            affected.append(db)
        if stats["missing_kb_ids"]:
            for model_name, kind, ename, kid in stats["missing_kb_ids"]:
                print(
                    f"[WARN] {db}/{model_name}.{ename} ({kind}): "
                    f"meta.kb_id={kid} not found in {db}_kb.jsonl",
                    file=sys.stderr,
                )
        for k in total:
            total[k] += stats[k]
        print(
            f"  [{'DRY' if dry_run else 'OK'}] {db}: "
            f"refreshed {stats['entities_refreshed']} entities "
            f"across {stats['models_visited']} models "
            f"(legacy plural skipped: {stats['skipped_legacy_plural']})"
        )

    print(
        f"\nTotal: {total['entities_refreshed']} entities refreshed across "
        f"{total['models_visited']} models. "
        f"Legacy multi-KB entities skipped: {total['skipped_legacy_plural']}."
    )
    if not dry_run and affected:
        print(f"Re-exporting {len(affected)} affected DB(s)...")
        for db in affected:
            await _export_async(db, source)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", help="Refresh one DB only (default: all DBs).")
    p.add_argument(
        "--source",
        default=str(DEFAULT_SLAYER_STORAGE),
        help=f"SLayer storage path (default: {DEFAULT_SLAYER_STORAGE}).",
    )
    p.add_argument(
        "--mini-interact-root",
        default=str(DEFAULT_MINI_INTERACT_ROOT),
        help=f"Mini-interact dataset root (default: {DEFAULT_MINI_INTERACT_ROOT}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk + classify without writing.",
    )
    args = p.parse_args()
    return asyncio.run(main_async(
        Path(args.source).resolve(),
        Path(args.mini_interact_root).resolve(),
        args.db,
        args.dry_run,
    ))


if __name__ == "__main__":
    sys.exit(main())
