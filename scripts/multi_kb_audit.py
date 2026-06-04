#!/usr/bin/env python3
"""Find every entity in a SLayer storage carrying ``meta.kb_ids`` (plural;
the multi-KB form), enriched with each KB's ``knowledge`` and
``definition`` so we can assess whether the conjunction is justified.

By default reads the live SLayer storage at ``~/.local/share/slayer``
(override via ``--source`` or ``$SLAYER_STORAGE``); the same storage the
SLayer MCP server writes to. Useful as the post-Step-4 acceptance gate
for DEV-1362's splitting pass: exits 0 only when zero multi-KB entities
remain.

Usage:
    python scripts/multi_kb_audit.py --all
    python scripts/multi_kb_audit.py --db households
    python scripts/multi_kb_audit.py --all --source slayer_models/polar
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from slayer.storage.sqlite_storage import SQLiteStorage
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents import paths

DEFAULT_SLAYER_STORAGE = Path(os.environ.get(
    "SLAYER_STORAGE",
    str(Path.home() / ".local" / "share" / "slayer"),
))
DEFAULT_MINI_INTERACT_ROOT = paths.benchmark_data_root("mini-interact")


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


def _multi_ids(meta: dict | None) -> list[int] | None:
    """Return the multi-KB id list from an entity's meta, or None.

    Only ``meta.kb_ids`` with len > 1 counts as multi-KB. A list of
    length 1 is a stylistic singular and gets normalised away by the
    refresh script in Step 5.
    """
    if not meta:
        return None
    ids = meta.get("kb_ids")
    if ids and len(ids) > 1:
        return [int(x) for x in ids]
    return None


async def _walk(storage, dbs: list[str] | None) -> list[tuple[str, str, str, str, list[int]]]:
    """Walk *storage* and return (db, model, kind, entity_name, kb_ids) tuples."""
    findings: list[tuple[str, str, str, str, list[int]]] = []
    if dbs is None:
        dbs = sorted(await storage.list_datasources())

    for db in dbs:
        try:
            model_names = await storage.list_models(data_source=db)
        except Exception as exc:
            print(f"  skip {db}: {exc}", file=sys.stderr)
            continue
        for name in model_names:
            try:
                model = await storage.get_model(name, data_source=db)
            except Exception as exc:
                print(f"  skip {db}/{name}: {exc}", file=sys.stderr)
                continue
            if model is None or model.data_source != db:
                continue
            mids = _multi_ids(model.meta)
            if mids:
                findings.append((db, name, "model", name, mids))
            for c in (model.columns or []):
                cids = _multi_ids(c.meta)
                if cids:
                    findings.append((db, name, "column", c.name, cids))
            for m in (model.measures or []):
                mmids = _multi_ids(m.meta)
                if mmids:
                    findings.append((db, name, "measure", m.name, mmids))
            for a in (model.aggregations or []):
                aids = _multi_ids(a.meta)
                if aids:
                    findings.append((db, name, "aggregation", a.name, aids))
    return findings


async def main_async(
    source: Path,
    dbs: list[str] | None,
    mini_interact_root: Path,
) -> int:
    storage = _open_source(source)
    findings = await _walk(storage, dbs)
    n_dbs = len(set(f[0] for f in findings))

    if not findings:
        scope = f"db={dbs[0]}" if dbs and len(dbs) == 1 else "all datasources"
        print(f"Found 0 multi-KB-id entities ({scope}, source={source}).")
        return 0

    print(
        f"Found {len(findings)} multi-KB-id entities across "
        f"{n_dbs} datasources (source={source}).\n"
    )
    by_db: dict[str, list] = defaultdict(list)
    for f in findings:
        by_db[f[0]].append(f)

    for db, items in by_db.items():
        kb = _kb_lookup(mini_interact_root, db)
        print(f"\n{'=' * 80}\n{db}  ({len(items)} entities)\n{'=' * 80}")
        for _, model, kind, ename, ids in items:
            print(f"\n  {model}.{ename}  ({kind})  kb_ids={ids}")
            for kid in ids:
                e = kb.get(kid)
                if not e:
                    print(f"    KB {kid}: <NOT FOUND in {db}_kb.jsonl>")
                    continue
                kn = e.get("knowledge", "?")
                tp = e.get("type", "?")
                df = (e.get("definition", "") or "").strip()
                print(f"    KB {kid:3d} [{tp}] {kn!r}")
                print(f"      def: {df[:240]}{'…' if len(df) > 240 else ''}")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit a SLayer storage for entities carrying meta.kb_ids "
            "(plural; multi-KB form). Exit 0 iff none found."
        ),
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--db", help="Audit one DB by name (e.g. 'households').")
    g.add_argument(
        "--all",
        action="store_true",
        help="Audit every datasource in the storage.",
    )
    p.add_argument(
        "--source",
        default=str(DEFAULT_SLAYER_STORAGE),
        help=(
            f"Path to the SLayer storage (default: {DEFAULT_SLAYER_STORAGE}). "
            "Override via $SLAYER_STORAGE."
        ),
    )
    p.add_argument(
        "--mini-interact-root",
        default=str(DEFAULT_MINI_INTERACT_ROOT),
        help=(
            "Root of the mini-interact dataset checkout, used to enrich "
            f"findings with KB text (default: {DEFAULT_MINI_INTERACT_ROOT})."
        ),
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    dbs = None if args.all else [args.db]
    return asyncio.run(main_async(
        Path(args.source).resolve(),
        dbs,
        Path(args.mini_interact_root).resolve(),
    ))


if __name__ == "__main__":
    sys.exit(main())
