#!/usr/bin/env python3
"""Generate per-DB W4d instruction files from the audit walk.

For every DB that still has multi-KB entities after Bucket-F redirect
(Step 2), emit ``bird-interact-agents/_w4d_instructions/<db>.md``
listing each entity, the heuristic bucket label, the KB excerpts, and
the recommended split shape.

The agent picks up the instruction file and applies the matching
``R-SPLIT-*`` recipe per entity. Heuristics here are conservative — when
ambiguous, the file says "(bucket: ?, agent judgment)" so the agent
knows it has to think rather than mechanically follow.

Usage:
    python scripts/build_w4d_instructions.py            # all affected DBs
    python scripts/build_w4d_instructions.py --db <db>  # one DB
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from slayer.storage.sqlite_storage import SQLiteStorage
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents import paths

INSTRUCTIONS_DIR = paths.main_checkout_root() / "_w4d_instructions"
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


def _classify_bucket(types: list[str], n: int) -> str:
    """Return the heuristic bucket label given the KB types and id count.

    Uses simple type-frequency rules; agent-decided cases are flagged
    with ``?`` so the instruction file makes clear that a human-curated
    split shape is needed.
    """
    if n >= 10:
        return "E"
    counts = Counter(types)
    n_calc = counts.get("calculation_knowledge", 0)
    n_dom = counts.get("domain_knowledge", 0)
    n_illu = counts.get("value_illustration", 0)
    if n == 2:
        if n_calc == 1 and n_dom == 1:
            return "A"
        if n_illu == 2 or (n_illu == 1 and n_calc == 0 and n_dom == 0):
            return "B"
        if n_calc == 2:
            return "D"
        return "?"
    if n == 3:
        if n_illu >= 1 and n_calc >= 1 and n_dom >= 1:
            return "C"
        if n_illu >= 2:
            return "B"
        if n_calc >= 2:
            return "D"
        return "?"
    if n >= 4:
        if n_illu >= n - 1:
            return "B"
        if n_calc >= n - 1:
            return "D"
        return "?"
    return "?"


def _bucket_hint(bucket: str) -> str:
    return {
        "A": "R-SPLIT-CALC-THRESH (calc + threshold/classification)",
        "B": "R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; "
             "JSON blob itself stays untagged per kb_id placement rule)",
        "C": "R-SPLIT-TRINITY (helper + calc + classification, 3-way split)",
        "D": "R-SPLIT-MULTI-FORMULA (one entity per formula)",
        "E": "R-SPLIT-MONSTER (aggressive split; agent judgment required)",
        "?": "Bucket unclear — agent reads KB texts and picks the best split shape",
    }[bucket]


async def _gather(storage, dbs: list[str] | None) -> dict[str, list[tuple[str, str, str, list[int]]]]:
    out: dict[str, list[tuple[str, str, str, list[int]]]] = {}
    if dbs is None:
        dbs = sorted(await storage.list_datasources())
    for db in dbs:
        try:
            model_names = await storage.list_models(data_source=db)
        except Exception:
            continue
        rows: list[tuple[str, str, str, list[int]]] = []
        for name in model_names:
            try:
                model = await storage.get_model(name, data_source=db)
            except Exception:
                continue
            if model is None or model.data_source != db:
                continue
            owners: list[tuple[str, str, object]] = [("model", name, model)]
            owners += [("column", c.name, c) for c in (model.columns or [])]
            owners += [("measure", m.name, m) for m in (model.measures or [])]
            owners += [("aggregation", a.name, a) for a in (model.aggregations or [])]
            for kind, ename, owner in owners:
                meta = getattr(owner, "meta", None)
                ids = (meta or {}).get("kb_ids") if meta else None
                if ids and len(ids) > 1:
                    rows.append((name, kind, ename, [int(x) for x in ids]))
        if rows:
            out[db] = rows
    return out


def _render_db_md(db: str, entities: list[tuple[str, str, str, list[int]]], kb: dict[int, dict]) -> str:
    lines: list[str] = []
    lines.append(f"# W4d: {db}")
    lines.append("")
    lines.append("Workflow: see `_w4d_instructions/_README.md`.")
    lines.append(
        "Skill order: read `kb-to-slayer-models` (recipes incl. "
        "R-SPLIT-* in \"Splitting multi-KB entities\"), then "
        "`translate-mini-interact-kb` (W4d refresh override)."
    )
    lines.append("")
    lines.append("## Multi-KB entities to split")
    lines.append("")
    for model_name, kind, ename, ids in entities:
        types = [(kb.get(i, {}).get("type", "?") or "?") for i in ids]
        bucket = _classify_bucket(types, len(ids))
        lines.append(f"### {model_name}.{ename}  ({kind})")
        lines.append(f"")
        lines.append(f"- Current `meta.kb_ids`: `{ids}`")
        lines.append(f"- Bucket: **{bucket}** — {_bucket_hint(bucket)}")
        lines.append(f"- Target: produce {len(ids)} single-KB entities, each carrying its own `meta.kb_id`.")
        lines.append("")
        lines.append("KB excerpts:")
        for kid in ids:
            e = kb.get(kid)
            if e is None:
                lines.append(f"  - KB {kid}: <NOT FOUND>")
                continue
            kn = (e.get("knowledge", "?") or "?").strip()
            tp = (e.get("type", "?") or "?").strip()
            df = (e.get("definition", "") or "").strip().replace("\n", " ")
            lines.append(f"  - KB {kid} [{tp}] \"{kn}\"")
            if df:
                trunc = df[:300] + ("…" if len(df) > 300 else "")
                lines.append(f"    def: {trunc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def main_async(source: Path, mini_interact_root: Path, dbs: list[str] | None) -> int:
    storage = _open_source(source)
    by_db = await _gather(storage, dbs)
    if not by_db:
        scope = f"db={dbs[0]}" if dbs and len(dbs) == 1 else "all DBs"
        print(f"No multi-KB entities found ({scope}). Nothing to write.")
        return 0
    INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for db, entities in sorted(by_db.items()):
        kb = _kb_lookup(mini_interact_root, db)
        path = INSTRUCTIONS_DIR / f"{db}.md"
        path.write_text(_render_db_md(db, entities, kb), encoding="utf-8")
        bucket_summary = Counter(
            _classify_bucket(
                [(kb.get(i, {}).get("type", "?") or "?") for i in ids],
                len(ids),
            )
            for _, _, _, ids in entities
        )
        summary = ", ".join(f"{b}×{c}" for b, c in sorted(bucket_summary.items()))
        print(f"  [OK] {db}: {len(entities)} entities ({summary}) → {path.relative_to(paths.main_checkout_root())}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", help="Generate one DB only (default: all DBs with multi-KB entities).")
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
    args = p.parse_args()
    dbs = [args.db] if args.db else None
    return asyncio.run(main_async(
        Path(args.source).resolve(),
        Path(args.mini_interact_root).resolve(),
        dbs,
    ))


if __name__ == "__main__":
    import sys
    sys.exit(main())
