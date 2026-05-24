#!/usr/bin/env python3
"""Redirect Bucket-F multi-KB entities (verbatim KB duplicates).

Walks the live SLayer storage, finds every entity carrying
``meta.kb_ids`` of length exactly 2, looks up both KB entries, and
flips entities whose two KBs are verbatim restatements of each other
into single-KB form:

- Primary kb_id (smaller numeric id) stays on the entity as
  ``meta.kb_id``; ``meta.kb_ids`` is dropped.
- Secondary id is documented in
  ``slayer_models/_notes/<db>.md`` with
  ``Status: not-applicable — duplicate of KB <primary>``.
- Affected DBs are re-exported via ``scripts/export_slayer_models``
  so the YAML tree under ``slayer_models/<db>/`` stays in sync.

Triage rules per entity:

- Pairwise definition+description SequenceMatcher ratio > 0.90, OR
  either KB explicitly says "duplicate of" / "equivalent
  restatement" / "equivalent to KB" referencing the other id →
  treat as Bucket-F duplicate, redirect.
- Ratio in (0.50, 0.90] → ambiguous; print and require human
  review (script exits non-zero).
- Ratio ≤ 0.50 → not a duplicate; leave for the agent splitting
  pass (Buckets A/C/D).

Entities with len(kb_ids) > 2 are skipped — those are by definition
Bucket B/E (over-grouping) or D (multi-formula), not source
duplicates.

Usage:
    python scripts/redirect_kb_duplicates.py [--dry-run]
    python scripts/redirect_kb_duplicates.py --source <slayer_storage>
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import sys
from pathlib import Path

from slayer.storage.sqlite_storage import SQLiteStorage
from slayer.storage.yaml_storage import YAMLStorage

# Reuse the export step so live storage and exported YAML stay in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_slayer_models import _export_async  # noqa: E402  # pyright: ignore[reportMissingImports]

from bird_interact_agents import paths  # noqa: E402

SLAYER_MODELS_DIR = paths.slayer_models_root()
NOTES_DIR = SLAYER_MODELS_DIR / "_notes"
DEFAULT_SLAYER_STORAGE = Path(os.environ.get(
    "SLAYER_STORAGE",
    str(Path.home() / ".local" / "share" / "slayer"),
))
DEFAULT_MINI_INTERACT_ROOT = paths.mini_interact_root()

DUP_PHRASES = ("duplicate of", "equivalent restatement", "equivalent to kb")

# Hard-coded Bucket-F baselines per DEV-1362. Listed here because some
# pairs share the same semantic but score below the 0.90 char-similarity
# bar (e.g. the same formula written with different LaTeX commas, or one
# KB stating the threshold in prose and the other in math). Format:
# (db, model_name, entity_kind, entity_name, sorted_kb_ids).
HARDCODED_DUPS: tuple[tuple[str, str, str, str, tuple[int, int]], ...] = (
    ("alien", "signals", "column", "bfr", (4, 51)),
    ("fake", "moderationaction", "column", "is_high_activity_account", (74, 77)),
    ("households", "service_types", "column", "socsupport", (9, 37)),
    ("polar", "extreme_weather_ready", "model", "extreme_weather_ready", (10, 50)),
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


def _kb_text(entry: dict) -> str:
    return ((entry.get("definition", "") or "") + " " + (entry.get("description", "") or "")).strip().lower()


def _has_dup_phrase(entry: dict) -> bool:
    blob = _kb_text(entry)
    return any(p in blob for p in DUP_PHRASES)


def _strip_kb_ids(meta: dict | None, primary: int) -> dict:
    new = dict(meta or {})
    new.pop("kb_ids", None)
    new["kb_id"] = primary
    return new


def _append_notes_entry(
    db: str,
    secondary: int,
    primary: int,
    model_name: str,
    entity_name: str,
    kb_secondary: dict,
) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    notes_path = NOTES_DIR / f"{db}.md"
    knowledge = kb_secondary.get("knowledge", "?")
    section = (
        f"\n## KB {secondary} — {knowledge}\n\n"
        f"Reason: Verbatim restatement of KB {primary}; encoded entity is "
        f"`{model_name}.{entity_name}` with `meta.kb_id = {primary}`.\n\n"
        f"Status: not-applicable — duplicate of KB {primary}\n"
    )
    if notes_path.exists():
        existing = notes_path.read_text(encoding="utf-8")
        if f"## KB {secondary} —" in existing:
            return
        notes_path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
    else:
        notes_path.write_text(
            f"# {db} — KB entries not encoded as model entities\n" + section,
            encoding="utf-8",
        )


async def _gather_pairs(storage) -> list[tuple[str, str, str, str, list[int]]]:
    out: list[tuple[str, str, str, str, list[int]]] = []
    for db in sorted(await storage.list_datasources()):
        try:
            model_names = await storage.list_models(data_source=db)
        except Exception as exc:
            print(f"  skip {db}: {exc}", file=sys.stderr)
            continue
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
                if ids and len(ids) == 2:
                    out.append((db, name, kind, ename, [int(x) for x in ids]))
    return out


async def _redirect_one(
    storage,
    db: str,
    model_name: str,
    kind: str,
    ename: str,
    primary: int,
) -> None:
    model = await storage.get_model(model_name, data_source=db)
    if model is None:
        raise RuntimeError(f"model vanished: {db}/{model_name}")
    if kind == "model":
        model.meta = _strip_kb_ids(model.meta, primary)
    elif kind == "column":
        for c in (model.columns or []):
            if c.name == ename:
                c.meta = _strip_kb_ids(c.meta, primary)
                break
    elif kind == "measure":
        for m in (model.measures or []):
            if m.name == ename:
                m.meta = _strip_kb_ids(m.meta, primary)
                break
    elif kind == "aggregation":
        for a in (model.aggregations or []):
            if a.name == ename:
                a.meta = _strip_kb_ids(a.meta, primary)
                break
    await storage.save_model(model)


async def main_async(
    source: Path,
    mini_interact_root: Path,
    dry_run: bool,
) -> int:
    storage = _open_source(source)
    pair_findings = await _gather_pairs(storage)

    redirected: list[tuple[str, str, str, str, int, int, dict]] = []
    ambiguous: list[tuple[str, str, str, str, list[int], float]] = []
    low: list[tuple[str, str, str, str, list[int], float]] = []

    hardcoded_set = {(db, mn, k, en, ids) for (db, mn, k, en, ids) in HARDCODED_DUPS}
    for db, model_name, kind, ename, ids in pair_findings:
        kb = _kb_lookup(mini_interact_root, db)
        a, b = sorted(ids)
        ka = kb.get(a)
        kb_ent = kb.get(b)
        if ka is None or kb_ent is None:
            print(f"[WARN] {db}/{model_name}.{ename}: KB id missing in jsonl ({ids})", file=sys.stderr)
            continue
        ratio = difflib.SequenceMatcher(None, _kb_text(ka), _kb_text(kb_ent)).ratio()
        explicit = _has_dup_phrase(ka) or _has_dup_phrase(kb_ent)
        hardcoded = (db, model_name, kind, ename, (a, b)) in hardcoded_set
        if hardcoded or ratio > 0.90 or explicit:
            redirected.append((db, model_name, kind, ename, a, b, kb_ent))
        elif ratio > 0.50:
            ambiguous.append((db, model_name, kind, ename, ids, ratio))
        else:
            low.append((db, model_name, kind, ename, ids, ratio))

    print(f"Found {len(pair_findings)} pairwise multi-KB entities.")
    print(f"  → {len(redirected)} Bucket-F duplicates (will redirect)")
    print(f"  → {len(ambiguous)} ambiguous (0.50 < ratio ≤ 0.90 — needs human review)")
    print(f"  → {len(low)} not duplicates (ratio ≤ 0.50 — leave for agent splits)\n")

    affected_dbs: set[str] = set()
    for db, model_name, kind, ename, primary, secondary, kb_secondary in redirected:
        print(f"  redirect {db}/{model_name}.{ename} ({kind}): "
              f"kb_ids=[{primary},{secondary}] → kb_id={primary}")
        if not dry_run:
            await _redirect_one(storage, db, model_name, kind, ename, primary)
            _append_notes_entry(db, secondary, primary, model_name, ename, kb_secondary)
        affected_dbs.add(db)

    if ambiguous:
        print("\nAmbiguous pairs (review manually):")
        for db, model_name, kind, ename, ids, ratio in ambiguous:
            print(f"  {db}/{model_name}.{ename} ({kind}): kb_ids={ids} ratio={ratio:.3f}")

    if not dry_run and affected_dbs:
        print(f"\nRe-exporting {len(affected_dbs)} affected DB(s)...")
        for db in sorted(affected_dbs):
            await _export_async(db, source)

    return 1 if ambiguous else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
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
        help="Print the planned redirects without writing.",
    )
    args = p.parse_args()
    return asyncio.run(main_async(
        Path(args.source).resolve(),
        Path(args.mini_interact_root).resolve(),
        args.dry_run,
    ))


if __name__ == "__main__":
    sys.exit(main())
