#!/usr/bin/env python3
"""One-off migration: bring on-disk data directories in line with DEV-1525 layout.

Idempotent — each step is a no-op when the source is already absent or the
destination already exists. Prints every action taken (or skipped).

Run once from the repo root after merging DEV-1525:

    uv run python scripts/migrate_data_dirs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from bird_interact_agents import paths as _paths

REPO_ROOT = _paths.main_checkout_root()

# Mini-interact DB names (used for cache cleanup)
_MINI_INTERACT_DBS = [
    "alien",
    "cold_chain_pharma_compliance",
    "credit",
    "exchange_traded_funds",
    "households",
    "hulushows",
    "labor_certification_applications",
    "organ_transplant",
    "planets_data",
    "reverse_logistics",
]


def _log(msg: str, dry: bool) -> None:
    prefix = "[DRY-RUN] " if dry else ""
    print(f"{prefix}{msg}")


def _rmtree(p: Path, dry: bool) -> None:
    if not p.exists():
        return
    _log(f"DELETE  {p}", dry)
    if not dry:
        shutil.rmtree(p)


def _rm(p: Path, dry: bool) -> None:
    if not p.exists():
        return
    _log(f"DELETE  {p}", dry)
    if not dry:
        p.unlink()


def _mv(src: Path, dst: Path, dry: bool) -> None:
    if not src.exists():
        _log(f"SKIP    {src} → {dst} (source absent)", dry)
        return
    if dst.exists():
        _log(f"SKIP    {src} → {dst} (destination exists)", dry)
        return
    _log(f"MOVE    {src} → {dst}", dry)
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _mkdir(p: Path, dry: bool) -> None:
    if p.exists():
        return
    _log(f"MKDIR   {p}", dry)
    if not dry:
        p.mkdir(parents=True, exist_ok=True)


def migrate(dry: bool) -> None:
    # ------------------------------------------------------------------
    # Step 1: delete old flat slayer_otf_cache/<db>/ dirs
    # The new layout is slayer_otf_cache/<benchmark>/<db>/; the old flat
    # layout had <db>/ directly under the root. Delete the old flat dirs.
    # ------------------------------------------------------------------
    otf_cache_root = REPO_ROOT / "slayer_otf_cache"
    for db in _MINI_INTERACT_DBS:
        _rmtree(otf_cache_root / db, dry)
    # Also delete households.v3bak if present
    _rmtree(otf_cache_root / "households.v3bak", dry)

    # ------------------------------------------------------------------
    # Step 2: delete old flat slayer_models_otf/<db>/ dirs + .build.lock
    # Same layout transition as step 1.
    # ------------------------------------------------------------------
    otf_models_root = REPO_ROOT / "slayer_models_otf"
    for db in _MINI_INTERACT_DBS:
        _rmtree(otf_models_root / db, dry)
        _rm(otf_models_root / f"{db}.build.lock", dry)
    _rmtree(otf_models_root / "households.v3bak", dry)

    # ------------------------------------------------------------------
    # Step 3: rename annotations/mini_interact/ → annotations/mini-interact/
    # ------------------------------------------------------------------
    ann_root = REPO_ROOT / "annotations"
    src_mi = ann_root / "mini_interact"
    dst_mi = ann_root / "mini-interact"
    if src_mi.exists() and not dst_mi.exists():
        _mv(src_mi, dst_mi, dry)
    elif src_mi.exists() and dst_mi.exists():
        # Merge: move subdirs from src that are not in dst
        _log(f"MERGE   {src_mi} → {dst_mi} (both exist, merging subdirs)", dry)
        if not dry:
            for sub in src_mi.iterdir():
                target = dst_mi / sub.name
                if not target.exists():
                    shutil.move(str(sub), str(target))
            src_mi.rmdir()  # only works if empty after merge
    else:
        _log(f"SKIP    {src_mi} → {dst_mi} (source absent or dest exists)", dry)

    # ------------------------------------------------------------------
    # Step 4: rename annotations/livesqlbench/ → annotations/livesqlbench-base-lite-sqlite/
    # ------------------------------------------------------------------
    src_lsb = ann_root / "livesqlbench"
    dst_lsb = ann_root / "livesqlbench-base-lite-sqlite"
    _mv(src_lsb, dst_lsb, dry)

    # ------------------------------------------------------------------
    # Step 5: delete stale _dev1515_convert_*.json index files
    # ------------------------------------------------------------------
    for p in ann_root.glob("_dev1515_convert*.json"):
        _rm(p, dry)

    # ------------------------------------------------------------------
    # Step 6: move audited_gold/mini-interact_audited.jsonl →
    #         audited_gold/mini-interact/mini-interact_audited.jsonl
    # ------------------------------------------------------------------
    ag_root = _paths.audited_gold_root()
    src_ag = ag_root / "mini-interact_audited.jsonl"
    dst_ag = ag_root / "mini-interact" / "mini-interact_audited.jsonl"
    _mkdir(ag_root / "mini-interact", dry)
    _mv(src_ag, dst_ag, dry)

    # ------------------------------------------------------------------
    # Step 7: delete stale audited_gold/mini_interact_audited.jsonl
    # ------------------------------------------------------------------
    _rm(ag_root / "mini_interact_audited.jsonl", dry)

    # ------------------------------------------------------------------
    # Step 8: delete stale audited_gold/livesqlbench_audited.jsonl
    # ------------------------------------------------------------------
    _rm(ag_root / "livesqlbench_audited.jsonl", dry)

    # ------------------------------------------------------------------
    # Step 8b: delete stale backup file
    # ------------------------------------------------------------------
    _rm(ag_root / "households_1_audited_smoke_backup_20260603.jsonl", dry)

    # ------------------------------------------------------------------
    # Step 9: delete empty top-level models/ and datasources/ dirs
    # ------------------------------------------------------------------
    for d in [REPO_ROOT / "models", REPO_ROOT / "datasources"]:
        if d.exists() and d.is_dir():
            contents = list(d.iterdir())
            if not contents:
                _log(f"DELETE  {d} (empty)", dry)
                if not dry:
                    d.rmdir()
            else:
                _log(f"SKIP    {d} (not empty: {[c.name for c in contents]})", dry)

    print("\nMigration complete." if not dry else "\nDry-run complete — no files changed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = ap.parse_args()
    migrate(dry=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
