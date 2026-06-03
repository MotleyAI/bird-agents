#!/usr/bin/env python3
"""One-off migration helper for DEV-1525: rename per-benchmark artifact dirs
to match the new canonical hyphenated benchmark names and new uniform nesting.

Run once from the main checkout root after deploying the updated code.
Idempotent: re-running on already-migrated or absent dirs is a no-op.

Usage:
    python scripts/migrate_benchmark_paths.py [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mv(src: Path, dst: Path, *, dry: bool) -> None:
    if not src.exists():
        print(f"  skip (absent): {src}")
        return
    if dst.exists():
        print(f"  skip (dest exists): {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  mv {src} → {dst}")
    if not dry:
        shutil.move(str(src), str(dst))


def migrate(root: Path, *, dry: bool) -> None:
    # --- OTF cache: flat dirs → nested under new canonical names ---
    # Old: slayer_otf_cache/          (mini-interact, legacy flat)
    # New: slayer_otf_cache/mini-interact/
    _mv(root / "slayer_otf_cache", root / "slayer_otf_cache" / "mini-interact", dry=dry)

    # Old: slayer_otf_cache_livesqlbench/
    # New: slayer_otf_cache/livesqlbench-base-lite-sqlite/
    _mv(
        root / "slayer_otf_cache_livesqlbench",
        root / "slayer_otf_cache" / "livesqlbench-base-lite-sqlite",
        dry=dry,
    )

    # --- OTF models: same consolidation ---
    # Old: slayer_models_otf/          (mini-interact, legacy flat)
    # New: slayer_models_otf/mini-interact/
    _mv(
        root / "slayer_models_otf",
        root / "slayer_models_otf" / "mini-interact",
        dry=dry,
    )

    # Old: slayer_models_otf_livesqlbench/
    # New: slayer_models_otf/livesqlbench-base-lite-sqlite/
    _mv(
        root / "slayer_models_otf_livesqlbench",
        root / "slayer_models_otf" / "livesqlbench-base-lite-sqlite",
        dry=dry,
    )

    # --- Audited gold: file basename uses old canonical name ---
    # Old: audited_gold/mini_interact_audited.jsonl
    # New: audited_gold/mini-interact_audited.jsonl
    _mv(
        root / "audited_gold" / "mini_interact_audited.jsonl",
        root / "audited_gold" / "mini-interact_audited.jsonl",
        dry=dry,
    )

    # Old: audited_gold/livesqlbench_audited.jsonl
    # New: audited_gold/livesqlbench-base-lite-sqlite_audited.jsonl
    _mv(
        root / "audited_gold" / "livesqlbench_audited.jsonl",
        root / "audited_gold" / "livesqlbench-base-lite-sqlite_audited.jsonl",
        dry=dry,
    )

    # --- Annotations: benchmark subdir uses old canonical name ---
    # Old: annotations/mini_interact/
    # New: annotations/mini-interact/
    _mv(
        root / "annotations" / "mini_interact",
        root / "annotations" / "mini-interact",
        dry=dry,
    )

    # Old: annotations/livesqlbench/
    # New: annotations/livesqlbench-base-lite-sqlite/
    _mv(
        root / "annotations" / "livesqlbench",
        root / "annotations" / "livesqlbench-base-lite-sqlite",
        dry=dry,
    )

    print(
        "\nNote: gated gold files (e.g. livesqlbench_sqlite_gt_kg_testcases_*.jsonl) "
        "should be placed under gated_gold/<benchmark>/ "
        "(e.g. gated_gold/livesqlbench-base-lite-sqlite/)."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without actually moving anything.",
    )
    p.add_argument(
        "--root",
        default=str(_repo_root()),
        help="Path to the main checkout root (default: auto-detected).",
    )
    args = p.parse_args()
    root = Path(args.root).resolve()
    print(f"Migration root: {root}")
    if args.dry_run:
        print("DRY RUN — no files will be moved.\n")
    migrate(root, dry=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
