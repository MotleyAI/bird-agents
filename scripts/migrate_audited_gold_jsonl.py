"""One-time migration: rewrite flat-row audited-gold JNSONLs to the grouped format.

Each consolidated JSONL had one line per (instance_id, variant_id). The new
schema groups all variants for a task into a single ``AuditedGoldRow`` line.

Run after landing the schema-redesign code:

    env -u SSH_AUTH_SOCK uv run python scripts/migrate_audited_gold_jsonl.py

Reads every ``<audited_gold_root>/**/*_audited.jsonl`` file, rewrites each in
place using ``AuditedGoldRow.from_flat_rows``. Skips files that already appear
to be in the new format (first non-blank line contains a ``variants`` key).
Backs up each original file as ``<file>.flat_backup`` before rewriting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_schema import AuditedGoldRow


def _migrate_file(jsonl_path: Path) -> tuple[int, int]:
    """Rewrite a single JSONL file. Returns (rows_written, rows_skipped_unrecoverable)."""
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not lines:
        print(f"  {jsonl_path.name}: empty — skipping")
        return 0, 0

    try:
        first = json.loads(lines[0])
    except json.JSONDecodeError as e:
        print(f"  {jsonl_path.name}: first line is not valid JSON ({e}) — skipping")
        return 0, 0
    if "variants" in first:
        print(f"  {jsonl_path.name}: already in grouped format — skipping")
        return 0, 0

    # Group flat rows by instance_id (preserving encounter order).
    ordered_iids: list[str] = []
    by_iid: dict[str, list[dict]] = {}
    errors: list[str] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"JSON decode: {e}")
            continue
        iid = row.get("instance_id")
        if not iid:
            errors.append(f"missing instance_id in: {line[:80]}")
            continue
        if iid not in by_iid:
            ordered_iids.append(iid)
        by_iid.setdefault(iid, []).append(row)

    if errors:
        print(f"  {jsonl_path.name}: {len(errors)} parse error(s):")
        for e in errors[:5]:
            print(f"    {e}")

    grouped: list[AuditedGoldRow] = []
    skipped = 0
    for iid in ordered_iids:
        flat_rows = by_iid[iid]
        try:
            grouped.append(AuditedGoldRow.from_flat_rows(flat_rows))
        except ValueError as e:
            print(f"  WARNING: {iid} skipped — {e}")
            skipped += 1

    backup = jsonl_path.with_suffix(".jsonl.flat_backup")
    if not backup.exists():
        backup.write_bytes(jsonl_path.read_bytes())
        print(f"  {jsonl_path.name}: backed up to {backup.name}")

    jsonl_path.write_text(
        "\n".join(r.model_dump_json() for r in grouped) + ("\n" if grouped else "")
    )
    print(f"  {jsonl_path.name}: {len(grouped)} grouped rows written, {skipped} skipped")
    return len(grouped), skipped


def main() -> None:
    root = paths.audited_gold_root()
    print(f"Audited gold root: {root}")
    total_written = 0
    total_skipped = 0
    for jsonl in sorted(root.rglob("*_audited.jsonl")):
        if ".flat_backup" in jsonl.name:
            continue
        print(f"\n{jsonl.relative_to(root)}:")
        w, s = _migrate_file(jsonl)
        total_written += w
        total_skipped += s
    print(f"\nDone. Total rows written: {total_written}, skipped: {total_skipped}")


if __name__ == "__main__":
    sys.exit(main())
