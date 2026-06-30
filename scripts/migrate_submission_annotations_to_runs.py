"""DEV-1533: migrate existing submission annotations from annotations/ to runs/.

Walks ``annotations/<benchmark>/<db>/<instance>.submission.<run_id>.json``
and copies each to
``runs/<benchmark>/<db>/<instance>/<run_id>.json`` (no-overwrite).

The source files in ``annotations/`` are NOT removed — they can be
cleaned up manually once the migration is confirmed correct.

Enrichment fields added in DEV-1533 (``submitted_sql``,
``predicted_result``, ``gold_result``, ``original_gold_annotated_correct``)
will be absent (``null``) in migrated files since the original annotation
files pre-date those fields.

Usage::

    uv run python scripts/migrate_submission_annotations_to_runs.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing anything")
    args = parser.parse_args(argv)
    dry_run: bool = args.dry_run

    from bird_interact_agents import paths
    from bird_interact_agents.eval.annotation_io import write_run_annotation
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
    ann_root = paths.annotations_root()
    runs_root = paths.runs_root()

    _SUB_RE = re.compile(r"^(.+)\.submission\.(.+)\.json$")

    copied = 0
    skipped = 0
    errors = 0

    for path in sorted(ann_root.rglob("*.submission.*.json")):
        m = _SUB_RE.match(path.name)
        if not m:
            continue
        instance_id, run_id = m.group(1), m.group(2)
        # Derive benchmark and db from relative path:
        # annotations/<benchmark>/<db>/<instance>.submission.<run_id>.json
        try:
            rel = path.relative_to(ann_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        benchmark, db = parts[0], parts[1]

        dest = runs_root / benchmark / db / instance_id / f"{run_id}.json"
        if dest.exists():
            skipped += 1
            print(f"  SKIP  {dest}")
            continue

        try:
            content = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  ERROR reading {path}: {exc}", file=sys.stderr)
            continue

        if dry_run:
            print(f"  WOULD copy  {path}")
            print(f"         →    {dest}")
            copied += 1
        else:
            # DEV-1591: route through write_run_annotation so a migrated record
            # gets its version/agent_model COPIED from the run's manifest (the
            # producer literal), like every other runs/ writer. Fall back to a
            # raw copy ONLY when a legacy file doesn't validate against the
            # current schema, so the migration never silently drops a record.
            # The raw-copy fallback is scoped to model_validate FAILURES alone —
            # a write/provenance/IO failure must NOT be masked as a legacy
            # record; it propagates to the outer error counter so the run stays
            # non-zero.
            try:
                ann = SubmissionAnnotation.model_validate(content)
            except Exception as exc:  # noqa: BLE001 — genuine legacy-schema record
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(content, indent=2) + "\n")
                print(f"  (raw copy — did not validate: {exc})", file=sys.stderr)
            else:
                try:
                    write_run_annotation(
                        ann, dest, benchmark=benchmark, run_id=run_id,
                    )
                except Exception as exc:  # noqa: BLE001 — real write/IO failure
                    errors += 1
                    print(f"  ERROR writing {dest}: {exc}", file=sys.stderr)
                    continue
            print(f"  COPY  {path}")
            print(f"      → {dest}")
            copied += 1

    print()
    action = "Would copy" if dry_run else "Copied"
    print(f"{action}: {copied}, Skipped (already exists): {skipped}, Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
