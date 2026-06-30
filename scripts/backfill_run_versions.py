"""Backfill ``version`` + ``agent_model`` onto existing ``runs/`` records.

DEV-1591 stream 2. New runs carry ``version`` + ``agent_model`` because the
PRODUCER writes them at grade/submit time (see
``grade_in_place._apply_config_provenance`` and ``driver.build_manifest``).
This script is kept "just in case" — to re-fill records that are missing the
fields by COPYING the literal the producer recorded, never reconstructing it.

For each ``runs/<benchmark>/<db>/<inst>/<run_id>.json`` it:
* loads the run's cloud manifest (local first, GCS fallback — reusing the
  cascade helper's caching read);
* fills a MISSING ``version`` by copying ``manifest["version"]`` (the producer
  literal); a record already stamped is left untouched (idempotent);
* fills ``agent_model`` from the manifest, preserving any value already on
  the record but WARNING when a present value disagrees with the manifest.

Usage::

    uv run python scripts/backfill_run_versions.py --benchmark mini-interact
    uv run python scripts/backfill_run_versions.py --benchmark mini-interact --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from bird_interact_agents import paths
from bird_interact_agents.eval import versioning
from bird_interact_agents.eval.annotation_io import (
    _canonical_benchmark,
    read_submission_annotation,
    write_run_annotation,
)

logger = logging.getLogger("backfill_run_versions")


# Reuse the cascade script's manifest reader (local-first, GCS fallback with
# local caching) so the backfill and the cascade agree on provenance. It's a
# top-level script, not an importable module, so load it by path.
_CASCADE = Path(__file__).resolve().parent / "cascade_for_combo.py"
_spec = importlib.util.spec_from_file_location("cascade_for_combo", _CASCADE)
_cascade = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("cascade_for_combo", _cascade)
_spec.loader.exec_module(_cascade)


def backfill(
    benchmark: str,
    *,
    allow_gcs: bool = True,
    dry_run: bool = False,
    gcs_client=None,
) -> dict:
    """Walk ``runs/<benchmark>/`` and stamp version + agent_model on each
    per-task record. Idempotent. Returns a counters dict."""
    runs_root = paths.runs_root() / _canonical_benchmark(benchmark)
    counters = {
        "scanned": 0,
        "updated": 0,
        "unchanged": 0,
        "no_manifest": 0,
        "agent_model_mismatch": 0,
        "unreadable": 0,
    }
    if not runs_root.exists():
        return counters

    manifest_cache: dict[str, Optional[dict]] = {}
    for path in sorted(runs_root.rglob("*.json")):
        if path.name.endswith(".trajectory.json"):
            continue
        try:
            rel = path.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        run_id = parts[-1][: -len(".json")]
        counters["scanned"] += 1
        try:
            ann = read_submission_annotation(path)
        except Exception as exc:  # noqa: BLE001
            counters["unreadable"] += 1
            logger.warning("unreadable %s: %s", path, exc)
            continue

        if run_id not in manifest_cache:
            # cascade.load_manifest covers benchmark-scoped local + GCS; fall
            # back to versioning.load_local_manifest for the legacy-flat
            # results/cloud/<run_id>/ layout it doesn't check (else a clean
            # claude_sdk_v1 legacy-flat run backfills as v0 with no model).
            manifest_cache[run_id] = _cascade.load_manifest(
                benchmark, run_id, gcs_client=gcs_client, allow_gcs=allow_gcs,
            ) or versioning.load_local_manifest(benchmark, run_id)
        manifest = manifest_cache[run_id]
        if manifest is None:
            counters["no_manifest"] += 1
        _, manifest_model = versioning.provenance_from_manifest(manifest)
        manifest_version = (manifest or {}).get(versioning.MANIFEST_VERSION_KEY)

        # Only fill a MISSING version by COPYING the literal the producer
        # recorded in the manifest — never reconstruct from the framework. A
        # record already stamped (by the producer at grade, or a prior
        # backfill) is authoritative and is left untouched.
        new_version = ann.version if ann.version is not None else manifest_version

        # agent_model: the record (producer-written) is authoritative; only
        # fill it from the manifest when MISSING — same copy-not-clobber rule
        # as version. Flag a disagreement but keep the record's own value.
        if manifest_model and ann.agent_model and ann.agent_model != manifest_model:
            counters["agent_model_mismatch"] += 1
            logger.warning(
                "%s: record agent_model %r != manifest %r (keeping record)",
                path, ann.agent_model, manifest_model,
            )
        new_model = ann.agent_model if ann.agent_model is not None else manifest_model

        changed = (ann.version != new_version) or (ann.agent_model != new_model)
        if not changed:
            counters["unchanged"] += 1
            continue
        counters["updated"] += 1
        if dry_run:
            continue
        ann.version = new_version
        ann.agent_model = new_model
        # Pass benchmark/run_id so the writer's own stamping is a no-op
        # (fields already set) and never re-derives a different value.
        write_run_annotation(ann, path, benchmark=benchmark, run_id=run_id)

    return counters


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", default="mini-interact")
    parser.add_argument(
        "--no-gcs", action="store_true",
        help="Don't fall back to GCS for missing local manifests.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
    )
    counters = backfill(
        args.benchmark, allow_gcs=not args.no_gcs, dry_run=args.dry_run,
    )
    print(json.dumps({"benchmark": args.benchmark, "dry_run": args.dry_run,
                      **counters}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
