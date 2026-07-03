"""Sync stable task annotations from GCS into the local annotations store.

The task annotation (``<annotations_root>/<benchmark>/<db>/<iid>.task.json``) is
the authoritative grading source: ``gold_variants``, ``evaluator_prompt``,
``masked_terms``, ``original_gold_is_correct``. When absent, the grader silently
falls back to an implicit N1-only annotation. Annotations produced by cloud
``annotate`` runs live in GCS and are only mirrored to a given checkout by a
prior ``bird-interact-cloud fetch`` merge or this sync.

DEV-1638: moved out of ``scripts/fetch_local_annotations.py`` into the package
so BOTH the local ``bird-interact`` run AND the cloud ``submit`` pre-build call
the SAME ``sync_annotations`` — guaranteeing the annotation set the local grader
reads equals the set the cloud image bakes. ``scripts/fetch_local_annotations.py``
is now a thin CLI wrapper.
"""
from __future__ import annotations

import argparse
import sys

from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.cloud import gcs as _gcs
# Reuse the SAME lightweight instance_id→selected_database resolver the cloud
# require-annotation gate uses (reads the raw tasks JSONL — no gated-gold merge /
# SELECT-filter), so sync and gate agree on the id→db mapping and neither needs
# the full benchmark load at submit time.
from bird_interact_agents.cloud._annotation_check import (
    _load_dataset_instance_db_map,
)
from bird_interact_agents.eval.annotation_io import task_annotation_path


def sync_annotations(
    benchmark: str,
    instance_ids: list[str] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Pull the stable ``*.task.json`` for ``instance_ids`` (or the whole
    benchmark) from GCS into the local annotations store.

    Returns ``{"fetched", "already_local", "missing_in_gcs"}`` counts.

    Computes the locally-missing set FIRST and only constructs the GCS client
    when at least one target is missing — so an all-local (or offline) run never
    touches GCS. A blob absent in GCS is COUNTED, never raised, so the caller's
    require-annotation gate can report precisely-missing ids. Genuine infra /
    auth errors from the GCS client DO propagate (the caller decides whether to
    swallow them).
    """
    bm = get_benchmark(benchmark).name
    inst_to_db = _load_dataset_instance_db_map(benchmark=get_benchmark(bm))
    if instance_ids:
        missing = [i for i in instance_ids if i not in inst_to_db]
        if missing:
            raise SystemExit(f"instance_ids not found in {bm}: {missing}")
    targets = instance_ids or list(inst_to_db)

    result = {"fetched": 0, "already_local": 0, "missing_in_gcs": 0}
    to_fetch: list[tuple[str, str]] = []  # (instance_id, db)
    for iid in targets:
        db = inst_to_db[iid]
        dest = task_annotation_path(
            benchmark=bm, selected_database=db, instance_id=iid,
        )
        if dest.exists() and not overwrite:
            result["already_local"] += 1
        else:
            to_fetch.append((iid, db))

    if not to_fetch:
        # Nothing missing → never build a GCS client (offline / all-local).
        return result

    client = _gcs.default_gcs_client()
    bucket = client.bucket(_gcs.BUCKET_NAME)
    for iid, db in to_fetch:
        blob = bucket.blob(_gcs.stable_task_annotation_blob(bm, db, iid))
        if not blob.exists():
            result["missing_in_gcs"] += 1
            print(f"  GCS missing: {iid}", file=sys.stderr)
            continue
        dest = task_annotation_path(
            benchmark=bm, selected_database=db, instance_id=iid,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob.download_as_bytes())
        result["fetched"] += 1

    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--instance-ids", default=None,
                    help="Comma-separated subset; default = all benchmark tasks.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if a local .task.json already exists.")
    args = ap.parse_args(argv)

    ids = ([s.strip() for s in args.instance_ids.split(",") if s.strip()]
           if args.instance_ids else None)
    result = sync_annotations(args.benchmark, ids, overwrite=args.overwrite)
    print(
        f"fetched={result['fetched']} already_local={result['already_local']} "
        f"missing_in_gcs={result['missing_in_gcs']}",
        file=sys.stderr,
    )
    return 1 if result["missing_in_gcs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
