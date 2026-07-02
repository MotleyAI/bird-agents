"""Sync stable task annotations from GCS into the local annotations store.

A local ``bird-interact`` run defaults to ``--require-annotation`` (the
annotation is the authoritative grading source: gold_variants, evaluator_prompt,
masked_terms, original_gold_is_correct). Annotations produced by cloud
``annotate`` runs live in GCS (``stable_task_annotation_blob``) and are only
present locally if a prior ``fetch``/merge pulled them. This script pulls the
stable ``*.task.json`` for the requested instance_ids (or the whole benchmark)
into ``<annotations_root>/<benchmark>/<db>/<instance_id>.task.json`` so a local
run's ``--require-annotation`` gate passes and grading has the annotation.

Usage:
    uv run python scripts/fetch_local_annotations.py \
        --benchmark livesqlbench-large --instance-ids id1,id2,...
    uv run python scripts/fetch_local_annotations.py \
        --benchmark livesqlbench-large            # all tasks in the benchmark
"""
from __future__ import annotations

import argparse
import sys

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.eval.annotation_io import _canonical_benchmark
from bird_interact_agents.cloud import gcs as _gcs
from bird_interact_agents.harness import load_benchmark_tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--instance-ids", default=None,
                    help="Comma-separated subset; default = all benchmark tasks.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if a local .task.json already exists.")
    args = ap.parse_args(argv)

    bm = get_benchmark(args.benchmark).name
    ids = ([s.strip() for s in args.instance_ids.split(",") if s.strip()]
           if args.instance_ids else None)
    rows = {
        r["instance_id"]: r
        for r in load_benchmark_tasks(
            bm, str(paths.benchmark_data_file(bm)), filter_ids=ids
        )
    }
    if ids:
        missing = [i for i in ids if i not in rows]
        if missing:
            raise SystemExit(f"instance_ids not found in {bm}: {missing}")
    targets = ids or list(rows)

    client = _gcs.default_gcs_client()
    bucket = client.bucket(_gcs.BUCKET_NAME)
    local_root = paths.annotations_root() / _canonical_benchmark(bm)

    n_new = n_have = n_missing = 0
    for iid in targets:
        db = rows[iid]["selected_database"]
        dest = local_root / db / f"{iid}.task.json"
        if dest.exists() and not args.overwrite:
            n_have += 1
            continue
        blob = bucket.blob(_gcs.stable_task_annotation_blob(bm, db, iid))
        if not blob.exists():
            n_missing += 1
            print(f"  GCS missing: {iid}", file=sys.stderr)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob.download_as_bytes())
        n_new += 1

    print(f"fetched={n_new} already_local={n_have} missing_in_gcs={n_missing}",
          file=sys.stderr)
    return 1 if n_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
