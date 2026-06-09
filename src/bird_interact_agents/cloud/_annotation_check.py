"""Submit-time guard for task-annotation availability.

In the post-DEV-1515 structure the task annotation
(``<annotations_root>/<benchmark>/<db>/<instance_id>.task.json``) is the
authoritative source for grading semantics: ``gold_variants`` (audited
rewrites of the original gold), ``evaluator_prompt`` (LLM judge for
``insufficient`` tasks), ``masked_terms`` (metadata anchors), and
``original_gold_is_correct`` (whether the audited gold overlay should
apply at all).

Audited-gold is derived from annotations: a task only gets an audited
row when its annotation says the original gold is wrong. Annotation
availability is therefore the authoritative gate; the old audited-gold
existence check was a proxy that missed exactly the cases the new
structure makes explicit (annotations where the original gold IS
correct don't need an audited row, and pre-annotation submits that
silently skip the LLM judge / novel-reading-judgment land wrong
verdicts mid-cloud-run).

Default ON: fail at submit, before bringing up the cluster, rather
than discovering the missing annotation in an actor log 10+ minutes
later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from bird_interact_agents import paths
from bird_interact_agents.benchmark import Benchmark, get_benchmark


def _load_dataset_instance_db_map(
    data_path: Optional[Path] = None,
    *,
    benchmark: Benchmark | None = None,
) -> dict[str, str]:
    """Map ``instance_id -> selected_database`` from the benchmark's data file."""
    if data_path is None:
        bench_name = benchmark.name if benchmark is not None else "mini-interact"
        data_path = paths.benchmark_data_file(bench_name)
    out: dict[str, str] = {}
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                td = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = td.get("instance_id")
            db = td.get("selected_database")
            if iid and db:
                out[iid] = db
    return out


def missing_annotation_ids(
    instance_ids: Iterable[str],
    *,
    benchmark: Benchmark | None = None,
    annotations_root: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> list[str]:
    """Return the subset of ``instance_ids`` that lack a task annotation file.

    An id is reported missing when:
      - its ``selected_database`` is unknown to the benchmark's data
        file (the caller passed a typo / stale id), OR
      - the file at
        ``<annotations_root>/<benchmark>/<db>/<iid>.task.json`` does not
        exist.

    Returns the missing ids in input order; an empty list means every
    id has an annotation.
    """
    bench = benchmark or get_benchmark("mini-interact")
    ann_root = annotations_root or paths.annotations_root()
    inst_to_db = _load_dataset_instance_db_map(data_path, benchmark=bench)

    missing: list[str] = []
    for iid in instance_ids:
        db = inst_to_db.get(iid)
        if db is None:
            missing.append(iid)
            continue
        path = ann_root / bench.name / db / f"{iid}.task.json"
        # `is_file()` rather than `exists()`: `exists()` returns True
        # for directories too, so a malformed checkout where someone
        # accidentally created a directory at the annotation path
        # would silently pass the submit-time guard and then fail
        # later when the harness tries to read the file. `is_file()`
        # matches the docstring + CLI help's "file" contract.
        if not path.is_file():
            missing.append(iid)
    return missing


def annotations_requiring_audited_gold_without_rows(
    instance_ids: Iterable[str],
    *,
    benchmark: Benchmark | None = None,
    annotations_root: Optional[Path] = None,
) -> list[str]:
    """DEV-1535 r2 (Codex): layered guard on top of
    ``missing_annotation_ids``.

    When an annotation says ``original_gold_is_correct=False`` the
    grader needs audited-gold variants to grade against — otherwise
    ``load_audited_gold_rows_for`` returns an empty list and the
    tolerant grader silently falls back to the (annotated-as-wrong)
    original gold. Pre-DEV-1535 the old ``--require-audited-gold``
    flag covered this; the replacement annotation guard (which was
    the right move — annotations are the authoritative source) loses
    this specific protection for the sync gap between annotation and
    audited_gold.

    Returns the subset of ``instance_ids`` whose annotation says the
    original gold is wrong AND have no row in the audited-gold
    sidecar. Empty list means every annotation-requiring-variants id
    has matching audited-gold rows. IIDs without an annotation file
    are skipped (the primary guard handles those).
    """
    from bird_interact_agents.eval.annotation_io import read_task_annotation
    from bird_interact_agents.eval.grade_in_place import (
        load_audited_gold_rows_for,
    )

    bench = benchmark or get_benchmark("mini-interact")
    ann_root = annotations_root or paths.annotations_root()
    inst_to_db = _load_dataset_instance_db_map(None, benchmark=bench)

    missing_audited: list[str] = []
    for iid in instance_ids:
        db = inst_to_db.get(iid)
        if db is None:
            continue
        ann_path = ann_root / bench.name / db / f"{iid}.task.json"
        if not ann_path.is_file():
            continue  # primary guard catches this
        try:
            ann = read_task_annotation(ann_path)
        except Exception:  # noqa: BLE001 — malformed annotation; skip
            continue
        # `original_gold_is_correct=True` (or None for back-compat)
        # means the original IS the gold; no audited row needed.
        if getattr(ann, "original_gold_is_correct", None) is not False:
            continue
        # Annotation says original is wrong — need at least one
        # audited row to grade against.
        rows = load_audited_gold_rows_for(
            benchmark=bench.name, instance_id=iid,
        )
        if not rows:
            missing_audited.append(iid)
    return missing_audited


__all__ = [
    "missing_annotation_ids",
    "annotations_requiring_audited_gold_without_rows",
]
