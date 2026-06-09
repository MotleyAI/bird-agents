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
    """Return the subset of ``instance_ids`` whose task annotation is
    missing OR unreadable.

    An id is reported missing when:
      - its ``selected_database`` is unknown to the benchmark's data
        file (the caller passed a typo / stale id), OR
      - the file at
        ``<annotations_root>/<benchmark>/<db>/<iid>.task.json`` does not
        exist, OR
      - the file is unparseable / fails schema validation
        (DEV-1535 r4 Codex): a malformed annotation file would
        otherwise pass the primary guard and then fail mid-cloud-run
        when the harness calls ``read_task_annotation``, recreating
        the exact delayed-failure mode the submit guard prevents.
        Pre-fix the schema-validation check only fired inside the
        layered audited-gold check, which was gated on
        ``--use-audited-gold-sql`` — so malformed files passed under
        ``--no-use-audited-gold-sql``.

    Returns the missing ids in input order; an empty list means every
    id has a present, parseable annotation.
    """
    from bird_interact_agents.eval.annotation_io import read_task_annotation

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
            continue
        # Schema validation: a present but malformed file is just as
        # bad as a missing one — same delayed-failure mode mid-cloud-run.
        try:
            read_task_annotation(path)
        except Exception:  # noqa: BLE001
            missing.append(iid)
    return missing


def _build_audited_gold_presence_index(
    benchmark_name: str,
    instance_to_db: Optional[dict[str, str]] = None,
) -> set[str]:
    """Read the consolidated audited-gold sidecar ONCE and return the set
    of instance_ids whose row passes the same validation
    ``apply_audited_gold_overlay`` enforces at runtime.

    DEV-1535 r3 (CodeRabbit perf) — build the index once instead of
    calling ``load_audited_gold_rows_for`` per-iid (O(N × file_size)
    pre-fix).

    DEV-1535 r5 (Codex) — validate every row against the SAME rules
    the runtime overlay enforces (was the original
    ``cloud._audited_gold_check`` module's job before this PR
    replaced it):
      - ``benchmark`` field, if present, must match the benchmark
        we're checking (cross-benchmark id collision guard — DB
        names overlap across benchmarks by design).
      - ``selected_database``, if present, must match the dataset's
        mapping for this id (same defence-in-depth — only applies
        when the dataset map is supplied).
      - ``audit_status`` of ``clean``/``original`` passes regardless of
        ``audited_sol_sql`` (the original IS the audited gold; overlay
        deliberately leaves ``sol_sql`` untouched).
      - Any other status (``edited``/``unrecoverable``) requires a
        non-empty ``audited_sol_sql`` list, otherwise the overlay
        silently falls back to original gold mid-cloud-run. Pre-r5
        a row with just ``{"instance_id": X}`` would have passed,
        reopening the exact gap the layered guard exists to close.

    Returns an empty set if the benchmark has no consolidated
    layout, the file is absent, or every row is unparseable — same
    graceful semantics as ``load_audited_gold_rows_for``."""
    from bird_interact_agents.benchmark import get_benchmark as _get_bench

    try:
        bench = _get_bench(benchmark_name)
    except Exception:  # noqa: BLE001
        return set()
    if getattr(bench, "audited_gold_layout", None) != "single_file":
        return set()
    sidecar = paths.audited_gold_root() / bench.name / f"{bench.name}_audited.jsonl"
    if not sidecar.is_file():
        return set()

    def _row_passes(audit_status: str | None,
                    audited_sol_sql: object) -> bool:
        if audit_status in ("clean", "original"):
            return True
        if audit_status in ("edited", "unrecoverable"):
            return (
                isinstance(audited_sol_sql, list)
                and bool(audited_sol_sql)
            )
        # Unknown / missing status — be conservative; treat as not present.
        return False

    present: set[str] = set()
    for line in sidecar.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        iid = d.get("instance_id")
        if not isinstance(iid, str):
            continue
        # DEV-1535 r6 (Codex): the cross-benchmark + cross-database
        # checks must REQUIRE the fields to be present (and matching),
        # not just "reject when present and wrong". The runtime overlay
        # in `harness.apply_audited_gold_overlay` treats absent
        # benchmark/selected_database as missing-row (silent fallback
        # to original gold); the submit-time index must mirror that
        # contract or the layered guard reopens the same silent-fallback
        # gap it was added to close.
        row_bench = d.get("benchmark")
        if not isinstance(row_bench, str) or row_bench != bench.name:
            continue
        row_db = d.get("selected_database")
        if not isinstance(row_db, str) or not row_db:
            continue
        if instance_to_db is not None:
            expected_db = instance_to_db.get(iid)
            if expected_db is not None and row_db != expected_db:
                continue

        # Grouped format: choose the primary variant; fall back to the
        # first variant if no primary is marked. Treat the whole row as
        # missing if no variant passes.
        if "variants" in d:
            variants = d.get("variants") or []
            if not isinstance(variants, list) or not variants:
                continue
            primary = next(
                (v for v in variants
                 if isinstance(v, dict) and v.get("primary")),
                variants[0] if isinstance(variants[0], dict) else None,
            )
            if not isinstance(primary, dict):
                continue
            status = primary.get("audit_status")
            sol_sql = primary.get("audited_sol_sql")
            if not _row_passes(status, sol_sql):
                continue
        else:
            # Legacy flat-row format — validation applies to the row itself.
            status = d.get("audit_status")
            sol_sql = d.get("audited_sol_sql")
            if not _row_passes(status, sol_sql):
                continue

        present.add(iid)
    return present


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

    DEV-1535 r3 (Codex): a malformed annotation file (parse error /
    schema validation failure) is REPORTED, not silently skipped. The
    submit-time guard exists to prevent these problems from surfacing
    mid-cloud-run; swallowing them recreates the exact delayed-failure
    mode the guard prevents. Reported as missing so the operator sees
    the broken annotation up-front.

    DEV-1535 r3 (CodeRabbit): the audited-gold sidecar is read ONCE
    into a presence set, not per-iid. The previous implementation
    re-scanned the JSONL per id (O(N × file_size)).
    """
    from bird_interact_agents.eval.annotation_io import read_task_annotation

    bench = benchmark or get_benchmark("mini-interact")
    ann_root = annotations_root or paths.annotations_root()
    inst_to_db = _load_dataset_instance_db_map(None, benchmark=bench)
    # Two-pass to avoid building the audited-gold presence index when
    # no annotation actually requires it.
    needs_audited: list[str] = []
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
        except Exception:  # noqa: BLE001
            # Malformed annotation: REPORT (don't skip). The submit
            # guard exists to surface these before the cluster comes
            # up; silently skipping recreates the delayed-failure
            # mode the guard prevents.
            missing_audited.append(iid)
            continue
        # `original_gold_is_correct=True` (or None for back-compat)
        # means the original IS the gold; no audited row needed.
        if getattr(ann, "original_gold_is_correct", None) is not False:
            continue
        needs_audited.append(iid)

    if needs_audited:
        # Pass the instance→db map so the index can enforce the
        # benchmark+database cross-checks the runtime overlay also runs.
        present = _build_audited_gold_presence_index(
            bench.name, instance_to_db=inst_to_db,
        )
        for iid in needs_audited:
            if iid not in present:
                missing_audited.append(iid)
    return missing_audited


__all__ = [
    "missing_annotation_ids",
    "annotations_requiring_audited_gold_without_rows",
]
