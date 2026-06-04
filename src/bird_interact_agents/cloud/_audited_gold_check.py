"""Submit-time guard for the audited-gold requirement (DEV-1478 follow-up).

The cloud CLI defaults to ``--use-audited-gold-sql`` + ``--require-audited-gold``
because silently falling back to the un-audited gold for a missing audited
row turns the cloud run into a meaningless mix of evaluations against
two different golds. The check below resolves each instance_id's
``audit_status`` and reports the subset that would silently fall back.

DEV-1510: now dispatches on `Benchmark.audited_gold_layout` so livesqlbench's
consolidated `audited_gold/<benchmark>_audited.jsonl` is supported alongside
mini-interact's per-db sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from bird_interact_agents import paths
from bird_interact_agents.benchmark import Benchmark


def _load_dataset_instance_db_map(
    data_path: Optional[Path] = None,
    *,
    benchmark: Benchmark | None = None,
) -> dict[str, str]:
    """Map ``instance_id`` -> ``selected_database`` from the benchmark's
    data file.

    Reading just two fields per row keeps this cheap (the file is ~3 MB,
    one line per task). Caches nothing — caller invokes once per submit.

    When ``data_path`` is omitted, resolves to the benchmark's data file
    via ``paths.benchmark_data_file(benchmark.name)`` so livesqlbench
    callers don't have to pass it explicitly. Default benchmark = mini
    interact (back-compat with the DEV-1478 callers).
    """
    if data_path is None:
        if benchmark is None or benchmark.name == "mini-interact":
            data_path = paths.benchmark_data_file("mini-interact")
        else:
            data_path = paths.benchmark_data_file(benchmark.name)
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


def _load_single_file_audit_index(
    audited_root: Path, benchmark: Benchmark,
) -> Optional[dict[str, tuple[str, bool, str, str]]]:
    """Return ``{instance_id: (audit_status, has_audited_sql, row_db, row_benchmark)}``
    for ``<root>/<benchmark.name>/<benchmark.name>_audited.jsonl`` or ``None`` if absent.

    ``row_db`` is the row's ``selected_database`` and ``row_benchmark``
    is the row's ``benchmark`` field; the caller uses both as defensive
    cross-benchmark guards (an instance_id collision that pointed at the
    wrong database or the wrong benchmark would silently apply the wrong
    audit otherwise).

    Supports both the new grouped format (``AuditedGoldRow`` — one line per
    task with a ``variants`` list) and the legacy flat-row format (one line
    per variant). Primary-variant preference is enforced via the ``primary``
    flag on ``AuditedGoldVariant``.
    """
    from bird_interact_agents.eval.annotation_schema import AuditedGoldRow

    path = audited_root / benchmark.name / f"{benchmark.name}_audited.jsonl"
    if not path.exists():
        return None
    out: dict[str, tuple[str, bool, str, str]] = {}
    flat_rows_by_iid: dict[str, list[dict]] = {}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(d, dict):
                continue

            if "variants" in d:
                # New grouped format.
                try:
                    row = AuditedGoldRow.model_validate(d)
                except Exception:
                    continue
                pv = row.primary_variant
                has_audited_sql = bool(pv.audited_sol_sql)
                out[row.instance_id] = (
                    pv.audit_status, has_audited_sql,
                    row.selected_database, row.benchmark,
                )
            else:
                # Legacy flat-row format — buffer and group by instance_id.
                iid = d.get("instance_id")
                if iid:
                    flat_rows_by_iid.setdefault(iid, []).append(d)

    # Convert buffered legacy flat rows by reading directly (no from_flat_rows).
    # Legacy rows may lack variant_id, have audit_status="unrecoverable", or
    # have empty audited_sol_sql for clean rows — all of which from_flat_rows
    # cannot handle cleanly.  Mirrors _load_db_audit_index's direct approach.
    for iid, flat_rows in flat_rows_by_iid.items():
        if iid in out:
            continue
        primary_row = next((r for r in flat_rows if r.get("primary")), flat_rows[0])
        status = primary_row.get("audit_status") or "missing-row"
        audited = primary_row.get("audited_sol_sql")
        has_audited_sql = isinstance(audited, list) and bool(audited)
        row_db = primary_row.get("selected_database") or ""
        row_benchmark = primary_row.get("benchmark") or ""
        out[iid] = (status, has_audited_sql, row_db, row_benchmark)
    return out


def _load_db_audit_index(
    db: str, audited_root: Path,
) -> Optional[dict[str, tuple[str, bool]]]:
    """Return ``{instance_id: (audit_status, has_audited_sql)}`` for
    ``<root>/<db>/<db>_audited.jsonl`` or ``None`` if the sidecar is absent.
    An empty file returns ``{}``.

    Codex DEV-1478 follow-up: track whether ``audited_sol_sql`` is a
    non-empty list on each row. ``apply_audited_gold_overlay`` only swaps
    ``sol_sql`` for ``edited`` / ``unrecoverable`` rows when that field is
    a non-empty list — so a sidecar row with status ``edited`` but missing
    or empty ``audited_sol_sql`` would silently fall back to the original
    un-audited gold mid-cloud-run, defeating the submit-time guard. We
    carry the presence bit so the guard can reject those rows up front.
    ``clean`` rows do not need ``audited_sol_sql`` (the original IS the
    audited gold).
    """
    path = audited_root / db / f"{db}_audited.jsonl"
    if not path.exists():
        return None
    out: dict[str, tuple[str, bool]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("instance_id")
            status = row.get("audit_status") or "missing-row"
            audited = row.get("audited_sol_sql")
            has_audited_sql = isinstance(audited, list) and bool(audited)
            if iid:
                out[iid] = (status, has_audited_sql)
    return out


def missing_audited_gold_ids(
    instance_ids: Iterable[str],
    *,
    audited_root: Optional[Path] = None,
    data_path: Optional[Path] = None,
    benchmark: Benchmark | None = None,
) -> list[str]:
    """Return the subset of ``instance_ids`` whose audited gold is missing.

    Dispatches on ``benchmark.audited_gold_layout`` (DEV-1510). Default
    benchmark = mini-interact, preserving the DEV-1478 call signature.

    An id is reported missing when:
      - its ``selected_database`` is unknown to the dataset file (the
        caller passed a typo / stale id), OR
      - the per_db sidecar / single_file file is absent, OR
      - the file has no row for the id, OR
      - (single_file only) the row's ``selected_database`` doesn't match
        the dataset's mapping for the id (cross-benchmark id collision —
        treating as missing avoids silently applying the wrong audit), OR
      - the row's ``audit_status`` is ``"edited"`` but ``audited_sol_sql``
        is missing or not a non-empty list (the overlay would silently fall
        back to the original un-audited gold for such rows, defeating the
        guard).

    A row with ``audit_status == "original"`` (or legacy ``"clean"``) passes
    regardless of ``audited_sol_sql`` because the overlay deliberately leaves
    ``sol_sql`` untouched for those rows — the original IS the audited gold
    by design. Returns the missing ids in input order; an empty list means
    everyone has audited gold.
    """
    audited_root = audited_root or paths.audited_gold_root()
    inst_to_db = _load_dataset_instance_db_map(data_path, benchmark=benchmark)
    layout = "per_db" if benchmark is None else benchmark.audited_gold_layout
    missing: list[str] = []

    if layout == "single_file":
        assert benchmark is not None  # narrowed for the type-checker
        single_index = _load_single_file_audit_index(audited_root, benchmark)
        for iid in instance_ids:
            db = inst_to_db.get(iid)
            if db is None:
                missing.append(iid)
                continue
            if single_index is None:
                missing.append(iid)
                continue
            entry = single_index.get(iid)
            if entry is None:
                missing.append(iid)
                continue
            status, has_audited_sql, row_db, row_benchmark = entry
            # Defensive cross-benchmark guard: a row is missing if its
            # `selected_database` is absent OR mismatches the dataset's
            # mapping for this id. The single_file layout is shared
            # across DBs, so applying an audit based on `instance_id`
            # alone (without verifying the per-DB discriminator) could
            # land the wrong audit. Mirrors the overlay's guard in
            # `harness.py::apply_audited_gold_overlay`.
            if not row_db or row_db != db:
                missing.append(iid)
                continue
            # Defence-in-depth: also verify the row's `benchmark` tag.
            # DB names overlap across benchmarks by design (alien, museum,
            # … exist in both mini-interact and livesqlbench) — without
            # this check, a row with the right (instance_id, db) but the
            # wrong benchmark would still pass. Schema requires it; an
            # absent field is treated the same as a mismatch.
            if not row_benchmark or row_benchmark != benchmark.name:
                missing.append(iid)
                continue
            if status in ("clean", "original"):
                continue
            if status in ("edited", "unrecoverable") and has_audited_sql:
                continue
            missing.append(iid)
        return missing

    # Default / per_db layout (mini-interact's historical contract). Kept
    # bit-identical to the pre-DEV-1510 path for back-compat.
    cache: dict[str, Optional[dict[str, tuple[str, bool]]]] = {}
    for iid in instance_ids:
        db = inst_to_db.get(iid)
        if db is None:
            missing.append(iid)
            continue
        if db not in cache:
            cache[db] = _load_db_audit_index(db, audited_root)
        index = cache[db]
        if index is None:
            missing.append(iid)
            continue
        entry = index.get(iid)
        if entry is None:
            missing.append(iid)
            continue
        status, has_audited_sql = entry
        if status in ("clean", "original"):
            continue
        if status in ("edited", "unrecoverable") and has_audited_sql:
            continue
        missing.append(iid)
    return missing
