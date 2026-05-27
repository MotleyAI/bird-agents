"""Submit-time guard for the audited-gold requirement (DEV-1478 follow-up).

The cloud CLI defaults to ``--use-audited-gold-sql`` + ``--require-audited-gold``
because silently falling back to the un-audited gold for a missing audited
row turns the cloud run into a meaningless mix of evaluations against
two different golds. The check below resolves each instance_id's
``audit_status`` and reports the subset that would silently fall back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from bird_interact_agents import paths


def _load_dataset_instance_db_map(
    data_path: Optional[Path] = None,
) -> dict[str, str]:
    """Map ``instance_id`` -> ``selected_database`` from mini_interact.jsonl.

    Reading just two fields per row keeps this cheap (the file is ~3 MB,
    one line per task). Caches nothing — caller invokes once per submit.
    """
    path = data_path or paths.mini_interact_data_file()
    out: dict[str, str] = {}
    with path.open() as f:
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
) -> list[str]:
    """Return the subset of ``instance_ids`` whose audited gold is missing.

    An id is reported missing when:
      - its ``selected_database`` is unknown to ``mini_interact.jsonl`` (the
        caller passed a typo / stale id), OR
      - the per-db sidecar ``<root>/<db>/<db>_audited.jsonl`` does not exist,
        OR
      - the sidecar has no row for the id, OR
      - the row's ``audit_status`` is ``edited`` or ``unrecoverable`` but
        ``audited_sol_sql`` is missing or not a non-empty list (Codex
        DEV-1478 follow-up: the overlay would silently fall back to the
        original un-audited gold for such rows, defeating the guard).

    A row with ``audit_status == "clean"`` passes regardless of
    ``audited_sol_sql`` because the overlay deliberately leaves
    ``sol_sql`` untouched for clean rows — the original IS the audited
    gold by design.
    Returns the missing ids in input order; an empty list means everyone
    has audited gold.
    """
    audited_root = audited_root or paths.audited_gold_root()
    inst_to_db = _load_dataset_instance_db_map(data_path)
    cache: dict[str, Optional[dict[str, tuple[str, bool]]]] = {}
    missing: list[str] = []
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
        if status == "clean":
            continue
        if status in ("edited", "unrecoverable") and has_audited_sql:
            continue
        missing.append(iid)
    return missing
