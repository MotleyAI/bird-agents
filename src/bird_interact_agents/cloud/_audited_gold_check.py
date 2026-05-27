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
) -> Optional[dict[str, str]]:
    """Return ``{instance_id: audit_status}`` for ``<root>/<db>/<db>_audited.jsonl``
    or ``None`` if the sidecar is absent. An empty file returns ``{}``."""
    path = audited_root / db / f"{db}_audited.jsonl"
    if not path.exists():
        return None
    out: dict[str, str] = {}
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
            status = row.get("audit_status")
            if iid:
                out[iid] = status or "missing-row"
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
      - the sidecar has no row for the id.

    A row with ``audit_status`` in ``("clean", "edited", "unrecoverable")`` is
    accepted — these are the three states for which the harness has an
    ``audited_sol_sql`` (or a deliberately-equal-to-original gold).
    Returns the missing ids in input order; an empty list means everyone
    has audited gold.
    """
    audited_root = audited_root or paths.audited_gold_root()
    inst_to_db = _load_dataset_instance_db_map(data_path)
    cache: dict[str, Optional[dict[str, str]]] = {}
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
        status = index.get(iid)
        if status is None:
            missing.append(iid)
            continue
        if status not in ("clean", "edited", "unrecoverable"):
            missing.append(iid)
    return missing
