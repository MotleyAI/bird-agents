#!/usr/bin/env python3
"""Verify the audited-gold sidecar for a mini-interact DB.

Usage:
    python scripts/verify_audited_gold.py --db households

Reads ``bird-interact-agents/audited_gold/<db>/<db>_audited.jsonl`` and
checks every entry:

- Schema: required keys present, audit_status is one of
  ``{clean, edited, unrecoverable}``, ``audited_sol_sql`` is a
  non-empty list[str].
- Every cited ``kb:<id>`` resolves in ``<db>_kb.jsonl``.
- Every cited ``column_meaning:<table>|<column>[|<sub>]`` resolves in
  ``<db>_column_meaning_base.json``.
- Every cited ``labeled_ambiguity:<term>`` resolves on the task's
  ``user_query_ambiguity.{critical,non_critical}_ambiguity[].term``
  or ``knowledge_ambiguity[].term``.
- Every cited ``knowledge_ambiguity:<term>`` resolves on the task's
  ``knowledge_ambiguity[].term``.
- ``audited_sol_sql[0]`` parses (sqlglot, sqlite dialect) and executes
  against ``<db>.sqlite`` without raising.

Exits 0 iff every entry passes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import sqlglot

from bird_interact_agents import paths

MINI_INTERACT_ROOT = paths.mini_interact_root()

REQUIRED_KEYS = {
    "instance_id",
    "selected_database",
    "audit_status",
    "original_sol_sql",
    "audited_sol_sql",
    "audited_sample_row",
    "changes",
    "reasoning_summary",
    "skill_version",
    "audited_at",
}
VALID_STATUS = {"clean", "edited", "unrecoverable", "ambiguous"}


def audited_root_for(audit_set: str) -> Path:
    """Return the on-disk root directory for the chosen audit set."""
    if audit_set == "inhouse":
        return paths.audited_gold_root()
    if audit_set == "sar":
        return paths.sar_audited_gold_root()
    raise ValueError(f"unknown audit-set {audit_set!r}")


def audited_filename_for(db: str, audit_set: str) -> str:
    if audit_set == "inhouse":
        return f"{db}_audited.jsonl"
    if audit_set == "sar":
        return f"{db}_sar_audited.jsonl"
    raise ValueError(f"unknown audit-set {audit_set!r}")


def load_audited(db: str, audit_set: str = "inhouse") -> list[dict]:
    # DEV-1515: inhouse mini-interact moved to single_file layout. Read
    # the consolidated `mini_interact_audited.jsonl` and filter by
    # `selected_database`. SAR audit set still uses the per_db layout.
    if audit_set == "inhouse":
        single = audited_root_for(audit_set) / "mini_interact_audited.jsonl"
        if single.exists():
            rows: list[dict] = []
            with single.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("selected_database") == db:
                        rows.append(d)
            if not rows:
                raise FileNotFoundError(
                    f"No rows for db={db!r} in {single}"
                )
            return rows
    path = audited_root_for(audit_set) / db / audited_filename_for(db, audit_set)
    if not path.exists():
        raise FileNotFoundError(f"No sidecar at {path}")
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_kb_ids(db: str) -> set[int]:
    path = MINI_INTERACT_ROOT / db / f"{db}_kb.jsonl"
    ids: set[int] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids.add(int(d["id"]))
    return ids


def load_column_meaning_keys(db: str) -> set[str]:
    """Return the set of valid column-meaning keys, including JSONB sub-fields.

    A top-level key like ``households|properties|dwelling_specs`` may itself
    point to a string (simple column) OR a dict with ``column_meaning`` +
    ``fields_meaning``. For each ``fields_meaning`` leaf, the audit may
    cite ``households|properties|dwelling_specs|<sub>``; we expand those
    here. Nested ``fields_meaning`` (e.g. vehicle_counts.Auto_Count) is
    also expanded one extra level deep.
    """
    path = MINI_INTERACT_ROOT / db / f"{db}_column_meaning_base.json"
    raw = json.loads(path.read_text())
    keys: set[str] = set()

    def walk_fields_meaning(prefix: str, blob: Any) -> None:
        if isinstance(blob, dict):
            for k, v in blob.items():
                composite = f"{prefix}|{k}"
                keys.add(composite)
                if isinstance(v, dict):
                    walk_fields_meaning(composite, v)

    for top_key, val in raw.items():
        keys.add(top_key)
        if isinstance(val, dict):
            fm = val.get("fields_meaning")
            if isinstance(fm, dict):
                walk_fields_meaning(top_key, fm)
    return keys


def load_task_records(db: str) -> dict[str, dict]:
    path = MINI_INTERACT_ROOT / "mini_interact.jsonl"
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("selected_database") == db:
                out[d["instance_id"]] = d
    return out


def labeled_terms(task: dict) -> set[str]:
    uqa = task.get("user_query_ambiguity") or {}
    terms: set[str] = set()
    for sub in ("critical_ambiguity", "non_critical_ambiguity"):
        for entry in uqa.get(sub) or []:
            t = entry.get("term")
            if t:
                terms.add(t)
    for entry in task.get("knowledge_ambiguity") or []:
        t = entry.get("term")
        if t:
            terms.add(t)
    return terms


def knowledge_ambig_terms(task: dict) -> set[str]:
    return {e.get("term") for e in (task.get("knowledge_ambiguity") or []) if e.get("term")}


def check_citations(entry: dict, kb_ids: set[int], cm_keys: set[str], task: dict, errs: list[str]) -> None:
    inst = entry["instance_id"]
    labeled = labeled_terms(task)
    ka_terms = knowledge_ambig_terms(task)
    for change in entry.get("changes", []):
        for cite in change.get("justified_by", []):
            if cite == "primitive":
                continue
            # `dialect:<engine>:<feature>` — mechanical SQL-dialect
            # correctness fix that isn't traceable to a KB/column-meaning
            # source (the dialect itself is the source). Used for post-hoc
            # bug fixes to audited SQL: integer vs floating-point division
            # on SQLite, lack of regex in REPLACE, etc. Free-form `feature`
            # is intentional — every dialect bug has its own shape.
            if cite.startswith("dialect:"):
                rest = cite.split(":", 2)
                if len(rest) != 3 or not rest[1] or not rest[2]:
                    errs.append(
                        f"{inst}: malformed dialect citation {cite!r} "
                        "(expected dialect:<engine>:<feature>)",
                    )
                continue
            if cite.startswith("kb:"):
                try:
                    kid = int(cite.split(":", 1)[1])
                except ValueError:
                    errs.append(f"{inst}: malformed citation {cite!r}")
                    continue
                if kid not in kb_ids:
                    errs.append(f"{inst}: cited kb:{kid} not in {entry['selected_database']}_kb.jsonl")
            elif cite.startswith("column_meaning:"):
                key = cite.split(":", 1)[1]
                if key not in cm_keys:
                    errs.append(f"{inst}: cited column_meaning:{key!r} not in column_meaning_base.json")
            elif cite.startswith("labeled_ambiguity:"):
                term = cite.split(":", 1)[1]
                if term not in labeled:
                    errs.append(f"{inst}: cited labeled_ambiguity:{term!r} not in task's user_query_ambiguity or knowledge_ambiguity terms")
            elif cite.startswith("knowledge_ambiguity:"):
                term = cite.split(":", 1)[1]
                if term not in ka_terms:
                    errs.append(f"{inst}: cited knowledge_ambiguity:{term!r} not in task's knowledge_ambiguity terms")
            else:
                errs.append(f"{inst}: unknown citation prefix in {cite!r}")


def check_schema(entry: dict, errs: list[str]) -> None:
    inst = entry.get("instance_id", "<missing>")
    missing = REQUIRED_KEYS - set(entry.keys())
    if missing:
        errs.append(f"{inst}: missing keys {sorted(missing)}")
    status = entry.get("audit_status")
    if status not in VALID_STATUS:
        errs.append(f"{inst}: audit_status={status!r} not in {sorted(VALID_STATUS)}")
    aud = entry.get("audited_sol_sql")
    if not isinstance(aud, list) or not aud or not all(isinstance(s, str) and s.strip() for s in aud):
        errs.append(f"{inst}: audited_sol_sql must be a non-empty list of non-empty strings")
    orig = entry.get("original_sol_sql")
    if not isinstance(orig, list) or not orig:
        errs.append(f"{inst}: original_sol_sql must be a non-empty list")
    if status == "clean" and entry.get("changes"):
        errs.append(f"{inst}: audit_status=clean but changes is non-empty")
    if status == "clean" and aud != orig:
        errs.append(f"{inst}: audit_status=clean but audited_sol_sql != original_sol_sql")
    # `edited` / `unrecoverable` carry per-clause justification. An empty
    # `changes` array breaks the sidecar contract: callers can't trace
    # what was changed (`edited`) or what was deemed unauthorised
    # (`unrecoverable`). Reject silent edits / silent deferrals.
    if status in {"edited", "unrecoverable", "ambiguous"} and not entry.get("changes"):
        errs.append(f"{inst}: audit_status={status} requires non-empty changes")


def check_sql(entry: dict, db_path: Path, errs: list[str]) -> None:
    inst = entry["instance_id"]
    sql = entry["audited_sol_sql"][0]
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as e:
        errs.append(f"{inst}: sqlglot parse failed — {e}")
        return
    try:
        # Open read-only + immutable to avoid creating a journal/WAL file;
        # makes the verifier runnable in sandboxes where the DB directory
        # isn't writable.
        con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        con.execute(sql).fetchone()
        con.close()
    except Exception as e:
        errs.append(f"{inst}: sqlite execution failed — {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="DB name (e.g. households)")
    ap.add_argument(
        "--audit-set",
        choices=["inhouse", "sar"],
        default="inhouse",
        help="Which audit JSONL to validate (inhouse=audit-gold-sql skill, sar=SAR-Agent audit)",
    )
    args = ap.parse_args()
    db = args.db
    audit_set = args.audit_set

    db_path = MINI_INTERACT_ROOT / db / f"{db}.sqlite"
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 2

    try:
        rows = load_audited(db, audit_set=audit_set)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        kb_ids = load_kb_ids(db)
        cm_keys = load_column_meaning_keys(db)
        tasks = load_task_records(db)
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2

    errs: list[str] = []
    seen: dict[str, int] = defaultdict(int)
    for entry in rows:
        seen[entry.get("instance_id", "")] += 1
        check_schema(entry, errs)
        if entry.get("instance_id") not in tasks:
            errs.append(f"{entry.get('instance_id')!r}: not found in mini_interact.jsonl for db={db}")
            continue
        check_citations(entry, kb_ids, cm_keys, tasks[entry["instance_id"]], errs)
        if entry.get("audited_sol_sql") and isinstance(entry["audited_sol_sql"], list):
            check_sql(entry, db_path, errs)

    # Duplicate instance_ids follow latest-wins (per the documented
    # sidecar contract). Surface as a warning to stderr rather than a
    # hard-fail, so an otherwise-valid sidecar isn't blocked by a
    # to-be-deduplicated row.
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        print(
            f"[WARN] {db} — duplicate instance_ids encountered; latest-wins will apply: {dupes}",
            file=sys.stderr,
        )

    if errs:
        print(f"[FAIL] {db} — {len(errs)} issue(s):")
        for e in errs:
            print(f"  - {e}")
        return 1

    by_status: dict[str, int] = defaultdict(int)
    for r in rows:
        by_status[r["audit_status"]] += 1
    print(f"[OK] {db} — {len(rows)} entries audited")
    for s in ("clean", "edited", "unrecoverable", "ambiguous"):
        print(f"  {s}: {by_status[s]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
