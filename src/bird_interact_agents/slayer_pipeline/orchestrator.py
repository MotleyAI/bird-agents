"""End-to-end orchestrator for phases 1–4 of the DB→SLayer-model rework.

Phases:

1. ``slayer datasources create`` + ``slayer ingest`` (subprocess) to
   register the SQLite DB and auto-generate one Column per table column.
2. ``column_meaning`` overlay — descriptions, deterministic type-token
   parsing, and DEV-1381 date-format annotations.
3. JSONB-leaf expansion — one Column per ``fields_meaning`` terminal
   leaf, full-path ``__`` naming, ``JSON_EXTRACT`` sql, copied
   description, ``meta.derived_from`` for idempotency.
4. LLM TEXT-as-date detection — sample values from SQLite, classify
   via the Anthropic API, retype confident matches to TIMESTAMP and
   rewrite ``Column.sql`` to a dialect-native parse expression.

KB encoding (phase 5) and ``verify_kb_coverage.py`` (phase 6) are
caller-side concerns. The orchestrator stops at the end of phase 4.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from slayer.storage.yaml_storage import YAMLStorage

from .. import paths
from ..config import get_default_llm_model
from .dates import detect_and_apply, make_anthropic_client
from .jsonb import detect_drift, expand_one_column, jsonb_meaning_entries
from .overlay import apply_overlay, load_meanings
from .portable_connection import absolute_sqlite_url

DEFAULT_MINI_INTERACT_ROOT = paths.mini_interact_root()
DEFAULT_RESULTS_ROOT = paths.results_root()
DEFAULT_SLAYER_STORAGE = Path(
    os.environ.get("SLAYER_STORAGE", str(Path.home() / ".local" / "share" / "slayer"))
)


def _wipe_db_from_storage(storage: Path, db: str) -> None:
    ds_yaml = storage / "datasources" / f"{db}.yaml"
    if ds_yaml.exists():
        ds_yaml.unlink()
    models_dir = storage / "models" / db
    if models_dir.exists():
        shutil.rmtree(models_dir)


def _slayer_ingest(
    conn_str: str, storage: Path, *, db: str, pg_password: str | None = None
) -> None:
    """Run ``slayer datasources create`` + ``slayer ingest`` for one DB.

    Extracted so tests can monkeypatch this without touching subprocess.
    ``conn_str`` is any valid SQLAlchemy connection string (sqlite or
    postgresql). ``db`` is the datasource name.

    ``pg_password``, when provided, is passed as the ``PGPASSWORD`` environment
    variable so it never appears in the subprocess command-line args (visible
    via ``ps``) or in the persisted ``datasources/<db>.yaml``.
    """
    storage.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SLAYER_STORAGE"] = str(storage)
    if pg_password is not None:
        env["PGPASSWORD"] = pg_password

    create = subprocess.run(
        ["slayer", "datasources", "create", conn_str, "--name", db],
        env=env,
        capture_output=True,
        text=True,
    )
    if create.returncode != 0 and "already exists" not in create.stderr:
        raise RuntimeError(
            f"slayer datasources create failed for {db}: {create.stderr.strip()}"
        )
    # Canonicalise the persisted datasource YAML's connection_string to the
    # SAME absolute URL — the slayer CLI may store a different slash form, and
    # later phases read this YAML. Robust line-rewrite (any prior value), not a
    # string-replace keyed on the input path.
    ds_yaml = storage / "datasources" / f"{db}.yaml"
    if ds_yaml.exists():
        text = ds_yaml.read_text()
        fixed = re.sub(
            r"(?m)^(\s*connection_string:\s*).*$",
            lambda m: f"{m.group(1)}{conn_str}",
            text,
        )
        if fixed != text:
            ds_yaml.write_text(fixed)

    ingest = subprocess.run(
        ["slayer", "ingest", "--datasource", db],
        env=env,
        capture_output=True,
        text=True,
    )
    if ingest.returncode != 0:
        raise RuntimeError(
            f"slayer ingest failed for {db}: {ingest.stderr.strip()}"
        )


def _phase1_ingest(
    db: str,
    storage: Path,
    *,
    sqlite_path: Optional[Path] = None,
    db_url: Optional[str] = None,
    pg_password: str | None = None,
) -> None:
    """Phase 1: ingest a DB into SLayer storage.

    Exactly one of ``sqlite_path`` or ``db_url`` must be provided.
    ``sqlite_path`` is the legacy SQLite path; ``db_url`` is a full
    SQLAlchemy URL (e.g. ``postgresql://...``) used for postgres backends.

    ``pg_password`` is forwarded to ``_slayer_ingest`` as the ``PGPASSWORD``
    env var so the credential never appears in subprocess args or YAML files.
    """
    if db_url is not None:
        conn_str = db_url
    elif sqlite_path is not None:
        # Canonical absolute URL, resolved env-independently — so a RELATIVE
        # `sqlite_path` (from a relative `--db-path`) can't become a broken
        # `sqlite:////../…` that SQLAlchemy reads at the filesystem root.
        conn_str = absolute_sqlite_url(sqlite_path)
    else:
        raise ValueError("_phase1_ingest: exactly one of sqlite_path or db_url must be provided")
    _slayer_ingest(conn_str, storage, db=db, pg_password=pg_password)


async def _phase2_overlay(
    storage: YAMLStorage, db: str, meanings_path: Path
) -> tuple[int, list[str]]:
    by_table = load_meanings(meanings_path)
    touched_total = 0
    warnings: list[str] = []
    for name in await storage.list_models(data_source=db):
        model = await storage.get_model(name, data_source=db)
        if model is None or model.data_source != db:
            continue
        touched, warns = apply_overlay(model, by_table)
        if touched or warns:
            await storage.save_model(model)
        touched_total += touched
        warnings.extend(warns)
    return touched_total, warnings


def _detect_jsonb_columns(meanings_path: Path) -> list:
    """Extract JSONB column entries from a column-meaning file.

    Extracted as a module-level function so tests can monkeypatch it to
    verify that postgres builds skip JSONB detection entirely.
    """
    import json as _json

    raw = _json.loads(meanings_path.read_text(encoding="utf-8"))
    return list(jsonb_meaning_entries(raw))


async def _phase3_jsonb(
    storage: YAMLStorage,
    db: str,
    meanings_path: Optional[Path] = None,
    sqlite_path: Optional[Path] = None,
    *,
    benchmark: Optional[object] = None,
) -> tuple[int, list[str], list[str]]:
    if getattr(benchmark, "db_backend", "sqlite") == "postgres":
        return 0, [], []

    if meanings_path is None:
        raise ValueError("_phase3_jsonb: meanings_path is required for non-postgres benchmarks")
    if sqlite_path is None:
        raise ValueError("_phase3_jsonb: sqlite_path is required for non-postgres benchmarks")

    added_total = 0
    typing_warnings: list[str] = []
    drift_warnings: list[str] = []
    for table, json_col, entry in _detect_jsonb_columns(meanings_path):
        model = await storage.get_model(table, data_source=db)
        if model is None or model.data_source != db:
            typing_warnings.append(
                f"{table}: JSONB column '{json_col}' declared in meanings but "
                f"no matching model in storage; skipped."
            )
            continue
        existing_by_name = {c.name: c for c in model.columns}
        new_cols, warns = expand_one_column(json_col, entry)
        typing_warnings.extend(warns)
        added_here = 0
        changed_here = False
        for col in new_cols:
            existing = existing_by_name.get(col.name)
            if existing is not None:
                # Refresh fields when derived_from matches — lets reruns
                # propagate fields_meaning edits to already-emitted
                # leaves. Hand-written columns (no derived_from match)
                # are left alone.
                if (existing.meta or {}).get("derived_from") == (
                    col.meta or {}
                ).get("derived_from"):
                    existing.description = col.description
                    existing.type = col.type
                    existing.sql = col.sql
                    existing.label = col.label
                    existing.meta = col.meta
                    changed_here = True
                continue
            model.columns.append(col)
            existing_by_name[col.name] = col
            added_here += 1
        # Top-level JSONB column gets a meta.jsonb=True flag so future
        # passes can find it without re-sniffing the description.
        jsonb_flagged = False
        for col in model.columns:
            if col.name.lower() == json_col.lower():
                meta = col.meta or {}
                if not meta.get("jsonb"):
                    meta["jsonb"] = True
                    col.meta = meta
                    jsonb_flagged = True
                break
        # Drift detection (top-level keys only, first cut).
        documented_keys = set(entry.get("fields_meaning", {}).keys())
        undoc, ghost = detect_drift(
            sqlite_path, table, json_col, documented_keys
        )
        for key in sorted(undoc):
            drift_warnings.append(
                f"{table}.{json_col}: undocumented top-level key in data: '{key}'"
            )
        for key in sorted(ghost):
            drift_warnings.append(
                f"{table}.{json_col}: documented key absent from sampled rows: '{key}'"
            )
        if added_here or changed_here or jsonb_flagged:
            await storage.save_model(model)
            added_total += added_here
    return added_total, typing_warnings, drift_warnings


async def _phase4_dates(
    storage: YAMLStorage,
    db: str,
    sqlite_path: Optional[Path] = None,
    llm_model: Optional[str] = None,
    *,
    benchmark: Optional[object] = None,
) -> tuple[int, list[str]]:
    if getattr(benchmark, "db_backend", "sqlite") == "postgres":
        return 0, []

    if sqlite_path is None:
        raise ValueError("_phase4_dates: sqlite_path is required for non-postgres benchmarks")
    if llm_model is None:
        raise ValueError("_phase4_dates: llm_model is required for non-postgres benchmarks")

    client = make_anthropic_client()
    retyped_total = 0
    warnings: list[str] = []
    for name in await storage.list_models(data_source=db):
        model = await storage.get_model(name, data_source=db)
        if model is None or model.data_source != db:
            continue
        retyped, warns = detect_and_apply(model, sqlite_path, client, llm_model)
        if retyped:
            await storage.save_model(model)
        retyped_total += retyped
        warnings.extend(warns)
    return retyped_total, warnings


def _write_warnings(results_dir: Path, filename: str, lines: list[str]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("(no warnings)\n", encoding="utf-8")
    return path


async def regenerate(
    db: str,
    mini_interact_root: Path = DEFAULT_MINI_INTERACT_ROOT,
    slayer_storage: Path = DEFAULT_SLAYER_STORAGE,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    llm_model: Optional[str] = None,
    wipe: bool = True,
    skip_phase1: bool = False,
    skip_phase4: bool = False,
) -> int:
    sqlite_path = mini_interact_root / db / f"{db}.sqlite"
    meanings_path = mini_interact_root / db / f"{db}_column_meaning_base.json"
    if not sqlite_path.is_file():
        print(f"[ERROR] SQLite not found: {sqlite_path}", file=sys.stderr)
        return 1
    if not meanings_path.is_file():
        print(f"[ERROR] column_meaning_base not found: {meanings_path}", file=sys.stderr)
        return 1

    results_dir = results_root / db

    if wipe and not skip_phase1:
        _wipe_db_from_storage(slayer_storage, db)

    if skip_phase1:
        print(f"[phase 1] skipped (--skip-phase1; assuming live storage already populated)")
    else:
        print(f"[phase 1] slayer datasources create + ingest ({db})")
        _phase1_ingest(db, slayer_storage, sqlite_path=sqlite_path)

    storage = YAMLStorage(base_dir=str(slayer_storage))

    print(f"[phase 2] column_meaning overlay ({db})")
    touched, p2_warns = await _phase2_overlay(storage, db, meanings_path)
    print(f"  {touched} columns touched, {len(p2_warns)} warnings")

    print(f"[phase 3] JSONB leaf expansion ({db})")
    added, jsonb_typing, drift = await _phase3_jsonb(
        storage, db, meanings_path=meanings_path, sqlite_path=sqlite_path,
    )
    print(f"  {added} leaf columns added, {len(jsonb_typing)} typing warnings, "
          f"{len(drift)} drift findings")

    typing_warnings_path = _write_warnings(
        results_dir,
        "column_typing_warnings.txt",
        p2_warns + jsonb_typing,
    )
    drift_warnings_path = _write_warnings(
        results_dir, "jsonb_drift_warnings.txt", drift
    )
    print(f"  -> {typing_warnings_path}")
    print(f"  -> {drift_warnings_path}")

    if skip_phase4:
        print("[phase 4] skipped (--skip-phase4)")
        return 0

    print(f"[phase 4] LLM TEXT-as-date detection ({db})")
    model_id = llm_model or get_default_llm_model()
    retyped, p4_warns = await _phase4_dates(storage, db, sqlite_path=sqlite_path, llm_model=model_id)
    print(f"  {retyped} columns retyped TIMESTAMP, {len(p4_warns)} warnings")
    date_warnings_path = _write_warnings(
        results_dir, "date_detection_warnings.txt", p4_warns
    )
    print(f"  -> {date_warnings_path}")

    return 0


def run(
    db: str,
    mini_interact_root: Path = DEFAULT_MINI_INTERACT_ROOT,
    slayer_storage: Path = DEFAULT_SLAYER_STORAGE,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    llm_model: Optional[str] = None,
    wipe: bool = True,
    skip_phase1: bool = False,
    skip_phase4: bool = False,
) -> int:
    return asyncio.run(
        regenerate(
            db=db,
            mini_interact_root=mini_interact_root,
            slayer_storage=slayer_storage,
            results_root=results_root,
            llm_model=llm_model,
            wipe=wipe,
            skip_phase1=skip_phase1,
            skip_phase4=skip_phase4,
        )
    )
