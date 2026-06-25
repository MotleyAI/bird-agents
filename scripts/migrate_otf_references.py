#!/usr/bin/env python
"""DEV-1605: migrate legacy FLAT OTF references into the versioned layout.

The pre-DEV-1605 layout stored one reference per ``(benchmark, db)`` at
``slayer_models_otf/<benchmark>/<db>/``. DEV-1605 versions references by
encoder model: ``slayer_models_otf/<benchmark>/<version>/<db>/``, and ABOLISHES
the legacy flat fallback — so a flat ``<db>/`` is no longer found by the
consumer. This one-shot, idempotent migration relocates each flat ``<db>/``
into ``<version>/<db>/``, deriving ``<version>`` from the reference's own
``_setup_usage.json`` (the ``setup_encoder::<model>`` breakdown → slug). When
the model can't be derived, the dir is moved under ``unknown/`` with a loud
warning so the operator can rebuild/relabel it. Already-versioned references
are left untouched.

Usage:
    uv run python scripts/migrate_otf_references.py [<benchmark> ...] [--dry-run]

With no benchmark args, migrates every known benchmark.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

from bird_interact_agents import paths
from bird_interact_agents.benchmark import benchmark_names
from bird_interact_agents.model_string import encoder_version_slug
from bird_interact_agents.slayer_otf.encoder_types import (
    EncoderMeta,
    EncoderMetaSettings,
)

_MARKER = "_reference_fp.txt"
_SETUP_USAGE = "_setup_usage.json"
_ENCODER_META = "_encoder_meta.json"
_UNKNOWN = "unknown"


class Move(BaseModel):
    """One planned/applied relocation of a flat ``<db>/`` → ``<version>/<db>/``."""

    benchmark: str
    db: str
    version: str
    derived: bool  # True when the version came from _setup_usage.json


class MigrationReport(BaseModel):
    moves: list[Move] = []


def _derive_version(db_dir: Path) -> tuple[str, bool]:
    """Return ``(version, derived)`` for a flat reference dir. Reads the
    ``setup_encoder`` breakdown model from ``_setup_usage.json``; falls back to
    ``unknown`` (derived=False) when absent / unparseable / empty."""
    usage_fp = db_dir / _SETUP_USAGE
    if not usage_fp.is_file():
        return _UNKNOWN, False
    try:
        data = json.loads(usage_fp.read_text())
    except (ValueError, OSError):
        return _UNKNOWN, False
    for row in data.get("breakdown", []):
        if row.get("scope") == "setup_encoder" and row.get("model"):
            try:
                return encoder_version_slug(row["model"]), True
            except ValueError:
                return _UNKNOWN, False
    return _UNKNOWN, False


def _is_flat_reference(db_dir: Path) -> bool:
    """A flat reference is a dir directly containing ``_reference_fp.txt``."""
    return (db_dir / _MARKER).is_file()


def _backfill_encoder_meta(
    dest_dir: Path, *, benchmark: str, db: str, version: str, derived: bool,
) -> None:
    """Write an ``_encoder_meta.json`` for a migrated reference if one is not
    already present, reconstructing what we can from the flat artefacts."""
    meta_fp = dest_dir / _ENCODER_META
    if meta_fp.is_file():
        return
    fp = (dest_dir / _MARKER).read_text().strip()
    encoder_model = _UNKNOWN
    usage_fp = dest_dir / _SETUP_USAGE
    if usage_fp.is_file():
        try:
            for row in json.loads(usage_fp.read_text()).get("breakdown", []):
                if row.get("scope") == "setup_encoder" and row.get("model"):
                    encoder_model = row["model"]
                    break
        except (ValueError, OSError):
            pass
    built_at = _dt.datetime.fromtimestamp(
        (dest_dir / _MARKER).stat().st_mtime, tz=_dt.timezone.utc,
    ).replace(microsecond=0).isoformat()
    meta = EncoderMeta(
        version=version,
        encoder_model=encoder_model,
        encoder_framework=_UNKNOWN,
        benchmark=benchmark,
        db=db,
        reference_fp=fp,
        built_at=built_at,
        settings=EncoderMetaSettings(version_was_explicit=not derived),
    )
    meta_fp.write_text(meta.model_dump_json(indent=2))


def migrate_benchmark(*, benchmark: str, dry_run: bool = False) -> MigrationReport:
    """Migrate every flat reference under ``<benchmark>`` into ``<version>/<db>``.

    Idempotent: dirs that are already versioned (no direct marker) are skipped.
    """
    report = MigrationReport()
    parent = paths.slayer_models_otf_root(benchmark=benchmark)  # version=None
    if not parent.is_dir():
        return report
    for db_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
        if not _is_flat_reference(db_dir):
            # Already a version dir (or an incomplete scrap) — skip.
            continue
        db = db_dir.name
        version, derived = _derive_version(db_dir)
        report.moves.append(
            Move(benchmark=benchmark, db=db, version=version, derived=derived)
        )
        if not derived:
            sys.stderr.write(
                f"[migrate_otf_references] WARNING: could not derive encoder "
                f"version for {benchmark}/{db} (no usable _setup_usage.json); "
                f"moving under '{_UNKNOWN}/'. Rebuild or relabel it.\n"
            )
        if dry_run:
            continue
        dest = paths.slayer_models_otf_root(
            benchmark=benchmark, version=version,
        ) / db
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            sys.stderr.write(
                f"[migrate_otf_references] WARNING: destination {dest} already "
                f"exists; leaving flat {db_dir} in place for manual review.\n"
            )
            report.moves.pop()
            continue
        os.rename(db_dir, dest)
        _backfill_encoder_meta(
            dest, benchmark=benchmark, db=db, version=version, derived=derived,
        )
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmarks", nargs="*",
        help="Benchmark tokens to migrate (default: all known benchmarks).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    targets = args.benchmarks or list(benchmark_names())
    total = 0
    for bench in targets:
        report = migrate_benchmark(benchmark=bench, dry_run=args.dry_run)
        for m in report.moves:
            verb = "would move" if args.dry_run else "moved"
            print(f"{verb} {m.benchmark}/{m.db} -> {m.version}/{m.db}")
        total += len(report.moves)
    print(f"{'(dry-run) ' if args.dry_run else ''}{total} reference(s) migrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
