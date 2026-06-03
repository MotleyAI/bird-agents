#!/usr/bin/env python3
"""DEV-1462 — materialize the stable per-DB sqlite for LiveSQLBench-Base-Lite.

The upstream dataset ships per-DB `<db>_template.sqlite` files as git-LFS
pointers. The OTF cache (`ensure_db_cache` + `fingerprint_of`) reads the
STABLE `<root>/<db>/<db>.sqlite` directly — so the dataset's flat layout
must be augmented with that stable file before any cache build.

This script does exactly that, per-DB:

  1. Refuse if `<db>_template.sqlite` is still an LFS pointer (file
     starts with ``version https://git-lfs…``). Exit non-zero with the
     ``git lfs pull`` remediation — silently ingesting a 132-byte
     pointer corrupts the fingerprint and poisons every downstream task.
  2. Otherwise copy ``<db>_template.sqlite`` → ``<db>.sqlite`` in the
     same dir. Idempotent (size+mtime skip); ``--force`` rewrites.

The script is small on purpose — the heavy plumbing (cache build, eval
reset) lives in ``slayer_otf`` and the upstream action handler.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Make the package importable when this script is run directly
# (`python scripts/prepare_livesqlbench.py`) from a src-layout checkout
# that hasn't `pip install -e`'d the package. Mirrors the bootstrap in
# scripts/export_slayer_models.py — a no-op once the package is installed
# (e.g. under `uv run`). (Codex review.)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bird_interact_agents import paths  # noqa: E402

_LFS_POINTER_PREFIX = b"version https://git-lfs"


def _is_lfs_pointer(p: Path) -> bool:
    """True iff ``p`` starts with the git-LFS pointer header. Small file
    read; we only need the first ~30 bytes."""
    try:
        with p.open("rb") as fh:
            head = fh.read(32)
    except OSError:
        return False
    return head.startswith(_LFS_POINTER_PREFIX)


def _materialise_one(
    db: str, root: Path, *, force: bool,
) -> tuple[str, str]:
    """Materialize ``<root>/<db>/<db>.sqlite`` from the template.

    Returns ``(status, detail)``:
      * ``("refused-lfs", path)`` — template is still an LFS pointer.
      * ``("missing-template", path)`` — template file absent.
      * ``("materialised", path)`` — copied template → stable.
      * ``("skipped", path)`` — stable file already present and up to date.
      * ``("forced", path)`` — overwrote a (possibly up-to-date) stable
        file under ``--force``.

    The copy is atomic: written into a tmp file in the same dir, then
    renamed onto ``<db>.sqlite``. Mirrors the marker-last write pattern
    used by the slayer_otf cache (no half-written file on crash).
    """
    db_dir = root / db
    template = db_dir / f"{db}_template.sqlite"
    stable = db_dir / f"{db}.sqlite"

    if not template.is_file():
        return ("missing-template", str(template))
    if _is_lfs_pointer(template):
        return ("refused-lfs", str(template))

    # Idempotence: skip when the stable file looks current (size+mtime
    # match the template). Cheap and avoids touching the file on every
    # rerun.
    if not force and stable.is_file():
        st_t = template.stat()
        st_s = stable.stat()
        if st_t.st_size == st_s.st_size and st_t.st_mtime_ns <= st_s.st_mtime_ns:
            return ("skipped", str(stable))

    # Atomic write: copy into a tmp sibling, then rename onto the target.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{db}.tmp-", suffix=".sqlite", dir=str(db_dir),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(template, tmp_path)
        os.replace(tmp_path, stable)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return ("forced" if force and stable.is_file() else "materialised", str(stable))


def prepare(
    root: Path,
    *,
    dbs: Iterable[str] | None = None,
    force: bool = False,
) -> tuple[bool, list[tuple[str, str, str]]]:
    """Run the prepare pass over ``root``. Returns
    ``(ok, [(db, status, detail), ...])``. ``ok`` is False if any DB was
    refused for an LFS pointer.

    ``dbs=None`` walks every immediate subdir of ``root`` that contains
    a ``<db>_template.sqlite`` file — i.e. the dataset's natural per-DB
    layout. Explicit ``dbs`` restricts to that subset.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"livesqlbench root not found: {root}")

    if dbs is None:
        dbs = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and (d / f"{d.name}_template.sqlite").exists()
        )

    rows: list[tuple[str, str, str]] = []
    ok = True
    for db in dbs:
        status, detail = _materialise_one(db, root, force=force)
        rows.append((db, status, detail))
        if status == "refused-lfs":
            ok = False
    return ok, rows


def _print_summary(rows: list[tuple[str, str, str]]) -> None:
    for db, status, detail in rows:
        print(f"  {db:14s} {status:18s} {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise per-DB stable <db>.sqlite from <db>_template.sqlite "
            "for LiveSQLBench-Base-Lite-SQLite. Refuses to ingest git-LFS "
            "pointers; rerun-safe by default; --force overwrites."
        ),
    )
    parser.add_argument(
        "--root", default=None,
        help=(
            "Dataset root (default: paths.benchmark_data_root('livesqlbench-base-lite-sqlite') — i.e. the "
            "sibling `livesqlbench-base-lite-sqlite/`, or whatever "
            "$BIRD_LIVESQLBENCH_ROOT points at)."
        ),
    )
    parser.add_argument(
        "--dbs", default=None,
        help=(
            "Comma-separated subset of DB names to materialise; default is "
            "every immediate subdir that has a <db>_template.sqlite."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite the stable <db>.sqlite even when it looks current.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else paths.benchmark_data_root("livesqlbench-base-lite-sqlite")
    dbs = (
        [s.strip() for s in args.dbs.split(",") if s.strip()]
        if args.dbs else None
    )

    try:
        ok, rows = prepare(root, dbs=dbs, force=args.force)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_summary(rows)
    if not ok:
        refused = [r for r in rows if r[1] == "refused-lfs"]
        print(
            "\n"
            "ERROR: one or more templates are still git-LFS pointers (not "
            "real sqlite files). Run\n"
            f"  cd {root}\n"
            "  git lfs pull\n"
            "and try again. Refused: "
            + ", ".join(r[0] for r in refused),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
