#!/usr/bin/env python3
"""Regenerate the SLayer datasource + models for one BIRD-Interact mini DB.

Runs phases 1-4 of the rework pipeline against the live SLayer storage
pointed to by ``$SLAYER_STORAGE`` (default ``~/.local/share/slayer``):

1. ``slayer datasources create`` + ``slayer ingest``
2. ``column_meaning`` overlay (descriptions + leading-type-token typing
   + DEV-1381 date-format annotations)
3. JSONB-leaf expansion (full-path-named Columns with ``JSON_EXTRACT``
   sql, copied descriptions, ``meta.derived_from``)
4. LLM TEXT-as-date detection (samples live values; retypes confident
   matches to TIMESTAMP with a SQLite-native parse expression)

KB encoding (phase 5) and ``scripts/verify_kb_coverage.py`` (phase 6)
are caller-side; the ``translate-mini-interact-kb`` skill wraps this
script and runs them.

Usage:
    python scripts/regenerate_slayer_model.py --db households
    python scripts/regenerate_slayer_model.py --db solar --skip-phase4
    BIRD_AGENTS_LLM_MODEL=claude-sonnet-4-6 \\
        python scripts/regenerate_slayer_model.py --db households
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `src/` importable when running directly without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bird_interact_agents.slayer_pipeline.orchestrator import (  # noqa: E402
    DEFAULT_MINI_INTERACT_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SLAYER_STORAGE,
    run,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="DB / datasource name (e.g. households).")
    p.add_argument(
        "--mini-interact-root",
        default=str(DEFAULT_MINI_INTERACT_ROOT),
        help=f"Path to mini-interact/ (default: {DEFAULT_MINI_INTERACT_ROOT}).",
    )
    p.add_argument(
        "--slayer-storage",
        default=os.environ.get("SLAYER_STORAGE", str(DEFAULT_SLAYER_STORAGE)),
        help=(
            "Live SLayer storage path (default $SLAYER_STORAGE or "
            f"{DEFAULT_SLAYER_STORAGE})."
        ),
    )
    p.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Where to write phase warnings (default: {DEFAULT_RESULTS_ROOT}).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the phase-4 LLM model (default: $BIRD_AGENTS_LLM_MODEL or claude-haiku-4-5).",
    )
    p.add_argument(
        "--no-wipe",
        action="store_true",
        help="Skip wiping the DB's existing models from SLAYER_STORAGE before phase 1.",
    )
    p.add_argument(
        "--skip-phase1",
        action="store_true",
        help=(
            "Skip `slayer datasources create` + `slayer ingest`. Use when live storage "
            "already has the models (e.g. ported from slayer_models/<db>/ via cp) and "
            "you only want to layer phases 2-4 on top."
        ),
    )
    p.add_argument(
        "--skip-phase4",
        action="store_true",
        help="Skip the LLM TEXT-as-date detection step (useful when ANTHROPIC_API_KEY is unset).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return run(
        db=args.db,
        mini_interact_root=Path(args.mini_interact_root).expanduser().resolve(),
        slayer_storage=Path(args.slayer_storage).expanduser().resolve(),
        results_root=Path(args.results_root).expanduser().resolve(),
        llm_model=args.model,
        wipe=not args.no_wipe,
        skip_phase1=args.skip_phase1,
        skip_phase4=args.skip_phase4,
    )


if __name__ == "__main__":
    sys.exit(main())
