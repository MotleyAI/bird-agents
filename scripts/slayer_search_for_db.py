#!/usr/bin/env python3
"""Per-DB SLayer search wrapper.

Boots a short-lived `SearchService` against `slayer_models/<db>/` so that
search results cannot leak across databases. Used by the encoder skill
(`kb-to-slayer-models`) to drive a search-first preamble, and by
`verify_kb_coverage.py` to check whether a KB id is documented as a
SLayer memory.

The session-wide `mcp__slayer__search` tool would search the shared
storage that the MCP server is rooted at; this script gives the same
SearchResponse shape but scoped to a single DB.

Usage:
    python scripts/slayer_search_for_db.py --db households \\
        --question "weighted score combining domestic help and social assistance" \\
        [--max-results 12] [--compact / --no-compact]

Prints the SearchResponse as JSON to stdout. Exits 0 on success, 1 if
the per-DB storage directory does not exist or the search fails.

DEV-1546 / DEV-1549: SLayer 0.7.2 collapsed the previous per-kind caps
(``--max-memories`` / ``--max-example-queries`` / ``--max-entities``)
into a single ``--max-results`` cap on the RRF-fused
``SearchResponse.results`` list. SLayer 0.7.3 added the ``compact``
flag, surfaced here as ``--compact`` / ``--no-compact``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from slayer.search.service import SearchService
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents import paths

DEFAULT_SLAYER_MODELS_DIR = paths.slayer_models_root()


async def _search(
    *,
    db: str,
    question: str,
    slayer_models_dir: Path,
    max_results: int,
    compact: bool,
) -> dict:
    """DEV-1546: slayer 0.7.2 collapsed the per-kind ``max_memories`` /
    ``max_example_queries`` / ``max_entities`` caps into a single
    ``max_results`` cap on the RRF-fused unified ``results`` list."""
    db_dir = slayer_models_dir / db
    if not db_dir.is_dir():
        raise FileNotFoundError(
            f"Per-DB SLayer storage not found: {db_dir}. "
            "Run the regenerate + KB-encoding pipeline first."
        )
    storage = YAMLStorage(base_dir=str(db_dir))
    service = SearchService(storage=storage)
    response = await service.search(
        question=question,
        max_results=max_results,
        compact=compact,
    )
    return response.model_dump(mode="json")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Per-DB SLayer search wrapper — boots a short-lived "
            "SearchService against one database's YAMLStorage view."
        ),
    )
    p.add_argument("--db", required=True, help="Database name (e.g. 'households').")
    p.add_argument(
        "--question", required=True,
        help="Free-text query routed to BM25 + tantivy + (optional) dense embeddings.",
    )
    p.add_argument(
        "--slayer-models-dir",
        default=str(DEFAULT_SLAYER_MODELS_DIR),
        help=(
            "Root of the per-DB SLayer storage trees "
            f"(default: {DEFAULT_SLAYER_MODELS_DIR})."
        ),
    )
    p.add_argument(
        "--max-results", type=int, default=12,
        help=(
            "DEV-1546: slayer 0.7.2 collapsed the previous per-kind caps "
            "(max-memories / max-example-queries / max-entities) into a "
            "single cap on the RRF-fused results list. Default 12 ≈ the "
            "sum of the pre-1546 defaults (5 + 2 + 5)."
        ),
    )
    compact_group = p.add_mutually_exclusive_group()
    compact_group.add_argument(
        "--compact", dest="compact", action="store_true", default=True,
        help="Compact mode (default): one-line `description` per hit.",
    )
    compact_group.add_argument(
        "--no-compact", dest="compact", action="store_false",
        help="Return the full `text` body on every hit (verbose).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        out = asyncio.run(
            _search(
                db=args.db,
                question=args.question,
                slayer_models_dir=Path(args.slayer_models_dir).expanduser().resolve(),
                max_results=args.max_results,
                compact=args.compact,
            )
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — surface every failure
        print(f"[ERROR] search failed: {e}", file=sys.stderr)
        return 1
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
