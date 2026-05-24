#!/usr/bin/env python3
"""Diff SAR-audit vs in-house audit for one mini-interact DB.

Usage:
    python scripts/compare_sar_vs_inhouse.py --db credit
                                            [--write-json /path/to/out.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.sar_audit import compare, loader


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare SAR-audit vs in-house audit.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--write-json", type=Path, default=None)
    args = ap.parse_args(argv)

    inhouse_path = paths.audited_gold_root() / args.db / f"{args.db}_audited.jsonl"
    sar_path = paths.sar_audited_gold_root() / args.db / f"{args.db}_sar_audited.jsonl"
    db_path = loader.locate_db_sqlite(db=args.db, mini_interact_root=paths.mini_interact_root())

    out = compare.compare_db(
        db=args.db, inhouse_path=inhouse_path, sar_path=sar_path, db_path=db_path
    )
    print(compare.render_markdown(out))

    json_path = args.write_json or (paths.sar_audited_gold_root() / args.db / "compare_to_inhouse.json")
    compare.write_json(out, json_path)
    print(f"\nJSON written to {json_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
