#!/usr/bin/env python3
"""Append a default deferred-section stub to a per-DB notes file for every
KB id the verifier currently flags as unaccounted.

One-shot fixup tool for W4b cases where an agent hit its usage limit
while writing the notes file. The stub uses a generic Reason / Status —
edit by hand for KB ids that warrant a more specific note.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import verify_kb_coverage as v  # type: ignore

from bird_interact_agents import paths

NOTES_DIR = paths.slayer_models_root() / "_notes"
DEFAULT_MINI_INTERACT_ROOT = paths.mini_interact_root()

STUB_TEMPLATE = """\
## KB {id} — {name}

Reason: not encoded during the W4b parallel translation pass (the
agent hit its usage limit before completing this entry).
Status: deferred to follow-up encoding pass.

"""


async def fixup(db: str, mini_interact_root: Path) -> int:
    notes_path = NOTES_DIR / f"{db}.md"
    if not notes_path.exists():
        notes_path.write_text(
            f"# {db} — KB entries not encoded as model entities\n\n",
            encoding="utf-8",
        )
    unaccounted, _overlap, knowledge = await v.verify_one(db, mini_interact_root)
    if not unaccounted:
        print(f"[OK] {db}: nothing to append")
        return 0
    with notes_path.open("a", encoding="utf-8") as f:
        for kb_id in sorted(unaccounted):
            f.write(STUB_TEMPLATE.format(id=kb_id, name=knowledge.get(kb_id, "")))
    print(f"[OK] {db}: appended {len(unaccounted)} deferred-section stubs")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="DB name")
    p.add_argument(
        "--mini-interact-root",
        default=str(DEFAULT_MINI_INTERACT_ROOT),
    )
    args = p.parse_args()
    return asyncio.run(fixup(args.db, Path(args.mini_interact_root).resolve()))


if __name__ == "__main__":
    sys.exit(main())
