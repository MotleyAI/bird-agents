#!/usr/bin/env python3
"""Pick a stable instance_id slice from mini_interact.jsonl.

Used to lock the same set of tasks across all three runners (original,
raw, slayer) so their results are comparable.

Usage:
    python scripts/select_tasks.py \
        --data /path/to/mini_interact.jsonl \
        --limit 30 \
        --out instance_ids.txt \
        --out-jsonl tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _non_negative_int(value: str) -> int:
    """argparse type that rejects negative integers with a clear message.

    The downstream pagination logic produces nonsensical selections for
    negative offsets/limits (negative `--start` skips no records but also
    drifts the record counter; negative `--limit` makes the take-while
    branch never trigger), so fail fast at parse time instead.
    """
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {n}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to mini_interact.jsonl")
    parser.add_argument(
        "--limit", type=_non_negative_int, default=30, help="Number of tasks to take"
    )
    parser.add_argument(
        "--start",
        type=_non_negative_int,
        default=0,
        help="Start offset (0-based, in records)",
    )
    parser.add_argument("--out", required=True, help="Where to write instance_ids.txt")
    parser.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional: also write the selected rows as a filtered JSONL "
        "(used by run_three_way.sh to lock the upstream runner to the "
        "same slice).",
    )
    args = parser.parse_args()

    ids: list[str] = []
    selected_lines: list[str] = []
    record_idx = 0
    with open(args.data) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if record_idx < args.start:
                record_idx += 1
                continue
            if len(ids) >= args.limit:
                break
            row = json.loads(stripped)
            ids.append(row["instance_id"])
            # Preserve the source record verbatim (minus the trailing
            # newline; we re-add separators when writing). Otherwise
            # downstream tools that diff `--out-jsonl` against the source
            # slice see spurious whitespace edits.
            selected_lines.append(raw_line.rstrip("\n"))
            record_idx += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ids) + "\n")
    print(f"Wrote {len(ids)} instance_ids to {args.out}")

    if args.out_jsonl:
        jsonl_path = Path(args.out_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("\n".join(selected_lines) + ("\n" if selected_lines else ""))
        print(f"Wrote {len(selected_lines)} rows to {args.out_jsonl}")


if __name__ == "__main__":
    main()
