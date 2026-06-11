"""Selection.jsonl loader.

Each line is a JSON object ``{"instance_id": "...", "run_id": "..."}``.
Duplicate ``instance_id`` is a hard error listing every duplicate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class DuplicateInstanceError(ValueError):
    pass


def load_selection(path: Path | str) -> list[tuple[str, str]]:
    """Return ``[(instance_id, run_id), …]`` in file order."""
    p = Path(path)
    out: list[tuple[str, str]] = []
    seen: list[str] = []
    with p.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                inst = obj["instance_id"]
                run = obj["run_id"]
            except KeyError as e:
                raise KeyError(
                    f"{p}:{line_no} selection entry missing required "
                    f"field {e.args[0]!r}: {obj!r}"
                )
            out.append((str(inst), str(run)))
            seen.append(str(inst))

    dupes = sorted(
        {iid for iid, count in Counter(seen).items() if count > 1}
    )
    if dupes:
        raise DuplicateInstanceError(
            f"{p}: selection has duplicate instance_id entries: "
            f"{', '.join(dupes)}. Each instance must map to exactly one run_id."
        )
    return out
