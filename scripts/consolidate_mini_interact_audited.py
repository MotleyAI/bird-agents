"""Consolidate mini-interact per-DB audited-gold JSONLs into a single file.

Merges every `audited_gold/<db>/<db>_audited.jsonl` into
`audited_gold/mini_interact_audited.jsonl` while:

* Adding `benchmark: "mini_interact"` (required by harness defence-in-depth
  for single_file layouts).
* Adding `variant_id: "primary"` and `primary: true` (DEV-1515 multi-variant
  support — all currently-existing rows are tagged as the primary variant;
  multi-variant tasks will add additional rows later).

Also retro-tags `audited_gold/livesqlbench_audited.jsonl` rows with the same
`variant_id`/`primary` fields for layout consistency across benchmarks.

Idempotent: skips fields that are already present with the expected values.
Run once; commit; delete per-DB files.
"""
from __future__ import annotations

import json

from bird_interact_agents import paths

AUDITED = paths.audited_gold_root()

MINI_INTERACT_OUT = AUDITED / "mini_interact_audited.jsonl"
LIVESQLBENCH = AUDITED / "livesqlbench_audited.jsonl"


def add_multivariant_fields(row: dict, *, benchmark: str) -> dict:
    """Return a row with benchmark/variant_id/primary fields ensured."""
    out = dict(row)
    if out.get("benchmark") not in (None, benchmark):
        raise RuntimeError(
            f"row has benchmark={out.get('benchmark')!r}; expected "
            f"{benchmark!r} (instance_id={out.get('instance_id')!r})"
        )
    out.setdefault("benchmark", benchmark)
    out.setdefault("variant_id", "primary")
    out.setdefault("primary", True)
    return out


def consolidate_mini_interact() -> None:
    per_db_files = sorted(
        p for p in AUDITED.iterdir()
        if p.is_dir() and (p / f"{p.name}_audited.jsonl").exists()
    )
    if not per_db_files:
        print("no per-DB JSONLs to consolidate")
        return
    print(f"found {len(per_db_files)} per-DB JSONLs")
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for db_dir in per_db_files:
        db = db_dir.name
        path = db_dir / f"{db}_audited.jsonl"
        n = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                row_db = d.get("selected_database")
                if row_db != db:
                    raise RuntimeError(
                        f"row in {path} has selected_database={row_db!r}, "
                        f"expected {db!r} (instance_id={d.get('instance_id')!r})"
                    )
                inst = d.get("instance_id")
                if not inst:
                    raise RuntimeError(f"row missing instance_id in {path}")
                key = (inst, d.get("variant_id", "primary"))
                if key in seen:
                    raise RuntimeError(
                        f"duplicate (instance_id, variant_id)={key} in {path}"
                    )
                seen.add(key)
                rows.append(add_multivariant_fields(d, benchmark="mini_interact"))
                n += 1
        print(f"  {db:40s}  {n:>3} rows")
    print(f"total rows: {len(rows)}")
    MINI_INTERACT_OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(f"wrote {MINI_INTERACT_OUT}")


def retro_tag_livesqlbench() -> None:
    if not LIVESQLBENCH.exists():
        print(f"{LIVESQLBENCH} absent; skipping retro-tag")
        return
    rows: list[dict] = []
    with LIVESQLBENCH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(add_multivariant_fields(d, benchmark="livesqlbench"))
    LIVESQLBENCH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(f"retro-tagged {len(rows)} rows in {LIVESQLBENCH}")


if __name__ == "__main__":
    consolidate_mini_interact()
    retro_tag_livesqlbench()
