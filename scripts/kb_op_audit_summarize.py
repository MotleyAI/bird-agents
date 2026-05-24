#!/usr/bin/env python3
"""Summarize the KB-vs-data audit into a markdown report + a per-row CSV.

Joins ``operator_tuples.jsonl`` and ``judgments.jsonl`` (outputs of the
extract + classify steps) and produces:

* ``KB_AUDIT.md`` — headline → per-bucket sections (each with per-DB
  sub-table + KB-silent examples) → KB-silent global table → per-bucket
  recommendation → methodology appendix.
* ``kb_silent.csv`` — every (instance_id, bucket, operator, expression,
  target_columns, reason) row whose verdict was "silent".

Usage::

    uv run python scripts/kb_op_audit_summarize.py \\
        --output-dir results/kb_op_audit_$(date +%Y%m%d)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Recommendation thresholds (KB-silent fraction within a bucket).
SILENT_RICH_ENOUGH = 0.05
SILENT_NEEDS_DSL = 0.50

BUCKET_NAMES: dict[str, str] = {
    "A": "String-hygiene",
    "B": "Null / conditional",
    "C": "Typing / JSON",
}
BUCKET_OPS: dict[str, list[str]] = {
    "A": ["LOWER", "UPPER", "TRIM", "REPLACE", "SUBSTR", "INSTR", "LENGTH",
          "||", "STRFTIME"],
    "B": ["CASE WHEN", "COALESCE", "NULLIF", "IS NULL", "IS NOT NULL"],
    "C": ["CAST", "JSON_EXTRACT", "->", "->>"],
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _hash_key(row: dict) -> tuple[str, int, str, str]:
    """Match ``kb_op_audit_classify._key`` so we can join."""
    import hashlib
    return (
        row["instance_id"],
        row["sql_index"],
        row["operator"],
        hashlib.sha1(row["expression"].encode("utf-8")).hexdigest()[:12],
    )


def join_tuples_with_judgments(
    tuples: list[dict], judgments: list[dict]
) -> list[dict]:
    """Return tuples enriched with verdict/reason/kb_evidence_ids.

    Tuples without a matching judgment are dropped (with a count) — those
    indicate the classifier hasn't completed for them yet.
    """
    j_by_key = {
        (
            j["instance_id"],
            j["sql_index"],
            j["operator"],
            j["expression_hash"],
        ): j
        for j in judgments
    }
    out: list[dict] = []
    missing = 0
    for t in tuples:
        k = _hash_key(t)
        if k in j_by_key:
            j = j_by_key[k]
            out.append({
                **t,
                "verdict": j["verdict"],
                "reason": j["reason"],
                "kb_evidence_ids": j["kb_evidence_ids"],
            })
        else:
            missing += 1
    if missing:
        print(f"WARN: {missing} tuples lack judgments (classifier incomplete)",
              file=sys.stderr)
    return out


def _silent_aware_counts(rows: Iterable[dict]) -> tuple[int, int, int]:
    """Return (n_total, n_aware, n_silent). ERROR verdicts go in n_total only."""
    n_total = n_aware = n_silent = 0
    for r in rows:
        n_total += 1
        v = r.get("verdict")
        if v == "aware":
            n_aware += 1
        elif v == "silent":
            n_silent += 1
    return n_total, n_aware, n_silent


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{n / total * 100:.1f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _example_blocks(
    rows: list[dict], verdict: str, n: int = 3, max_chars: int = 280
) -> list[str]:
    """Return up to *n* short markdown blocks for examples of the given verdict."""
    candidates = [r for r in rows if r.get("verdict") == verdict]
    candidates.sort(key=lambda r: (r["selected_database"], r["instance_id"]))
    out: list[str] = []
    for r in candidates[:n]:
        expr = r["expression"]
        if len(expr) > max_chars:
            expr = expr[: max_chars - 1] + "…"
        evidence = r.get("kb_evidence_ids", [])
        ev_str = (
            f" (KB evidence: {evidence})" if evidence else ""
        )
        reason = r.get("reason", "").replace("\n", " ")
        out.append(
            f"- **{r['instance_id']}** ({r['operator']} in `{r['cte_or_clause']}`)"
            f"{ev_str}\n"
            f"  - SQL: `{expr}`\n"
            f"  - Reason: {reason}"
        )
    return out


def _per_op_table(rows: list[dict], bucket: str) -> str:
    """Markdown table: per-operator total / aware / silent within *bucket*."""
    by_op: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["bucket"] == bucket:
            by_op[r["operator"]].append(r)
    headers = ["Operator", "Total", "Aware", "Silent", "Silent %"]
    table_rows: list[list[str]] = []
    # Order by canonical operator order, then any extras alphabetically.
    canonical = BUCKET_OPS[bucket]
    extras = sorted(set(by_op.keys()) - set(canonical))
    for op in canonical + extras:
        if op not in by_op:
            continue
        n_total, n_aware, n_silent = _silent_aware_counts(by_op[op])
        table_rows.append([
            f"`{op}`",
            str(n_total),
            str(n_aware),
            str(n_silent),
            _pct(n_silent, n_total),
        ])
    return _md_table(headers, table_rows)


def _per_db_table(rows: list[dict], bucket: str) -> str:
    """Markdown table: per-DB total / aware / silent within *bucket*. Sorted
    by silent count desc so the worst-coverage DBs surface."""
    by_db: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["bucket"] == bucket:
            by_db[r["selected_database"]].append(r)
    headers = ["DB", "Total", "Aware", "Silent", "Silent %"]
    triples = []
    for db, items in by_db.items():
        triples.append((db, _silent_aware_counts(items)))
    triples.sort(key=lambda x: (-x[1][2], x[0]))
    table_rows = [
        [
            db,
            str(t[0]),
            str(t[1]),
            str(t[2]),
            _pct(t[2], t[0]),
        ]
        for db, t in triples
    ]
    return _md_table(headers, table_rows)


def _bucket_recommendation(rows: list[dict], bucket: str) -> str:
    bucket_rows = [r for r in rows if r["bucket"] == bucket]
    n_total, _, n_silent = _silent_aware_counts(bucket_rows)
    silent_frac = n_silent / n_total if n_total else 0.0
    by_op = Counter(
        r["operator"] for r in bucket_rows if r.get("verdict") == "silent"
    )
    top_silent_ops = ", ".join(
        f"`{op}` ({n})" for op, n in by_op.most_common(5)
    )
    if silent_frac < SILENT_RICH_ENOUGH:
        verdict = (
            f"**KB rich enough** — only {silent_frac:.1%} of usages are "
            f"KB-silent. Encode the transforms in `Column.sql` and move on; "
            f"DSL growth is not justified here."
        )
    elif silent_frac < SILENT_NEEDS_DSL:
        verdict = (
            f"**KB needs annotation enrichment** — {silent_frac:.1%} of "
            f"usages are KB-silent. The KB names the right concepts but "
            f"misses the transform-cuing details. Top silent operators: "
            f"{top_silent_ops}. Address by adding annotations (e.g. "
            f"value-format notes, JSON-shape notes, NULL-policy notes) to "
            f"the relevant KB items rather than growing the DSL."
        )
    else:
        verdict = (
            f"**DSL needs to grow** — {silent_frac:.1%} of usages are "
            f"KB-silent. Across 26 DBs the KB simply doesn't carry the "
            f"information; encoders cannot bake the transforms into "
            f"`Column.sql` without out-of-bound data inspection. Top "
            f"silent operators: {top_silent_ops}. Add these to the DSL as "
            f"inline transforms so Mode B can apply them per-query."
        )
    return verdict


def _write_kb_silent_csv(rows: list[dict], path: Path) -> int:
    silent_rows = [r for r in rows if r.get("verdict") == "silent"]
    silent_rows.sort(
        key=lambda r: (r["bucket"], r["selected_database"], r["instance_id"])
    )
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id",
            "selected_database",
            "is_hard8",
            "bucket",
            "operator",
            "expression",
            "target_columns",
            "literal_args",
            "cte_or_clause",
            "reason",
        ])
        for r in silent_rows:
            writer.writerow([
                r["instance_id"],
                r["selected_database"],
                r["is_hard8"],
                r["bucket"],
                r["operator"],
                r["expression"],
                ";".join(r["target_columns"]),
                ";".join(str(x) for x in r["literal_args"]),
                r["cte_or_clause"],
                r.get("reason", ""),
            ])
    return len(silent_rows)


def build_markdown(rows: list[dict]) -> str:
    sections: list[str] = []
    sections.append(
        "# KB-vs-data audit — gold-SQL operators in BIRD-Interact mini\n"
    )
    sections.append(
        "Per-task audit of every gold-SQL operator usage in "
        "`mini-interact/mini_interact.jsonl`, classified by Claude Haiku 4.5 "
        "as **KB-aware** (a competent encoder/agent could derive the need for "
        "the transform from the eval-time text bundle alone — KB jsonl + "
        "schema + column_meaning, post HARD-8 deletions where applicable) "
        "or **KB-silent** (the transform is only discoverable by inspecting "
        "raw data, which Mode-B SLayer DSL cannot do at runtime). See "
        "`~/.claude/plans/kb-string-operator-audit.md` for the full spec.\n"
    )

    n_total, n_aware, n_silent = _silent_aware_counts(rows)
    sections.append("## Headline\n")
    sections.append(
        f"Across {n_total} operator usages from "
        f"{len(set(r['instance_id'] for r in rows))} tasks across "
        f"{len(set(r['selected_database'] for r in rows))} DBs: "
        f"**{n_aware} ({_pct(n_aware, n_total)}) KB-aware**, "
        f"**{n_silent} ({_pct(n_silent, n_total)}) KB-silent**.\n"
    )
    bucket_rows: list[list[str]] = []
    for b in ("A", "B", "C"):
        items = [r for r in rows if r["bucket"] == b]
        bn_total, bn_aware, bn_silent = _silent_aware_counts(items)
        bucket_rows.append([
            f"{b}. {BUCKET_NAMES[b]}",
            str(bn_total),
            str(bn_aware),
            str(bn_silent),
            _pct(bn_silent, bn_total),
        ])
    sections.append(
        _md_table(
            ["Bucket", "Total", "Aware", "Silent", "Silent %"],
            bucket_rows,
        )
        + "\n"
    )

    for b in ("A", "B", "C"):
        items = [r for r in rows if r["bucket"] == b]
        if not items:
            continue
        sections.append(f"## Bucket {b}: {BUCKET_NAMES[b]}\n")
        sections.append(f"### Per-operator within bucket {b}\n")
        sections.append(_per_op_table(rows, b) + "\n")
        sections.append(f"### Per-DB within bucket {b}\n")
        sections.append(_per_db_table(rows, b) + "\n")
        sections.append(f"### Recommendation for bucket {b}\n")
        sections.append(_bucket_recommendation(rows, b) + "\n")
        sections.append(f"### Example KB-silent rows (bucket {b})\n")
        examples = _example_blocks(items, "silent", n=5)
        if examples:
            sections.append("\n".join(examples) + "\n")
        else:
            sections.append("(no silent rows in this bucket)\n")
        sections.append(f"### Example KB-aware rows (bucket {b})\n")
        aware_examples = _example_blocks(items, "aware", n=3)
        if aware_examples:
            sections.append("\n".join(aware_examples) + "\n")
        else:
            sections.append("(no aware rows in this bucket)\n")

    sections.append("## KB-silent rows (full table)\n")
    sections.append(
        "Full KB-silent list also written to `kb_silent.csv` for "
        "spreadsheet consumption.\n"
    )

    sections.append("## Methodology\n")
    sections.append(
        "- **Operator extraction**: sqlglot AST traversal of every "
        "`sol_sql` string (`scripts/kb_op_audit_extract.py`).\n"
        "- **Classifier**: Claude `claude-haiku-4-5-20251001` with per-"
        "(db, deletion-set) cached system prompt holding the eval-time "
        "text bundle (`scripts/kb_op_audit_classify.py`). For HARD-8 "
        "tasks the KB jsonl is post-deletion. Schema + column_meaning "
        "are always included.\n"
        "- **KB-aware bar (definition-derivable)**: the transform must "
        "be derivable from the eval-time text without inspecting raw "
        "data. A KB item that *defines* a metric whose computation "
        "requires the operator counts as a cue; a KB item that merely "
        "names the column does not.\n"
        "- **Excluded sources**: `execute()` SQL sampling at runtime "
        "(by design — Mode B can't apply transforms not baked into "
        "Column.sql at encode time).\n"
        "- **Recommendation thresholds**: bucket KB-silent fraction "
        f"<{SILENT_RICH_ENOUGH:.0%} → 'KB rich enough'; "
        f"<{SILENT_NEEDS_DSL:.0%} → 'KB needs annotation enrichment'; "
        f"≥{SILENT_NEEDS_DSL:.0%} → 'DSL needs to grow'.\n"
    )
    return "\n".join(sections)


def run(*, output_dir: Path) -> int:
    tuples_path = output_dir / "operator_tuples.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    audit_md_path = output_dir / "KB_AUDIT.md"
    csv_path = output_dir / "kb_silent.csv"

    tuples = _load_jsonl(tuples_path)
    judgments = _load_jsonl(judgments_path)
    if not tuples:
        print(f"ERROR: missing or empty {tuples_path}", file=sys.stderr)
        return 2
    if not judgments:
        print(f"ERROR: missing or empty {judgments_path}", file=sys.stderr)
        return 2

    rows = join_tuples_with_judgments(tuples, judgments)
    if not rows:
        print(
            "ERROR: no rows after joining operator_tuples.jsonl with "
            "judgments.jsonl; check classifier completeness / key compatibility.",
            file=sys.stderr,
        )
        return 2
    md = build_markdown(rows)
    audit_md_path.write_text(md)
    n_silent = _write_kb_silent_csv(rows, csv_path)

    _, n_aware, n_silent_count = _silent_aware_counts(rows)
    print(f"Rows joined:   {len(rows)}")
    print(f"  KB-aware:    {n_aware}")
    print(f"  KB-silent:   {n_silent_count}")
    print(f"Wrote: {audit_md_path}")
    print(f"Wrote: {csv_path}  ({n_silent} silent rows)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return run(output_dir=Path(args.output_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
