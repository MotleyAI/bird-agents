#!/usr/bin/env python3
"""Classify each (operator, surrounding-context) tuple as KB-aware or
KB-silent against the eval-time text the agent has at runtime.

Reads ``operator_tuples.jsonl`` (output of
``kb_op_audit_extract.py``) and, for each tuple, calls Claude Haiku 4.5
with a per-(db, HARD-8-deletion-set)-grouped system prompt containing
the eval-time text bundle: the DB knowledge base (post-deletion for
HARD-8 tasks), the schema, and the column-meaning sidecar.

Judgments are appended to ``judgments.jsonl`` keyed by
``(instance_id, sql_index, operator, expression_hash)``. Re-runs skip
keys already in the file, so the script is resumable across crashes.

Usage::

    ANTHROPIC_API_KEY=... uv run python scripts/kb_op_audit_classify.py \\
        --output-dir results/kb_op_audit_$(date +%Y%m%d) \\
        --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
)


from bird_interact_agents import paths

DEFAULT_MINI_INTERACT_ROOT = paths.benchmark_data_root("mini-interact")

MODEL = "claude-haiku-4-5-20251001"

# Pricing (per 1M tokens) for cost-estimate output. Approximate; if
# Anthropic publishes new rates, edit here.
COST_INPUT_PER_MTOK = 1.0
COST_INPUT_CACHE_WRITE_PER_MTOK = 1.25
COST_INPUT_CACHE_READ_PER_MTOK = 0.10
COST_OUTPUT_PER_MTOK = 5.0


SYSTEM_PROMPT_INSTRUCTIONS = """\
You are auditing a SQL benchmark for whether the eval-time text bundle
shown below is rich enough that a competent encoder/agent could derive
the need for a specific SQL transform without inspecting raw data.

You will be asked, per call, about ONE operator usage from gold SQL.
Your job is to judge whether the eval-time text bundle below makes the
transform "definition-derivable":

A. KB-aware (return verdict="aware") iff at least one of:
   1. A KB item, schema comment, or column-meaning entry EXPLICITLY
      cues the transform itself (e.g. "values are stored case-mixed",
      "column stores JSON with these keys").
   2. A KB item DEFINES the metric/concept whose computation requires
      this operator (e.g. "TOLS Category = bucket Low/Medium/High by
      these thresholds" implies CASE WHEN).
   3. A KB item ENUMERATES the actual values in a way that surfaces an
      irregularity (case-mix, whitespace, JSON shape). Clean
      enumeration does NOT count.

B. KB-silent (return verdict="silent") iff none of A.1, A.2, A.3 hold
   — the only way to discover the transform's necessity is by sampling
   raw data, which is OUT OF SCOPE.

Return strict JSON in this exact shape (no surrounding prose):
  {"verdict": "aware" | "silent", "reason": "<two sentences max>",
   "kb_evidence_ids": [<int ids>]}
``kb_evidence_ids`` lists the KB ids you cited (empty list when silent
or when evidence is in schema/column_meaning rather than KB).

=== ELIGIBLE EVAL-TIME TEXT (the agent has tool-access to all of this) ===
"""


USER_PROMPT_TEMPLATE = (
    "Operator usage to classify:\n\n"
    "- DB: {db}\n"
    "- Operator: {operator}\n"
    "- Bucket: {bucket}\n"
    "- SQL fragment: {expression}\n"
    "- Target columns: {target_columns}\n"
    "- Literal arg values: {literal_args}\n"
    "- Surrounding clause: {cte_or_clause}\n"
    "- Task instance_id: {instance_id}\n\n"
    "Apply the definition-derivable bar from the system instructions.\n"
    "Return strict JSON only.\n"
)


def _hash_expression(expr: str) -> str:
    return hashlib.sha1(expr.encode("utf-8")).hexdigest()[:12]


def _key(row: dict) -> tuple[str, int, str, str]:
    return (
        row["instance_id"],
        row["sql_index"],
        row["operator"],
        _hash_expression(row["expression"]),
    )


def _load_kb(mini_interact_root: Path, db: str) -> list[dict]:
    p = mini_interact_root / db / f"{db}_kb.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_schema(mini_interact_root: Path, db: str) -> str:
    p = mini_interact_root / db / f"{db}_schema.txt"
    return p.read_text() if p.exists() else "(missing)"


def _load_column_meaning(mini_interact_root: Path, db: str) -> str:
    p = mini_interact_root / db / f"{db}_column_meaning_base.json"
    if not p.exists():
        return "(missing)"
    raw = json.loads(p.read_text())
    return json.dumps(raw, indent=2, ensure_ascii=False)


def _format_kb(kb_items: list[dict], deleted_ids: set[int]) -> str:
    visible = [k for k in kb_items if int(k["id"]) not in deleted_ids]
    if not visible:
        return "(empty after deletions)"
    lines: list[str] = []
    for k in visible:
        kid = k.get("id")
        knowledge = (k.get("knowledge") or "").strip()
        ktype = k.get("type", "")
        defn = (k.get("definition") or "").strip()
        desc = (k.get("description") or "").strip()
        body = " — ".join(b for b in [defn, desc] if b)
        lines.append(f"[KB {kid}] [{ktype}] {knowledge}\n  {body}")
    return "\n".join(lines)


def build_system_prompt(
    *,
    db: str,
    deleted_ids: set[int],
    mini_interact_root: Path,
) -> str:
    kb_items = _load_kb(mini_interact_root, db)
    schema = _load_schema(mini_interact_root, db)
    column_meaning = _load_column_meaning(mini_interact_root, db)
    kb_text = _format_kb(kb_items, deleted_ids)
    deleted_note = (
        f", with KB ids {sorted(deleted_ids)} masked"
        if deleted_ids else ""
    )
    visible_count = sum(1 for k in kb_items if int(k["id"]) not in deleted_ids)
    return (
        SYSTEM_PROMPT_INSTRUCTIONS
        + f"\n--- KB knowledge base ({db} — {visible_count} items"
        + f"{deleted_note}) ---\n"
        + kb_text
        + f"\n\n--- Schema ({db}) ---\n"
        + schema
        + f"\n\n--- Column meaning ({db}) ---\n"
        + column_meaning
        + "\n"
    )


def build_user_prompt(row: dict) -> str:
    return USER_PROMPT_TEMPLATE.format(
        db=row["selected_database"],
        operator=row["operator"],
        bucket=row["bucket"],
        expression=row["expression"],
        target_columns=row["target_columns"],
        literal_args=row["literal_args"],
        cte_or_clause=row["cte_or_clause"],
        instance_id=row["instance_id"],
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict_json(text: str) -> dict:
    """Extract a JSON dict from Claude's response, tolerant of leading
    chatter or trailing whitespace.

    Raises ``ValueError`` if no JSON dict can be parsed.
    """
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"no JSON found in response: {text[:200]}")
    body = m.group(0)
    return json.loads(body)


class JudgmentCache:
    """Append-only JSONL cache of completed judgments, keyed by tuple key."""

    def __init__(self, path: Path):
        self.path = path
        self.seen: set[tuple[str, int, str, str]] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.seen.add(
                    (
                        rec["instance_id"],
                        rec["sql_index"],
                        rec["operator"],
                        rec["expression_hash"],
                    )
                )

    def already_done(self, row: dict) -> bool:
        return _key(row) in self.seen

    def append(self, judgment: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(judgment, ensure_ascii=False) + "\n")
        self.seen.add(
            (
                judgment["instance_id"],
                judgment["sql_index"],
                judgment["operator"],
                judgment["expression_hash"],
            )
        )


async def classify_one(
    *,
    client: AsyncAnthropic,
    system_prompt: str,
    row: dict,
    semaphore: asyncio.Semaphore,
    max_attempts: int = 3,
) -> dict:
    """Classify one tuple, retrying transient errors. Returns a judgment dict."""
    user_prompt = build_user_prompt(row)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        retry_delay: Optional[int] = None
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=400,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                # Defer the sleep until after the semaphore releases so
                # a burst of 429/5xx/timeouts doesn't tie up concurrency
                # slots with sleeping coroutines.
                retry_delay = 2 ** attempt
        if retry_delay is not None:
            await asyncio.sleep(retry_delay)
            continue
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        try:
            verdict = parse_verdict_json(text)
            if verdict.get("verdict") not in {"aware", "silent"}:
                raise ValueError(f"invalid verdict payload: {verdict!r}")
            reason = verdict.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError(f"invalid reason type: {type(reason).__name__}")
            evidence = verdict.get("kb_evidence_ids", []) or []
            if not isinstance(evidence, list) or not all(
                isinstance(x, int) for x in evidence
            ):
                raise ValueError(f"invalid kb_evidence_ids: {evidence!r}")
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            await asyncio.sleep(1)
            continue
        usage = response.usage
        return {
            "instance_id": row["instance_id"],
            "selected_database": row["selected_database"],
            "sql_index": row["sql_index"],
            "operator": row["operator"],
            "bucket": row["bucket"],
            "expression_hash": _hash_expression(row["expression"]),
            "verdict": verdict.get("verdict", "ERROR"),
            "reason": verdict.get("reason", ""),
            "kb_evidence_ids": verdict.get("kb_evidence_ids", []) or [],
            "deleted_kb_ids": row["deleted_kb_ids"],
            "is_hard8": row["is_hard8"],
            "_usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    usage, "cache_read_input_tokens", 0
                ),
            },
        }
    raise RuntimeError(
        f"giving up on {_key(row)} after {max_attempts} attempts: {last_exc}"
    )


def estimate_cost(usage_totals: dict) -> float:
    return (
        usage_totals.get("input_tokens", 0) / 1_000_000 * COST_INPUT_PER_MTOK
        + usage_totals.get("cache_creation_input_tokens", 0) / 1_000_000
        * COST_INPUT_CACHE_WRITE_PER_MTOK
        + usage_totals.get("cache_read_input_tokens", 0) / 1_000_000
        * COST_INPUT_CACHE_READ_PER_MTOK
        + usage_totals.get("output_tokens", 0) / 1_000_000 * COST_OUTPUT_PER_MTOK
    )


async def run_async(
    *,
    output_dir: Path,
    mini_interact_root: Path,
    concurrency: int,
) -> int:
    tuples_path = output_dir / "operator_tuples.jsonl"
    if not tuples_path.exists():
        print(f"ERROR: missing {tuples_path}; run extract step first.", file=sys.stderr)
        return 2
    judgments_path = output_dir / "judgments.jsonl"
    cache = JudgmentCache(judgments_path)

    rows: list[dict] = []
    for line in tuples_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))

    pending = [r for r in rows if not cache.already_done(r)]
    print(
        f"Tuples total={len(rows)}  cached={len(rows) - len(pending)}  "
        f"pending={len(pending)}"
    )
    if not pending:
        print("Nothing to do; all judgments already cached.")
        return 0

    # Group pending by (db, frozenset(deleted_kb_ids)) so within a group
    # we can reuse the cached system prompt (5-min TTL) and avoid
    # rebuilding KB text per call.
    groups: dict[tuple[str, tuple[int, ...]], list[dict]] = defaultdict(list)
    for r in pending:
        groups[(r["selected_database"], tuple(sorted(r["deleted_kb_ids"])))].append(r)
    print(f"Cache groups: {len(groups)}  (mean tuples/group={len(pending)/len(groups):.1f})")

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    usage_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    n_done = 0
    n_aware = 0
    n_silent = 0
    n_err = 0
    t0 = time.time()

    # Process group by group so within-group cache hits dominate.
    for (db, del_ids), group_rows in sorted(groups.items()):
        deleted_set = set(del_ids)
        system_prompt = build_system_prompt(
            db=db,
            deleted_ids=deleted_set,
            mini_interact_root=mini_interact_root,
        )

        # Fan out within the group with the shared system prompt.
        tasks = [
            classify_one(
                client=client,
                system_prompt=system_prompt,
                row=r,
                semaphore=semaphore,
            )
            for r in group_rows
        ]
        for coro in asyncio.as_completed(tasks):
            try:
                judgment = await coro
            except Exception as exc:  # exhausted retries
                n_err += 1
                print(f"  ERROR: {exc}", file=sys.stderr)
                continue
            usage = judgment.pop("_usage")
            for k, v in usage.items():
                usage_totals[k] += v
            cache.append(judgment)
            n_done += 1
            if judgment["verdict"] == "aware":
                n_aware += 1
            elif judgment["verdict"] == "silent":
                n_silent += 1
            else:
                n_err += 1
            if n_done % 100 == 0:
                cost = estimate_cost(usage_totals)
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(
                    f"  done={n_done}/{len(pending)}  aware={n_aware}  "
                    f"silent={n_silent}  err={n_err}  "
                    f"~${cost:.2f}  ~{rate:.1f}/s"
                )

    cost = estimate_cost(usage_totals)
    elapsed = time.time() - t0
    print(
        f"\nClassifier done: total={n_done}  aware={n_aware}  silent={n_silent}  "
        f"err={n_err}  elapsed={elapsed:.0f}s  cost~${cost:.2f}"
    )
    print(f"Token totals: {usage_totals}")
    print(f"Judgments cache: {judgments_path}")
    return 0 if n_err == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--mini-interact-root",
        default=str(DEFAULT_MINI_INTERACT_ROOT),
        help=f"(default: {DEFAULT_MINI_INTERACT_ROOT})",
    )
    p.add_argument("--concurrency", type=int, default=10)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.concurrency < 1:
        print("ERROR: --concurrency must be >= 1", file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    return asyncio.run(
        run_async(
            output_dir=Path(args.output_dir).resolve(),
            mini_interact_root=Path(args.mini_interact_root).resolve(),
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
