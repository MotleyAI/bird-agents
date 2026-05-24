"""Side-by-side comparison of original / raw / slayer eval results.

Reads:
    <dir>/original/results.jsonl                (original harness)
    <dir>/original/token_usage_<model>.json     (analyze_tokens output)
    <dir>/raw/eval.json                         (our run.py output)
    <dir>/slayer/eval.json                      (our run.py output)

Writes <dir>/comparison.json and prints a summary table.

Each version's per-task record is normalised to:
    {instance_id, phase1_passed, phase2_passed, total_reward, error}

Token usage is aggregated per leg into:
    {prompt_tokens, completion_tokens, agent_cost_usd, user_sim_cost_usd,
     cost_usd, partial}
The original leg only records agent-side tokens (user-sim is not
captured by upstream `analyze_tokens`), so its `user_sim_cost_usd`
field is `None` and the "user-sim tok" cell renders as a dash.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from bird_interact_agents.usage import TokenUsage

VERSIONS = ("original", "raw", "slayer")


def _first_present(row: dict, *keys: str, default: Any = None) -> Any:
    """Return the first key whose value is not None.

    Unlike `a or b`, this preserves explicit `False` and `0` — important for
    `phase1_passed=False` / `total_reward=0` not getting flipped to the next
    fallback's value.
    """
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _norm_orig(row: dict) -> dict | None:
    """Upstream main.py serialises the SampleStatus dataclass directly:
    instance_id is nested under `original_data`, phase status is reported
    via `phase1_completed` / `task_finished` (the latter signals the
    follow-up phase finished), reward via `last_reward`. We keep the
    bird-interact-agents-shaped fallbacks first so this also accepts
    ad-hoc rows in our own format.

    Returns None when no `instance_id` can be recovered — caller drops the
    row rather than collapsing it under an empty key (which would let one
    malformed record overwrite another).
    """
    od = row.get("original_data") or {}
    iid = (
        row.get("instance_id")
        or od.get("instance_id")
        or row.get("task_id")
    )
    if not iid:
        return None
    return {
        "instance_id": iid,
        "phase1_passed": bool(
            _first_present(row, "phase1_passed", "phase1_completed", default=False)
        ),
        "phase2_passed": bool(
            _first_present(row, "phase2_passed", "task_finished", default=False)
        ),
        "total_reward": float(
            _first_present(row, "total_reward", "last_reward", default=0.0)
        ),
        "error": row.get("error"),
    }


def _norm_ours(row: dict) -> dict | None:
    """Like `_norm_orig` but for our own `eval.json` shape. Returns None on
    missing instance_id so the caller drops the row instead of merging
    multiple rows under the empty-string key."""
    iid = row.get("instance_id") or row.get("task_id")
    if not iid:
        return None
    return {
        "instance_id": iid,
        "phase1_passed": bool(row.get("phase1_passed")),
        "phase2_passed": bool(row.get("phase2_passed")),
        "total_reward": float(row.get("total_reward") or 0.0),
        "error": row.get("error"),
        "submitted_sql": row.get("submitted_sql"),
        "submitted_query": row.get("submitted_query"),
        "ground_truth_sql": row.get("ground_truth_sql"),
        "duration_s": float(row.get("duration_s") or 0.0),
    }


def _last_submit_sql(history: list) -> str | None:
    """Walk a SampleStatus.interaction_history for the most recent
    `submit_sql(...)` action. Returns None if the agent never submitted.
    """
    for entry in reversed(history):
        action = entry.get("action") or "" if isinstance(entry, dict) else ""
        if action.startswith("submit_sql("):
            # Crude extraction — between first '(' and last ')'.
            return action[len("submit_sql("):-1] if action.endswith(")") else None
    return None


def _latest_run_in_db(
    conn: sqlite3.Connection, query_mode: str
) -> tuple[str, str] | None:
    """Pick the (run_id, framework) for the most recent task in this leg's
    DB. `task_results` is keyed by (run_id, framework, mode, query_mode,
    instance_id), so reusing a leg dir across runs leaves stale rows
    behind — without scoping, `INSERT OR REPLACE` plus duplicate
    instance_ids would silently mix runs together. Returns `None` when
    no row matches the requested `query_mode`.
    """
    row = conn.execute(
        "SELECT run_id, framework FROM task_results "
        "WHERE query_mode = ? "
        "ORDER BY started_at DESC LIMIT 1",
        (query_mode,),
    ).fetchone()
    return (row[0], row[1]) if row else None


def _load_ours_from_db(db_path: Path, query_mode: str) -> list[dict] | None:
    """Read this leg's task rows from `results.db` if it exists.

    Returns `None` (not [] ) when the DB is missing so callers can fall
    back to the legacy `eval.json` reader without conflating "no rows"
    with "no DB". When the DB is present but holds rows from multiple
    runs in the same leg dir, only the newest `(run_id, framework)`
    slice is read so `summary` / `per_task` stay deterministic.
    """
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        latest = _latest_run_in_db(conn, query_mode)
        if latest is None:
            return []
        run_id, framework = latest
        rows = list(conn.execute(
            "SELECT instance_id, phase1_passed, phase2_passed, total_reward, "
            "error, submitted_sql, submitted_query, ground_truth_sql, "
            "duration_s FROM task_results "
            "WHERE query_mode = ? AND run_id = ? AND framework = ? "
            "ORDER BY instance_id",
            (query_mode, run_id, framework),
        ))
    finally:
        conn.close()
    out: list[dict] = []
    for r in rows:
        out.append({
            "instance_id": r[0],
            "phase1_passed": bool(r[1]),
            "phase2_passed": bool(r[2]),
            "total_reward": float(r[3] or 0.0),
            "error": r[4],
            "submitted_sql": r[5],
            "submitted_query": r[6],
            "ground_truth_sql": r[7],
            "duration_s": float(r[8] or 0.0),
        })
    return out


def _load_ours_aggregates_from_db(
    db_path: Path, query_mode: str
) -> dict | None:
    """Aggregate usage + timing for the latest run's slice in `db_path`.

    Returns a dict ``{"usage": ..., "timing": ...}`` shaped like
    `_load_ours_usage` / `_load_ours_timing`'s outputs, or `None` when
    no DB / no matching rows. Used as the partial-run fallback: when
    `eval.json` is missing, the per-task `usage_json` + `duration_s`
    columns still hold the cost/latency the run paid before it died,
    and zeroing them out (the old behaviour) under-reports exactly the
    case the per-task sink was added to preserve.
    """
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        latest = _latest_run_in_db(conn, query_mode)
        if latest is None:
            return None
        run_id, framework = latest
        rows = list(conn.execute(
            "SELECT usage_json, duration_s FROM task_results "
            "WHERE query_mode = ? AND run_id = ? AND framework = ?",
            (query_mode, run_id, framework),
        ))
    finally:
        conn.close()
    if not rows:
        return None

    total = TokenUsage()
    durations: list[float] = []
    for usage_json, duration_s in rows:
        try:
            blob = json.loads(usage_json or "{}")
        except json.JSONDecodeError:
            blob = {}
        if blob:
            total.merge(TokenUsage.model_validate(blob))
        durations.append(float(duration_s or 0.0))

    agent_in = sum(
        b.prompt_tokens for b in total.breakdown if b.scope == "agent"
    )
    agent_out = sum(
        b.completion_tokens for b in total.breakdown if b.scope == "agent"
    )
    user_in = sum(
        b.prompt_tokens for b in total.breakdown if b.scope == "user_sim"
    )
    user_out = sum(
        b.completion_tokens for b in total.breakdown if b.scope == "user_sim"
    )
    usage = {
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "reasoning_tokens": total.reasoning_tokens,
        "cache_read_tokens": total.cache_read_tokens,
        "n_calls": total.n_calls,
        "agent_prompt_tokens": agent_in,
        "agent_completion_tokens": agent_out,
        "user_sim_prompt_tokens": user_in,
        "user_sim_completion_tokens": user_out,
        "agent_cost_usd": total.agent_cost_usd,
        "user_sim_cost_usd": total.user_sim_cost_usd,
        "cost_usd": total.cost_usd,
        # Aggregating from per-task rows always reflects a partial picture
        # — eval.json is the only place a "run finished" signal lives.
        "partial": True,
    }

    durations_sorted = sorted(durations)
    n = len(durations_sorted)
    timing = {
        "total_duration_s": sum(durations_sorted),
        "avg_duration_s": sum(durations_sorted) / n if n else 0.0,
        "p50_duration_s": durations_sorted[n // 2] if n else 0.0,
        "max_duration_s": max(durations_sorted) if durations_sorted else 0.0,
    }
    return {"usage": usage, "timing": timing}


def _load_original(path: Path, *, allow_missing: bool) -> list[dict]:
    """Load the original leg's task rows.

    Prefer `<original_dir>/results.db` (populated by
    `bird_interact_agents.original_ingest` after the upstream run). Fall
    back to parsing `<original_dir>/results.jsonl` directly so output
    dirs produced before the ingest hook still work.

    `allow_missing=True` lets callers skip missing files; otherwise
    `FileNotFoundError` is raised so a typo in the output dir is loud.
    """
    orig_dir = path.parent
    db_path = orig_dir / "results.db"
    db_rows = _load_ours_from_db(db_path, "raw")
    if db_rows is not None and db_rows:
        return db_rows

    if not path.is_file():
        if allow_missing:
            return []
        raise FileNotFoundError(
            f"Expected results file not found: {path}. "
            "Pass --allow-missing to treat absent files as empty results."
        )
    rows: list[dict] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            normalised = _norm_orig(row)
            if normalised is None:
                skipped += 1
                continue
            # Pluck the submitted SQL: upstream stores it as
            # `successful_phase1_sql` once phase 1 passes; otherwise we
            # walk `interaction_history` for the last submit_sql action.
            normalised["submitted_sql"] = (
                row.get("successful_phase1_sql")
                or _last_submit_sql(row.get("interaction_history") or [])
            )
            od = row.get("original_data") or {}
            sol = od.get("sol_sql")
            if isinstance(sol, list) and sol:
                normalised["ground_truth_sql"] = sol[0]
            elif isinstance(sol, str):
                normalised["ground_truth_sql"] = sol
            rows.append(normalised)
    if skipped:
        print(f"[compare_results] {path}: skipped {skipped} row(s) with missing instance_id")
    return rows


def _load_ours(path: Path, *, allow_missing: bool) -> list[dict]:
    """Load this leg's task rows.

    Prefer `<leg_dir>/results.db` (the per-task SQLite sink written by
    `run.py` after each task completes). Fall back to `eval.json` when
    no DB exists, e.g. on output dirs produced before the DB sink
    landed.
    """
    leg_dir = path.parent
    # Infer query_mode from the leg directory name (raw / slayer).
    query_mode = leg_dir.name
    db_path = leg_dir / "results.db"
    db_rows = _load_ours_from_db(db_path, query_mode)
    if db_rows is not None:
        return db_rows
    if not path.is_file():
        if allow_missing:
            return []
        raise FileNotFoundError(
            f"Expected results file not found: {path}. "
            "Pass --allow-missing to treat absent files as empty results."
        )
    blob = json.loads(path.read_text())
    rows: list[dict] = []
    skipped = 0
    for r in blob.get("results", []):
        norm = _norm_ours(r)
        if norm is None:
            skipped += 1
            continue
        rows.append(norm)
    if skipped:
        print(f"[compare_results] {path}: skipped {skipped} row(s) with missing instance_id")
    return rows


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "phase1_rate": sum(r["phase1_passed"] for r in rows) / n,
        "phase2_rate": sum(r["phase2_passed"] for r in rows) / n,
        "avg_reward": sum(r["total_reward"] for r in rows) / n,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def _empty_usage() -> dict:
    return {
        "prompt_tokens": 0, "completion_tokens": 0,
        "reasoning_tokens": 0, "cache_read_tokens": 0,
        "n_calls": 0, "cost_usd": 0.0,
        "agent_cost_usd": 0.0, "user_sim_cost_usd": 0.0,
        "partial": False,
    }


def _load_ours_usage(eval_path: Path) -> dict:
    if not eval_path.is_file():
        # Run died before writing eval.json — pull aggregate from
        # the per-task DB sink so partial cost is still reported.
        leg_dir = eval_path.parent
        agg = _load_ours_aggregates_from_db(
            leg_dir / "results.db", leg_dir.name
        )
        if agg is not None:
            return agg["usage"]
        return _empty_usage()
    blob = json.loads(eval_path.read_text())
    u = blob.get("total_usage")
    if not u:
        return _empty_usage()
    # Split prompt/completion totals by scope from breakdown[] so the
    # comparison table can show agent vs user-sim columns separately.
    agent_in = sum(
        r["prompt_tokens"] for r in u.get("breakdown", []) if r["scope"] == "agent"
    )
    agent_out = sum(
        r["completion_tokens"] for r in u.get("breakdown", []) if r["scope"] == "agent"
    )
    user_in = sum(
        r["prompt_tokens"] for r in u.get("breakdown", []) if r["scope"] == "user_sim"
    )
    user_out = sum(
        r["completion_tokens"] for r in u.get("breakdown", []) if r["scope"] == "user_sim"
    )
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "reasoning_tokens": u.get("reasoning_tokens", 0),
        "cache_read_tokens": u.get("cache_read_tokens", 0),
        "n_calls": u.get("n_calls", 0),
        "agent_prompt_tokens": agent_in,
        "agent_completion_tokens": agent_out,
        "user_sim_prompt_tokens": user_in,
        "user_sim_completion_tokens": user_out,
        "agent_cost_usd": u.get("agent_cost_usd", 0.0),
        "user_sim_cost_usd": u.get("user_sim_cost_usd", 0.0),
        "cost_usd": u.get("cost_usd", 0.0),
        "partial": u.get("partial", False),
    }


def _load_orig_usage(orig_dir: Path) -> dict:
    """Sum per-turn aggregates produced by upstream `analyze_tokens`.

    Upstream only records agent-side tokens — user-sim is not captured —
    so we leave `user_sim_cost_usd=None` and reuse `litellm.cost_per_token`
    to compute the agent cost so original / raw / slayer are priced
    consistently.
    """
    empty = {
        **_empty_usage(),
        "agent_prompt_tokens": 0, "agent_completion_tokens": 0,
        "user_sim_prompt_tokens": 0, "user_sim_completion_tokens": 0,
        "user_sim_cost_usd": None,
        "partial": True,
    }
    if not orig_dir.is_dir():
        return empty
    # When several `token_usage_*.json` files are present (re-runs with
    # different agent models leave one per model), pick the freshest by
    # mtime — `glob()` order is filesystem-dependent and would otherwise
    # make totals non-reproducible.
    matches = sorted(
        orig_dir.glob("token_usage_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        # Fall back to summing per-turn `token_usage` blobs directly.
        out = _load_orig_usage_from_turns(orig_dir)
        if out["prompt_tokens"] == 0:
            return empty
        # We don't know which model produced these tokens here — skip
        # cost calc; caller can add `--price-overrides` or rerun
        # `analyze_tokens` separately to fill it in.
        return out

    blob = json.loads(matches[0].read_text())
    prompt = 0
    completion = 0
    reasoning = 0
    n_calls = 0
    model = ""
    for stats in blob.values():
        n = int(stats.get("num_samples") or 0)
        prompt += int(stats.get("avg_prompt_tokens") or 0) * n
        completion += int(stats.get("avg_completion_tokens") or 0) * n
        reasoning += int(stats.get("avg_reasoning_tokens") or 0) * n
        n_calls += n
        model = stats.get("model", model)

    try:
        import litellm
        p_cost, c_cost = litellm.cost_per_token(
            model=model, prompt_tokens=prompt, completion_tokens=completion,
        )
        agent_cost = p_cost + c_cost
    except Exception:
        agent_cost = 0.0

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cache_read_tokens": 0,
        "n_calls": n_calls,
        "agent_prompt_tokens": prompt,
        "agent_completion_tokens": completion,
        "user_sim_prompt_tokens": 0,
        "user_sim_completion_tokens": 0,
        "agent_cost_usd": agent_cost,
        "user_sim_cost_usd": None,
        "cost_usd": agent_cost,
        "partial": True,  # user-sim missing
    }


def _fmt_tok(n: int) -> str:
    return f"{n / 1000:.1f}K" if n else "—"


def _fmt_usd(v) -> str:
    return f"${v:.2f}" if isinstance(v, (int, float)) else "—"


def _any_sql_captured(by_id: dict[str, dict[str, dict]]) -> bool:
    """True if any leg actually persisted a submitted/ground-truth SQL.

    Legacy `eval.json`-only output dirs don't carry these fields, so
    we suppress `per_task_sql.md` rather than emit a useless file with
    every cell saying "n/a".
    """
    for legs in by_id.values():
        for row in legs.values():
            if any(
                row.get(k) for k in (
                    "submitted_sql", "submitted_query", "ground_truth_sql",
                )
            ):
                return True
    return False


def _render_sql_block(label: str, sql: str | None) -> str:
    if not sql:
        return f"**{label}**: _(none submitted)_\n"
    return f"**{label}**:\n```sql\n{sql.strip()}\n```\n"


def _outcome_summary(legs: dict[str, dict]) -> str:
    """One-line summary of which legs passed phase 1, for the section header."""
    bits = []
    for v in VERSIONS:
        row = legs.get(v) or {}
        if row.get("error"):
            bits.append(f"{v}: ERROR")
        elif row.get("phase1_passed"):
            bits.append(f"{v}: ✓")
        elif v in legs:
            bits.append(f"{v}: ✗")
    return " · ".join(bits) if bits else ""


def _render_per_task_sql_md(by_id: dict[str, dict[str, dict]]) -> str:
    """Side-by-side per-instance dump of ground-truth, original, raw,
    and slayer SQL (slayer also includes the JSON DSL the agent
    submitted)."""
    parts = ["# Per-task SQL submissions\n"]
    for iid in sorted(by_id):
        legs = by_id[iid]
        parts.append(f"\n## {iid}\n")
        summary = _outcome_summary(legs)
        if summary:
            parts.append(f"_{summary}_\n")

        # Pick whichever leg recorded ground truth (they should agree;
        # we read it from task_data.sol_sql in run.py).
        gt = next(
            (
                row.get("ground_truth_sql")
                for row in legs.values()
                if row.get("ground_truth_sql")
            ),
            None,
        )
        parts.append(_render_sql_block("Ground truth", gt))

        for v in VERSIONS:
            row = legs.get(v) or {}
            if v == "slayer":
                # Slayer carries both the JSON DSL submitted by the agent
                # and the SQL the SLayer engine rendered from it.
                if row.get("submitted_query"):
                    parts.append(
                        f"**slayer (JSON query)**:\n```json\n"
                        f"{row['submitted_query'].strip()}\n```\n"
                    )
                parts.append(_render_sql_block("slayer (rendered SQL)", row.get("submitted_sql")))
            else:
                parts.append(_render_sql_block(v, row.get("submitted_sql")))
    return "".join(parts)


def _fmt_dur(v) -> str:
    if not isinstance(v, (int, float)) or v <= 0:
        return "—"
    if v < 60:
        return f"{v:.1f}s"
    return f"{v / 60:.1f}m"


def _empty_timing() -> dict:
    return {
        "total_duration_s": 0.0,
        "avg_duration_s": 0.0,
        "p50_duration_s": 0.0,
        "max_duration_s": 0.0,
    }


def _load_ours_timing(eval_path: Path) -> dict:
    if not eval_path.is_file():
        # Same partial-run rationale as `_load_ours_usage`: prefer the
        # per-row `duration_s` over zeros when eval.json is absent.
        leg_dir = eval_path.parent
        agg = _load_ours_aggregates_from_db(
            leg_dir / "results.db", leg_dir.name
        )
        if agg is not None:
            return agg["timing"]
        return _empty_timing()
    blob = json.loads(eval_path.read_text())
    return {
        "total_duration_s": float(blob.get("total_duration_s") or 0.0),
        "avg_duration_s": float(blob.get("avg_duration_s") or 0.0),
        "p50_duration_s": float(blob.get("p50_duration_s") or 0.0),
        "max_duration_s": float(blob.get("max_duration_s") or 0.0),
    }


def _load_orig_timing(orig_dir: Path) -> dict:
    """Read leg-total wall-clock from `original/duration.json`. Per-task
    timing is not captured for the original leg, so `avg/p50/max_duration_s`
    only reflect the leg-wide average if `n_tasks` was recorded."""
    path = orig_dir / "duration.json"
    if not path.is_file():
        return _empty_timing()
    blob = json.loads(path.read_text())
    total = float(blob.get("total_duration_s") or 0.0)
    avg = float(blob.get("avg_duration_s") or 0.0)
    return {
        "total_duration_s": total,
        "avg_duration_s": avg,
        # We don't have per-task durations to compute p50/max; report avg.
        "p50_duration_s": avg,
        "max_duration_s": avg,
    }


def _load_orig_usage_from_turns(orig_dir: Path) -> dict:
    """Sum `token_usage` from each `results.jsonl.agent_raw_turn_*.jsonl`
    when `analyze_tokens` couldn't run (e.g. because its hard-coded
    pricing whitelist rejects the model). Each per-turn file contains
    one line per task processed in that turn.

    If `duration.json` records `agent_model`, price the totals via
    `litellm.cost_per_token` so the original leg's `$ agent` column
    populates without re-running upstream's analyze_tokens.
    """
    prompt = 0
    completion = 0
    reasoning = 0
    n_calls = 0
    for f in sorted(orig_dir.glob("results.jsonl.agent_raw_turn_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                u = row.get("token_usage") or {}
                if not u:
                    continue
                prompt += int(u.get("prompt_tokens") or 0)
                completion += int(u.get("completion_tokens") or 0)
                reasoning += int(u.get("reasoning_tokens") or 0)
                n_calls += 1

    agent_cost = 0.0
    duration_path = orig_dir / "duration.json"
    if duration_path.is_file():
        try:
            model = json.loads(duration_path.read_text()).get("agent_model")
        except Exception:
            model = None
        if model and prompt:
            try:
                import litellm
                p_cost, c_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                )
                agent_cost = p_cost + c_cost
            except Exception:
                agent_cost = 0.0

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cache_read_tokens": 0,
        "n_calls": n_calls,
        "agent_prompt_tokens": prompt,
        "agent_completion_tokens": completion,
        "user_sim_prompt_tokens": 0,
        "user_sim_completion_tokens": 0,
        "agent_cost_usd": agent_cost,
        "user_sim_cost_usd": None,
        "cost_usd": agent_cost,
        "partial": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", help="Directory containing original/, raw/, slayer/")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Treat absent results files as empty datasets instead of failing. "
        "Off by default so a misrouted/failed run doesn't silently turn into "
        "a misleading 0%/0-reward report.",
    )
    args = parser.parse_args()

    base = Path(args.dir)
    rows_by_version = {
        "original": _load_original(
            base / "original" / "results.jsonl", allow_missing=args.allow_missing
        ),
        "raw": _load_ours(
            base / "raw" / "eval.json", allow_missing=args.allow_missing
        ),
        "slayer": _load_ours(
            base / "slayer" / "eval.json", allow_missing=args.allow_missing
        ),
    }
    by_id: dict[str, dict[str, dict]] = {}
    for v, rows in rows_by_version.items():
        for r in rows:
            by_id.setdefault(r["instance_id"], {})[v] = r

    summary = {v: _aggregate(rows) for v, rows in rows_by_version.items()}
    usage = {
        "original": _load_orig_usage(base / "original"),
        "raw": _load_ours_usage(base / "raw" / "eval.json"),
        "slayer": _load_ours_usage(base / "slayer" / "eval.json"),
    }
    timing = {
        "original": _load_orig_timing(base / "original"),
        "raw": _load_ours_timing(base / "raw" / "eval.json"),
        "slayer": _load_ours_timing(base / "slayer" / "eval.json"),
    }
    out = {
        "summary": summary, "usage": usage, "timing": timing,
        "per_task": by_id,
    }
    (base / "comparison.json").write_text(json.dumps(out, indent=2))

    # Per-task SQL dump — only emit when at least one leg actually
    # captured a `submitted_sql`/`submitted_query`/`ground_truth_sql`
    # field (DB-backed runs do; legacy eval.json-only runs don't).
    if _any_sql_captured(by_id):
        (base / "per_task_sql.md").write_text(_render_per_task_sql_md(by_id))

    # ── Markdown table to stdout ────────────────────────────────────────
    print("\n## Aggregate\n")
    print(
        "| version | n | P1 rate | P2 rate | avg reward | errors | "
        "agent tok in/out | user-sim tok in/out | $ agent | $ user-sim | $ total | "
        "avg dur | total dur |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in VERSIONS:
        s = summary[v]
        u = usage[v]
        t = timing[v]
        agent_in = u.get("agent_prompt_tokens", 0)
        agent_out = u.get("agent_completion_tokens", 0)
        user_in_n = u.get("user_sim_prompt_tokens", 0)
        user_out_n = u.get("user_sim_completion_tokens", 0)
        user_cell = (
            "— / —"
            if u["user_sim_cost_usd"] is None
            else f"{_fmt_tok(user_in_n)}/{_fmt_tok(user_out_n)}"
        )
        # Original leg only has leg-total wall-clock; show the avg derived
        # from total / n_tasks but not p50/max.
        avg_dur_cell = (
            "—" if v == "original" and s["n"] == 0 else _fmt_dur(t["avg_duration_s"])
        )
        print(
            f"| {v} | {s['n']} | {s['phase1_rate']:.2%} | "
            f"{s['phase2_rate']:.2%} | {s['avg_reward']:.3f} | {s['errors']} | "
            f"{_fmt_tok(agent_in)}/{_fmt_tok(agent_out)} | "
            f"{user_cell} | "
            f"{_fmt_usd(u['agent_cost_usd'])} | "
            f"{_fmt_usd(u['user_sim_cost_usd'])} | "
            f"{_fmt_usd(u['cost_usd'])} | "
            f"{avg_dur_cell} | {_fmt_dur(t['total_duration_s'])} |"
        )
    if any(usage[v]["partial"] for v in VERSIONS):
        print(
            "\n> Note: rows marked partial omit user-sim tokens "
            "(upstream `analyze_tokens` records agent-side only) or "
            "agent-side tokens (mcp_agent SDK does not expose them)."
        )

    print("\n## Per-task P1\n")
    print("| instance_id | original | raw | slayer |")
    print("|---|---|---|---|")
    for iid in sorted(by_id):
        cells = [
            "✓" if by_id[iid].get(v, {}).get("phase1_passed") else
            ("✗" if v in by_id[iid] else "—")
            for v in VERSIONS
        ]
        print(f"| {iid} | {cells[0]} | {cells[1]} | {cells[2]} |")

    print(f"\nWrote {base / 'comparison.json'}")


if __name__ == "__main__":
    main()
