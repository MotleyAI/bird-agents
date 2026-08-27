"""First-submission success rate from stored session logs.

For a ``(benchmark, agent_model, mode)`` combination this walks
``runs/<benchmark>/<db>/<iid>/`` exactly like ``scripts/cascade_for_combo.py``
(latest non-``eval_failed`` submission annotation per ``(db, iid, mode)``, by
``annotated_at``), then opens each selected run's **session log** and asks a
different question than the cascade: did the agent's **very first** ``submit``
already return the correct answer, before any retry within the run's budget?

The signal comes straight from the harness. Each run submits via a tool whose
result reports the match:

* raw mode → ``submit_sql`` → tool result ``"... Result match: True/False"``
* slayer mode → ``submit_query`` → generates+runs SQL, result is either
  ``"... Result match: True/False"`` or ``"Result: SQL execution error"`` (a
  failed first submission — the generated SQL didn't run).

We anchor on the FIRST submit call's result and classify it:

* ``correct`` — first submit reported ``Result match: True``
* ``wrong``   — first submit reported ``Result match: False``
* ``error``   — first submit produced no match line (dry-run DB error, SQL
  execution error, …) → it did not give the correct answer
* ``nosubmit``— the log has no submit call at all

Session logs live at ``results/<cloud_run_id>/<trajectory_path>`` (the run
record's ``submission.trajectory_path``, e.g. ``rows/<iid>/attempt-1.json``).
A selected run whose log was never written / not recovered is counted under
``no_log`` and excluded from the first-submission denominator.

Two trajectory encodings exist in the wild and both are handled: the older
``repr``-string form (``AssistantMessage(content=[ToolUseBlock(...)])``) and
the newer structured-dict form (``{"content": [{"id":..,"name":..}]}``).

Usage
-----
    uv run python scripts/scan_first_submission.py \\
        --benchmark livesqlbench-large --agent-model opus --paired

    uv run python scripts/scan_first_submission.py \\
        --benchmark livesqlbench-large --mode slayer --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from bird_interact_agents import paths


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable — no filesystem / subprocess side effects)
# --------------------------------------------------------------------------- #
_RESULT_MATCH_RE = re.compile(r"Result match:\s*(True|False)")
_REPR_TOOLUSE_RE = re.compile(r"ToolUseBlock\(id='([^']+)', name='([^']+)'")
_REPR_TOOLRESULT_RE = re.compile(r"ToolResultBlock\(tool_use_id='([^']+)'")


def extract_submissions(
    trajectory: list,
) -> "tuple[list[str], dict[str, str]]":
    """Return ``(ordered submit tool_use_ids, {tool_use_id: result_text})``.

    Handles both trajectory encodings. Only tool-uses whose name contains
    ``"submit"`` are collected (``submit_sql`` for raw, ``submit_query`` for
    slayer); result text is accumulated per ``tool_use_id`` so the caller can
    look up the outcome of any specific submission.
    """
    submit_ids: list[str] = []
    result_text: dict[str, str] = {}
    for event in trajectory or []:
        data = event.get("data") if isinstance(event, dict) else None
        if isinstance(data, str):
            # repr-string encoding
            for tid, name in _REPR_TOOLUSE_RE.findall(data):
                if "submit" in name.lower():
                    submit_ids.append(tid)
            for tid in _REPR_TOOLRESULT_RE.findall(data):
                # The whole UserMessage repr (which embeds the result text)
                # is attributed to the result's tool_use_id; the Result-match
                # regex then reads it back.
                result_text[tid] = result_text.get(tid, "") + data
        elif isinstance(data, dict):
            # structured-dict encoding
            content = data.get("content")
            if isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("id") and blk.get("name") and (
                        "submit" in str(blk["name"]).lower()
                    ):
                        submit_ids.append(blk["id"])
                    if blk.get("tool_use_id") is not None and "content" in blk:
                        c = blk["content"]
                        txt = c if isinstance(c, str) else json.dumps(c)
                        tid = blk["tool_use_id"]
                        result_text[tid] = result_text.get(tid, "") + txt
    return submit_ids, result_text


def first_submission_outcome(trajectory: list) -> str:
    """``correct`` | ``wrong`` | ``error`` | ``nosubmit`` for the FIRST submit."""
    submit_ids, result_text = extract_submissions(trajectory)
    if not submit_ids:
        return "nosubmit"
    m = _RESULT_MATCH_RE.search(result_text.get(submit_ids[0], ""))
    if m:
        return "correct" if m.group(1) == "True" else "wrong"
    return "error"


def is_opus_match(agent_model: str, needle: str) -> bool:
    """Case-insensitive substring match (``opus`` ⊂ ``anthropic/claude-opus-4-8``)."""
    return needle.lower() in (agent_model or "").lower()


# --------------------------------------------------------------------------- #
# Record loading + latest-per-task selection (mirrors cascade_for_combo)
# --------------------------------------------------------------------------- #
def iter_run_records(runs_root: Path, benchmark: str):
    """Yield ``(path, record_dict)`` for every stored submission annotation."""
    bdir = runs_root / benchmark
    if not bdir.is_dir():
        return
    for rec in bdir.glob("*/*/*.json"):
        if rec.name.endswith(".trajectory.json"):
            continue
        try:
            d = json.loads(rec.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("kind") != "submission_annotation":
            continue
        yield rec, d


def select_latest(
    runs_root: Path, benchmark: str, agent_model: str,
) -> "dict[tuple[str, str, str], dict]":
    """``(db, iid, mode) -> record`` for the latest non-eval_failed opus run."""
    best: dict[tuple[str, str, str], tuple[str, dict]] = {}
    for _, d in iter_run_records(runs_root, benchmark):
        sub = d.get("submission") or {}
        cfg = sub.get("config") or {}
        am = cfg.get("agent_model") or d.get("agent_model") or ""
        if not is_opus_match(am, agent_model):
            continue
        mode = cfg.get("query_mode")
        if mode not in ("raw", "slayer"):
            continue
        db, iid = d.get("selected_database"), d.get("instance_id")
        if not (db and iid):
            continue
        if (d.get("evaluation") or {}).get("verdict") == "eval_failed":
            continue
        at = d.get("annotated_at") or ""
        key = (db, iid, mode)
        if key not in best or at > best[key][0]:
            best[key] = (at, d)
    return {k: v[1] for k, v in best.items()}


def log_path_for(results_root: Path, record: dict) -> Optional[Path]:
    sub = record.get("submission") or {}
    rid, tp = sub.get("cloud_run_id"), sub.get("trajectory_path")
    if not (rid and tp):
        return None
    return results_root / rid / tp


def final_pass(record: dict) -> bool:
    return (record.get("evaluation") or {}).get(
        "phase1_against_original_gold"
    ) == "pass"


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def scan(
    runs_root: Path, results_root: Path, benchmark: str, agent_model: str,
    modes: list[str], paired: bool,
) -> dict:
    latest = select_latest(runs_root, benchmark, agent_model)
    tasks_by_mode = {
        m: {(db, iid) for (db, iid, mm) in latest if mm == m} for m in modes
    }
    if paired and len(modes) == 2:
        common = tasks_by_mode[modes[0]] & tasks_by_mode[modes[1]]
        scope = {m: common for m in modes}
    else:
        scope = tasks_by_mode

    out: dict = {
        "benchmark": benchmark, "agent_model": agent_model,
        "paired": paired, "modes": {},
    }
    logged: dict[str, set] = {}
    for m in modes:
        buckets = {
            "tasks": 0, "has_log": 0, "no_log": 0,
            "correct": 0, "wrong": 0, "error": 0, "nosubmit": 0,
            "final_pass": 0,
        }
        logged[m] = set()
        for (db, iid) in sorted(scope[m]):
            rec = latest[(db, iid, m)]
            buckets["tasks"] += 1
            if final_pass(rec):
                buckets["final_pass"] += 1
            p = log_path_for(results_root, rec)
            if p is None or not p.exists():
                buckets["no_log"] += 1
                continue
            try:
                traj = json.loads(p.read_text()).get("trajectory") or []
            except (OSError, json.JSONDecodeError):
                buckets["no_log"] += 1
                continue
            buckets["has_log"] += 1
            logged[m].add((db, iid))
            buckets[first_submission_outcome(traj)] += 1
        out["modes"][m] = buckets

    if paired and len(modes) == 2:
        both = logged[modes[0]] & logged[modes[1]]
        out["both_logged"] = sorted(f"{db}/{iid}" for (db, iid) in both)
        out["both_logged_count"] = len(both)
        for m in modes:
            correct = sum(
                1 for t in both
                if first_submission_outcome(
                    json.loads(log_path_for(results_root, latest[(t[0], t[1], m)])
                               .read_text()).get("trajectory") or []
                ) == "correct"
            )
            out["modes"][m]["first_correct_on_both_logged"] = correct
    return out


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "n/a"


def print_report(out: dict) -> None:
    print(f"benchmark={out['benchmark']}  agent_model~={out['agent_model']}  "
          f"paired={out['paired']}")
    for m, b in out["modes"].items():
        print(f"\n===== {m.upper()} =====")
        print(f"  tasks selected        : {b['tasks']}")
        print(f"  with session log      : {b['has_log']}  "
              f"(no_log={b['no_log']})")
        print(f"  first submit CORRECT  : {b['correct']} / {b['has_log']} "
              f"= {_pct(b['correct'], b['has_log'])}")
        print(f"    first submit wrong  : {b['wrong']}")
        print(f"    first submit error  : {b['error']}")
        print(f"    no submit in log    : {b['nosubmit']}")
        print(f"  final pass (N1, after retries): {b['final_pass']} / "
              f"{b['tasks']} = {_pct(b['final_pass'], b['tasks'])}")
        if "first_correct_on_both_logged" in b:
            fc = b["first_correct_on_both_logged"]
            n = out["both_logged_count"]
            print(f"  first CORRECT on both-logged subset: {fc} / {n} "
                  f"= {_pct(fc, n)}")
    if "both_logged_count" in out:
        print(f"\nboth-logged (apples-to-apples) task count: "
              f"{out['both_logged_count']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", default="livesqlbench-large")
    ap.add_argument("--agent-model", default="opus",
                    help="case-insensitive substring of the stamped agent_model")
    ap.add_argument("--mode", choices=("raw", "slayer", "both"), default="both")
    ap.add_argument("--paired", action="store_true",
                    help="restrict to tasks that ran in BOTH raw and slayer")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    modes = ["raw", "slayer"] if args.mode == "both" else [args.mode]
    out = scan(
        paths.runs_root(), paths.results_root(),
        args.benchmark, args.agent_model, modes, args.paired,
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print_report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
