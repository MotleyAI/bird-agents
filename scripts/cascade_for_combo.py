"""Cascade-stat summary for one (benchmark, agent_model, mode) combination.

For each ``(db, instance_id)`` under ``runs/<benchmark>/``, picks the latest
submission annotation (by ``annotated_at``) produced by a cloud run whose
manifest matches the requested ``mode`` (raw|slayer) AND ``agent-model``
substring (case-insensitive — ``opus`` matches
``anthropic/claude-opus-4-7``), SKIPPING runs whose
``evaluation.verdict == "eval_failed"``. Those are stale grader-infrastructure
failures that would otherwise wrongly override the genuine earlier verdict.

Emits BOTH views, since both are useful:

* Cumulative N1..N9 — each ``n_k`` is "task passes AT OR BELOW cascade tier
  k" (monotone, ``n1 ⊆ n2 ⊆ ... ⊆ n9``). ``n1`` is the headline original-gold
  pass rate. ``Δ vs prev`` shows the incremental tier contribution.
* Mutually-exclusive partition L1..L11 — each task → exactly one tier. This
  is the "where did pass/fail land" view.

Manifest lookup: reads ``results/<benchmark>/cloud/<run_id>/manifest.json``
when present; otherwise downloads from GCS (cached under that same path).
``--no-gcs`` disables the GCS fallback (run files whose manifest isn't local
get skipped, with a warning).

Usage
-----
    uv run python scripts/cascade_for_combo.py \\
        --benchmark mini-interact --mode slayer --agent-model opus
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from bird_interact_agents import paths
from bird_interact_agents.cloud import gcs
from bird_interact_agents.eval.annotation_io import read_submission_annotation
from bird_interact_agents.eval.versioning import DEFAULT_VERSION
from bird_interact_agents.eval.cascading_report import (
    _annotation_cascade_bools,
    _assign_partition_tier,
    _PARTITION_IS_PASS,
    _PARTITION_LABELS,
)


logger = logging.getLogger("cascade_for_combo")


# The mode slot is delimited by ``-`` OR ``_`` and may sit mid-name (cloud
# ``<ts>-<framework>-<mode>-<hash>.json``) OR at the very end (local
# ``<ts>_<framework>_<mode>.json`` / ``local_<ts>_<framework>_<mode>.json``).
# Anchor on a leading ``-``/``_`` and a trailing ``-``/``_``/``.`` or end so the
# trailing-slug local form is matched too (the old ``-(raw|slayer)-`` missed it).
_FILE_MODE_RE = re.compile(r"[-_](raw|slayer|cube)(?=[-_.]|$)")


def mode_from_filename(name: str) -> Optional[str]:
    """Return ``"raw"`` / ``"slayer"`` / ``None`` from the mode slot token in a
    run filename — BOTH the cloud form (``<ts>-<framework>-<mode>-<hash>.json``)
    and the local form (``<ts>_<framework>_<mode>.json``, optionally
    ``local_``-prefixed)."""
    m = _FILE_MODE_RE.search(name)
    return m.group(1) if m else None


def load_manifest(
    benchmark: str, run_id: str, *, gcs_client=None, allow_gcs: bool = True,
    cache: bool = True,
) -> Optional[dict]:
    """Return the cloud manifest for ``run_id``, or ``None`` if unavailable.

    Reads from ``results/<benchmark>/cloud/<run_id>/manifest.json`` when
    present. Falls back to GCS (caching the result locally) unless
    ``allow_gcs`` is False. Pass ``cache=False`` to suppress the local
    cache write on a GCS fetch — used by callers (e.g. a ``--dry-run``) that
    must not mutate the results tree.
    """
    local = (
        paths.results_root() / benchmark / "cloud" / run_id / "manifest.json"
    )
    if local.exists():
        try:
            return json.loads(local.read_text())
        except json.JSONDecodeError:
            logger.warning("corrupt local manifest %s; re-fetching", local)
    if not allow_gcs:
        return None
    try:
        manifest = gcs.read_manifest(run_id, client=gcs_client)
    except Exception as exc:  # noqa: BLE001 — any GCS error is "skip this run"
        logger.warning("GCS manifest for %s unavailable: %s", run_id, exc)
        return None
    if cache:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def model_matches(manifest_model: Optional[str], requested: str) -> bool:
    """Case-insensitive substring match. ``opus`` matches
    ``anthropic/claude-opus-4-7``."""
    if not manifest_model:
        return False
    return requested.lower() in manifest_model.lower()


def _effective_verdict(data: dict) -> str:
    """Mirror ``SubmissionAnnotation._migrate_invalid_verdict`` for raw-JSON
    reads: legacy ``verdict="invalid"`` rows with
    ``failure_classification.primary="other"`` are infra failures and must
    be classified as ``"eval_failed"`` so they don't sneak past the
    "skip eval_failed" filter."""
    ev = data.get("evaluation") or {}
    fc = data.get("failure_classification") or {}
    raw = ev.get("verdict") or ""
    if raw == "invalid" and fc.get("primary") == "other":
        return "eval_failed"
    return raw


def collect_latest_per_task(
    *,
    benchmark: str,
    mode: str,
    agent_model: str,
    runs_root: Optional[Path] = None,
    allow_gcs: bool = True,
    gcs_client=None,
    version: Optional[str] = None,
) -> tuple[list[Path], dict[str, int]]:
    """Pick the latest non-eval_failed run per ``(db, instance_id)`` that
    matches ``(mode, agent_model)`` — and, when ``version`` is set, the
    requested code version. Returns the list of chosen file paths plus a
    small bookkeeping dict (files seen / skipped / overrides used).

    DEV-1591: the per-task RECORD is the authoritative source for
    ``agent_model`` and ``version`` (self-describing); the cloud manifest is
    consulted only as a fallback for legacy records that pre-date stamping.
    Mode comes from the run-id filename. A run is therefore NOT skipped just
    because its manifest is missing/stale, provided the record carries the
    fields — that was the bug that let manifest gaps drop stamped records.
    """
    runs_root = runs_root or (paths.runs_root() / benchmark)
    manifest_cache: dict[str, Optional[dict]] = {}
    candidates: dict[tuple[str, str], list[tuple[str, Path, str]]] = {}
    counters = {
        "files_scanned": 0,
        "matched_mode": 0,
        "matched_model": 0,
        "skipped_no_manifest": 0,
        "skipped_wrong_version": 0,
        "stale_eval_failed_overridden": 0,
        "skipped_eval_failed_only": 0,
    }
    if not runs_root.exists():
        return [], counters

    for path in runs_root.rglob("*.json"):
        if path.name.endswith(".trajectory.json"):
            continue
        counters["files_scanned"] += 1
        file_mode = mode_from_filename(path.name)
        if file_mode != mode:
            continue
        counters["matched_mode"] += 1
        try:
            rel = path.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        db, iid, run_file = parts
        run_id = run_file[:-5]
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("unreadable annotation %s: %s", path, exc)
            continue
        # Record is authoritative; the manifest is only a fallback for legacy
        # records that pre-date stamping. Defer the (potentially GCS-backed)
        # manifest load until the record itself can't name the model — a
        # stamped record never pays for a lookup it doesn't need.
        eff_model = data.get("agent_model")
        if not eff_model:
            if run_id not in manifest_cache:
                manifest_cache[run_id] = load_manifest(
                    benchmark, run_id,
                    gcs_client=gcs_client, allow_gcs=allow_gcs,
                )
            eff_model = (manifest_cache[run_id] or {}).get("agent_model")
        if not eff_model:
            # Unattributable: neither the record nor a manifest names a model.
            counters["skipped_no_manifest"] += 1
            continue
        if not model_matches(eff_model, agent_model):
            continue
        counters["matched_model"] += 1
        if version is not None:
            eff_version = data.get("version") or DEFAULT_VERSION
            if eff_version != version:
                counters["skipped_wrong_version"] += 1
                continue
        verdict = _effective_verdict(data)
        annotated_at = data.get("annotated_at") or ""
        candidates.setdefault((db, iid), []).append(
            (annotated_at, path, verdict),
        )

    chosen: list[Path] = []
    for recs in candidates.values():
        recs.sort()
        real = [r for r in recs if r[2] != "eval_failed"]
        if not real:
            counters["skipped_eval_failed_only"] += 1
            continue
        if recs[-1][2] == "eval_failed":
            counters["stale_eval_failed_overridden"] += 1
        chosen.append(real[-1][1])
    return chosen, counters


def aggregate(paths_in: list[Path]) -> dict:
    n_counts = {f"n{i}": 0 for i in range(1, 10)}
    p_counts = {k: 0 for k in _PARTITION_LABELS}
    n = 0
    for p in paths_in:
        ann = read_submission_annotation(p)
        n += 1
        for k, v in _annotation_cascade_bools(ann).items():
            if v:
                n_counts[k.split("_", 1)[0]] += 1
        p_counts[_assign_partition_tier(ann)] += 1
    return {"n": n, "n_counts": n_counts, "p_counts": p_counts}


_N_LABELS = {
    "n1": "original gold",
    "n2": "audited primary",
    "n3": "any audited variant",
    "n4": "correct up to tie order",
    "n5": "LLM judge",
    "n6": "numeric epsilon",
    "n7": "trailing whitespace",
    "n8": "column order",
    "n9": "case fold",
}


def render(
    *,
    benchmark: str,
    agent_model: str,
    mode: str,
    agg: dict,
    counters: dict,
) -> str:
    n = agg["n"]
    lines: list[str] = []
    bar = "=" * 92
    lines.append(bar)
    lines.append(
        f"Cascade summary  benchmark={benchmark}  "
        f"agent_model~={agent_model!r}  mode={mode}  "
        f"n={n}  (latest non-eval_failed run per task)"
    )
    lines.append(bar)
    if not n:
        lines.append("")
        lines.append(
            "No matching tasks. Counters: " + json.dumps(counters, indent=2)
        )
        return "\n".join(lines)

    lines.append("")
    lines.append(
        "Cumulative N-tier  (each n_k = tasks passing AT OR BELOW cascade tier k)"
    )
    lines.append("")
    lines.append(
        f"  {'tier':<6}{'description':<30}"
        f"{'cum_pass':>10}{'rate':>10}{'Δ vs prev':>12}"
    )
    prev = 0
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"):
        c = agg["n_counts"][k]
        delta = c - prev
        lines.append(
            f"  {k:<6}{_N_LABELS[k]:<30}"
            f"{c:>10}{c/n*100:>9.1f}%{delta:>+12d}"
        )
        prev = c

    lines.append("")
    lines.append(
        "Mutually-exclusive partition  (each task → exactly one tier)"
    )
    lines.append("")
    lines.append(
        f"  {'tier':<32}{'cnt':>5}{'rate':>9}{'cumsum':>9}{'cum_rate':>11}"
    )
    cum = 0
    cum_pass = 0
    for label in _PARTITION_LABELS:
        c = agg["p_counts"][label]
        cum += c
        is_pass = _PARTITION_IS_PASS[label]
        if is_pass:
            cum_pass += c
        marker = "+" if is_pass else "-"
        lines.append(
            f"  {marker} {label:<30}{c:>5}{c/n*100:>8.1f}%"
            f"{cum:>9}{cum/n*100:>10.1f}%"
        )
    lines.append("")
    lines.append(
        f"Cascade PASS = {cum_pass}/{n} ({cum_pass/n*100:.1f}%)   "
        f"L2 diagnostic = {agg['p_counts']['l2_wrong_original']}   "
        f"L11 hard fail = {agg['p_counts']['l11_fail']}"
    )

    lines.append("")
    lines.append(
        "Counters: "
        f"files_scanned={counters['files_scanned']}  "
        f"matched_mode={counters['matched_mode']}  "
        f"matched_model={counters['matched_model']}  "
        f"skipped_no_manifest={counters['skipped_no_manifest']}  "
        f"skipped_wrong_version={counters['skipped_wrong_version']}  "
        f"stale_eval_failed_overridden={counters['stale_eval_failed_overridden']}  "
        f"skipped_eval_failed_only={counters['skipped_eval_failed_only']}"
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark", default="mini-interact",
        help="Benchmark canonical name (default: mini-interact).",
    )
    parser.add_argument(
        "--mode", required=True, choices=("raw", "slayer", "cube"),
        help="Query mode of the runs to include.",
    )
    parser.add_argument(
        "--agent-model", required=True,
        help=(
            "Case-insensitive substring match against the cloud manifest's "
            "agent_model field (e.g. 'opus' matches "
            "'anthropic/claude-opus-4-7')."
        ),
    )
    parser.add_argument(
        "--version", default=None,
        help=(
            "Filter to one agent code version (e.g. 'v0' for the clean "
            "origin/main baseline). Read from each record's stamped "
            "'version' field, defaulting a missing value to "
            f"'{DEFAULT_VERSION}'. Omit to include all versions (legacy "
            "behaviour). See eval/versioning.py for the taxonomy."
        ),
    )
    parser.add_argument(
        "--no-gcs", action="store_true",
        help=(
            "Don't fall back to GCS for missing local manifests. Run files "
            "whose manifest isn't already cached locally get skipped."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of the human table.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s",
    )

    # GCS client is constructed lazily on the first cache miss inside
    # ``gcs.read_manifest`` — eagerly constructing here would force every
    # caller to have ADC even when the local cache is complete and no GCS
    # fallback is actually needed.
    chosen, counters = collect_latest_per_task(
        benchmark=args.benchmark,
        mode=args.mode,
        agent_model=args.agent_model,
        allow_gcs=not args.no_gcs,
        version=args.version,
    )
    agg = aggregate(chosen)

    if args.json:
        payload = {
            "benchmark": args.benchmark,
            "mode": args.mode,
            "agent_model_filter": args.agent_model,
            "version_filter": args.version,
            "n_tasks": agg["n"],
            "cumulative_n_counts": agg["n_counts"],
            "partition_l_counts": agg["p_counts"],
            "counters": counters,
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(
        render(
            benchmark=args.benchmark,
            agent_model=args.agent_model,
            mode=args.mode,
            agg=agg,
            counters=counters,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
