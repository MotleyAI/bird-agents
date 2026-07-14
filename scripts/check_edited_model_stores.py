#!/usr/bin/env python
"""DEV-1671: scan saved edited-model stores and report KB-passing-state findings.

Resolves, per task, the inputs the deterministic checker needs and runs it:
  * store models    — untar ``edited_models.tar.gz`` → ``YAMLStorage``
  * baseline models — the OTF cache ``slayer_otf_cache/<benchmark>/<db>`` (guarded
                      by ``cache_fp`` == the store's ``_edited_models_meta.json``)
  * kb_rows         — ``_kb_rows.json`` inside the store
  * relevant_kb_ids — the task's ``external_knowledge`` (gated gold; advisory)
  * winning_query   — ``submitted_query`` of the latest successful attempt

Prints a per-store report and exits non-zero if any store has an ERROR finding
(``--fail-on flag`` also fails on flags). Read-only — never mutates a store.

Usage:
  uv run python scripts/check_edited_model_stores.py --benchmark livesqlbench-large \
      --instance-ids reverse_logistics_4,residential_data_4,fake_account_6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tarfile
import tempfile
from pathlib import Path

from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents import paths
from bird_interact_agents.slayer_otf import edited_models as _edited_models
from bird_interact_agents.slayer_otf.store_kb_checker import StoreCheckReport, check_store

_GATED_GOLD = {
    "livesqlbench-large": "gated_gold/livesqlbench-large/"
    "livesqlbench_large_v1_gt_kg_testcases_20260302(1).jsonl",
}


def _db_of(instance_id: str) -> str:
    return "_".join(instance_id.split("_")[:-1]) + "_large"


def _load_external_knowledge(benchmark: str, instance_id: str) -> list[int]:
    rel = _GATED_GOLD.get(benchmark)
    if not rel:
        return []
    path = _repo_root() / rel
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("instance_id") == instance_id:
            ek = row.get("external_knowledge") or []
            return [int(x) for x in ek]
    return []


def _latest_slayer_attempt(benchmark: str, instance_id: str) -> tuple[dict | None, str | None]:
    """Return (attempt_dict, run_id) for the store's supporting query: the LATEST
    (by run-id timestamp) SLAYER save-models attempt that carries a ``submitted_query``.

    A store is only written by a slayer run (``--save-edited-models``), and the archive
    is latest-wins-overwrite — so the current store reflects the latest slayer run.
    Raw runs (no ``submitted_query``) and same-timestamp-tied non-slayer runs are ignored
    (the earlier ``reward>=`` heuristic wrongly picked the chronologically-last _raw_ run)."""
    results_root = _repo_root() / "results" / benchmark
    if not results_root.exists():
        return None, None
    candidates: list[tuple[str, str, dict]] = []
    for run_dir in results_root.iterdir():
        if "slayer" not in run_dir.name:
            continue
        row_dir = run_dir / "rows" / instance_id
        if not row_dir.is_dir():
            continue
        for att in sorted(row_dir.glob("attempt-*.json")):
            try:
                d = json.loads(att.read_text())
            except Exception:
                continue
            if d.get("submitted_query"):
                candidates.append((run_dir.name, att.name, d))
    if not candidates:
        return None, None
    # latest by (run-id timestamp, attempt file) so a tie on run-id picks the last attempt
    run_id, _att, attempt = max(candidates, key=lambda c: (c[0], c[1]))
    return attempt, run_id


def _repo_root() -> Path:
    # gitignored data lives in the MAIN checkout (worktree-safe, DEV path rule).
    return Path(paths.main_checkout_root())


async def _check_one(benchmark: str, instance_id: str) -> StoreCheckReport:
    db = _db_of(instance_id)
    archive = _edited_models.run_edited_models_archive(
        benchmark=benchmark, selected_database=db, instance_id=instance_id,
    )
    if not Path(archive).exists():
        return StoreCheckReport.from_findings(
            [], instance_id=instance_id, db=db, benchmark=benchmark, baseline_available=False,
        )
    attempt, _run = _latest_slayer_attempt(benchmark, instance_id)
    winning_query = None
    reward = None
    if attempt is not None:
        raw = attempt.get("submitted_query")
        winning_query = json.loads(raw) if isinstance(raw, str) else raw
        reward = attempt.get("total_reward")
    if winning_query is None:
        return StoreCheckReport.from_findings(
            [], instance_id=instance_id, db=db, benchmark=benchmark, reward=reward,
            baseline_available=False,
        )

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(archive) as t:
            t.extractall(tmp)
        store_dir = Path(tmp) / db
        # A cleaned store that REFORMULATED its query (e.g. demoted an answer-baking
        # measure into the query) records the supported query as a sidecar; prefer
        # it over the (now-stale) attempt query.
        sidecar = store_dir / "_winning_query.json"
        if sidecar.exists():
            winning_query = json.loads(sidecar.read_text())
        kb_rows = json.loads((store_dir / "_kb_rows.json").read_text())

        # Baseline = the OTF cache models. The store's cache_fp can differ from the
        # current cache_fp when the cache was REBUILT (slayer/embedding version bump)
        # since the store was saved — but that is a COSMETIC fingerprint change: the
        # baseline MODEL CONTENT is stable (verified: 48/53 fake_account models
        # byte-identical, the rest being the agent's own [kb] edits). So use the cache
        # whenever present; skip only when the cache is genuinely absent.
        cache_dir = paths.slayer_otf_cache_root(benchmark=benchmark) / db
        baseline_storage = None
        if cache_dir.exists():
            baseline_storage = YAMLStorage(base_dir=str(cache_dir))

        store_storage = YAMLStorage(base_dir=str(store_dir))
        relevant = _load_external_knowledge(benchmark, instance_id)
        return await check_store(
            store_storage=store_storage,
            baseline_storage=baseline_storage,
            db=db,
            kb_rows=kb_rows,
            relevant_kb_ids=relevant,
            winning_query=winning_query,
            instance_id=instance_id,
            benchmark=benchmark,
            reward=reward,
        )


def _print_report(rep: StoreCheckReport) -> None:
    status = "OK" if rep.ok else "FAIL"
    if not rep.baseline_available:
        status = "SKIPPED (baseline unavailable)"
    print(f"\n=== {rep.instance_id} [{rep.db}] reward={rep.reward} -> {status} ===")
    for f in rep.findings:
        kb = f" kb={f.kb_id}" if f.kb_id is not None else ""
        loc = ".".join(x for x in (f.model, f.entity) if x)
        print(f"  [{f.level.upper():5}] {f.category:28} {loc}{kb}: {f.detail}")
    if not rep.findings and rep.baseline_available:
        print("  (no findings — passing state)")


async def _amain(args: argparse.Namespace) -> int:
    ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    reports = [await _check_one(args.benchmark, iid) for iid in ids]
    if args.json:
        print(json.dumps([r.model_dump() for r in reports], indent=2))
    else:
        for r in reports:
            _print_report(r)
    has_error = any(not r.ok and r.baseline_available for r in reports)
    has_flag = any(r.findings for r in reports)
    if args.fail_on == "flag" and has_flag:
        return 1
    return 1 if has_error else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan saved edited-model stores (DEV-1671).")
    ap.add_argument("--benchmark", default="livesqlbench-large")
    ap.add_argument("--instance-ids", required=True, help="comma-separated instance ids")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on", choices=["error", "flag"], default="error")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
