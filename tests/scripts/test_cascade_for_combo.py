"""Tests for ``scripts/cascade_for_combo.py``.

Covers the contract that matters: pick the latest non-eval_failed run per
(db, instance_id) AFTER filtering by manifest's ``query_mode`` AND
``agent_model`` substring. The cumulative-N and partition-L aggregations
reuse the production helpers in
``bird_interact_agents.eval.cascading_report`` so are not re-tested here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "cascade_for_combo.py"
)
_spec = importlib.util.spec_from_file_location("cascade_for_combo", SCRIPT)
cascade_for_combo = importlib.util.module_from_spec(_spec)
sys.modules["cascade_for_combo"] = cascade_for_combo
_spec.loader.exec_module(cascade_for_combo)


_BENCHMARK = "mini-interact"


def _write_annotation(
    *,
    runs_root: Path,
    db: str,
    iid: str,
    run_id: str,
    annotated_at: str,
    n1: bool = False,
    n2: bool = False,
    verdict: str = "correct",
    version=None,
    agent_model=None,
) -> Path:
    """Write a minimal SubmissionAnnotation JSON. Mode is encoded into the
    run_id (caller threads ``-raw-`` / ``-slayer-``)."""
    dest = runs_root / _BENCHMARK / db / iid / f"{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "version": version,
                "agent_model": agent_model,
                "schema_version": 1,
                "kind": "submission_annotation",
                "instance_id": iid,
                "selected_database": db,
                "task_annotation_ref": (
                    f"annotations/{_BENCHMARK}/{db}/{iid}.task.json"
                ),
                "annotated_by": "test",
                "annotated_at": annotated_at,
                "submission": {
                    "cloud_run_id": run_id,
                    "trajectory_path": f"rows/{iid}/attempt-1.json",
                    "submitted_sql_path": None,
                    "predicted_row_count": 1,
                    "duration_s": 1.0,
                    "cost_usd_agent": 0.0,
                    "cost_usd_user_sim": 0.0,
                    "n_agent_turns": 1,
                    "n_ask_user_calls": 0,
                },
                "evaluation": {
                    "phase1_against_original_gold": "pass" if n1 else "fail",
                    "phase1_against_audited_primary": "pass" if n2 else "fail",
                    "phase1_against_any_audited_variant": "pass" if n2 else "fail",
                    "phase1_against_variants": [],
                    "correct_up_to_tie_order": False,
                    "novel_reading_judgment": None,
                    "correct_under_numeric_epsilon": False,
                    "correct_under_trailing_whitespace": False,
                    "correct_under_column_order": False,
                    "correct_under_case_fold": False,
                    "numeric_epsilon": 1e-6,
                    "verdict": verdict,
                    "matched_variant_id": "primary" if n2 else None,
                    "rationale": "",
                },
                "failure_classification": {
                    "primary": "no_fail" if n1 or n2 else "agent_miss",
                    "secondary": [],
                    "agent_at_fault": not (n1 or n2),
                    "remediation_target": "other",
                    "remediation_text": "",
                    "details": "",
                },
                "decision_point": None,
                "user_sim_interaction": {
                    "n_asks": 0, "key_responses": [],
                    "disclosed_resolutions": [],
                    "undisclosed_resolutions": [],
                },
                "original_gold_annotated_correct": True,
            }
        )
    )
    return dest


def _write_manifest(
    *,
    results_root: Path,
    run_id: str,
    query_mode: str,
    agent_model: str,
) -> None:
    dest = (
        results_root / _BENCHMARK / "cloud" / run_id / "manifest.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {"query_mode": query_mode, "agent_model": agent_model}
        )
    )


@pytest.fixture
def isolated_tree(tmp_path, monkeypatch):
    """Point runs_root + results_root at tmp_path, and disable GCS lookup."""
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(results))
    return runs, results


def test_filters_by_mode_and_model_substring(isolated_tree):
    """Files outside the requested mode or model are excluded; substring
    match is case-insensitive."""
    runs, results = isolated_tree
    # opus / slayer — wanted, n1 pass
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_1",
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00", n1=True, n2=True,
    )
    _write_manifest(
        results_root=results,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        query_mode="slayer",
        agent_model="anthropic/claude-OPUS-4-7",  # uppercase — must still match
    )
    # haiku / slayer — same task, must be filtered out by --agent-model opus
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_2",
        run_id="20260602t1000-claudes-slayer-bbbbbb",
        annotated_at="2026-06-02T10:00:00+00:00", n1=True,
    )
    _write_manifest(
        results_root=results,
        run_id="20260602t1000-claudes-slayer-bbbbbb",
        query_mode="slayer",
        agent_model="anthropic/claude-haiku-4-5",
    )
    # opus / raw — same task, must be filtered out by --mode slayer
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_3",
        run_id="20260603t1000-claudesdk-raw-cccccc",
        annotated_at="2026-06-03T10:00:00+00:00", n1=True,
    )
    _write_manifest(
        results_root=results,
        run_id="20260603t1000-claudesdk-raw-cccccc",
        query_mode="raw",
        agent_model="anthropic/claude-opus-4-7",
    )

    chosen, counters = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK,
        mode="slayer",
        agent_model="opus",
        allow_gcs=False,
    )
    assert [p.name for p in chosen] == [
        "20260601t1000-claudes-slayer-aaaaaa.json"
    ], counters
    assert counters["matched_mode"] == 2  # two slayer files seen
    assert counters["matched_model"] == 1  # only the opus one passed
    assert counters["skipped_no_manifest"] == 0


def test_eval_failed_latest_falls_back_to_prior_real_verdict(isolated_tree):
    """If chronologically-latest is eval_failed AND a prior real verdict
    exists, the prior wins (and the counter records it)."""
    runs, results = isolated_tree
    iid = "alien_1"
    # earlier real verdict
    _write_annotation(
        runs_root=runs, db="alien", iid=iid,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00",
        n1=True, verdict="correct",
    )
    _write_manifest(
        results_root=results,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )
    # later eval_failed regrade — must NOT be picked
    _write_annotation(
        runs_root=runs, db="alien", iid=iid,
        run_id="20260612t1000-claudes-slayer-bbbbbb",
        annotated_at="2026-06-12T10:00:00+00:00",
        n1=False, verdict="eval_failed",
    )
    _write_manifest(
        results_root=results,
        run_id="20260612t1000-claudes-slayer-bbbbbb",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )

    chosen, counters = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK,
        mode="slayer",
        agent_model="opus",
        allow_gcs=False,
    )
    assert len(chosen) == 1
    assert "20260601t1000" in chosen[0].name
    assert counters["stale_eval_failed_overridden"] == 1


def test_eval_failed_only_task_is_omitted_from_aggregate(isolated_tree):
    """If every run for a task is ``eval_failed`` (no genuine verdict ever
    came through), the task is omitted from ``chosen`` so it doesn't get
    silently miscounted as an L11 hard fail in the cascade aggregate."""
    runs, results = isolated_tree
    iid = "alien_1"
    _write_annotation(
        runs_root=runs, db="alien", iid=iid,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00",
        n1=False, verdict="eval_failed",
    )
    _write_manifest(
        results_root=results,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )
    _write_annotation(
        runs_root=runs, db="alien", iid=iid,
        run_id="20260602t1000-claudes-slayer-bbbbbb",
        annotated_at="2026-06-02T10:00:00+00:00",
        n1=False, verdict="eval_failed",
    )
    _write_manifest(
        results_root=results,
        run_id="20260602t1000-claudes-slayer-bbbbbb",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )

    chosen, counters = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK,
        mode="slayer", agent_model="opus", allow_gcs=False,
    )
    assert chosen == []
    assert counters["skipped_eval_failed_only"] == 1
    assert counters["stale_eval_failed_overridden"] == 0


def test_legacy_invalid_plus_other_is_treated_as_eval_failed(isolated_tree):
    """Annotations from before the ``invalid`` → ``eval_failed`` migration
    (``verdict="invalid"`` + ``failure_classification.primary="other"``)
    must be classified as infra failures, mirroring
    ``SubmissionAnnotation._migrate_invalid_verdict``."""
    runs, results = isolated_tree
    iid = "alien_1"
    # earlier real verdict
    _write_annotation(
        runs_root=runs, db="alien", iid=iid,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00",
        n1=True, verdict="correct",
    )
    _write_manifest(
        results_root=results,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )
    # later LEGACY infra failure (verdict=invalid + primary=other) — must
    # be treated as eval_failed, NOT pick over the earlier correct verdict
    later = _write_annotation(
        runs_root=runs, db="alien", iid=iid,
        run_id="20260612t1000-claudes-slayer-bbbbbb",
        annotated_at="2026-06-12T10:00:00+00:00",
        n1=False, verdict="invalid",
    )
    raw = json.loads(later.read_text())
    raw["failure_classification"]["primary"] = "other"
    later.write_text(json.dumps(raw))
    _write_manifest(
        results_root=results,
        run_id="20260612t1000-claudes-slayer-bbbbbb",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )

    chosen, counters = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK,
        mode="slayer", agent_model="opus", allow_gcs=False,
    )
    assert len(chosen) == 1
    assert "20260601t1000" in chosen[0].name
    assert counters["stale_eval_failed_overridden"] == 1


# ---------------------------------------------------------------------------
# DEV-1591 stream 2 — version filter
# ---------------------------------------------------------------------------
def test_version_filter_excludes_newer_wrong_version(isolated_tree):
    """The pollution bug: a newer v2 run must NOT override the clean v0
    baseline when filtering ``--version v0``. The wrong-version record is
    excluded BEFORE the latest-per-task pick, so the older v0 wins."""
    runs, results = isolated_tree
    iid = "households_1"
    _write_annotation(
        runs_root=runs, db="households", iid=iid,
        run_id="20260625t1317-claudes-slayer-1330bf",
        annotated_at="2026-06-25T13:17:00+00:00",
        n1=True, verdict="correct",
        version="v0", agent_model="anthropic/claude-opus-4-7",
    )
    _write_manifest(
        results_root=results, run_id="20260625t1317-claudes-slayer-1330bf",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )
    # newer v2 (this branch) run for the SAME task
    _write_annotation(
        runs_root=runs, db="households", iid=iid,
        run_id="20260629t1209-claudes-slayer-cea364",
        annotated_at="2026-06-29T12:09:00+00:00",
        n1=False, verdict="agent_miss",
        version="v2", agent_model="anthropic/claude-opus-4-7",
    )
    _write_manifest(
        results_root=results, run_id="20260629t1209-claudes-slayer-cea364",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )

    chosen, _ = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK, mode="slayer", agent_model="opus",
        allow_gcs=False, version="v0",
    )
    assert [p.name for p in chosen] == [
        "20260625t1317-claudes-slayer-1330bf.json"
    ]
    # No version filter → latest (the v2 run) wins, reproducing the pollution.
    chosen_all, _ = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK, mode="slayer", agent_model="opus",
        allow_gcs=False,
    )
    assert [p.name for p in chosen_all] == [
        "20260629t1209-claudes-slayer-cea364.json"
    ]


def test_missing_version_defaults_to_v0(isolated_tree):
    """A legacy record (no ``version`` field) is treated as v0 for the
    filter (read-time default)."""
    runs, results = isolated_tree
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_1",
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00", n1=True,
        version=None,  # legacy — no version stamped
        agent_model="anthropic/claude-opus-4-7",
    )
    _write_manifest(
        results_root=results, run_id="20260601t1000-claudes-slayer-aaaaaa",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )
    chosen, _ = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK, mode="slayer", agent_model="opus",
        allow_gcs=False, version="v0",
    )
    assert len(chosen) == 1


def test_record_agent_model_used_when_manifest_missing(isolated_tree):
    """Codex High #2: a record stamped with ``agent_model`` must be matched
    on the RECORD, not require a present/matching manifest. With the manifest
    absent (and GCS disabled), the record is still selected."""
    runs, _results = isolated_tree
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_1",
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00", n1=True,
        version="v0", agent_model="anthropic/claude-opus-4-7",
    )
    # NB: no manifest written for this run_id.
    chosen, counters = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK, mode="slayer", agent_model="opus",
        allow_gcs=False, version="v0",
    )
    assert len(chosen) == 1, counters
    assert counters["matched_model"] == 1


def test_no_manifest_with_no_gcs_skips_run(isolated_tree):
    """When the local manifest is missing AND ``allow_gcs=False``, the run
    is skipped and the counter records it."""
    runs, _results = isolated_tree
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_1",
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00", n1=True,
    )
    # (no manifest written)
    chosen, counters = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK,
        mode="slayer",
        agent_model="opus",
        allow_gcs=False,
    )
    assert chosen == []
    assert counters["skipped_no_manifest"] == 1


def test_main_does_not_require_gcs_when_local_cache_is_complete(
    isolated_tree, monkeypatch, capsys,
):
    """``main()`` must not construct a GCS client when every needed
    manifest is already on disk — the canonical local-reporting invocation
    has to work on machines without ADC."""
    runs, results = isolated_tree
    _write_annotation(
        runs_root=runs, db="alien", iid="alien_1",
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        annotated_at="2026-06-01T10:00:00+00:00", n1=True, n2=True,
    )
    _write_manifest(
        results_root=results,
        run_id="20260601t1000-claudes-slayer-aaaaaa",
        query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
    )

    def _no_adc(*_a, **_kw):
        raise RuntimeError("ADC not available")

    monkeypatch.setattr(cascade_for_combo.gcs, "default_gcs_client", _no_adc)

    rc = cascade_for_combo.main([
        "--benchmark", _BENCHMARK,
        "--mode", "slayer", "--agent-model", "opus", "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_tasks"] == 1
    assert payload["cumulative_n_counts"]["n1"] == 1


def test_aggregate_partitions_correctly(isolated_tree):
    """End-to-end: two n1-pass + one n2-only + one fail should produce
    n1=2, n2=3, partition L1=2, L3=1, L11=1."""
    runs, results = isolated_tree
    cases = [
        ("alien_1", True,  True),   # n1+n2 → L1
        ("alien_2", True,  True),   # n1+n2 → L1
        ("alien_3", False, True),   # n2 only → L3
        ("alien_4", False, False),  # nothing → L11
    ]
    for i, (iid, n1, n2) in enumerate(cases):
        run_id = f"20260601t100{i}-claudes-slayer-{'a'*6}"
        _write_annotation(
            runs_root=runs, db="alien", iid=iid, run_id=run_id,
            annotated_at=f"2026-06-01T10:0{i}:00+00:00",
            n1=n1, n2=n2,
            verdict="correct" if (n1 or n2) else "agent_miss",
        )
        _write_manifest(
            results_root=results, run_id=run_id,
            query_mode="slayer", agent_model="anthropic/claude-opus-4-7",
        )
    chosen, _ = cascade_for_combo.collect_latest_per_task(
        benchmark=_BENCHMARK,
        mode="slayer", agent_model="opus", allow_gcs=False,
    )
    agg = cascade_for_combo.aggregate(chosen)
    assert agg["n"] == 4
    assert agg["n_counts"]["n1"] == 2
    assert agg["n_counts"]["n2"] == 3
    assert agg["p_counts"]["l1_correct_original"] == 2
    assert agg["p_counts"]["l3_audited_primary"] == 1
    assert agg["p_counts"]["l11_fail"] == 1
