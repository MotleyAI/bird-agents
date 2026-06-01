"""DEV-1515: offline re-grade CLI for already-completed runs.

`python -m bird_interact_agents.eval.regrade --run-id <id>
 [--instance-ids ...] [--benchmark ...] [--force-llm-judge]`

Contract:
* Walks `<results>/cloud/<run-id>/rows/<inst>/attempt-1.json` for each
  instance.
* Re-runs `grade_submission` with the locally-loaded LLM-judge cache
  at `<results>/cloud/<run-id>/llm_judge_cache.json`.
* OVERWRITES `<main_checkout>/annotations/<benchmark>/<db>/<inst>.submission.<run-id>.json`
  (this is the explicit re-grade path; distinct from `fetch`'s
  no-overwrite merge).
* Writes a fresh `<results>/cloud/<run-id>/eval_regraded.json` —
  the historical `eval.json` is NOT mutated.
* `--instance-ids` filters which rows get re-graded.
* `--force-llm-judge` invalidates cache entries for affected rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_attempt(run_dir: Path, instance_id: str, *, submitted_sql: str = "S"):
    d = run_dir / "rows" / instance_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "attempt-1.json").write_text(json.dumps({
        "instance_id": instance_id,
        "selected_database": "alien",
        "submitted_sql": submitted_sql,
        "trajectory": [],
        "usage": {"cost_usd_agent": 0.0, "cost_usd_user_sim": 0.0,
                  "n_agent_turns": 0, "n_ask_user_calls": 0},
        "duration_s": 0.0,
        "predicted_row_count": 0,
        "sol_sql": ["SELECT gold"],
        "original_sol_sql": ["SELECT gold"],
    }))


def test_regrade_walks_run_artefacts(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    for inst in ("alien_1", "alien_2"):
        _write_attempt(run_dir, inst)

    from bird_interact_agents.eval.regrade import regrade_run

    class StubGrader:
        def __call__(self, **kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary",
                novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    report = regrade_run(
        run_id="r1",
        benchmark="mini-interact",
        run_dir=run_dir,
        instance_ids=None,
        force_llm_judge=False,
        grader=StubGrader(),
        repo_root=tmp_path,
    )
    assert report.regraded == 2


def test_regrade_overwrites_existing_submission_annotation(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    # Pre-existing submission annotation at the destination.
    dest_dir = tmp_path / "annotations" / "mini_interact" / "alien"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "alien_1.submission.r1.json"
    dest.write_text('{"annotated_by": "stale", "kind": "submission_annotation"}')

    run_dir = tmp_path / "results" / "cloud" / "r1"
    _write_attempt(run_dir, "alien_1")

    from bird_interact_agents.eval.regrade import regrade_run

    class StubGrader:
        def __call__(self, **kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary",
                novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None, force_llm_judge=False,
        grader=StubGrader(), repo_root=tmp_path,
    )
    refreshed = json.loads(dest.read_text())
    assert refreshed["annotated_by"] != "stale"


def test_regrade_respects_instance_id_filter(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    for inst in ("alien_1", "alien_2", "alien_3"):
        _write_attempt(run_dir, inst)

    seen: list[str] = []

    class StubGrader:
        def __call__(self, **kw):
            seen.append(kw.get("submitted_sql", "?"))
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary",
                novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    from bird_interact_agents.eval.regrade import regrade_run

    regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=["alien_2"], force_llm_judge=False,
        grader=StubGrader(), repo_root=tmp_path,
    )
    assert len(seen) == 1


def test_regrade_writes_eval_regraded_not_eval_json(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    _write_attempt(run_dir, "alien_1")
    # Historical eval.json — must NOT be overwritten by regrade.
    eval_json = run_dir / "eval.json"
    eval_json.write_text('{"phase1_count": 999}')

    class StubGrader:
        def __call__(self, **kw):
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=True, n2_audited_primary=True,
                n3_any_audited_variant=True, n4_tie_order=True,
                n5_llm_judge=True, n6_numeric_epsilon=True,
                n7_trailing_whitespace=True, n8_column_order=True,
                n9_case_fold=True,
                matched_variant_id="primary",
                novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    from bird_interact_agents.eval.regrade import regrade_run

    regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None, force_llm_judge=False,
        grader=StubGrader(), repo_root=tmp_path,
    )

    eval_regraded = run_dir / "eval_regraded.json"
    assert eval_regraded.exists()
    # Historical preserved.
    assert json.loads(eval_json.read_text())["phase1_count"] == 999


def test_regrade_force_llm_judge_clears_cache_entries(tmp_path, monkeypatch):
    """`--force-llm-judge` MUST drop matching keys from
    `<results>/cloud/<run-id>/llm_judge_cache.json` before re-grading.

    Cache entries embed `instance_id` so the clearer can filter. The
    contract: after `clear_llm_judge_cache(..., instance_ids=["alien_1"])`
    NO cached entry whose key/value references `alien_1` remains; entries
    for OTHER instances are preserved.
    """
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    _write_attempt(run_dir, "alien_1")

    cache_path = run_dir / "llm_judge_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "k_for_alien_1": {"instance_id": "alien_1", "verdict": True},
        "k_for_alien_2": {"instance_id": "alien_2", "verdict": False},
    }))

    from bird_interact_agents.eval.regrade import clear_llm_judge_cache

    clear_llm_judge_cache(
        cache_path=cache_path, instance_ids=["alien_1"],
    )
    remaining = json.loads(cache_path.read_text())
    assert "k_for_alien_1" not in remaining, (
        "force_llm_judge must drop the alien_1 cache entry"
    )
    assert "k_for_alien_2" in remaining, (
        "entries for other instances must be preserved"
    )


def test_regrade_run_force_llm_judge_reinvokes_judge(tmp_path, monkeypatch):
    """End-to-end: `regrade_run(..., force_llm_judge=True)` calls the
    grader exactly once and the cache is empty afterward for the
    filtered instances."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    _write_attempt(run_dir, "alien_1")
    cache_path = run_dir / "llm_judge_cache.json"
    cache_path.write_text(json.dumps({
        "k_for_alien_1": {"instance_id": "alien_1", "verdict": True},
    }))

    calls: list[dict] = []

    class StubGrader:
        def __call__(self, **kw):
            calls.append(kw)
            from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
            return CascadeVerdict(
                n1_original_gold=False, n2_audited_primary=False,
                n3_any_audited_variant=False, n4_tie_order=False,
                n5_llm_judge=False, n6_numeric_epsilon=False,
                n7_trailing_whitespace=False, n8_column_order=False,
                n9_case_fold=False,
                matched_variant_id=None, novel_reading_judgment=None,
                variant_matches=[], rowset_relations=[],
            )

    from bird_interact_agents.eval.regrade import regrade_run

    regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=["alien_1"], force_llm_judge=True,
        grader=StubGrader(), repo_root=tmp_path,
    )
    assert len(calls) == 1
    remaining = json.loads(cache_path.read_text())
    assert "k_for_alien_1" not in remaining
