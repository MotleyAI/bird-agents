"""DEV-1515: offline re-grade CLI for already-completed runs.

`python -m bird_interact_agents.eval.regrade --run-id <id>
 [--instance-ids ...] --benchmark ...`

Contract:
* Walks `<results>/cloud/<run-id>/rows/<inst>/attempt-N.json` for each
  instance (highest N wins).
* Re-runs `grade_submission` with the locally-loaded LLM-judge cache
  at `<results>/cloud/<run-id>/llm_judge_cache.json`. The judge uses
  the agent's own model (read from `<run_dir>/manifest.json`) and is
  invoked automatically when `metadata_sufficiency.verdict ==
  "insufficient"`.
* OVERWRITES `<main_checkout>/annotations/<benchmark>/<db>/<inst>.submission.<run-id>.json`
  (this is the explicit re-grade path; distinct from `fetch`'s
  no-overwrite merge).
* Writes a fresh `<results>/cloud/<run-id>/eval_regraded.json` —
  the historical `eval.json` is NOT mutated.
* `--instance-ids` filters which rows get re-graded.
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
        instance_ids=None,
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

    report = regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=["alien_2"],
        grader=StubGrader(), repo_root=tmp_path,
    )
    assert len(seen) == 1, "grader should be called exactly once (for alien_2)"
    assert report.regraded == 1
    assert report.regraded_instances == ["alien_2"]
    assert report.skipped == 2  # alien_1 and alien_3 filtered out


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
        instance_ids=None,
        grader=StubGrader(), repo_root=tmp_path,
    )

    eval_regraded = run_dir / "eval_regraded.json"
    assert eval_regraded.exists()
    # Historical preserved.
    assert json.loads(eval_json.read_text())["phase1_count"] == 999


# ---------------------------------------------------------------------------
# Codex round 6: ``_build_original_sql_index`` MUST accept string-shaped
# ``sol_sql``. The mini_interact JSONL carries both shapes (post-DEV-1478
# is list; older rows / tests / fixtures pass a bare string). Pre-fix the
# ``isinstance(sol, list)`` filter silently dropped the string rows, so
# regrade fell through to ``original_sql_by_inst.get(iid) → []`` and N1
# could never pass for those instances.
# ---------------------------------------------------------------------------


def test_build_original_sql_index_accepts_string_sol_sql(tmp_path, monkeypatch):
    """Mini-interact data file with both string + list shapes: both
    must land in the index after normalize_sol_sql wraps the string."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.eval.regrade import _build_original_sql_index

    data_file = tmp_path / "mini_interact.jsonl"
    data_file.write_text(
        json.dumps({
            "instance_id": "alien_string_sol",
            "sol_sql": "SELECT 1 FROM t",
        }) + "\n"
        + json.dumps({
            "instance_id": "alien_list_sol",
            "sol_sql": ["SELECT 2 FROM u"],
        }) + "\n"
        + json.dumps({
            "instance_id": "alien_no_sol",
            # ``sol_sql`` absent — expected to be skipped.
        }) + "\n"
    )

    # Pretend the data file lives at this temp path.
    monkeypatch.setattr(
        paths_mod, "benchmark_data_file",
        lambda benchmark: data_file,
    )

    out = _build_original_sql_index("mini_interact")
    assert out["alien_string_sol"] == ["SELECT 1 FROM t"], (
        "string-shaped sol_sql must be wrapped as a 1-item list, NOT "
        "dropped or character-split"
    )
    assert out["alien_list_sol"] == ["SELECT 2 FROM u"]
    assert "alien_no_sol" not in out


def test_regrade_picks_latest_attempt_after_resubmit(tmp_path, monkeypatch):
    """Codex r10: regrade MUST read the highest ``attempt-N.json``, not
    the hardcoded attempt-1. Pre-fix a resubmit's attempt-2 was either
    skipped (when attempt-1 was absent) or its SQL got silently
    overwritten by stale attempt-1 data."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    d = run_dir / "rows" / "alien_1"
    d.mkdir(parents=True)
    # Stale attempt-1: this SQL must NOT be the one graded.
    (d / "attempt-1.json").write_text(json.dumps({
        "instance_id": "alien_1",
        "selected_database": "alien",
        "submitted_sql": "STALE_ATTEMPT_1_SQL",
        "trajectory": [],
        "usage": {},
        "duration_s": 0.0,
        "sol_sql": ["SELECT gold"],
        "original_sol_sql": ["SELECT gold"],
    }))
    # Fresh attempt-3: this is the one regrade should pick.
    (d / "attempt-3.json").write_text(json.dumps({
        "instance_id": "alien_1",
        "selected_database": "alien",
        "submitted_sql": "FRESH_ATTEMPT_3_SQL",
        "trajectory": [],
        "usage": {},
        "duration_s": 0.0,
        "sol_sql": ["SELECT gold"],
        "original_sol_sql": ["SELECT gold"],
    }))

    captured: list[dict] = []

    class StubGrader:
        def __call__(self, *, instance_id, submitted_sql, task_row):
            captured.append({
                "instance_id": instance_id,
                "submitted_sql": submitted_sql,
            })
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
        instance_ids=None,
        grader=StubGrader(), repo_root=tmp_path,
    )
    assert len(captured) == 1
    assert captured[0]["submitted_sql"] == "FRESH_ATTEMPT_3_SQL", (
        f"regrade must pick the HIGHEST attempt-N.json, got "
        f"{captured[0]['submitted_sql']!r} — pre-fix this would be "
        f"'STALE_ATTEMPT_1_SQL'"
    )


def test_regrade_grades_when_only_later_attempt_exists(tmp_path, monkeypatch):
    """Companion case: a resubmit may produce ONLY attempt-2 (the
    earlier attempt-1.json was never written or was cleaned up). Pre-fix
    that instance was silently skipped because the hardcoded path
    didn't exist."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    d = run_dir / "rows" / "alien_1"
    d.mkdir(parents=True)
    (d / "attempt-2.json").write_text(json.dumps({
        "instance_id": "alien_1",
        "selected_database": "alien",
        "submitted_sql": "ONLY_ATTEMPT_2",
        "trajectory": [],
        "usage": {},
        "duration_s": 0.0,
    }))

    captured: list[dict] = []

    class StubGrader:
        def __call__(self, *, instance_id, submitted_sql, task_row):
            captured.append({
                "instance_id": instance_id,
                "submitted_sql": submitted_sql,
            })
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

    report = regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None,
        grader=StubGrader(), repo_root=tmp_path,
    )
    assert report.regraded == 1
    assert captured[0]["submitted_sql"] == "ONLY_ATTEMPT_2"


def test_build_original_sql_index_does_not_char_split_string(
    tmp_path, monkeypatch,
):
    """Defensive: a long-string ``sol_sql`` returns a 1-element list
    whose element is the verbatim SQL, NOT a per-character list."""
    from bird_interact_agents import paths as paths_mod
    from bird_interact_agents.eval.regrade import _build_original_sql_index

    sql = "WITH cte AS (SELECT * FROM t) SELECT a, b FROM cte WHERE x = 1"
    data_file = tmp_path / "mini_interact.jsonl"
    data_file.write_text(
        json.dumps({"instance_id": "x_1", "sol_sql": sql}) + "\n",
    )
    monkeypatch.setattr(
        paths_mod, "benchmark_data_file",
        lambda benchmark: data_file,
    )

    out = _build_original_sql_index("mini_interact")
    assert out["x_1"] == [sql]
    assert len(out["x_1"]) == 1
