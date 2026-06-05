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

    # Pre-existing run annotation at the destination (DEV-1533: runs/).
    dest_dir = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "r1.json"
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


def test_regrade_preserves_dev1533_run_result_fields(tmp_path, monkeypatch):
    """DEV-1533: regrade must populate ``submitted_sql``,
    ``predicted_result``, ``gold_result`` AND
    ``original_gold_annotated_correct`` on the rewritten annotation.
    Pre-fix the regrade-built annotation omitted all four, so OVERWRITING
    the runs/ store erased the data that DEV-1533 was created to keep
    (and flipped the L1/L2 partition tier signal to None)."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    run_dir = tmp_path / "results" / "cloud" / "r1"
    # Hand-craft the attempt so it carries the same run-data fields the
    # harness writes at run time.
    d = run_dir / "rows" / "alien_1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "attempt-1.json").write_text(json.dumps({
        "instance_id": "alien_1",
        "selected_database": "alien",
        "submitted_sql": "SELECT agent_sql",
        "trajectory": [],
        "usage": {"cost_usd_agent": 0.0, "cost_usd_user_sim": 0.0,
                  "n_agent_turns": 0, "n_ask_user_calls": 0},
        "duration_s": 0.0,
        "predicted_row_count": 2,
        "sol_sql": ["SELECT gold"],
        "original_sol_sql": ["SELECT gold"],
        "predicted_result_json": json.dumps([[1, "x"], [2, "y"]]),
        "gold_result_json": json.dumps([[1, "x"], [2, "y"]]),
    }))

    # Pre-populate a task annotation marking the original gold as wrong;
    # the regrade MUST surface that on
    # ``original_gold_annotated_correct`` so the cascade's L1/L2
    # partition tier comes out right.
    from bird_interact_agents.eval.implicit_annotation import (
        implicit_task_annotation,
    )
    ta = implicit_task_annotation(
        instance_id="alien_1", selected_database="alien",
        benchmark="mini-interact", amb_user_query="q",
    )
    ta = ta.model_copy(update={"original_gold_is_correct": False})
    task_dir = tmp_path / "annotations" / "mini-interact" / "alien"
    task_dir.mkdir(parents=True)
    (task_dir / "alien_1.task.json").write_text(
        ta.model_dump_json(indent=2, exclude_none=False) + "\n"
    )

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

    written = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r1.json"
    assert written.exists()
    body = json.loads(written.read_text())
    assert body["submitted_sql"] == "SELECT agent_sql"
    assert body["predicted_result"] == [[1, "x"], [2, "y"]]
    assert body["gold_result"] == [[1, "x"], [2, "y"]]
    assert body["original_gold_annotated_correct"] is False, (
        "regrade must read original_gold_is_correct from the task "
        "annotation; got %r" % body.get("original_gold_annotated_correct")
    )


def test_regrade_with_explicit_repo_root_aggregates_from_that_root(
    tmp_path, monkeypatch,
):
    """When the caller passes ``repo_root=X`` (not the default
    ``paths.main_checkout_root()``), the writer lands annotations under
    ``X/runs/`` AND the aggregator MUST read from the SAME root so
    ``eval_regraded.json`` reflects the just-written annotations
    instead of computing zero counts off an unrelated default tree."""
    from bird_interact_agents import paths as paths_mod
    # Deliberately pin the default ``main_checkout_root`` to a SEPARATE
    # directory so the aggregator falling back to it would read an empty
    # tree. The fix threads ``repo_root`` through so the right tree is
    # used.
    default_root = tmp_path / "default-root"
    default_root.mkdir()
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: default_root)

    alt_root = tmp_path / "alt-root"
    alt_root.mkdir()

    run_dir = alt_root / "results" / "cloud" / "r1"
    _write_attempt(run_dir, "alien_1")

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
        grader=StubGrader(), repo_root=alt_root,
    )

    # Annotation went to alt_root.
    written = alt_root / "runs" / "mini-interact" / "alien" / "alien_1" / "r1.json"
    assert written.exists(), "regrade should write under alt_root/runs/"
    # The default root must NOT have been written to.
    assert not (default_root / "runs").exists(), (
        "explicit repo_root must not leak writes into the default root"
    )

    # The aggregator must read from alt_root too — otherwise eval_regraded.json
    # reports zero counts off the (empty) default root.
    eval_regraded = run_dir / "eval_regraded.json"
    assert eval_regraded.exists()
    body = json.loads(eval_regraded.read_text())
    assert body["phase1_count"] == 1, (
        f"eval_regraded.json must reflect annotations from alt_root; "
        f"got phase1_count={body.get('phase1_count')} (likely the "
        f"aggregator fell back to the default root and saw no annotations)"
    )


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

    out = _build_original_sql_index("mini-interact")
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


def test_latest_attempt_file_returns_none_for_missing_dir(tmp_path):
    """_latest_attempt_file returns None for a non-existent directory.
    Regression: previously called iterdir() directly, raising FileNotFoundError."""
    from bird_interact_agents.eval.regrade import _latest_attempt_file

    missing = tmp_path / "no_such_instance"
    assert _latest_attempt_file(missing) is None


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

    out = _build_original_sql_index("mini-interact")
    assert out["x_1"] == [sql]
    assert len(out["x_1"]) == 1
