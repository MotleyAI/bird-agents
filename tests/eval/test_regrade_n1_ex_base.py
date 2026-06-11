"""Tests for `scripts/regrade_n1_ex_base.py` (mini-interact backfill).

The script:
- walks `runs/mini-interact/<db>/<iid>/<run_id>.json`
- loads `submitted_sql` from JSON + `sol_sql` + `conditions` from
  the per-task annotation file
- detects mutation-bearing SQL (skip + log status `state-sensitive`)
- resolves SQLite db at `paths.benchmark_data_root("mini-interact") / db / f"{db}.sqlite"`
- RE-RUNS the FULL cascade (Codex finding #1 — N1 propagates into N2/N3
  and into `failure_classification`; the regrade must regenerate the
  entire `evaluation` block + classifier output, not flip one field)
- rewrites the JSON in place; idempotent on re-run.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_benchmark_root(tmp_path: Path, monkeypatch) -> Path:
    """A fake mini-interact root with one DB ``alien`` containing a single
    table ``t(val REAL)`` populated with `1.2345`. The same data shape is
    used for both the agent's submitted SQL and the gold."""
    parent = tmp_path / "benchmarks"
    parent.mkdir()
    root = parent / "mini-interact"
    db_dir = root / "alien"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "alien.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (val REAL)")
    conn.execute("INSERT INTO t (val) VALUES (1.2345)")
    conn.commit()
    conn.close()
    # `paths.benchmark_data_root('mini-interact')` honours
    # `BIRD_BENCHMARKS_ROOT` (parent dir), not `BIRD_BENCHMARK_DATA_ROOT`.
    monkeypatch.setenv("BIRD_BENCHMARKS_ROOT", str(parent))
    return root


@pytest.fixture
def fake_runs_root(tmp_path: Path, monkeypatch) -> Path:
    """A fake runs/ root (BIRD_RUNS_ROOT) for both the script's walk and
    the test's setup."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(root))
    return root


@pytest.fixture
def fake_annotations_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "annotations"
    root.mkdir()
    monkeypatch.setenv("BIRD_ANNOTATIONS_ROOT", str(root))
    return root


def _seed_result_json(
    runs_root: Path,
    annotations_root: Path,
    *,
    db: str,
    iid: str,
    run_id: str,
    submitted_sql: str,
    sol_sql: list[str],
    n1_original_gold: bool,
    failure_primary: str,
) -> Path:
    """Write a synthetic SubmissionAnnotation JSON + matching TaskAnnotation,
    using the project's own schema models so any field default that the
    regrade reads is realistic."""
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, MetadataSufficiency, Provenance,
        SubmissionAnnotation, SubmissionEvaluation, SubmissionMetadata,
        TaskAnnotation,
    )

    (annotations_root / "mini-interact" / db).mkdir(parents=True, exist_ok=True)
    task = TaskAnnotation(
        instance_id=iid,
        selected_database=db,
        annotated_by="test",
        annotated_at="2026-01-01",
        amb_user_query="How many rows?",
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="test",
        ),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=iid,
        ),
    )
    # Pin sol_sql onto the task annotation so the regrade picks it up.
    task_dict = task.model_dump()
    task_dict["sol_sql"] = sol_sql
    task_dict["conditions"] = None
    (annotations_root / "mini-interact" / db / f"{iid}.task.json").write_text(
        json.dumps(task_dict, indent=2)
    )

    inst_dir = runs_root / "mini-interact" / db / iid
    inst_dir.mkdir(parents=True, exist_ok=True)
    ev = SubmissionEvaluation(
        phase1_against_original_gold="pass" if n1_original_gold else "fail",
        phase1_against_audited_primary="pass" if n1_original_gold else "fail",
        phase1_against_any_audited_variant="pass" if n1_original_gold else "fail",
        phase1_against_variants=[],
        correct_up_to_tie_order=False,
        novel_reading_judgment=None,
        correct_under_numeric_epsilon=False,
        correct_under_trailing_whitespace=False,
        correct_under_column_order=False,
        correct_under_case_fold=False,
        numeric_epsilon=1e-6,
        verdict="correct" if n1_original_gold else "agent_miss",
        matched_variant_id=None,
        rationale="",
        miss_diagnostics=None,
    )
    fc = FailureClassification(
        primary=failure_primary,
        secondary=[],
        agent_at_fault=(failure_primary == "agent_miss"),
        remediation_target="other" if failure_primary == "no_fail" else "agent",
        remediation_text="",
        details="seeded by test",
    )
    submission = SubmissionMetadata(
        cloud_run_id=run_id,
        trajectory_path=f"rows/{iid}/attempt-1.json",
        submitted_sql_path=None,
        predicted_row_count=None,
        duration_s=1.0,
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        n_agent_turns=None,
        n_ask_user_calls=None,
    )
    ann = SubmissionAnnotation(
        schema_version=1,
        kind="submission_annotation",
        instance_id=iid,
        selected_database=db,
        task_annotation_ref=f"annotations/mini-interact/{db}/{iid}.task.json",
        annotated_by="seed",
        annotated_at="2026-06-11T00:00:00Z",
        submission=submission,
        evaluation=ev,
        failure_classification=fc,
        decision_point=None,
        user_sim_interaction=None,
        autopsy=None,
        submitted_sql=submitted_sql,
        predicted_result=None,
        gold_result=None,
        original_gold_annotated_correct=None,
    )
    out = inst_dir / f"{run_id}.json"
    out.write_text(json.dumps(ann.model_dump(mode="json"), indent=2))
    return out


def _run_regrade(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "regrade_n1_ex_base.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd or repo),
        check=False,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_regrade_script_flips_float_precision_case_full_cascade(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    """A task that previously failed N1 (4th-decimal float divergence)
    must flip to pass N1 AND propagate through every dependent field:
    `phase1_against_*`, `verdict`, `failure_classification.primary`,
    `failure_classification.agent_at_fault`."""
    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_1",
        run_id="20260611t1200-claudes-slayer-aaa111",
        # Pred returns 1.2345; gold returns 1.23 -> rounds to 1.23 at 2 dp.
        submitted_sql="SELECT val FROM t",
        sol_sql=["SELECT 1.23 AS val"],
        n1_original_gold=False,
        failure_primary="numerical_precision",
    )
    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    after = json.loads(p.read_text())
    # N1 flipped to pass.
    assert after["evaluation"]["phase1_against_original_gold"] == "pass"
    # Audited tiers and verdict consistent with the new N1.
    assert after["evaluation"]["phase1_against_audited_primary"] == "pass"
    assert after["evaluation"]["phase1_against_any_audited_variant"] == "pass"
    assert after["evaluation"]["verdict"] == "correct"
    # failure_classification re-derived.
    assert after["failure_classification"]["primary"] == "no_fail"
    assert after["failure_classification"]["agent_at_fault"] is False


def test_regrade_script_idempotent(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    """Running the script twice produces 0 additional flips and the JSON
    is byte-equal across the second run."""
    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_1",
        run_id="20260611t1200-claudes-slayer-aaa222",
        submitted_sql="SELECT val FROM t",
        sol_sql=["SELECT 1.23 AS val"],
        n1_original_gold=False,
        failure_primary="numerical_precision",
    )
    proc1 = _run_regrade()
    assert proc1.returncode == 0, proc1.stderr
    first = p.read_text()
    proc2 = _run_regrade()
    assert proc2.returncode == 0, proc2.stderr
    second = p.read_text()
    assert first == second
    # Second-run report names 0 flips.
    assert "regraded_flipped=0" in (proc2.stdout + proc2.stderr)


def test_regrade_script_skips_state_sensitive_mutation_pred(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    """Submitted SQL containing INSERT/UPDATE/etc is skipped — the
    backfill DB is pristine while inline grading saw the post-mutation
    state. The JSON is left untouched."""
    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_2",
        run_id="20260611t1200-claudes-slayer-bbb333",
        submitted_sql="INSERT INTO t (val) VALUES (9.99); SELECT val FROM t",
        sol_sql=["SELECT val FROM t"],
        n1_original_gold=False,
        failure_primary="agent_miss",
    )
    before = p.read_text()
    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    after = p.read_text()
    assert before == after
    assert "skipped_state_sensitive=1" in (proc.stdout + proc.stderr)


def test_regrade_script_skips_state_sensitive_mutation_gold(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    """Gold SQL containing mutations is also state-sensitive."""
    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_3",
        run_id="20260611t1200-claudes-slayer-ccc444",
        submitted_sql="SELECT val FROM t",
        sol_sql=["UPDATE t SET val = 1.23", "SELECT val FROM t"],
        n1_original_gold=False,
        failure_primary="agent_miss",
    )
    before = p.read_text()
    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    assert p.read_text() == before
    assert "skipped_state_sensitive=1" in (proc.stdout + proc.stderr)


def test_regrade_script_refuses_non_mini_interact_paths(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path, tmp_path: Path,
):
    """A result JSON under runs/livesqlbench-* is untouched (the script
    is mini-interact-only)."""
    # Seed a mini-interact result so the script has at least one task.
    p_mi = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_4",
        run_id="20260611t1200-claudes-slayer-ddd555",
        submitted_sql="SELECT val FROM t",
        sol_sql=["SELECT val FROM t"],
        n1_original_gold=True,
        failure_primary="no_fail",
    )
    # Drop a sibling fake livesqlbench result that the script must skip.
    lsb_path = (
        fake_runs_root / "livesqlbench-base-lite-sqlite" / "alien" / "alien_99"
        / "20260611t1200-claudes-slayer-eee666.json"
    )
    lsb_path.parent.mkdir(parents=True, exist_ok=True)
    lsb_path.write_text("{}")
    before = lsb_path.read_text()
    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    assert lsb_path.read_text() == before


def test_regrade_script_resolves_sqlite_path_via_benchmark_data_root(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    """Codex finding #4: the SQLite db_path is
    `benchmark_data_root("mini-interact") / db / f"{db}.sqlite"`, NOT
    `mini_interact_root() / 'databases' / db / f"{db}.sqlite"`. The
    fake_benchmark_root fixture lays out the former; the script must
    find the DB at that location."""
    _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_5",
        run_id="20260611t1200-claudes-slayer-fff777",
        submitted_sql="SELECT val FROM t",
        sol_sql=["SELECT val FROM t"],
        n1_original_gold=False,
        failure_primary="agent_miss",
    )
    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    # The script did not error with "db not found"; it processed the task.
    assert "regraded" in (proc.stdout + proc.stderr).lower()


def test_regrade_script_dry_run_does_not_write(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_6",
        run_id="20260611t1200-claudes-slayer-ggg888",
        submitted_sql="SELECT val FROM t",
        sol_sql=["SELECT 1.23 AS val"],
        n1_original_gold=False,
        failure_primary="numerical_precision",
    )
    before = p.read_text()
    proc = _run_regrade("--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert p.read_text() == before  # untouched
    # Dry-run reports the would-be flip and the write-counter is zero
    # (Codex round-2 finding #13).
    output = proc.stdout + proc.stderr
    assert "would_flip=1" in output
    assert "regraded_flipped=0" in output


# ---------------------------------------------------------------------------
# Codex round-2 additions
# ---------------------------------------------------------------------------


def test_regrade_script_rewrites_stale_tolerance_booleans(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path,
):
    """Codex round-2 finding #9: prove the script RECOMPUTES the full
    cascade rather than patching the N1 field. Seed a JSON with a
    deliberate pred-vs-gold mismatch AND tolerance booleans stamped to
    True (which the recompute will overwrite to False, since the rows
    truly don't overlap under any relaxed tolerance either). A naive
    'flip N1 only' impl would leave the stamped True values intact."""
    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="alien", iid="alien_stale",
        run_id="20260611t1200-claudes-slayer-stale01",
        # Pred returns 1.5 (CAST of stored 1.2345 to int via something);
        # gold returns 99.99 — guaranteed disjoint. Even under case-fold
        # / trailing-whitespace / column-order / numeric-epsilon, the
        # recompute will yield False.
        submitted_sql="SELECT 1.5 AS val",
        sol_sql=["SELECT 99.99 AS val"],
        n1_original_gold=False,
        failure_primary="agent_miss",
    )
    # Stamp deliberately stale `True` values for every tolerance boolean.
    # If the regrade only patched N1, these would survive verbatim.
    payload = json.loads(p.read_text())
    payload["evaluation"]["correct_up_to_tie_order"] = True
    payload["evaluation"]["correct_under_numeric_epsilon"] = True
    payload["evaluation"]["correct_under_trailing_whitespace"] = True
    payload["evaluation"]["correct_under_column_order"] = True
    payload["evaluation"]["correct_under_case_fold"] = True
    p.write_text(json.dumps(payload, indent=2))

    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    after = json.loads(p.read_text())
    # Pred (1.5) and gold (99.99) are disjoint, so the recompute writes
    # False everywhere — proving the script's recompute path overwrote
    # the stamped stale True values rather than copying them forward.
    assert after["evaluation"]["phase1_against_original_gold"] == "fail"
    assert after["evaluation"]["correct_up_to_tie_order"] is False
    assert after["evaluation"]["correct_under_numeric_epsilon"] is False
    assert after["evaluation"]["correct_under_trailing_whitespace"] is False
    assert after["evaluation"]["correct_under_column_order"] is False
    assert after["evaluation"]["correct_under_case_fold"] is False


def test_regrade_script_falls_back_to_jsonl_when_annotation_missing(
    fake_benchmark_root: Path, fake_runs_root: Path,
    fake_annotations_root: Path, tmp_path: Path, monkeypatch,
):
    """Codex round-2 finding #10: when the annotation file is missing,
    the script falls back to the canonical `mini_interact.jsonl` row for
    sol_sql + conditions. Seed a JSONL on disk, point the script at it
    via `BIRD_MINI_INTERACT_DATA_PATH`, omit the annotation file, run
    the regrade — must process the task, not skip it as
    `skipped_missing_inputs`."""
    iid = "alien_jsonl_only"
    db = "alien"
    inst_dir = fake_runs_root / "mini-interact" / db / iid
    inst_dir.mkdir(parents=True, exist_ok=True)

    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation,
        SubmissionEvaluation, SubmissionMetadata,
    )
    submission = SubmissionMetadata(
        cloud_run_id="20260611t1200-claudes-slayer-jsonl",
        trajectory_path=f"rows/{iid}/attempt-1.json",
    )
    ev = SubmissionEvaluation(
        phase1_against_original_gold="fail",
        phase1_against_audited_primary="fail",
        phase1_against_any_audited_variant="fail",
        phase1_against_variants=[],
        correct_up_to_tie_order=False,
        novel_reading_judgment=None,
        correct_under_numeric_epsilon=False,
        correct_under_trailing_whitespace=False,
        correct_under_column_order=False,
        correct_under_case_fold=False,
        numeric_epsilon=1e-6,
        verdict="agent_miss",
        matched_variant_id=None,
        rationale="",
        miss_diagnostics=None,
    )
    fc = FailureClassification(
        primary="agent_miss", secondary=[], agent_at_fault=True,
        remediation_target="agent", remediation_text="", details="seed",
    )
    ann = SubmissionAnnotation(
        instance_id=iid, selected_database=db,
        task_annotation_ref=f"annotations/mini-interact/{db}/{iid}.task.json",
        annotated_by="seed", annotated_at="2026-06-11T00:00:00Z",
        submission=submission, evaluation=ev, failure_classification=fc,
        submitted_sql="SELECT val FROM t",
    )
    result_path = inst_dir / "20260611t1200-claudes-slayer-jsonl.json"
    result_path.write_text(json.dumps(ann.model_dump(mode="json"), indent=2))

    # Write a canonical jsonl carrying sol_sql + conditions for this iid.
    jsonl_path = tmp_path / "mini_interact.jsonl"
    jsonl_path.write_text(json.dumps({
        "instance_id": iid,
        "selected_database": db,
        "sol_sql": ["SELECT val FROM t"],
        "conditions": None,
    }) + "\n")
    monkeypatch.setenv("BIRD_MINI_INTERACT_DATA_PATH", str(jsonl_path))

    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    output = proc.stdout + proc.stderr
    # Should be processed (not skipped_missing_inputs).
    assert "skipped_missing_inputs=0" in output
    assert "regraded" in output.lower()


def test_regrade_script_skips_when_sqlite_db_missing(
    fake_runs_root: Path, fake_annotations_root: Path, tmp_path: Path,
    monkeypatch,
):
    """Codex round-2 finding #11: when the SQLite DB file is missing
    (or unreadable), the script does NOT crash — it logs the task with
    a `missing_db` status and leaves the JSON untouched. The other
    tasks in the run still get processed."""
    parent = tmp_path / "benchmarks_missing"
    (parent / "mini-interact").mkdir(parents=True)
    monkeypatch.setenv("BIRD_BENCHMARKS_ROOT", str(parent))
    # benchmark_data_root('mini-interact')/<db>/<db>.sqlite is missing
    # for db='missing_db'.

    p = _seed_result_json(
        fake_runs_root, fake_annotations_root,
        db="missing_db", iid="missing_db_1",
        run_id="20260611t1200-claudes-slayer-missdb",
        submitted_sql="SELECT 1",
        sol_sql=["SELECT 1"],
        n1_original_gold=False,
        failure_primary="agent_miss",
    )
    before = p.read_text()
    proc = _run_regrade()
    assert proc.returncode == 0, proc.stderr
    assert p.read_text() == before
    assert "skipped_missing_inputs" in (proc.stdout + proc.stderr).lower() or \
           "missing_db" in (proc.stdout + proc.stderr).lower()
