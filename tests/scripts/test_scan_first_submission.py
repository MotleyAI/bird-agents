"""Tests for ``scripts/scan_first_submission.py``.

Covers the two contracts that matter: (1) the first-submission classifier
reads the ``Result match:`` signal out of BOTH trajectory encodings and the
error/no-submit fallbacks, and (2) the scan picks the latest non-eval_failed
opus run per ``(db, iid, mode)`` and joins it to the right session log.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_first_submission.py"
)
_spec = importlib.util.spec_from_file_location("scan_first_submission", SCRIPT)
scan_first_submission = importlib.util.module_from_spec(_spec)
sys.modules["scan_first_submission"] = scan_first_submission
_spec.loader.exec_module(scan_first_submission)
m = scan_first_submission


# --------------------------------------------------------------------------- #
# trajectory-event builders for the two encodings
# --------------------------------------------------------------------------- #
def _repr_submit(tid: str, tool: str = "mcp__bird-interact-tools__submit_sql"):
    return {
        "type": "AssistantMessage",
        "data": (
            f"AssistantMessage(content=[ToolUseBlock(id='{tid}', "
            f"name='{tool}', input={{'sql': 'SELECT 1'}})], model='x')"
        ),
    }


def _repr_result(tid: str, text: str):
    return {
        "type": "UserMessage",
        "data": (
            f"UserMessage(content=[ToolResultBlock(tool_use_id='{tid}', "
            f"content=[{{'type': 'text', 'text': '{text}'}}], is_error=None)])"
        ),
    }


def _dict_submit(tid: str, tool: str = "mcp__bird-interact-tools__submit_query"):
    return {
        "type": "AssistantMessage",
        "data": {"content": [{"id": tid, "name": tool, "input": {}}]},
    }


def _dict_result(tid: str, text: str):
    return {
        "type": "UserMessage",
        "data": {"content": [{"tool_use_id": tid, "content": text}]},
    }


# --------------------------------------------------------------------------- #
# first_submission_outcome — both encodings, all branches
# --------------------------------------------------------------------------- #
def test_repr_first_submit_correct():
    traj = [_repr_submit("a"), _repr_result("a", "Submitted. Result match: True")]
    assert m.first_submission_outcome(traj) == "correct"


def test_repr_first_submit_wrong():
    traj = [_repr_submit("a"), _repr_result("a", "Submitted. Result match: False")]
    assert m.first_submission_outcome(traj) == "wrong"


def test_dict_first_submit_correct():
    traj = [_dict_submit("z"), _dict_result("z", "Generated SQL: ...\nResult match: True")]
    assert m.first_submission_outcome(traj) == "correct"


def test_dict_first_submit_error_execution():
    # slayer submit_query whose generated SQL failed → no Result-match line
    traj = [_dict_submit("z"), _dict_result("z", "Generated SQL: ...\nResult: SQL execution error")]
    assert m.first_submission_outcome(traj) == "error"


def test_repr_first_submit_dry_run_error():
    traj = [_repr_submit("a"),
            _repr_result("a", "Submitted SQL failed dry-run (DB error): UndefinedColumn")]
    assert m.first_submission_outcome(traj) == "error"


def test_no_submit_call():
    traj = [{"type": "AssistantMessage",
             "data": "AssistantMessage(content=[ToolUseBlock(id='x', "
                     "name='mcp__bird-interact-tools__execute_sql', input={})])"}]
    assert m.first_submission_outcome(traj) == "nosubmit"


def test_only_first_submit_counts():
    # first wrong, second correct → outcome is 'wrong' (we score the FIRST)
    traj = [
        _repr_submit("a"), _repr_result("a", "Result match: False"),
        _repr_submit("b"), _repr_result("b", "Result match: True"),
    ]
    ids, _ = m.extract_submissions(traj)
    assert ids == ["a", "b"]
    assert m.first_submission_outcome(traj) == "wrong"


def test_is_opus_match():
    assert m.is_opus_match("anthropic/claude-opus-4-8", "opus")
    assert m.is_opus_match("anthropic/claude-OPUS-4-8", "opus")
    assert not m.is_opus_match("anthropic/claude-sonnet-5", "opus")
    assert not m.is_opus_match("", "opus")


# --------------------------------------------------------------------------- #
# select_latest + end-to-end scan against a synthetic runs/results tree
# --------------------------------------------------------------------------- #
_BM = "livesqlbench-large"


def _write_run(runs_root: Path, *, db, iid, mode, run_id, at,
               n1_pass, verdict="correct", agent_model="anthropic/claude-opus-4-8"):
    dest = runs_root / _BM / db / iid / f"{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "kind": "submission_annotation",
        "instance_id": iid,
        "selected_database": db,
        "annotated_at": at,
        "agent_model": agent_model,
        "submission": {
            "cloud_run_id": run_id,
            "trajectory_path": f"rows/{iid}/attempt-1.json",
            "config": {"query_mode": mode, "agent_model": agent_model,
                       "dataset": _BM},
        },
        "evaluation": {
            "phase1_against_original_gold": "pass" if n1_pass else "agent_miss",
            "verdict": verdict,
        },
    }))
    return dest


def _write_log(results_root: Path, *, run_id, iid, trajectory):
    dest = results_root / run_id / "rows" / iid / "attempt-1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"trajectory": trajectory}))


def test_select_latest_skips_eval_failed_and_picks_newest(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, db="d1", iid="d1_1", mode="raw", run_id="r_old",
               at="20260101t0000", n1_pass=False)
    _write_run(runs, db="d1", iid="d1_1", mode="raw", run_id="r_new",
               at="20260201t0000", n1_pass=True)
    # a still-newer run that eval_failed must NOT win
    _write_run(runs, db="d1", iid="d1_1", mode="raw", run_id="r_broken",
               at="20260301t0000", n1_pass=False, verdict="eval_failed")
    latest = m.select_latest(runs, _BM, "opus")
    assert set(latest) == {("d1", "d1_1", "raw")}
    assert latest[("d1", "d1_1", "raw")]["submission"]["cloud_run_id"] == "r_new"


def test_select_latest_filters_non_opus(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, db="d1", iid="d1_1", mode="raw", run_id="r1",
               at="20260101t0000", n1_pass=True,
               agent_model="anthropic/claude-sonnet-5")
    assert m.select_latest(runs, _BM, "opus") == {}


def test_scan_paired_end_to_end(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    # paired task t1: raw first-correct, slayer first-wrong-then-final-pass
    _write_run(runs, db="d1", iid="t1", mode="raw", run_id="raw1",
               at="20260101t0000", n1_pass=True)
    _write_log(results, run_id="raw1", iid="t1",
               trajectory=[_repr_submit("a"), _repr_result("a", "Result match: True")])
    _write_run(runs, db="d1", iid="t1", mode="slayer", run_id="sla1",
               at="20260101t0000", n1_pass=True)
    _write_log(results, run_id="sla1", iid="t1", trajectory=[
        _dict_submit("z1"), _dict_result("z1", "Result match: False"),
        _dict_submit("z2"), _dict_result("z2", "Result match: True")])
    # raw-only task t2 (drops out of the paired scope)
    _write_run(runs, db="d1", iid="t2", mode="raw", run_id="raw2",
               at="20260101t0000", n1_pass=False)
    _write_log(results, run_id="raw2", iid="t2",
               trajectory=[_repr_submit("a"), _repr_result("a", "Result match: False")])

    out = m.scan(runs, results, _BM, "opus", ["raw", "slayer"], paired=True)
    raw, sla = out["modes"]["raw"], out["modes"]["slayer"]
    assert raw["tasks"] == 1 and sla["tasks"] == 1          # paired → only t1
    assert raw["correct"] == 1 and raw["has_log"] == 1
    assert sla["wrong"] == 1 and sla["correct"] == 0        # first submit wrong
    assert sla["final_pass"] == 1                            # but final N1 pass
    assert out["both_logged_count"] == 1
    assert raw["first_correct_on_both_logged"] == 1
    assert sla["first_correct_on_both_logged"] == 0


def test_scan_counts_missing_log(tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    _write_run(runs, db="d1", iid="t1", mode="raw", run_id="raw1",
               at="20260101t0000", n1_pass=True)
    # no log written for raw1
    out = m.scan(runs, results, _BM, "opus", ["raw"], paired=False)
    assert out["modes"]["raw"]["no_log"] == 1
    assert out["modes"]["raw"]["has_log"] == 0
