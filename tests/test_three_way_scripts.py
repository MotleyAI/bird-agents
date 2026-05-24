"""Tests for scripts/select_tasks.py and scripts/compare_results.py.

Imports the script modules by file path so they can be tested without
installing them as a package.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compare_results = _load_module("compare_results", SCRIPTS_DIR / "compare_results.py")


# ── compare_results._norm_orig ───────────────────────────────────────────


def test_norm_orig_preserves_false_phase_flag():
    row = {
        "instance_id": "t1",
        "phase1_passed": False,
        "phase1_completed": True,
    }
    out = compare_results._norm_orig(row)
    assert out["phase1_passed"] is False, "explicit False must not be flipped to fallback"


def test_norm_orig_preserves_zero_reward():
    row = {
        "instance_id": "t1",
        "total_reward": 0,
        "last_reward": 1,
    }
    out = compare_results._norm_orig(row)
    assert out["total_reward"] == 0.0, "explicit 0 must not be flipped to fallback"


def test_norm_orig_falls_back_when_primary_missing():
    row = {
        "instance_id": "t1",
        "phase1_completed": True,
        "task_finished": True,
        "last_reward": 0.7,
    }
    out = compare_results._norm_orig(row)
    assert out["phase1_passed"] is True
    assert out["phase2_passed"] is True
    assert out["total_reward"] == pytest.approx(0.7)


def test_norm_orig_defaults_when_all_missing():
    out = compare_results._norm_orig({"instance_id": "t1"})
    assert out["phase1_passed"] is False
    assert out["phase2_passed"] is False
    assert out["total_reward"] == 0.0


def test_norm_orig_treats_none_as_missing():
    row = {
        "instance_id": "t1",
        "phase1_passed": None,
        "phase1_completed": True,
    }
    out = compare_results._norm_orig(row)
    assert out["phase1_passed"] is True


def test_norm_orig_drops_row_with_missing_instance_id():
    """A row without any usable instance_id is dropped (returns None) rather
    than collapsed under an empty string — otherwise multiple malformed
    records would silently overwrite one another in the per-task merge."""
    assert compare_results._norm_orig({"phase1_passed": True}) is None
    assert compare_results._norm_orig({"original_data": {}, "task_id": ""}) is None


def test_norm_ours_drops_row_with_missing_instance_id():
    assert compare_results._norm_ours({"phase1_passed": True}) is None
    assert compare_results._norm_ours({"task_id": None}) is None


def test_load_original_raises_when_file_missing(tmp_path: Path):
    """Default mode: missing expected file is a hard error so a misrouted
    run can't masquerade as a 0% baseline."""
    missing = tmp_path / "results.jsonl"
    with pytest.raises(FileNotFoundError, match="--allow-missing"):
        compare_results._load_original(missing, allow_missing=False)


def test_load_ours_raises_when_file_missing(tmp_path: Path):
    missing = tmp_path / "eval.json"
    with pytest.raises(FileNotFoundError, match="--allow-missing"):
        compare_results._load_ours(missing, allow_missing=False)


def test_load_original_allows_missing_with_flag(tmp_path: Path):
    assert compare_results._load_original(tmp_path / "x.jsonl", allow_missing=True) == []


def test_load_ours_allows_missing_with_flag(tmp_path: Path):
    assert compare_results._load_ours(tmp_path / "x.json", allow_missing=True) == []



# ── select_tasks: record-based pagination + --out-jsonl ─────────────────


def _write_jsonl_with_blanks(path: Path, rows: list[dict]) -> None:
    """Write rows interleaved with blank lines, to exercise the bug
    where raw-line pagination drifts off the record sequence."""
    chunks: list[str] = []
    for r in rows:
        chunks.append(json.dumps(r))
        chunks.append("")  # blank line after each record
    path.write_text("\n".join(chunks) + "\n")


def _run_select_tasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "select_tasks.py"), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_select_tasks_pagination_skips_records_not_lines(tmp_path: Path):
    data = tmp_path / "tasks.jsonl"
    _write_jsonl_with_blanks(
        data, [{"instance_id": f"t{i}"} for i in range(6)]
    )
    out_ids = tmp_path / "ids.txt"
    _run_select_tasks(
        "--data", str(data),
        "--start", "2",
        "--limit", "3",
        "--out", str(out_ids),
    )
    selected = out_ids.read_text().strip().splitlines()
    # Records 2, 3, 4 — NOT raw lines 2..4 (which would include blanks).
    assert selected == ["t2", "t3", "t4"]


def test_select_tasks_emits_filtered_jsonl(tmp_path: Path):
    data = tmp_path / "tasks.jsonl"
    rows = [{"instance_id": f"t{i}", "payload": i} for i in range(5)]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out_ids = tmp_path / "ids.txt"
    out_jsonl = tmp_path / "tasks_filtered.jsonl"
    _run_select_tasks(
        "--data", str(data),
        "--limit", "3",
        "--out", str(out_ids),
        "--out-jsonl", str(out_jsonl),
    )

    selected_ids = out_ids.read_text().strip().splitlines()
    assert selected_ids == ["t0", "t1", "t2"]

    emitted = [
        json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()
    ]
    assert [r["instance_id"] for r in emitted] == selected_ids
    assert [r["payload"] for r in emitted] == [0, 1, 2]


def test_select_tasks_rejects_negative_limit(tmp_path: Path):
    """Negative --limit fails at parse time with a clear message rather
    than silently producing an unintuitive selection."""
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"instance_id": "t0"}) + "\n")
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPTS_DIR / "select_tasks.py"),
            "--data", str(data),
            "--limit", "-1",
            "--out", str(tmp_path / "ids.txt"),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "non-negative" in proc.stderr


def test_select_tasks_rejects_negative_start(tmp_path: Path):
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"instance_id": "t0"}) + "\n")
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPTS_DIR / "select_tasks.py"),
            "--data", str(data),
            "--start", "-5",
            "--limit", "1",
            "--out", str(tmp_path / "ids.txt"),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "non-negative" in proc.stderr



def test_select_tasks_no_jsonl_when_flag_omitted(tmp_path: Path):
    data = tmp_path / "tasks.jsonl"
    data.write_text(json.dumps({"instance_id": "only"}) + "\n")
    out_ids = tmp_path / "ids.txt"
    _run_select_tasks(
        "--data", str(data),
        "--limit", "1",
        "--out", str(out_ids),
    )
    # No --out-jsonl path provided → nothing extra written.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ids.txt", "tasks.jsonl"]
