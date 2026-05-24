"""`scripts/run_sar_audit.py` CLI: exit code on success vs failure."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_SCRIPT = REPO_ROOT / "scripts" / "run_sar_audit.py"


def _stage_minimal_db(root: Path, db: str = "fake_cli") -> None:
    """Stage one mini-interact task + KB + column_meanings + sqlite under root."""
    db_dir = root / db
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / f"{db}_kb.jsonl").write_text(json.dumps({"id": 1, "knowledge": "x"}) + "\n")
    (db_dir / f"{db}_column_meaning_base.json").write_text(json.dumps({"t|x": "col"}))
    (root / "mini_interact.jsonl").write_text(
        json.dumps(
            {
                "instance_id": f"{db}_1",
                "selected_database": db,
                "sol_sql": ["SELECT x FROM t ORDER BY x LIMIT 1"],
                "amb_user_query": "smallest",
                "external_knowledge": [],
                "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
                "knowledge_ambiguity": [],
            }
        )
        + "\n"
    )
    sqlite_path = db_dir / f"{db}.sqlite"
    con = sqlite3.connect(sqlite_path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()


def _cli_env(tmp_path: Path) -> dict:
    """Env vars that point the CLI at our synthetic mini-interact tree, and
    flip on the always-fail injection knob used by the test stubs."""
    return {
        **os.environ,
        "BIRD_DB_PATH": str(tmp_path / "mini_interact"),
        "BIRD_DATA_PATH": str(tmp_path / "mini_interact" / "mini_interact.jsonl"),
        "BIRD_SAR_AUDITED_GOLD_ROOT": str(tmp_path / "sar_audited_gold"),
        # When set, the CLI substitutes a stub SAR factory that always raises.
        "SAR_AUDIT_FORCE_FAILURE": "1",
    }


def test_cli_returns_nonzero_when_any_task_fails(tmp_path: Path):
    _stage_minimal_db(tmp_path / "mini_interact", db="fake_cli")
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--db", "fake_cli"],
        capture_output=True,
        text=True,
        env=_cli_env(tmp_path),
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode != 0, (
        f"CLI should exit non-zero on failure, got 0.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # CLI warned that the test-hook mode is active so accidental production
    # runs don't silently produce synthetic artifacts.
    assert "SAR_AUDIT_FORCE_FAILURE" in result.stderr


def test_cli_returns_zero_on_clean_success(tmp_path: Path):
    _stage_minimal_db(tmp_path / "mini_interact", db="fake_cli")
    env = _cli_env(tmp_path)
    # The clean-success knob: stub returns one clean verdict per task.
    env.pop("SAR_AUDIT_FORCE_FAILURE", None)
    env["SAR_AUDIT_FORCE_CLEAN"] = "1"
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--db", "fake_cli"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"CLI should exit 0 on clean success, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # The CLI surfaces the test-hook mode loudly on stderr so accidental
    # production runs with the env var set don't silently produce
    # synthetic artifacts.
    assert "SAR_AUDIT_FORCE_CLEAN" in result.stderr

    # Output JSONL exists at the canonical path.
    output_jsonl = tmp_path / "sar_audited_gold" / "fake_cli" / "fake_cli_sar_audited.jsonl"
    assert output_jsonl.exists()
