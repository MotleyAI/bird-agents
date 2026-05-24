"""Verifier accepts the SAR audit set and the `ambiguous` status."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPO_ROOT / "scripts" / "verify_audited_gold.py"


def _make_sar_row(instance_id, audit_status, sql, *, changes=None, sample_row=None):
    return {
        "instance_id": instance_id,
        "selected_database": "fake",
        "audit_status": audit_status,
        "original_sol_sql": [sql],
        "audited_sol_sql": [sql],
        "audited_sample_row": sample_row,
        "audited_sample_row_status": "ok" if sample_row is not None else "empty",
        "audited_sample_row_error": None,
        "changes": changes if changes is not None else [],
        "reasoning_summary": "x",
        "skill_version": "sar-agent/1.0",
        "audited_at": "2026-05-21T12:00:00+00:00",
        "sar_correctness_flag": audit_status == "clean",
        "sar_ambiguity_flag": audit_status == "ambiguous",
        "revised_question": None,
        "step_count": 1,
        "cost_usd": 0.0,
        "audit_model_requested": "claude-opus-4-7",
        "audit_model_actual": "claude-opus-4-7-20260121",
        "raw_trajectory": None,
    }


def test_valid_status_includes_ambiguous():
    """Code-level constant check — independent of the CLI."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_audited_gold", VERIFIER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "ambiguous" in mod.VALID_STATUS


def test_verifier_accepts_audit_set_flag():
    """`--audit-set sar` is a known option."""
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--audit-set" in result.stdout
    assert "inhouse" in result.stdout
    assert "sar" in result.stdout


def test_sar_ambiguous_row_with_change_passes(tmp_path: Path, monkeypatch):
    """Synthetic ambiguous SAR row with one sar_ambiguous change validates."""
    _stage_sar_db(tmp_path, "fake_db_ambig")

    sar_dir = tmp_path / "sar_audited_gold" / "fake_db_ambig"
    sar_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _make_sar_row(
            "fake_db_ambig_1",
            "ambiguous",
            "SELECT x FROM t LIMIT 1",
            changes=[
                {
                    "clause_kind": "sar_ambiguous",
                    "source": "sar_agent",
                    "original": "SELECT x FROM t LIMIT 1",
                    "replacement": "",
                    "why_unjustified": "ambiguous question",
                    "justified_by": [],
                }
            ],
            sample_row=[1],
        )
    ]
    (sar_dir / "fake_db_ambig_sar_audited.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )

    result = _run_verifier(tmp_path, "fake_db_ambig", "sar")
    assert result.returncode == 0, result.stderr + result.stdout


def test_sar_unrecoverable_row_with_change_passes(tmp_path: Path):
    """Synthetic unrecoverable SAR row with one sar_unrecoverable change validates."""
    _stage_sar_db(tmp_path, "fake_db_unrec")

    sar_dir = tmp_path / "sar_audited_gold" / "fake_db_unrec"
    sar_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _make_sar_row(
            "fake_db_unrec_1",
            "unrecoverable",
            "SELECT x FROM t LIMIT 1",
            changes=[
                {
                    "clause_kind": "sar_unrecoverable",
                    "source": "sar_agent",
                    "original": "SELECT x FROM t LIMIT 1",
                    "replacement": "",
                    "why_unjustified": "cannot determine intent",
                    "justified_by": [],
                }
            ],
            sample_row=[1],
        )
    ]
    (sar_dir / "fake_db_unrec_sar_audited.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )

    result = _run_verifier(tmp_path, "fake_db_unrec", "sar")
    assert result.returncode == 0, result.stderr + result.stdout


def test_ambiguous_with_empty_changes_fails(tmp_path: Path):
    """Tightened rule: `ambiguous` rows must also carry a synthesized change.
    Mirrors the `edited`/`unrecoverable` contract — no silent ambiguity."""
    _stage_sar_db(tmp_path, "fake_db_ambig_bad")

    sar_dir = tmp_path / "sar_audited_gold" / "fake_db_ambig_bad"
    sar_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _make_sar_row(
            "fake_db_ambig_bad_1",
            "ambiguous",
            "SELECT x FROM t LIMIT 1",
            changes=[],
            sample_row=[1],
        )
    ]
    (sar_dir / "fake_db_ambig_bad_sar_audited.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )

    result = _run_verifier(tmp_path, "fake_db_ambig_bad", "sar")
    assert result.returncode != 0
    assert "requires non-empty changes" in (result.stdout + result.stderr)


def test_unrecoverable_with_empty_changes_still_fails(tmp_path: Path):
    """The pre-existing 'no silent deferrals' rule must remain."""
    _stage_sar_db(tmp_path, "fake_db_bad")

    sar_dir = tmp_path / "sar_audited_gold" / "fake_db_bad"
    sar_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _make_sar_row(
            "fake_db_bad_1",
            "unrecoverable",
            "SELECT x FROM t LIMIT 1",
            changes=[],
            sample_row=[1],
        )
    ]
    (sar_dir / "fake_db_bad_sar_audited.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )

    result = _run_verifier(tmp_path, "fake_db_bad", "sar")
    assert result.returncode != 0
    assert "requires non-empty changes" in (result.stdout + result.stderr)


def test_existing_inhouse_credit_still_validates_clean():
    """Regression guard: real audited_gold/credit/credit_audited.jsonl
    still passes with `--audit-set inhouse`."""
    result = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--db",
            "credit",
            "--audit-set",
            "inhouse",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"verify_audited_gold.py regression on inhouse credit: "
        f"{result.stdout}\n{result.stderr}"
    )


# ---- helpers -----------------------------------------------------------------


def _stage_sar_db(tmp_path: Path, db: str):
    """Create the minimum on-disk artefacts for a synthetic DB so the
    verifier can run against it: kb.jsonl, column_meaning_base.json,
    mini_interact.jsonl, and a sqlite file with table t."""
    mi_root = tmp_path / "mini_interact"
    db_dir = mi_root / db
    db_dir.mkdir(parents=True, exist_ok=True)

    (db_dir / f"{db}_kb.jsonl").write_text(json.dumps({"id": 1, "knowledge": "x"}) + "\n")
    (db_dir / f"{db}_column_meaning_base.json").write_text(json.dumps({"t|x": "col"}))

    (mi_root / "mini_interact.jsonl").write_text(
        json.dumps(
            {
                "instance_id": f"{db}_1",
                "selected_database": db,
                "sol_sql": ["SELECT x FROM t LIMIT 1"],
                "amb_user_query": "x",
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


def _run_verifier(tmp_path: Path, db: str, audit_set: str):
    """Runs the verifier with env vars pointing at the synthetic dirs."""
    env_overrides = {
        "BIRD_DB_PATH": str(tmp_path / "mini_interact"),
        "BIRD_DATA_PATH": str(tmp_path / "mini_interact" / "mini_interact.jsonl"),
        "BIRD_AUDITED_GOLD_ROOT": str(tmp_path / "audited_gold"),
        "BIRD_SAR_AUDITED_GOLD_ROOT": str(tmp_path / "sar_audited_gold"),
    }
    import os

    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--db",
            db,
            "--audit-set",
            audit_set,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
