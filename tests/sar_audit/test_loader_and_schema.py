"""High-level driver entry point (`run_db_by_name`) + schema renderer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from bird_interact_agents.sar_audit import driver, loader, schema_renderer
from tests.sar_audit._stubs import StubSARRunResult, StubSARVerdict


def _stage(root: Path, db: str = "fake_load"):
    """Stage one task + KB + column meanings + sqlite at canonical paths."""
    db_dir = root / db
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / f"{db}_kb.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"id": 1, "knowledge": "KB one entry"},
                {"id": 2, "knowledge": "KB two entry"},
            ]
        )
        + "\n"
    )
    (db_dir / f"{db}_column_meaning_base.json").write_text(
        json.dumps(
            {
                "t|x": "the x column",
                "t|y": {
                    "column_meaning": "JSON payload",
                    "fields_meaning": {"k": "sub key"},
                },
            }
        )
    )

    sqlite_path = db_dir / f"{db}.sqlite"
    con = sqlite3.connect(sqlite_path)
    con.execute("CREATE TABLE t (x INTEGER, y TEXT)")
    con.execute("INSERT INTO t VALUES (1, 'first')")
    con.execute("INSERT INTO t VALUES (2, 'second')")
    con.commit()
    con.close()

    tasks = [
        {
            "instance_id": f"{db}_1",
            "selected_database": db,
            "sol_sql": ["SELECT x FROM t ORDER BY x LIMIT 1"],
            "amb_user_query": "smallest x",
            "external_knowledge": [1],
            "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
            "knowledge_ambiguity": [],
        }
    ]
    (root / "mini_interact.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks) + "\n"
    )
    return db


# --- loader ------------------------------------------------------------------


def test_load_task_list_filters_by_db(tmp_path: Path):
    db = _stage(tmp_path)
    # Also write a task from a different db so the loader must filter.
    other_task = {
        "instance_id": "other_1",
        "selected_database": "other",
        "sol_sql": ["SELECT 1"],
        "amb_user_query": "x",
        "external_knowledge": [],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
        "knowledge_ambiguity": [],
    }
    mi_path = tmp_path / "mini_interact.jsonl"
    with mi_path.open("a") as f:
        f.write(json.dumps(other_task) + "\n")

    tasks = loader.load_task_list(db=db, mini_interact_path=mi_path)
    assert len(tasks) == 1
    assert tasks[0]["instance_id"] == f"{db}_1"


def test_load_kb(tmp_path: Path):
    db = _stage(tmp_path)
    kb = loader.load_kb(db=db, mini_interact_root=tmp_path)
    assert len(kb) == 2
    assert kb[0]["id"] == 1
    assert kb[1]["knowledge"] == "KB two entry"


def test_load_column_meanings(tmp_path: Path):
    db = _stage(tmp_path)
    cm = loader.load_column_meanings(db=db, mini_interact_root=tmp_path)
    assert "t|x" in cm
    assert cm["t|x"] == "the x column"
    assert "t|y" in cm


def test_locate_db_sqlite(tmp_path: Path):
    db = _stage(tmp_path)
    path = loader.locate_db_sqlite(db=db, mini_interact_root=tmp_path)
    assert path.exists()
    assert path.name == f"{db}.sqlite"


# --- schema renderer ---------------------------------------------------------


def test_schema_renderer_emits_ddl(tmp_path: Path):
    db = _stage(tmp_path)
    db_path = loader.locate_db_sqlite(db=db, mini_interact_root=tmp_path)
    cm = loader.load_column_meanings(db=db, mini_interact_root=tmp_path)
    out = schema_renderer.render_schema(db_path=db_path, column_meanings=cm)
    assert "CREATE TABLE" in out
    assert "t" in out
    assert "x" in out and "y" in out
    assert "INTEGER" in out
    assert "TEXT" in out


def test_schema_renderer_includes_column_meanings(tmp_path: Path):
    db = _stage(tmp_path)
    db_path = loader.locate_db_sqlite(db=db, mini_interact_root=tmp_path)
    cm = loader.load_column_meanings(db=db, mini_interact_root=tmp_path)
    out = schema_renderer.render_schema(db_path=db_path, column_meanings=cm)
    assert "the x column" in out


def test_schema_renderer_handles_jsonb_subfields(tmp_path: Path):
    db = _stage(tmp_path)
    db_path = loader.locate_db_sqlite(db=db, mini_interact_root=tmp_path)
    cm = loader.load_column_meanings(db=db, mini_interact_root=tmp_path)
    out = schema_renderer.render_schema(db_path=db_path, column_meanings=cm)
    # Nested fields_meaning sub-keys appear in the rendered schema string.
    assert "JSON payload" in out
    assert "sub key" in out


# --- high-level run_db_by_name -----------------------------------------------


def test_run_db_by_name_loads_and_writes_canonical_path(
    tmp_path, stub_sar_agent, stub_upstream, fixed_now, read_jsonl
):
    """End-to-end: stage files at canonical paths, call `run_db_by_name`,
    assert it loads everything and writes to
    `sar_audited_gold_root/<db>/<db>_sar_audited.jsonl`."""
    db = _stage(tmp_path)
    factory, handle = stub_sar_agent

    handle.queue(
        StubSARRunResult(
            verdict=StubSARVerdict(correctness_flag=True, ambiguity_flag=False),
            audit_model_actual="claude-opus-4-7-20260121",
        )
    )

    sar_root = tmp_path / "sar_audited_gold"
    result = driver.run_db_by_name(
        db=db,
        mini_interact_root=tmp_path,
        sar_audited_gold_root=sar_root,
        audit_model="claude-opus-4-7",
        max_steps=5,
        sar_agent_factory=factory,
    )

    output_path = sar_root / db / f"{db}_sar_audited.jsonl"
    assert output_path.exists(), f"expected canonical path {output_path} to exist"

    rows = read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["instance_id"] == f"{db}_1"
    assert rows[0]["audit_status"] == "clean"
    assert rows[0]["selected_database"] == db
    assert result.audited == 1
    assert result.failed == 0
