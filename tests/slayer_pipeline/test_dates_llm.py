"""LLM-mocked tests for phase 4 — TEXT-as-date detection.

The Anthropic client is stubbed end-to-end (no network). Covers:
- Confident, consistent response retypes the column to TIMESTAMP,
  rewrites Column.sql for non-ISO formats, and caches
  ``meta.date_source_format`` + ``meta.detected_by='llm'``.
- Low-confidence response leaves the column TEXT and emits a warning.
- LLM-proposed format that fails the strptime gate leaves the column
  TEXT and emits a warning.
- Columns that already carry ``meta.date_source_format`` are skipped
  (idempotency on re-run).
- JSONB-derived columns (carrying ``meta.derived_from``) are skipped
  in this pass.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from slayer.core.models import Column, DataType, SlayerModel

from bird_interact_agents.slayer_pipeline.dates import detect_and_apply


class _FakeBlock:
    """Stands in for `anthropic.types.TextBlock`."""

    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeAnthropic:
    """Minimal stand-in for `anthropic.Anthropic` — covers only `.messages.create`."""

    def __init__(self, response_text: str) -> None:
        self.messages = _FakeMessages(response_text)


# Monkey-patch `isinstance(block, TextBlock)` so the fake block passes the
# real type check inside `detect_and_apply`.
@pytest.fixture(autouse=True)
def _patch_textblock(monkeypatch: pytest.MonkeyPatch) -> None:
    from bird_interact_agents.slayer_pipeline import dates as dates_module

    monkeypatch.setattr(dates_module, "TextBlock", _FakeBlock)


def _make_sqlite(tmp_path: Path, table: str, col: str, values: list[str]) -> Path:
    db = tmp_path / "fake.sqlite"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(f'CREATE TABLE "{table}" ("{col}" TEXT)')
        conn.executemany(
            f'INSERT INTO "{table}" ("{col}") VALUES (?)',
            [(v,) for v in values],
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _model_with(col: Column) -> SlayerModel:
    return SlayerModel(
        name="orders",
        sql_table="orders",
        data_source="demo",
        columns=[col],
    )


def test_confident_iso_response_retypes_passthrough(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(
        tmp_path, "orders", "created_at",
        ["2025-01-01", "2025-02-15", "2025-03-30", "2025-12-31"],
    )
    col = Column(name="created_at", sql="created_at", type=DataType.TEXT)
    model = _model_with(col)
    client = _FakeAnthropic(
        json.dumps({"is_date": True, "source_format": "%Y-%m-%d", "confidence": 0.95})
    )
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 1
    assert warns == []
    assert col.type == DataType.TIMESTAMP
    assert col.sql == "created_at"
    assert col.meta["date_source_format"] == "%Y-%m-%d"
    assert col.meta["detected_by"] == "llm"


def test_confident_dd_mm_yyyy_rewrites_sql(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(
        tmp_path, "orders", "beginday",
        ["31/12/2024", "01/01/2025", "15/06/2025"],
    )
    col = Column(name="beginday", sql="beginday", type=DataType.TEXT)
    model = _model_with(col)
    client = _FakeAnthropic(
        json.dumps({"is_date": True, "source_format": "%d/%m/%Y", "confidence": 0.9})
    )
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 1
    assert warns == []
    assert col.type == DataType.TIMESTAMP
    assert "SUBSTR(beginday, 7, 4)" in col.sql


def test_low_confidence_leaves_text_and_warns(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(
        tmp_path, "orders", "memo",
        ["maybe a date", "or not", "12/13/14"],
    )
    col = Column(name="memo", sql="memo", type=DataType.TEXT)
    model = _model_with(col)
    client = _FakeAnthropic(
        json.dumps({"is_date": True, "source_format": None, "confidence": 0.3})
    )
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 0
    assert col.type == DataType.TEXT
    assert len(warns) == 1


def test_strptime_gate_rejects_wrong_format(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(
        tmp_path, "orders", "d",
        ["2024-01-01", "2024-02-29", "2025-12-31"],
    )
    col = Column(name="d", sql="d", type=DataType.TEXT)
    model = _model_with(col)
    # Model lies: says %d/%m/%Y but the data is %Y-%m-%d.
    client = _FakeAnthropic(
        json.dumps({"is_date": True, "source_format": "%d/%m/%Y", "confidence": 0.99})
    )
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 0
    assert col.type == DataType.TEXT
    assert any("failed strptime" in w for w in warns)


def test_skip_when_already_typed(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(tmp_path, "orders", "d", ["2024-01-01"])
    col = Column(
        name="d", sql="d", type=DataType.TEXT,
        meta={"date_source_format": "%Y-%m-%d", "detected_by": "llm"},
    )
    model = _model_with(col)
    client = _FakeAnthropic(json.dumps({"is_date": True, "source_format": "%d/%m/%Y", "confidence": 0.9}))
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 0
    assert warns == []
    # The fake client was never called.
    assert client.messages.calls == []


def test_skip_jsonb_derived_leaves(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(tmp_path, "orders", "json_blob", ["{}"])
    col = Column(
        name="json_blob__when",
        sql="JSON_EXTRACT(json_blob, '$.when')",
        type=DataType.TEXT,
        meta={"derived_from": {"json_col": "json_blob", "path": ["when"]}},
    )
    model = _model_with(col)
    client = _FakeAnthropic(json.dumps({"is_date": True}))
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 0
    assert warns == []
    assert client.messages.calls == []


def test_unsupported_format_leaves_text_and_warns(tmp_path: Path) -> None:
    """If the LLM proposes a format that passes strptime against the
    samples but our SQLite rewrite table can't translate, leave the
    column TEXT — do NOT cache it as TIMESTAMP under a non-functional
    Column.sql."""
    sqlite_path = _make_sqlite(
        tmp_path, "orders", "d",
        ["31.12.2024", "01.01.2025", "15.06.2025"],
    )
    col = Column(name="d", sql="d", type=DataType.TEXT)
    model = _model_with(col)
    client = _FakeAnthropic(
        json.dumps({"is_date": True, "source_format": "%d.%m.%Y", "confidence": 0.99})
    )
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 0
    assert col.type == DataType.TEXT
    assert (col.meta or {}).get("date_source_format") is None
    assert any("not supported by the SQLite rewrite path" in w for w in warns)


def test_is_date_false_leaves_text(tmp_path: Path) -> None:
    sqlite_path = _make_sqlite(tmp_path, "orders", "name", ["alice", "bob", "carol"])
    col = Column(name="name", sql="name", type=DataType.TEXT)
    model = _model_with(col)
    client = _FakeAnthropic(json.dumps({"is_date": False}))
    retyped, warns = detect_and_apply(model, sqlite_path, client, "claude-haiku-4-5")
    assert retyped == 0
    assert warns == []
    assert col.type == DataType.TEXT
