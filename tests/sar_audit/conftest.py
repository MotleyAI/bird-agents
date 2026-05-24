"""Shared fixtures + stub SARAgent for SAR-audit tests.

The driver, prompt wrapper, and adapter all reach upstream SAR-Agent via
`bird_interact_agents.sar_audit._upstream_import`. Tests monkey-patch the
symbols on that module so the submodule does NOT need to be initialised
for unit tests to pass. The one exception is
`test_submodule_import_smoke.py`, which is skipped when the submodule
is absent.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.sar_audit._stubs import StubSARRunResult, StubSARVerdict  # noqa: F401


class _StubSARAgentState:
    """Module-level state so the stub class behaves like a callable factory.

    Each test resets `queued_results` and `constructed_with` via the
    `stub_sar_agent` fixture.
    """

    queued_results: list[StubSARRunResult] = []
    constructed_with: list[dict] = []
    raise_on_run: Exception | None = None


def _make_stub_sar_agent_factory() -> Callable[..., Any]:
    """Returns a callable conforming to the driver's `sar_agent_factory` shape."""

    class StubSARAgent:
        def __init__(self, **kwargs):
            _StubSARAgentState.constructed_with.append(kwargs)
            self._kwargs = kwargs

        def run(self, *, max_steps: int) -> StubSARRunResult:
            if _StubSARAgentState.raise_on_run is not None:
                raise _StubSARAgentState.raise_on_run
            if not _StubSARAgentState.queued_results:
                raise RuntimeError("StubSARAgent.run: no queued result")
            return _StubSARAgentState.queued_results.pop(0)

    return StubSARAgent


@pytest.fixture
def stub_sar_agent(monkeypatch):
    """Resets stub state, returns the factory + a handle to queue results."""

    _StubSARAgentState.queued_results = []
    _StubSARAgentState.constructed_with = []
    _StubSARAgentState.raise_on_run = None
    factory = _make_stub_sar_agent_factory()

    class Handle:
        @staticmethod
        def queue(result: StubSARRunResult) -> None:
            _StubSARAgentState.queued_results.append(result)

        @staticmethod
        def raise_on_run(exc: Exception) -> None:
            _StubSARAgentState.raise_on_run = exc

        @property
        def constructed_with(self) -> list[dict]:
            return _StubSARAgentState.constructed_with

    return factory, Handle()


@pytest.fixture
def stub_upstream(monkeypatch):
    """Monkey-patches `_upstream_import.load_prompt_template` with a stub
    template containing the same `{question}/{schema}/{external_knowledge}/
    {gold_query}` placeholders as upstream, so tests don't need the real
    submodule.
    """
    from bird_interact_agents.sar_audit import _upstream_import

    calls: list[dict] = []

    STUB_TEMPLATE = (
        "<<UPSTREAM_BIRD_PROMPT_BEGIN>>\n"
        "Question: {question}\n"
        "Schema: {schema}\n"
        "External Knowledge: {external_knowledge}\n"
        "Annotated Query: {gold_query}\n"
        "<<UPSTREAM_BIRD_PROMPT_END>>"
    )

    def fake_load_template() -> str:
        calls.append({})
        return STUB_TEMPLATE

    monkeypatch.setattr(_upstream_import, "load_prompt_template", fake_load_template)

    class Handle:
        @property
        def calls(self) -> list[dict]:
            return calls

    return Handle()


@pytest.fixture
def fake_db(tmp_path: Path) -> Path:
    """Tiny sqlite DB with table `t(x INTEGER)` containing 1,2,3."""
    db_path = tmp_path / "fake.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()
    return db_path


@pytest.fixture
def fake_empty_db(tmp_path: Path) -> Path:
    """Sqlite DB with table `t(x INTEGER)` and no rows."""
    db_path = tmp_path / "fake_empty.sqlite"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.commit()
    con.close()
    return db_path


@pytest.fixture
def fake_task() -> dict:
    """Minimal mini-interact task shape the driver consumes."""
    return {
        "instance_id": "fake_1",
        "selected_database": "fake",
        "sol_sql": ["SELECT x FROM t ORDER BY x LIMIT 1"],
        "amb_user_query": "Show me the smallest x.",
        "external_knowledge": [2],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {"term": "smallest", "sql_snippet": "ORDER BY x ASC LIMIT 1"}
            ],
            "non_critical_ambiguity": [
                {"term": "show me", "sql_snippet": "SELECT x"}
            ],
        },
        "knowledge_ambiguity": [{"term": "x value"}],
    }


@pytest.fixture
def fake_kb() -> list[dict]:
    return [
        {"id": 1, "knowledge": "KB ONE: rows are integers."},
        {"id": 2, "knowledge": "KB TWO: smallest means ORDER BY ASC LIMIT 1."},
        {"id": 3, "knowledge": "KB THREE: do not return NULL."},
    ]


@pytest.fixture
def fake_column_meanings() -> dict:
    return {
        "t|x": "the integer payload column",
        "t|extra": {
            "column_meaning": "JSON blob",
            "fields_meaning": {"k": "the key field"},
        },
    }


@pytest.fixture
def write_jsonl():
    def _write(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return _write


@pytest.fixture
def read_jsonl():
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    return _read


@pytest.fixture
def fixed_now(monkeypatch):
    """Pins the audited_at timestamp injected by the driver/normalizer."""
    from bird_interact_agents.sar_audit import normalizer

    monkeypatch.setattr(normalizer, "_now_iso", lambda: "2026-05-21T12:00:00+00:00")
    return "2026-05-21T12:00:00+00:00"
