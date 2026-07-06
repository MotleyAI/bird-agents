"""DEV-1648: orchestrator phase wiring for Postgres.

- Phase 2 collects DB-wide PK/join keys and threads backend + key set +
  sampler into ``apply_overlay``.
- Phase 3 now RUNS for postgres (leaf expansion), but skips the
  SQLite-only ``detect_drift``.
- Phase 4 still skips postgres, and on SQLite threads the key guard into
  ``detect_and_apply``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from slayer.core.models import Column, DataType, ModelJoin, SlayerModel

from bird_interact_agents.slayer_pipeline import orchestrator


class FakeStorage:
    def __init__(self, models: list[SlayerModel]) -> None:
        self._m = {m.name: m for m in models}
        self.saved: list[str] = []

    async def list_models(self, data_source=None):
        return list(self._m)

    async def get_model(self, name, data_source=None):
        return self._m.get(name)

    async def save_model(self, model):
        self._m[model.name] = model
        self.saved.append(model.name)


def _pk_model(db: str) -> SlayerModel:
    return SlayerModel(
        name="transplant_matching", sql_table="transplant_matching",
        data_source=db,
        columns=[
            Column(name="match_rec_registry", sql="match_rec_registry",
                   type=DataType.TEXT, primary_key=True),
            Column(name="donor_ref_reg", sql="donor_ref_reg", type=DataType.TEXT),
            Column(name="score_val", sql="score_val", type=DataType.TEXT),
        ],
        joins=[ModelJoin(target_model="demographics",
                         join_pairs=[["donor_ref_reg", "donor_registry"]])],
    )


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------


def test_phase2_threads_backend_and_dbwide_keys(monkeypatch, tmp_path) -> None:
    db = "pgdb"
    storage = FakeStorage([_pk_model(db)])
    calls = []

    def fake_apply(model, by_table, *, backend="sqlite",
                   key_columns=frozenset(), pg_sampler=None):
        calls.append((backend, key_columns, pg_sampler))
        return 0, []

    monkeypatch.setattr(orchestrator, "apply_overlay", fake_apply)
    monkeypatch.setattr(orchestrator, "load_meanings", lambda p: {})
    sampler = lambda t, c: []

    asyncio.run(orchestrator._phase2_overlay(
        storage, db, tmp_path / "m.json", backend="postgres", pg_sampler=sampler,
    ))

    assert calls, "apply_overlay must be called per model"
    backend, keys, passed_sampler = calls[0]
    assert backend == "postgres"
    assert passed_sampler is sampler
    # DB-wide keys: PK + local join side + referenced join side.
    assert ("transplant_matching", "match_rec_registry") in keys
    assert ("transplant_matching", "donor_ref_reg") in keys
    assert ("demographics", "donor_registry") in keys
    assert ("transplant_matching", "score_val") not in keys


# ---------------------------------------------------------------------------
# Phase 3 — now runs for postgres, skips SQLite drift
# ---------------------------------------------------------------------------


def test_phase3_expands_leaves_for_postgres_and_skips_drift(monkeypatch, tmp_path) -> None:
    db = "pgdb"
    model = SlayerModel(
        name="households", sql_table="households", data_source=db,
        columns=[Column(name="socioeconomic", sql="socioeconomic", type=DataType.TEXT)],
    )
    storage = FakeStorage([model])
    b = SimpleNamespace(db_backend="postgres")

    monkeypatch.setattr(
        orchestrator, "_detect_jsonb_columns",
        lambda p: [("households", "socioeconomic",
                    {"fields_meaning": {"Bath_Count": "REAL. baths."}})],
    )
    drift_called = {"hit": False}

    def fake_drift(*a, **k):
        drift_called["hit"] = True
        return set(), set()

    monkeypatch.setattr(orchestrator, "detect_drift", fake_drift)
    monkeypatch.setattr(orchestrator, "SlayerQueryEngine", lambda storage=None: object())

    refresh_calls = {"n": 0}

    async def fake_refresh(**k):
        refresh_calls["n"] += 1
        return []

    monkeypatch.setattr(orchestrator, "refresh_table_backed_model_sampled", fake_refresh)

    sentinel_sampler = lambda t, e: []
    added, typ_warns, drift_warns = asyncio.run(orchestrator._phase3_jsonb(
        storage, db, meanings_path=tmp_path / "m.json", sqlite_path=None,
        benchmark=b, pg_extract_sampler=sentinel_sampler,
    ))

    assert added >= 1  # leaf expansion happened for postgres
    assert drift_called["hit"] is False  # SQLite drift skipped
    assert drift_warns == []
    assert "households" in storage.saved            # model persisted
    assert refresh_calls["n"] >= 1                  # leaves sample-refreshed
    leaf = next(c for c in storage._m["households"].columns
                if c.name == "socioeconomic__Bath_Count")
    # Postgres-native extract, not JSON_EXTRACT.
    assert "jsonb_extract_path_text" in leaf.sql
    assert "JSON_EXTRACT" not in leaf.sql


def test_phase3_threads_extract_sampler_and_backend(monkeypatch, tmp_path) -> None:
    db = "pgdb"
    model = SlayerModel(
        name="households", sql_table="households", data_source=db,
        columns=[Column(name="socioeconomic", sql="socioeconomic", type=DataType.TEXT)],
    )
    storage = FakeStorage([model])
    b = SimpleNamespace(db_backend="postgres")
    sentinel_sampler = lambda t, e: ["2025-01-01 00:00:00"]
    captured = {}

    monkeypatch.setattr(
        orchestrator, "_detect_jsonb_columns",
        lambda p: [("households", "socioeconomic",
                    {"fields_meaning": {"event_ts": "TIMESTAMP. when."}})],
    )

    def fake_expand(json_col, entry, *, backend="sqlite", table="",
                    pg_extract_sampler=None):
        captured["backend"] = backend
        captured["table"] = table
        captured["sampler"] = pg_extract_sampler
        leaf = Column(
            name=f"{json_col}__event_ts",
            sql=f'jsonb_extract_path_text("{json_col}", \'event_ts\')',
            type=DataType.TIMESTAMP,
            meta={"derived_from": {"json_col": json_col, "path": ["event_ts"]}},
        )
        return [leaf], []

    monkeypatch.setattr(orchestrator, "expand_one_column", fake_expand)
    monkeypatch.setattr(orchestrator, "SlayerQueryEngine", lambda storage=None: object())

    async def fake_refresh(**k):
        return []

    monkeypatch.setattr(orchestrator, "refresh_table_backed_model_sampled", fake_refresh)

    asyncio.run(orchestrator._phase3_jsonb(
        storage, db, meanings_path=tmp_path / "m.json", sqlite_path=None,
        benchmark=b, pg_extract_sampler=sentinel_sampler,
    ))
    assert captured["backend"] == "postgres"
    assert captured["table"] == "households"
    assert captured["sampler"] is sentinel_sampler


# ---------------------------------------------------------------------------
# Phase 4 — still skips postgres
# ---------------------------------------------------------------------------


def test_phase4_still_skipped_for_postgres(monkeypatch, tmp_path) -> None:
    b = SimpleNamespace(db_backend="postgres")
    storage = FakeStorage([])

    called = {"hit": False}
    monkeypatch.setattr(orchestrator, "detect_and_apply",
                        lambda *a, **k: called.__setitem__("hit", True) or (0, []))

    retyped, _ = asyncio.run(orchestrator._phase4_dates(
        storage, "pgdb", sqlite_path=None, benchmark=b,
    ))
    assert retyped == 0
    assert called["hit"] is False


def test_phase4_sqlite_threads_key_guard(monkeypatch, tmp_path) -> None:
    db = "sqdb"
    storage = FakeStorage([_pk_model(db)])
    b = SimpleNamespace(db_backend="sqlite")
    captured = {}

    def fake_detect(model, sqlite_path, client, llm_model, *, key_columns=frozenset()):
        captured["keys"] = key_columns
        return 0, []

    monkeypatch.setattr(orchestrator, "detect_and_apply", fake_detect)
    monkeypatch.setattr(orchestrator, "make_anthropic_client", lambda: object())

    asyncio.run(orchestrator._phase4_dates(
        storage, db, sqlite_path=tmp_path / "x.sqlite", llm_model="m", benchmark=b,
    ))
    assert ("transplant_matching", "match_rec_registry") in captured["keys"]
    assert ("transplant_matching", "donor_ref_reg") in captured["keys"]
