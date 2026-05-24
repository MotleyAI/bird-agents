"""Tests for `scripts/export_slayer_models.py`.

Focus on the memory-export filter: the destination tree must carry
only memories whose ``linked_entities`` resolves under the target
DB, matching the verifier's scope rule.

Smoke-covers the model + datasource path too so regressions there
can't sneak past unnoticed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from slayer.storage.yaml_storage import YAMLStorage

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_slayer_models.py"
spec = importlib.util.spec_from_file_location("export_slayer_models", SCRIPT)
export_slayer_models = importlib.util.module_from_spec(spec)
sys.modules["export_slayer_models"] = export_slayer_models
spec.loader.exec_module(export_slayer_models)


async def _seed_source(src_dir: Path) -> None:
    """Build a source YAMLStorage with two datasources, one model
    per datasource, and three memories that exercise the filter:
    households-only, solar-only, and a cross-linked memory whose
    entities touch both datasources.
    """
    from slayer.core.models import Column, DatasourceConfig, SlayerModel

    src = YAMLStorage(base_dir=str(src_dir))

    await src.save_datasource(DatasourceConfig(name="households", type="sqlite"))
    await src.save_datasource(DatasourceConfig(name="solar", type="sqlite"))

    await src.save_model(
        SlayerModel(
            name="m_h",
            data_source="households",
            sql_table="t",
            columns=[Column(name="c", sql="c", type="INT")],
        )
    )
    await src.save_model(
        SlayerModel(
            name="m_s",
            data_source="solar",
            sql_table="t",
            columns=[Column(name="c", sql="c", type="INT")],
        )
    )

    await src.save_memory(
        learning="KB 12 — keep me: a households memory.",
        entities=["households.m_h.c"],
    )
    await src.save_memory(
        learning="KB 99 — drop me: a solar memory.",
        entities=["solar.m_s.c"],
    )
    await src.save_memory(
        learning=(
            "KB 7 — cross-linked: this memory references both "
            "DBs and should still land in the households tree because "
            "at least one entity starts with 'households.'."
        ),
        entities=["solar.m_s.c", "households.m_h.c"],
    )


def test_export_filters_memories_by_db_prefix(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    dest_root = tmp_path / "out"

    asyncio.run(_seed_source(src_dir))

    monkeypatch.setattr(export_slayer_models, "DEST_ROOT", dest_root)

    rc = asyncio.run(export_slayer_models._export_async("households", str(src_dir)))
    assert rc == 0

    dest_dir = dest_root / "households"
    assert (dest_dir / "datasources" / "households.yaml").is_file()
    assert (dest_dir / "models" / "households" / "m_h.yaml").is_file()
    # Cross-DB datasource and model must not bleed in.
    assert not (dest_dir / "datasources" / "solar.yaml").exists()
    assert not (dest_dir / "models" / "solar").exists()

    dest = YAMLStorage(base_dir=str(dest_dir))
    memories = asyncio.run(dest.list_memories())
    learnings = sorted(m.learning for m in memories)

    # Exactly the two memories that touch households.* — the solar-only
    # memory must be filtered out.
    assert len(memories) == 2
    assert any(l.startswith("KB 12 —") for l in learnings)
    assert any(l.startswith("KB 7 —") for l in learnings)
    assert not any("drop me" in l for l in learnings)


def test_export_with_no_memories_succeeds(tmp_path, monkeypatch):
    """A DB with no in-scope memories still exports cleanly and the
    destination tree simply has no memories.yaml."""
    src_dir = tmp_path / "src"
    dest_root = tmp_path / "out"

    from slayer.core.models import Column, DatasourceConfig, SlayerModel

    src = YAMLStorage(base_dir=str(src_dir))
    asyncio.run(src.save_datasource(DatasourceConfig(name="solar", type="sqlite")))
    asyncio.run(src.save_model(
        SlayerModel(
            name="m_s",
            data_source="solar",
            sql_table="t",
            columns=[Column(name="c", sql="c", type="INT")],
        ),
    ))
    # No memories at all.

    monkeypatch.setattr(export_slayer_models, "DEST_ROOT", dest_root)
    rc = asyncio.run(export_slayer_models._export_async("solar", str(src_dir)))
    assert rc == 0

    dest_dir = dest_root / "solar"
    assert (dest_dir / "datasources" / "solar.yaml").is_file()
    assert (dest_dir / "models" / "solar" / "m_s.yaml").is_file()
    # memories.yaml should be absent (or empty) — verifier handles both.
    mem_file = dest_dir / "memories.yaml"
    if mem_file.exists():
        assert mem_file.read_text().strip() in ("", "[]")


def test_export_is_idempotent(tmp_path, monkeypatch):
    """Re-exporting unchanged source memories produces a bit-identical
    memories.yaml on the destination side. This guards against the
    id/created_at-churn regression: ``save_memory`` would otherwise
    allocate a fresh id and timestamp on every run.
    """
    src_dir = tmp_path / "src"
    dest_root = tmp_path / "out"

    asyncio.run(_seed_source(src_dir))
    monkeypatch.setattr(export_slayer_models, "DEST_ROOT", dest_root)

    rc = asyncio.run(export_slayer_models._export_async("households", str(src_dir)))
    assert rc == 0
    first = (dest_root / "households" / "memories.yaml").read_bytes()

    rc = asyncio.run(export_slayer_models._export_async("households", str(src_dir)))
    assert rc == 0
    second = (dest_root / "households" / "memories.yaml").read_bytes()

    assert first == second, "re-export should be byte-identical"


def test_open_source_accepts_sqlite3_and_uri(tmp_path):
    """``_open_source`` should accept ``.sqlite3`` files, ``sqlite://``
    URIs, and ``yaml://`` URIs in addition to plain directory paths.
    Delegates to SLayer's own resolver, so any backend the CLI accepts
    works here too.
    """
    # All four shapes should resolve without raising; the returned
    # backend should support the methods we need on the read side
    # (list_models / list_memories / get_datasource).
    sqlite3_file = tmp_path / "x.sqlite3"
    sqlite3_file.touch()
    backend = export_slayer_models._open_source(str(sqlite3_file))
    assert hasattr(backend, "list_models")
    assert hasattr(backend, "list_memories")

    sqlite_uri = f"sqlite:///{sqlite3_file}"
    backend = export_slayer_models._open_source(sqlite_uri)
    assert hasattr(backend, "list_models")

    yaml_dir = tmp_path / "yaml_storage"
    yaml_dir.mkdir()
    backend = export_slayer_models._open_source(str(yaml_dir))
    assert hasattr(backend, "list_models")

    yaml_uri = f"yaml://{yaml_dir}"
    backend = export_slayer_models._open_source(yaml_uri)
    assert hasattr(backend, "list_models")
