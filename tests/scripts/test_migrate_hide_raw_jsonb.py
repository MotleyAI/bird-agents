"""DEV-1672 review: the migration CLI's archive sweep must be scoped to the
requested benchmark's ``runs/<benchmark>/`` subtree — NOT walk all of
``runs/`` (which would rewrite other benchmarks' archives and, worse, the
``runs/_edited_models_backups/`` copies). Pins the Codex round-3 finding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_hide_raw_jsonb.py"
_spec = importlib.util.spec_from_file_location("migrate_hide_raw_jsonb", SCRIPT)
migrate = importlib.util.module_from_spec(_spec)
sys.modules["migrate_hide_raw_jsonb"] = migrate
_spec.loader.exec_module(migrate)


def _touch_archive(runs: Path, *rel: str) -> Path:
    d = runs.joinpath(*rel)
    d.mkdir(parents=True, exist_ok=True)
    arc = d / "edited_models.tar.gz"
    arc.write_bytes(b"")
    return arc


async def test_sweep_archives_scopes_to_benchmark(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    want = _touch_archive(runs, "livesqlbench-large", "sports_events_large", "sports_events_1")
    _touch_archive(runs, "mini-interact", "alien", "alien_1")  # other benchmark
    _touch_archive(runs, "_edited_models_backups", "20260713", "livesqlbench-large",
                   "sports_events_large", "sports_events_1")  # a backup copy

    called: list[Path] = []

    async def fake_hide_archive(p):
        called.append(Path(p))
        return 0

    monkeypatch.setattr(migrate, "hide_archive", fake_hide_archive)

    await migrate._sweep_archives("livesqlbench-large", None)

    assert called == [want], "must touch ONLY the scoped benchmark's archive"


async def test_sweep_archives_honours_instance_id_filter(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(runs))
    keep = _touch_archive(runs, "livesqlbench-large", "sports_events_large", "sports_events_1")
    _touch_archive(runs, "livesqlbench-large", "polar_equipment_large", "polar_equipment_12")

    called: list[Path] = []

    async def fake_hide_archive(p):
        called.append(Path(p))
        return 0

    monkeypatch.setattr(migrate, "hide_archive", fake_hide_archive)

    await migrate._sweep_archives("livesqlbench-large", {"sports_events_1"})

    assert called == [keep]
