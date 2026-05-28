"""Tests for the submit-time audited-gold guard (DEV-1478 follow-up + Codex
second-pass review).

The helper at ``cloud._audited_gold_check.missing_audited_gold_ids`` is the
guard behind ``bird-interact-cloud submit --require-audited-gold`` (default
on). It must mirror ``harness.apply_audited_gold_overlay``'s actual
behaviour at task time, not just the row's ``audit_status`` string —
otherwise a corrupt sidecar row silently slips past the submit guard and
falls back to the un-audited gold mid-cloud-run.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_dataset(data_path: Path, rows: list[dict]) -> None:
    data_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )


def _write_audited(audited_root: Path, db: str, rows: list[dict]) -> Path:
    """Lay down ``<audited_root>/<db>/<db>_audited.jsonl`` with the given rows."""
    sidecar = audited_root / db / f"{db}_audited.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return sidecar


def test_clean_row_passes_regardless_of_audited_sol_sql(tmp_path: Path) -> None:
    """``audit_status == 'clean'`` means the original IS the audited gold;
    the overlay deliberately leaves ``sol_sql`` untouched for clean rows.
    The helper must NOT require ``audited_sol_sql`` to be present for
    clean rows."""
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        # No ``audited_sol_sql`` field — clean rows don't need one.
        {"instance_id": "alien_1", "audit_status": "clean"},
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == []


def test_edited_row_with_nonempty_audited_sol_sql_passes(tmp_path: Path) -> None:
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {
            "instance_id": "alien_1",
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT 1"],
        },
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == []


def test_unrecoverable_row_with_nonempty_audited_sol_sql_passes(
    tmp_path: Path,
) -> None:
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {
            "instance_id": "alien_1",
            "audit_status": "unrecoverable",
            "audited_sol_sql": ["-- marker SQL", "SELECT 1"],
        },
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == []


def test_edited_row_missing_audited_sol_sql_is_reported(tmp_path: Path) -> None:
    """Codex DEV-1478 follow-up: a sidecar row with ``audit_status == 'edited'``
    but missing ``audited_sol_sql`` would silently fall back to the
    original un-audited gold in the overlay. The guard must flag it as
    missing at submit time."""
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        # `audited_sol_sql` field absent entirely — overlay won't swap.
        {"instance_id": "alien_1", "audit_status": "edited"},
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_1"]


def test_edited_row_with_empty_audited_sol_sql_is_reported(
    tmp_path: Path,
) -> None:
    """Empty list also fails ``apply_audited_gold_overlay``'s
    ``bool(audited)`` check, so the guard must flag it too."""
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {
            "instance_id": "alien_1",
            "audit_status": "edited",
            "audited_sol_sql": [],
        },
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_1"]


def test_edited_row_with_non_list_audited_sol_sql_is_reported(
    tmp_path: Path,
) -> None:
    """``apply_audited_gold_overlay`` requires ``isinstance(audited, list)``.
    A string-typed value would fall back too — the guard must catch it."""
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {
            "instance_id": "alien_1",
            "audit_status": "edited",
            "audited_sol_sql": "SELECT 1",  # type: ignore[dict-item]
        },
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_1"]


def test_unrecoverable_row_missing_audited_sol_sql_is_reported(
    tmp_path: Path,
) -> None:
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {"instance_id": "alien_1", "audit_status": "unrecoverable"},
    ])

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_1"]


def test_missing_row_in_sidecar_is_reported(tmp_path: Path) -> None:
    """Pre-existing contract preserved: a sidecar that exists but has no
    row for the requested id reports missing."""
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
        {"instance_id": "alien_2", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {"instance_id": "alien_1", "audit_status": "clean"},
    ])

    missing = missing_audited_gold_ids(
        ["alien_1", "alien_2"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_2"]


def test_missing_db_sidecar_file_is_reported(tmp_path: Path) -> None:
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    # No file at audited_root/alien/alien_audited.jsonl.

    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_1"]


def test_unknown_instance_id_is_reported(tmp_path: Path) -> None:
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {"instance_id": "alien_1", "audit_status": "clean"},
    ])

    missing = missing_audited_gold_ids(
        ["alien_9999"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_9999"]


def test_partial_batch_reports_only_failing_ids(tmp_path: Path) -> None:
    """A typical batch — some clean, some edited-with-sql, some
    edited-without-sql — must report only the silently-fallback ones."""
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
        {"instance_id": "alien_2", "selected_database": "alien"},
        {"instance_id": "alien_3", "selected_database": "alien"},
        {"instance_id": "alien_4", "selected_database": "alien"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_audited(audited_root, "alien", [
        {"instance_id": "alien_1", "audit_status": "clean"},
        {
            "instance_id": "alien_2", "audit_status": "edited",
            "audited_sol_sql": ["SELECT 1"],
        },
        # alien_3: status edited but missing sql — silent fallback.
        {"instance_id": "alien_3", "audit_status": "edited"},
        # alien_4: not in sidecar at all.
    ])

    missing = missing_audited_gold_ids(
        ["alien_1", "alien_2", "alien_3", "alien_4"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == ["alien_3", "alien_4"]
