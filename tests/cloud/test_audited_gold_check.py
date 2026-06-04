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


# ---------------------------------------------------------------------------
# DEV-1510: `missing_audited_gold_ids` learns a `benchmark` kwarg so the
# submit-time guard supports BOTH layouts symmetrically. Pre-fix the cloud CLI
# skipped this check entirely for `gold_required` benchmarks (livesqlbench)
# because the per-db sidecar didn't exist. After this change:
#   * mini-interact (per_db layout): existing behavior unchanged.
#   * livesqlbench (single_file layout): reads
#     `<audited_root>/livesqlbench_audited.jsonl`, looks up by instance_id,
#     enforces the same `clean passes` / `edited+empty sql fails` semantics
#     as per_db, and (defensively) treats a row whose `selected_database`
#     doesn't match the task's as missing-row.
# All tests below mirror the per_db tests above; the only delta is the
# audit-file layout and the dataset shape.
# ---------------------------------------------------------------------------


def _write_lsb_dataset(data_path: Path, rows: list[dict]) -> None:
    """Mirror of `_write_dataset` but for livesqlbench's data file (same
    JSON-lines shape; only the filename and the field set the dataset's
    instance_db_map loader keys off differ)."""
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_lsb_audited(audited_root: Path, rows: list[dict]) -> Path:
    """Lay down `<audited_root>/livesqlbench_audited.jsonl` for the
    single_file layout.

    Injects ``benchmark: "livesqlbench-base-lite-sqlite"`` into any row that doesn't
    already set it, so existing test cases (which only set
    ``selected_database``) keep passing the new benchmark-field guard.
    Tests that DELIBERATELY omit ``benchmark`` to exercise the guard
    should set ``benchmark`` to ``None`` (which the helper preserves)
    or set an explicit wrong value.
    """
    audited_root.mkdir(parents=True, exist_ok=True)
    sidecar = audited_root / "livesqlbench-base-lite-sqlite_audited.jsonl"
    normalised = []
    for r in rows:
        if "benchmark" not in r:
            r = {**r, "benchmark": "livesqlbench-base-lite-sqlite"}
        normalised.append(r)
    sidecar.write_text("\n".join(json.dumps(r) for r in normalised) + "\n")
    return sidecar


def test_lsb_clean_row_passes(tmp_path: Path) -> None:
    """Same `clean passes regardless of audited_sol_sql` semantics as the
    per_db case — the overlay deliberately leaves `sol_sql` untouched
    for clean rows, so missing `audited_sol_sql` is benign."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_9", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_9", "selected_database": "museum",
         "audit_status": "clean"},
    ])

    missing = missing_audited_gold_ids(
        ["museum_9"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == []


def test_lsb_edited_with_audited_sol_sql_passes(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_7", "selected_database": "museum",
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == []


def test_lsb_edited_missing_sol_sql_is_reported(tmp_path: Path) -> None:
    """Same silent-fallback risk as per_db: a row with `audit_status=edited`
    but no `audited_sol_sql` would silently keep the original gold."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_7", "selected_database": "museum",
         "audit_status": "edited"},  # no audited_sol_sql
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"]


def test_lsb_missing_row_is_reported(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_1", "selected_database": "museum"},
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_1", "selected_database": "museum",
         "audit_status": "clean"},
        # museum_7 not in the audit file.
    ])

    missing = missing_audited_gold_ids(
        ["museum_1", "museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"]


def test_lsb_missing_file_reports_every_id(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_1", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    # No livesqlbench_audited.jsonl at all.

    missing = missing_audited_gold_ids(
        ["museum_1"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_1"]


def test_lsb_row_db_mismatch_is_reported(tmp_path: Path) -> None:
    """Defensive: a single_file audit row whose `selected_database`
    differs from the dataset's mapping for that instance_id is corrupt
    (cross-benchmark id collision). The guard must NOT trust it; it
    reports the id as missing so the submitter sees the discrepancy
    instead of silently applying the wrong audit."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_7", "selected_database": "WRONG",
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"]


def test_lsb_row_with_no_selected_database_is_reported(tmp_path: Path) -> None:
    """Codex DEV-1510 review follow-up: a row that's missing
    `selected_database` entirely must also be reported. The submit-time
    guard's cross-benchmark check has to mirror the overlay's — if the
    overlay rejects a row with no discriminator (and falls back to the
    original gold), the submit-time guard must too, otherwise the
    cluster comes up + the actor silently falls back mid-run."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        # NB: no `selected_database` field at all.
        {"instance_id": "museum_7",
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"], (
        "row with no selected_database must be reported; the overlay "
        "rejects it as missing-row, so the submit-time guard has to too"
    )


def test_lsb_row_with_empty_selected_database_is_reported(tmp_path: Path) -> None:
    """Same guard, with `selected_database` present but empty-string —
    the guard must reject both None and empty-string the same way."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_7", "selected_database": "",
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"]


def test_lsb_row_with_wrong_benchmark_is_reported(tmp_path: Path) -> None:
    """Codex post-review-fix follow-up: DB names overlap across benchmarks
    by design (alien, museum, ...). A row with the right (instance_id,
    selected_database) but the wrong `benchmark` tag would slip past the
    selected_database check. The submit-time guard must mirror the
    overlay's check on the `benchmark` field too."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_7", "selected_database": "museum",
         "benchmark": "mini-interact",  # wrong benchmark
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"]


def test_lsb_row_with_no_benchmark_field_is_reported(tmp_path: Path) -> None:
    """Same guard, with `benchmark` absent — the schema requires it, so
    missing is treated the same as mismatch. Bypasses the helper's
    default-injection by setting `benchmark=None` explicitly (which the
    helper preserves)."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        # benchmark=None — JSON-serialises as null, which the loader's
        # `row.get("benchmark") or ""` coerces to "" so the guard fires.
        {"instance_id": "museum_7", "selected_database": "museum",
         "benchmark": None,
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
    ])

    missing = missing_audited_gold_ids(
        ["museum_7"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7"]


def test_benchmark_kwarg_default_preserves_per_db_behavior(tmp_path: Path) -> None:
    """Same back-compat posture as `apply_audited_gold_overlay`: existing
    `missing_audited_gold_ids(..., data_path=mini_interact_file)` callers
    (the cloud CLI's mini-interact path) keep working without the kwarg."""
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

    # NO `benchmark=` kwarg — defaults to per_db, mini-interact.
    missing = missing_audited_gold_ids(
        ["alien_1"],
        audited_root=audited_root, data_path=data,
    )
    assert missing == []


def test_lsb_partial_batch_reports_only_failing_ids(tmp_path: Path) -> None:
    """End-to-end batch sanity: a partial fix lands museum_1/2/9 in the
    single-file audit; museum_7's row was malformed (status=edited but
    no audited_sol_sql) and museum_10 was never added. Guard reports
    only the two that would silently fall back."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        missing_audited_gold_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_lsb_dataset(data, [
        {"instance_id": f"museum_{i}", "selected_database": "museum"}
        for i in (1, 2, 7, 9, 10)
    ])
    audited_root = tmp_path / "audited_gold"
    _write_lsb_audited(audited_root, [
        {"instance_id": "museum_1", "selected_database": "museum",
         "audit_status": "clean"},
        {"instance_id": "museum_2", "selected_database": "museum",
         "audit_status": "edited", "audited_sol_sql": ["SELECT 1"]},
        # museum_7 row malformed: edited + no audited_sol_sql.
        {"instance_id": "museum_7", "selected_database": "museum",
         "audit_status": "edited"},
        {"instance_id": "museum_9", "selected_database": "museum",
         "audit_status": "clean"},
        # museum_10 not in the file.
    ])

    missing = missing_audited_gold_ids(
        ["museum_1", "museum_2", "museum_7", "museum_9", "museum_10"],
        audited_root=audited_root, data_path=data,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
    )
    assert missing == ["museum_7", "museum_10"]
