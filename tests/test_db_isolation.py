"""DEV-1462 — `harness.materialize_task_db`.

LiveSQLBench-Base-Lite-SQLite ships per-DB `<db>_template.sqlite` and the
upstream eval rm+copies it onto `<db>.sqlite` on every submit. If multiple
tasks share the stable dataset `<db>.sqlite`, concurrent resets race the
OTF cache build that ALSO reads it. Solution: each task gets its own
`db_file_path` in `$TMPDIR/.../<instance_id>/<db>.sqlite`, with the
template symlinked in alongside it — so the upstream reset operates inside
the per-task dir and never touches the stable file.

The marker that gates all of this is `task["dataset"] == "livesqlbench"`
(the loader stamps it). Mini-interact tasks must be a no-op.

These tests drive the REAL upstream `reset_and_restore_database` path so
the contract on `dirname(db_file_path)/<base>_template.sqlite` is verified
end-to-end. They run unsandboxed (sqlite writes).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from tests._livesqlbench_fixtures import make_tiny_sqlite


# All 18 real LiveSQLBench DB names — the upstream reset extracts
# `base_db_name` by splitting on `_ephemeral_`/`_process_`, so any new
# name added here must not contain those tokens (Codex #5).
_REAL_LIVESQLBENCH_DBS = [
    "alien", "archeology", "credit", "cross_db", "crypto", "cybermarket",
    "disaster", "fake", "gaming", "insider", "mental", "museum", "news",
    "polar", "robot", "solar", "vaccine", "virtual",
]


def _make_dataset_dir(root: Path, db: str) -> Path:
    """Build `<root>/<db>/<db>_template.sqlite` with a tiny real sqlite.
    Returns the dataset root."""
    root.mkdir(parents=True, exist_ok=True)
    make_tiny_sqlite(root / db / f"{db}_template.sqlite")
    return root


def test_no_op_for_mini_interact_task(tmp_path):
    """No `dataset` marker (or `dataset != livesqlbench`) → returns None
    AND leaves the task dict untouched."""
    from bird_interact_agents.harness import materialize_task_db

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = {"instance_id": "alien_1", "selected_database": "alien"}
    before = dict(task)
    out = materialize_task_db(task, str(root))
    assert out is None, "must be a no-op for non-livesqlbench tasks"
    assert task == before, (
        f"materialize_task_db must NOT mutate a non-livesqlbench task; "
        f"before={before!r} after={task!r}"
    )


def test_sets_db_file_path_to_per_instance_dir(tmp_path):
    from bird_interact_agents.harness import materialize_task_db

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "dataset": "livesqlbench",
    }
    out = materialize_task_db(task, str(root))
    assert out is not None
    assert task["db_file_path"] == out
    p = Path(out)
    assert p.name == "alien.sqlite"
    # Per-instance dir; the parent dirname is the part the upstream reset
    # uses to find the template.
    assert "alien_1" in p.parent.parts, (
        f"db_file_path must live under a per-instance dir; got {p!r}"
    )


def test_per_instance_dir_carries_template_as_symlink_to_real(tmp_path):
    """The upstream reset reads
    `<dirname(db_file_path)>/<base>_template.sqlite` — so the template
    MUST sit next to the future working sqlite in the per-task dir.

    Plan B0 requires a SYMLINK (not a copy) to avoid duplicating ~2-25 MB
    templates per task across 180 tasks (Codex #4 pinned target-equality
    on the symlink so a stale dir is rebuilt against the current root)."""
    from bird_interact_agents.harness import materialize_task_db

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "dataset": "livesqlbench",
    }
    materialize_task_db(task, str(root))
    template_in_task_dir = Path(task["db_file_path"]).parent / "alien_template.sqlite"
    assert template_in_task_dir.is_symlink(), (
        "the per-task `<db>_template.sqlite` MUST be a SYMLINK (B0); a "
        "real copy would multiply storage by 180 tasks."
    )
    # The symlink's target MUST be the canonical dataset template — the
    # whole point of the stale-symlink rebuild check (Codex #4) is to
    # verify this target equality on reruns.
    resolved = template_in_task_dir.resolve()
    expected = (root / "alien" / "alien_template.sqlite").resolve()
    assert resolved == expected, (
        f"per-task template symlink must point at the dataset template; "
        f"got {resolved!r}, expected {expected!r}"
    )


def test_idempotent(tmp_path):
    """Two calls with the same task MUST be a no-op the second time —
    return the same db_file_path, don't tear down the dir."""
    from bird_interact_agents.harness import materialize_task_db

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "dataset": "livesqlbench",
    }
    out1 = materialize_task_db(task, str(root))
    out2 = materialize_task_db(task, str(root))
    assert out1 == out2


def test_stale_symlink_rebuilds_when_data_path_changes(tmp_path):
    """Codex #4: per-instance dir keyed only by `instance_id` means a
    rerun against a different `--db-path` could reuse a STALE symlink.
    The helper MUST validate the symlink target matches the current
    `data_path_base/<db>/<db>_template.sqlite` and rebuild if not."""
    from bird_interact_agents.harness import materialize_task_db

    root1 = _make_dataset_dir(tmp_path / "v1", "alien")
    root2 = _make_dataset_dir(tmp_path / "v2", "alien")
    # Make the two templates byte-distinct so we can detect which one
    # the per-task dir resolves to.
    (root2 / "alien" / "alien_template.sqlite").write_bytes(
        (root2 / "alien" / "alien_template.sqlite").read_bytes() + b"\x00"
    )
    task = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "dataset": "livesqlbench",
    }
    materialize_task_db(task, str(root1))
    first_template = Path(task["db_file_path"]).parent / "alien_template.sqlite"
    first_bytes = first_template.read_bytes()
    # Drop db_file_path so we exercise the rebuild branch, not just a
    # cached pass-through.
    task.pop("db_file_path", None)
    materialize_task_db(task, str(root2))
    second_template = Path(task["db_file_path"]).parent / "alien_template.sqlite"
    second_bytes = second_template.read_bytes()
    assert second_bytes != first_bytes, (
        "stale-symlink path must rebuild against the current dataset "
        "root, not silently reuse the previous run's template"
    )


def test_refuses_db_name_with_ephemeral_or_process_substring(tmp_path):
    """Upstream reset splits `base_db_name` on `_ephemeral_`/`_process_`
    — a db name containing either substring would mislead the template
    lookup. Refuse defensively (Codex #5)."""
    from bird_interact_agents.harness import materialize_task_db

    for offending in ("foo_ephemeral_bar", "x_process_y"):
        root = _make_dataset_dir(tmp_path / offending, offending)
        task = {
            "instance_id": f"{offending}_1",
            "selected_database": offending,
            "dataset": "livesqlbench",
        }
        with pytest.raises((ValueError, RuntimeError)):
            materialize_task_db(task, str(root))


def test_all_18_real_db_names_pass_defensive_check(tmp_path):
    """All 18 real LiveSQLBench DB names must NOT trip the defensive
    refusal — that would be a self-inflicted block on the real data."""
    from bird_interact_agents.harness import materialize_task_db

    for db in _REAL_LIVESQLBENCH_DBS:
        root = _make_dataset_dir(tmp_path / db, db)
        task = {
            "instance_id": f"{db}_1",
            "selected_database": db,
            "dataset": "livesqlbench",
        }
        materialize_task_db(task, str(root))  # must not raise.


def test_real_upstream_reset_uses_per_task_dir(tmp_path):
    """End-to-end: after `materialize_task_db`, call the real upstream
    `execute_submit_action`. It MUST reset the per-task `<db>.sqlite`
    in the per-instance dir AND leave the stable dataset `<db>.sqlite`
    untouched (Codex #8)."""
    from bird_interact_agents.harness import (
        SampleStatus,
        execute_submit_action,
        materialize_task_db,
    )

    root = _make_dataset_dir(tmp_path / "data", "alien")
    # Also materialise the stable file (post-prepare state); we'll verify
    # the upstream reset DOES NOT touch it.
    stable = root / "alien" / "alien.sqlite"
    make_tiny_sqlite(stable)
    stable_bytes_before = stable.read_bytes()

    task = {
        "instance_id": "alien_x1",
        "selected_database": "alien",
        "dataset": "livesqlbench",
        "sol_sql": ["SELECT id FROM widgets"],
        "category": "Query",
        "conditions": {"decimal": [], "distinct": False, "order": False},
        "test_cases": [],
    }
    materialize_task_db(task, str(root))
    status = SampleStatus(idx=0, original_data=task)
    # The submit path resets `db_file_path`'s dir; we don't care about
    # the eval verdict here — just the FS effects.
    execute_submit_action("SELECT id FROM widgets", status, str(root))

    # The per-task working file must now exist with non-zero bytes.
    task_working = Path(task["db_file_path"])
    assert task_working.is_file()
    assert task_working.stat().st_size > 0

    # The STABLE dataset sqlite must be byte-identical to what we wrote
    # before the submit — the eval reset never touched it.
    assert stable.read_bytes() == stable_bytes_before, (
        "execute_submit_action MUST not rewrite the stable dataset "
        "<db>.sqlite — the per-task isolation is the whole point."
    )


def test_concurrent_tasks_get_disjoint_per_task_dirs(tmp_path):
    """Two tasks on the same DB MUST land in DISJOINT per-task dirs so
    their concurrent resets cannot collide."""
    from bird_interact_agents.harness import materialize_task_db

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task_a = {
        "instance_id": "alien_a",
        "selected_database": "alien",
        "dataset": "livesqlbench",
    }
    task_b = {
        "instance_id": "alien_b",
        "selected_database": "alien",
        "dataset": "livesqlbench",
    }
    materialize_task_db(task_a, str(root))
    materialize_task_db(task_b, str(root))
    a_dir = Path(task_a["db_file_path"]).parent
    b_dir = Path(task_b["db_file_path"]).parent
    assert a_dir != b_dir
    assert a_dir.exists() and b_dir.exists()
