"""DEV-1509 — `harness.materialize_task_db` pre-copies the working SQLite.

LiveSQLBench tasks set ``task["db_file_path"]`` to a per-task path that the
upstream evaluator populates on the first ``reset_and_restore_database``.
Before that first reset, anything that opens the path read-only sees either
a missing file (no-op) or an EMPTY SQLite (upstream's RW
``sqlite3.connect`` creates the file if missing). The dry-run gate in
``agents/_submit.py::_dry_run_sql`` opens the path with
``sqlite3.connect("file:<db_file_path>?mode=ro", uri=True)`` and, against
an empty file, returns ``OperationalError: no such table: <name>`` for
every referenced table — surfacing as the casing rabbit hole described in
the Linear issue (the agent rationalises the spurious error as a
casing mismatch).

Fix: ``materialize_task_db`` pre-copies the canonical template to the
working ``<db>.sqlite`` (atomically, via tmp + ``os.replace``). The
template entry in the per-task dir remains a SYMLINK — copying templates
per task would multiply storage by 180; copying the WORKING file is one
SQLite per task instance, bounded.

These tests pin the new invariant, the strengthened idempotence guards
(SQLite-magic header), the LFS-pointer rejection, the atomic-rename
implementation pattern, and the documentation-via-code that slayer
ingest preserves table casing and SQLite identifier lookup is
case-insensitive (rebuts the Linear issue's casing framing).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from tests._livesqlbench_fixtures import make_lfs_pointer, make_tiny_sqlite


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _make_dataset_dir(root: Path, db: str) -> Path:
    """Build ``<root>/<db>/<db>_template.sqlite`` with a tiny real sqlite
    (one table ``widgets``). Returns the dataset root."""
    root.mkdir(parents=True, exist_ok=True)
    make_tiny_sqlite(root / db / f"{db}_template.sqlite")
    return root


def _make_task(instance_id: str, db: str) -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": db,
        "dataset": "livesqlbench-base-lite-sqlite",
    }


def _materialize(task: dict, root: Path) -> Path:
    """Wrapper around ``materialize_task_db`` that returns the working
    DB path as a non-None ``Path`` (collapsing the ``str | None`` return
    into a stricter shape for tests; non-livesqlbench tasks are not
    exercised here)."""
    from bird_interact_agents.harness import materialize_task_db

    out = materialize_task_db(task, str(root))
    assert out is not None, "materialize_task_db must not be a no-op here"
    return Path(out)


# ---------------------------------------------------------------------------
# 1. Core invariant: working file present + populated after materialize
# ---------------------------------------------------------------------------


def test_working_db_file_present_after_materialize_for_livesqlbench(tmp_path):
    """After ``materialize_task_db`` returns, the path it set on
    ``db_file_path`` MUST be a real file with non-zero bytes. Pre-fix it
    is absent (upstream reset hasn't run yet)."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_1", "alien")
    p = _materialize(task, root)
    assert p.is_file(), (
        f"working <db>.sqlite must exist after materialize_task_db; "
        f"got {p!r}"
    )
    assert p.stat().st_size > 0, (
        f"working <db>.sqlite must be non-empty; size={p.stat().st_size}"
    )


def test_working_db_file_byte_identical_to_template_after_materialize(tmp_path):
    """The pre-copy MUST produce a byte-identical copy of the dataset
    template (not a touched file, not a partial copy)."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    template = root / "alien" / "alien_template.sqlite"
    task = _make_task("alien_1", "alien")
    p = _materialize(task, root)
    assert p.read_bytes() == template.read_bytes(), (
        "working <db>.sqlite must be byte-identical to the dataset template"
    )


# ---------------------------------------------------------------------------
# 2-3. The bug: dry-run gate against the materialised path
# ---------------------------------------------------------------------------


def test_dry_run_against_materialized_working_db_finds_real_tables(tmp_path):
    """Regression test for the actual cloud failure. The dry-run gate in
    ``agents/_submit.py::_dry_run_sql`` MUST return ``None`` (success) for
    a valid SQL string against the materialised working DB.

    Strengthened per Codex: assert the working file is present and
    populated BEFORE invoking _dry_run_sql, so a future regression that
    re-removes the pre-copy would not silently false-pass via the
    ``not os.path.exists(db_path)`` early-return in ``_dry_run_sql``.
    """
    from bird_interact_agents.agents._submit import _dry_run_sql

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_dry1", "alien")
    p = _materialize(task, root)

    # Precondition: the precopy actually happened.
    assert p.is_file() and p.stat().st_size > 0, (
        "precondition for the dry-run check: materialize_task_db must "
        "have left a populated working <db>.sqlite (otherwise the "
        "_dry_run_sql None-on-missing-file branch would mask the bug); "
        f"got is_file={p.is_file()} size={p.stat().st_size if p.exists() else 'N/A'}"
    )

    err = _dry_run_sql(
        "SELECT id FROM widgets",
        data_path_base=str(root),
        db_name="alien",
        db_file_path=str(p),
    )
    assert err is None, (
        f"dry-run against the materialised working DB must succeed for "
        f"a valid query; got error={err!r}"
    )


def test_dry_run_against_materialized_working_db_with_mode_ro_works(tmp_path):
    """Narrower companion to test 3 — pins ``?mode=ro`` (the gate's
    actual mode) by hand. Failure mode: an empty SQLite opened ?mode=ro
    raises ``OperationalError: no such table: widgets`` (we verified this
    empirically while diagnosing DEV-1509)."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_dry2", "alien")
    p = _materialize(task, root)

    uri = f"file:{p}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert "widgets" in tables, (
        f"the working <db>.sqlite opened ?mode=ro must expose `widgets` "
        f"in sqlite_master; got {tables!r}"
    )


# ---------------------------------------------------------------------------
# 4. Template stays a symlink (storage-cost guard)
# ---------------------------------------------------------------------------


def test_template_in_per_task_dir_remains_a_symlink(tmp_path):
    """The pre-copy applies to the WORKING <db>.sqlite only; the per-task
    <db>_template.sqlite MUST remain a symlink. Copying templates per
    task would multiply storage by 180 (DEV-1462 Plan B0; existing test
    `test_per_instance_dir_carries_template_as_symlink_to_real` already
    pins this, restated here so an accidental switch to copy is caught
    by the DEV-1509 surface too)."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_sym", "alien")
    p = _materialize(task, root)
    link = p.parent / "alien_template.sqlite"
    assert link.is_symlink(), (
        "per-task <db>_template.sqlite must remain a SYMLINK after the "
        "pre-copy change; a real copy would blow per-template storage"
    )


# ---------------------------------------------------------------------------
# 5-8. Idempotence + rebuild branches around the working-file guard
# ---------------------------------------------------------------------------


def test_idempotent_when_working_file_intact(monkeypatch, tmp_path):
    """A second call with the same task dict + intact working file MUST
    take the fast path and NOT re-copy. Spy on ``harness.shutil.copy2``
    to assert zero invocations on the second call."""
    import bird_interact_agents.harness as harness

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_idem", "alien")
    harness.materialize_task_db(task, str(root))

    calls: list[tuple[str, str]] = []
    real_copy2 = harness.shutil.copy2

    def _spy(src, dst, *a, **kw):
        calls.append((str(src), str(dst)))
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(harness.shutil, "copy2", _spy)
    out2 = harness.materialize_task_db(task, str(root))
    assert out2 == task["db_file_path"]
    assert calls == [], (
        f"second materialize_task_db with intact working file must NOT "
        f"re-copy; got calls={calls!r}"
    )


def test_idempotent_rebuilds_when_working_file_missing(tmp_path):
    """If the working file goes missing between calls (e.g., an external
    cleanup), the second call MUST rebuild it (re-copy template). Pins
    the ``is_file()`` half of the strengthened fast-path guard."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_missing", "alien")
    p = _materialize(task, root)

    p.unlink()
    assert not p.exists()

    p2 = _materialize(task, root)
    assert p2 == p
    assert p2.is_file() and p2.stat().st_size > 0, (
        f"missing working file must be rebuilt by the second call; "
        f"got is_file={p2.is_file()} size={p2.stat().st_size if p2.exists() else 'N/A'}"
    )


def test_idempotent_rebuilds_when_working_file_empty(tmp_path):
    """If the working file gets truncated to 0 bytes (the SPECIFIC failure
    mode observed in the cloud — empty SQLite created by an upstream
    RW connect before reset), the second call MUST rebuild. Pins the
    ``stat().st_size > 0`` half of the guard (the SQLite-magic check
    subsumes this; we test it explicitly because 0-byte is THE observed
    failure mode)."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_empty", "alien")
    p = _materialize(task, root)

    # Truncate to zero bytes (simulates an empty SQLite created by
    # upstream get_db_connection's default-RW sqlite3.connect).
    open(p, "w").close()
    assert p.stat().st_size == 0

    p2 = _materialize(task, root)
    assert p2.stat().st_size > 0, (
        "0-byte working file must be rebuilt; this is the observed "
        "failure mode in DEV-1509"
    )
    template = root / "alien" / "alien_template.sqlite"
    assert p2.read_bytes() == template.read_bytes()


def test_idempotent_rebuilds_when_working_file_is_lfs_pointer(tmp_path):
    """If the working file ends up as a 132-byte git-LFS pointer (e.g.,
    a partial copy or a foreign artifact), the second call MUST rebuild
    it. Pins the SQLite-magic header check (the file is non-empty and
    a simple ``size > 0`` guard would silently accept it, but the
    SQLite header is missing). Codex MEDIUM finding."""
    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_lfs", "alien")
    p = _materialize(task, root)

    # Overwrite with an LFS-pointer-shaped file (real fixture used
    # elsewhere in the suite to test prepare_livesqlbench refusal).
    make_lfs_pointer(p)
    assert p.stat().st_size > 0, "precondition: LFS pointer is non-empty"
    assert not p.read_bytes().startswith(_SQLITE_MAGIC), (
        "precondition: LFS pointer must NOT start with the SQLite magic"
    )

    p2 = _materialize(task, root)
    header = p2.read_bytes()[:16]
    assert header == _SQLITE_MAGIC, (
        f"LFS pointer at working <db>.sqlite must be rebuilt to a real "
        f"SQLite (header `SQLite format 3\\x00`); got header={header!r}"
    )


# ---------------------------------------------------------------------------
# 9. Stale-symlink rebuild path also copies the new template
# ---------------------------------------------------------------------------


def test_stale_symlink_rebuild_also_copies_working(tmp_path):
    """The existing ``test_stale_symlink_rebuilds_when_data_path_changes``
    pins the symlink rebuild against a different ``data_path_base``.
    With the pre-copy in place, the WORKING file must also be re-copied
    from the NEW template (not silently inherited from the previous
    run's template)."""
    root1 = _make_dataset_dir(tmp_path / "v1", "alien")
    root2 = _make_dataset_dir(tmp_path / "v2", "alien")
    # Byte-distinct templates so we can tell which one the rebuild used.
    (root2 / "alien" / "alien_template.sqlite").write_bytes(
        (root2 / "alien" / "alien_template.sqlite").read_bytes() + b"\x00"
    )

    task = _make_task("alien_stale", "alien")
    p1 = _materialize(task, root1)
    out1_bytes = p1.read_bytes()

    # Drop db_file_path so the rebuild branch runs (matches the existing
    # test's setup).
    task.pop("db_file_path", None)
    p2 = _materialize(task, root2)

    new_template_bytes = (root2 / "alien" / "alien_template.sqlite").read_bytes()
    out2_bytes = p2.read_bytes()
    assert out2_bytes == new_template_bytes, (
        "rebuild against a new --db-path must re-copy the NEW template "
        "to the working file, not silently keep the previous one"
    )
    assert out2_bytes != out1_bytes, (
        "sanity check: the two templates were byte-distinct, so the "
        "working file bytes must have changed"
    )


# ---------------------------------------------------------------------------
# 10. Upstream reset still works (and overwrites a corrupted working DB)
# ---------------------------------------------------------------------------


def test_upstream_reset_still_works_after_pre_copy(tmp_path):
    """End-to-end mirror of the existing ``test_real_upstream_reset_uses_
    per_task_dir`` strengthened per Codex LOW finding: after
    ``materialize_task_db`` (which now pre-copies) corrupt the working
    DB, then call the real upstream ``execute_submit_action``. The
    reset's ``os.remove`` + ``shutil.copy2`` MUST restore the working
    DB byte-equal to the template, AND the stable dataset sqlite (if
    it existed) MUST stay untouched."""
    from bird_interact_agents.harness import (
        SampleStatus,
        execute_submit_action,
    )

    root = _make_dataset_dir(tmp_path / "data", "alien")
    template = root / "alien" / "alien_template.sqlite"
    stable = root / "alien" / "alien.sqlite"
    make_tiny_sqlite(stable)
    stable_before = stable.read_bytes()

    task = {
        "instance_id": "alien_reset",
        "selected_database": "alien",
        "dataset": "livesqlbench-base-lite-sqlite",
        "sol_sql": ["SELECT id FROM widgets"],
        "category": "Query",
        "conditions": {"decimal": [], "distinct": False, "order": False},
        "test_cases": [],
    }
    working = _materialize(task, root)

    # Corrupt the pre-copied working DB to prove the upstream reset
    # actually overwrites it (and the pre-copy doesn't fight the reset).
    working.write_bytes(b"\x00" * 64)
    assert not working.read_bytes().startswith(_SQLITE_MAGIC), (
        "precondition: working DB is corrupted"
    )

    status = SampleStatus(idx=0, original_data=task)
    execute_submit_action("SELECT id FROM widgets", status, str(root))

    assert working.is_file() and working.stat().st_size > 0
    # `execute_submit_action` connects to the working DB after the reset's
    # `shutil.copy2`, so SQLite touches the header (e.g. byte 18 — file-
    # format write version). Don't require byte-equality of the whole
    # file; instead check the reset's semantic contract: the working DB
    # starts with the SQLite magic and exposes the same tables/rows as
    # the template (i.e. the corruption was wiped).
    assert working.read_bytes().startswith(_SQLITE_MAGIC), (
        "after upstream reset, working DB must be a real SQLite file "
        "(the reset wiped our intentional corruption)"
    )
    with sqlite3.connect(working) as wconn, sqlite3.connect(template) as tconn:
        wtables = sorted(r[0] for r in wconn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        ttables = sorted(r[0] for r in tconn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        assert wtables == ttables, (
            f"working DB tables {wtables!r} must match template {ttables!r}"
        )
        for tbl in ttables:
            wrows = list(wconn.execute(f"SELECT * FROM {tbl}"))
            trows = list(tconn.execute(f"SELECT * FROM {tbl}"))
            assert wrows == trows, (
                f"working DB rows in {tbl!r} must match template after reset"
            )
    assert stable.read_bytes() == stable_before, (
        "execute_submit_action must NOT rewrite the stable dataset "
        "<db>.sqlite — DEV-1462 per-task isolation is preserved"
    )


# ---------------------------------------------------------------------------
# 11. Documentation-via-code: rebut the Linear issue's casing framing
# ---------------------------------------------------------------------------


def test_slayer_ingest_preserves_table_casing_and_sqlite_case_insensitive(tmp_path):
    """The Linear issue (DEV-1509) framed the bug as a SLayer-introspection-
    vs-SQLite casing mismatch. This test pins the two real upstream
    invariants that rebut that framing — and is the reason the
    casing-regression-test bullet from the Linear AC is intentionally
    NOT honoured as written.

    Invariant A: ``slayer.engine.ingestion.ingest_datasource`` preserves
    the SQLite-stored casing of table names verbatim, surfacing it as
    BOTH ``SlayerModel.name`` AND ``SlayerModel.sql_table``. So a
    mixed-case SQLite produces mixed-case model names.

    Invariant B: SQLite is case-INsensitive for identifier lookup
    (unquoted AND quoted), so a SLayer model that emits SQL with one
    casing matches a SQLite table stored with a different casing. The
    cloud failure mode in DEV-1509 cannot be a true casing mismatch."""
    from slayer.core.models import DatasourceConfig
    from slayer.engine.ingestion import ingest_datasource

    sqlite_path = tmp_path / "mixed.sqlite"
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            'CREATE TABLE "ArtifactsCore" '
            '(id INTEGER PRIMARY KEY, name TEXT NOT NULL)'
        )
        conn.execute("INSERT INTO ArtifactsCore(id, name) VALUES (1, 'a')")
        conn.commit()
    finally:
        conn.close()

    # Invariant A: slayer ingest preserves the SQLite-stored casing.
    config = DatasourceConfig(
        name="mixedds",
        type="sqlite",
        connection_string=f"sqlite:///{sqlite_path}",
    )
    models = ingest_datasource(config)
    assert len(models) == 1
    m = models[0]
    assert m.name == "ArtifactsCore", (
        f"slayer ingest must preserve the SQLite-stored table casing "
        f"on SlayerModel.name; got name={m.name!r}"
    )
    assert m.sql_table == "ArtifactsCore", (
        f"slayer ingest must preserve the SQLite-stored table casing "
        f"on SlayerModel.sql_table; got sql_table={m.sql_table!r}"
    )

    # Invariant B: SQLite is case-insensitive for identifiers in BOTH
    # unquoted and quoted forms. Any of these three queries must succeed
    # against the same `ArtifactsCore` table.
    for sql in (
        "SELECT * FROM ArtifactsCore",
        "SELECT * FROM artifactscore",
        'SELECT * FROM "ArtifactsCore"',
    ):
        uri = f"file:{sqlite_path}?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            cur = c.cursor()
            cur.execute(sql)
            row = cur.fetchone()
        finally:
            c.close()
        assert row is not None, (
            f"SQLite identifier lookup must be case-insensitive; query "
            f"{sql!r} unexpectedly returned no rows against ArtifactsCore"
        )


# ---------------------------------------------------------------------------
# 12. Atomic copy: tmp + os.replace (so concurrent calls can't half-copy)
# ---------------------------------------------------------------------------


def test_materialize_uses_atomic_rename_for_working_db_copy(monkeypatch, tmp_path):
    """The new implementation MUST use ``shutil.copy2(template, tmp);
    os.replace(tmp, expected_db_file)`` so concurrent calls on the same
    instance_id can interleave without producing a partial/corrupt
    working DB. (Codex MEDIUM finding on same-instance concurrency.)

    We verify the implementation pattern by spying on
    ``harness.shutil.copy2`` and ``harness.os.replace`` and asserting:
      (a) ``shutil.copy2`` is called with destination ≠ expected_db_file
          (i.e. it writes to a tmp path),
      (b) the tmp destination lives in the same dir as expected_db_file
          (so ``os.replace`` is cross-rename-safe on every filesystem),
      (c) ``os.replace`` is called with ``(tmp_dst, expected_db_file)``."""
    import bird_interact_agents.harness as harness

    root = _make_dataset_dir(tmp_path / "data", "alien")
    task = _make_task("alien_atomic", "alien")

    copy2_calls: list[tuple[str, str]] = []
    replace_calls: list[tuple[str, str]] = []
    real_copy2 = harness.shutil.copy2
    real_replace = harness.os.replace

    def _spy_copy2(src, dst, *a, **kw):
        copy2_calls.append((str(src), str(dst)))
        return real_copy2(src, dst, *a, **kw)

    def _spy_replace(src, dst, *a, **kw):
        replace_calls.append((str(src), str(dst)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(harness.shutil, "copy2", _spy_copy2)
    monkeypatch.setattr(harness.os, "replace", _spy_replace)

    out = harness.materialize_task_db(task, str(root))
    assert out is not None
    expected_db_file = Path(out)

    # `materialize_task_db` installs BOTH the template symlink and the
    # working DB atomically via per-call unique .part- paths + os.replace
    # (the template entry is a symlink, so no shutil.copy2 there). Filter
    # to the working-DB replace — that's the one this test pins.
    assert len(copy2_calls) == 1, (
        f"expected exactly one shutil.copy2 call (the working DB); "
        f"got {copy2_calls!r}"
    )
    working_replace_calls = [
        c for c in replace_calls if Path(c[1]) == expected_db_file
    ]
    assert len(working_replace_calls) == 1, (
        f"expected exactly one os.replace call targeting the working DB "
        f"({expected_db_file!r}); got {replace_calls!r}"
    )
    copy_src, copy_dst = copy2_calls[0]
    rep_src, rep_dst = working_replace_calls[0]

    assert Path(copy_dst) != expected_db_file, (
        f"shutil.copy2 must NOT write directly to expected_db_file "
        f"({expected_db_file!r}); writing to a tmp + os.replace is what "
        f"makes the copy atomic. Got copy2 dst={copy_dst!r}"
    )
    assert Path(copy_dst).parent == expected_db_file.parent, (
        f"tmp file must live in the same dir as expected_db_file so "
        f"os.replace is atomic (same filesystem). Got tmp parent="
        f"{Path(copy_dst).parent!r}, expected parent="
        f"{expected_db_file.parent!r}"
    )
    assert Path(rep_src) == Path(copy_dst), (
        f"os.replace src must equal shutil.copy2 dst (the tmp file); "
        f"got rep_src={rep_src!r} copy_dst={copy_dst!r}"
    )
    assert Path(rep_dst) == expected_db_file, (
        f"os.replace dst must equal expected_db_file; got "
        f"rep_dst={rep_dst!r} expected={expected_db_file!r}"
    )
