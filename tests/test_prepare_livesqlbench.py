"""DEV-1462 — `scripts/prepare_livesqlbench.py`.

The upstream dataset ships per-DB `<db>_template.sqlite` files as git-LFS
pointers; `ensure_db_cache` reads the STABLE `<db>/<db>.sqlite` directly.
The prepare script's job is to detect the LFS-pointer state (and refuse
loudly) and to copy each template into the stable filename, idempotently.

Tests verify the script's CLI contract end-to-end via `subprocess.run`
(rather than monkeypatching internals) so the same artefact that a user
invokes lands in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests._livesqlbench_fixtures import (
    make_lsb_dataset,
    make_lfs_pointer,
    make_tiny_sqlite,
    public_task,
)


def _script_path() -> Path:
    """The script lives at <repo>/scripts/prepare_livesqlbench.py."""
    return Path(__file__).resolve().parents[1] / "scripts" / "prepare_livesqlbench.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Drive the script with `sys.executable` to inherit the venv."""
    return subprocess.run(
        [sys.executable, str(_script_path()), *args],
        capture_output=True, text=True, check=False,
    )


def test_script_file_exists():
    """Pin the path so the rest of the suite has something to drive."""
    assert _script_path().is_file(), (
        f"prepare_livesqlbench.py must live at {_script_path()}"
    )


def test_copies_template_to_stable_sqlite(tmp_path):
    root = tmp_path / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(root, dbs=["alien", "credit"], tasks=[])
    res = _run(["--root", str(root)])
    assert res.returncode == 0, (
        f"prepare_livesqlbench must succeed on real templates; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    for db in ("alien", "credit"):
        stable = root / db / f"{db}.sqlite"
        assert stable.is_file(), f"missing stable sqlite for {db}: {stable}"
        # Content matches the template (same byte count is a cheap proxy
        # — the fixture's tiny_sqlite is deterministic).
        template = root / db / f"{db}_template.sqlite"
        assert stable.stat().st_size == template.stat().st_size


def test_refuses_lfs_pointer_template(tmp_path, capsys):
    """An LFS-pointer template MUST exit non-zero with `git lfs pull`
    guidance — silently ingesting a 132-byte pointer would corrupt the
    OTF cache fingerprint AND poison every downstream task."""
    root = tmp_path / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(
        root, dbs=["alien"], tasks=[], template_as_lfs_pointer=True,
    )
    res = _run(["--root", str(root)])
    assert res.returncode != 0, (
        "prepare_livesqlbench must exit non-zero on an LFS-pointer template"
    )
    combined = (res.stdout or "") + (res.stderr or "")
    assert "git lfs pull" in combined, (
        "prepare_livesqlbench must print the `git lfs pull` remediation "
        f"in its error output; got: {combined!r}"
    )
    # Crucially, it must NOT have written a 132-byte stable sqlite.
    assert not (root / "alien" / "alien.sqlite").exists()


def test_is_idempotent(tmp_path):
    """Re-running the script must not re-copy when the stable file is up to date."""
    root = tmp_path / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(root, dbs=["alien"], tasks=[])
    res1 = _run(["--root", str(root)])
    assert res1.returncode == 0
    stable = root / "alien" / "alien.sqlite"
    first_mtime_ns = stable.stat().st_mtime_ns
    # Second run — must be a no-op for `alien`.
    res2 = _run(["--root", str(root)])
    assert res2.returncode == 0
    assert stable.stat().st_mtime_ns == first_mtime_ns, (
        "second run must not rewrite the stable sqlite when it's already current"
    )


def test_force_overwrites_existing_stable(tmp_path):
    """`--force` must rewrite even if the stable file looks up to date.

    Drive the difference by truncating the stable file after the first run;
    `--force` must re-copy it from the template (restoring the byte count).
    """
    root = tmp_path / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(root, dbs=["alien"], tasks=[])
    res1 = _run(["--root", str(root)])
    assert res1.returncode == 0
    stable = root / "alien" / "alien.sqlite"
    template = root / "alien" / "alien_template.sqlite"
    assert stable.stat().st_size == template.stat().st_size
    # Truncate to a sentinel size so we can prove the force path re-ran.
    stable.write_bytes(b"x")
    assert stable.stat().st_size == 1
    res2 = _run(["--root", str(root), "--force"])
    assert res2.returncode == 0
    assert stable.stat().st_size == template.stat().st_size, (
        "--force must overwrite the truncated stable sqlite from the template"
    )


def test_dbs_subset_only_acts_on_listed_dbs(tmp_path):
    """`--dbs db1,db2` must restrict materialisation to that set; an
    omitted DB keeps no stable file."""
    root = tmp_path / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(root, dbs=["alien", "credit", "fake"], tasks=[])
    res = _run(["--root", str(root), "--dbs", "alien,credit"])
    assert res.returncode == 0
    assert (root / "alien" / "alien.sqlite").exists()
    assert (root / "credit" / "credit.sqlite").exists()
    assert not (root / "fake" / "fake.sqlite").exists(), (
        "--dbs must NOT materialise dbs outside the listed set"
    )


def test_prints_per_db_summary(tmp_path):
    """The script must surface a per-DB status line so a user can verify
    what landed without re-listing the dir."""
    root = tmp_path / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(root, dbs=["alien"], tasks=[])
    res = _run(["--root", str(root)])
    assert res.returncode == 0
    combined = (res.stdout or "") + (res.stderr or "")
    assert "alien" in combined, (
        f"prepare_livesqlbench must mention each DB in its summary; "
        f"got: {combined!r}"
    )


def test_default_root_is_paths_livesqlbench_root(monkeypatch, tmp_path):
    """With no `--root`, the script falls back to
    `paths.benchmark_data_root('livesqlbench-base-lite-sqlite')`, which is
    driven by `BIRD_BENCHMARKS_ROOT`."""
    benchmarks_root = tmp_path / "benchmarks"
    root = benchmarks_root / "livesqlbench-base-lite-sqlite"
    make_lsb_dataset(root, dbs=["alien"], tasks=[])
    env = {"BIRD_BENCHMARKS_ROOT": str(benchmarks_root)}
    res = subprocess.run(
        [sys.executable, str(_script_path())],
        capture_output=True, text=True, check=False,
        env={**__import__("os").environ, **env},
    )
    assert res.returncode == 0, (
        f"prepare_livesqlbench must honour BIRD_BENCHMARKS_ROOT when "
        f"--root is omitted; stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert (root / "alien" / "alien.sqlite").exists()
