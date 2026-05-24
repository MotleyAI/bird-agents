"""T1–T5: image-hash / build helpers.

Each test corresponds to a row in SPEC_DEV-1453.md §10. The cloud package
doesn't exist yet — these tests fail-for-the-right-reason (import error)
and turn green as `bird_interact_agents.cloud.image` is implemented.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from bird_interact_agents.cloud import image  # noqa: E402  (intentional import)


# ---------------------------------------------------------------------------
# T1 — data_hash is content-based AND covers every input listed in §6.2.
# ---------------------------------------------------------------------------


def test_data_hash_stable_across_touches(fake_repo_root: Path,
                                         fake_mini_interact: Path) -> None:
    """Identical bytes ⇒ identical hash even after `touch` changes mtimes."""
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    time.sleep(0.01)
    for p in list(fake_mini_interact.rglob("*")) + list(
        (fake_repo_root / "slayer_models").rglob("*")
    ) + list((fake_repo_root / "audited_gold").rglob("*")):
        if p.is_file():
            os.utime(p, None)
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 == h2


def test_data_hash_changes_on_mini_interact_edit(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    (fake_mini_interact / "db_a" / "schema.sql").write_text(
        "CREATE TABLE t (x INT, y INT);\n"
    )
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 != h2


def test_data_hash_changes_on_slayer_models_edit(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    (fake_repo_root / "slayer_models" / "db_a" / "model.yaml").write_text(
        "models: [{name: extra}]\n"
    )
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 != h2


def test_data_hash_changes_on_audited_gold_edit(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    (fake_repo_root / "audited_gold" / "db_a" / "db_a_audited.jsonl").write_text(
        '{"instance_id":"db_a_1","sol_sql":"SELECT 2"}\n'
    )
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 != h2


def test_data_hash_changes_on_slayer_ingest_version_bump(
    monkeypatch: pytest.MonkeyPatch,
    fake_repo_root: Path,
    fake_mini_interact: Path,
) -> None:
    """§6.2: data_hash must change when the `slayer ingest` CLI version
    changes, since that re-derives `/data/slayer_dbs/`."""
    monkeypatch.setattr(image, "_slayer_ingest_version", lambda: "0.6.9")
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    monkeypatch.setattr(image, "_slayer_ingest_version", lambda: "0.6.10")
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 != h2


def test_data_hash_includes_dockerfile_data_section(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    df = (fake_repo_root / "Dockerfile.cloud").read_text()
    df_new = df.replace(
        "COPY mini-interact/ /data/mini-interact/\n",
        "COPY mini-interact/ /data/mini-interact/  # edited\n",
    )
    (fake_repo_root / "Dockerfile.cloud").write_text(df_new)
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 != h2


def test_data_hash_ignores_dockerfile_code_section(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    df = (fake_repo_root / "Dockerfile.cloud").read_text()
    df_new = df.replace(
        "RUN uv sync --extra all --extra dev\n",
        "RUN uv sync --extra all --extra dev  # edited\n",
    )
    (fake_repo_root / "Dockerfile.cloud").write_text(df_new)
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 == h2


def test_data_hash_unchanged_on_unrelated_untracked_files(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    """Untracked files outside image-input paths don't move data_hash."""
    h1 = image.data_hash(fake_repo_root, fake_mini_interact)
    (fake_repo_root / "scratch.txt").write_text("notes\n")
    h2 = image.data_hash(fake_repo_root, fake_mini_interact)
    assert h1 == h2


# ---------------------------------------------------------------------------
# T2 — code_hash covers every code input AND ignores data inputs.
# ---------------------------------------------------------------------------


def test_code_hash_stable_across_touches(fake_repo_root: Path,
                                          make_git) -> None:
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    time.sleep(0.01)
    for p in (fake_repo_root / "src").rglob("*"):
        if p.is_file():
            os.utime(p, None)
    h2 = image.code_hash(fake_repo_root, allow_dirty=False)
    assert h1 == h2


def test_code_hash_changes_on_src_edit(fake_repo_root: Path,
                                        make_git) -> None:
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    (fake_repo_root / "src" / "bird_interact_agents" / "run.py").write_text(
        "def main(): return 42\n"
    )
    make_git(status_porcelain=" M src/bird_interact_agents/run.py")
    h2 = image.code_hash(fake_repo_root, allow_dirty=True)
    assert h1 != h2


def test_code_hash_changes_on_uv_lock_edit(fake_repo_root: Path,
                                            make_git) -> None:
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    (fake_repo_root / "uv.lock").write_text("# lock v2\n")
    make_git(status_porcelain=" M uv.lock")
    h2 = image.code_hash(fake_repo_root, allow_dirty=True)
    assert h1 != h2


def test_code_hash_changes_on_pyproject_edit(fake_repo_root: Path,
                                              make_git) -> None:
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    (fake_repo_root / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.2.0'\n"
    )
    make_git(status_porcelain=" M pyproject.toml")
    h2 = image.code_hash(fake_repo_root, allow_dirty=True)
    assert h1 != h2


def test_code_hash_tracks_dockerfile_code_section(
    fake_repo_root: Path, make_git
) -> None:
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    df = (fake_repo_root / "Dockerfile.cloud").read_text()
    df_new = df.replace(
        "COPY src/ /app/bird-interact-agents/src/\n",
        "COPY src/ /app/bird-interact-agents/src/  # edit\n",
    )
    (fake_repo_root / "Dockerfile.cloud").write_text(df_new)
    h2 = image.code_hash(fake_repo_root, allow_dirty=False)
    assert h1 != h2


def test_code_hash_ignores_dockerfile_data_section(
    fake_repo_root: Path, make_git
) -> None:
    """The DATA-LAYERS sentinel block belongs to data_hash, not code_hash."""
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    df = (fake_repo_root / "Dockerfile.cloud").read_text()
    df_new = df.replace(
        "COPY mini-interact/ /data/mini-interact/\n",
        "COPY mini-interact/ /data/mini-interact/  # edit\n",
    )
    (fake_repo_root / "Dockerfile.cloud").write_text(df_new)
    h2 = image.code_hash(fake_repo_root, allow_dirty=False)
    assert h1 == h2


def test_code_hash_ignores_data_files(
    fake_repo_root: Path, fake_mini_interact: Path, make_git
) -> None:
    """Editing slayer_models / audited_gold / mini-interact must NOT move code_hash."""
    make_git(status_porcelain="")
    h1 = image.code_hash(fake_repo_root, allow_dirty=False)
    (fake_repo_root / "slayer_models" / "db_a" / "model.yaml").write_text(
        "models: [{name: extra}]\n"
    )
    (fake_mini_interact / "db_a" / "schema.sql").write_text(
        "CREATE TABLE t (x INT, y INT);\n"
    )
    # Pretend nothing changed under code-input paths (these are data paths).
    h2 = image.code_hash(fake_repo_root, allow_dirty=False)
    assert h1 == h2


# ---------------------------------------------------------------------------
# T3 — image_tag = data_hash[:12] + "-" + code_hash[:12] (+ -dirty when applicable).
# ---------------------------------------------------------------------------


def test_image_tag_composition(fake_repo_root: Path, fake_mini_interact: Path,
                                make_git) -> None:
    make_git(status_porcelain="")
    tag = image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=False)
    expected = (
        image.data_hash(fake_repo_root, fake_mini_interact)[:12]
        + "-"
        + image.code_hash(fake_repo_root, allow_dirty=False)[:12]
    )
    assert tag == expected
    assert "dirty" not in tag


def test_image_tag_idempotent(fake_repo_root: Path, fake_mini_interact: Path,
                               make_git) -> None:
    make_git(status_porcelain="")
    a = image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=False)
    b = image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=False)
    assert a == b


def test_image_tag_dirty_suffix(fake_repo_root: Path,
                                 fake_mini_interact: Path,
                                 make_git) -> None:
    make_git(status_porcelain=" M src/bird_interact_agents/run.py")
    tag = image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=True)
    assert tag.endswith("-dirty")


# ---------------------------------------------------------------------------
# T4 — dirty worktree refusal by default; allow_dirty=True succeeds.
# ---------------------------------------------------------------------------


def test_dirty_worktree_refused_by_default(fake_repo_root: Path,
                                            fake_mini_interact: Path,
                                            make_git) -> None:
    make_git(status_porcelain=" M src/bird_interact_agents/run.py")
    with pytest.raises(image.DirtyWorktreeError):
        image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=False)


def test_dirty_worktree_ignores_unrelated_changes(fake_repo_root: Path,
                                                    fake_mini_interact: Path,
                                                    make_git) -> None:
    """Changes outside image-input paths shouldn't trip the dirty check."""
    make_git(status_porcelain=" M README.md")  # README isn't an image input
    tag = image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=False)
    assert "dirty" not in tag


def test_dirty_worktree_includes_untracked_input_files(
    fake_repo_root: Path, fake_mini_interact: Path, make_git
) -> None:
    """Untracked files under image-input paths trip the dirty check."""
    # `??` is git's marker for untracked.
    make_git(status_porcelain="?? src/bird_interact_agents/scratch.py")
    with pytest.raises(image.DirtyWorktreeError):
        image.image_tag(fake_repo_root, fake_mini_interact, allow_dirty=False)


# ---------------------------------------------------------------------------
# Fail-closed defences — CR#7 (Dockerfile sentinels) + CR#8 (git failures).
# ---------------------------------------------------------------------------


def test_data_hash_raises_when_dockerfile_missing(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    (fake_repo_root / "Dockerfile.cloud").unlink()
    with pytest.raises(image.DockerfileSentinelError):
        image.data_hash(fake_repo_root, fake_mini_interact)


def test_data_hash_raises_when_sentinel_missing(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    dockerfile = fake_repo_root / "Dockerfile.cloud"
    text = dockerfile.read_text().replace("# DATA-LAYERS", "# nope")
    dockerfile.write_text(text)
    with pytest.raises(image.DockerfileSentinelError):
        image.data_hash(fake_repo_root, fake_mini_interact)


def test_code_hash_raises_on_git_failure(
    fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When git status fails for a reason other than 'not a git repo',
    code_hash must raise — failing open would let dirty trees through."""
    import subprocess as _sp

    def fake_run(argv, *_args, **_kw):
        if argv[:2] == ["git", "status"]:
            return _sp.CompletedProcess(
                argv, 1, stdout="",
                stderr="fatal: detected dubious ownership in repository\n",
            )
        return _sp.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    with pytest.raises(image.GitStatusUnavailable):
        image.code_hash(fake_repo_root, allow_dirty=False)


def test_code_hash_not_a_git_repo_is_silent(
    fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the repo_root isn't under git at all, fall back to empty set
    (nothing to check) instead of raising."""
    import subprocess as _sp

    def fake_run(argv, *_args, **_kw):
        if argv[:2] == ["git", "status"]:
            return _sp.CompletedProcess(
                argv, 128, stdout="",
                stderr="fatal: not a git repository (or any of the parent dirs)\n",
            )
        return _sp.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    # Doesn't raise; just hashes whatever's on disk.
    image.code_hash(fake_repo_root, allow_dirty=False)


# (T5 — `image_tag(..., detach=True)` rejection — lived here in v1 but was
# the wrong API boundary; CLI mutual-exclusion is asserted in test_cli.py
# `test_detach_and_allow_dirty_mutually_exclusive`.)
