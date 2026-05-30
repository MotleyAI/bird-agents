"""T1–T5: image-hash / build helpers.

De-bake (this chunk): the benchmark dataset (mini-interact / livesqlbench) is
NO LONGER baked into the image — it is uploaded to GCS at submit and the actor
downloads it per node. So ``data_hash`` no longer takes a ``mini_interact_root``
and no longer depends on the dataset bytes; ``audited_gold/`` (code-like
corrections) is the only remaining baked DATA layer.
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


def test_data_hash_stable_across_touches(fake_repo_root: Path) -> None:
    """Identical bytes ⇒ identical hash even after `touch` changes mtimes."""
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    time.sleep(0.01)
    for p in list((fake_repo_root / "slayer_models").rglob("*")) + list(
        (fake_repo_root / "audited_gold").rglob("*")
    ):
        if p.is_file():
            os.utime(p, None)
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 == h2


def test_data_hash_unchanged_on_dataset_edit(
    fake_repo_root: Path, fake_mini_interact: Path
) -> None:
    """De-bake: the dataset is no longer a build input, so ``data_hash`` does
    not even take it as an argument — editing the dataset dir cannot move the
    hash. (Regression guard: a future signature change must not re-introduce a
    dataset dependency that forces full image rebuilds on every dataset bump.)"""
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    (fake_mini_interact / "db_a" / "schema.sql").write_text(
        "CREATE TABLE t (x INT, y INT);\n"
    )
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 == h2


def test_data_hash_unchanged_on_slayer_models_edit(
    fake_repo_root: Path
) -> None:
    """DEV-1468: slayer_models/ is uploaded to GCS at submit, not baked into
    the image, so it is no longer a data-layer input. Editing it must NOT
    move data_hash (else every local slayer-model rebuild would force a full
    image rebuild for nothing)."""
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    (fake_repo_root / "slayer_models" / "db_a" / "model.yaml").write_text(
        "models: [{name: extra}]\n"
    )
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 == h2


def test_data_hash_changes_on_audited_gold_edit(
    fake_repo_root: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    (fake_repo_root / "audited_gold" / "db_a" / "db_a_audited.jsonl").write_text(
        '{"instance_id":"db_a_1","sol_sql":"SELECT 2"}\n'
    )
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 != h2


def test_data_hash_changes_on_livesqlbench_audited_gold_edit(
    fake_repo_root: Path
) -> None:
    """DEV-1510: the livesqlbench audit ships as a SINGLE top-level file
    `audited_gold/livesqlbench_audited.jsonl` (not a per-db sidecar dir).
    The image already bakes audited_gold/ recursively, so the new file
    rides in for free — but data_hash must include it so a content change
    forces an image rebuild (same contract as for per-db files)."""
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    (fake_repo_root / "audited_gold" / "livesqlbench_audited.jsonl").write_text(
        '{"instance_id":"museum_7","audit_status":"edited",'
        '"audited_sol_sql":["SELECT 1"]}\n'
    )
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 != h2, (
        "data_hash must include audited_gold/livesqlbench_audited.jsonl; "
        "otherwise an audited-gold edit doesn't force the image to rebake "
        "and cloud actors would keep the stale gold."
    )


def test_dockerfile_audited_gold_copy_covers_top_level_files() -> None:
    """The production Dockerfile.cloud must copy the WHOLE audited_gold
    dir (not `audited_gold/*/`), so a top-level
    `audited_gold/livesqlbench_audited.jsonl` is baked exactly like the
    per-db files. Regression guard: a future Dockerfile edit that
    narrows the COPY to per-db dirs (or accidentally drops the
    destination) would silently exclude the livesqlbench audit."""
    import re as _re

    repo_root = Path(__file__).resolve().parents[2]
    df = (repo_root / "Dockerfile.cloud").read_text()
    # Strip comment-only lines so a comment that happens to contain the
    # string `COPY --from=audited-gold . ` can't satisfy the test.
    non_comment_lines = [
        ln for ln in df.splitlines()
        if not ln.lstrip().startswith("#")
    ]
    body = "\n".join(non_comment_lines)
    # Anchored regex: COPY --from=audited-gold . <DEST_DIR_ENDING_IN audited_gold/>
    # The `.` source captures the entire context (so top-level files
    # like `livesqlbench_audited.jsonl` ride along), and the destination
    # must land under `audited_gold/` for the in-container path the
    # actor reads from.
    pattern = _re.compile(
        r"^\s*COPY\s+--from=audited-gold\s+\.\s+\S*audited_gold/?\s*$",
        flags=_re.MULTILINE,
    )
    assert pattern.search(body), (
        "Dockerfile.cloud must contain a `COPY --from=audited-gold . "
        "<dest>/audited_gold/` line (`.` source = whole context so "
        "top-level files like livesqlbench_audited.jsonl are baked). "
        f"Got (non-comment Dockerfile body):\n{body}"
    )


def test_data_hash_no_longer_depends_on_slayer_ingest_version() -> None:
    """DEV-1468: the image no longer runs `slayer ingest` (no baked
    /data/slayer_dbs), so the slayer-ingest CLI version is no longer a
    data-layer input. The helper is removed; data_hash must not reference it."""
    assert not hasattr(image, "_slayer_ingest_version"), (
        "_slayer_ingest_version should be removed once the ingest layer is gone"
    )


def test_real_dockerfile_no_longer_bakes_dataset_slayer_or_ingest() -> None:
    """Guard the PRODUCTION Dockerfile.cloud (not just the test fixture): the
    slayer setup is uploaded to GCS at submit, and (de-bake) so is the dataset,
    so the image must not COPY slayer_models/ / mini-interact, pre-ingest into
    /data/slayer_dbs, or pin BIRD_DB_PATH/BIRD_DATA_PATH."""
    repo_root = Path(__file__).resolve().parents[2]
    df = (repo_root / "Dockerfile.cloud").read_text()
    assert "COPY slayer_models" not in df
    assert "/data/slayer_dbs" not in df
    assert "slayer ingest" not in df
    # De-bake: the dataset is no longer baked, and the data-path envs are set
    # by the actor at runtime (not pinned in the image).
    assert "COPY --from=mini-interact" not in df
    assert "/data/mini-interact" not in df
    # Precise: the ENV ASSIGNMENT must be gone (the explanatory comment may
    # still name the vars to say the actor sets them at runtime).
    assert "BIRD_DB_PATH=" not in df
    assert "BIRD_DATA_PATH=" not in df
    # audited_gold (code-like corrections) stays baked.
    assert "audited_gold" in df


def test_data_hash_includes_dockerfile_data_section(
    fake_repo_root: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    df = (fake_repo_root / "Dockerfile.cloud").read_text()
    df_new = df.replace(
        "COPY audited_gold/  /app/bird-interact-agents/audited_gold/\n",
        "COPY audited_gold/  /app/bird-interact-agents/audited_gold/  # edited\n",
    )
    assert df_new != df  # the line we edit must exist in the DATA section
    (fake_repo_root / "Dockerfile.cloud").write_text(df_new)
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 != h2


def test_data_hash_ignores_dockerfile_code_section(
    fake_repo_root: Path
) -> None:
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    df = (fake_repo_root / "Dockerfile.cloud").read_text()
    df_new = df.replace(
        "RUN uv sync --extra all --extra dev\n",
        "RUN uv sync --extra all --extra dev  # edited\n",
    )
    (fake_repo_root / "Dockerfile.cloud").write_text(df_new)
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    assert h1 == h2


def test_data_hash_unchanged_on_unrelated_untracked_files(
    fake_repo_root: Path
) -> None:
    """Untracked files outside image-input paths don't move data_hash."""
    h1 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
    (fake_repo_root / "scratch.txt").write_text("notes\n")
    h2 = image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")
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
        "COPY audited_gold/  /app/bird-interact-agents/audited_gold/\n",
        "COPY audited_gold/  /app/bird-interact-agents/audited_gold/  # edit\n",
    )
    assert df_new != df  # the line we edit must exist in the DATA section
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


def test_image_tag_composition(fake_repo_root: Path, make_git) -> None:
    make_git(status_porcelain="")
    tag = image.image_tag(
        fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=False
    )
    expected = (
        image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")[:12]
        + "-"
        + image.code_hash(fake_repo_root, allow_dirty=False)[:12]
    )
    assert tag == expected
    assert "dirty" not in tag


def test_image_tag_idempotent(fake_repo_root: Path, make_git) -> None:
    make_git(status_porcelain="")
    a = image.image_tag(
        fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=False
    )
    b = image.image_tag(
        fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=False
    )
    assert a == b


def test_image_tag_dirty_suffix(fake_repo_root: Path, make_git) -> None:
    make_git(status_porcelain=" M src/bird_interact_agents/run.py")
    tag = image.image_tag(
        fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=True
    )
    assert tag.endswith("-dirty")


# ---------------------------------------------------------------------------
# T4 — dirty worktree refusal by default; allow_dirty=True succeeds.
# ---------------------------------------------------------------------------


def test_dirty_worktree_refused_by_default(fake_repo_root: Path,
                                            make_git) -> None:
    make_git(status_porcelain=" M src/bird_interact_agents/run.py")
    with pytest.raises(image.DirtyWorktreeError):
        image.image_tag(
            fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=False
        )


def test_dirty_worktree_ignores_unrelated_changes(fake_repo_root: Path,
                                                    make_git) -> None:
    """Changes outside image-input paths shouldn't trip the dirty check."""
    make_git(status_porcelain=" M README.md")  # README isn't an image input
    tag = image.image_tag(
        fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=False
    )
    assert "dirty" not in tag


def test_dirty_worktree_includes_untracked_input_files(
    fake_repo_root: Path, make_git
) -> None:
    """Untracked files under image-input paths trip the dirty check."""
    # `??` is git's marker for untracked.
    make_git(status_porcelain="?? src/bird_interact_agents/scratch.py")
    with pytest.raises(image.DirtyWorktreeError):
        image.image_tag(
            fake_repo_root, fake_repo_root / "audited_gold", allow_dirty=False
        )


# ---------------------------------------------------------------------------
# Fail-closed defences — CR#7 (Dockerfile sentinels) + CR#8 (git failures).
# ---------------------------------------------------------------------------


def test_data_hash_raises_when_dockerfile_missing(
    fake_repo_root: Path
) -> None:
    (fake_repo_root / "Dockerfile.cloud").unlink()
    with pytest.raises(image.DockerfileSentinelError):
        image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")


def test_data_hash_raises_when_sentinel_missing(
    fake_repo_root: Path
) -> None:
    dockerfile = fake_repo_root / "Dockerfile.cloud"
    text = dockerfile.read_text().replace("# DATA-LAYERS", "# nope")
    dockerfile.write_text(text)
    with pytest.raises(image.DockerfileSentinelError):
        image.data_hash(fake_repo_root, fake_repo_root / "audited_gold")


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


# ---------------------------------------------------------------------------
# DEV-1470: worktree-safety — `audited_gold/` lives in the MAIN checkout, NOT
# in worktrees. The image build must (a) hash its contents from the explicit
# `audited_gold_root` parameter (not `repo_root / audited_gold`), (b) wire it
# to docker via `--build-context audited-gold=<path>`, and (c) not surface
# edits to `audited_gold/` via the worktree's `git status` (it's gitignored,
# so it can never appear there anyway — but the dirty-prefix list shouldn't
# claim it does either).
# ---------------------------------------------------------------------------


def test_data_hash_uses_audited_gold_root_not_repo_subdir(
    fake_repo_root: Path, tmp_path: Path,
) -> None:
    """The hash MUST reflect the bytes under the passed `audited_gold_root`,
    NOT the (potentially absent) `repo_root / 'audited_gold'`. Worktree
    callers pass `paths.audited_gold_root()` (main-checkout-anchored); the
    worktree's own `audited_gold/` dir is empty or missing, but the hash
    must NOT change because of that."""
    # Real audited_gold lives at fake_repo_root/audited_gold (the fixture's
    # default). Baseline hash with this co-located layout:
    h_colocated = image.data_hash(
        fake_repo_root, fake_repo_root / "audited_gold",
    )

    # Now simulate a WORKTREE: pretend the worktree is a different dir that
    # lacks the audited_gold subdir, but the canonical audited_gold root
    # is still the original location.
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()
    # Worktree gets a copy of just the code-input scaffolding (uv.lock,
    # pyproject.toml, Dockerfile.cloud, src/) — NOT audited_gold/.
    for rel in ("uv.lock", "pyproject.toml", "Dockerfile.cloud"):
        (worktree_root / rel).write_text((fake_repo_root / rel).read_text())
    import shutil
    shutil.copytree(fake_repo_root / "src", worktree_root / "src")

    h_worktree = image.data_hash(
        worktree_root, fake_repo_root / "audited_gold",
    )
    assert h_colocated == h_worktree, (
        "data_hash differs between main-checkout and worktree callers — the "
        "function is leaking the repo_root path into the digest. Worktree "
        "submits would force needless rebuilds (or, worse, pull from an "
        "empty audited_gold and miss real data)."
    )


def test_data_hash_ignores_stray_audited_gold_under_worktree(
    fake_repo_root: Path, tmp_path: Path,
) -> None:
    """If a stale or unrelated `audited_gold/` sits under the worktree's
    `repo_root`, `data_hash` MUST IGNORE it — the only authoritative source
    is the explicit `audited_gold_root` arg."""
    worktree_root = tmp_path / "worktree"
    worktree_root.mkdir()
    for rel in ("uv.lock", "pyproject.toml", "Dockerfile.cloud"):
        (worktree_root / rel).write_text((fake_repo_root / rel).read_text())
    import shutil
    shutil.copytree(fake_repo_root / "src", worktree_root / "src")
    # Stale audited_gold sitting under the worktree — must NOT influence the hash.
    (worktree_root / "audited_gold" / "db_a").mkdir(parents=True)
    (worktree_root / "audited_gold" / "db_a" / "STALE.jsonl").write_text(
        '{"instance_id":"STALE","sol_sql":"SELECT 999"}\n'
    )

    h_stale_present = image.data_hash(
        worktree_root, fake_repo_root / "audited_gold",
    )
    h_baseline = image.data_hash(
        fake_repo_root, fake_repo_root / "audited_gold",
    )
    assert h_stale_present == h_baseline, (
        "data_hash reads from `worktree_root / 'audited_gold'` — that's the bug. "
        "It must read ONLY from the explicit `audited_gold_root` argument."
    )


def test_build_and_push_wires_audited_gold_and_not_dataset(
    fake_repo_root: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`build_and_push` must pass `--build-context audited-gold=<path>` to
    docker. De-bake: it must NOT pass a `mini-interact` (dataset) build
    context anymore — the dataset is delivered via GCS, not baked."""
    import subprocess as _sp

    captured: list[list[str]] = []

    def fake_run(argv, *_a, **_kw):
        captured.append(list(argv))
        # `manifest inspect` returns non-zero → forces a build path
        rc = 1 if argv[:3] == ["docker", "manifest", "inspect"] else 0
        return _sp.CompletedProcess(argv, rc, stdout="", stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    # Stub the config + paths resolution.
    from bird_interact_agents.cloud import config as _config
    monkeypatch.setattr(_config, "image_uri_prefix", lambda: "registry.example/x/runner")

    audited_root = tmp_path / "main-checkout" / "audited_gold"
    audited_root.mkdir(parents=True)
    (audited_root / "db_a").mkdir()
    (audited_root / "db_a" / "db_a_audited.jsonl").write_text('{"x":1}\n')

    uri = image.build_and_push(
        "deadbeef-cafebabe",
        fake_repo_root,
        audited_gold_root=audited_root,
        force=False,
    )
    assert uri == "registry.example/x/runner:deadbeef-cafebabe"

    # Find the docker build invocation and verify --build-context entries.
    build_argv = next(a for a in captured if a[:2] == ["docker", "build"])
    pairs: list[tuple[str, str]] = []
    for i, tok in enumerate(build_argv):
        if tok == "--build-context" and i + 1 < len(build_argv):
            k, _, v = build_argv[i + 1].partition("=")
            pairs.append((k, v))
    assert ("audited-gold", str(audited_root)) in pairs, (
        f"audited-gold build-context not passed; saw {pairs}"
    )
    ctx_names = [k for k, _ in pairs]
    assert "mini-interact" not in ctx_names, (
        f"dataset build-context must be gone after de-bake; saw {pairs}"
    )


def test_dirty_input_paths_no_longer_includes_audited_gold(
    fake_repo_root: Path, make_git,
) -> None:
    """`audited_gold/` is gitignored and lives in the MAIN checkout, not the
    worktree. It's content-hashed via `data_hash(..., audited_gold_root=...)`
    instead, so a status line under `audited_gold/` (impossible in practice
    since it's gitignored) MUST NOT trip the dirty check. Regression for the
    pre-DEV-1470 contract where editing audited_gold/ would block a submit."""
    make_git(status_porcelain=" M audited_gold/db_a/db_a_audited.jsonl")
    # allow_dirty=False — would raise if audited_gold/ were still an input
    # prefix. With the DEV-1470 fix it sails through.
    image.image_tag(
        fake_repo_root,
        fake_repo_root / "audited_gold",
        allow_dirty=False,
    )
