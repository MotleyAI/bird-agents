"""DEV-1515: annotations/ bake-in to the cloud image.

Worktree-safety regression mirroring the existing audited_gold tests
in `tests/cloud/test_image.py`. The contract:

* `paths.annotations_root()` anchors at the main checkout.
* `image.data_hash` includes annotation content (so an annotation
  edit invalidates the cached image).
* `image.build_and_push` calls `docker build --build-context
  annotations=<paths.annotations_root()>` so the build runs cleanly
  from any worktree (where annotations/ may not exist locally).
* `Dockerfile.cloud` copies the build context to
  `/app/bird-interact-agents/annotations/` — matching how
  audited_gold is mounted (not `/workspace/...`).
"""
from __future__ import annotations

from pathlib import Path


def _make_annotations_dir(tmp: Path) -> Path:
    d = tmp / "annotations" / "mini-interact" / "alien"
    d.mkdir(parents=True)
    (d / "alien_1.task.json").write_text(
        '{"schema_version": 1, "kind": "task_annotation"}'
    )
    return tmp / "annotations"


def test_data_hash_includes_annotations_content(fake_repo_root):
    from bird_interact_agents.cloud import image

    annotations_root = _make_annotations_dir(fake_repo_root)
    audited_gold_root = fake_repo_root / "audited_gold"

    h1 = image.data_hash(
        fake_repo_root, audited_gold_root, annotations_root=annotations_root,
    )

    # Mutate one annotation file; hash must change.
    (annotations_root / "mini-interact" / "alien" / "alien_1.task.json").write_text(
        '{"schema_version": 1, "kind": "task_annotation", "x": 1}'
    )
    h2 = image.data_hash(
        fake_repo_root, audited_gold_root, annotations_root=annotations_root,
    )
    assert h1 != h2, "annotations content must be hashed into data_hash"


def test_data_hash_invariant_to_host_path_of_annotations_root(tmp_path):
    """The hash key is the on-image path `annotations/<rel>`, not the
    host path — so a worktree build (where the annotations root sits at
    a different absolute path) produces the SAME digest as a
    main-checkout build with identical content.
    """
    from bird_interact_agents.cloud import image

    # Use the cloud-test fake_repo_root fixture indirectly — re-create
    # the minimal layout under two different host paths.
    def _make_repo(base: Path) -> tuple[Path, Path, Path]:
        (base / "src" / "bird_interact_agents").mkdir(parents=True)
        (base / "src" / "bird_interact_agents" / "__init__.py").write_text(
            "VERSION = '0.1.0'\n"
        )
        (base / "src" / "bird_interact_agents" / "run.py").write_text(
            "def main(): pass\n"
        )
        (base / "uv.lock").write_text("# lock file\n")
        (base / "pyproject.toml").write_text("[project]\nname='x'\n")
        # Existing audited_gold also stable across hosts.
        ag = base / "audited_gold"
        ag.mkdir()
        (ag / "x_audited.jsonl").write_text("{}\n")
        # Annotations dir with identical content.
        ann = base / "annotations" / "mini-interact" / "alien"
        ann.mkdir(parents=True)
        (ann / "alien_1.task.json").write_text(
            '{"schema_version": 1, "kind": "task_annotation"}'
        )
        # Minimal Dockerfile (data_hash reads sentinel sections).
        (base / "Dockerfile.cloud").write_text(
            "# DATA-LAYERS\nCOPY x .\n# CODE-LAYERS\nCOPY y .\n"
        )
        return base, ag, base / "annotations"

    base_a = tmp_path / "host_a"
    base_b = tmp_path / "host_b"
    repo_a, ag_a, ann_a = _make_repo(base_a)
    repo_b, ag_b, ann_b = _make_repo(base_b)

    h_a = image.data_hash(repo_a, ag_a, annotations_root=ann_a)
    h_b = image.data_hash(repo_b, ag_b, annotations_root=ann_b)
    assert h_a == h_b, (
        "data_hash must be invariant to the host path of annotations_root "
        "— it must key by `annotations/<rel>` only"
    )


def test_dockerfile_cloud_copies_annotations_to_checkout_root():
    """Critical: copy target is `/app/bird-interact-agents/annotations/`
    (alongside `/app/bird-interact-agents/audited_gold/`), NOT
    `/workspace/...` — because `paths.main_checkout_root()` resolves
    to `/app/bird-interact-agents/` inside the worker."""
    from bird_interact_agents import paths as paths_mod

    dockerfile = Path(paths_mod.__file__).parent.parent.parent / "Dockerfile.cloud"
    text = dockerfile.read_text()
    assert "COPY --from=annotations . /app/bird-interact-agents/annotations/" in text


def test_build_and_push_passes_annotations_build_context(monkeypatch, fake_repo_root):
    """Build invocation must include `--build-context annotations=<root>`."""
    from bird_interact_agents.cloud import image

    captured: list[list[str]] = []

    def fake_run(cmd, **_kw):  # noqa: ARG001
        captured.append(list(cmd))

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(image.subprocess, "run", fake_run)

    annotations_root = _make_annotations_dir(fake_repo_root)
    audited_gold_root = fake_repo_root / "audited_gold"
    # DEV-1550 round 7: pass explicit upstream grader eval roots so this
    # test is hermetic and doesn't depend on the dev/CI machine having
    # sibling `BIRD-Interact` / `livesqlbench` checkouts. Each root must
    # carry the markers `_ensure_upstream_grader_tree_present` requires.
    bird_interact_eval = fake_repo_root / "_bird_interact_eval"
    bird_interact_eval.mkdir()
    livesqlbench_eval = fake_repo_root / "_livesqlbench_eval"
    livesqlbench_eval.mkdir()
    for d in (bird_interact_eval, livesqlbench_eval):
        (d / "test_utils.py").write_text("")
        (d / "db_utils.py").write_text("")

    image.build_and_push(
        "test-tag",
        fake_repo_root,
        audited_gold_root=audited_gold_root,
        annotations_root=annotations_root,
        bird_interact_evaluation_root=bird_interact_eval,
        livesqlbench_evaluation_root=livesqlbench_eval,
        force=True,
    )
    # The actual build invocation is the call containing "docker build" —
    # there may be a prior `docker manifest inspect` probe; force=True
    # skips it but be defensive in case the implementation reorders.
    build_cmds = [c for c in captured if "build" in c]
    assert build_cmds, "expected at least one `docker build` invocation"
    cmd = " ".join(build_cmds[0])
    assert "--build-context" in cmd
    assert f"annotations={annotations_root}" in cmd
