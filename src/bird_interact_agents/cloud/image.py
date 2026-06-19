"""Image hashing and build/push helpers.

`data_hash` and `code_hash` are content-based SHA-256 hashes over the
files that compose each side of the image. They are deliberately
*mtime-independent* so identical bytes yield identical hashes regardless
of checkout history or `touch` operations.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


class DirtyWorktreeError(RuntimeError):
    """Raised when image_tag() is invoked without `allow_dirty=True` while
    the worktree has uncommitted changes touching image-input paths."""


class UpstreamGraderUnavailableError(RuntimeError):
    """Raised by ``build_and_push`` when an upstream grader tree the
    Dockerfile expects to bake via BuildKit ``--build-context`` is
    missing on the host. Surfacing this at submit time (instead of
    letting ``docker build`` fail with an opaque BuildKit error) gives
    the user an actionable remediation: clone the upstream tree next to
    the main checkout, or set the corresponding env var. Without it the
    cascade-tier-N1 dispatch in the cloud actor silently degrades to
    legacy ``_set_equal`` — the exact regression this PR was meant to
    eliminate. (Codex round 6.)"""


# Paths whose contents bind into the docker image's CODE layers.
_CODE_RELATIVE_PATHS: tuple[str, ...] = (
    "src",
    "uv.lock",
    "pyproject.toml",
)


def _iter_files_under(root: Path) -> Iterable[Path]:
    if not root.exists():
        return iter(())
    if root.is_file():
        return iter([root])
    return (p for p in sorted(root.rglob("*")) if p.is_file())


def _iter_upstream_grader_files(root: Path) -> Iterable[Path]:
    """Iterate hashable source files under an upstream-grader tree.

    Filters out ``__pycache__/`` (rebuilt non-deterministically by any
    local import, which would otherwise churn the data-layer hash and
    force needless rebuilds) and only keeps Python source (the loader
    only reads ``.py`` — ``.sh`` / ``.ipynb`` siblings don't influence
    the grader's behaviour).

    Symlinks are skipped on purpose: a stray ``foo.py`` symlink whose
    target is under ``__pycache__/`` (or a path outside the grader root
    entirely) would otherwise bypass both the ``__pycache__`` parts
    filter (the symlink's own ``parts`` don't contain it) and the
    in-tree assumption (``read_bytes()`` follows the link). Hashing
    out-of-context bytes would make the image tag depend on whatever
    the symlink resolves to. (Codex round 6.)
    """
    if not root.exists():
        return iter(())
    return (
        p for p in sorted(root.rglob("*.py"))
        if p.is_file()
        and not p.is_symlink()
        and "__pycache__" not in p.parts
    )


def _hash_files(files: Iterable[Path], *, base: Path) -> str:
    """SHA-256 over (relative_path, bytes) for each file, in sorted order."""
    h = hashlib.sha256()
    for f in files:
        try:
            rel = f.relative_to(base)
        except ValueError:
            rel = f
        h.update(str(rel).encode())
        h.update(b"\x00")
        h.update(f.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


class DockerfileSentinelError(RuntimeError):
    """Raised when Dockerfile.cloud is missing the # DATA-LAYERS or
    # CODE-LAYERS sentinel. Failing closed here is essential — if we
    silently returned empty sections, Dockerfile edits would stop affecting
    `data_hash`/`code_hash` and the wrong image could be reused."""


def _split_dockerfile_sections(repo_root: Path) -> dict[str, str]:
    """Parse Dockerfile.cloud into {DATA, CODE} sections delimited by the
    `# DATA-LAYERS` and `# CODE-LAYERS` sentinels.

    Raises `DockerfileSentinelError` when Dockerfile.cloud is missing or
    either sentinel is absent."""
    path = repo_root / "Dockerfile.cloud"
    if not path.exists():
        raise DockerfileSentinelError(
            f"Dockerfile.cloud not found under {repo_root}"
        )
    text = path.read_text()
    data_marker = "# DATA-LAYERS"
    code_marker = "# CODE-LAYERS"
    if data_marker not in text or code_marker not in text:
        missing = [m for m in (data_marker, code_marker) if m not in text]
        raise DockerfileSentinelError(
            f"Dockerfile.cloud missing required sentinel(s): {missing}"
        )
    di = text.index(data_marker)
    ci = text.index(code_marker)
    if di < ci:
        return {"DATA": text[di:ci], "CODE": text[ci:]}
    return {"CODE": text[ci:di], "DATA": text[di:]}


def data_hash(
    repo_root: Path,
    audited_gold_root: Path,
    *,
    annotations_root: Path | None = None,
    bird_interact_evaluation_root: Path | None = None,
    livesqlbench_evaluation_root: Path | None = None,
) -> str:
    """Content-based hash over the inputs that compose the DATA layers
    of `Dockerfile.cloud`: ``audited_gold/`` and the Dockerfile's DATA
    section.

    Worktree-safe: ``audited_gold_root`` is passed in by the caller (driver
    resolves it via ``paths.audited_gold_root()``, which anchors at the main
    checkout via git's common dir). The Dockerfile pulls it in via BuildKit's
    ``--build-context`` so the build is runnable from any git worktree, where
    the dir doesn't physically live. The hash key always uses the on-image
    path (``audited_gold/<rel>``) so the digest is stable regardless of where
    the inputs live on the host.

    DEV-1468: ``slayer_models/`` is no longer baked (uploaded to GCS at
    submit), and the image no longer runs ``slayer ingest``, so neither the
    slayer-model files nor the slayer-ingest CLI version are data-layer
    inputs anymore.

    De-bake (this chunk): the benchmark dataset (mini-interact /
    livesqlbench) is ALSO no longer baked — it is delivered via GCS at submit
    (``cloud/benchmark_data.ensure_uploaded`` → content-hashed prefix → the
    actor downloads it per node), so the dataset tree is no longer a
    data-layer input and ``data_hash`` no longer depends on it. The gated
    gold sidecar rides along inside that GCS dataset upload (it lives in the
    data root), so it is not baked either. ``audited_gold/`` (author-produced
    corrections, code-like) stays baked + hashed.

    DEV-1515: ``annotations_root`` is optional; pass ``paths.annotations_root()``
    to include annotations content. Defaults to ``None`` (no annotations hashed)
    so test callers that don't supply it remain hermetic.

    DEV-1550: ``bird_interact_evaluation_root`` /
    ``livesqlbench_evaluation_root`` are the upstream grader ``evaluation/``
    subdirs that ``Dockerfile.cloud`` bakes via ``COPY --from=`` so the
    cascade tier N1 dispatch can load ``test_utils.py`` + ``db_utils.py``
    in the cloud actor. Hashed under stable on-image keys so the digest
    is independent of where the host files live (the build context is
    relocatable). Default ``None`` keeps hermetic tests hermetic; the
    runtime driver populates both via ``image.build_and_push``."""
    h = hashlib.sha256()

    # audited_gold (main-checkout-anchored, gitignored). Keyed under
    # ``audited_gold/<rel>`` to match the on-image path — hash is stable
    # whether the caller resolves the root from main checkout or worktree
    # (the bytes are identical either way; the path that produced them is
    # not part of the digest).
    for f in _iter_files_under(audited_gold_root):
        rel = f.relative_to(audited_gold_root)
        h.update(b"repo/")
        h.update(f"audited_gold/{rel.as_posix()}".encode())
        h.update(b"\x00")
        h.update(f.read_bytes())
        h.update(b"\x00")

    # DEV-1515: annotations/ — same posture as audited_gold/. Keyed
    # under ``annotations/<rel>`` so a worktree build produces the
    # same digest as a main-checkout build with identical content.
    if annotations_root is not None and annotations_root.exists():
        for f in _iter_files_under(annotations_root):
            rel = f.relative_to(annotations_root)
            h.update(b"repo/")
            h.update(f"annotations/{rel.as_posix()}".encode())
            h.update(b"\x00")
            h.update(f.read_bytes())
            h.update(b"\x00")

    # DEV-1550: upstream grader subtrees. Keys mirror the in-image bake
    # paths under ``upstream_graders/{bird-interact,livesqlbench}/`` so
    # the digest stays the same whether the host root is the developer's
    # checkout sibling or a CI mount. ``_iter_upstream_grader_files``
    # filters ``__pycache__`` so a stale .pyc on the host doesn't churn
    # the hash.
    for grader_root, hash_key_prefix in (
        (bird_interact_evaluation_root, "upstream_graders/bird-interact/"),
        (livesqlbench_evaluation_root, "upstream_graders/livesqlbench/"),
    ):
        if grader_root is None or not grader_root.exists():
            continue
        for f in _iter_upstream_grader_files(grader_root):
            rel = f.relative_to(grader_root)
            h.update(b"repo/")
            h.update((hash_key_prefix + rel.as_posix()).encode())
            h.update(b"\x00")
            h.update(f.read_bytes())
            h.update(b"\x00")

    # Dockerfile DATA-LAYERS section
    sections = _split_dockerfile_sections(repo_root)
    h.update(b"dockerfile-data/")
    h.update(sections["DATA"].encode())
    h.update(b"\x00")

    return h.hexdigest()


def code_hash(repo_root: Path, allow_dirty: bool) -> str:
    """Content-based hash over the inputs that compose the CODE layers
    of `Dockerfile.cloud`: `src/`, `uv.lock`, `pyproject.toml`, and the
    Dockerfile's CODE section.

    Raises `DirtyWorktreeError` if the worktree is dirty under any
    code-input path and `allow_dirty=False`.
    """
    dirty = _dirty_image_input_paths(repo_root)
    if dirty and not allow_dirty:
        raise DirtyWorktreeError(
            f"uncommitted changes under image-input paths: {sorted(dirty)}"
        )

    h = hashlib.sha256()
    for rel in _CODE_RELATIVE_PATHS:
        target = repo_root / rel
        if target.is_dir():
            files = list(_iter_files_under(target))
        elif target.is_file():
            files = [target]
        else:
            files = []
        for f in files:
            try:
                rp = f.relative_to(repo_root)
            except ValueError:
                rp = f
            h.update(b"code/")
            h.update(str(rp).encode())
            h.update(b"\x00")
            h.update(f.read_bytes())
            h.update(b"\x00")

    sections = _split_dockerfile_sections(repo_root)
    h.update(b"dockerfile-code/")
    h.update(sections["CODE"].encode())
    h.update(b"\x00")

    return h.hexdigest()


def image_tag(
    repo_root: Path,
    audited_gold_root: Path,
    *,
    allow_dirty: bool,
    annotations_root: Path | None = None,
    bird_interact_evaluation_root: Path | None = None,
    livesqlbench_evaluation_root: Path | None = None,
) -> str:
    """`<data_hash[:12]>-<code_hash[:12]>` (+ `-dirty` when `allow_dirty=True`
    and the worktree is dirty).

    See :func:`data_hash` for why ``audited_gold_root`` is a separate input
    (worktree-safety). ``annotations_root`` is the DEV-1515 sibling input —
    same rationale. ``bird_interact_evaluation_root`` /
    ``livesqlbench_evaluation_root`` (DEV-1550) point at the upstream
    grader ``evaluation/`` subdirs Dockerfile.cloud bakes into the image."""
    dh = data_hash(
        repo_root,
        audited_gold_root,
        annotations_root=annotations_root,
        bird_interact_evaluation_root=bird_interact_evaluation_root,
        livesqlbench_evaluation_root=livesqlbench_evaluation_root,
    )
    ch = code_hash(repo_root, allow_dirty=allow_dirty)
    tag = f"{dh[:12]}-{ch[:12]}"
    if allow_dirty and _dirty_image_input_paths(repo_root):
        tag += "-dirty"
    return tag


class GitStatusUnavailable(RuntimeError):
    """Raised when `git status --porcelain` can't run or fails — failing
    open here would let dirty trees through `image_tag(allow_dirty=False)`
    silently. Callers that are explicitly outside a git checkout (e.g.
    `build_and_push` from an unpacked tarball) must opt in via
    `allow_dirty=True` to bypass the check."""


def _dirty_image_input_paths(repo_root: Path) -> set[str]:
    """Set of porcelain status lines touching image-input paths.

    Raises `GitStatusUnavailable` when git is missing or `git status`
    fails, EXCEPT when the repo_root simply isn't under a git checkout
    (status returns 128 with a specific error message) — in that case we
    return an empty set, because there's nothing meaningful to check."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as e:
        raise GitStatusUnavailable(f"git unavailable: {e}") from e
    if res.returncode != 0:
        stderr = (res.stderr or "").lower()
        # `not a git repository` → not a checkout at all; nothing to check.
        if "not a git repository" in stderr:
            return set()
        raise GitStatusUnavailable(
            f"git status --porcelain failed in {repo_root}: {res.stderr.strip()}"
        )
    dirty: set[str] = set()
    # Image-input prefixes (anything that lands inside the docker image).
    # DEV-1468: slayer_models/ is uploaded to GCS at submit, not baked, so it
    # is no longer an image input — editing it must not block a submit.
    # DEV-1470: ``audited_gold/`` lives in the main checkout (gitignored),
    # NOT in the worktree, so editing it never appears in this worktree's
    # ``git status --porcelain`` anyway — and the build now pulls it in
    # via ``--build-context audited-gold=<paths.audited_gold_root()>``,
    # which is content-hashed by :func:`data_hash` (so a content change
    # rebuilds the image regardless of git tracking).
    input_prefixes = (
        "src/",
        "uv.lock",
        "pyproject.toml",
        "Dockerfile.cloud",
    )
    for line in res.stdout.splitlines():
        if len(line) < 4:
            continue
        # Porcelain format: "XY path" (XY is 2 chars + space, but `??` for untracked)
        status = line[:2]
        path = line[3:].strip()
        # `??` = untracked, `M`/`A`/`D` etc. = tracked changes
        if status.strip() and any(
            path.startswith(p) for p in input_prefixes
        ):
            dirty.add(path)
    return dirty


def _ensure_upstream_grader_tree_present(
    eval_root: Path, *, env_var: str, tree_label: str,
) -> None:
    """Raise :class:`UpstreamGraderUnavailableError` when the upstream
    grader tree at ``eval_root`` is missing or doesn't contain the
    required ``test_utils.py`` marker.

    Without this check, ``docker build --build-context <name>=<path>``
    fails downstream with an opaque BuildKit error. Catching it up
    front lets us point the user at the env-var override + the expected
    sibling-of-checkout layout.
    """
    marker = eval_root / "test_utils.py"
    if marker.is_file():
        return
    if not eval_root.is_dir():
        problem = f"directory not found"
    else:
        problem = f"directory exists but is missing required 'test_utils.py'"
    raise UpstreamGraderUnavailableError(
        f"Upstream {tree_label} grader tree {problem}: {eval_root}. "
        f"Either clone the upstream {tree_label} repo next to the "
        f"bird-agents main checkout, or set ${env_var} to the upstream "
        f"repo root. Without it, the cloud actor's cascade-tier-N1 "
        f"dispatch silently falls back to legacy `_set_equal`."
    )


def default_grader_eval_roots() -> tuple[Path, Path]:
    """Host-side eval-dir paths the cloud build bakes into the image.

    Returns ``(bird_interact_evaluation_root, livesqlbench_evaluation_root)``.
    Centralised so the driver / CLI callsites pass the same paths to
    :func:`image_tag` (data-layer hash) AND :func:`build_and_push`
    (BuildKit ``--build-context``) — drift between the two would mean
    rebuilding under a stale tag.
    """
    from bird_interact_agents import paths

    bird_interact_eval = (
        paths.bird_interact_upstream_root()
        / "mini_interact" / "knowledge_based"
        / "mini_interact_conv" / "evaluation"
    )
    livesqlbench_eval = (
        paths.livesqlbench_upstream_root() / "evaluation" / "src"
    )
    return bird_interact_eval, livesqlbench_eval


def build_and_push(
    tag: str,
    repo_root: Path,
    *,
    image_uri_prefix: str | None = None,
    audited_gold_root: Path | None = None,
    annotations_root: Path | None = None,
    bird_interact_evaluation_root: Path | None = None,
    livesqlbench_evaluation_root: Path | None = None,
    force: bool = False,
) -> str:
    """Build (if needed) and push the image, returning the full URI.

    `image_uri_prefix` defaults to `config.image_uri_prefix()` so any
    `BIRD_INTERACT_CLOUD_*` env-var overrides take effect.

    `audited_gold_root` defaults to `paths.audited_gold_root()` — the
    gitignored audited-gold dir that lives in the MAIN checkout, NOT in
    worktrees. It is wired in via BuildKit's
    ``--build-context audited-gold=<path>`` so the Dockerfile's
    ``COPY --from=audited-gold`` works from any worktree without the user
    having to mirror the dir.

    De-bake (this chunk): the benchmark dataset is no longer a build context
    (no ``--build-context mini-interact``) — it is delivered to the cluster
    via GCS at submit. ``audited_gold`` is the only remaining baked DATA
    layer.

    Skip-if-exists: if `docker manifest inspect <uri>:<tag>` succeeds and
    `force=False`, we don't rebuild.
    """
    from bird_interact_agents import paths
    from bird_interact_agents.cloud import config

    if image_uri_prefix is None:
        image_uri_prefix = config.image_uri_prefix()
    if audited_gold_root is None:
        audited_gold_root = paths.audited_gold_root()
    if annotations_root is None:
        annotations_root = paths.annotations_root()
    # DEV-1550: upstream BIRD-Interact + livesqlbench grader subtrees.
    # The build contexts point at the host evaluation/ subdirs that
    # `Dockerfile.cloud`'s `COPY --from=bird-interact-evaluation` /
    # `COPY --from=livesqlbench-evaluation` lines bake into the image.
    # `_MINI_INTERACT_REL` / `_LIVESQLBENCH_REL` in `upstream_ex_base`
    # are anchored under the eval grader root, so the dirs we point at
    # here must be the deeper `evaluation/` subdirs, not the upstream
    # repo roots.
    if bird_interact_evaluation_root is None:
        bird_interact_evaluation_root = (
            paths.bird_interact_upstream_root()
            / "mini_interact" / "knowledge_based"
            / "mini_interact_conv" / "evaluation"
        )
    if livesqlbench_evaluation_root is None:
        livesqlbench_evaluation_root = (
            paths.livesqlbench_upstream_root() / "evaluation" / "src"
        )
    # Fail-fast prereq check (Codex round 6): a missing upstream-grader
    # tree would otherwise blow up `docker build` with an opaque BuildKit
    # error about an unresolved `--build-context`. Surface an actionable
    # message naming the env-var remediation so the user can clone the
    # tree or repoint the var instead of decoding BuildKit's output.
    _ensure_upstream_grader_tree_present(
        bird_interact_evaluation_root,
        env_var="BIRD_BIRD_INTERACT_ROOT",
        tree_label="BIRD-Interact",
    )
    _ensure_upstream_grader_tree_present(
        livesqlbench_evaluation_root,
        env_var="BIRD_LIVESQLBENCH_ROOT",
        tree_label="livesqlbench",
    )
    uri = f"{image_uri_prefix}:{tag}"
    if not force:
        probe = subprocess.run(
            ["docker", "manifest", "inspect", uri],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return uri
    subprocess.run(
        [
            "docker", "build",
            "--build-context", f"audited-gold={audited_gold_root}",
            "--build-context", f"annotations={annotations_root}",
            "--build-context",
            f"bird-interact-evaluation={bird_interact_evaluation_root}",
            "--build-context",
            f"livesqlbench-evaluation={livesqlbench_evaluation_root}",
            "-t", uri,
            "-f", "Dockerfile.cloud",
            ".",
        ],
        cwd=str(repo_root),
        check=True,
    )
    subprocess.run(["docker", "push", uri], check=True)
    return uri
