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
) -> str:
    # DEV-1515: default to paths.annotations_root() so callers that omit
    # the argument still include annotations content in the hash.
    if annotations_root is None:
        from bird_interact_agents import paths as _paths
        annotations_root = _paths.annotations_root()
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
    corrections, code-like) stays baked + hashed."""
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
) -> str:
    """`<data_hash[:12]>-<code_hash[:12]>` (+ `-dirty` when `allow_dirty=True`
    and the worktree is dirty).

    See :func:`data_hash` for why ``audited_gold_root`` is a separate input
    (worktree-safety). ``annotations_root`` is the DEV-1515 sibling input —
    same rationale."""
    dh = data_hash(repo_root, audited_gold_root, annotations_root=annotations_root)
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


def build_and_push(
    tag: str,
    repo_root: Path,
    *,
    image_uri_prefix: str | None = None,
    audited_gold_root: Path | None = None,
    annotations_root: Path | None = None,
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
            "-t", uri,
            "-f", "Dockerfile.cloud",
            ".",
        ],
        cwd=str(repo_root),
        check=True,
    )
    subprocess.run(["docker", "push", uri], check=True)
    return uri
