"""Shared pre-encoded-mode plumbing for the four SLayer ``claude_sdk`` OTF
agents (DEV-1586).

The on-the-fly OTF agents (``claude_sdk_otf{,_ainteract}{,_v1}``) encode KB
items into a per-task SLayer store at task time, using SLayer WRITE tools
(``create_model`` / ``edit_model`` / ``save_memory`` / ``validate_models``)
and a deterministic cache (base models + KB-as-memories) resolved via
:func:`bird_interact_agents.slayer_otf.resolve_otf_task_storage_dir`.

This module adds the *pre-encoded* alternative: the agent runs against an
ALREADY-encoded SLayer datasource (KB items already materialised as named
columns / measures), so it only needs INTROSPECTION tools. The four agents
share — verbatim, no copy-paste — the source-root selection, the
benchmark-aware per-task storage resolver (with HARD-8 deleted-KB masking),
and the write-tool filtering that lives here.

Two sources, selected by the ``--pre-encoded-models`` CLI flag:

* ``otf`` (default sense): the encoding-agent output at
  :func:`paths.slayer_models_otf_root` (``slayer_models_otf/<benchmark>/<db>``).
* ``custom``: the hand-curated committed models at
  :func:`paths.slayer_models_root` (``slayer_models/<db>``).

Per-task storage reuses the committed-reference path
(:func:`build_task_variant_storage`) rather than the OTF cache path, because
that helper already applies HARD-8 deleted-KB masking by ``meta.kb_id`` on
encoded columns / measures / aggregations / models — exactly the ablation
the pre-encoded consumer needs (parity with the ``pydantic_ai`` adapter).
Unlike ``harness.resolve_task_storage_dir`` it is benchmark-aware: it threads
``mini_interact_root`` / ``db_root`` from ``data_path_base`` so the per-task
datasource connection string re-anchors correctly for benchmark-scoped OTF
roots and non-mini benchmarks (Codex DEV-1586 High#1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from bird_interact_agents import paths
from bird_interact_agents.hard8_preprocessor import (
    build_task_variant_storage,
    extract_deleted_kb_ids,
)
from bird_interact_agents.harness import _task_variant_workdir

# Accepted values for the user-facing ``--pre-encoded-models`` flag.
PRE_ENCODED_SOURCES: tuple[str, ...] = ("otf", "custom")

# Back-compat default for an OLD manifest that carries
# ``slayer_setup == "pre-encoded"`` but no ``pre_encoded_source`` (those
# pre-DEV-1586 runs always meant the committed ``slayer_models`` reference).
LEGACY_PRE_ENCODED_SOURCE = "custom"

# Bare SLayer MCP tool names that MUTATE the models. The pre-encoded agents
# strip these so they can only introspect. ``save_memory`` is included even
# though it does not edit a model directly — it writes into the SLayer store,
# which the read-only consumer must not do.
WRITE_SLAYER_TOOLS: frozenset[str] = frozenset(
    {"create_model", "edit_model", "save_memory", "validate_models"}
)

# The same names as the model sees them (``mcp__slayer__`` prefixed).
WRITE_SLAYER_TOOL_NAMES: frozenset[str] = frozenset(
    f"mcp__slayer__{t}" for t in WRITE_SLAYER_TOOLS
)


class PreEncodedSetupError(RuntimeError):
    """Raised when a pre-encoded reference is missing or unusable.

    The read-only consumer has no write tools / LLM-encode step, so it can
    never build the reference itself; it fails loudly and points the caller
    at the encoding step (``scripts/build_otf_references.py`` for the ``otf``
    source, or curating ``slayer_models/<db>`` with built embeddings for the
    ``custom`` source).
    """


def derive_slayer_setup(pre_encoded_source: str | None) -> str:
    """Derive the internal ``slayer_setup`` value from the user-facing flag.

    ``--pre-encoded-models`` set (``otf``/``custom``) → ``"pre-encoded"``;
    omitted → ``"on-the-fly"`` (the default for the ``claude_sdk`` agents).
    The internal value is still consumed by the cloud routing / fingerprint /
    merge paths, so it is derived rather than retired outright (DEV-1586).
    """
    return "pre-encoded" if pre_encoded_source else "on-the-fly"


def validate_pre_encoded_source(pre_encoded_source: str | None) -> None:
    """Raise ``ValueError`` for an out-of-vocabulary source value.

    ``None`` (omitted) is valid — it selects the on-the-fly path.
    """
    if pre_encoded_source is None:
        return
    if pre_encoded_source not in PRE_ENCODED_SOURCES:
        raise ValueError(
            f"pre_encoded_source must be one of {PRE_ENCODED_SOURCES} or None; "
            f"got {pre_encoded_source!r}"
        )


def pre_encoded_source_root(source: str, *, benchmark: str) -> Path:
    """Return the canonical reference root for ``source``.

    * ``otf`` → ``paths.slayer_models_otf_root(benchmark=...)``
      (benchmark-scoped encoding-agent output).
    * ``custom`` → ``paths.slayer_models_root()`` (benchmark-agnostic
      hand-curated reference).

    Both are MAIN-checkout paths via the ``paths`` helpers — never a
    worktree-relative path (CLAUDE.md worktree-safety contract).
    """
    validate_pre_encoded_source(source)
    if source == "otf":
        return paths.slayer_models_otf_root(benchmark=benchmark)
    return paths.slayer_models_root()


def strip_write_slayer_tools(bare_names: Iterable[str]) -> list[str]:
    """Drop the mutating SLayer tools from a list of BARE tool names
    (the ``SLAYER_MCP_TOOLS`` whitelist form), order-preserving."""
    return [n for n in bare_names if n not in WRITE_SLAYER_TOOLS]


def strip_write_tool_names(prefixed_names: Iterable[str]) -> list[str]:
    """Drop the mutating SLayer tools from a list of MCP-prefixed tool names
    (the ``allowed_tools`` / partition-constant form), order-preserving.

    DEV-1581 R2 moved the SLayer tools off the ``slayer`` stdio server and onto
    the in-process ``bird-interact-tools`` server, so the names are now
    ``mcp__bird-interact-tools__create_model`` rather than
    ``mcp__slayer__create_model``. We therefore strip by the BARE tool suffix
    (server-prefix-agnostic) so both forms — and any future server name — are
    handled by the same predicate.
    """
    def _bare(name: str) -> str:
        return name.split("__")[-1] if name.startswith("mcp__") else name

    return [n for n in prefixed_names if _bare(n) not in WRITE_SLAYER_TOOLS]


def _assert_reference_present(source: str, db_dir: Path) -> None:
    """Fail-clear when the per-DB reference is missing or lacks embeddings.

    Presence semantics mirror ``driver._artifact_present``:
    * ``otf`` → the ``_reference_fp.txt`` completeness marker must exist.
    * ``custom`` → a non-empty committed dir (no marker convention).

    In BOTH cases a non-empty ``embeddings.db`` is required, because the
    pre-encoded agents run the SLayer MCP with ``ingest_on_startup=False``
    (startup ingest hangs the Claude Agent SDK session — no MCP
    startup-timeout knob), so semantic ``search`` depends on the embeddings
    shipped with the reference. We assert presence + non-empty only — NOT a
    model-match — relying on the project's locked-slayer-version invariant
    (DEV-1586 decision; Codex High#3 acknowledged as a documented limit).
    """
    if source == "otf":
        if not (db_dir / "_reference_fp.txt").is_file():
            raise PreEncodedSetupError(
                f"pre-encoded 'otf' reference missing for {db_dir.name!r}: "
                f"no _reference_fp.txt under {db_dir}. Build it first with "
                f"`scripts/build_otf_references.py <benchmark>` (runs the LLM "
                f"encoder), or submit a `pydantic_ai_otf_encode` run for the DB."
            )
    else:  # custom
        if not (db_dir.is_dir() and any(db_dir.iterdir())):
            raise PreEncodedSetupError(
                f"pre-encoded 'custom' reference missing for {db_dir.name!r}: "
                f"{db_dir} is absent or empty. Curate the committed models "
                f"under slayer_models/<db>/ first."
            )
    emb = db_dir / "embeddings.db"
    if not (emb.is_file() and emb.stat().st_size > 0):
        raise PreEncodedSetupError(
            f"pre-encoded {source!r} reference for {db_dir.name!r} has no "
            f"usable embeddings.db under {db_dir} (required because the "
            f"pre-encoded agents run with ingest_on_startup=False). For the "
            f"'custom' source, build embeddings before use; for 'otf', "
            f"rebuild the reference."
        )


async def resolve_pre_encoded_storage_dir(
    *,
    db_name: str,
    task_data: dict,
    data_path_base: str,
    benchmark: str,
    source: str,
) -> tuple[str, list[int]]:
    """Resolve the per-task SLayer storage for the pre-encoded path.

    Materialises a fresh per-task copy of the chosen reference
    (``<source-root>/<db>``) under the per-instance variant work dir, with
    this task's ``deleted_knowledge`` KB items masked (HARD-8) on any encoded
    column / measure / aggregation / model tagged ``meta.kb_id``. The
    committed reference is therefore read-only at runtime; SLayer's
    first-load type-refinement writes land in the per-task scratch dir.

    Returns ``(slayer_storage_dir, deleted_kb_ids)``. Raises
    :class:`PreEncodedSetupError` (fail-clear) when the reference is missing
    or has no usable embeddings.

    ``benchmark`` / ``data_path_base`` are threaded so the per-task datasource
    connection string re-anchors correctly for benchmark-scoped OTF roots and
    non-mini benchmarks — the gap that makes raw ``resolve_task_storage_dir``
    unsafe here (Codex DEV-1586 High#1).
    """
    validate_pre_encoded_source(source)
    root = pre_encoded_source_root(source, benchmark=benchmark)
    db_dir = root / db_name
    _assert_reference_present(source, db_dir)

    deleted = sorted(extract_deleted_kb_ids(task_data))
    instance_id = task_data["instance_id"]
    # ``.resolve()`` is load-bearing: the per-task datasource connection
    # string is re-anchored at this root, and a relative path would root at
    # ``/<rel>/...`` (mirrors resolve_otf_task_storage_dir's rationale).
    db_root = Path(data_path_base).resolve()
    variant_dir = await build_task_variant_storage(
        canonical_storage_root=root,
        db_name=db_name,
        deleted_kb_ids=set(deleted),
        work_dir=_task_variant_workdir(instance_id),
        mini_interact_root=db_root,
        db_root=db_root,
    )
    return str(variant_dir), deleted
