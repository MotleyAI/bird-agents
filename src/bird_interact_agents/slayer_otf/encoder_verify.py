"""DEV-1589: shared hard-check / auto-wire helpers for the build-time OTF
reference encoders.

These run AFTER a per-KB agent session closes — when the SLayer stdio
subprocess is dead and the parent process is the single writer of the YAML.
Each helper opens a FRESH ``YAMLStorage(base_dir=build_dir)`` per call (never a
cached handle) so it always sees the subprocess's just-committed writes.

The four hard checks:

* **HC-present** — every claimed entity exists AND carries ``meta.kb_id`` (the
  sole HARD-8 deletion key). Datasource-SCOPED lookups (mini-interact and
  LiveSQLBench share model names like ``alien``).
* **HC-depuse** — a dependent KB must REFERENCE each successfully-encoded
  *declared* dependency's entity by name (structured-field inspection, in
  identifier positions only — a name inside a string literal does not count),
  never re-derive/inline it.
* **HC-desc** (:func:`autowire_descriptions`) — each encoded entity's
  description carries the verbatim KB row.
* **HC-mem** (:func:`autowire_memory_backrefs`) — each entity ref is recorded on
  the ``<db>_kb_<id>`` memory.

:func:`purge_kb_entities_and_backrefs` cleans up on any non-encoded / downgraded
outcome: it deletes every ``meta.kb_id``-tagged entity for the KB and prunes
those refs from the memory, so no orphan entity or dangling backref survives
into the committed reference.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_otf.encoder_types import (
    EncodedEntity,
    EncoderResult,
)

logger = logging.getLogger(__name__)


# A quoted string literal ('...' or "...") — stripped before tokenising so a
# dep name inside a literal is NOT counted as a structural reference.
_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# A SQL/identifier token.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _fresh_storage(build_dir: Path | str) -> YAMLStorage:
    return YAMLStorage(base_dir=str(build_dir))


def meta_has_kb_id(meta: Any, kb_id: int) -> bool:
    """True iff ``meta`` carries ``kb_id`` equal to ``kb_id`` (int or str)."""
    if not isinstance(meta, dict):
        return False
    raw = meta.get("kb_id")
    if raw is None:
        return False
    try:
        return int(raw) == int(kb_id)
    except (TypeError, ValueError):
        return False


def referenced_identifiers(text: Any) -> set[str]:
    """Identifier tokens in ``text`` with quoted string literals removed.

    The string-literal strip is what makes the dep-use check structural rather
    than a raw substring match: ``WHERE label = 'premium_revenue'`` yields
    ``{'label'}`` — ``premium_revenue`` (a literal value, not a reference) does
    NOT appear."""
    if not text:
        return set()
    stripped = _LITERAL_RE.sub(" ", str(text))
    return set(_IDENT_RE.findall(stripped))


def _iter_strings(value: Any):
    """Yield every string LEAF of a structured value (list/dict/pydantic),
    recursively. Dict KEYS are not yielded — only values."""
    if value is None:
        return
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            value = dump()
        except Exception:  # noqa: BLE001
            value = str(value)
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _iter_strings(v)


def _structured_ref_tokens(value: Any) -> set[str]:
    """Identifier tokens from the string VALUES of a STRUCTURED field
    (`source_queries` / `source_model_origin`). Unlike
    :func:`referenced_identifiers`, it does NOT strip quoted literals — in these
    structures a dependency reference IS a string value (e.g.
    ``source_model: "premium_revenue"``), so stripping it would lose the
    reference; and only values (not dict keys / field names) are tokenised, so
    structural field names can't create accidental matches (Codex review)."""
    toks: set[str] = set()
    for s in _iter_strings(value):
        toks |= set(_IDENT_RE.findall(s))
    return toks


def _leaf_buckets(model: Any) -> dict[str, list]:
    return {
        "column": list(getattr(model, "columns", None) or []),
        "measure": list(getattr(model, "measures", None) or []),
        "aggregation": list(getattr(model, "aggregations", None) or []),
    }


def _leaf_definition_text(item: Any) -> str:
    """The structured definition field of a leaf entity (Column.sql /
    Measure.formula / Aggregation.formula). Description is intentionally NOT
    included — a dep mention in prose is not a reference."""
    sql = getattr(item, "sql", None)
    formula = getattr(item, "formula", None)
    return " ".join(p for p in (sql, formula) if p)


async def entity_present_and_tagged(
    ent: EncodedEntity, storage: Any, db: str, kb_id: int,
) -> bool:
    """True iff ``ent`` exists in ``db`` AND carries ``meta.kb_id == kb_id``.

    Datasource-SCOPED (Codex r1 #5): a bare ``get_model`` resolves by datasource
    priority, so a same-named model in another datasource would be mis-read.
    """
    try:
        if ent.kind == "model":
            model = await storage.get_model(ent.name, data_source=db)
            return (
                model is not None
                and model.data_source == db
                and meta_has_kb_id(getattr(model, "meta", None), kb_id)
            )
        if ent.host_model is None:
            return False
        model = await storage.get_model(ent.host_model, data_source=db)
        if model is None or model.data_source != db:
            return False
        bucket = _leaf_buckets(model).get(ent.kind)
        if bucket is None:
            return False
        for item in bucket:
            if item.name == ent.name:
                return meta_has_kb_id(getattr(item, "meta", None), kb_id)
        return False
    except Exception:  # noqa: BLE001 — storage may raise on a missing model
        return False


async def _entity_reference_tokens(
    ent: EncodedEntity, storage: Any, db: str,
) -> set[str]:
    """Identifier tokens this entity's structured definition references."""
    try:
        if ent.kind == "model":
            model = await storage.get_model(ent.name, data_source=db)
            if model is None:
                return set()
            toks: set[str] = referenced_identifiers(getattr(model, "sql", None))
            # Query-backed models persist their structural dependency references
            # in backing_query_sql / source_queries / source_model_origin — a
            # model that references a dep through one of these must NOT be falsely
            # downgraded (Codex review).
            toks |= referenced_identifiers(getattr(model, "backing_query_sql", None))
            toks |= _structured_ref_tokens(getattr(model, "source_queries", None))
            toks |= _structured_ref_tokens(getattr(model, "source_model_origin", None))
            for bucket in _leaf_buckets(model).values():
                for item in bucket:
                    toks |= referenced_identifiers(_leaf_definition_text(item))
            return toks
        if ent.host_model is None:
            return set()
        model = await storage.get_model(ent.host_model, data_source=db)
        if model is None:
            return set()
        for item in _leaf_buckets(model).get(ent.kind, []):
            if item.name == ent.name:
                return referenced_identifiers(_leaf_definition_text(item))
        return set()
    except Exception:  # noqa: BLE001
        return set()


async def _depuse_failures(
    entities: list[EncodedEntity],
    encoded_deps: Iterable[EncoderResult],
    storage: Any,
    db: str,
) -> list[str]:
    """For each successfully-encoded *declared* dependency, require ≥1 of its
    entity names to appear in an identifier position of ≥1 of THIS KB's
    entities. ``encoded_deps`` is the caller-filtered declared+encoded set."""
    tokens: set[str] = set()
    for ent in entities:
        tokens |= await _entity_reference_tokens(ent, storage, db)
    failures: list[str] = []
    for dep in encoded_deps:
        dep_names = {e.name for e in dep.entities}
        if dep_names and not (dep_names & tokens):
            failures.append(
                f"KB {dep.kb_id} was already encoded as "
                f"{sorted(dep_names)} — reference it by name, do not re-derive "
                f"its logic."
            )
    return failures


async def hard_failures(
    build_dir: Path | str,
    db: str,
    kb_id: int,
    entities: list[EncodedEntity],
    encoded_deps: Iterable[EncoderResult],
) -> list[str]:
    """Unified HC-present + HC-depuse. Returns human-readable failure strings
    (fed verbatim into the corrective re-prompt); empty list ⇒ all checks pass.
    Opens a FRESH storage so it sees the subprocess's just-written YAML."""
    storage = _fresh_storage(build_dir)
    failures: list[str] = []
    for ent in entities:
        # The reported entity_ref MUST be the canonical one for this entity —
        # otherwise a typo'd/wrong-datasource ref passes the presence check and
        # autowire_memory_backrefs later stores the bad ref (Codex review).
        expected_ref = (
            f"{db}.{ent.name}" if ent.kind == "model"
            else f"{db}.{ent.host_model}.{ent.name}"
        )
        if ent.entity_ref != expected_ref:
            failures.append(
                f"entity_ref {ent.entity_ref!r} is not canonical for this "
                f"entity (expected {expected_ref!r})."
            )
            continue
        if not await entity_present_and_tagged(ent, storage, db, kb_id):
            failures.append(
                f"{ent.entity_ref} is missing or not tagged meta.kb_id={kb_id} "
                f"in storage — write it (and tag it) before submitting."
            )
    failures.extend(
        await _depuse_failures(entities, encoded_deps, storage, db)
    )
    return failures


async def autowire_descriptions(
    build_dir: Path | str,
    db: str,
    entities: list[EncodedEntity],
    verbatim_desc: str,
) -> None:
    """Ensure each encoded entity's description carries ``verbatim_desc``
    (the ``[kb=N]`` + verbatim KB row). Idempotent — skips if already present."""
    storage = _fresh_storage(build_dir)
    # Group leaf updates by host model so each model is rewritten once.
    by_host: dict[str, list[EncodedEntity]] = {}
    model_ents: list[EncodedEntity] = []
    for ent in entities:
        if ent.kind == "model" or ent.host_model is None:
            model_ents.append(ent)
        else:
            by_host.setdefault(ent.host_model, []).append(ent)

    for host, ents in by_host.items():
        model = await storage.get_model(host, data_source=db)
        if model is None:
            continue
        buckets = {
            "column": list(model.columns or []),
            "measure": list(model.measures or []),
            "aggregation": list(model.aggregations or []),
        }
        wanted = {(e.kind, e.name) for e in ents}
        for kind, items in buckets.items():
            for i, item in enumerate(items):
                if (kind, item.name) in wanted:
                    items[i] = item.model_copy(update={
                        "description": _prepend(verbatim_desc, item.description),
                    })
        await storage.save_model(model.model_copy(update={
            "columns": buckets["column"],
            "measures": buckets["measure"],
            "aggregations": buckets["aggregation"],
        }))

    for ent in model_ents:
        model = await storage.get_model(ent.name, data_source=db)
        if model is None:
            continue
        await storage.save_model(model.model_copy(update={
            "description": _prepend(verbatim_desc, model.description),
        }))


def _prepend(verbatim_desc: str, existing: str | None) -> str:
    existing = existing or ""
    if verbatim_desc in existing:
        return existing
    return verbatim_desc if not existing else f"{verbatim_desc}\n\n{existing}"


async def autowire_memory_backrefs(
    build_dir: Path | str, db: str, kb_id: int, entities: list[EncodedEntity],
) -> None:
    """Append each entity ref to the ``<db>_kb_<kb_id>`` memory. Idempotent."""
    storage = _fresh_storage(build_dir)
    mem_id = f"{db}_kb_{kb_id}"
    mem = await storage.get_memory_row(mem_id)
    if mem is None:
        return
    new_entities = list(mem.entities)
    changed = False
    for ent in entities:
        if ent.entity_ref not in new_entities:
            new_entities.append(ent.entity_ref)
            changed = True
    if not changed:
        return
    saved = await storage.save_memory(
        learning=mem.learning, entities=new_entities, query=mem.query,
        id=mem_id, description=mem.description,
    )
    await _refresh_memory_embedding(storage, saved)


async def purge_kb_entities_and_backrefs(
    build_dir: Path | str, db: str, kb_id: int,
) -> None:
    """Delete every ``meta.kb_id == kb_id`` entity for ``db`` and prune those
    refs from the ``<db>_kb_<kb_id>`` memory. Used on every non-encoded /
    downgraded outcome so no orphan entity or dangling backref survives."""
    storage = _fresh_storage(build_dir)
    try:
        names = await storage.list_models(data_source=db)
    except ValueError:
        return
    removed_refs: set[str] = set()
    # When a whole model is deleted, its leaf backrefs (`<db>.<model>.<leaf>`)
    # must also be pruned from memory — exact-match removal alone would leave
    # them dangling (CodeRabbit). Tracked as prefixes.
    removed_model_prefixes: set[str] = set()
    for name in names:
        model = await storage.get_model(name, data_source=db)
        if model is None:
            continue
        # A query-backed model entity tagged for this KB → delete the model.
        if meta_has_kb_id(getattr(model, "meta", None), kb_id):
            removed_refs.add(f"{db}.{name}")
            removed_model_prefixes.add(f"{db}.{name}.")
            try:
                # Scope the delete to `db` — a bare name resolves by datasource
                # priority and could drop a same-named model in another
                # datasource (Codex review).
                await storage.delete_model(name, data_source=db)
            except Exception:  # noqa: BLE001 — already absent is fine
                logger.debug("purge: model %r already absent", name)
            continue
        cols, drop_cols = _split_tagged(model.columns, kb_id)
        meas, drop_meas = _split_tagged(model.measures, kb_id)
        aggs, drop_aggs = _split_tagged(model.aggregations, kb_id)
        if not (drop_cols or drop_meas or drop_aggs):
            continue
        for leaf in (*drop_cols, *drop_meas, *drop_aggs):
            removed_refs.add(f"{db}.{name}.{leaf.name}")
        await storage.save_model(model.model_copy(update={
            "columns": cols, "measures": meas, "aggregations": aggs,
        }))

    mem_id = f"{db}_kb_{kb_id}"
    mem = await storage.get_memory_row(mem_id)
    if mem is None:
        return
    prefixes = tuple(removed_model_prefixes)
    pruned = [
        e for e in mem.entities
        if e not in removed_refs and not (prefixes and e.startswith(prefixes))
    ]
    if len(pruned) != len(mem.entities):
        saved = await storage.save_memory(
            learning=mem.learning, entities=pruned, query=mem.query,
            id=mem_id, description=mem.description,
        )
        await _refresh_memory_embedding(storage, saved)


def _split_tagged(items: Any, kb_id: int) -> tuple[list, list]:
    """Return ``(keep, drop)`` partitioning ``items`` on ``meta.kb_id``."""
    keep, drop = [], []
    for item in items or []:
        if meta_has_kb_id(getattr(item, "meta", None), kb_id):
            drop.append(item)
        else:
            keep.append(item)
    return keep, drop


async def _refresh_memory_embedding(storage: Any, saved: Any) -> None:
    """Fire the SearchService upsert hook (best-effort; short-circuits when no
    embedding client is configured — safe in CI / offline)."""
    try:
        from slayer.search.service import SearchService

        await SearchService(storage=storage).upsert_memory(saved)
    except Exception:  # noqa: BLE001 — embedding refresh is best-effort
        logger.debug("encoder_verify: embedding refresh skipped")
