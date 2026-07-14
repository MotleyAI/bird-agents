"""DEV-1672: propagate raw-JSON-column hiding to stores the build-time encoder
change does NOT rebuild.

The phase-3 encoder change hides raw JSON columns in freshly-built OTF caches,
and the ``_PHASE3_IMPL_TOKEN`` bump forces warm caches to rebuild. But two store
kinds are not covered by that:

* saved edited-model archives (``runs/<bench>/<db>/<iid>/edited_models.tar.gz``)
  that ``--apply-edited-models`` reuses — they were snapshotted from the OLD
  cache and are validated against the full ``_cache_fp.txt`` (unchanged), so
  they are still accepted with visible raw columns;
* OTF reference stores (``slayer_models_otf/<db>``) that are merged, not
  cache-rebuilt.

``hide_store_dir`` / ``hide_archive`` flip ``hidden=True`` on the already-expanded
raw JSON columns in those stores, idempotently, reusing the single
``hide_expanded_jsonb_columns`` predicate. ``hide_archive`` preserves the
archive's ``_edited_models_meta.json`` (its stamped ``cache_fp``) so
``apply_or_none`` keeps accepting it.

NB: apply-time self-heal (``edited_models.materialize_from_saved_store``) is the
correctness *guarantee* — this module is on-disk cleanup that lets an operator
migrate stores ahead of time and also covers the reference stores.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_pipeline.jsonb import hide_expanded_jsonb_columns


async def hide_store_dir(base_dir, *, data_source: str | None = None) -> int:
    """Flip ``hidden=True`` on every already-expanded raw JSON column in the YAML
    store at *base_dir*. When *data_source* is given, confine the sweep to that
    datasource's models; otherwise sweep every datasource in the store.
    Idempotent; returns the number of columns newly hidden."""
    storage = YAMLStorage(base_dir=str(base_dir))
    sources = [data_source] if data_source else await storage.list_datasources()
    total = 0
    for ds in sources:
        for name in await storage.list_models(data_source=ds):
            model = await storage.get_model(name, data_source=ds)
            if model is None:
                continue
            flipped = hide_expanded_jsonb_columns(model)
            if flipped:
                await storage.save_model(model)
                total += flipped
    return total


def _extract_all(tar: tarfile.TarFile, dest: Path) -> None:
    # PEP 706 ``filter="data"`` exists on 3.11.4+/3.12+; gate on the marker so
    # older 3.11 interpreters (which raise on the kwarg) still work.
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    # pragma: no cover - only on pre-3.11.4 interpreters. Path-traversal-safe
    # fallback mirroring ``edited_models._safe_extractall`` (kept inline to
    # avoid a circular import: edited_models already imports this module):
    # reject absolute paths / ``..`` escapes / links before extracting.
    resolved = Path(dest).resolve()
    for m in tar.getmembers():
        target = (resolved / m.name).resolve()
        if target != resolved and resolved not in target.parents:
            raise tarfile.TarError(f"unsafe path in archive: {m.name!r}")
        if m.issym() or m.islnk():
            raise tarfile.TarError(f"link in archive: {m.name!r}")
    tar.extractall(dest)  # noqa: S202 — members validated just above


async def hide_archive(archive_path) -> int:
    """Unpack a saved ``edited_models.tar.gz``, hide its raw JSON columns, and
    repack IN PLACE — preserving the whole tree (including
    ``_edited_models_meta.json`` / its ``cache_fp``) so ``apply_or_none`` still
    accepts the archive. Idempotent; returns the number of columns newly hidden.
    A no-op archive (already hidden) is left byte-for-byte untouched."""
    archive_path = Path(archive_path)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with tarfile.open(archive_path, "r:gz") as tar:
            top = {n.split("/", 1)[0] for n in tar.getnames() if n and n != "."}
            _extract_all(tar, tmp)
        # The archive is rooted at a single ``<db>/`` dir (see save_edited_store).
        top.discard("")
        if len(top) != 1:
            raise ValueError(
                f"{archive_path}: expected exactly one top-level dir, got {sorted(top)}"
            )
        db = next(iter(top))
        store = tmp / db
        flipped = await hide_store_dir(store, data_source=db)
        if flipped:
            _repack(store, archive_path, db)
        return flipped


def _repack(store: Path, archive_path: Path, db: str) -> None:
    """Re-tar *store* to *archive_path* under ``arcname=db`` (includes the
    preserved meta file). Atomic via a sibling tmp + ``os.replace``."""
    tmp_out = archive_path.parent / f".{archive_path.name}.tmp"
    try:
        with tarfile.open(tmp_out, "w:gz") as tar:
            tar.add(store, arcname=db)
        os.replace(tmp_out, archive_path)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
