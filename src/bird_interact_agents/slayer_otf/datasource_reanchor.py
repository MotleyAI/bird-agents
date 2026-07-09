"""Per-task datasource ``connection_string`` re-anchoring (DEV-1649).

Extracted from :mod:`bird_interact_agents.slayer_otf.runtime` so that BOTH the
cache-copy path (``prepare_task_storage``) AND the saved-store apply path
(``edited_models.materialize_from_saved_store``) can re-anchor a copied
datasource without a circular import between ``runtime`` and ``edited_models``.

``runtime`` keeps its private ``_rewrite_datasource_connection_string`` name as a
thin alias for back-compat (pinned by ``tests/test_recursive_runtime_db_root``).
"""

from __future__ import annotations

from pathlib import Path

from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.slayer_pipeline.portable_connection import (
    reanchor_connection_string,
)


async def rewrite_datasource_connection_string(
    *,
    db: str,
    scratch: Path,
    mini_interact_root: Path,
    db_root: Path | None = None,
) -> None:
    """Re-anchor the copied datasource's ``connection_string`` to the current
    mini-interact root.

    The cache/store was built by ``slayer datasources create`` against whatever
    root the FIRST call used. If a subsequent call's ``--db-path`` points
    elsewhere (a different worktree, a fresh clone, or a different machine —
    the DEV-1478 cloud bug), the baked-in absolute path is stale.

    Delegates to :func:`reanchor_connection_string`, which force-rewrites the
    stale-foreign-absolute form while resolving the relative form against the
    root. Root precedence (DEV-1462): an explicit ``db_root`` wins over
    ``$BIRD_DB_PATH``.
    """
    storage = YAMLStorage(base_dir=str(scratch))
    ds = await storage.get_datasource(db)
    if ds is None:
        raise RuntimeError(
            f"slayer_otf: cached scratch is missing datasource {db!r}; "
            f"cache_dir layout may have changed."
        )
    if ds.connection_string is None:
        return

    resolved = reanchor_connection_string(
        ds.connection_string, db, mini_interact_root, db_root=db_root,
    )
    if resolved != ds.connection_string:
        ds = ds.model_copy(update={"connection_string": resolved})
        await storage.save_datasource(ds)
