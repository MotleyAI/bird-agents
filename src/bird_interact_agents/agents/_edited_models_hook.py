"""DEV-1649: shared finalize-and-persist hook for the on-the-fly slayer
interact agents.

Wraps ``harness.finalize_result_row`` (kept a pure stamping function) with the
edited-models save side + applied-from provenance, so all five agents share one
call site instead of duplicating the gating logic. Saving is gated (success +
flag + actual edits) inside ``maybe_save_edited_models`` — this helper is safe
to call from any post-resolve return.
"""

from __future__ import annotations

from pathlib import Path

from bird_interact_agents.harness import finalize_result_row
from bird_interact_agents.slayer_otf import edited_models as _edited_models


def finalize_with_edited_models_save(
    row: dict,
    *,
    deleted_kb_ids,
    slayer_storage_dir: str,
    benchmark: str | None = None,
    save_edited_models: bool = False,
    task_data: dict | None = None,
) -> dict:
    """Stamp the row (``finalize_result_row``), record ``edited_models_applied_from``
    from the resolver's stash, and — on success + ``--save-edited-models`` — persist
    the edited store. Returns the finalized row."""
    td = task_data or {}
    applied_from = td.get("_edited_models_applied_from")
    if applied_from:
        row["edited_models_applied_from"] = applied_from
    row = finalize_result_row(
        row, deleted_kb_ids=deleted_kb_ids, slayer_storage_dir=slayer_storage_dir,
    )
    _edited_models.maybe_save_edited_models(
        row,
        benchmark=benchmark,
        save_edited_models=save_edited_models,
        work_dir=Path(slayer_storage_dir).parent if slayer_storage_dir else None,
        slayer_storage_dir=slayer_storage_dir,
        deleted_kb_ids=deleted_kb_ids,
        cache_fp=td.get("_edited_models_cache_fp", ""),
    )
    return row
