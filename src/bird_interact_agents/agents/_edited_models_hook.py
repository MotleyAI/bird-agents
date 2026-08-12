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
from bird_interact_agents.slayer_otf.timing import log_otf_event


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
    + DEV-1778 ``consumed_edited_models`` from the resolver's stash, and — on
    success + ``--save-edited-models`` — persist the edited store. Returns the
    finalized row."""
    td = task_data or {}
    applied_from = td.get("_edited_models_applied_from")
    if applied_from:
        row["edited_models_applied_from"] = applied_from
        # DEV-1778: stamp which store STATE was consumed. Requires the
        # fingerprint AND both identity fields; else omit rather than build a
        # None-field record that would fail annotation validation downstream.
        store_fp = td.get("_edited_models_consumed_store_fp")
        db = row.get("database") or td.get("selected_database")
        iid = row.get("instance_id") or td.get("instance_id")
        if store_fp and db and iid:
            row["consumed_edited_models"] = {
                "db": db, "instance_id": iid, "store_fp": store_fp,
            }
        elif store_fp:
            log_otf_event(
                "otf.edited_models.consumed_stamp_skipped",
                db=db, instance_id=iid,
            )
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
