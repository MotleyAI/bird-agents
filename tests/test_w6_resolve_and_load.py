"""W6 wiring tests: ``resolve_task_storage_dir`` against the real
per-DB YAML trees in ``slayer_models/``.

Exercises the full preprocessor + resolver path end-to-end with a real
task shape so we know the harness will hand the agent a YAMLStorage
that actually has the deletion applied.

Relies on the canonical ``slayer_models/households/`` tree shipped in
this repo. ``households_1`` from ``mini_interact.jsonl`` deletes KB 15
(``properties.bathroom_ratio``); we use the same KB id here so the
test mirrors the real benchmark contract.
"""

from pathlib import Path

from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.harness import resolve_task_storage_dir


SLAYER_MODELS_ROOT = Path(__file__).resolve().parent.parent / "slayer_models"


def _task(instance_id: str, db_name: str, deleted: int | None) -> dict:
    ka = []
    if deleted is not None:
        ka.append({"term": "stub", "deleted_knowledge": deleted})
    return {
        "instance_id": instance_id,
        "selected_database": db_name,
        "knowledge_ambiguity": ka,
    }


async def test_resolve_raw_mode_returns_empty():
    """Raw mode bypasses the preprocessor entirely."""
    path, deleted = await resolve_task_storage_dir(
        slayer_storage_root=str(SLAYER_MODELS_ROOT),
        db_name="households",
        task_data=_task("t1", "households", 15),
        query_mode="raw",
    )
    assert path == ""
    assert deleted == []


async def test_resolve_slayer_no_deletions_materialises_per_task_copy():
    """Slayer mode with no deletions still materialises a per-task copy
    under $TMPDIR so the canonical reference stays read-only at runtime
    (SLayer's first-load type-refinement and any agent
    create_model/edit_model calls are confined to the per-task scratch
    dir)."""
    path, deleted = await resolve_task_storage_dir(
        slayer_storage_root=str(SLAYER_MODELS_ROOT),
        db_name="households",
        task_data=_task("t1", "households", None),
        query_mode="slayer",
    )
    assert deleted == []
    # Per-task path, not the canonical one.
    assert path != f"{SLAYER_MODELS_ROOT}/households"
    assert Path(path).exists()

    # Copy contains every model from the canonical reference.
    src_names = sorted(
        await YAMLStorage(base_dir=str(SLAYER_MODELS_ROOT / "households")).list_models()
    )
    dst_names = sorted(await YAMLStorage(base_dir=path).list_models())
    assert dst_names == src_names


async def test_resolve_slayer_with_deletion_drops_real_entity():
    """Slayer mode with a real deletion produces a variant missing the
    targeted column."""
    path, deleted = await resolve_task_storage_dir(
        slayer_storage_root=str(SLAYER_MODELS_ROOT),
        db_name="households",
        task_data=_task("households_1", "households", 15),
        query_mode="slayer",
    )
    assert deleted == [15]
    assert path != f"{SLAYER_MODELS_ROOT}/households"

    variant = YAMLStorage(base_dir=path)
    properties = await variant.get_model("properties")
    assert properties is not None
    column_names = {c.name for c in properties.columns}
    # ``bathroom_ratio`` carries meta.kb_id=15 in the canonical YAML; the
    # variant should not expose it.
    assert "bathroom_ratio" not in column_names
    # And the rest of the model survived.
    assert "household_density" in column_names or len(column_names) > 5


async def test_resolve_slayer_canonical_still_has_entity():
    """Sanity check: the canonical YAML still has the deleted entity —
    we're confirming the variant is what's filtered, not the source."""
    canonical = YAMLStorage(base_dir=str(SLAYER_MODELS_ROOT / "households"))
    properties = await canonical.get_model("properties")
    assert properties is not None
    column_names = {c.name for c in properties.columns}
    assert "bathroom_ratio" in column_names, (
        "Canonical YAML lost bathroom_ratio — preprocessor is mutating the source"
    )


async def test_resolve_slayer_root_unset_returns_empty():
    """Missing slayer_storage_root short-circuits even in slayer mode."""
    path, deleted = await resolve_task_storage_dir(
        slayer_storage_root=None,
        db_name="households",
        task_data=_task("t1", "households", 15),
        query_mode="slayer",
    )
    assert path == ""
    assert deleted == []
