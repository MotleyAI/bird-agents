"""DEV-1535 — `finalize_result_row` backfills `n_agent_turns` from the
trajectory for adapters that don't compute it explicitly.

Pre-fix only the pydantic_ai* adapters set this field; every claude_sdk*
adapter left it None, so the CascadingReport / budget analyses had to
fall back on `len(trajectory)` as a proxy. The fix lives in
`finalize_result_row` so every adapter benefits via its existing
result-builder call — no per-adapter diff needed.
"""

from __future__ import annotations

from bird_interact_agents.harness import finalize_result_row


def _row(trajectory=None, **kwargs):
    base = {"trajectory": trajectory if trajectory is not None else []}
    base.update(kwargs)
    return base


def test_backfills_n_agent_turns_from_assistant_messages():
    """The canonical case: count AssistantMessage entries in a trajectory
    produced by the claude_sdk loop."""
    trajectory = [
        {"type": "SystemMessage", "data": {"subtype": "init"}},
        {"type": "AssistantMessage", "data": {}},
        {"type": "UserMessage", "data": {}},
        {"type": "AssistantMessage", "data": {}},
        {"type": "UserMessage", "data": {}},
        {"type": "AssistantMessage", "data": {}},
        {"type": "ResultMessage", "data": {"num_turns": 3}},
    ]
    row = finalize_result_row(
        _row(trajectory=trajectory),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["n_agent_turns"] == 3


def test_zero_when_trajectory_has_no_assistant_messages():
    """An error-path trajectory (system init only, or empty) yields 0,
    not None — distinguishes 'agent never ran' from 'we don't know'."""
    row = finalize_result_row(
        _row(trajectory=[{"type": "SystemMessage", "data": {}}]),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["n_agent_turns"] == 0


def test_zero_when_trajectory_empty():
    row = finalize_result_row(
        _row(trajectory=[]),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["n_agent_turns"] == 0


def test_preserves_explicit_value_set_by_adapter():
    """Adapters that compute their own turn count (pydantic_ai* family)
    pre-populate `n_agent_turns`; the backfill must NOT clobber it."""
    trajectory = [
        {"type": "AssistantMessage", "data": {}},
        {"type": "AssistantMessage", "data": {}},
    ]
    row = finalize_result_row(
        _row(trajectory=trajectory, n_agent_turns=99),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["n_agent_turns"] == 99


def test_no_op_when_trajectory_field_missing():
    """A row without a trajectory key gets no backfill (defensive — the
    caller didn't give us anything to count)."""
    row = finalize_result_row(
        {"task_id": "x"},
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row.get("n_agent_turns") is None


def test_no_op_when_trajectory_is_not_a_list():
    """Defensive against malformed adapter output (e.g. an exception
    string stuffed into the field) — don't crash, just leave the count
    None for the caller to interpret."""
    row = finalize_result_row(
        _row(trajectory="this is not a list"),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row.get("n_agent_turns") is None


def test_non_dict_trajectory_items_are_ignored():
    """A claude_sdk fallback path can stuff `str(msg)` into the trajectory
    when `dataclasses.asdict(msg)` raises. Those entries lack a `type`
    key and must not contribute to the count."""
    trajectory = [
        "raw stringified msg from a non-dataclass path",
        {"type": "AssistantMessage", "data": {}},
        None,
        {"type": "AssistantMessage", "data": {}},
    ]
    row = finalize_result_row(
        _row(trajectory=trajectory),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["n_agent_turns"] == 2


def test_other_finalize_responsibilities_unchanged():
    """The backfill is additive — variant_storage_path and deleted_kb_ids
    plumbing must keep behaving as before."""
    row = finalize_result_row(
        _row(trajectory=[]),
        deleted_kb_ids=[7, 12], slayer_storage_dir="/data/x",
    )
    assert row["deleted_kb_ids"] == [7, 12]
    assert row["variant_storage_path"] == "/data/x"

    row2 = finalize_result_row(
        _row(trajectory=[]),
        deleted_kb_ids=[], slayer_storage_dir="/data/x",
    )
    assert row2["variant_storage_path"] is None
