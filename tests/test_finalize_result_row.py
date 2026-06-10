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


# ---------------------------------------------------------------------------
# DEV-1535 follow-up — usage.n_ask_user_calls chokepoint backstop +
# predicted_row_count backfill from the captured result snapshot.
# ---------------------------------------------------------------------------


def test_backfills_n_ask_user_calls_to_zero_when_usage_dict_missing_field():
    """Per-adapter edits cover the known callers; this is the defensive
    backstop for future adapters that forget the convention. Field is
    set to 0 (not None) so consumers can rely on its presence."""
    row = finalize_result_row(
        _row(trajectory=[], usage={"prompt_tokens": 100, "agent_cost_usd": 1.5}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["usage"]["n_ask_user_calls"] == 0


def test_preserves_explicit_n_ask_user_calls():
    """Adapter pre-set value of 5 (e.g. claude_sdk_otf_ainteract's
    `ctx_dict.asks_used`) survives the backstop."""
    row = finalize_result_row(
        _row(trajectory=[], usage={"prompt_tokens": 100, "n_ask_user_calls": 5}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["usage"]["n_ask_user_calls"] == 5


def test_initialises_usage_when_row_has_no_usage_field():
    """DEV-1535 r2 (CodeRabbit): early-error claude_sdk paths build a
    row with NO `usage` field at all (e.g.
    `claude_sdk_otf/agent.py:393-406`). Pre-fix the backfills here
    skipped silently and the annotation lost every usage-side
    telemetry field. The chokepoint now initialises `usage = {}`
    BEFORE applying backfills so those rows get the same defaults
    everyone else does."""
    row = finalize_result_row(
        _row(trajectory=[]),  # NO usage field
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    # n_agent_turns mirrors in (computed as 0 from empty trajectory),
    # n_ask_user_calls defaults to 0. No SDK metadata (no ResultMessage)
    # and no tool_call_stats (empty trajectory fails the discriminator).
    assert row["usage"] == {"n_agent_turns": 0, "n_ask_user_calls": 0}


def test_replaces_non_dict_usage_with_initialised_dict():
    """Defensive: a row whose `usage` was set to a non-dict (e.g. None
    in a very-early error path) gets replaced with an initialised dict
    so the chokepoint backfills always have a target."""
    row = finalize_result_row(
        _row(trajectory=[], usage=None),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["usage"] == {"n_agent_turns": 0, "n_ask_user_calls": 0}


def test_backfills_predicted_row_count_from_snapshot_dict():
    """`capture_result_snapshot` writes a dict; finalize reads `row_count`."""
    row = finalize_result_row(
        _row(
            trajectory=[],
            predicted_result_json={
                "columns": ["a", "b"], "row_count": 42,
                "sample_rows": [[1, 2]],
            },
        ),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["predicted_row_count"] == 42


def test_backfills_predicted_row_count_from_json_encoded_snapshot():
    """The same snapshot can arrive JSON-encoded (the cloud actor stuffs
    it through GCS as a string). Finalize transparently decodes."""
    import json as _json
    row = finalize_result_row(
        _row(
            trajectory=[],
            predicted_result_json=_json.dumps({
                "columns": ["a"], "row_count": 7, "sample_rows": [],
            }),
        ),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["predicted_row_count"] == 7


def test_preserves_explicit_predicted_row_count():
    """Adapter pre-set value must NOT be clobbered by the backfill."""
    row = finalize_result_row(
        _row(
            trajectory=[],
            predicted_row_count=99,
            predicted_result_json={"row_count": 7},
        ),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["predicted_row_count"] == 99


def test_predicted_row_count_no_op_on_malformed_snapshot_json():
    """Garbage in the snapshot slot stays as None — defensive against
    early-error rows where the field is a fragment."""
    row = finalize_result_row(
        _row(trajectory=[], predicted_result_json="not valid json"),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row.get("predicted_row_count") is None


def test_predicted_row_count_no_op_on_error_snapshot():
    """`capture_result_snapshot` returns `{"error": "..."}` on runtime
    failure (no `row_count` field) — stays None."""
    row = finalize_result_row(
        _row(
            trajectory=[],
            predicted_result_json={"error": "Connection failed"},
        ),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row.get("predicted_row_count") is None


def test_predicted_row_count_no_op_when_field_missing():
    """No `predicted_result_json` key on the row → stays None."""
    row = finalize_result_row(
        _row(trajectory=[]),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row.get("predicted_row_count") is None


# ---------------------------------------------------------------------------
# DEV-1535 follow-up — tool_call_stats backfill for claude_sdk
# trajectories + sdk_* metadata folded under `usage`.
# ---------------------------------------------------------------------------


def _claude_sdk_min_trajectory(num_turns: int = 3) -> list[dict]:
    """Minimal claude_sdk-shaped trajectory: one tool_use + matching
    tool_result + a final ResultMessage. Used by the backfill tests
    below."""
    return [
        {
            "type": "AssistantMessage",
            "data": {"content": [{"type": "tool_use",
                                   "name": "execute_sql", "id": "t1"}]},
        },
        {
            "type": "UserMessage",
            "data": {"content": [{"type": "tool_result",
                                   "tool_use_id": "t1",
                                   "is_error": False,
                                   "content": [{"type": "text", "text": "ok"}]}]},
        },
        {
            "type": "ResultMessage",
            "data": {"num_turns": num_turns,
                     "stop_reason": "end_turn",
                     "duration_ms": 5000,
                     "duration_api_ms": 4500},
        },
    ]


def test_backfills_tool_call_stats_for_claude_sdk_trajectory():
    """A claude_sdk-shaped trajectory triggers the walker — row gains
    `tool_call_stats` matching the pydantic_ai shape."""
    row = finalize_result_row(
        _row(trajectory=_claude_sdk_min_trajectory(), usage={}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["tool_call_stats"] is not None
    assert row["tool_call_stats"]["total_calls"] == 1
    assert row["tool_call_stats"]["total_errors"] == 0
    assert row["tool_call_stats"]["per_tool"] == [
        {"tool": "execute_sql", "n_calls": 1, "n_errors": 0},
    ]


def test_preserves_explicit_tool_call_stats():
    """pydantic_ai-style adapters set tool_call_stats during the run —
    the chokepoint backfill must NOT clobber that value."""
    explicit = {"per_tool": [{"tool": "foo", "n_calls": 99, "n_errors": 0}],
                "total_calls": 99, "total_errors": 0, "error_samples": []}
    row = finalize_result_row(
        _row(trajectory=_claude_sdk_min_trajectory(),
             usage={},
             tool_call_stats=explicit),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["tool_call_stats"] is explicit


def test_no_tool_call_stats_for_non_claude_sdk_shape():
    """A pydantic_ai trajectory doesn't match the discriminator — the
    walker returns None and the row's tool_call_stats stays absent.
    (pydantic_ai adapters populate it themselves.)"""
    pydantic_ai_shaped = [{"kind": "response", "parts": []}]
    row = finalize_result_row(
        _row(trajectory=pydantic_ai_shaped, usage={}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert "tool_call_stats" not in row or row.get("tool_call_stats") is None


def test_folds_sdk_result_metadata_under_usage():
    """The final ResultMessage's metadata lands as four `sdk_*` keys
    under `usage` — namespaced rather than top-level so results.db's
    `usage_json` carries them without further plumbing."""
    row = finalize_result_row(
        _row(trajectory=_claude_sdk_min_trajectory(num_turns=7), usage={}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["usage"]["sdk_num_turns"] == 7
    assert row["usage"]["sdk_stop_reason"] == "end_turn"
    assert row["usage"]["sdk_duration_ms"] == 5000
    assert row["usage"]["sdk_duration_api_ms"] == 4500


def test_sdk_metadata_does_not_clobber_explicit_values():
    """If an adapter has already populated one of the sdk_* keys (e.g.
    for a hand-rolled multi-phase run), the chokepoint backfill respects
    that. Backfill uses setdefault, not assignment."""
    row = finalize_result_row(
        _row(
            trajectory=_claude_sdk_min_trajectory(num_turns=7),
            usage={"sdk_stop_reason": "explicit_override"},
        ),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["usage"]["sdk_stop_reason"] == "explicit_override"
    # Other fields still backfilled.
    assert row["usage"]["sdk_num_turns"] == 7


def test_sdk_metadata_no_op_when_no_resultmessage():
    """No ResultMessage in the trajectory → walker returns None → no
    sdk_* keys added. Defensive against very-early agent crashes."""
    trajectory = [
        {"type": "AssistantMessage", "data": {"content": []}},
        {"type": "UserMessage", "data": {"content": []}},
    ]
    row = finalize_result_row(
        _row(trajectory=trajectory, usage={"prompt_tokens": 100}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert "sdk_num_turns" not in row["usage"]
    assert "sdk_stop_reason" not in row["usage"]


def test_sdk_metadata_lands_even_when_starting_usage_was_not_a_dict():
    """DEV-1535 r2 (CodeRabbit): with the chokepoint usage-init in
    place, a row starting with `usage=None` still gets the SDK
    metadata folded in — the fix makes the early-error claude_sdk
    paths analyzable for the first time."""
    row = finalize_result_row(
        _row(trajectory=_claude_sdk_min_trajectory(num_turns=4),
             usage=None),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert isinstance(row["usage"], dict)
    assert row["usage"]["sdk_num_turns"] == 4
    assert row["usage"]["sdk_stop_reason"] == "end_turn"
    assert row["usage"]["n_ask_user_calls"] == 0  # also initialised


# ---------------------------------------------------------------------------
# DEV-1535 r2 (Codex) — `n_agent_turns` backfill must mirror into `usage`
# too. All annotation writers read `usage_blob.get("n_agent_turns")`, so
# the top-level row key was dead for the very runs the backfill exists
# to fix.
# ---------------------------------------------------------------------------


def test_n_agent_turns_backfill_mirrors_into_usage():
    """The trajectory-derived `n_agent_turns` count lands BOTH at the
    top level (legacy compatibility) AND under `usage`, where the
    cloud + local annotation writers read it from."""
    trajectory = [
        {"type": "AssistantMessage", "data": {}},
        {"type": "AssistantMessage", "data": {}},
        {"type": "AssistantMessage", "data": {}},
    ]
    row = finalize_result_row(
        _row(trajectory=trajectory),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    assert row["n_agent_turns"] == 3
    assert row["usage"]["n_agent_turns"] == 3


def test_n_agent_turns_mirror_does_not_clobber_explicit_usage_value():
    """If an adapter already set `usage["n_agent_turns"]` explicitly
    (e.g. via the SDK's ResultMessage), the mirror must NOT overwrite
    it. The top-level backfill computes from trajectory length; the
    adapter value (which may differ slightly) wins."""
    trajectory = [
        {"type": "AssistantMessage", "data": {}},
        {"type": "AssistantMessage", "data": {}},
    ]
    row = finalize_result_row(
        _row(trajectory=trajectory, usage={"n_agent_turns": 99}),
        deleted_kb_ids=[], slayer_storage_dir="",
    )
    # Top-level backfill skipped because n_agent_turns is non-None
    # in usage already? Actually it checks row["n_agent_turns"], which
    # is None — so the trajectory-count IS computed and stored at
    # row["n_agent_turns"]. The MIRROR is what's gated on the existing
    # usage value.
    assert row["usage"]["n_agent_turns"] == 99

    row2 = finalize_result_row(
        _row(trajectory=[]),
        deleted_kb_ids=[], slayer_storage_dir="/data/x",
    )
    assert row2["variant_storage_path"] is None
