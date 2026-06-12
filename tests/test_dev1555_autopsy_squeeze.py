"""DEV-1555 Stage 1: autopsy prompt hygiene.

1. Echo strip — ``_compress_trajectory_for_autopsy`` replaces the
   ``tool_use_result`` SDK echo (a full duplicate of every tool result,
   ~half the serialized trajectory) with a size marker. Dict-shaped items
   only; legacy string items pass through untouched (Codex r1 #7).
2. ``_estimate_tokens`` — chars/3.5, ceil.
3. ``fit_trajectory_for_autopsy`` — deterministic progressive squeeze:
   under budget → unchanged; over budget → elide tool-result block bodies
   oldest-first, never touching assistant text, tool inputs, or the last
   ``keep_last`` items; if still over → drop middle items behind a single
   ``ElidedItems`` marker, preserving the first ``keep_head`` and last
   ``keep_last`` items.
4. ``run_autopsy`` applies the squeeze so the final prompt fits
   ``context_window_for(model) * 0.75 - 4096`` tokens.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Synthetic trajectory builders (mirror the live SDK dump shape)
# ---------------------------------------------------------------------------

def _assistant_item(i: int, *, with_thinking: bool = False) -> dict:
    content = [
        {"id": f"tu_{i}", "name": "mcp__slayer__query", "input": {"q": f"q{i}"}},
        {"text": f"assistant text {i}"},
    ]
    if with_thinking:
        content.insert(0, {"thinking": "T" * 50, "signature": "sig"})
    return {"type": "AssistantMessage", "data": {"content": content}}


def _tool_result_item(i: int, *, body_chars: int = 2000) -> dict:
    body = f"r{i}:" + ("Z" * body_chars)
    return {
        "type": "UserMessage",
        "data": {
            "content": [
                {"tool_use_id": f"tu_{i}", "content": body, "is_error": False},
            ],
            "tool_use_result": {"echo": body, "extra": body},
        },
    }


def _legacy_string_item(i: int) -> dict:
    return {"type": "UserMessage", "data": f"legacy repr {i}"}


def _traj(n_pairs: int, *, body_chars: int = 2000) -> list[dict]:
    items: list[dict] = []
    for i in range(n_pairs):
        items.append(_assistant_item(i))
        items.append(_tool_result_item(i, body_chars=body_chars))
    return items


# ---------------------------------------------------------------------------
# 1. Echo strip
# ---------------------------------------------------------------------------

def test_compress_strips_tool_use_result_echo_with_size_marker():
    from bird_interact_agents.eval.autopsy import (
        _compress_trajectory_for_autopsy,
    )

    original = _tool_result_item(0)
    echo_chars = len(json.dumps(original["data"]["tool_use_result"]))
    out = _compress_trajectory_for_autopsy([original])

    assert out[0]["data"]["tool_use_result"] == (
        f"[tool_use_result: {echo_chars} chars]"
    )
    # The model-visible tool result block is untouched.
    assert out[0]["data"]["content"] == original["data"]["content"]
    # Input not mutated.
    assert isinstance(original["data"]["tool_use_result"], dict)


def test_compress_still_strips_thinking_and_passes_strings_through():
    from bird_interact_agents.eval.autopsy import (
        _compress_trajectory_for_autopsy,
    )

    traj = [_assistant_item(0, with_thinking=True), _legacy_string_item(1)]
    out = _compress_trajectory_for_autopsy(traj)
    thinking_block = out[0]["data"]["content"][0]
    assert thinking_block["thinking"] == "[thinking: 50 chars]"
    assert out[1] == _legacy_string_item(1)


# ---------------------------------------------------------------------------
# 2. Token estimator
# ---------------------------------------------------------------------------

def test_estimate_tokens_is_chars_over_3_5_ceil():
    from bird_interact_agents.eval.autopsy import _estimate_tokens

    assert _estimate_tokens("x" * 35) == 10
    assert _estimate_tokens("x" * 36) == 11
    assert _estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# 3. fit_trajectory_for_autopsy
# ---------------------------------------------------------------------------

def _fit(traj, budget, **kw):
    from bird_interact_agents.eval.autopsy import fit_trajectory_for_autopsy

    return fit_trajectory_for_autopsy(traj, budget_tokens=budget, **kw)


def _ctraj(n_pairs: int, *, body_chars: int = 2000) -> list[dict]:
    """Compressed synthetic trajectory — ``fit`` always runs AFTER the
    echo strip in production (run_autopsy compresses first), so the fit
    tests exercise the same pipeline."""
    from bird_interact_agents.eval.autopsy import (
        _compress_trajectory_for_autopsy,
    )

    return _compress_trajectory_for_autopsy(_traj(n_pairs, body_chars=body_chars))


def _size_tokens(traj) -> int:
    from bird_interact_agents.eval.autopsy import _estimate_tokens

    return _estimate_tokens(json.dumps(traj))


def test_fit_returns_input_unchanged_when_under_budget():
    traj = _ctraj(3)
    assert _fit(traj, 10**9) == traj


def test_fit_elides_oldest_tool_results_first_and_fits_budget():
    traj = _ctraj(40)  # 80 items; ~80K chars of tool-result bodies
    budget = int(_size_tokens(traj) * 0.6)
    out = _fit(traj, budget)

    assert _size_tokens(out) <= budget
    assert len(out) == len(traj)
    # Last 20 items bit-identical.
    assert out[-20:] == traj[-20:]
    # Oldest tool-result body elided with the size marker.
    first_result = out[1]["data"]["content"][0]
    orig_body = traj[1]["data"]["content"][0]["content"]
    n = len(json.dumps(orig_body))
    assert first_result["content"] == f"[tool result elided: {n} chars]"
    # Assistant text and tool inputs never touched.
    for got, want in zip(out, traj):
        if got["type"] != "AssistantMessage":
            continue
        assert got["data"]["content"] == want["data"]["content"]


def test_fit_is_deterministic():
    traj = _ctraj(40)
    budget = int(_size_tokens(traj) * 0.6)
    assert _fit(traj, budget) == _fit(traj, budget)


def test_fit_drops_middle_items_behind_single_marker_when_elision_insufficient():
    traj = _ctraj(40)  # 80 items
    # Below what body elision alone can reach (~34% here) but above the
    # head+tail floor (~26%) — forces middle drops, keeps it feasible.
    budget = int(_size_tokens(traj) * 0.30)
    out = _fit(traj, budget)

    assert _size_tokens(out) <= budget
    markers = [it for it in out if it.get("type") == "ElidedItems"]
    assert len(markers) == 1
    assert isinstance(markers[0]["data"], str) and markers[0]["data"]
    # Head and tail survive as items (head bodies may be elided, but the
    # items themselves are present in order).
    assert [it["type"] for it in out[:5]] == [it["type"] for it in traj[:5]]
    tail = [it for it in out if it.get("type") != "ElidedItems"][-20:]
    assert tail == traj[-20:]


def test_fit_does_not_elide_inside_protected_tail():
    traj = _ctraj(15)  # 30 items — tail protection covers the last 20
    budget = int(_size_tokens(traj) * 0.8)
    out = _fit(traj, budget)
    assert out[-20:] == traj[-20:]


def test_fit_leaves_legacy_string_items_untouched():
    traj = [_legacy_string_item(0), *_ctraj(30)]
    budget = int(_size_tokens(traj) * 0.5)
    out = _fit(traj, budget)
    assert out[0] == _legacy_string_item(0)


# ---------------------------------------------------------------------------
# 4. run_autopsy budget integration
# ---------------------------------------------------------------------------

def _minimal_task_annotation():
    from bird_interact_agents.eval.annotation_schema import (
        MetadataSufficiency,
        Provenance,
        TaskAnnotation,
    )

    return TaskAnnotation(
        instance_id="test_1",
        selected_database="testdb",
        annotated_by="test",
        annotated_at="2026-01-01",
        amb_user_query="How many rows?",
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="test"
        ),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="test_1",
        ),
    )


@pytest.mark.asyncio
async def test_run_autopsy_squeezes_prompt_to_model_window(
    tmp_path, monkeypatch,
):
    """With a small context window, the prompt actually sent to the API
    must fit window*0.75 - 4096 tokens and carry elision markers."""
    from bird_interact_agents.eval import autopsy as autopsy_mod
    from bird_interact_agents.eval.autopsy import _estimate_tokens, run_autopsy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        autopsy_mod, "context_window_for", lambda model: 40_000,
    )

    tool_input = {
        "pattern": "other",
        "other_details": "x",
        "narrative": "n",
        "remediation": "r",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
    }
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = tool_input
    mock_response = MagicMock()
    mock_response.content = [mock_tool]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    big_traj = _traj(60, body_chars=4000)  # far over the 40K-token window

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=_minimal_task_annotation(),
            trajectory=big_traj,
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )

    assert result.error is None
    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0][
        "content"
    ]
    budget = int(40_000 * 0.75) - 4096
    assert _estimate_tokens(sent_prompt) <= budget
    assert "[tool result elided:" in sent_prompt
    assert "[tool_use_result:" in sent_prompt


@pytest.mark.asyncio
async def test_run_autopsy_small_trajectory_not_squeezed(tmp_path, monkeypatch):
    """Under-budget trajectories must reach the prompt without elision."""
    from bird_interact_agents.eval.autopsy import run_autopsy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    tool_input = {
        "pattern": "other",
        "other_details": "x",
        "narrative": "n",
        "remediation": "r",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
    }
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = tool_input
    mock_response = MagicMock()
    mock_response.content = [mock_tool]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=_minimal_task_annotation(),
            trajectory=_traj(2, body_chars=100),
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )

    assert result.error is None
    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert "[tool result elided:" not in sent_prompt
