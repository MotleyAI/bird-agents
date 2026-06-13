"""DEV-1555 Stage 2: autopsy on open-weight backends.

1. `_build_anthropic_client(model)` becomes registry-aware: moonshot →
   AsyncAnthropic(base_url=<anthropic-format endpoint>, api_key=$MOONSHOT_API_KEY);
   anthropic models keep the OAuth→API-key env resolution.
2. Deterministic text-JSON fallback when the response has no tool_use block
   (third-party endpoints may not honor forced tool_choice): prefer a fenced
   ```json block, else the first balanced {...} that parses; schema-validate
   exactly once; validation failure → AutopsyError(kind="validation_error");
   no parseable candidate → kind="missing_tool_use".
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_KIMI = "moonshot/kimi-k2.7-code"

_VALID_ONE_SHOT_PAYLOAD = {
    "pattern": "other",
    "other_details": "x",
    "narrative": "n",
    "remediation": "r",
    "decision_point_trajectory_index": None,
    "decision_point_description": None,
}


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


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _mock_client(content_blocks):
    response = MagicMock()
    response.content = content_blocks
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


async def _run(monkeypatch, content_blocks):
    from bird_interact_agents.eval.autopsy import run_autopsy

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with patch("anthropic.AsyncAnthropic", return_value=_mock_client(content_blocks)):
        return await run_autopsy(
            task_annotation=_minimal_task_annotation(),
            trajectory=[{"type": "T", "data": "x"}],
            slayer_storage_dir="/nonexistent",
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )


# ---------------------------------------------------------------------------
# Text-JSON fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_use_block_still_preferred(monkeypatch):
    tool = MagicMock()
    tool.type = "tool_use"
    tool.name = "autopsy_output"
    tool.input = dict(_VALID_ONE_SHOT_PAYLOAD)
    result = await _run(monkeypatch, [tool])
    assert result.error is None
    assert result.analysis is not None


@pytest.mark.asyncio
async def test_fallback_parses_fenced_json_block(monkeypatch):
    text = (
        "Here is my analysis:\n```json\n"
        + json.dumps(_VALID_ONE_SHOT_PAYLOAD)
        + "\n```\nthanks"
    )
    result = await _run(monkeypatch, [_text_block(text)])
    assert result.error is None
    assert result.analysis is not None
    assert result.analysis.pattern == "other"


@pytest.mark.asyncio
async def test_fallback_parses_first_balanced_object(monkeypatch):
    text = "noise before " + json.dumps(_VALID_ONE_SHOT_PAYLOAD) + " noise after"
    result = await _run(monkeypatch, [_text_block(text)])
    assert result.error is None
    assert result.analysis is not None


@pytest.mark.asyncio
async def test_fallback_schema_invalid_candidate_is_validation_error(monkeypatch):
    bad = dict(_VALID_ONE_SHOT_PAYLOAD, pattern="not_a_real_pattern")
    result = await _run(
        monkeypatch, [_text_block("```json\n" + json.dumps(bad) + "\n```")],
    )
    assert result.analysis is None
    assert result.error is not None
    assert result.error.kind == "validation_error"


@pytest.mark.asyncio
async def test_fallback_no_parseable_candidate_is_missing_tool_use(monkeypatch):
    result = await _run(
        monkeypatch,
        [_text_block("no json here { definitely not json } end")],
    )
    assert result.analysis is None
    assert result.error is not None
    assert result.error.kind == "missing_tool_use"


@pytest.mark.asyncio
async def test_requires_thinking_model_gets_thinking_and_auto_tool_choice(
    monkeypatch, tmp_path,
):
    """Probed live (2026-06-12): kimi-k2.7-code requires thinking enabled,
    and forced tool_choice is incompatible with thinking — the autopsy
    request must switch to thinking + tool_choice=auto (the text-JSON
    fallback covers responses that skip the tool)."""
    from bird_interact_agents.eval.autopsy import run_autopsy

    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    tool = MagicMock()
    tool.type = "tool_use"
    tool.name = "autopsy_output"
    tool.input = dict(_VALID_ONE_SHOT_PAYLOAD)
    client = _mock_client([tool])

    with patch("anthropic.AsyncAnthropic", return_value=client):
        result = await run_autopsy(
            task_annotation=_minimal_task_annotation(),
            trajectory=[{"type": "T", "data": "x"}],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model=_KIMI,
            is_one_shot=True,
        )

    assert result.error is None
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "kimi-k2.7-code"
    assert kwargs["thinking"]["type"] == "enabled"
    assert kwargs["tool_choice"] == {"type": "auto"}
    assert kwargs["max_tokens"] > kwargs["thinking"]["budget_tokens"]


@pytest.mark.asyncio
async def test_anthropic_autopsy_request_shape_unchanged(monkeypatch, tmp_path):
    tool = MagicMock()
    tool.type = "tool_use"
    tool.name = "autopsy_output"
    tool.input = dict(_VALID_ONE_SHOT_PAYLOAD)
    client = _mock_client([tool])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    from bird_interact_agents.eval.autopsy import run_autopsy

    with patch("anthropic.AsyncAnthropic", return_value=client):
        await run_autopsy(
            task_annotation=_minimal_task_annotation(),
            trajectory=[{"type": "T", "data": "x"}],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    kwargs = client.messages.create.call_args.kwargs
    assert "thinking" not in kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "autopsy_output"}
    assert kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Registry-aware client construction
# ---------------------------------------------------------------------------

def test_build_client_moonshot_uses_registry(monkeypatch):
    from bird_interact_agents.eval.autopsy import _build_anthropic_client

    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.delenv("BIRD_MOONSHOT_ANTHROPIC_BASE_URL", raising=False)
    # Ambient anthropic creds must not interfere with provider routing.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-be-used")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")

    client = _build_anthropic_client(_KIMI)
    assert str(client.base_url).rstrip("/") == "https://api.moonshot.ai/anthropic"
    assert client.api_key == "ms-key-1"


def test_build_client_moonshot_missing_key_raises(monkeypatch):
    from bird_interact_agents.eval.autopsy import _build_anthropic_client

    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOONSHOT_API_KEY"):
        _build_anthropic_client(_KIMI)


def test_build_client_anthropic_resolution_unchanged(monkeypatch):
    from bird_interact_agents.eval.autopsy import _build_anthropic_client

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-key")
    client = _build_anthropic_client("anthropic/claude-sonnet-4-5")
    assert client.api_key == "anth-key"

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-y")
    client = _build_anthropic_client("anthropic/claude-sonnet-4-5")
    assert client.auth_token == "sk-ant-oat01-y"
