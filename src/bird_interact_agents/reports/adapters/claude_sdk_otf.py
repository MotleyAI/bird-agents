"""Claude Agent SDK trajectory walker (claude_sdk_otf_* family).

Consumes the SDK-native message stream
(``SystemMessage`` / ``AssistantMessage`` / ``UserMessage`` /
``ResultMessage``) and yields one ``Turn`` per ``tool_use`` block. Pure-
text / thinking assistant messages without tool_use fold into the NEXT
tool-using turn so the leaderboard never sees no-op rows.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from bird_interact_agents.reports.adapters.base import Turn


def _normalize_content_to_text(content: Any) -> str:
    """Collapse mixed Anthropic-SDK content shapes into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(json.dumps(item, separators=(",", ":")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    if isinstance(content, dict):
        return json.dumps(content, separators=(",", ":"))
    return str(content)


def _render_response(
    pending_thinking: str,
    pending_text: str,
    tool_use: dict[str, Any],
) -> str:
    """Render the assistant message contents as a JSON list of content
    items. Tests can strip ``"thinking"``-typed items from this JSON
    string at output time for the ``--no-thinking`` flag — the literal
    token ``"thinking"`` therefore appears iff a thinking block survives.
    """
    items: list[dict[str, Any]] = []
    if pending_thinking:
        items.append({"type": "thinking", "thinking": pending_thinking})
    if pending_text:
        items.append({"type": "text", "text": pending_text})
    items.append(
        {
            "type": "tool_use",
            "id": tool_use.get("id", ""),
            "name": tool_use.get("name", ""),
            "input": tool_use.get("input", {}),
        }
    )
    return json.dumps(items, separators=(",", ":"))


def _is_tool_use(block: dict[str, Any]) -> bool:
    return block.get("type") == "tool_use" or (
        "name" in block and "input" in block and "id" in block
    )


def walk_trajectory(steps: list[dict[str, Any]]) -> Iterable[Turn]:
    """Yield one ``Turn`` per assistant tool_use block.

    Pass 1 builds the ``tool_use_id -> tool_result_text`` map; pass 2
    walks the trajectory in order, accumulating prompt/thinking/text
    buffers and emitting a Turn at every tool_use.
    """
    # Pass 1: build the tool_use_id → observation map. tool_result
    # content can arrive as a string, a list of text blocks, or a bare
    # list of strings — _normalize_content_to_text handles all three.
    tool_result_by_id: dict[str, str] = {}
    for msg in steps:
        if msg.get("type") != "UserMessage":
            continue
        data = msg.get("data") or {}
        for c in data.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                tool_result_by_id[c.get("tool_use_id", "")] = (
                    _normalize_content_to_text(c.get("content"))
                )

    # Pass 2.
    prompt_buf: list[str] = []
    pending_thinking = ""
    pending_text = ""
    last_model = ""

    def _append_prompt(s: str) -> None:
        if not s:
            return
        if prompt_buf and not prompt_buf[-1].endswith("\n"):
            prompt_buf.append("\n")
        prompt_buf.append(s)

    for msg in steps:
        t = msg.get("type")
        data = msg.get("data") or {}
        if t == "UserMessage":
            for c in data.get("content") or []:
                if not isinstance(c, dict):
                    _append_prompt(str(c))
                    continue
                kind = c.get("type")
                if kind == "tool_result":
                    _append_prompt(
                        _normalize_content_to_text(c.get("content"))
                    )
                elif kind == "text" or "text" in c:
                    _append_prompt(str(c.get("text", "")))
        elif t == "AssistantMessage":
            model = data.get("model") or last_model
            last_model = model or last_model
            for c in data.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if "thinking" in c and not _is_tool_use(c):
                    pending_thinking += str(c.get("thinking") or "")
                    continue
                if _is_tool_use(c):
                    tu_id = str(c.get("id", ""))
                    observation = tool_result_by_id.get(tu_id, "")
                    yield Turn(
                        model=last_model,
                        prompt="".join(prompt_buf).rstrip(),
                        response_raw=_render_response(
                            pending_thinking, pending_text, c
                        ),
                        tool_name=str(c.get("name", "")),
                        tool_input=dict(c.get("input") or {}),
                        tool_use_id=tu_id,
                        observation=observation,
                    )
                    prompt_buf = []
                    pending_thinking = ""
                    pending_text = ""
                elif "text" in c:
                    pending_text += str(c.get("text") or "")
        # SystemMessage / ResultMessage / other → skip silently.
