"""SAR-Agent audit loop driven against the Anthropic Messages API directly.

Reuses upstream's prompt template (`prompts/prompt_user_bird.txt`) and tool
schemas (`db_interface.get_function_call_bird()`), but drives the loop here
so we don't need to monkey-patch upstream's openai-tinted LLMInterface.

The loop:
1. Send the wrapped prompt as the initial user message.
2. Anthropic returns either text or tool_use blocks.
3. For each tool_use: execute the tool (read_sqlite_query against our DB,
   or `terminate` which ends the loop with the analyse_result text).
4. Cap at `max_steps` (matching upstream's 30-step budget).
5. Parse the final analyse_result text into a SARVerdict.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel

from .normalizer import SARVerdict


# Anthropic tool schema for read_sqlite_query (mapped from upstream's
# OpenAI-style schema in db_interface.get_function_call_bird()).
_READ_SQLITE_TOOL = {
    "name": "read_sqlite_query",
    "description": "Read the result of a sqlite query (read-only, results capped at 50 rows).",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The sqlite query to execute."},
            "explaination": {
                "type": "string",
                "description": "Why this query is being executed.",
            },
        },
        "required": ["query", "explaination"],
    },
}

_TERMINATE_TOOL = {
    "name": "terminate",
    "description": "Call when audit analysis is complete. Pass the final analyse_result block.",
    "input_schema": {
        "type": "object",
        "properties": {
            "analyze_result": {
                "type": "string",
                "description": (
                    "Final analysis using the format:\n"
                    "Correctness: Yes/No, brief reason\n"
                    "Is_ambiguous: Yes/No, brief reason\n"
                    "Explanation: detailed reasoning\n"
                    "Revised: revised SQL and/or question, if needed"
                ),
            }
        },
        "required": ["analyze_result"],
    },
}

ANTHROPIC_TOOLS = [_READ_SQLITE_TOOL, _TERMINATE_TOOL]


class SARRunResult(BaseModel):
    verdict: SARVerdict
    step_count: int
    cost_usd: float | None = None
    audit_model_actual: str | None = None
    raw_trajectory: list | None = None


class SARAuditLoop:
    """One audit attempt for a single task. Returns a SARRunResult."""

    def __init__(
        self,
        *,
        model: str,
        prompt: str,
        api_key: str | None,
        db_path: Path,
    ):
        self.model = model
        self.prompt = prompt
        self.db_path = db_path
        self.client = (
            anthropic.Anthropic(api_key=api_key)
            if api_key is not None
            else anthropic.Anthropic()
        )

    def run(self, *, max_steps: int) -> SARRunResult:
        messages: list[dict] = [{"role": "user", "content": self.prompt}]
        trajectory: list[dict] = []
        terminal_text: str | None = None
        actual_model: str | None = None
        total_input_tokens = 0
        total_output_tokens = 0
        step_count = 0

        for step in range(max_steps):
            step_count = step + 1
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                tools=ANTHROPIC_TOOLS,
                max_tokens=4096,
            )
            actual_model = getattr(response, "model", actual_model)
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_input_tokens += getattr(usage, "input_tokens", 0)
                total_output_tokens += getattr(usage, "output_tokens", 0)

            content = response.content
            trajectory.append({"step": step + 1, "stop_reason": response.stop_reason, "blocks": _summarise_blocks(content)})

            # Append assistant turn to messages.
            messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in content]})

            tool_uses = [b for b in content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                # No tool call → model is rambling. Push it back to call terminate.
                if response.stop_reason == "end_turn":
                    text_block = next(
                        (b for b in content if getattr(b, "type", None) == "text"),
                        None,
                    )
                    if text_block is not None:
                        terminal_text = getattr(text_block, "text", "")
                        break
                messages.append(
                    {
                        "role": "user",
                        "content": "Please call read_sqlite_query or terminate next.",
                    }
                )
                continue

            tool_results: list[dict] = []
            for tu in tool_uses:
                name = getattr(tu, "name", "")
                inp = getattr(tu, "input", {}) or {}
                tu_id = getattr(tu, "id", "")
                if name == "terminate":
                    terminal_text = str(inp.get("analyze_result", ""))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": "Terminated.",
                        }
                    )
                elif name == "read_sqlite_query":
                    sql = str(inp.get("query", ""))
                    result_str = self._run_read_only(sql)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": result_str,
                        }
                    )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu_id,
                            "content": f"Unknown tool: {name}",
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

            if terminal_text is not None:
                break

        if terminal_text is None:
            # Model exhausted `max_steps` without calling `terminate`.
            # Don't silently mark the audit clean — surface as a task
            # failure so the driver routes it to the failures sidecar.
            raise RuntimeError(
                f"SARAuditLoop exhausted max_steps={max_steps} without a terminate call"
            )

        verdict = parse_verdict(terminal_text)
        return SARRunResult(
            verdict=verdict,
            step_count=step_count,
            cost_usd=_estimate_cost(
                model=self.model,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
            audit_model_actual=actual_model,
            raw_trajectory=trajectory,
        )

    def _run_read_only(self, sql: str) -> str:
        try:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro&immutable=1", uri=True)
        except sqlite3.Error as exc:
            return f"Error opening DB: {exc}"
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(50)
        except sqlite3.Error as exc:
            return f"Error: {exc}"
        finally:
            con.close()
        if not rows:
            return "Result: (no rows)"
        return "Result: " + str(rows)


def _block_to_dict(block: Any) -> dict:
    """Convert an anthropic content block to a serialisable dict for the
    next request's `messages[].content`."""
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}),
        }
    if btype == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", ""),
            "content": getattr(block, "content", ""),
        }
    return {"type": btype or "unknown"}


def _summarise_blocks(content: Any) -> list[dict]:
    out: list[dict] = []
    for b in content:
        btype = getattr(b, "type", None)
        if btype == "text":
            text = getattr(b, "text", "")
            out.append({"type": "text", "preview": text[:200]})
        elif btype == "tool_use":
            out.append({"type": "tool_use", "name": getattr(b, "name", "")})
        else:
            out.append({"type": btype or "unknown"})
    return out


def parse_verdict(text: str) -> SARVerdict:
    """Parse the upstream `Correctness/Is_ambiguous/Explanation/Revised` format
    into a SARVerdict.

    Fail-closed defaults: when the verdict text is empty / malformed /
    missing the `Correctness:` header, we return `correctness_flag=False`
    so the audit is recorded as `unrecoverable` rather than silently
    marked clean. `ambiguity_flag` defaults to `False` (we don't claim
    ambiguity we didn't observe)."""
    sections = _split_sections(text)

    correctness_str = sections.get("correctness", "")
    ambiguity_str = sections.get("is_ambiguous", "")
    explanation = sections.get("explanation", "")
    revised = sections.get("revised", "")

    correctness_flag = _parse_yes_no(correctness_str, default=False)
    ambiguity_flag = _parse_yes_no(ambiguity_str, default=False)
    revised_sql, revised_question = _split_revised(revised)

    return SARVerdict(
        correctness_flag=correctness_flag,
        ambiguity_flag=ambiguity_flag,
        revised_sql=revised_sql,
        revised_question=revised_question,
        reasoning=explanation.strip() or text.strip(),
    )


_SECTION_HEADERS = ("correctness", "is_ambiguous", "explanation", "revised")


def _split_sections(text: str) -> dict[str, str]:
    """Split the analyse_result into header-keyed sections."""
    out: dict[str, str] = {}
    pattern = re.compile(
        r"^(Correctness|Is_ambiguous|Explanation|Revised)\s*:\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[start:end].strip()
    return out


def _parse_yes_no(blob: str, *, default: bool) -> bool:
    """Look for an explicit Yes/No token at the start of the section value."""
    if not blob:
        return default
    first = blob.strip().split(None, 1)[0].rstrip(",.;").lower() if blob.strip() else ""
    if first.startswith("yes"):
        return True
    if first.startswith("no"):
        return False
    return default


_REVISED_LABEL_RE = re.compile(
    r"^\s*(?:Revised\s+(?:SQL|Query|SQL\s+query)|SQL|Query)\s*:\s*",
    re.IGNORECASE,
)
# Require SELECT/WITH to appear at a clause boundary (start-of-string,
# newline, or after `.`/`;`/`:`) so prose like "select the option" or
# "with the hint" doesn't get parsed as SQL.
_SQL_START_RE = re.compile(r"(?:^|[\n.;:]\s*)(SELECT|WITH)\b", re.IGNORECASE)


def _split_revised(blob: str) -> tuple[str | None, str | None]:
    """Heuristic split of the Revised section into a SQL string and a
    rephrased-question string. Handles three forms:

    1. Fenced block: ```sql … ``` (any preceding/trailing prose → question).
    2. Prose-prefixed: "Revised SQL: WITH …" — strip the label, treat as SQL.
    3. Bare SELECT/WITH anywhere — split into (pre-question, SQL).
    """
    if not blob.strip():
        return None, None
    text = blob.strip()

    # Form 1: fenced SQL block.
    fence_match = re.search(r"```(?:sql)?\s*(.+?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        sql_candidate = fence_match.group(1).strip() or None
        remainder = (text[: fence_match.start()] + text[fence_match.end():]).strip()
        question_candidate = remainder if remainder else None
        return sql_candidate, question_candidate

    # Form 2: strip a "Revised SQL:" / "SQL:" / "Query:" prefix if present.
    stripped = _REVISED_LABEL_RE.sub("", text, count=1).strip()
    if stripped != text:
        # Label was present → the whole remainder is SQL.
        return (stripped or None), None

    # Form 3: SELECT/WITH appears at a clause boundary somewhere in the body.
    sql_match = _SQL_START_RE.search(text)
    if sql_match:
        sql_start = sql_match.start(1)  # the keyword group, not the boundary
        pre = text[:sql_start].strip()
        sql_part = text[sql_start:].strip()
        # Trim trailing prose after the SQL (rare but possible). For now,
        # take everything from SELECT/WITH to end — the executor will
        # validate via sqlglot at run time.
        question = pre if pre else None
        return sql_part, question

    # No SQL detected → all prose, treat as question revision.
    return None, text


# Anthropic pricing (input/output USD per 1M tokens). Best-effort — keys
# match the model id Anthropic returns in responses.
_PRICING_USD_PER_M = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _estimate_cost(*, model: str, input_tokens: int, output_tokens: int) -> float | None:
    base = model.split("-2")[0] if "-2" in model else model
    pricing = None
    for k, v in _PRICING_USD_PER_M.items():
        if base.startswith(k):
            pricing = v
            break
    if pricing is None:
        return None
    in_cost = pricing[0] * input_tokens / 1_000_000
    out_cost = pricing[1] * output_tokens / 1_000_000
    return round(in_cost + out_cost, 6)
