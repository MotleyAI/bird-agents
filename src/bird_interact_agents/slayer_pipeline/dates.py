"""LLM TEXT-as-date detection (phase 4 of regenerate_slayer_model.py).

For every Column whose type is still ``TEXT`` after the deterministic
phase-2 overlay, sample up to ``SAMPLE_VALUES_PER_COLUMN`` distinct
non-NULL values from SQLite and ask the LLM to classify whether the
column is date-valued and (if so) what the source strftime format is.

On a confident, consistent response we retype the column to
``TIMESTAMP``, rewrite ``Column.sql`` to the SQLite-native parse
expression for the detected format, and cache the inference in
``Column.meta`` (``date_source_format``, ``detected_by='llm'``) for
idempotency. Re-runs skip any column that already carries
``meta.date_source_format``.

On low confidence or mixed formats, the column stays TEXT and a line
is appended to the caller's warning sink for human review.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from anthropic.types import TextBlock
from pydantic import BaseModel, Field

from slayer.core.models import DataType, SlayerModel

from .overlay import ISO_TEXT_DATE_FORMATS, _sqlite_reformat_sql

SAMPLE_VALUES_PER_COLUMN = 20
CONFIDENCE_THRESHOLD = 0.8
MAX_TOKENS = 256


class _DateLLMResponse(BaseModel):
    is_date: bool
    source_format: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None


def sample_text_values(
    sqlite_path: Path, table: str, col: str
) -> list[str]:
    """Return up to ``SAMPLE_VALUES_PER_COLUMN`` distinct non-NULL,
    non-empty values from *table.col* (sqlite). Returns ``[]`` on any
    error (the column may have been renamed, or the table may not
    exist yet)."""
    if not sqlite_path.exists():
        return []
    try:
        # immutable=1 — mini-interact dirs aren't sandbox-writable, so
        # SQLite can't create a journal there.
        conn = sqlite3.connect(f"file:{sqlite_path}?immutable=1", uri=True)
    except sqlite3.Error:
        return []
    try:
        cur = conn.execute(
            f'SELECT DISTINCT "{col}" FROM "{table}" '
            f'WHERE "{col}" IS NOT NULL AND "{col}" != \'\' '
            f'LIMIT {SAMPLE_VALUES_PER_COLUMN}'
        )
        return [str(row[0]) for row in cur if row and row[0] is not None]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


_PROMPT_TEMPLATE = """\
You are given a SQL table column and a sample of its non-NULL values.
Classify whether the column holds date or timestamp values stored as text.

Column qualifier: {qualifier}
Sample values (distinct, up to {n}):
{values}

Return JSON ONLY with this shape, no prose:

{{"is_date": true|false, "source_format": "<strftime>", "confidence": 0.0-1.0, "notes": "..."}}

Rules:
- If the column is NOT date/timestamp-valued, return {{"is_date": false}}.
- If it IS, "source_format" MUST be a Python strftime string that round-trips
  EVERY sampled value via datetime.strptime. If the samples mix formats,
  return is_date=true with confidence < 0.5 and source_format=null.
- "confidence" reflects how unambiguously the samples fit a single format.
- Prefer ISO when ambiguous; flag day-month-vs-month-day ambiguity in "notes".
"""


def _llm_classify(
    client: Anthropic,
    model: str,
    qualifier: str,
    samples: list[str],
) -> _DateLLMResponse:
    prompt = _PROMPT_TEMPLATE.format(
        qualifier=qualifier,
        n=SAMPLE_VALUES_PER_COLUMN,
        values="\n".join(f"  - {v!r}" for v in samples),
    )
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    ).strip()
    # Strip markdown fences if the model adds them.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    parsed_json = json.loads(_first_json_object(text))
    return _DateLLMResponse.model_validate(parsed_json)


def _first_json_object(text: str) -> str:
    """Return the substring of *text* that is the first balanced ``{...}``
    JSON object, ignoring anything before/after. Lets us tolerate
    trailing prose the LLM sometimes appends despite instructions.
    """
    start = text.find("{")
    if start == -1:
        return text  # let json.loads raise the original error
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _validate_format_against_samples(fmt: str, samples: list[str]) -> bool:
    """Confirm EVERY sample parses under ``fmt`` via ``datetime.strptime``.

    Belt-and-braces over the LLM: if the model said "%d/%m/%Y" but the
    samples are actually "%m/%d/%Y", at least one will throw. We then
    refuse the conversion and leave the column TEXT.
    """
    from datetime import datetime

    if not samples:
        return False
    try:
        for value in samples:
            datetime.strptime(value, fmt)
    except ValueError:
        return False
    return True


def detect_and_apply(
    model: SlayerModel,
    sqlite_path: Path,
    client: Anthropic,
    llm_model: str,
) -> tuple[int, list[str]]:
    """For each TEXT column in *model* with no ``meta.date_source_format``
    cached, sample-and-classify. Returns ``(num_retyped, warnings)``.
    """
    table = (model.sql_table or model.name)
    retyped = 0
    warnings: list[str] = []
    for column in model.columns:
        if column.type != DataType.TEXT:
            continue
        if (column.meta or {}).get("date_source_format"):
            continue
        if (column.meta or {}).get("derived_from"):
            # JSONB-leaf column: live-data sampling via JSON_EXTRACT
            # is not implemented yet; skip.
            continue
        samples = sample_text_values(sqlite_path, table, column.name)
        if len(samples) < 3:
            continue
        try:
            decision = _llm_classify(
                client=client,
                model=llm_model,
                qualifier=f"{table}.{column.name}",
                samples=samples,
            )
        except Exception as exc:
            warnings.append(
                f"{table}.{column.name}: LLM date-classify failed "
                f"({exc.__class__.__name__}: {exc}); column left TEXT."
            )
            continue
        if not decision.is_date:
            continue
        if decision.confidence < CONFIDENCE_THRESHOLD:
            warnings.append(
                f"{table}.{column.name}: LLM said date but confidence "
                f"{decision.confidence:.2f} < {CONFIDENCE_THRESHOLD}; "
                f"column left TEXT. notes={decision.notes!r}"
            )
            continue
        fmt = decision.source_format
        if not fmt:
            warnings.append(
                f"{table}.{column.name}: LLM said date but no "
                f"source_format returned; column left TEXT."
            )
            continue
        if not _validate_format_against_samples(fmt, samples):
            warnings.append(
                f"{table}.{column.name}: LLM-proposed format '{fmt}' "
                f"failed strptime on at least one sample; column left TEXT."
            )
            continue
        new_sql = _sqlite_reformat_sql(column.name, fmt)
        if new_sql is None and fmt not in ISO_TEXT_DATE_FORMATS:
            warnings.append(
                f"{table}.{column.name}: LLM-proposed format '{fmt}' is "
                f"not supported by the SQLite rewrite path; column left TEXT."
            )
            continue
        column.type = DataType.TIMESTAMP
        if new_sql is not None:
            column.sql = new_sql
        meta = column.meta or {}
        meta["date_source_format"] = fmt
        meta["detected_by"] = "llm"
        column.meta = meta
        retyped += 1
    return retyped, warnings


def make_anthropic_client() -> Anthropic:
    """Construct an Anthropic client from the ambient environment.

    The Anthropic SDK reads ``ANTHROPIC_API_KEY`` automatically. We
    centralise construction here so tests can monkey-patch this.
    """
    return Anthropic()
