"""Tests for the DEV-1550 A2 plumbing: ``_build_one`` /
``encode_kb_as_memories`` thread ``row["description"]`` into the encoded
memory dict so SLayer 0.7.3's compact-mode renderer picks up the clean
one-line summary instead of falling back to first-paragraph-of-learning.

Scope: ``_build_one`` (via ``encode_kb_as_memories``) only. The companion
re-save site in ``slayer_otf.reference_build._annotate_memories`` is
covered by ``tests/test_pydantic_ai_otf_encode_memory_annotation.py``.

The encoder remains a pure function: ``(db, kb_rows, deleted_kb_ids)`` ->
list of dicts. The tests below assert on those dicts directly; one case
also round-trips through ``Memory.model_validate`` so we pin the
blank-vs-missing-vs-oversized semantics SLayer 0.7.3 owns.
"""

from __future__ import annotations

from typing import Any, Union

import pytest
from pydantic import ValidationError
from slayer.memories.models import MEMORY_DESCRIPTION_MAX_CHARS, Memory

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)


DB = "tinydb"


def _kb(
    kb_id: int,
    knowledge: str = "knowledge text",
    children_knowledge: Union[int, list, None] = -1,
    description: Any = "desc",
    definition: str = "def",
    type_: str = "calculation_knowledge",
    include_description: bool = True,
) -> dict[str, Any]:
    """Mirror the encoder-test helper but with explicit description control."""
    row: dict[str, Any] = {
        "id": kb_id,
        "knowledge": knowledge,
        "definition": definition,
        "type": type_,
        "children_knowledge": children_knowledge,
    }
    if include_description:
        row["description"] = description
    return row


def _one(row: dict[str, Any]) -> dict[str, Any]:
    mems = encode_kb_as_memories(DB, [row], deleted_kb_ids=set())
    assert len(mems) == 1
    return mems[0]


# ---------------------------------------------------------------------------
# A2.1 — `_build_one` threads `description` into the encoded dict.
# ---------------------------------------------------------------------------


def test_description_populated_from_row_happy_path():
    """Happy path: KB row's ``description`` survives into the encoded
    memory dict unchanged."""
    summary = "Explains the market classification system"
    row = _kb(1, description=summary)
    mem = _one(row)
    assert mem["description"] == summary


def test_description_defaults_to_empty_string_when_missing_key():
    """Defensive fallback: a row that LACKS the ``description`` key must
    yield ``description == ""`` in the encoded dict (NEVER a ``KeyError``).

    SLayer 0.7.3's ``Memory._normalise_description`` will coerce that empty
    string to ``None`` at load time, which is the deliberate fallback path
    to first-paragraph-of-``learning``. The encoder's job is just to avoid
    blowing up here.
    """
    row = _kb(2, include_description=False)
    mem = _one(row)
    assert mem["description"] == ""


def test_description_is_independent_of_learning_body():
    """``description`` and ``learning`` are different surfaces; populating
    ``description`` MUST NOT replace, mangle, or truncate the verbatim
    ``learning`` body (which the agent needs for full drill-in).

    Note: the verbatim block inside ``learning`` is a YAML dump of the
    entire KB row, which by design echoes the ``description`` field — so
    we DON'T assert "summary not in learning"; we assert "learning is not
    *replaced by* description and the structural anchors survive".
    """
    summary = "one-line summary"
    knowledge = "the long-form knowledge sentence the KB row carries"
    row = _kb(3, knowledge=knowledge, description=summary)
    mem = _one(row)

    assert mem["description"] == summary
    # learning was not silently replaced by description (top-level shape).
    assert mem["learning"] != summary
    # The KB header (``KB N — knowledge``) is the contract HARD-8 parsing
    # depends on and must remain intact.
    assert mem["learning"].startswith(f"KB 3 — {knowledge}\n\n")
    # The verbatim block trailer (used by reference_build's annotation
    # ordering check) is still present.
    assert "KB item (verbatim from tinydb_kb.jsonl):" in mem["learning"]
    # The verbatim block echoes the ``description`` field by design (YAML
    # dump of the row), so we'd see ``summary`` appear inside ``learning``
    # — that's expected and NOT a regression.


def test_other_dict_keys_unchanged_no_regression():
    """Every key the pre-DEV-1550 shape carried (``version``, ``id``,
    ``learning``, ``entities``, ``query``, ``created_at``) is still present
    and unchanged. ``description`` is the only new key."""
    row = _kb(4, description="x")
    mem = _one(row)
    expected_keys = {
        "version", "id", "learning", "description", "entities", "query",
        "created_at",
    }
    assert set(mem) == expected_keys
    assert mem["version"] == 1
    assert mem["id"] == f"{DB}_kb_4"
    assert mem["entities"] == [DB]
    assert mem["query"] is None
    # created_at is the encoder's deterministic EPOCH constant — sanity
    # check that A2 didn't disturb it.
    from bird_interact_agents.slayer_otf.kb_memory_encoder import EPOCH
    assert mem["created_at"] == EPOCH


# ---------------------------------------------------------------------------
# A2.1 — `Memory.model_validate` round-trip (codex Low #2).
# Locks the blank-vs-missing-vs-oversized semantics the encoder delegates
# to SLayer 0.7.3. These are NOT prompt-content tests; they pin the load-
# time behaviour the plan documents as "SLayer handles it".
# ---------------------------------------------------------------------------


def test_missing_description_round_trips_to_none_via_memory_validate():
    """missing key → encoded ``""`` → ``Memory.model_validate`` coerces to None."""
    row = _kb(10, include_description=False)
    mem_dict = _one(row)
    assert mem_dict["description"] == ""

    memory = Memory.model_validate(mem_dict)
    assert memory.description is None


def test_explicit_none_description_round_trips_to_none():
    """Explicit ``description=None`` in the row stays None through the
    encoder and through validation."""
    row = _kb(11, description=None)
    mem_dict = _one(row)
    assert mem_dict["description"] is None

    memory = Memory.model_validate(mem_dict)
    assert memory.description is None


def test_whitespace_description_coerced_to_none_at_validate():
    """Whitespace-only ``description`` ALSO falls through SLayer's
    normaliser to None — so the compact renderer falls back to
    first-paragraph-of-``learning`` rather than rendering empty space."""
    row = _kb(12, description="   \n  \t ")
    mem_dict = _one(row)
    # Encoder passes through verbatim (no strip).
    assert mem_dict["description"] == "   \n  \t "

    memory = Memory.model_validate(mem_dict)
    assert memory.description is None


def test_oversized_description_raises_at_validate():
    """A description above SLayer's 500-char hard cap raises
    ``ValidationError`` at YAML load time — fail-loud, never silent
    truncation. If a real bird-interact KB corpus ever trips this, it's a
    follow-up issue (truncation policy belongs in the encoder or SLayer,
    not silently here)."""
    oversized = "x" * (MEMORY_DESCRIPTION_MAX_CHARS + 1)
    row = _kb(13, description=oversized)
    mem_dict = _one(row)
    assert mem_dict["description"] == oversized  # encoder is pass-through

    with pytest.raises(ValidationError):
        Memory.model_validate(mem_dict)
