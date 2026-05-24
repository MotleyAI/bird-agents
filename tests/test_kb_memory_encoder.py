"""Tests for ``slayer_otf.kb_memory_encoder.encode_kb_as_memories``.

The encoder is a pure function: ``(db, kb_rows, deleted_kb_ids)`` →
list of dicts ready to dump into ``memories.yaml``. Tests exercise
body format, cross-ref representation, datasource-eligibility entity,
deletion semantics, defensive ``children_knowledge`` normalisation,
and determinism.

No I/O. No fixtures beyond bare dicts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import pytest
import yaml

from slayer.memories.models import Memory, is_valid_memory_id

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    EPOCH,
    encode_kb_as_memories,
)


DB = "tinydb"


def _kb(
    kb_id: int,
    knowledge: str,
    children_knowledge: Union[int, list, None] = -1,
    description: str = "desc",
    definition: str = "def",
    type_: str = "calculation_knowledge",
) -> dict[str, Any]:
    return {
        "id": kb_id,
        "knowledge": knowledge,
        "description": description,
        "definition": definition,
        "type": type_,
        "children_knowledge": children_knowledge,
    }


# ---------------------------------------------------------------------------
# Happy path + cross-ref shape
# ---------------------------------------------------------------------------


def test_happy_path_three_kbs_b_references_c():
    """B → C cross-ref ends up as ``memory:<db>_kb_<C_id>`` in B's entities.

    Independent / unreferenced KBs (A, C here) get only the bare datasource
    entity, no cross-refs.
    """
    a = _kb(1, "A title", children_knowledge=[])
    b = _kb(2, "B title", children_knowledge=[3])
    c = _kb(3, "C title", children_knowledge=-1)

    mems = encode_kb_as_memories(DB, [a, b, c], deleted_kb_ids=set())

    by_id = {m["id"]: m for m in mems}
    assert set(by_id) == {f"{DB}_kb_1", f"{DB}_kb_2", f"{DB}_kb_3"}

    assert by_id[f"{DB}_kb_2"]["entities"] == [DB, f"memory:{DB}_kb_3"]
    assert by_id[f"{DB}_kb_1"]["entities"] == [DB]
    assert by_id[f"{DB}_kb_3"]["entities"] == [DB]


def test_bare_datasource_is_always_first_entity():
    """Every KB memory must carry the bare datasource id as ``entities[0]``
    so it survives ``SearchService._filter_memories_by_datasource``."""
    rows = [
        _kb(1, "first", children_knowledge=-1),
        _kb(2, "second", children_knowledge=[1]),
        _kb(3, "third", children_knowledge=[]),
        _kb(4, "fourth", children_knowledge=None),
    ]
    mems = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    for m in mems:
        assert m["entities"][0] == DB, (
            f"memory {m['id']} must lead with bare datasource entity"
        )


def test_memory_id_is_db_prefixed_with_kb_id():
    """Memory id = ``f'{db}_kb_{kb_id}'``. KB ids are per-DB; the prefix
    keeps cross-DB collisions away."""
    rows = [_kb(0, "zero"), _kb(42, "forty-two")]
    mems = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    ids = sorted(m["id"] for m in mems)
    assert ids == [f"{DB}_kb_0", f"{DB}_kb_42"]


def test_memory_body_starts_with_kb_prefix_marker():
    """``KB <id> — `` (em-dash U+2014) marker is the contract shared with
    ``hard8_preprocessor._KB_PREFIX_RE``. Body must lead with it."""
    rows = [_kb(7, "Pole Position")]
    [m] = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    assert m["learning"].startswith("KB 7 — Pole Position\n\n"), (
        f"body should start with the KB prefix marker, got: {m['learning'][:60]!r}"
    )


def test_memory_body_contains_verbatim_kb_fields():
    """Body must contain all six KB fields verbatim — id + knowledge +
    description + definition + type + children_knowledge — under a
    header that names the source file, so the agent (and semantic
    search) can reason about the original row without paraphrase."""
    row = _kb(
        13,
        "Signal-to-Noise Quality Indicator (SNQI)",
        children_knowledge=[1, 2, 3],
        description="Combines SNR and noise floor into a single quality metric.",
        definition="SNQI = SnrRatio - 0.1 * |NoiseFloorDbm|",
        type_="calculation_knowledge",
    )
    [m] = encode_kb_as_memories(DB, [row], deleted_kb_ids=set())
    body = m["learning"]
    # Verbatim-from header naming the source file.
    assert f"KB item (verbatim from {DB}_kb.jsonl):" in body, (
        f"body should advertise where the verbatim block came from; got "
        f"first 200 chars: {body[:200]!r}"
    )
    # All field values present.
    assert "Signal-to-Noise Quality Indicator (SNQI)" in body
    assert "Combines SNR and noise floor into a single quality metric." in body
    assert "SNQI = SnrRatio - 0.1 * |NoiseFloorDbm|" in body
    assert "calculation_knowledge" in body
    # Id appears in the verbatim block, not only in the prefix marker.
    # Use a YAML-looking match so a future change that drops `id:` from
    # the dump (and only keeps it in the prefix) is caught.
    assert "id: 13" in body
    # children list appears as YAML — match the leading dash form most
    # likely to be emitted by safe_dump.
    assert "- 1" in body and "- 2" in body and "- 3" in body


def test_pipe_format_knowledge_preserved_in_body():
    """The 10 cybermarket-style ``db|table|column`` knowledge strings
    survive verbatim in the memory body (no extra entity attach — semantic
    search hits the string in the body)."""
    row = _kb(0, "cybermarket|markets|mktclass")
    [m] = encode_kb_as_memories(DB, [row], deleted_kb_ids=set())
    assert "cybermarket|markets|mktclass" in m["learning"]
    # And NO extra entity link beyond the bare db (option locked in by user).
    assert m["entities"] == [DB]


# ---------------------------------------------------------------------------
# children_knowledge — defensive normalisation
# ---------------------------------------------------------------------------


def test_children_knowledge_minus_one_yields_no_cross_refs():
    """The ``-1`` sentinel (679/1621 corpus rows) means 'no children'."""
    rows = [_kb(1, "lonely", children_knowledge=-1)]
    [m] = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    assert m["entities"] == [DB]


def test_children_knowledge_empty_list_yields_no_cross_refs():
    rows = [_kb(1, "lonely", children_knowledge=[])]
    [m] = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    assert m["entities"] == [DB]


def test_children_knowledge_missing_or_none_yields_no_cross_refs():
    """Defensive: future schema drift shouldn't crash the encoder."""
    row_missing = {
        "id": 1, "knowledge": "k", "description": "d", "definition": "def",
        "type": "x",
    }
    row_none = {**row_missing, "id": 2, "children_knowledge": None}
    mems = encode_kb_as_memories(DB, [row_missing, row_none], deleted_kb_ids=set())
    for m in mems:
        assert m["entities"] == [DB]


def test_children_knowledge_as_positive_int_normalised():
    """A future row using a positive int (instead of list[int]) is
    treated as a single-element list. Today the corpus never emits this,
    but the corpus audit found no rule against it."""
    rows = [_kb(1, "ref-int", children_knowledge=5), _kb(5, "target")]
    mems = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    by_id = {m["id"]: m for m in mems}
    assert by_id[f"{DB}_kb_1"]["entities"] == [DB, f"memory:{DB}_kb_5"]


def test_dangling_child_id_dropped_and_warned(caplog):
    """A child id absent from kb_rows AND from deleted_kb_ids is dropped
    from the encoded entities list (keeps the YAML valid for SLayer's
    strict resolver) AND triggers a ``WARNING`` log so the drift is
    visible to whoever is encoding the corpus."""
    rows = [_kb(1, "refers-bogus", children_knowledge=[2, 999])]
    rows.append(_kb(2, "real target"))
    with caplog.at_level(logging.WARNING):
        mems = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    by_id = {m["id"]: m for m in mems}
    assert by_id[f"{DB}_kb_1"]["entities"] == [DB, f"memory:{DB}_kb_2"]
    # Bogus 999 is gone from entities.
    assert f"memory:{DB}_kb_999" not in by_id[f"{DB}_kb_1"]["entities"]
    # And the encoder warned about it.
    assert any("999" in rec.getMessage() for rec in caplog.records), (
        f"expected a warning mentioning the dangling child id 999; "
        f"got: {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Deletion semantics
# ---------------------------------------------------------------------------


def test_deletion_drops_the_memory_itself():
    a = _kb(1, "A")
    b = _kb(2, "B")
    c = _kb(3, "C")
    mems = encode_kb_as_memories(DB, [a, b, c], deleted_kb_ids={2})
    ids = {m["id"] for m in mems}
    assert ids == {f"{DB}_kb_1", f"{DB}_kb_3"}


def test_deletion_strips_dangling_cross_refs_from_survivors():
    """The whole point of DEV-1455's deletion path: when KB B is deleted,
    KB A's cross-ref to B must NOT survive in the output. SLayer does
    NOT auto-strip these (entities is a free-form list)."""
    a = _kb(1, "A", children_knowledge=[2, 3])
    b = _kb(2, "B")
    c = _kb(3, "C")
    mems = encode_kb_as_memories(DB, [a, b, c], deleted_kb_ids={2})
    by_id = {m["id"]: m for m in mems}
    # B is gone, A's ref to B is stripped, A's ref to C survives.
    assert f"{DB}_kb_2" not in by_id
    assert by_id[f"{DB}_kb_1"]["entities"] == [DB, f"memory:{DB}_kb_3"]


def test_deletion_with_no_surviving_refs_leaves_bare_db_only():
    a = _kb(1, "A", children_knowledge=[2])
    b = _kb(2, "B")
    mems = encode_kb_as_memories(DB, [a, b], deleted_kb_ids={2})
    by_id = {m["id"]: m for m in mems}
    assert by_id[f"{DB}_kb_1"]["entities"] == [DB]


def test_deletion_of_referenced_only_does_not_warn(caplog):
    """A child id that is in ``deleted_kb_ids`` is dropped silently —
    not a warning. Drift warning only fires for children that are
    neither in kb_rows nor deleted."""
    a = _kb(1, "A", children_knowledge=[2])
    b = _kb(2, "B")
    with caplog.at_level(logging.WARNING):
        encode_kb_as_memories(DB, [a, b], deleted_kb_ids={2})
    # No warning records about dangling refs.
    assert not any("dangl" in rec.getMessage().lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# SLayer compatibility
# ---------------------------------------------------------------------------


def test_memory_id_passes_slayer_charset_validation():
    """``<db>_kb_<id>`` must pass ``Memory.id`` charset rules: no ``:``,
    no ``/``, no ``?``, no ``#``, no whitespace, no ASCII control."""
    for kb_id in (0, 1, 42, 1621):
        mid = f"{DB}_kb_{kb_id}"
        assert is_valid_memory_id(mid), f"{mid!r} should pass charset rules"


def test_each_encoded_dict_round_trips_through_memory_model():
    """Every encoder output dict must construct a ``Memory`` without error.
    Catches schema drift in SLayer's Memory model (e.g., DEV-1428).
    """
    a = _kb(1, "A", children_knowledge=[2])
    b = _kb(2, "B", children_knowledge=-1)
    mems = encode_kb_as_memories(DB, [a, b], deleted_kb_ids=set())
    for m in mems:
        # Memory.model_validate accepts dicts; raises on schema violations.
        Memory.model_validate(m)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_calls_produce_byte_identical_yaml():
    """No clock, no PRNG, no os-state: same input → same output, every
    time. Required so byte-equal cache fingerprints are meaningful."""
    rows = [
        _kb(1, "alpha", children_knowledge=[2]),
        _kb(2, "beta", children_knowledge=-1),
        _kb(3, "gamma", children_knowledge=[1, 2]),
    ]
    m1 = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    m2 = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    y1 = yaml.safe_dump(m1, sort_keys=False)
    y2 = yaml.safe_dump(m2, sort_keys=False)
    assert y1 == y2


def test_created_at_is_fixed_epoch():
    """No wall clock — ``created_at`` is the fixed EPOCH constant so two
    runs on the same input produce byte-identical YAML even across hosts.
    """
    [m] = encode_kb_as_memories(DB, [_kb(1, "k")], deleted_kb_ids=set())
    # Encoder may emit a datetime or an ISO string; accept either as long
    # as it represents the EPOCH.
    raw = m["created_at"]
    if hasattr(raw, "isoformat"):
        assert raw == EPOCH
    else:
        assert str(raw).startswith("1970-01-01")


# ---------------------------------------------------------------------------
# Real-corpus shape sanity (skipped when mini-interact data isn't present)
# ---------------------------------------------------------------------------


def _has_resolvable_child(row: dict, kb_ids: set[int]) -> bool:
    """Whether ``row`` cites at least one child that the encoder would
    emit a cross-ref for: present in ``kb_ids`` and not a self-reference.
    Mirrors the encoder's drop rules so test predicates don't outpace
    the implementation."""
    cks = row.get("children_knowledge")
    self_id = int(row["id"])
    if isinstance(cks, list):
        return any(
            int(c) in kb_ids and int(c) != self_id for c in cks
        )
    if isinstance(cks, int) and cks >= 0:
        return cks in kb_ids and cks != self_id
    return False


def _find_smallest_kb_file() -> tuple[str, Path, list[dict]] | None:
    """Return (db_name, kb_path, kb_rows) for the smallest mini-interact
    DB by KB row count, or None if the data is not available locally.

    Used by the real-corpus sanity test below — keeps it CI-friendly.
    """
    from bird_interact_agents import paths

    root = paths.mini_interact_root()
    if not root.exists():
        return None
    candidates: list[tuple[int, str, Path, list[dict]]] = []
    for db_dir in root.iterdir():
        if not db_dir.is_dir() or db_dir.name.startswith("."):
            continue
        kb_path = db_dir / f"{db_dir.name}_kb.jsonl"
        if not kb_path.is_file():
            continue
        rows: list[dict] = []
        for line in kb_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        if not rows:
            continue
        candidates.append((len(rows), db_dir.name, kb_path, rows))
    if not candidates:
        return None
    candidates.sort()
    _, db, kb_path, rows = candidates[0]
    return db, kb_path, rows


def test_real_corpus_round_trips_through_memory_model():
    """Sanity: every KB row in the smallest real mini-interact DB encodes
    into a Memory that ``Memory.model_validate`` accepts. Catches surprises
    in upstream KB shape that toy fixtures wouldn't reveal."""
    found = _find_smallest_kb_file()
    if found is None:
        pytest.skip("mini-interact data not available")
    db, _kb_path, rows = found
    mems = encode_kb_as_memories(db, rows, deleted_kb_ids=set())
    assert len(mems) == len(rows)
    for m in mems:
        Memory.model_validate(m)
        assert m["entities"][0] == db


def test_real_corpus_at_least_one_cross_ref_emitted():
    """At least one of the smallest DB's KB rows must produce a
    ``memory:<db>_kb_<n>`` cross-ref entity — proves the encoder is
    actually wiring up the children graph on real data."""
    found = _find_smallest_kb_file()
    if found is None:
        pytest.skip("mini-interact data not available")
    db, _kb_path, rows = found
    mems = encode_kb_as_memories(db, rows, deleted_kb_ids=set())
    cross_refs = [
        e
        for m in mems
        for e in m["entities"]
        if e.startswith(f"memory:{db}_kb_")
    ]
    # In the corpus, 57% of rows reference at least one other KB. The
    # encoder drops self-references and dangling refs (children not in
    # the kb id set), so this predicate mirrors that: only count cross-
    # refs that the encoder is required to emit. Otherwise a corpus
    # where every child id is dangling would falsely fail the assert.
    kb_ids = {int(r["id"]) for r in rows}
    expected = any(_has_resolvable_child(r, kb_ids) for r in rows)
    if expected:
        assert cross_refs, "expected at least one cross-ref on real corpus"
    else:
        assert cross_refs == [], (
            f"smallest DB has no children_knowledge refs in source data; "
            f"encoder should not invent any. Got: {cross_refs[:5]}"
        )
