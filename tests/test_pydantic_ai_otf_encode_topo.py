"""Tests for the topological sort used by `_register_kb_to_slayer`.

Two helpers under test:

* `_walk_children(seed_ids, kb_rows_by_id) -> set[int]` — returns seed
  ∪ all transitive children resolvable from KB rows.
* `_topo_sort(ids, kb_rows_by_id) -> list[int]` — returns the ids in
  dep-before-dependent order. Cycles are detected: the SCC members are
  returned in their detected order with a paired `cycle_ids: set[int]`
  the caller uses to short-circuit them to `status="error"` per Codex
  finding 5.

The dep graph is built from each KB's memory `entities` list — the
`memory:<db>_kb_<n>` tokens that DEV-1455's `_resolve_cross_refs`
already filtered for deleted/unknown ids (Codex finding 6). This test
file uses a stand-in `kb_rows_by_id` shaped as the loader produces,
plus an `entities` mapping the helper consumes.

The helper signature accepts `entities_by_id: dict[int, list[int]]`
— the resolved-children edge list, not the raw `children_knowledge`
from the YAML body.
"""

from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# _walk_children
# ---------------------------------------------------------------------------


def test_walk_children_linear_chain():
    """5 -> 4 -> 3 — seed [5] expands to {3, 4, 5}."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _walk_children,
    )

    edges = {3: [], 4: [3], 5: [4]}
    out = _walk_children({5}, edges)
    assert out == {3, 4, 5}


def test_walk_children_diamond():
    """5 -> {3, 4}; 3 -> 2; 4 -> 2 — seed [5] expands to {2, 3, 4, 5}."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _walk_children,
    )

    edges = {2: [], 3: [2], 4: [2], 5: [3, 4]}
    out = _walk_children({5}, edges)
    assert out == {2, 3, 4, 5}


def test_walk_children_handles_multiple_seeds():
    """Seeds with overlapping descendants are unioned correctly."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _walk_children,
    )

    edges = {1: [], 2: [], 3: [1], 4: [2], 5: [3], 6: [4]}
    out = _walk_children({5, 6}, edges)
    assert out == {1, 2, 3, 4, 5, 6}


def test_walk_children_empty_seed():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _walk_children,
    )

    assert _walk_children(set(), {1: [], 2: [1]}) == set()


def test_walk_children_raises_on_unknown_seed():
    """Seed id not present in `entities_by_id` is a bug in the caller —
    raise rather than silently dropping."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _walk_children,
    )

    with pytest.raises(KeyError):
        _walk_children({99}, {1: [], 2: [1]})


def test_walk_children_skips_unknown_child_with_warning(caplog):
    """A child id present in `children_knowledge` but absent from the
    row map (DEV-1455 already filters but defense-in-depth) is dropped
    silently — the upstream filter is the source of truth. No raise."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _walk_children,
    )

    edges = {1: [], 2: [1, 999]}  # 999 absent
    out = _walk_children({2}, edges)
    assert out == {1, 2}


# ---------------------------------------------------------------------------
# _topo_sort
# ---------------------------------------------------------------------------


def test_topo_sort_linear_chain_returns_deps_first():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _topo_sort,
    )

    edges = {3: [], 4: [3], 5: [4]}
    order, cycle_ids = _topo_sort({3, 4, 5}, edges)
    assert order == [3, 4, 5]
    assert cycle_ids == set()


def test_topo_sort_diamond_deterministic_tie_break_ascending_id():
    """5 -> {3, 4}, both depend on 2 — pinned tie-break is ascending id.
    With ties broken consistently, every run produces the same order;
    tests depending on the topo ordering get a stable contract."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _topo_sort,
    )

    edges = {2: [], 3: [2], 4: [2], 5: [3, 4]}
    order, cycle_ids = _topo_sort({2, 3, 4, 5}, edges)
    assert order == [2, 3, 4, 5]
    assert cycle_ids == set()


def test_topo_sort_empty_input():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _topo_sort,
    )

    order, cycle_ids = _topo_sort(set(), {})
    assert order == []
    assert cycle_ids == set()


def test_topo_sort_detects_cycle_and_returns_scc_members(caplog):
    """Codex finding 5: a cycle (2 -> 3 -> 2) is NOT silently broken.
    The helper detects the SCC and returns its members in `cycle_ids`
    so the caller can short-circuit them to `status='error'` without
    running an encoder."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _topo_sort,
    )

    edges = {2: [3], 3: [2], 4: []}
    with caplog.at_level(logging.WARNING):
        order, cycle_ids = _topo_sort({2, 3, 4}, edges)
    assert cycle_ids == {2, 3}
    # 4 has no deps, so it's still encodeable; it appears in order.
    assert 4 in order
    # Cycle members appear in `cycle_ids` and NOT in the topo order.
    assert 2 not in order
    assert 3 not in order


def test_topo_sort_isolates_cycle_from_acyclic_dependents():
    """If 4 depends on cyclic 2, then 4 transitively can't be encoded
    either — it shows up in `cycle_ids` (or in a separate
    `unencodeable_ids` set) so the caller short-circuits it too.

    The simplest contract: every id that transitively touches a cycle
    is in `cycle_ids`, every id that doesn't is in `order`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _topo_sort,
    )

    edges = {1: [], 2: [3], 3: [2], 4: [2]}
    order, cycle_ids = _topo_sort({1, 2, 3, 4}, edges)
    assert cycle_ids == {2, 3, 4}
    assert order == [1]


def test_topo_sort_seed_ignores_ids_not_in_input_set():
    """`_topo_sort` orders ONLY the input set; edges to ids outside
    the set are ignored (they were either already encoded earlier or
    out of scope for this call)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _topo_sort,
    )

    # 5 -> 4 -> 3, but only sort {4, 5} — 3 is treated as a leaf.
    edges = {3: [], 4: [3], 5: [4]}
    order, cycle_ids = _topo_sort({4, 5}, edges)
    assert order == [4, 5]
    assert cycle_ids == set()
