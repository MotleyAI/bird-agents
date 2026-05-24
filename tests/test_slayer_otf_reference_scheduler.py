"""Tests for the parallel setup-encode scheduler
``slayer_otf.reference_build._encode_all`` (DEV-1454).

The scheduler walks the KB dependency DAG and runs the (injected) per-KB
encoder ``run_one(kb_id, row, deps_results)`` such that:

* each KB is encoded **exactly once** (per-KB spawn-time lock), even under a
  diamond where two dependents share one dependency;
* a dependent only runs **after** its dependencies finish (its ``deps_results``
  are the finished dependency EncoderResults);
* independent subtrees run **concurrently**, bounded by a semaphore;
* dependency **cycles** are pre-excluded and surfaced as ``deferred`` results
  without ever calling ``run_one`` for the cycle members — and without hanging.
"""

from __future__ import annotations

import asyncio


def _rows(spec: dict[int, list[int]]) -> list[dict]:
    """Build KB rows from an ``id -> children`` spec."""
    out = []
    for kb_id, children in spec.items():
        out.append({
            "id": kb_id,
            "knowledge": f"K{kb_id}",
            "definition": "x",
            "children_knowledge": children if children else -1,
        })
    return out


async def _run(spec, run_one, *, concurrency=6):
    from bird_interact_agents.slayer_otf.reference_build import (
        _edges_from_kb_rows, _encode_all,
    )

    rows = _rows(spec)
    edges = _edges_from_kb_rows(rows)
    return await asyncio.wait_for(
        _encode_all(
            kb_rows=rows, edges=edges, run_one=run_one,
            concurrency=concurrency,
        ),
        timeout=2,  # pure in-memory scheduler; a hang means a cycle-deadlock bug
    )


async def test_each_kb_encoded_exactly_once_diamond():
    """Diamond: 4→{2,3}, 2→1, 3→1. KB 1 is a shared dep of 2 and 3; it
    must be encoded EXACTLY ONCE despite two dependents needing it."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    calls: list[int] = []

    async def run_one(kb_id, row, deps_results):
        calls.append(kb_id)
        await asyncio.sleep(0)  # let the loop interleave
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    results = await _run({1: [], 2: [1], 3: [1], 4: [2, 3]}, run_one)

    assert sorted(calls) == [1, 2, 3, 4]
    assert calls.count(1) == 1, "shared dependency must encode exactly once"
    assert {r.kb_id for r in results} == {1, 2, 3, 4}


async def test_dependency_finishes_before_dependent_runs():
    """A dependent's ``deps_results`` must contain its dependency's finished
    result, proving the dep completed first."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    seen_deps: dict[int, list[int]] = {}

    async def run_one(kb_id, row, deps_results):
        seen_deps[kb_id] = [d.kb_id for d in deps_results]
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    await _run({1: [], 2: [1]}, run_one)
    assert seen_deps[1] == []
    assert seen_deps[2] == [1], "dependent must receive its dep's result"


async def test_semaphore_bounds_concurrency():
    """Independent KBs run concurrently but never exceed the semaphore cap."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    inflight = 0
    peak = 0

    async def run_one(kb_id, row, deps_results):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.02)
        inflight -= 1
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    # 6 independent KBs, cap 2.
    await _run({i: [] for i in range(1, 7)}, run_one, concurrency=2)
    assert peak <= 2, f"concurrency cap exceeded: peak={peak}"
    assert peak >= 2, "independent items should run concurrently up to the cap"


async def test_cycle_members_deferred_without_calling_encoder():
    """A dependency cycle (1→2→1) must not be scheduled: its members come
    back ``status='deferred'`` and ``run_one`` is never called for them
    (and the whole thing must not hang — enforced by the wait_for above)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    called: list[int] = []

    async def run_one(kb_id, row, deps_results):
        called.append(kb_id)
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    results = await _run({1: [2], 2: [1], 3: []}, run_one)
    by_id = {r.kb_id: r for r in results}

    assert by_id[1].status == "deferred"
    assert by_id[2].status == "deferred"
    assert 1 not in called and 2 not in called, (
        "cycle members must not be passed to the encoder"
    )
    # The acyclic KB 3 still encodes.
    assert by_id[3].status == "encoded"
    assert 3 in called


async def test_deferred_dependency_is_visible_to_dependent():
    """When a dependency is deferred, its dependent receives that deferred
    result in ``deps_results`` so the encoder can choose to defer too
    (cascade is the encoder's call; the scheduler just wires the result)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    async def run_one(kb_id, row, deps_results):
        if kb_id == 1:
            return EncoderResult(
                kb_id=1, status="deferred", entities=[], notes="ambiguous",
            )
        # KB 2 cascades: defer because its dep is not encoded.
        dep_ok = all(d.status == "encoded" for d in deps_results)
        return EncoderResult(
            kb_id=kb_id, status="encoded" if dep_ok else "deferred",
            entities=[], notes="",
        )

    results = await _run({1: [], 2: [1]}, run_one)
    by_id = {r.kb_id: r for r in results}
    assert by_id[1].status == "deferred"
    assert by_id[2].status == "deferred", "dependent should see the deferred dep"
