"""DEV-1671: shared helpers for the edited-model-store cleanup recipe.

The cleanup mutates a saved store (delete unused scratch, reference inlined KB
defs, demote answer-baking) via SLayer's edit API, then must prove the mutation
is behaviour-preserving: the reformulated winning query returns the IDENTICAL
result to the original. These helpers own the two non-obvious pieces of that
gate — parsing the SLayer ``query`` tool's ``format="json"`` output into value
rows, and comparing two result sets POSITIONALLY (headers ignored, like the
grader) with Decimal-precision numeric equality and explicit NULL handling.

The dynamic query execution (Component B) lives in :func:`run_store_query`; it
needs a live datasource (local postgres) and is exercised by the cleanup driver,
not the unit suite. The comparison helpers are pure and unit-tested.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any


def extract_result_rows(result: Any) -> list[list[Any]]:
    """Parse the SLayer ``query`` / ``query_nested`` tool output (``format="json"``)
    into a list of positional value rows. Column NAMES are dropped (the grader is
    positional). Tolerant of a leading JSON array possibly followed by trailing
    prose (the tool appends a "Measure attributes:" block)."""
    if isinstance(result, list):
        rows = result
    else:
        text = str(result).lstrip()
        # the JSON payload is the first top-level [...] array; take up to its close
        if not text.startswith("["):
            return []
        depth = 0
        end = None
        for i, ch in enumerate(text):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return []
        rows = json.loads(text[:end])
    out: list[list[Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(list(row.values()))
        elif isinstance(row, (list, tuple)):
            out.append(list(row))
        else:
            out.append([row])
    return out


def _num(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None


def values_equal(a: Any, b: Any, *, round_ndigits: int | None = None) -> bool:
    """Positional value equality: numerics compared as Decimal (optionally rounded
    to ``round_ndigits``), NULLs explicit, everything else by string form."""
    if a is None or b is None:
        return a is None and b is None
    da, db = _num(a), _num(b)
    if da is not None and db is not None:
        if round_ndigits is not None:
            q = Decimal(1).scaleb(-round_ndigits)
            return da.quantize(q) == db.quantize(q)
        return da == db
    return str(a) == str(b)


def results_identical(
    a: Any, b: Any, *, round_ndigits: int | None = None, order_sensitive: bool = True
) -> bool:
    """True iff two SLayer query results are identical as ordered positional value
    tuples (headers ignored). Set ``order_sensitive=False`` to compare as a
    multiset (for queries whose ORDER BY has ties without a deterministic
    tiebreak — compare within equal sort-key groups upstream instead when possible)."""
    ra, rb = extract_result_rows(a), extract_result_rows(b)
    if len(ra) != len(rb):
        return False
    if not order_sensitive:
        ra = sorted(ra, key=lambda r: [str(x) for x in r])
        rb = sorted(rb, key=lambda r: [str(x) for x in r])
    for row_a, row_b in zip(ra, rb):
        if len(row_a) != len(row_b):
            return False
        for va, vb in zip(row_a, row_b):
            if not values_equal(va, vb, round_ndigits=round_ndigits):
                return False
    return True


async def materialize_store(
    *,
    benchmark: str,
    db: str,
    instance_id: str,
    work_dir: str,
    backup_label: str = "20260713_dev1671-cleanup",
) -> dict:
    """Back up the store (once — never overwrites an existing backup) and untar it into
    ``work_dir``. Returns ``{scratch, meta, kb_rows, archive, backup}``. The caller then
    edits ``scratch`` with the REAL SLayer MCP tools (``create_mcp_server(YAMLStorage(scratch))``
    → ``create_model`` / ``edit_model`` / ``delete_model`` / ``query`` / ``inspect`` / …) and
    finishes with :func:`verify_and_repack`. This helper deliberately does NOT wrap the edit
    tools — the agent drives SLayer's honest tool surface directly."""
    import json as _json
    import shutil
    import tarfile
    from pathlib import Path

    from bird_interact_agents import paths
    from bird_interact_agents.slayer_otf import edited_models as _EM

    archive = Path(_EM.run_edited_models_archive(
        benchmark=benchmark, selected_database=db, instance_id=instance_id))
    if not archive.exists():
        raise FileNotFoundError(f"store archive missing: {archive}")
    backup = (paths.main_checkout_root() / "runs" / "_edited_models_backups" / backup_label
              / benchmark / db / instance_id / "edited_models.tar.gz")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(archive, backup)
    work = Path(work_dir)
    with tarfile.open(archive) as t:
        t.extractall(work)
    scratch = work / db
    return {
        "scratch": str(scratch),
        "meta": _json.loads((scratch / "_edited_models_meta.json").read_text()),
        "kb_rows": _json.loads((scratch / "_kb_rows.json").read_text()),
        "archive": str(archive),
        "backup": str(backup),
    }


async def verify_and_repack(
    *,
    benchmark: str,
    db: str,
    instance_id: str,
    work_dir: str,
    scratch: str,
    meta: dict,
    kb_rows: list[dict],
    relevant_kb_ids: list[int],
    baseline_result: str,
    nice_query: dict | list,
    record_query: bool = False,
    round_ndigits: int | None = None,
    order_sensitive: bool = True,
) -> dict:
    """GATE + repack a store the caller has already edited (via the real SLayer tools).
    Runs ``nice_query`` against the edited ``scratch``, asserts the result is IDENTICAL to
    ``baseline_result`` (the original winning query's result, captured before edits), runs the
    checker, and — only if the result is identical — repacks in place. Returns
    ``{ok, clean, remaining, reason}``. If the result differs, the archive is left UNTOUCHED
    (never repacks a store whose behaviour changed). Set ``record_query`` when ``nice_query``
    differs from the original (writes ``_winning_query.json`` so a future scan validates it)."""
    import json as _json
    from pathlib import Path

    from slayer.storage.yaml_storage import YAMLStorage

    from bird_interact_agents import paths
    from bird_interact_agents.slayer_otf import edited_models as _EM
    from bird_interact_agents.slayer_otf.store_kb_checker import check_store

    scratch_p = Path(scratch)

    nice_res = await run_store_query(str(scratch_p), nice_query)
    if not results_identical(baseline_result, nice_res, round_ndigits=round_ndigits,
                             order_sensitive=order_sensitive):
        return {"ok": False, "reason": "identical-result gate failed",
                "baseline": baseline_result.splitlines()[0] if baseline_result else "",
                "nice": nice_res.splitlines()[0] if nice_res else ""}

    cache_dir = paths.slayer_otf_cache_root(benchmark=benchmark) / db
    baseline_storage = YAMLStorage(base_dir=str(cache_dir)) if cache_dir.exists() else None
    rep = await check_store(
        store_storage=YAMLStorage(base_dir=str(scratch_p)), baseline_storage=baseline_storage,
        db=db, kb_rows=kb_rows, relevant_kb_ids=relevant_kb_ids, winning_query=nice_query,
        instance_id=instance_id, benchmark=benchmark)

    if record_query:
        (scratch_p / "_winning_query.json").write_text(_json.dumps(nice_query))

    _EM.save_edited_store(
        benchmark=benchmark, db=db, instance_id=instance_id, work_dir=Path(work_dir),
        scratch=scratch_p, deleted_kb_ids=meta.get("deleted_kb_ids", []),
        cache_fp=meta.get("cache_fp", ""))
    return {"ok": True, "clean": rep.ok, "remaining": [f.model_dump() for f in rep.findings]}


async def run_store_query(scratch_dir: str, query: dict | list, *, fmt: str = "json") -> str:
    """Run a ``SlayerQuery`` (single dict or nested list of stages) against a
    materialised store via SLayer's own MCP ``query`` / ``query_nested`` tool
    (the exact path the agents use), returning the raw tool output. Needs a live
    datasource — the store's ``connection_string`` must resolve (local postgres)."""
    from slayer.mcp.server import create_mcp_server
    from slayer.storage.yaml_storage import YAMLStorage

    storage = YAMLStorage(base_dir=scratch_dir)
    mcp = create_mcp_server(storage)
    try:
        if isinstance(query, list):
            fn = mcp._tool_manager._tools["query_nested"].fn
            return await fn(queries=query, format=fmt)
        fn = mcp._tool_manager._tools["query"].fn
        return await fn(**{**query, "format": fmt})
    finally:
        engine = getattr(mcp, "_slayer_engine", None)
        if engine is not None and hasattr(engine, "aclose"):
            await engine.aclose()
