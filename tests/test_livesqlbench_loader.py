"""DEV-1462 — `harness.load_livesqlbench_tasks`.

Mirrors the existing `harness.load_tasks` shape (path + limit), with two
extra responsibilities:

  * **Gold merge** by `instance_id` — sets `sol_sql`, `external_knowledge`,
    `test_cases` on each public task (the public dataset ships these empty;
    the gated `--gold-file` carries them).
  * **`query` → `amb_user_query` shim** so every agent + `_ambiguity_count`
    keeps reading `amb_user_query` unchanged.
  * **Dataset marker** `task["dataset"] = "livesqlbench-base-lite-sqlite"` — the loader-stamped
    irreducible source of truth for per-task DB isolation + one-shot
    `run_task`'s programmatic guard.
  * **SELECT filter** to `category == "Query"` (180 rows on the real
    dataset).
  * **`filter_ids`-aware fail-fast** on empty post-merge `sol_sql` — only
    for tasks that will actually run, so a partial gold sidecar matched
    against `--instance-id` doesn't trip on un-targeted rows.
  * **Limit applied AFTER filtering** so `--limit 180` doesn't yield < 180.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._livesqlbench_fixtures import (
    make_lsb_dataset,
    make_lsb_gold,
    public_task,
)


def _public_tasks_180_query_90_management() -> list[dict]:
    """Mirror the real dataset shape: a deterministic 18-DB × 15-task
    mix where exactly 180 are Query and 90 are Management — so the
    loader's "assert exactly 180" full-run check has something concrete
    to compare against without shipping the actual data."""
    dbs = [
        "alien", "archeology", "credit", "cross_db", "crypto", "cybermarket",
        "disaster", "fake", "gaming", "insider", "mental", "museum", "news",
        "polar", "robot", "solar", "vaccine", "virtual",
    ]
    out: list[dict] = []
    for db in dbs:
        # 10 Query + 5 Management per DB = 18 × 15 = 270; Query total = 180.
        for n in range(1, 11):
            out.append(public_task(f"{db}_{n}", db, category="Query"))
        for n in range(1, 6):
            inst_id = f"{db}_M_{n}"
            out.append(public_task(inst_id, db, category="Management"))
    return out


def test_loader_merges_gold_by_instance_id(tmp_path):
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_1", "alien"),
            public_task("alien_2", "alien"),
        ],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl", rows=[
            {"instance_id": "alien_1",
             "sol_sql": ["SELECT id FROM widgets WHERE id = 1"],
             "external_knowledge": [{"k": "v"}],
             "test_cases": []},
            {"instance_id": "alien_2",
             "sol_sql": ["SELECT name FROM widgets"],
             "external_knowledge": [],
             "test_cases": []},
        ],
    )
    rows = load_livesqlbench_tasks(str(data), str(gold), limit=100)
    assert {r["instance_id"] for r in rows} == {"alien_1", "alien_2"}
    by_id = {r["instance_id"]: r for r in rows}
    assert by_id["alien_1"]["sol_sql"] == ["SELECT id FROM widgets WHERE id = 1"]
    assert by_id["alien_1"]["external_knowledge"] == [{"k": "v"}]
    assert by_id["alien_2"]["sol_sql"] == ["SELECT name FROM widgets"]


def test_loader_maps_query_to_amb_user_query(tmp_path):
    """Every agent + `_ambiguity_count` reads `amb_user_query`; the shim
    must populate it from the public `query` field (and keep `query` for
    traceability)."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_1", "alien", query="how many widgets?"),
        ],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_1",
               "sol_sql": ["SELECT COUNT(*) FROM widgets"]}],
    )
    rows = load_livesqlbench_tasks(str(data), str(gold), limit=100)
    assert rows[0]["amb_user_query"] == "how many widgets?"
    assert rows[0]["query"] == "how many widgets?", (
        "`query` must be preserved alongside `amb_user_query` for "
        "downstream traceability."
    )


def test_loader_stamps_dataset_marker(tmp_path):
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"],
        tasks=[public_task("alien_1", "alien")],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_1"}],
    )
    rows = load_livesqlbench_tasks(str(data), str(gold), limit=100)
    assert all(r.get("dataset") == "livesqlbench-base-lite-sqlite" for r in rows), (
        "loader MUST stamp task['dataset']='livesqlbench' — it's the "
        "irreducible marker for materialize_task_db + one-shot run_task."
    )


def test_loader_filters_to_select_only(tmp_path):
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_1", "alien", category="Query"),
            public_task("alien_M_1", "alien", category="Management"),
            public_task("alien_2", "alien", category="Query"),
        ],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[
            {"instance_id": "alien_1"},
            {"instance_id": "alien_M_1"},  # gold present, but filtered out
            {"instance_id": "alien_2"},
        ],
    )
    rows = load_livesqlbench_tasks(str(data), str(gold), limit=100)
    assert {r["instance_id"] for r in rows} == {"alien_1", "alien_2"}
    assert all(r["category"] == "Query" for r in rows)


def test_loader_asserts_180_on_full_run(tmp_path):
    """A real full-run load returns exactly 180 SELECT rows — the loader
    must assert that so a silently truncated dataset surfaces immediately."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    tasks = _public_tasks_180_query_90_management()
    data = make_lsb_dataset(tmp_path / "lsb", dbs=[
        "alien", "archeology", "credit", "cross_db", "crypto", "cybermarket",
        "disaster", "fake", "gaming", "insider", "mental", "museum", "news",
        "polar", "robot", "solar", "vaccine", "virtual",
    ], tasks=tasks)
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": t["instance_id"]} for t in tasks],
    )
    rows = load_livesqlbench_tasks(str(data), str(gold), limit=None)
    assert len(rows) == 180


def test_loader_assertion_fires_on_short_full_run(tmp_path):
    """If the dataset is silently truncated to fewer than 180 SELECT
    rows, the full-run loader MUST raise — not return a short list and
    let downstream metrics lie."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_1", "alien"),
            public_task("alien_2", "alien"),
        ],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_1"}, {"instance_id": "alien_2"}],
    )
    # Codex review: the guard is a `raise ValueError`, not an `assert`,
    # so it survives `python -O` (which strips assertions).
    with pytest.raises(ValueError):
        load_livesqlbench_tasks(str(data), str(gold), limit=None)


def test_loader_limit_applied_after_filtering(tmp_path):
    """A `--limit 180` request against a dataset whose first 180 rows
    are NOT all `category=="Query"` must still return 180 Query rows.
    Apply-limit-before-filter would silently undercount."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    # 3 Management at the head, then 5 Query.
    head = [public_task(f"alien_M_{n}", "alien", category="Management")
            for n in range(1, 4)]
    tail = [public_task(f"alien_{n}", "alien", category="Query")
            for n in range(1, 6)]
    data = make_lsb_dataset(tmp_path / "lsb", dbs=["alien"], tasks=head + tail)
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": t["instance_id"]} for t in head + tail],
    )
    rows = load_livesqlbench_tasks(str(data), str(gold), limit=5)
    assert {r["instance_id"] for r in rows} == {
        "alien_1", "alien_2", "alien_3", "alien_4", "alien_5",
    }
    assert all(r["category"] == "Query" for r in rows)


def test_loader_fail_fast_on_missing_gold_file(tmp_path):
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"],
        tasks=[public_task("alien_1", "alien")],
    )
    missing = tmp_path / "no_such_gold.jsonl"
    with pytest.raises((FileNotFoundError, OSError)):
        load_livesqlbench_tasks(str(data), str(missing), limit=None)


def test_loader_fail_fast_on_invalid_gold_json(tmp_path):
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"],
        tasks=[public_task("alien_1", "alien")],
    )
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not-a-json-line\n")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        load_livesqlbench_tasks(str(data), str(bad), limit=None)


def test_loader_fail_fast_on_kept_task_with_empty_sol_sql(tmp_path):
    """A kept SELECT task with empty `sol_sql` post-merge means the gold
    sidecar is incomplete for that task — must fail-fast, not let it run
    and score zero."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"],
        tasks=[public_task("alien_1", "alien")],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_1", "sol_sql": []}],
    )
    with pytest.raises((ValueError, AssertionError)):
        load_livesqlbench_tasks(str(data), str(gold), limit=None)


def test_loader_filter_ids_narrows_before_empty_sol_sql_check(tmp_path):
    """A partial-gold scenario MUST work when the user pins `filter_ids`
    to the rows they DO have gold for — the fail-fast must only consider
    the tasks that will actually run (Codex #6)."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_1", "alien"),
            public_task("alien_2", "alien"),
            public_task("alien_3", "alien"),
        ],
    )
    # Gold provided only for alien_2 (the one the user wants to run).
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_2",
               "sol_sql": ["SELECT name FROM widgets"]}],
    )
    rows = load_livesqlbench_tasks(
        str(data), str(gold), limit=None, filter_ids=["alien_2"],
    )
    assert {r["instance_id"] for r in rows} == {"alien_2"}


def test_loader_logs_warning_on_m_underscore_id_with_query_category(tmp_path, caplog):
    """Plan B2 step 5: `category=="Query"` is authoritative; an `_M_`
    substring in `instance_id` is a DEFENSIVE cross-check. The loader
    must log (not raise) when the two signals disagree — so a real
    upstream relabeling surfaces in the run log without breaking the run."""
    import logging

    from bird_interact_agents.harness import load_livesqlbench_tasks

    # alien_M_2 has `_M_` in the id (looks Management) BUT category="Query"
    # — that's the disagreement we want to surface.
    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_M_2", "alien", category="Query"),
        ],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_M_2"}],
    )
    with caplog.at_level(logging.WARNING):
        rows = load_livesqlbench_tasks(
            str(data), str(gold), limit=None, filter_ids=["alien_M_2"],
        )
    # The task is KEPT (authoritative signal is category="Query")…
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "alien_M_2"
    # …but the cross-check disagreement is surfaced in the log.
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "alien_M_2" in msg, (
        "loader must log a warning when `_M_` cross-check disagrees with "
        f"category=='Query'; got log: {msg!r}"
    )


def test_loader_filter_ids_assert_180_skipped_when_filter_present(tmp_path):
    """When `filter_ids` (or `limit`) narrows the set, the "assert 180"
    full-run check MUST NOT fire."""
    from bird_interact_agents.harness import load_livesqlbench_tasks

    data = make_lsb_dataset(
        tmp_path / "lsb", dbs=["alien"], tasks=[
            public_task("alien_1", "alien"),
        ],
    )
    gold = make_lsb_gold(
        tmp_path / "gold.jsonl",
        rows=[{"instance_id": "alien_1"}],
    )
    # No raise even though we have far fewer than 180 rows.
    rows = load_livesqlbench_tasks(
        str(data), str(gold), limit=None, filter_ids=["alien_1"],
    )
    assert len(rows) == 1


def test_load_benchmark_tasks_nongold_applies_limit_after_filter(tmp_path):
    """Non-gold dispatch (mini_interact): `limit` must apply AFTER `filter_ids`.

    Limiting first can truncate a requested instance_id away before filtering —
    e.g. filtering to the 3rd row with limit=2 would load only [a, b], then
    filter to [] (the requested `c` dropped). Limit-after-filter yields [c]
    (CodeRabbit, mirrors the LiveSQLBench path)."""
    from bird_interact_agents.harness import load_benchmark_tasks

    p = tmp_path / "mini.jsonl"
    p.write_text("".join(
        json.dumps({"instance_id": i, "selected_database": "db"}) + "\n"
        for i in ("a", "b", "c")
    ))
    rows = load_benchmark_tasks(
        "mini-interact", str(p), None, limit=2, filter_ids=["c"],
    )
    assert [r["instance_id"] for r in rows] == ["c"]
