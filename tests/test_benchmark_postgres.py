"""Benchmark registry tests for the two new postgres-backed benchmarks.

Pins the resolution contract (canonical name, dataset marker, cli tokens)
and the per-benchmark facts the harness keys off (db_backend, modes,
isolation, gold requirements, etc.).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bird_interact_agents.benchmark import (
    Benchmark,
    all_benchmarks,
    benchmark_names,
    cli_dataset_tokens,
    get_benchmark,
)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolves_livesqlbench_postgres():
    b = get_benchmark("livesqlbench-base-lite")
    assert b.name == "livesqlbench-base-lite"


def test_resolves_mini_interact_postgres():
    b = get_benchmark("bird-interact-lite-exp")
    assert b.name == "bird-interact-lite-exp"


def test_resolves_dataset_markers():
    assert get_benchmark("livesqlbench-base-lite").name == "livesqlbench-base-lite"
    assert get_benchmark("bird-interact-lite-exp").name == "bird-interact-lite-exp"


def test_all_benchmarks_includes_postgres():
    names = {b.name for b in all_benchmarks()}
    assert "livesqlbench-base-lite" in names
    assert "bird-interact-lite-exp" in names


def test_benchmark_names_includes_postgres():
    names = set(benchmark_names())
    assert "livesqlbench-base-lite" in names
    assert "bird-interact-lite-exp" in names


def test_cli_dataset_tokens_includes_postgres():
    tokens = set(cli_dataset_tokens())
    assert "livesqlbench-base-lite" in tokens
    assert "bird-interact-lite-exp" in tokens
    # every advertised token resolves
    for t in tokens:
        get_benchmark(t)


# ---------------------------------------------------------------------------
# db_backend field
# ---------------------------------------------------------------------------


def test_sqlite_benchmarks_have_sqlite_backend():
    assert get_benchmark("mini-interact").db_backend == "sqlite"
    assert get_benchmark("livesqlbench-base-lite-sqlite").db_backend == "sqlite"


def test_postgres_benchmarks_have_postgres_backend():
    assert get_benchmark("livesqlbench-base-lite").db_backend == "postgres"
    assert get_benchmark("bird-interact-lite-exp").db_backend == "postgres"


def test_db_backend_default_is_sqlite():
    """A new Benchmark without an explicit db_backend must default to sqlite
    so existing descriptors are unaffected."""
    assert Benchmark.model_fields["db_backend"].default == "sqlite"


def test_db_backend_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Benchmark(
            name="x",
            dataset_marker="x",
            data_file="x.jsonl",
            supported_modes=("one-shot",),
            one_shot=True,
            gold_required=False,
            per_task_db_isolation=False,
            container_data_dir="/data/x",
            db_backend="mysql",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# livesqlbench_postgres facts
# ---------------------------------------------------------------------------


def test_livesqlbench_postgres_facts():
    b = get_benchmark("livesqlbench-base-lite")
    assert b.db_backend == "postgres"
    assert b.data_file == "livesqlbench_data.jsonl"
    assert b.one_shot is True
    assert b.gold_required is True
    assert b.per_task_db_isolation is False
    assert set(b.supported_modes) == {"one-shot"}
    assert b.audited_gold_layout == "single_file"
    assert b.container_data_dir == "/data/livesqlbench-base-lite"
    assert b.dataset_marker == "livesqlbench-base-lite"


# ---------------------------------------------------------------------------
# mini_interact_postgres facts
# ---------------------------------------------------------------------------


def test_mini_interact_postgres_facts():
    b = get_benchmark("bird-interact-lite-exp")
    assert b.db_backend == "postgres"
    assert b.data_file == "mini_interact.jsonl"
    assert b.one_shot is False
    assert b.gold_required is False
    assert b.per_task_db_isolation is False
    assert "a-interact" in b.supported_modes
    assert "one-shot" not in b.supported_modes
    assert b.audited_gold_layout == "single_file"
    assert b.container_data_dir == "/data/bird-interact-lite-exp"
    assert b.dataset_marker == "bird-interact-lite-exp"


# ---------------------------------------------------------------------------
# Uniqueness contract (extends test_benchmark.py)
# ---------------------------------------------------------------------------


def test_all_benchmarks_have_distinct_container_dirs():
    bs = list(all_benchmarks())
    dirs = [b.container_data_dir for b in bs]
    assert len(dirs) == len(set(dirs)), f"duplicate container_data_dir: {dirs}"


def test_all_benchmarks_have_distinct_name_file_pairs():
    """Each benchmark has a unique (name, data_file) combination."""
    bs = list(all_benchmarks())
    pairs = [(b.name, b.data_file) for b in bs]
    assert len(pairs) == len(set(pairs))


def test_all_benchmarks_have_distinct_dataset_markers():
    bs = list(all_benchmarks())
    markers = [b.dataset_marker for b in bs]
    assert len(markers) == len(set(markers))


# ---------------------------------------------------------------------------
# Loader dataset-marker threading (DEV-1523 Codex finding)
# ---------------------------------------------------------------------------


def _make_minimal_livesqlbench_jsonl(path, instance_ids):
    """Write a minimal livesqlbench JSONL with the given instance_ids."""
    import json
    with open(path, "w") as f:
        for iid in instance_ids:
            f.write(json.dumps({
                "instance_id": iid,
                "selected_database": "alien",
                "category": "Query",
                "query": "How many rows?",
            }) + "\n")


def _make_gold_jsonl(path, instance_ids):
    """Write a minimal gold sidecar with the given instance_ids."""
    import json
    with open(path, "w") as f:
        for iid in instance_ids:
            f.write(json.dumps({
                "instance_id": iid,
                "sol_sql": ["SELECT COUNT(*) FROM t"],
            }) + "\n")


def test_load_benchmark_tasks_postgres_stamps_correct_marker(tmp_path):
    """load_benchmark_tasks for livesqlbench_postgres must stamp
    row['dataset'] = 'livesqlbench_postgres', not 'livesqlbench'.
    The wrong marker causes execute_submit_action to route postgres tasks
    through the SQLite path (Codex finding, DEV-1523)."""
    from bird_interact_agents.harness import load_benchmark_tasks

    data_file = tmp_path / "data.jsonl"
    gold_file = tmp_path / "gold.jsonl"
    _make_minimal_livesqlbench_jsonl(data_file, ["alien_pg_1"])
    _make_gold_jsonl(gold_file, ["alien_pg_1"])

    tasks = load_benchmark_tasks(
        "livesqlbench-base-lite",
        str(data_file),
        gold_file=str(gold_file),
        filter_ids=["alien_pg_1"],
    )
    assert tasks, "expected at least one task"
    assert tasks[0]["dataset"] == "livesqlbench-base-lite", (
        f"expected dataset_marker='livesqlbench_postgres', got {tasks[0]['dataset']!r}; "
        "harness dispatch keys off this marker to route postgres vs sqlite"
    )


def test_load_benchmark_tasks_mini_interact_postgres_stamps_correct_marker(tmp_path):
    """load_benchmark_tasks for mini_interact_postgres must stamp
    row['dataset'] = 'mini_interact_postgres', not 'mini_interact'.
    The wrong marker routes postgres tasks through the SQLite path (Codex, DEV-1523)."""
    import json
    from bird_interact_agents.harness import load_benchmark_tasks

    data_file = tmp_path / "data.jsonl"
    with open(data_file, "w") as f:
        f.write(json.dumps({
            "instance_id": "alien_pg_1",
            "selected_database": "alien",
            "amb_user_query": "How many rows?",
        }) + "\n")

    tasks = load_benchmark_tasks(
        "bird-interact-lite-exp",
        str(data_file),
        filter_ids=["alien_pg_1"],
    )
    assert tasks, "expected at least one task"
    assert tasks[0]["dataset"] == "bird-interact-lite-exp", (
        f"expected dataset='mini_interact_postgres', got {tasks[0]['dataset']!r}"
    )


def test_load_benchmark_tasks_sqlite_mini_interact_still_stamps_mini_interact(tmp_path):
    """Backward-compat: the SQLite mini_interact still gets dataset='mini_interact'."""
    import json
    from bird_interact_agents.harness import load_benchmark_tasks

    data_file = tmp_path / "data.jsonl"
    with open(data_file, "w") as f:
        f.write(json.dumps({
            "instance_id": "alien_1",
            "selected_database": "alien",
            "amb_user_query": "How many rows?",
        }) + "\n")

    tasks = load_benchmark_tasks(
        "mini-interact",
        str(data_file),
        filter_ids=["alien_1"],
    )
    assert tasks
    assert tasks[0]["dataset"] == "mini-interact"


def test_load_benchmark_tasks_sqlite_livesqlbench_still_stamps_livesqlbench(tmp_path):
    """Backward-compat: the SQLite livesqlbench still gets dataset='livesqlbench'."""
    from bird_interact_agents.harness import load_benchmark_tasks

    data_file = tmp_path / "data.jsonl"
    gold_file = tmp_path / "gold.jsonl"
    _make_minimal_livesqlbench_jsonl(data_file, ["alien_1"])
    _make_gold_jsonl(gold_file, ["alien_1"])

    tasks = load_benchmark_tasks(
        "livesqlbench-base-lite-sqlite",
        str(data_file),
        gold_file=str(gold_file),
        filter_ids=["alien_1"],
    )
    assert tasks
    assert tasks[0]["dataset"] == "livesqlbench-base-lite-sqlite"


# ---------------------------------------------------------------------------
# _pg_execute_submit_action — finished flag respects benchmark.one_shot
# ---------------------------------------------------------------------------


def _make_sample_status(dataset: str, db: str, sol_sql: list[str]):
    """Minimal SampleStatus-like object with original_data for harness tests."""
    from types import SimpleNamespace
    return SimpleNamespace(
        original_data={
            "dataset": dataset,
            "selected_database": db,
            "sol_sql": sol_sql,
        }
    )


def test_pg_submit_one_shot_finishes_on_wrong_answer():
    """livesqlbench_postgres (one_shot=True): finished=True even when p1=False."""
    from unittest.mock import patch, MagicMock
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["SELECT 1"])

    with patch("bird_interact_agents.harness.make_db_connection") as mock_conn_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value = ([(99,)], ["n"])      # predicted — wrong
        mock_conn.execute_sequence.return_value = ([(1,)], ["n"])  # gold — different
        mock_conn_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        _, _, p1, _, finished = _pg_execute_submit_action("SELECT 99", ss, "/data")

    assert not p1
    assert finished, "one-shot benchmark must finish even on wrong answer"


def test_pg_submit_interactive_does_not_finish_on_wrong_answer():
    """mini_interact_postgres (one_shot=False): finished=False when p1=False."""
    from unittest.mock import patch, MagicMock
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("bird-interact-lite-exp", "alien", ["SELECT 1"])

    with patch("bird_interact_agents.harness.make_db_connection") as mock_conn_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value = ([(99,)], ["n"])      # predicted — wrong
        mock_conn.execute_sequence.return_value = ([(1,)], ["n"])  # gold — different
        mock_conn_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        _, _, p1, _, finished = _pg_execute_submit_action("SELECT 99", ss, "/data")

    assert not p1
    assert not finished, "interactive benchmark must NOT finish on a wrong answer"


def test_pg_submit_interactive_finishes_on_correct_answer():
    """mini_interact_postgres: finished=True when p1=True."""
    from unittest.mock import patch, MagicMock
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("bird-interact-lite-exp", "alien", ["SELECT 1"])

    with patch("bird_interact_agents.harness.make_db_connection") as mock_conn_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.return_value = ([(1,)], ["n"])           # predicted — correct
        mock_conn.execute_sequence.return_value = ([(1,)], ["n"])  # gold — matches
        mock_conn_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        _, _, p1, _, finished = _pg_execute_submit_action("SELECT 1", ss, "/data")

    assert p1
    assert finished, "interactive benchmark must finish when p1=True"


def test_pg_submit_interactive_does_not_finish_on_sql_error():
    """mini_interact_postgres: SQL error => finished=False so agent can retry."""
    from unittest.mock import patch, MagicMock
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("bird-interact-lite-exp", "alien", ["SELECT 1"])

    with patch("bird_interact_agents.harness.make_db_connection") as mock_conn_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError("syntax error")
        mock_conn_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        obs, _, _, _, finished = _pg_execute_submit_action("BAD SQL", ss, "/data")

    assert "error" in obs.lower()
    assert not finished, "interactive benchmark must NOT finish on SQL execution error"


def test_pg_submit_one_shot_finishes_on_sql_error():
    """livesqlbench_postgres: SQL error => finished=True (task over regardless)."""
    from unittest.mock import patch, MagicMock
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["SELECT 1"])

    with patch("bird_interact_agents.harness.make_db_connection") as mock_conn_ctx:
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError("syntax error")
        mock_conn_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        _, _, _, _, finished = _pg_execute_submit_action("BAD SQL", ss, "/data")

    assert finished, "one-shot benchmark must finish even on SQL execution error"


# ---------------------------------------------------------------------------
# Sequential gold SQL execution semantics
# ---------------------------------------------------------------------------


def test_pg_submit_gold_sequence_uses_execute_sequence():
    """sol_sqls must be passed to execute_sequence (one transaction) not
    to execute per-statement.  execute_sequence is called exactly once on
    the gold connection with the full list in order."""
    from unittest.mock import patch
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["stmt1", "stmt2"])

    connections_opened = []

    class _TrackingConn:
        def __init__(self, idx):
            self.idx = idx
            self.sequence_calls: list[list[str]] = []

        def execute(self, q):
            # Only called for the predicted SQL (conn 0)
            return [(42,)], ["n"]

        def execute_sequence(self, sqls):
            self.sequence_calls.append(list(sqls))
            return [(42,)], ["n"]

    class _TrackingCM:
        def __enter__(self):
            c = _TrackingConn(len(connections_opened))
            connections_opened.append(c)
            return c

        def __exit__(self, *_):
            return False

    with patch("bird_interact_agents.harness.make_db_connection", return_value=_TrackingCM()):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT 42", ss, "/data")

    assert p1, "predicted and gold match — expected p1=True"
    assert len(connections_opened) == 2, (
        f"expected exactly 2 DB connections (1 predicted + 1 gold sequence), "
        f"got {len(connections_opened)}"
    )
    gold_conn = connections_opened[1]
    assert gold_conn.sequence_calls == [["stmt1", "stmt2"]], (
        f"execute_sequence must be called once with the full list; "
        f"got {gold_conn.sequence_calls}"
    )


def test_pg_submit_gold_sequence_uses_last_result():
    """When sol_sqls has multiple statements, the last statement's result
    (as returned by execute_sequence) is compared against predicted."""
    from unittest.mock import patch
    from bird_interact_agents.harness import _pg_execute_submit_action

    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["setup_stmt", "final_select"])

    connections_opened = []

    class _TrackingConn:
        def __init__(self, idx):
            self.idx = idx

        def execute(self, q):
            return [(99,)], ["n"]  # predicted

        def execute_sequence(self, sqls):
            # Returns the final result (99 matches predicted)
            return [(99,)], ["n"]

    class _TrackingCM:
        def __enter__(self):
            c = _TrackingConn(len(connections_opened))
            connections_opened.append(c)
            return c

        def __exit__(self, *_):
            return False

    with patch("bird_interact_agents.harness.make_db_connection", return_value=_TrackingCM()):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT 99", ss, "/data")

    assert p1, "execute_sequence result matches predicted — expected p1=True"


# ---------------------------------------------------------------------------
# JSONB / unhashable cell handling and order-sensitive grading
# ---------------------------------------------------------------------------


def _make_sample_status_with_conditions(
    dataset: str, db: str, sol_sql: list[str], conditions: dict
):
    from types import SimpleNamespace
    return SimpleNamespace(
        original_data={
            "dataset": dataset,
            "selected_database": db,
            "sol_sql": sol_sql,
            "conditions": conditions,
        }
    )


def _patch_pg_conn(pred_rows, gold_rows):
    """Return a context-manager patch that returns pred_rows for execute
    and gold_rows for execute_sequence."""
    from unittest.mock import patch

    class _Conn:
        def execute(self, q):
            return pred_rows, ["col"]

        def execute_sequence(self, sqls):
            return gold_rows, ["col"]

    class _CM:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *_):
            return False

    return patch("bird_interact_agents.harness.make_db_connection", return_value=_CM())


def test_pg_submit_jsonb_cells_are_hashable():
    """JSONB columns returned as dict by psycopg2 must not crash Counter comparison."""
    from bird_interact_agents.harness import _pg_execute_submit_action

    pred = [({"key": "val"},)]
    gold = [({"key": "val"},)]
    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["SELECT col FROM t"])

    with _patch_pg_conn(pred, gold):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT col FROM t", ss, "/data")

    assert p1, "matching JSONB rows should compare equal"


def test_pg_submit_jsonb_cells_differ():
    """Differing JSONB rows must produce p1=False."""
    from bird_interact_agents.harness import _pg_execute_submit_action

    pred = [({"key": "a"},)]
    gold = [({"key": "b"},)]
    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["SELECT col FROM t"])

    with _patch_pg_conn(pred, gold):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT col FROM t", ss, "/data")

    assert not p1, "differing JSONB rows should not match"


def test_pg_submit_ordered_match_correct_order():
    """conditions[order]=True: same rows in same order => p1=True."""
    from bird_interact_agents.harness import _pg_execute_submit_action

    pred = [(1,), (2,)]
    gold = [(1,), (2,)]
    ss = _make_sample_status_with_conditions(
        "livesqlbench-base-lite", "alien", ["SELECT n FROM t ORDER BY n"], {"order": True}
    )

    with _patch_pg_conn(pred, gold):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT n FROM t ORDER BY n", ss, "/data")

    assert p1, "ordered match with correct order should pass"


def test_pg_submit_ordered_match_wrong_order():
    """conditions[order]=True: same rows in wrong order => p1=False."""
    from bird_interact_agents.harness import _pg_execute_submit_action

    pred = [(2,), (1,)]
    gold = [(1,), (2,)]
    ss = _make_sample_status_with_conditions(
        "livesqlbench-base-lite", "alien", ["SELECT n FROM t ORDER BY n"], {"order": True}
    )

    with _patch_pg_conn(pred, gold):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT n FROM t ORDER BY n", ss, "/data")

    assert not p1, "ordered match with wrong order should fail"


def test_pg_submit_unordered_ignores_row_order():
    """conditions[order] absent/False: same rows different order => p1=True."""
    from bird_interact_agents.harness import _pg_execute_submit_action

    pred = [(2,), (1,)]
    gold = [(1,), (2,)]
    ss = _make_sample_status("livesqlbench-base-lite", "alien", ["SELECT n FROM t"])

    with _patch_pg_conn(pred, gold):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT n FROM t", ss, "/data")

    assert p1, "unordered match should ignore row order"


def test_pg_submit_empty_sol_sql_never_passes():
    """Missing/empty sol_sql must produce p1=False regardless of pred rows."""
    from bird_interact_agents.harness import _pg_execute_submit_action

    # pred returns empty result — which would falsely match empty gold
    # if we didn't guard against missing sol_sql
    ss = _make_sample_status("livesqlbench-base-lite", "alien", [])

    class _EmptyConn:
        def execute(self, q):
            return [], []

        def execute_sequence(self, sqls):
            return [], []

    class _CM:
        def __enter__(self):
            return _EmptyConn()

        def __exit__(self, *_):
            return False

    from unittest.mock import patch
    with patch("bird_interact_agents.harness.make_db_connection", return_value=_CM()):
        _, _, p1, _, _ = _pg_execute_submit_action("SELECT 1", ss, "/data")

    assert not p1, "missing sol_sql must never produce p1=True"


def test_slayer_mcp_config_derives_pgpassword_from_bird_pg_password(monkeypatch, tmp_path):
    """slayer_mcp_stdio_config must forward BIRD_PG_PASSWORD as PGPASSWORD
    when PGPASSWORD is not already set, so postgres SLayer datasources can
    authenticate without embedding the password in the datasource URL."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    monkeypatch.setenv("BIRD_PG_PASSWORD", "mysecret")
    monkeypatch.delenv("PGPASSWORD", raising=False)

    storage_dir = str(tmp_path / "storage")
    (tmp_path / "storage").mkdir()

    from unittest.mock import patch
    with patch("bird_interact_agents.harness._resolve_slayer_command", return_value="/slayer"):
        cfg = slayer_mcp_stdio_config(storage_dir, ingest_on_startup=False)

    assert cfg["env"].get("PGPASSWORD") == "mysecret"


def test_slayer_mcp_config_does_not_override_existing_pgpassword(monkeypatch, tmp_path):
    """If PGPASSWORD is already set, it must not be overwritten."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    monkeypatch.setenv("PGPASSWORD", "explicit")
    monkeypatch.setenv("BIRD_PG_PASSWORD", "other")

    storage_dir = str(tmp_path / "storage")
    (tmp_path / "storage").mkdir()

    from unittest.mock import patch
    with patch("bird_interact_agents.harness._resolve_slayer_command", return_value="/slayer"):
        cfg = slayer_mcp_stdio_config(storage_dir, ingest_on_startup=False)

    assert cfg["env"].get("PGPASSWORD") == "explicit"
