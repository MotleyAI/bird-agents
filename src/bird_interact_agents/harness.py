"""Thin adapter that imports BIRD-Interact's existing harness components.

`mini-interact-agent` is installed from the MotleyAI fork via
`uv sync --extra original` (see pyproject.toml); its `batch_run_bird_interact`
and `src.envs` packages are then importable from site-packages directly.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

from collections import Counter

from bird_interact_agents.benchmark import Benchmark, get_benchmark
from bird_interact_agents.db_connection import make_db_connection

logger = logging.getLogger(__name__)
from bird_interact_agents.hard8_preprocessor import (
    build_task_variant_storage,
    extract_deleted_kb_ids,
)

# ---------------------------------------------------------------------------
# Re-export harness components
# ---------------------------------------------------------------------------

# Action execution (SQLite DB operations + submission evaluation)
from batch_run_bird_interact.action_handler_sqlite import (
    execute_env_action as _sqlite_execute_env_action,
    execute_submit_action as _sqlite_execute_submit_action,
    load_db_data_if_needed,
    close_db_connection,
    get_db_connection,
    reset_and_reconnect_db,
    _schema_cache,
    _column_meanings_cache,
    _external_knowledge_cache,
    _filter_knowledge_for_agent,
    KNOWLEDGE_VISIBLE_FIELDS,
    MAX_RESULT_LENGTH,
)

# User simulator prompt building
from batch_run_bird_interact.prompt_utils import (
    build_user_encoder_prompt,
    build_user_decoder_prompt,
    parse_encoder_response,
)

# Sample status dataclass
from batch_run_bird_interact.sample_status import SampleStatus


# ---------------------------------------------------------------------------
# Postgres dispatch for execute_env_action / execute_submit_action
# ---------------------------------------------------------------------------


def _pg_execute_env_action(
    action: str, sample_status: "SampleStatus", data_path_base: str
) -> tuple[str, bool]:
    """Postgres-backend implementation of execute_env_action.

    Cache-based actions (schema, column meanings, external knowledge) work
    identically to the SQLite path — they read flat files populated by
    ``load_db_data_if_needed``. Only ``execute(...)`` is different: it
    opens a ``PostgresDbConnection`` instead of a SQLite connection.
    """
    db_name = sample_status.original_data["selected_database"]
    load_db_data_if_needed(db_name, data_path_base)

    try:
        if action.startswith("execute("):
            sql = action[8:-1].strip().strip("'\"")
            benchmark = get_benchmark(sample_status.original_data["dataset"])
            with make_db_connection(
                db_name, data_path_base=data_path_base, benchmark=benchmark, read_only=True
            ) as conn:
                rows, cols = conn.execute(sql)
            if rows:
                formatted = []
                for row in rows:
                    parts = []
                    for cell in row:
                        s = str(cell)
                        if len(s) > 100:
                            s = s[:97] + "..."
                        parts.append(s)
                    formatted.append(" | ".join(parts))
                observation = "\n".join(formatted)
                words = len(observation.split())
                if words > MAX_RESULT_LENGTH:
                    observation = " ".join(observation.split()[:MAX_RESULT_LENGTH]) + "..."
            else:
                observation = "Query executed, empty result set."
            return observation, True

        if action == "get_schema()":
            return _schema_cache.get(db_name, "Schema not available"), True

        if action == "get_all_column_meanings()":
            return json.dumps(_column_meanings_cache.get(db_name, {}), indent=2), True

        if action.startswith("get_column_meaning("):
            m = re.search(r"get_column_meaning\((.*)\)", action)
            if m:
                parts = [p.strip().strip("'\"") for p in m.group(1).strip().split(",")]
                if len(parts) == 2:
                    key = f"{db_name}|{parts[0].lower()}|{parts[1].lower()}"
                    return _column_meanings_cache.get(db_name, {}).get(key, "Column meaning not found"), True
                return "Error: Invalid arguments for get_column_meaning. Expected table_name, column_name.", False
            return "Error: Could not parse arguments for get_column_meaning.", False

        if action == "get_all_external_knowledge_names()":
            agent_kb = _filter_knowledge_for_agent(db_name, sample_status.original_data)
            return str(list(agent_kb.keys())), True

        if action.startswith("get_knowledge_definition("):
            m = re.search(r"get_knowledge_definition\((.*)\)", action)
            if m:
                name = m.group(1).strip().strip("'\"")
                agent_kb = _filter_knowledge_for_agent(db_name, sample_status.original_data)
                if name in agent_kb:
                    kb = agent_kb[name]
                    return json.dumps({k: kb[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in kb}, indent=2), True
                return "Knowledge not found or not accessible.", True
            return "Error: Could not parse arguments for get_knowledge_definition.", False

        if action == "get_all_knowledge_definitions()":
            agent_kb = _filter_knowledge_for_agent(db_name, sample_status.original_data)
            visible = [{k: v[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in v} for v in agent_kb.values()]
            return json.dumps(visible, indent=2), True

        return f"Unknown action: {action}", False

    except Exception as e:  # noqa: BLE001
        return f"SQL execution error: {e}", False


def _pg_execute_submit_action(
    sql: str, sample_status: "SampleStatus", data_path_base: str
) -> tuple[str, float, bool, bool, bool]:
    """Postgres-backend implementation of execute_submit_action.

    Runs ``sql`` (predicted) and each gold SQL in ``sol_sql`` against the
    shared postgres DB, compares result sets (unordered), and returns the
    standard ``(observation, reward, p1, p2, finished)`` tuple.
    """
    db_name = sample_status.original_data["selected_database"]
    sol_sqls = sample_status.original_data.get("sol_sql") or []
    if isinstance(sol_sqls, str):
        sol_sqls = [sol_sqls]
    benchmark = get_benchmark(sample_status.original_data["dataset"])

    def _run(query: str) -> tuple[list, bool]:
        try:
            with make_db_connection(
                db_name, data_path_base=data_path_base, benchmark=benchmark, read_only=True
            ) as conn:
                rows, _ = conn.execute(query)
            return rows, False
        except Exception:  # noqa: BLE001
            return [], True  # error flag

    pred_rows, pred_err = _run(sql)
    if pred_err:
        # For interactive benchmarks (one_shot=False) an execution error lets
        # the agent retry; for one-shot benchmarks the task is over.
        return "SQL execution error", 0.0, False, False, benchmark.one_shot

    # Execute all gold SQLs sequentially on a single shared connection so that
    # multi-statement sequences (CTEs, temp tables) work correctly.  This mirrors
    # tolerant_grader._multi_sql_execute semantics: the list is a sequence of
    # dependent steps, not a set of independent alternatives.
    def _run_gold_sequence(sqls: list[str]) -> tuple[list, bool]:
        try:
            with make_db_connection(
                db_name, data_path_base=data_path_base, benchmark=benchmark, read_only=True
            ) as conn:
                # execute_sequence issues ONE BEGIN READ ONLY / ROLLBACK for
                # the whole list, so temp tables created in step N remain
                # visible to step N+1 (unlike calling conn.execute per step,
                # which rolls back each statement individually).
                rows, _ = conn.execute_sequence(sqls)  # type: ignore[union-attr]
                return rows, False
        except Exception:  # noqa: BLE001
            return [], True

    gold_rows, gold_err = _run_gold_sequence(sol_sqls)
    # Counter comparison preserves duplicate rows (mirrors tolerant_grader)
    p1 = not gold_err and Counter(map(tuple, pred_rows)) == Counter(map(tuple, gold_rows))

    reward = 1.0 if p1 else 0.0
    obs = f"Submitted. Result match: {p1}"
    # p2=False: no second-pass grading metric for postgres benchmarks yet.
    # finished: one-shot benchmarks always end after one submission; interactive
    # ones only end when the agent's answer is correct.
    finished = benchmark.one_shot or p1
    return obs, reward, p1, False, finished


def execute_env_action(
    action: str, sample_status: "SampleStatus", data_path_base: str
) -> tuple[str, bool]:
    """Dispatch ``execute_env_action`` to the postgres or SQLite implementation."""
    dataset = sample_status.original_data.get("dataset", "")
    try:
        b = get_benchmark(dataset)
    except ValueError:
        b = None
    if b is not None and b.db_backend == "postgres":
        return _pg_execute_env_action(action, sample_status, data_path_base)
    return _sqlite_execute_env_action(action, sample_status, data_path_base)


def execute_submit_action(
    sql: str, sample_status: "SampleStatus", data_path_base: str
) -> tuple[str, float, bool, bool, bool]:
    """Dispatch ``execute_submit_action`` to the postgres or SQLite implementation."""
    dataset = sample_status.original_data.get("dataset", "")
    try:
        b = get_benchmark(dataset)
    except ValueError:
        b = None
    if b is not None and b.db_backend == "postgres":
        return _pg_execute_submit_action(sql, sample_status, data_path_base)
    return _sqlite_execute_submit_action(sql, sample_status, data_path_base)


# ---------------------------------------------------------------------------
# Upstream bugfix monkey-patch (kept here, not in the vendored source, so
# the fix travels with the rest of our changes and doesn't drift if the
# BIRD-Interact tree is ever re-checked-out).
#
# BIRD-Interact's `ex_base` returns 0 (fail) whenever EITHER the predicted
# or the gold result set is empty. That breaks tasks whose audited gold
# legitimately returns 0 rows (e.g. households_15 — "highly supported
# AND financially secure" is empty in this dataset). For those, empty
# == empty is a correct match and should pass. We replace `ex_base` with
# a copy that treats `[] == []` as a match while preserving all other
# behaviour.
def _patch_ex_base_empty_match() -> None:
    from src.envs.bird_interact_env.test_case_utils_sqlite import (  # noqa: E402
        test_utils as _tu,
    )

    _original_ex_base = _tu.ex_base

    def ex_base(pred_sqls, sol_sqls, db_path, conn, conditions=None):
        if not pred_sqls or not sol_sqls:
            return 0
        predicted_res, pred_err, pred_to = _tu.execute_queries(
            pred_sqls, db_path, conn, None, "",
        )
        ground_res, gt_err, gt_to = _tu.execute_queries(
            sol_sqls, db_path, conn, None, "",
        )
        if any([pred_err, pred_to, gt_err, gt_to]):
            return 0
        predicted_res = _tu.preprocess_results(predicted_res)
        ground_res = _tu.preprocess_results(ground_res)
        # BUGFIX: when both sides legitimately return 0 rows, that's a match.
        if not predicted_res and not ground_res:
            return 1
        if not predicted_res or not ground_res:
            return 0
        if conditions is not None and conditions.get("order", False):
            return 1 if predicted_res == ground_res else 0
        return 1 if set(predicted_res) == set(ground_res) else 0

    # Keep the original around for tests / introspection.
    ex_base._wraps = _original_ex_base  # type: ignore[attr-defined]
    _tu.ex_base = ex_base


_patch_ex_base_empty_match()

# Maximum number of assistant turns per task. Consumed by every adapter to
# cap runaway loops independent of the bird-coin budget.
MAX_MODEL_TURNS = 60

# Budget calculation helpers
ACTION_COSTS = {
    "execute_sql": 1,
    "get_schema": 1,
    "get_all_column_meanings": 1,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1,
    "ask_user": 2,
    "submit_sql": 3,
    "submit_query": 3,
    # SLayer tools
    "help": 0.5,
    "list_datasources": 0.5,
    "models_summary": 1,
    "inspect_model": 0.5,
    "search": 0.5,
    "query": 1,
}


def _ambiguity_count(task_data: dict) -> int:
    n = 0
    user_query_ambiguity = task_data.get("user_query_ambiguity", {})
    if "critical_ambiguity" in user_query_ambiguity:
        n += len(user_query_ambiguity["critical_ambiguity"])
    if "knowledge_ambiguity" in task_data:
        n += len(task_data["knowledge_ambiguity"])
    return n


def calculate_budget(
    task_data: dict, patience: int = 3, mode: str = "a-interact"
) -> float:
    """Calculate bird-coin budget for a task.

    a-interact: ENV_INTERACT(3) + SUBMIT(3) + 2*amb + 2*patience.
        Default patience=3 reproduces the original mini_interact_agent
        result with user_patience_budget=6 (= 12 + 2*amb).
    c-interact: ask_cost*(amb + patience) + submit_cost.
        Reproduces ADK's discrete turn budget (n_amb+patience clarification
        turns + 1 submit) using the existing coin plumbing.
    one-shot: fixed 30 (DEV-1462). One-shot strips ``ask_user`` from every
        role, so the pool is only drawn down by ``submit_query`` (cost 3).
        Turn-capping via ``MAX_MODEL_TURNS``/``request_limit`` is the real
        bound; the pool just needs to outlast ~9 submit attempts before
        ``force_submit`` trips (Plan B3).
    """
    amb = _ambiguity_count(task_data)
    if mode == "a-interact":
        return 6 + 2 * amb + 2 * patience
    if mode == "c-interact":
        return ACTION_COSTS["ask_user"] * (amb + patience) + ACTION_COSTS["submit_sql"]
    if mode == "one-shot":
        return 30.0
    raise ValueError(f"Unsupported budget mode: {mode}")


def update_budget(status: "SampleStatus", action_name: str) -> tuple[float, bool]:
    """Decrement remaining_budget by the cost of action_name.

    Mirrors the bookkeeping in the original mini_interact_agent's
    `update_budget` (see batch_run_bird_interact/main.py): subtract the cost,
    set force_submit when at-or-below cost. Returns the new remaining budget
    and the force_submit flag.
    """
    cost = ACTION_COSTS.get(action_name, 0)
    status.remaining_budget = max(0.0, status.remaining_budget - cost)
    if status.remaining_budget <= ACTION_COSTS["submit_sql"]:
        status.force_submit = True
    return status.remaining_budget, status.force_submit


def load_tasks(jsonl_path: str, limit: int | None = None) -> list[dict]:
    """Load tasks from a JSONL file."""
    import json

    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def load_benchmark_tasks(
    dataset: str,
    data_path: str,
    gold_file: str | None = None,
    *,
    limit: int | None = None,
    filter_ids: list[str] | None = None,
) -> list[dict]:
    """Benchmark-aware task loader — the single dispatch point both the local
    runner and the cloud actor call, so the loader selection lives in ONE place.

    A benchmark with a gated gold sidecar (``gold_required``) uses
    :func:`load_livesqlbench_tasks` (merges the gold by instance_id, stamps the
    dataset marker, filters to SELECT, is filter_ids-aware). Otherwise the plain
    :func:`load_tasks` path (gold is inline in the data JSONL), with an optional
    instance-id filter applied here.
    """
    b = get_benchmark(dataset)
    if b.gold_required:
        if not gold_file:
            raise ValueError(
                f"benchmark {b.name!r} requires a gold sidecar (gold_file); "
                "got none.",
            )
        return load_livesqlbench_tasks(
            data_path, gold_file, limit=limit, filter_ids=filter_ids,
            dataset_marker=b.dataset_marker,
        )
    # Apply `limit` AFTER `filter_ids` (not via load_tasks' own limit), so a
    # caller passing both doesn't have requested instance_ids truncated away
    # before filtering — mirrors the LiveSQLBench path (CodeRabbit).
    tasks = load_tasks(data_path)
    if filter_ids is not None:
        wanted = set(filter_ids)
        tasks = [t for t in tasks if t.get("instance_id") in wanted]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


# ---------------------------------------------------------------------------
# DEV-1462: LiveSQLBench loader.
#
# The public LiveSQLBench-Base-Lite jsonl ships task records WITHOUT
# `sol_sql`/`test_cases`/`external_knowledge` (they are gated). The user
# supplies the gold sidecar via ``--gold-file``; this loader merges it
# in by `instance_id`, maps `query`→`amb_user_query` (so existing
# agents and ``_ambiguity_count`` keep working without rewriting), stamps
# the dataset marker that drives per-task DB isolation + the one-shot
# `run_task` programmatic guard, and filters to SELECT tasks
# (`category=="Query"`). The full SELECT set is exactly 180.
# ---------------------------------------------------------------------------


# Expected SELECT-task count on a full unfiltered LiveSQLBench-Base-Lite run.
_LIVESQLBENCH_SELECT_FULL_RUN_COUNT = 180


def load_livesqlbench_tasks(
    data_path: str,
    gold_file: str,
    *,
    limit: int | None = None,
    filter_ids: list[str] | None = None,
    dataset_marker: str = "livesqlbench",
) -> list[dict]:
    """Load + merge a LiveSQLBench task batch.

    Pipeline (Plan B2):

      1. Read the public task jsonl at ``data_path``.
      2. Read the gated gold sidecar at ``gold_file`` and merge `sol_sql`,
         `external_knowledge`, `test_cases` per `instance_id`.
      3. Map `query` → `amb_user_query` (keep `query` for traceability).
      4. Stamp `task["dataset"] = "livesqlbench"` — the loader's irreducible
         marker for ``materialize_task_db`` + the one-shot ``run_task``
         programmatic guard.
      5. Filter to `category == "Query"` (authoritative); log a warning
         when an `_M_` instance_id appears with `category=="Query"` (or
         vice versa) — the substring is a DEFENSIVE cross-check, never
         a filter (Plan B2 step 5).
      6. Apply `filter_ids` if given so the empty-`sol_sql` fail-fast
         (step 8) only inspects tasks that will actually run (Codex #6).
      7. Assert exactly 180 SELECT tasks when neither `limit` nor
         `filter_ids` narrows the set — a silent dataset truncation
         must surface immediately, not produce a deceptive partial-run
         metric.
      8. Fail-fast on any KEPT task with empty `sol_sql` after the merge
         — that means the gold sidecar is incomplete for at least one
         task that would otherwise run.
      9. Apply `limit` AFTER the SELECT + `filter_ids` narrowing, so
         `--limit 180` doesn't accidentally yield < 180 if the dataset
         interleaves Management rows.

    Raises:
        FileNotFoundError: data_path or gold_file missing.
        json.JSONDecodeError / ValueError: malformed JSONL.
        AssertionError: full unfiltered run yielded != 180 SELECT rows.
        ValueError: a kept task has empty `sol_sql` post-merge.
    """
    import json

    data_path_p = Path(data_path)
    gold_path = Path(gold_file)
    if not data_path_p.is_file():
        raise FileNotFoundError(f"livesqlbench data file missing: {data_path_p}")
    if not gold_path.is_file():
        raise FileNotFoundError(
            f"livesqlbench gold-file missing (required for "
            f"--dataset livesqlbench): {gold_path}",
        )

    # Step 1 — public rows.
    public_rows: list[dict] = []
    with data_path_p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                public_rows.append(json.loads(line))

    # Step 2 — gold sidecar keyed by instance_id.
    gold_by_id: dict[str, dict] = {}
    with gold_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)  # bubbles JSONDecodeError on malformed lines
            inst = entry.get("instance_id")
            if inst:
                gold_by_id[inst] = entry

    # Steps 3-4 — merge + shim + stamp.
    merged: list[dict] = []
    for row in public_rows:
        inst = row.get("instance_id")
        gold = gold_by_id.get(inst, {}) if inst else {}
        # Field-by-field merge so the public row's other keys survive.
        if "sol_sql" in gold:
            row["sol_sql"] = gold["sol_sql"]
        if "external_knowledge" in gold:
            row["external_knowledge"] = gold["external_knowledge"]
        if "test_cases" in gold:
            row["test_cases"] = gold["test_cases"]
        # `query` → `amb_user_query` shim. Keep `query` for traceability.
        if "query" in row and "amb_user_query" not in row:
            row["amb_user_query"] = row["query"]
        row["dataset"] = dataset_marker
        merged.append(row)

    # Step 5 — SELECT filter; defensive `_M_` cross-check.
    select_rows: list[dict] = []
    for row in merged:
        cat = row.get("category")
        inst = row.get("instance_id", "")
        has_m = "_M_" in inst
        is_query = cat == "Query"
        if is_query and has_m:
            logger.warning(
                "livesqlbench loader: instance_id=%r has `_M_` substring "
                "but category==Query — keeping (authoritative signal is "
                "category); cross-check disagrees",
                inst,
            )
        elif (not is_query) and (not has_m) and cat == "Management":
            logger.warning(
                "livesqlbench loader: instance_id=%r is Management but "
                "lacks `_M_` substring — defensive cross-check disagrees",
                inst,
            )
        if is_query:
            select_rows.append(row)

    # Step 6 — filter_ids narrowing BEFORE the assert (so a partial-gold
    # run targeted via --instance-id doesn't trip step 8 on un-run rows).
    if filter_ids is not None:
        wanted = set(filter_ids)
        select_rows = [r for r in select_rows if r.get("instance_id") in wanted]

    # Step 7 — full-run check. Only when neither limit nor filter
    # narrows the set; otherwise a smaller count is expected by design.
    # NOT an `assert` because production guards must survive `python -O`
    # / `PYTHONOPTIMIZE`, which strips assertions (Codex review).
    if limit is None and filter_ids is None:
        if len(select_rows) != _LIVESQLBENCH_SELECT_FULL_RUN_COUNT:
            raise ValueError(
                f"expected exactly {_LIVESQLBENCH_SELECT_FULL_RUN_COUNT} "
                f"SELECT tasks on a full unfiltered LiveSQLBench run; got "
                f"{len(select_rows)}. Has the dataset been truncated?"
            )

    # Step 8 — empty `sol_sql` fail-fast on the kept set.
    missing_gold = [
        r["instance_id"] for r in select_rows
        if not r.get("sol_sql")
    ]
    if missing_gold:
        raise ValueError(
            f"livesqlbench gold sidecar is incomplete: "
            f"{len(missing_gold)} kept SELECT task(s) have empty `sol_sql` "
            f"after merge: {missing_gold[:5]}"
            + ("..." if len(missing_gold) > 5 else ""),
        )

    # Step 9 — limit AFTER filter.
    if limit is not None:
        select_rows = select_rows[:limit]
    return select_rows


def apply_audited_gold_overlay(
    tasks: list[dict],
    audited_root: str | Path,
    *,
    benchmark: Benchmark | None = None,
) -> dict[str, str]:
    """Swap each task's ``sol_sql`` for the audited version when available
    and record the pre-overlay gold so EVERY task scores against the
    original gold too.

    The on-disk layout depends on ``benchmark.audited_gold_layout``:

    * ``per_db`` (mini-interact's historical contract, also the default
      when ``benchmark is None``): one sidecar per DB at
      ``<audited_root>/<db>/<db>_audited.jsonl``. Cached on first read
      per DB.
    * ``single_file`` (DEV-1510 — livesqlbench): one consolidated JSONL
      at ``<audited_root>/<benchmark.name>_audited.jsonl``, with
      ``instance_id`` as the lookup key and ``selected_database`` as
      the per-DB discriminator on each row. Read ONCE per call.

    For each task whose ``instance_id`` is in the audit set AND whose
    ``audit_status`` is ``edited`` or ``unrecoverable``, replaces
    ``task["sol_sql"]`` in-place with the audited list. ``clean`` rows
    keep their ``sol_sql``. The upstream ``execute_submit_action`` reads
    ``sample_status.original_data["sol_sql"]`` by reference (see
    ``BIRD-Interact/.../action_handler.py``), so mutating the dict
    before ``SampleStatus`` is constructed is sufficient.

    For EVERY task (clean, edited, unrecoverable, and missing-sidecar
    alike), the pre-overlay gold is preserved as
    ``task["original_sol_sql"]`` so downstream dual-evaluation ALWAYS
    scores the agent's submission against the canonical/original gold —
    not just the rows the overlay rewrote.

    Returns a dict mapping ``instance_id`` -> overlay status
    (``"edited"|"unrecoverable"|"clean"|"missing-row"|"missing-file"``).
    Missing files / missing rows leave the task's ``sol_sql`` untouched
    (but still get ``original_sol_sql`` set, equal to that ``sol_sql``).

    A single-file row whose ``selected_database`` does not match the
    task's ``selected_database`` is treated as ``missing-row`` and logs
    a warning — defensive against cross-benchmark instance_id collision
    in the consolidated file.
    """
    import json

    layout = "per_db" if benchmark is None else benchmark.audited_gold_layout
    audited_root = Path(audited_root)
    overlay_log: dict[str, str] = {}

    if layout == "single_file":
        # `single_file` only ever appears when a Benchmark descriptor set it,
        # so `benchmark is not None` inside this branch — narrowed for the
        # type-checker.
        assert benchmark is not None
        # Read the consolidated audit file ONCE for the whole task list.
        # An absent file is benign: log once + return "missing-file" for
        # every task (without ever opening a per-db path that doesn't
        # exist in this layout).
        single_file_path = audited_root / f"{benchmark.name}_audited.jsonl"
        single_rows: dict[str, dict] | None
        if not single_file_path.exists():
            logger.warning(
                "audited-gold single_file missing for benchmark=%s: %s — "
                "falling back to original gold",
                benchmark.name, single_file_path,
            )
            single_rows = None
        else:
            single_rows = {}
            with single_file_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "invalid audited-gold JSON for benchmark=%s at %s: %s",
                            benchmark.name, single_file_path, e,
                        )
                        continue
                    inst_id = d.get("instance_id")
                    if not inst_id:
                        logger.warning(
                            "audited-gold row missing instance_id for benchmark=%s at %s",
                            benchmark.name, single_file_path,
                        )
                        continue
                    # Codex r9: DEV-1515 multi-variant audits ship N
                    # rows per instance_id (one ``primary=True`` plus
                    # non-primary alternates). Latest-wins would let a
                    # later-listed alternate overwrite the primary's
                    # audited_sol_sql, applying the wrong reading at
                    # overlay time. Prefer primary; once recorded,
                    # never overwrite. (A non-primary recorded first
                    # gets overwritten by the primary later in the
                    # file.)
                    existing = single_rows.get(inst_id)
                    if existing is None:
                        single_rows[inst_id] = d
                    elif existing.get("primary") is True:
                        # Already have the primary — keep it.
                        continue
                    elif d.get("primary") is True:
                        # Upgrade non-primary → primary.
                        single_rows[inst_id] = d
                    # else: both non-primary, keep the first one (no
                    # ordering preference between alternates).
        for task in tasks:
            inst = task.get("instance_id")
            db = task.get("selected_database")
            if not inst or not db:
                continue
            # Always snapshot the pre-overlay gold (same posture as per_db).
            pre_overlay = task.get("sol_sql")
            task["original_sol_sql"] = (
                list(pre_overlay) if isinstance(pre_overlay, list) else pre_overlay
            )
            if single_rows is None:
                overlay_log[inst] = "missing-file"
                continue
            entry = single_rows.get(inst)
            if entry is None:
                overlay_log[inst] = "missing-row"
                continue
            row_db = entry.get("selected_database")
            # Reject BOTH missing and mismatching `selected_database`. A row
            # with no `selected_database` is corrupt — applying it based on
            # `instance_id` alone would defeat the cross-benchmark collision
            # protection entirely (single_file is shared across DBs).
            if not row_db:
                logger.warning(
                    "audited-gold row for instance_id=%s has no "
                    "selected_database — treating as missing-row "
                    "(single_file layout requires the per-DB discriminator)",
                    inst,
                )
                overlay_log[inst] = "missing-row"
                continue
            if row_db != db:
                logger.warning(
                    "audited-gold row for instance_id=%s has "
                    "selected_database=%r but task carries selected_database=%r "
                    "— treating as missing-row to avoid applying the wrong "
                    "audit (single_file layout cross-benchmark guard)",
                    inst, row_db, db,
                )
                overlay_log[inst] = "missing-row"
                continue
            # Defence-in-depth: also verify the row's `benchmark` tag matches
            # the active benchmark. A row with the right (instance_id,
            # selected_database) but the wrong `benchmark` would still slip
            # past the selected_database check above (DB names overlap across
            # benchmarks by design — that's why we have single_file in the
            # first place). The schema requires `benchmark`, so an absent
            # field is treated the same as a mismatch.
            row_benchmark = entry.get("benchmark")
            if not row_benchmark or row_benchmark != benchmark.name:
                logger.warning(
                    "audited-gold row for instance_id=%s has "
                    "benchmark=%r but active benchmark is %r "
                    "— treating as missing-row to avoid applying the wrong "
                    "audit (single_file layout cross-benchmark guard)",
                    inst, row_benchmark, benchmark.name,
                )
                overlay_log[inst] = "missing-row"
                continue
            status = entry.get("audit_status")
            overlay_log[inst] = status or "missing-row"
            if status in ("edited", "unrecoverable"):
                audited = entry.get("audited_sol_sql")
                if isinstance(audited, list) and audited:
                    task["sol_sql"] = list(audited)
        return overlay_log

    # Default / per_db layout — the historical mini-interact path. Kept
    # bit-identical to the pre-DEV-1510 implementation so existing tests
    # and the cloud upload-back/merge contract don't drift.
    cache: dict[str, dict[str, dict]] = {}
    for task in tasks:
        inst = task.get("instance_id")
        db = task.get("selected_database")
        if not inst or not db:
            continue
        pre_overlay = task.get("sol_sql")
        task["original_sol_sql"] = (
            list(pre_overlay) if isinstance(pre_overlay, list) else pre_overlay
        )
        if db not in cache:
            path = audited_root / db / f"{db}_audited.jsonl"
            if not path.exists():
                cache[db] = {}
                logger.warning(
                    "audited-gold sidecar missing for db=%s: %s — falling back to original gold",
                    db, path,
                )
            else:
                rows: dict[str, dict] = {}
                with path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                "invalid audited-gold JSON for db=%s at %s: %s",
                                db, path, e,
                            )
                            continue
                        inst_id = d.get("instance_id")
                        if not inst_id:
                            logger.warning(
                                "audited-gold row missing instance_id for db=%s at %s",
                                db, path,
                            )
                            continue
                        rows[inst_id] = d  # latest-wins
                cache[db] = rows
        rows = cache[db]
        if not rows:
            overlay_log[inst] = "missing-file"
            continue
        entry = rows.get(inst)
        if entry is None:
            overlay_log[inst] = "missing-row"
            continue
        status = entry.get("audit_status")
        overlay_log[inst] = status or "missing-row"
        if status in ("edited", "unrecoverable"):
            audited = entry.get("audited_sol_sql")
            if isinstance(audited, list) and audited:
                task["sol_sql"] = list(audited)
    return overlay_log


def evaluate_dual_gold(
    *,
    pred_sql: str,
    audited_sol_sqls: list[str],
    original_sol_sqls: list[str],
    status,
    data_path_base: str,
) -> dict[str, dict]:
    """Score `pred_sql` against TWO golds — the audited reference (what
    the agent interacted with) and the original reference (the canonical
    benchmark). Returns ``{"audited": {...}, "original": {...}}`` with
    each side carrying ``{"p1": bool, "p2": bool, "reward": float,
    "observation": str}``.

    When ``audited_sol_sqls == original_sol_sqls`` (overlay no-op for a
    clean task), only ONE evaluator call runs and the result is copied
    to both sides — avoids doubling the wall on a 300-task run when
    most tasks are clean.

    If an evaluator call raises (malformed gold / infra blip), that
    side's payload carries ``p1=False`` and the exception text in
    ``observation``; the other side is unaffected. The caller's
    ``status.original_data["sol_sql"]`` is restored to its pre-call
    value before this helper returns, regardless of how the upstream
    evaluator interacts with it.
    """
    def _one_call(sol_sqls: list[str]) -> dict:
        # Upstream `execute_submit_action` reads gold from
        # `status.original_data["sol_sql"]` — swap it in for the call
        # and restore afterwards. Using try/finally so an exception
        # never leaves the status corrupted.
        original_data = status.original_data
        prev = original_data.get("sol_sql")
        original_data["sol_sql"] = list(sol_sqls)
        try:
            obs, reward, p1, p2, finished = execute_submit_action(
                pred_sql, status, data_path_base,
            )
            return {
                "p1": bool(p1),
                "p2": bool(p2),
                "finished": bool(finished),
                "reward": float(reward) if reward is not None else 0.0,
                "observation": str(obs),
            }
        except Exception as e:  # noqa: BLE001 — evaluator can raise broadly
            return {
                "p1": False,
                "p2": False,
                "finished": False,
                "reward": 0.0,
                "observation": f"evaluator raised: {type(e).__name__}: {e}",
            }
        finally:
            original_data["sol_sql"] = prev

    audited_result = _one_call(audited_sol_sqls)
    if list(audited_sol_sqls) == list(original_sol_sqls):
        return {"audited": audited_result, "original": dict(audited_result)}
    original_result = _one_call(original_sol_sqls)
    return {"audited": audited_result, "original": original_result}


# ---------------------------------------------------------------------------
# SLayer MCP server (stdio) — used by all framework agents in slayer mode.
# Each task spawns a per-DB instance pointing at the right model storage.
# ---------------------------------------------------------------------------

def _resolve_slayer_command() -> str:
    """Locate the slayer CLI binary.

    Prefers `.venv/bin/slayer` next to our package (so the spawned subprocess
    uses the same Python environment), falls back to `slayer` on PATH.
    """
    # The .venv lives at the repo root; src/bird_interact_agents/harness.py
    # is two levels deep below repo root.
    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_slayer = repo_root / ".venv" / "bin" / "slayer"
    if venv_slayer.is_file() and os.access(venv_slayer, os.X_OK):
        return str(venv_slayer)
    on_path = shutil.which("slayer")
    if on_path:
        return on_path
    raise RuntimeError(
        "slayer CLI not found. Install with `uv pip install 'motley-slayer[embedding-search]'` "
        "or `uv pip install -e '../slayer[embedding-search]'` and try again."
    )


def finalize_result_row(
    row: dict,
    *,
    deleted_kb_ids: list[int],
    slayer_storage_dir: str,
) -> dict:
    """Stamp HARD-8 bookkeeping onto an adapter's result row.

    ``variant_storage_path`` is set only when the row's task actually
    used a deletion variant (i.e. ``deleted_kb_ids`` is non-empty);
    otherwise it stays ``None`` so canonical-storage rows can be told
    apart from variant rows in the results JSON.
    """
    row["deleted_kb_ids"] = deleted_kb_ids
    row["variant_storage_path"] = slayer_storage_dir if deleted_kb_ids else None
    return row


def _task_variant_workdir(instance_id: str) -> Path:
    """Per-task scratch dir for HARD-8 variant storage.

    Lives under ``$TMPDIR/bird_interact_w5_variants/<instance_id>/``.
    Reused (overwritten) across runs of the same task — content is
    rewritten from scratch each time so stale deletions don't leak.
    """
    p = Path(tempfile.gettempdir()) / "bird_interact_w5_variants" / instance_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# DEV-1462: per-task DB isolation for LiveSQLBench.
#
# LiveSQLBench ships per-DB ``<db>_template.sqlite`` and the upstream eval
# rm+copies it onto ``<db>.sqlite`` on every submit. If multiple tasks share
# the stable dataset ``<db>.sqlite``, concurrent resets race the OTF cache
# build that ALSO reads it. Each LiveSQLBench task gets its own
# ``db_file_path`` in ``$TMPDIR/.../<instance_id>/<db>.sqlite``, with the
# template symlinked alongside — so the upstream reset operates inside
# the per-task dir and never touches the stable file OTF ingests.
# ---------------------------------------------------------------------------


def _livesqlbench_task_dbdir(instance_id: str) -> Path:
    """Per-task scratch dir for the LiveSQLBench per-task working sqlite.

    Lives under ``$TMPDIR/bird_interact_livesqlbench_db/<instance_id>/``.
    Created lazily on demand; never auto-cleaned (OS tmp hygiene owns it).
    """
    p = (
        Path(tempfile.gettempdir())
        / "bird_interact_livesqlbench_db"
        / instance_id
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


# The 16-byte magic at the start of every SQLite 3 database file. Used by
# `materialize_task_db`'s fast-path to reject 0-byte and foreign-shape
# files (e.g. LFS pointers) at the per-task working `<db>.sqlite` (DEV-1509).
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _has_sqlite_magic(path: Path) -> bool:
    """Return True iff the first 16 bytes of `path` are the SQLite 3 magic.

    Cheap (one open + 16-byte read). Catches the empirically observed
    0-byte case and foreign-shape files (LFS pointers, etc.). NOTE:
    does NOT detect deeper corruption of a SQLite-magic-starting file —
    that's out of scope; the upstream `reset_and_restore_database` is
    the authoritative restoration path.
    """
    try:
        with path.open("rb") as fh:
            return fh.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def materialize_task_db(
    task_data: dict, data_path_base: str | Path,
) -> str | None:
    """Materialise a per-task isolated working sqlite for a LiveSQLBench task.

    No-op (returns ``None``) for any task without
    ``task_data["dataset"] == "livesqlbench"`` — mini-interact + every
    other dataset keeps the existing shared-``<db>.sqlite`` flow.

    For LiveSQLBench tasks:

    * Refuse with a clear error if the DB name contains
      ``_ephemeral_``/``_process_`` (the upstream
      ``reset_and_restore_database`` splits ``base_db_name`` on these
      tokens; a matching DB name would mis-derive the template path).
      None of the 18 real LiveSQLBench DB names triggers this — the
      defensive check is a guard against a future name collision.
    * Create ``$TMPDIR/bird_interact_livesqlbench_db/<instance_id>/`` and
      symlink ``<data_path_base>/<db>/<db>_template.sqlite`` into it
      (the upstream reset reads
      ``<dirname(db_file_path)>/<base>_template.sqlite`` so the template
      MUST sit next to the working file).
    * Set ``task_data["db_file_path"] = <dir>/<db>.sqlite`` (the
      upstream ``_resolve_sqlite_db_path`` honours this when present).
    * Idempotent + stale-safe: if ``db_file_path`` is already set AND the
      symlinked template's target equals the current
      ``data_path_base/<db>/<db>_template.sqlite``, return unchanged.
      Otherwise rebuild the per-task dir so a rerun against a different
      ``--db-path`` does not silently reuse a stale symlink.

    Returns the resolved ``db_file_path`` (str) or ``None`` for the no-op
    branch. Called from ``run_oracle_task`` and each one-shot
    ``run_task`` setup before the first submit.
    """
    dataset = task_data.get("dataset", "")
    try:
        b = get_benchmark(dataset)
        if not b.per_task_db_isolation:
            return None
    except ValueError:
        return None

    db = task_data.get("selected_database")
    instance_id = task_data.get("instance_id")
    if not db or not instance_id:
        raise ValueError(
            "materialize_task_db: task_data is missing required keys "
            "`selected_database` and/or `instance_id`",
        )
    if "_ephemeral_" in db or "_process_" in db:
        raise ValueError(
            f"materialize_task_db: DB name {db!r} contains "
            f"`_ephemeral_`/`_process_` — the upstream "
            f"reset_and_restore_database would split base_db_name on "
            f"those tokens and look for the wrong template. Refusing "
            f"to materialise.",
        )

    data_root = Path(data_path_base)
    dataset_template = (data_root / db / f"{db}_template.sqlite").resolve()
    task_dir = _livesqlbench_task_dbdir(instance_id)
    expected_db_file = task_dir / f"{db}.sqlite"
    expected_template_link = task_dir / f"{db}_template.sqlite"

    # Fast-path: idempotence. Reuse the per-task dir iff (a) the task
    # already carries our `db_file_path`, (b) the template symlink
    # targets the current dataset template, AND (c) the working file is
    # a real SQLite (DEV-1509: pre-fix, any earlier RW-mode connect
    # could leave a 0-byte file here that ?mode=ro readers misinterpret
    # as "no such table"; the magic-header check rejects 0-byte AND
    # foreign shapes like LFS pointers).
    existing = task_data.get("db_file_path")
    if existing and Path(existing) == expected_db_file:
        if (
            expected_template_link.is_symlink()
            and expected_db_file.is_file()
            and _has_sqlite_magic(expected_db_file)
        ):
            try:
                existing_target = expected_template_link.resolve()
            except OSError:
                existing_target = None
            if existing_target == dataset_template:
                return str(expected_db_file)
        # Else fall through to rebuild — stale symlink, missing working
        # file, or corrupted/foreign-shape working file.

    # Rebuild the per-task dir. Both the template symlink AND the
    # working file are installed via per-call unique tmp paths +
    # `os.replace`, so two concurrent calls on the same instance_id
    # cannot collide on a shared `.part` slot or race os.symlink's
    # FileExistsError. `os.replace` atomically swaps the destination on
    # POSIX, regardless of whether it's missing, a regular file, or a
    # symlink — so the up-front `unlink()` of stale state is redundant.
    uniq = uuid.uuid4().hex[:8]

    # Template entry: SYMLINK ONLY — 18 templates × N tasks copied
    # would blow storage (DEV-1462 Plan B0).
    tmp_link = expected_template_link.with_name(
        f"{expected_template_link.name}.part-{uniq}"
    )
    os.symlink(dataset_template, tmp_link)
    os.replace(tmp_link, expected_template_link)

    # Working file: ATOMIC PRE-COPY (DEV-1509). Without this, the dry-run
    # gate in `agents/_submit.py::_dry_run_sql` opens the path ?mode=ro
    # and — if anything (e.g. upstream `get_db_connection` in default RW
    # mode) created an empty file here before the first reset — gets
    # `OperationalError: no such table: <name>`, which the agent
    # misdiagnoses as a casing problem. shutil.copy2 to a sibling tmp
    # then os.replace makes the destination switch atomic, so a
    # mid-copy failure (ENOSPC etc.) cannot leave a half-written file
    # at expected_db_file. The upstream `reset_and_restore_database`
    # is idempotent over this pre-copy (it `os.remove`s and
    # `shutil.copy2`s from template on every submit).
    tmp_db_file = expected_db_file.with_name(
        f"{expected_db_file.name}.part-{uniq}"
    )
    shutil.copy2(dataset_template, tmp_db_file)
    os.replace(tmp_db_file, expected_db_file)

    task_data["db_file_path"] = str(expected_db_file)
    return str(expected_db_file)


async def resolve_task_storage_dir(
    *,
    slayer_storage_root: Optional[str],
    db_name: str,
    task_data: dict,
    query_mode: str,
) -> Tuple[str, list[int]]:
    """Resolve the per-task SLayer storage path.

    Returns ``(slayer_storage_dir, deleted_kb_ids)``.

    - In raw mode or when ``slayer_storage_root`` is unset: returns
      ``("", [])``. (The downstream slayer MCP launch is gated on
      ``query_mode == "slayer"`` in each adapter, so the empty string
      never reaches ``slayer_mcp_stdio_config``.)
    - In slayer mode: ALWAYS materialises a per-task copy of the
      canonical ``<root>/<db_name>`` under
      ``$TMPDIR/bird_interact_w5_variants/<instance_id>/<db_name>/``,
      optionally with HARD-8 deletions applied. The canonical
      ``slayer_models/`` reference is therefore read-only at runtime —
      SLayer's first-load type-refinement writes and the agent's
      ``create_model`` / ``edit_model`` / ``delete_model`` calls land
      in the per-task scratch dir, never in the committed reference.
    """
    if query_mode != "slayer" or not slayer_storage_root:
        return "", []
    deleted = sorted(extract_deleted_kb_ids(task_data))
    instance_id = task_data["instance_id"]
    variant_dir = await build_task_variant_storage(
        canonical_storage_root=Path(slayer_storage_root),
        db_name=db_name,
        deleted_kb_ids=set(deleted),
        work_dir=_task_variant_workdir(instance_id),
    )
    return str(variant_dir), deleted


# Startup-handshake budget (seconds) for the SLayer stdio MCP server when
# `ingest_on_startup=True` (the committed-reference path — committed
# `slayer_models/<db>/` may have been built under a different embedding model
# than the one the runtime is using, so the slayer server's `--ingest-on-
# startup` refresh is the safety net). That refresh RE-REFLECTS the datasource
# schema and rebuilds the in-memory semantic layer before answering
# pydantic-ai's `initialize()` handshake. Reflection is pure CPU and scales
# with schema size — ~30-50s uncontended for a large schema (e.g. alien,
# 30+ models) — and balloons under multi-actor CPU contention. The prior 300s
# budget tripped exactly that way (DEV-1478). 1800s gives ≈30-50x margin.
#
# OTF callers (DEV-1508) opt out via `slayer_mcp_stdio_config(...,
# ingest_on_startup=False)` because the deterministic OTF cache IS the
# post-ingestion state; on those paths the handshake is sub-second and this
# budget is moot.
SLAYER_MCP_STARTUP_TIMEOUT_S = 1800


def slayer_mcp_stdio_config(
    storage_dir: str, *, ingest_on_startup: bool = True,
) -> dict:
    """Return a stdio MCP server config for the per-task slayer storage.

    Frameworks adapt this dict to their own MCP-server config type.

    Args:
        storage_dir: Per-task SLayer storage dir. Non-empty (a Path("")
            silently aliases to CWD on `.resolve()`, which would point
            SLayer at whatever directory the run happens to start in).
        ingest_on_startup: When True (default), pass ``--ingest-on-startup``
            to ``slayer mcp`` so the server refreshes datasource / model /
            column / measure / aggregation embeddings against the active
            embedding model at boot. Required for the committed-reference
            adapters (``pydantic_ai``, ``claude_sdk``, ``smolagents``,
            ``mcp_agent``, ``agno``) whose ``slayer_models/<db>/`` may have
            been embedded against a different model.

            When False (DEV-1508), the flag is omitted. OTF callers whose
            storage comes through ``cache.ensure_db_cache`` pass False:
            today that's ``claude_sdk_otf`` and ``pydantic_ai_recursive`` in
            on-the-fly mode. Their per-task storage is `cp -r`'d from the
            deterministic OTF cache, which already contains a fully
            ingested ``embeddings.db`` / ``memories.yaml`` / model YAML —
            re-ingesting is wasted work, and on the Claude Agent SDK path
            (no MCP startup-timeout knob) it leaves slayer
            ``status='pending'`` for the entire agent session. The cache's
            ``_impl_fp.txt`` marker (``slayer.__version__`` + embedding
            model) is recomputed on reuse and forces a rebuild on drift,
            so dropping this flag is safe under the project's locked-
            slayer-version invariant.

            ``pydantic_ai_otf_encode`` does NOT yet opt out: its storage
            comes from ``reference_build.ensure_db_reference``, whose
            reuse path is presence-gated on ``_reference_fp.txt`` without
            the impl-fingerprint check. Extending the impl-fp split to
            ``ensure_db_reference`` is a follow-up; until then the encoder
            keeps the startup ingest as its drift safety net.

    Keys:
        command: absolute path to the slayer binary
        args:    [`mcp`] (or [`mcp`, `--ingest-on-startup`] when enabled)
        env:     full env dict with SLAYER_STORAGE pointing at the per-DB store

    Raises:
        ValueError: if storage_dir is empty/None.
    """
    if not storage_dir:
        raise ValueError(
            "slayer_mcp_stdio_config requires a non-empty storage_dir; "
            "set --slayer-storage-root (or pass slayer_storage_root explicitly) "
            "when running slayer mode."
        )
    env = os.environ.copy()
    env["SLAYER_STORAGE"] = str(Path(storage_dir).resolve())
    args = ["mcp"]
    if ingest_on_startup:
        args.append("--ingest-on-startup")
    return {
        "command": _resolve_slayer_command(),
        "args": args,
        "env": env,
    }
