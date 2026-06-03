"""Shared per-task helpers used by every framework adapter.

`submit_raw_sql`, `submit_slayer_query`, `ask_user_impl`, and
`run_env_action` are the authoritative implementations of the
operations each adapter exposes as tools. Adapter files contain only
the framework-specific decoration; the bodies live here.

State is duck-typed — every adapter's per-task object (`TaskDeps`,
`TaskState`, contextvar dict) just needs to expose the attributes the
helpers touch (`status`, `data_path_base`, `usage`, `result`,
`user_sim_model`, `user_sim_prompt_version`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from typing import Any, Callable

from pydantic import ValidationError

from bird_interact_agents.agents._tool_specs import ToolSpec, render_action
from bird_interact_agents.benchmark import get_benchmark as _get_benchmark
from bird_interact_agents.db_connection import make_db_connection
from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_query_payload,
)
from bird_interact_agents.harness import (
    ACTION_COSTS,
    _schema_cache,
    build_user_decoder_prompt,
    build_user_encoder_prompt,
    evaluate_dual_gold,
    execute_env_action,
    execute_submit_action,
    parse_encoder_response,
    update_budget,
)
from bird_interact_agents.usage import acompletion_tracked

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic capture: snapshot result rows for offline failure analysis.
# ---------------------------------------------------------------------------

# Number of rows kept in the snapshot's `sample_rows`. Bigger samples make the
# results.db payloads heavy without adding much analytic value — most failure
# modes show up in the first row or in the column header.
_SNAPSHOT_SAMPLE_SIZE = 5
# Total rows examined to compute `row_count`. Past this we mark the snapshot
# truncated so analysis code can be aware (BIRD-Interact's own MAX_ROWS=10000).
_SNAPSHOT_MAX_ROWS = 10000


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    return value


def _first_sql(value: Any) -> str | None:
    """sol_sql is sometimes a string, sometimes a list; pick the first
    non-empty string."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
    return None


def capture_result_snapshot(
    sql: str | None,
    db_name: str,
    data_path_base: str,
    db_file_path: str | None = None,
    benchmark: Any = None,
) -> dict | None:
    """Run ``sql`` against the DB and return a serialisable snapshot —
    column names + inferred Python types + row count + a small head sample.

    Routes through ``DbConnection`` so it works for both SQLite and Postgres
    benchmarks. Returns None when ``sql`` is empty or (SQLite only) the DB
    file is absent. On any runtime error returns ``{"error": "..."}`` rather
    than raising, so failures here don't sink the run.
    """
    if not sql or not sql.strip():
        return None

    if getattr(benchmark, "db_backend", "sqlite") == "postgres":
        try:
            # Wrap in a subquery so we cap the fetch at source rather than
            # loading the entire result set with fetchall() before slicing.
            _snapshot_sql = (
                f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _snap"
                f" LIMIT {_SNAPSHOT_MAX_ROWS + 1}"
            )
            with make_db_connection(db_name, benchmark=benchmark, read_only=True) as conn:
                rows, col_names = conn.execute(_snapshot_sql)
            truncated = len(rows) > _SNAPSHOT_MAX_ROWS
            if truncated:
                rows = rows[:_SNAPSHOT_MAX_ROWS]
            sample = rows[:_SNAPSHOT_SAMPLE_SIZE]
            types: list[str] = []
            for i, _ in enumerate(col_names):
                inferred = "null"
                for row in rows:
                    if i < len(row) and row[i] is not None:
                        inferred = type(row[i]).__name__
                        break
                types.append(inferred)
            return {
                "columns": [
                    {"name": n, "type": t} for n, t in zip(col_names, types, strict=True)
                ],
                "row_count": len(rows),
                "row_count_truncated": truncated,
                "sample_rows": [
                    [_jsonable(v) for v in row] for row in sample
                ],
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("capture_result_snapshot failed for %s: %s", db_name, e)
            return {"error": f"{type(e).__name__}: {e}"}

    # SQLite path
    if db_file_path:
        db_path = db_file_path
    else:
        db_path = os.path.join(data_path_base, db_name, f"{db_name}.sqlite")
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(_SNAPSHOT_MAX_ROWS + 1)
            truncated = len(rows) > _SNAPSHOT_MAX_ROWS
            if truncated:
                rows = rows[:_SNAPSHOT_MAX_ROWS]
            col_names = [d[0] for d in (cur.description or [])]
            sample = rows[:_SNAPSHOT_SAMPLE_SIZE]
            types_: list[str] = []
            for i, _ in enumerate(col_names):
                inferred = "null"
                for row in rows:
                    if i < len(row) and row[i] is not None:
                        inferred = type(row[i]).__name__
                        break
                types_.append(inferred)
            return {
                "columns": [
                    {"name": n, "type": t} for n, t in zip(col_names, types_, strict=True)
                ],
                "row_count": len(rows),
                "row_count_truncated": truncated,
                "sample_rows": [
                    [_jsonable(v) for v in row] for row in sample
                ],
            }
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 — capture every failure mode
        logger.debug("capture_result_snapshot failed for %s: %s", db_name, e)
        return {"error": f"{type(e).__name__}: {e}"}


def classify_submission(
    *,
    p1: bool,
    p2: bool,
    observation: str | None,
    json_failed: bool = False,
    translation_failed: bool = False,
    dry_run_failed: bool = False,
    infrastructure_failed: bool = False,
) -> str:
    """Bucket one submission outcome into a coarse status string.

    The string is the primary axis the user uses for offline failure-mode
    analysis. SQL-runtime vs. wrong-result is detected by string-matching the
    canonical evaluator's observation message (see
    `execute_submit_action`).
    """
    if json_failed:
        return "json_error"
    if translation_failed:
        return "translation_error"
    if dry_run_failed:
        return "dry_run_error"
    if infrastructure_failed:
        return "infrastructure_error"
    if p2:
        return "passed_phase2"
    if p1:
        return "passed_phase1"
    obs = (observation or "").lower()
    if (
        "error executing submitted sql" in obs
        or "submitted sql execution timed out" in obs
        or "error processing submission" in obs
    ):
        return "sql_runtime_error"
    return "wrong_result"


def _diagnostic_payload(
    *,
    submitted_sql: str | None,
    sample_status: Any,
    data_path_base: str,
    observation: str | None,
    p1: bool,
    p2: bool,
    benchmark: Any = None,
    json_failed: bool = False,
    translation_failed: bool = False,
    dry_run_failed: bool = False,
    infrastructure_failed: bool = False,
    phase1_observation_audited: str | None = None,
    phase1_observation_original: str | None = None,
) -> dict[str, Any]:
    """Build the dict that `submit_*` writes onto `state.result` for the
    diagnostic columns: predicted/gold snapshots + classifier verdict +
    the raw evaluator observation, keyed by the phase the call ran in."""
    db_name = sample_status.original_data["selected_database"]
    sol_sql = _first_sql(sample_status.original_data.get("sol_sql"))
    pre_phase = getattr(sample_status, "current_phase", 1)
    db_file_path = sample_status.original_data.get("db_file_path")

    # Dry-run failures still capture predicted/gold snapshots — the
    # predicted snapshot will surface the same DB error as a
    # `{"error": ...}` blob, which is useful offline. JSON / translation
    # failures have no submitted_sql to snapshot.
    skip_snapshots = json_failed or translation_failed
    predicted = (
        capture_result_snapshot(
            submitted_sql, db_name, data_path_base, db_file_path=db_file_path,
            benchmark=benchmark,
        )
        if not skip_snapshots else None
    )
    gold = (
        capture_result_snapshot(
            sol_sql, db_name, data_path_base, db_file_path=db_file_path,
            benchmark=benchmark,
        )
        if not skip_snapshots else None
    )

    payload: dict[str, Any] = {
        "submission_status": classify_submission(
            p1=p1, p2=p2, observation=observation,
            json_failed=json_failed,
            translation_failed=translation_failed,
            dry_run_failed=dry_run_failed,
            infrastructure_failed=infrastructure_failed,
        ),
        "predicted_result_json": (
            json.dumps(predicted, default=str)
            if predicted is not None else None
        ),
        "gold_result_json": (
            json.dumps(gold, default=str) if gold is not None else None
        ),
    }
    if pre_phase == 2:
        payload["phase2_observation"] = observation
    else:
        payload["phase1_observation"] = observation
    # DEV-1515: per-task pass-fail bool fields against audited/original
    # gold have been REMOVED — all per-task verdicts now live in the
    # SubmissionAnnotation (produced inline by grade_and_write). The
    # observation snapshots are kept as diagnostic-only context for
    # log inspection.
    if phase1_observation_audited is not None:
        payload["phase1_observation_audited"] = phase1_observation_audited
    if phase1_observation_original is not None:
        payload["phase1_observation_original"] = phase1_observation_original
    return payload


# ---------------------------------------------------------------------------
# Budget bookkeeping. Centralised so every adapter shares one authoritative
# budget-update path; `claude_sdk.agent._gate` does pre-call rejection only.
# ---------------------------------------------------------------------------

def _budget_note(state: Any) -> str:
    status = state.status
    return (
        f"\n\n[Remaining budget: {status.remaining_budget:.1f}"
        f" / {status.total_budget:.1f}]"
    )


def gate_or_none(state: Any, action_name: str, query_mode: str) -> str | None:
    """Return a "you must submit" message if the call should be rejected,
    or None if it should proceed.

    Submit tools always proceed (they're the way out of force_submit).
    Honors `state.status.force_submit` and respects the right submit-tool
    name for the active query mode.
    """
    if action_name.startswith("submit_"):
        return None
    submit_tool = "submit_query" if query_mode == "slayer" else "submit_sql"
    submit_cost = ACTION_COSTS[submit_tool]
    cost = ACTION_COSTS.get(action_name, 0)
    if state.status.force_submit or state.status.remaining_budget < cost + submit_cost:
        return (
            f"Budget exhausted ({state.status.remaining_budget:.1f} remaining, "
            f"{action_name} costs {cost}). You MUST call {submit_tool} now "
            "with your best answer."
        )
    return None


# ---------------------------------------------------------------------------
# bird-interact discovery + submission
# ---------------------------------------------------------------------------

def run_env_action(
    state: Any, spec: ToolSpec, query_mode: str = "raw", **kwargs: str,
) -> str:
    """Render `spec` to an action string and dispatch via the harness.

    Applies budget gating + bookkeeping so a non-submit tool is rejected
    when `state.status.force_submit` is set or budget would drop below
    submit cost. Successful calls update_budget and append a remaining-
    budget note.
    """
    err = gate_or_none(state, spec.name, query_mode)
    if err is not None:
        return err
    action = render_action(spec, **kwargs)
    observation, _ = execute_env_action(action, state.status, state.data_path_base)
    update_budget(state.status, spec.name)
    return str(observation) + _budget_note(state)


def _dispatch_eval(state: Any, sql: str):
    """Run the submission evaluator and return
    ``(observation, reward, p1, p2, finished, audited_passes, original_passes,
       audited_observation, original_observation)``.

    On single-eval runs (no overlay applied to the task) the dual fields
    are ``None``. On dual-eval runs (``task["original_sol_sql"]`` set by
    `apply_audited_gold_overlay`) the helper runs both golds via
    `evaluate_dual_gold` — the AUDITED side drives the agent-visible
    return values (so the agent grades against what it interacted with);
    the ORIGINAL side goes into the diagnostic payload only.
    """
    original_sol_sql = state.status.original_data.get("original_sol_sql")
    if not original_sol_sql:
        # Single-eval path — current behavior, unchanged.
        observation, reward, p1, p2, finished = execute_submit_action(
            sql, state.status, state.data_path_base,
        )
        return (observation, reward, p1, p2, finished,
                None, None, None, None)

    # Dual-eval: audited gold drives `state.result` + agent feedback;
    # original gold lands in the diagnostic columns for offline scoring.
    audited_sol_sql = state.status.original_data["sol_sql"]
    dual = evaluate_dual_gold(
        pred_sql=sql,
        audited_sol_sqls=list(audited_sol_sql) if isinstance(audited_sol_sql, list) else [audited_sol_sql],
        original_sol_sqls=list(original_sol_sql) if isinstance(original_sol_sql, list) else [original_sol_sql],
        status=state.status,
        data_path_base=state.data_path_base,
    )
    aud = dual["audited"]
    orig = dual["original"]
    return (
        aud["observation"],
        aud["reward"],
        aud["p1"],
        aud["p2"],
        # `finished` comes from upstream verbatim — for mini-interact
        # (no phase 2) a phase-1 pass returns finished=True even when
        # p2=False, and we must preserve that for downstream consumers.
        aud["finished"],
        aud["p1"],
        orig["p1"],
        aud["observation"],
        orig["observation"],
    )


def _dry_run_sql(
    sql: str,
    *,
    data_path_base: str,
    db_name: str,
    db_file_path: str | None = None,
    benchmark: Any = None,
) -> str | None:
    """Execute ``sql`` read-only against the per-task DB and return None on
    success or a short error string on failure.

    Used as a FREE pre-eval gate inside ``submit_raw_sql`` and
    ``submit_slayer_query``: if the candidate SQL has a DB-side error
    (missing column / syntax / etc.), the agent gets that error back without
    burning the 3-coin submit cost.

    For Postgres benchmarks routes through ``DbConnection`` (read_only=True
    wraps in BEGIN/ROLLBACK so writes are never committed).

    For SQLite: prefer the template DB at
    ``data_path_base/<db_name>/<db_name>_template.sqlite`` if present
    (canonical reset state), else the live DB. Returns None when the DB file
    is missing — misconfigured envs don't block valid submissions.
    """
    if not sql or not sql.strip():
        return None

    if getattr(benchmark, "db_backend", "sqlite") == "postgres":
        try:
            with make_db_connection(db_name, benchmark=benchmark, read_only=True) as conn:
                conn.execute(sql)
            return None
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    # SQLite path
    if db_file_path:
        db_path = db_file_path
    else:
        template = os.path.join(
            data_path_base, db_name, f"{db_name}_template.sqlite",
        )
        if os.path.exists(template):
            db_path = template
        else:
            db_path = os.path.join(data_path_base, db_name, f"{db_name}.sqlite")
    if not os.path.exists(db_path):
        return None
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=30)
    except sqlite3.Error as e:
        return f"{type(e).__name__}: {e}"
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchone()
    except sqlite3.Error as e:
        return f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    return None


def _dry_run_error_message(error: str) -> str:
    """Standard agent-facing wrapper for a dry-run failure. Names the
    no-charge guarantee and points at the cheap schema-inspection tools
    so the next attempt is informed."""
    return (
        f"Submitted SQL failed dry-run (DB error): {error}\n\n"
        "Submission was NOT charged — fix the SQL and retry. To verify "
        "schema names before resubmitting, call `inspect_model` or "
        "`models_summary` (1 coin each)."
    )


def submit_raw_sql(state: Any, sql: str) -> str:
    """Submit a raw SQL query and record the submission on `state.result`.

    `submit_sql` is exempt from gate rejection — it's the way out of
    force_submit — but it still calls update_budget so subsequent reward
    accounting matches the upstream harness.
    """
    pre_phase = getattr(state.status, "current_phase", 1)

    # FREE dry-run gate: catch DB-side errors (missing function /
    # missing column) before paying the 3-coin submit cost. Mirrors
    # `json_failed` / `translation_failed` short-circuits in
    # submit_slayer_query.
    db_name = state.status.original_data["selected_database"]
    db_file_path = state.status.original_data.get("db_file_path")
    _dataset = state.status.original_data.get("dataset", "")
    _benchmark = None
    if _dataset:
        try:
            _benchmark = _get_benchmark(_dataset)
        except ValueError:
            pass
    dry_err = _dry_run_sql(
        sql,
        data_path_base=state.data_path_base,
        db_name=db_name,
        db_file_path=db_file_path,
        benchmark=_benchmark,
    )
    if dry_err is not None:
        msg = _dry_run_error_message(dry_err)
        diag = _diagnostic_payload(
            submitted_sql=sql,
            sample_status=state.status,
            data_path_base=state.data_path_base,
            observation=msg,
            p1=False, p2=False,
            benchmark=_benchmark,
            dry_run_failed=True,
        )
        prior = state.result or {}
        state.result = {
            **prior,
            "phase1_passed": False,
            "phase2_passed": False,
            "total_reward": 0.0,
            "finished": False,
            "submitted_sql": sql,
            "submitted_query": None,
            **diag,
        }
        if pre_phase == 1:
            state.result["phase2_observation"] = prior.get("phase2_observation")
        else:
            state.result["phase1_observation"] = prior.get("phase1_observation")
        return msg + _budget_note(state)

    infra_failed = False
    audited_obs = original_obs = None
    try:
        (observation, reward, p1, p2, finished,
         _audited_p1, _original_p1, audited_obs, original_obs) = _dispatch_eval(state, sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("execute_submit_action raised on %s", sql[:80])
        observation = f"Error processing submission: {e}"
        reward, p1, p2, finished = 0.0, False, False, False
        infra_failed = True
    update_budget(state.status, "submit_sql")
    diag = _diagnostic_payload(
        submitted_sql=sql,
        sample_status=state.status,
        data_path_base=state.data_path_base,
        observation=observation,
        p1=p1,
        p2=p2,
        benchmark=_benchmark,
        infrastructure_failed=infra_failed,
        phase1_observation_audited=audited_obs,
        phase1_observation_original=original_obs,
    )
    prior = state.result or {}
    state.result = {
        **prior,
        "phase1_passed": p1,
        "phase2_passed": p2,
        "total_reward": reward if reward is not None else 0.0,
        "finished": finished,
        "submitted_sql": sql,
        "submitted_query": None,
        # Diagnostic fields. _diagnostic_payload writes phaseN_observation
        # only for the phase this call ran in; we keep the other one from
        # `prior` so a phase-2 submission doesn't blank out a stored phase-1
        # message.
        **diag,
    }
    if pre_phase == 1:
        state.result["phase2_observation"] = prior.get("phase2_observation")
    else:
        state.result["phase1_observation"] = prior.get("phase1_observation")
    return str(observation) + _budget_note(state)


_SHAPE_ERROR_TEMPLATE = (
    "Invalid query JSON: expected either a single SlayerQuery object "
    '{{"source_model": ..., ...}} or a nested-DAG array '
    '[ {{"name": ..., "source_model": ..., ...}}, ..., '
    '{{"source_model": ..., ...}} ] (last element is the DAG root). '
    "Got: {got}."
)

_WRAPPED_KEYS = ("queries", "nested_queries")


def _shape_error_message(parsed: Any) -> str:
    """Build the user-facing message for a JSON that's neither a single
    SlayerQuery object nor a nested-DAG array. The Got: snippet is
    truncated so a giant blob doesn't dominate the LLM context."""
    snippet = json.dumps(parsed, default=str)
    if len(snippet) > 200:
        snippet = snippet[:200] + "..."
    return _SHAPE_ERROR_TEMPLATE.format(got=snippet)


def _is_wrapped_shape(parsed: Any) -> bool:
    """Detect the wrapped-shape attempts the mental_4 trajectory burned
    bird-coins on: `{queries: [...]}`, `{nested_queries: [...]}`, and
    `{source_model: ..., queries: [...]}`.

    Rule: a top-level dict carrying a list value under `queries` or
    `nested_queries` is treated as a wrapped variant. We catch these
    before sql_sync so the rejection message can name the two valid
    shapes (instead of letting Pydantic surface 'Extra inputs not
    permitted')."""
    if not isinstance(parsed, dict):
        return False
    return any(isinstance(parsed.get(k), list) for k in _WRAPPED_KEYS)


def _compile_sql(client: Any, parsed: Any) -> str:
    """Compile a parsed SLayer query (single-stage dict or nested list)
    to SQL.

    Primary path: `client.sql_sync(parsed)`. For lists, this works
    natively once slayer relaxes the dict-only gate in `query_sync`
    (tracked under DEV-1437).

    Brace for current slayer: when `sql_sync(list)` raises, fall back
    to `client._engine.execute_sync(query=parsed, dry_run=True).sql` —
    the engine has always accepted lists. AttributeError covers
    slayer 0.6.8's HTTP path (which reaches `query.model_dump(...)`
    on a list); ValidationError and TypeError cover the plausible
    rejection modes a refactored `query_sync` would emit.

    The brace gets removed once `pyproject.toml`'s slayer floor bumps
    past the DEV-1437 release.
    """
    try:
        return client.sql_sync(parsed)
    except (AttributeError, TypeError, ValidationError):
        if isinstance(parsed, list):
            engine = getattr(client, "_engine", None)
            if engine is not None:
                return engine.execute_sync(query=parsed, dry_run=True).sql
        raise


def submit_slayer_query(
    state: Any,
    query_json: str,
    slayer_client_factory: Callable[[Any], Any],
) -> str:
    """Submit a SLayer JSON query: render to SQL, then evaluate.

    The JSON string may decode to either shape:

    * **Single-stage**: a JSON object that validates as a
      ``SlayerQuery`` — e.g. ``{"source_model": "orders",
      "dimensions": ["status"], "measures": ["amount:sum"]}``.
    * **Nested DAG**: a JSON array of stage dicts where the last
      element is the DAG root — e.g. ``[{"name": "monthly",
      "source_model": "orders", ...}, {"source_model": "monthly",
      ...}]``. Same shape ``query_nested`` MCP tool accepts.

    Any other top-level shape (scalar, wrapped object like
    ``{"queries": [...]}``) is rejected with a sharp error message
    naming the two valid shapes, before ``sql_sync`` is called. See
    DEV-1435 for the failure mode this prevents.

    Records both the original JSON DSL and the rendered SQL on
    ``state.result``. Budget bookkeeping mirrors ``submit_raw_sql``.
    """
    pre_phase = getattr(state.status, "current_phase", 1)
    prior = state.result or {}
    _benchmark = None  # populated below after the JSON/shape gates

    def _record(*, sql: str | None, observation: str | None,
                reward: float, p1: bool, p2: bool, finished: bool,
                json_failed: bool = False, translation_failed: bool = False,
                dry_run_failed: bool = False,
                infrastructure_failed: bool = False,
                phase1_observation_audited: str | None = None,
                phase1_observation_original: str | None = None) -> None:
        diag = _diagnostic_payload(
            submitted_sql=sql,
            sample_status=state.status,
            data_path_base=state.data_path_base,
            observation=observation,
            p1=p1, p2=p2,
            benchmark=_benchmark,
            json_failed=json_failed,
            translation_failed=translation_failed,
            dry_run_failed=dry_run_failed,
            infrastructure_failed=infrastructure_failed,
            phase1_observation_audited=phase1_observation_audited,
            phase1_observation_original=phase1_observation_original,
        )
        state.result = {
            **prior,
            "phase1_passed": p1,
            "phase2_passed": p2,
            "total_reward": reward,
            "finished": finished,
            "submitted_sql": sql,
            "submitted_query": query_json,
            **diag,
        }
        if pre_phase == 1:
            state.result["phase2_observation"] = prior.get("phase2_observation")
        else:
            state.result["phase1_observation"] = prior.get("phase1_observation")

    # DEV-1432 broader budget rule: `submit_query` charges the 3-coin
    # submit cost IFF the call reached `execute_submit_action`. Every
    # deterministic pre-eval failure (json_failed, shape-error,
    # translation_failed) is FREE — the agent can retry without burning
    # bird-coins on shape-debugging.
    try:
        parsed = json.loads(query_json)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON — submission aborted: {e}"
        _record(sql=None, observation=msg, reward=0.0,
                p1=False, p2=False, finished=False, json_failed=True)
        return msg + _budget_note(state)

    # Top-level shape gate. Anything that isn't a single-stage object or
    # a nested-DAG array (and any wrapped-object variant the agent might
    # guess) gets a sharp error pointing at the two valid shapes —
    # cheaper than letting Pydantic surface "Extra inputs not permitted"
    # downstream.
    if not isinstance(parsed, (dict, list)) or _is_wrapped_shape(parsed):
        msg = _shape_error_message(parsed)
        _record(sql=None, observation=msg, reward=0.0,
                p1=False, p2=False, finished=False, json_failed=True)
        return msg + _budget_note(state)

    # DEV-1478: deterministically normalize text-equality FILTER predicates
    # (wrap the column in lower(trim(...)), lowercase the literal) so a NL
    # question that carries no casing info matches all case/whitespace
    # variants. Only the structured `filters` clause is touched — projections /
    # dimensions / joins are left alone. The recorded `submitted_query` below
    # stays the agent's ORIGINAL DSL; only the compiled `submitted_sql`
    # reflects the normalization.
    parsed = normalize_query_payload(parsed)

    client = slayer_client_factory(state)
    try:
        sql = _compile_sql(client, parsed)
    except Exception as e:
        msg = f"Could not generate SQL — submission aborted: {e}"
        _record(sql=None, observation=msg, reward=0.0,
                p1=False, p2=False, finished=False, translation_failed=True)
        return msg + _budget_note(state)

    # FREE dry-run gate: catch DB-side errors in the rendered SQL
    # (missing function / missing column / etc.) before paying the
    # 3-coin submit cost. Mirrors the `json_failed` / `translation_failed`
    # short-circuits above.
    db_name = state.status.original_data["selected_database"]
    db_file_path = state.status.original_data.get("db_file_path")
    _dataset = state.status.original_data.get("dataset", "")
    _benchmark = None
    if _dataset:
        try:
            _benchmark = _get_benchmark(_dataset)
        except ValueError:
            pass
    dry_err = _dry_run_sql(
        sql,
        data_path_base=state.data_path_base,
        db_name=db_name,
        db_file_path=db_file_path,
        benchmark=_benchmark,
    )
    if dry_err is not None:
        msg = _dry_run_error_message(dry_err)
        _record(sql=sql, observation=msg, reward=0.0,
                p1=False, p2=False, finished=False, dry_run_failed=True)
        return msg + _budget_note(state)

    infra_failed = False
    audited_obs = original_obs = None
    try:
        (observation, reward, p1, p2, finished,
         _audited_p1, _original_p1, audited_obs, original_obs) = _dispatch_eval(state, sql)
    except Exception as e:  # noqa: BLE001
        logger.exception("execute_submit_action raised on slayer-rendered SQL")
        observation = f"Error processing submission: {e}"
        reward, p1, p2, finished = 0.0, False, False, False
        infra_failed = True
    update_budget(state.status, "submit_query")
    _record(
        sql=sql,
        observation=observation,
        reward=reward if reward is not None else 0.0,
        p1=p1, p2=p2, finished=finished,
        infrastructure_failed=infra_failed,
        phase1_observation_audited=audited_obs,
        phase1_observation_original=original_obs,
    )
    return f"Generated SQL:\n{sql}\n\nResult: {observation}" + _budget_note(state)


# ---------------------------------------------------------------------------
# User simulator (encoder + decoder via litellm)
# ---------------------------------------------------------------------------

async def ask_user_impl(
    state: Any, question: str, query_mode: str | None = None,
) -> str:
    """Two-stage user simulator: encoder extracts an intent, decoder
    renders the user's reply. Both LLM calls are routed through
    `acompletion_tracked` so usage lands on `state.usage`.

    When `query_mode` is provided, applies the same budget gating as
    `run_env_action`: a force_submit-set status returns a "you must
    submit" message instead of running the user-sim. Successful calls
    decrement the budget by the cost of `ask_user`.
    """
    if query_mode is not None:
        err = gate_or_none(state, "ask_user", query_mode)
        if err is not None:
            return err

    db_name = state.status.original_data["selected_database"]
    schema = _schema_cache.get(db_name, "")

    encoder_prompt = build_user_encoder_prompt(
        question, state.status, schema, state.user_sim_prompt_version,
    )
    encoder_resp = await acompletion_tracked(
        state.usage,
        scope="user_sim",
        model=state.user_sim_model,
        messages=[{"role": "user", "content": encoder_prompt}],
    )
    encoder_text = encoder_resp.choices[0].message.content or ""
    encoder_action = parse_encoder_response(encoder_text)

    transcript = getattr(state, "user_sim_transcript", None)
    if transcript is not None:
        transcript.append({
            "phase": "encoder",
            "agent_question": question,
            "prompt": encoder_prompt,
            "response": encoder_text,
        })

    decoder_prompt = build_user_decoder_prompt(
        question, encoder_action, state.status, schema, state.user_sim_prompt_version,
    )
    decoder_resp = await acompletion_tracked(
        state.usage,
        scope="user_sim",
        model=state.user_sim_model,
        messages=[{"role": "user", "content": decoder_prompt}],
    )
    raw_response = decoder_resp.choices[0].message.content or ""

    if transcript is not None:
        transcript.append({
            "phase": "decoder",
            "agent_question": question,
            "prompt": decoder_prompt,
            "response": raw_response,
        })

    match = re.search(r"<s>(.*?)</s>", raw_response, re.DOTALL)
    answer = match.group(1).strip() if match else raw_response.strip()

    if query_mode is not None:
        update_budget(state.status, "ask_user")
        return answer + _budget_note(state)
    return answer
