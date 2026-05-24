"""Thin adapter that imports BIRD-Interact's existing harness components.

`mini-interact-agent` is installed from the MotleyAI fork via
`uv sync --extra original` (see pyproject.toml); its `batch_run_bird_interact`
and `src.envs` packages are then importable from site-packages directly.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

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
    execute_env_action,
    execute_submit_action,
    load_db_data_if_needed,
    close_db_connection,
    get_db_connection,
    reset_and_reconnect_db,
    _schema_cache,
    _column_meanings_cache,
    _external_knowledge_cache,
    _filter_knowledge_for_agent,
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
    """
    amb = _ambiguity_count(task_data)
    if mode == "a-interact":
        return 6 + 2 * amb + 2 * patience
    if mode == "c-interact":
        return ACTION_COSTS["ask_user"] * (amb + patience) + ACTION_COSTS["submit_sql"]
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


def apply_audited_gold_overlay(
    tasks: list[dict],
    audited_root: str | Path,
) -> dict[str, str]:
    """Swap each task's ``sol_sql`` for the audited version when available.

    Looks for ``<audited_root>/<db>/<db>_audited.jsonl`` per task's
    ``selected_database``. For each task whose ``instance_id`` is in
    the sidecar AND whose ``audit_status`` is ``edited`` or
    ``unrecoverable``, replaces ``task["sol_sql"]`` in-place with the
    audited list. ``clean`` rows are a no-op. The upstream
    ``execute_submit_action`` reads
    ``sample_status.original_data["sol_sql"]`` by reference (see
    ``BIRD-Interact/.../action_handler.py``), so mutating the dict
    before ``SampleStatus`` is constructed is sufficient.

    When the overlay rewrites ``sol_sql`` (i.e. for ``edited`` /
    ``unrecoverable`` rows), the pre-overlay value is preserved as
    ``task["original_sol_sql"]`` so downstream dual-evaluation can
    score the agent's submission against both golds. ``clean`` and
    missing-row tasks have no ``original_sol_sql`` key (since
    ``sol_sql`` IS the original — no point double-evaluating identical
    SQL).

    Returns a dict mapping ``instance_id`` -> overlay status
    (``"edited"|"unrecoverable"|"clean"|"missing-row"|"missing-file"``).
    Missing files / missing rows leave the task's gold untouched.
    """
    import json

    audited_root = Path(audited_root)
    overlay_log: dict[str, str] = {}
    cache: dict[str, dict[str, dict]] = {}

    for task in tasks:
        inst = task.get("instance_id")
        db = task.get("selected_database")
        if not inst or not db:
            continue
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
                # Preserve the pre-overlay gold so downstream dual-eval
                # can score against both the agent's reference (audited)
                # and the canonical reference (original). Copying the
                # list defensively — caller may still mutate task in
                # other ways.
                pre_overlay = task.get("sol_sql")
                if isinstance(pre_overlay, list):
                    task["original_sol_sql"] = list(pre_overlay)
                else:
                    task["original_sol_sql"] = pre_overlay
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

import os as _os
import shutil as _shutil
from pathlib import Path as _Path


def _resolve_slayer_command() -> str:
    """Locate the slayer CLI binary.

    Prefers `.venv/bin/slayer` next to our package (so the spawned subprocess
    uses the same Python environment), falls back to `slayer` on PATH.
    """
    # The .venv lives at the repo root; src/bird_interact_agents/harness.py
    # is two levels deep below repo root.
    repo_root = _Path(__file__).resolve().parent.parent.parent
    venv_slayer = repo_root / ".venv" / "bin" / "slayer"
    if venv_slayer.is_file() and _os.access(venv_slayer, _os.X_OK):
        return str(venv_slayer)
    on_path = _shutil.which("slayer")
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


def slayer_mcp_stdio_config(storage_dir: str) -> dict:
    """Return a stdio MCP server config for the per-task slayer storage.

    Frameworks adapt this dict to their own MCP-server config type.

    Keys:
        command: absolute path to the slayer binary
        args:    [`mcp`]
        env:     full env dict with SLAYER_STORAGE pointing at the per-DB store

    Raises:
        ValueError: if storage_dir is empty/None. We refuse to silently fall
            back to CWD because Path("").resolve() does — that would point
            SLayer at whatever directory the run happens to start in.
    """
    if not storage_dir:
        raise ValueError(
            "slayer_mcp_stdio_config requires a non-empty storage_dir; "
            "set --slayer-storage-root (or pass slayer_storage_root explicitly) "
            "when running slayer mode."
        )
    env = _os.environ.copy()
    env["SLAYER_STORAGE"] = str(_Path(storage_dir).resolve())
    return {
        "command": _resolve_slayer_command(),
        # --ingest-on-startup refreshes datasource / model / column /
        # measure / aggregation embeddings against the active embedding
        # model when the server boots. Memory embeddings remain stale
        # (DEV-1416) — the kb-to-slayer-models skill writes them once
        # at save-time. The per-task variant carries the canonical
        # store's `embeddings.db` verbatim (see hard8_preprocessor's
        # `_copy_memories_and_embeddings`) so memory rows are already
        # present; this flag covers the entity side.
        "args": ["mcp", "--ingest-on-startup"],
        "env": env,
    }
