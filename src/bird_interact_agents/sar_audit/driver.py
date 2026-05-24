"""SAR-audit driver — orchestrates SAR-Agent over a mini-interact DB."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from bird_interact_agents import paths

from . import io, loader, prompt_wrapper, schema_renderer
from .normalizer import SampleRowResult, SARVerdict, to_normalized_row


SKILL_VERSION = "sar-agent/1.0"


class RunResult(BaseModel):
    db: str
    audited: int
    skipped: int
    failed: int
    audited_path: Path
    failures_path: Path


def open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    """Open `db_path` in read-only, immutable mode (no journal/WAL creation)."""
    target = f"file:{db_path}?mode=ro&immutable=1"
    return sqlite3.connect(target, uri=True)


def execute_sample_row(db_path: Path, sql: str) -> SampleRowResult:
    """Execute `sql` against `db_path` in read-only mode; return first row."""
    try:
        con = open_readonly_sqlite(db_path)
    except sqlite3.Error as exc:
        return SampleRowResult(row=None, status="error", error=str(exc))
    try:
        cur = con.execute(sql)
        row = cur.fetchone()
        if row is None:
            return SampleRowResult(row=None, status="empty")
        return SampleRowResult(row=list(row), status="ok")
    except sqlite3.Error as exc:
        return SampleRowResult(row=None, status="error", error=str(exc))
    finally:
        con.close()


def run_db_by_name(
    *,
    db: str,
    mini_interact_root: Path | None = None,
    sar_audited_gold_root: Path | None = None,
    audit_model: str = "claude-opus-4-7",
    max_steps: int = 30,
    save_trajectory: bool = False,
    force: bool = False,
    redo_instance: str | None = None,
    limit: int | None = None,
    sar_agent_factory: Callable | None = None,
    anthropic_client_factory: Callable | None = None,
) -> RunResult:
    """High-level entry point: load canonical files for `db`, run audit,
    write to `sar_audited_gold_root/<db>/<db>_sar_audited.jsonl`.
    """
    mi_root = mini_interact_root or paths.mini_interact_root()
    sar_root = sar_audited_gold_root or paths.sar_audited_gold_root()

    tasks = loader.load_task_list(db=db, mini_interact_path=mi_root / "mini_interact.jsonl")
    full_kb = loader.load_kb(db=db, mini_interact_root=mi_root)
    full_column_meanings = loader.load_column_meanings(db=db, mini_interact_root=mi_root)
    db_path = loader.locate_db_sqlite(db=db, mini_interact_root=mi_root)

    return run_db(
        db=db,
        tasks=tasks,
        db_path=db_path,
        full_kb=full_kb,
        full_column_meanings=full_column_meanings,
        audit_model=audit_model,
        max_steps=max_steps,
        save_trajectory=save_trajectory,
        force=force,
        redo_instance=redo_instance,
        limit=limit,
        output_dir=sar_root / db,
        sar_agent_factory=sar_agent_factory,
        anthropic_client_factory=anthropic_client_factory,
    )


def run_db(
    *,
    db: str,
    tasks: list[dict],
    db_path: Path,
    full_kb: list[dict],
    full_column_meanings: dict,
    audit_model: str = "claude-opus-4-7",
    max_steps: int = 30,
    save_trajectory: bool = False,
    force: bool = False,
    redo_instance: str | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
    sar_agent_factory: Callable | None = None,
    anthropic_client_factory: Callable | None = None,
) -> RunResult:
    """Low-level driver: caller provides everything explicit."""
    if output_dir is None:
        output_dir = paths.sar_audited_gold_root() / db
    output_dir.mkdir(parents=True, exist_ok=True)
    audited_path = output_dir / f"{db}_sar_audited.jsonl"
    failures_path = output_dir / f"{db}_sar_failures.jsonl"

    existing_rows = io.read_existing_rows(audited_path)
    existing_index = io.index_by_instance_id(existing_rows)

    if limit is not None:
        tasks = list(tasks)[:limit]

    schema_str = schema_renderer.render_schema(
        db_path=db_path, column_meanings=full_column_meanings
    )

    if sar_agent_factory is None:
        sar_agent_factory = _default_sar_agent_factory(
            anthropic_client_factory=anthropic_client_factory
        )

    audited_count = 0
    skipped_count = 0
    failed_count = 0

    for task in tasks:
        instance_id = task["instance_id"]
        should_redo = (
            force
            or instance_id == redo_instance
            or _should_redo(existing_index.get(instance_id), audit_model)
        )

        if instance_id in existing_index and not should_redo:
            skipped_count += 1
            continue

        wrapped_prompt = prompt_wrapper.render_prompt(
            task=task,
            full_kb=full_kb,
            full_column_meanings=full_column_meanings,
            schema_str=schema_str,
        )

        try:
            with tempfile.TemporaryDirectory() as td, contextlib.chdir(td):
                sar_agent = sar_agent_factory(
                    model=audit_model,
                    prompt=wrapped_prompt,
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    db_path=db_path,
                )
                run_result = sar_agent.run(max_steps=max_steps)
        except Exception as exc:  # noqa: BLE001 — any LLM/loop error is a task failure
            io.append_failure(
                failures_path,
                {
                    "instance_id": instance_id,
                    "selected_database": db,
                    "failed_at": _now_iso(),
                    "audit_model_requested": audit_model,
                    "error_class": type(exc).__name__,
                    "error_message": str(exc),
                    "step_count": getattr(exc, "step_count", 0),
                    "skill_version": SKILL_VERSION,
                },
            )
            failed_count += 1
            continue

        verdict_obj = run_result.verdict
        if not isinstance(verdict_obj, SARVerdict):
            verdict_obj = SARVerdict(
                correctness_flag=verdict_obj.correctness_flag,
                ambiguity_flag=verdict_obj.ambiguity_flag,
                revised_sql=verdict_obj.revised_sql,
                revised_question=verdict_obj.revised_question,
                reasoning=verdict_obj.reasoning,
            )

        # Choose the SQL to execute for sample_row.
        # Mirror normalizer's choice without computing the full row twice.
        executed_sql = _sample_sql_for(verdict_obj, task["sol_sql"][0])
        sample_row_result = execute_sample_row(db_path, executed_sql)

        row = to_normalized_row(
            task=task,
            verdict=verdict_obj,
            sample_row_result=sample_row_result,
            audit_model_requested=audit_model,
            audit_model_actual=getattr(run_result, "audit_model_actual", None),
            step_count=getattr(run_result, "step_count", 0),
            cost_usd=getattr(run_result, "cost_usd", None),
            skill_version=SKILL_VERSION,
            raw_trajectory=(
                getattr(run_result, "raw_trajectory", None) if save_trajectory else None
            ),
        )

        # If we're redoing a row that already exists, remove the old row first.
        if instance_id in existing_index:
            _rewrite_without(audited_path, instance_id)
        io.append_row(audited_path, row)
        existing_index[instance_id] = row
        audited_count += 1

    return RunResult(
        db=db,
        audited=audited_count,
        skipped=skipped_count,
        failed=failed_count,
        audited_path=audited_path,
        failures_path=failures_path,
    )


def _should_redo(existing: dict | None, audit_model: str) -> bool:
    if existing is None:
        return False
    if existing.get("audit_model_requested") != audit_model:
        return True
    if existing.get("skill_version") != SKILL_VERSION:
        return True
    return False


def _sample_sql_for(verdict: SARVerdict, original_sql: str) -> str:
    """Mirror normalizer.audited_sol_sql selection so the sample row is for
    the SQL we actually emit."""
    has_revision = verdict.revised_sql is not None and verdict.revised_sql.strip() != ""
    if verdict.ambiguity_flag and has_revision:
        return verdict.revised_sql  # type: ignore[return-value]
    if not verdict.correctness_flag and not verdict.ambiguity_flag and has_revision:
        return verdict.revised_sql  # type: ignore[return-value]
    return original_sql


def _rewrite_without(path: Path, instance_id: str) -> None:
    """Drop the row matching `instance_id` from the JSONL (used by redo).
    Reads + rewrites atomically via temp file + rename."""
    if not path.exists():
        return
    rows = io.read_existing_rows(path)
    kept = [r for r in rows if r["instance_id"] != instance_id]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_sar_agent_factory(
    *, anthropic_client_factory: Callable | None = None
) -> Callable:
    """Returns a factory that constructs a `SARAuditLoop` per task.

    Path B: we don't import upstream's `SARAgent` class. We reuse upstream's
    prompt template (in `prompt_wrapper`) and tool schemas (in `audit_loop`),
    but drive the Anthropic call loop ourselves. `anthropic_client_factory`
    is unused — kept for backward-compat with the driver signature; tests
    inject their own factory directly.
    """
    from .audit_loop import SARAuditLoop

    def factory(*, model: str, prompt: str, api_key: str | None, db_path: Path):
        return SARAuditLoop(
            model=model,
            prompt=prompt,
            api_key=api_key,
            db_path=db_path,
        )

    return factory
