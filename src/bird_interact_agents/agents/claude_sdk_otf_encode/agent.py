"""DEV-1609: a first-class framework adapter around the DEV-1589 claude_sdk
reference encoder, so the cloud submit path can OTF-encode with claude_sdk
(any registry / open-weight model) — not just the legacy pydantic path.

``ClaudeSDKOtfEncodeAgent`` is BUILD-ONLY by design (DEV-1609 D1): its
``run_task`` does exactly what ``scripts/build_otf_references.py`` does
locally — construct the claude_sdk build-encoder and call
``ensure_db_reference`` to build the canonical per-DB
``slayer_models_otf/<benchmark>/<db>`` reference, which the cloud merge-back
uploads home. There is NO per-task HARD-8 masking and NO agentic eval loop;
that machinery belongs to the (deprecated) interactive adapters. This keeps
cloud encoding behaviourally identical to local encoding.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
    make_claude_sdk_build_encoder,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.harness import (
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.slayer_otf import ensure_db_reference
from bird_interact_agents.slayer_otf.reference_build import _SETUP_USAGE
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


def _encode_result_row(
    *,
    instance_id: str,
    db_name: str,
    kb_encoded: list,
    usage: dict,
    error: str | None,
) -> dict:
    """Finalized result row for a build-only encode task. Mirrors the legacy
    encode's minimal-row key set (parity for batch evaluation / serialisation)
    but carries NO submission/grading — an encode run only builds a reference.
    """
    row = {
        "task_id": instance_id,
        "instance_id": instance_id,
        "database": db_name,
        "phase1_passed": False,
        "phase2_passed": False,
        "total_reward": 0.0,
        "submitted_sql": None,
        "submitted_query": None,
        "trajectory": {"final_output_excerpt": "", "agents": []},
        "error": error,
        "usage": usage,
        "submission_status": "never_submitted",
        "phase1_observation": None,
        "phase2_observation": None,
        "predicted_result_json": None,
        "gold_result_json": None,
        "n_agent_turns": None,
        "tool_call_stats": None,
        "phase1_observation_audited": None,
        "phase1_observation_original": None,
        "kb_encoded": kb_encoded,
    }
    # No per-task masking in an encode run → no deletion variant.
    return finalize_result_row(row, deleted_kb_ids=[], slayer_storage_dir="")


def _read_setup_usage(reference_dir: Path) -> dict:
    """Best-effort read of the encoder's ``_setup_usage.json`` from the built
    reference dir. Present only on a fresh build; absent on a cache reuse (no
    LLM ran → zero usage is correct)."""
    fp = Path(reference_dir) / _SETUP_USAGE
    try:
        return TokenUsage.model_validate_json(fp.read_text()).model_dump()
    except Exception:  # noqa: BLE001 — telemetry is best-effort
        return TokenUsage().model_dump()


class ClaudeSDKOtfEncodeAgent:
    """Build-only OTF *reference* encoder driven by the claude_sdk SDK."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        reasoning_effort: str | None = None,
        slayer_setup: str = "on-the-fly",
    ) -> None:
        if slayer_setup != "on-the-fly":
            raise ValueError(
                "claude_sdk_otf_encode requires slayer_setup='on-the-fly'; "
                f"got {slayer_setup!r}"
            )
        # The claude_sdk encoder takes the RAW model string (e.g.
        # ``zai/glm-5.2``); the hermetic session layers provider auth.
        self.slayer_storage_root = slayer_storage_root
        self.model_id = model
        self.reasoning_effort = reasoning_effort
        self.slayer_setup = slayer_setup

    async def run_task(
        self,
        task_data: dict,
        data_path_base: str,
        budget: float,
        query_mode: str,
        eval_mode: str = "a-interact",
        user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version: str = "v2",
    ) -> dict:
        if query_mode != "slayer":
            raise ValueError(
                "claude_sdk_otf_encode supports only --query-mode slayer; "
                f"got {query_mode!r}"
            )
        if eval_mode != "a-interact":
            raise ValueError(
                "claude_sdk_otf_encode supports only --mode a-interact; "
                f"got {eval_mode!r}"
            )

        _dataset = task_data.get("dataset")
        if not _dataset:
            raise ValueError("task_data missing required 'dataset' field")
        benchmark_obj = get_benchmark(_dataset)
        benchmark: str = benchmark_obj.name

        db_name = task_data["selected_database"]
        instance_id = task_data["instance_id"]

        try:
            load_db_data_if_needed(db_name, data_path_base)
            # DEV-1462 B0: LiveSQLBench tasks get a per-task isolated
            # ``db_file_path`` (no-op for mini-interact).
            materialize_task_db(task_data, data_path_base)

            build_encoder = make_claude_sdk_build_encoder(
                model=self.model_id,
                self_model_id=native_model_id(self.model_id),
                reasoning_effort=self.reasoning_effort,
            )
            db_root_resolved = Path(data_path_base).resolve()
            # Build (or reuse) the canonical full-KB reference — the exact call
            # `scripts/build_otf_references.py::_build_one` makes. The cloud
            # merge-back uploads `slayer_models_otf/<benchmark>/<db>` home.
            entry = await ensure_db_reference(
                db_name,
                reference_root=paths.slayer_models_otf_root(
                    benchmark=benchmark
                ),
                cache_root=paths.slayer_otf_cache_root(benchmark=benchmark),
                mini_interact_root=db_root_resolved,
                build_encoder=build_encoder,
                db_root=db_root_resolved,
                benchmark=benchmark_obj,
            )
            kb_encoded = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in (entry.setup_results or [])
            ]
            usage = _read_setup_usage(entry.reference_dir)
            return _encode_result_row(
                instance_id=instance_id,
                db_name=db_name,
                kb_encoded=kb_encoded,
                usage=usage,
                error=None,
            )
        except Exception as e:  # noqa: BLE001 — finalize an error row, keep batch alive
            logger.exception(
                "claude_sdk OTF-encode error on %s (%s): %s",
                instance_id, db_name, e,
            )
            return _encode_result_row(
                instance_id=instance_id,
                db_name=db_name,
                kb_encoded=[],
                usage=TokenUsage().model_dump(),
                error=str(e),
            )
