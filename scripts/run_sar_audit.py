#!/usr/bin/env python3
"""CLI wrapper for `bird_interact_agents.sar_audit.driver.run_db_by_name`.

Usage:
    python scripts/run_sar_audit.py --db credit [--limit N]
                                   [--audit-model claude-opus-4-7]
                                   [--max-steps 30] [--force]
                                   [--redo-instance credit_5]
                                   [--save-trajectory]

Exits non-zero if any task in the run failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from bird_interact_agents.sar_audit import driver


def _stub_upstream() -> None:
    """Test hook: when forced clean/failure mode is on, stub
    `load_prompt_template` so the prompt wrapper doesn't trip on a missing
    submodule."""
    from bird_interact_agents.sar_audit import _upstream_import

    STUB = (
        "<<TEST_STUB_TEMPLATE>>\n"
        "Question: {question}\n"
        "Schema: {schema}\n"
        "External Knowledge: {external_knowledge}\n"
        "Annotated Query: {gold_query}\n"
    )

    _upstream_import.load_prompt_template = lambda: STUB  # type: ignore[assignment]


def _force_failure_factory() -> Callable:
    """Test hook: SAR_AUDIT_FORCE_FAILURE=1 makes every task raise."""

    def factory(**kwargs):
        class Erroring:
            def run(self, *, max_steps: int):
                raise RuntimeError("SAR_AUDIT_FORCE_FAILURE injected error")

        return Erroring()

    return factory


def _force_clean_factory() -> Callable:
    """Test hook: SAR_AUDIT_FORCE_CLEAN=1 makes every task return clean."""
    from bird_interact_agents.sar_audit.normalizer import SARVerdict
    from pydantic import BaseModel

    class _Result(BaseModel):
        verdict: SARVerdict
        step_count: int = 1
        cost_usd: float | None = 0.0
        audit_model_actual: str | None = "test-model"
        raw_trajectory: list | None = None

    def factory(**kwargs):
        class Cleanish:
            def run(self, *, max_steps: int):
                return _Result(
                    verdict=SARVerdict(
                        correctness_flag=True,
                        ambiguity_flag=False,
                        reasoning="forced clean for test",
                    )
                )

        return Cleanish()

    return factory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run SAR-Agent audit on a mini-interact DB.")
    ap.add_argument("--db", required=True, help="mini-interact database name (e.g. credit)")
    ap.add_argument("--limit", type=int, default=None, help="cap tasks (debug)")
    ap.add_argument(
        "--audit-model",
        default="claude-opus-4-7",
        help="LiteLLM-style model string (default: claude-opus-4-7)",
    )
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="redo every task even if present")
    ap.add_argument("--redo-instance", default=None)
    ap.add_argument("--save-trajectory", action="store_true")
    args = ap.parse_args(argv)

    factory = None
    if os.environ.get("SAR_AUDIT_FORCE_FAILURE") == "1":
        print(
            "WARNING: SAR_AUDIT_FORCE_FAILURE=1 — outputs are synthetic test artifacts, "
            "not real SAR-Agent audits.",
            file=sys.stderr,
        )
        factory = _force_failure_factory()
        _stub_upstream()
    elif os.environ.get("SAR_AUDIT_FORCE_CLEAN") == "1":
        print(
            "WARNING: SAR_AUDIT_FORCE_CLEAN=1 — outputs are synthetic test artifacts, "
            "not real SAR-Agent audits.",
            file=sys.stderr,
        )
        factory = _force_clean_factory()
        _stub_upstream()

    result = driver.run_db_by_name(
        db=args.db,
        audit_model=args.audit_model,
        max_steps=args.max_steps,
        save_trajectory=args.save_trajectory,
        force=args.force,
        redo_instance=args.redo_instance,
        limit=args.limit,
        sar_agent_factory=factory,
    )

    print(
        f"db={result.db} audited={result.audited} skipped={result.skipped} failed={result.failed}"
    )
    print(f"audited_path={result.audited_path}")
    if result.failed > 0:
        print(f"failures_path={result.failures_path}")
    return 1 if result.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
