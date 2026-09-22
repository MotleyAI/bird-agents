"""Submission output writer: submission.jsonl + email_title.txt + manifest.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bird_interact_agents.reports.cost import (
    FIXED_COSTS,
    SECTION_VI_THRESHOLDS,
)
from bird_interact_agents.reports.schema import SubmissionRow
from bird_interact_agents.reports.tokens import DEFAULT_MODEL


_BENCHMARK_TO_SPLIT: dict[str, str] = {
    "bird-interact-lite-exp": "lite",
    "bird-interact-full": "full",
    "mini-interact": "mini-interact",
}


def build_email_title(
    *, benchmark: str, setting: str, team: str, method: str
) -> str:
    if benchmark not in _BENCHMARK_TO_SPLIT:
        raise ValueError(
            f"benchmark {benchmark!r} is not an a-Interact split; "
            f"supported: {sorted(_BENCHMARK_TO_SPLIT)}"
        )
    split = _BENCHMARK_TO_SPLIT[benchmark]
    return f"[BIRD-INTERACT-1.0-{split}][{setting}][{team}][{method}]"


@dataclass
class ManifestPlan:
    benchmark: str
    setting: str
    split: str
    team: str
    method: str
    tag: str
    selection_mode: str
    source_run_ids: list[str]
    generated_at: str
    instances: list[dict[str, Any]] = field(default_factory=list)
    patience_resolution: list[dict[str, Any]] = field(default_factory=list)
    leakage_check: dict[str, Any] | None = None
    warnings_by_instance: list[dict[str, Any]] = field(default_factory=list)


def write_submission(
    *, rows: list[SubmissionRow], plan: ManifestPlan, out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. submission.jsonl
    jsonl_path = out_dir / "submission.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            obj = row.model_dump()
            f.write(json.dumps(obj, separators=(",", ":")))
            f.write("\n")

    # 2. email_title.txt — no trailing newline.
    title = build_email_title(
        benchmark=plan.benchmark,
        setting=plan.setting,
        team=plan.team,
        method=plan.method,
    )
    (out_dir / "email_title.txt").write_text(title)

    # 3. manifest.json
    manifest = {
        "schema_version": 1,
        "kind": "bird_interact_submission_manifest",
        "generated_at": plan.generated_at,
        "benchmark": plan.benchmark,
        "split": plan.split,
        "setting": plan.setting,
        "team": plan.team,
        "method": plan.method,
        "tag": plan.tag,
        "n_instances": len(rows),
        "selection_mode": plan.selection_mode,
        "source_run_ids": plan.source_run_ids,
        "instances": plan.instances,
        "patience_resolution": plan.patience_resolution,
        "section_vi_threshold": SECTION_VI_THRESHOLDS,
        "fixed_costs": FIXED_COSTS,
        "tokenizer": f"anthropic.messages.count_tokens(model={DEFAULT_MODEL})",
        "leakage_check": plan.leakage_check,
        "warnings_by_instance": plan.warnings_by_instance,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir
