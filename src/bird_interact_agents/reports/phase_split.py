"""Per-submit phase classifier.

The bird-interact-tools submit tool emits one of the verdict strings in
``action_handler_sqlite``:

* ``Phase 1 SQL Correct! (Reward: X points). Moving to Phase 2.``
* ``Phase 1 SQL Correct! (Reward: X points). No Phase 2. Task finished.``
* ``Phase 2 SQL Correct! (Reward: X points). Task finished.``
* ``Submitted SQL failed test case in Phase {1|2}. Reason: …``

We classify each observation directly from the marker text; only when
markers are missing or ordered inconsistently do we fall back to a
warning-emitting heuristic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_RE_CORRECT = re.compile(r"Phase\s+([12])\s+SQL\s+Correct", re.IGNORECASE)
_RE_WRONG = re.compile(
    r"Submitted\s+SQL\s+failed\s+test\s+case\s+in\s+Phase\s+([12])",
    re.IGNORECASE,
)


def _to_text(observation: Any) -> str:
    """Normalise observation to a single string. Tool_result content can
    arrive as a string, a list of text-blocks ({type: text, text: …}),
    or a bare list of strings."""
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, list):
        chunks: list[str] = []
        for item in observation:
            if isinstance(item, dict):
                # Anthropic SDK content block.
                if "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(str(item))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(observation)


def classify_submit_observation(observation: Any) -> tuple[str | None, str | None]:
    """Return ``(phase, verdict)`` for one submit observation.

    ``phase`` ∈ {"phase1", "phase2", None}. ``verdict`` ∈ {"correct",
    "wrong", None}. Both ``None`` when no marker is present.
    """
    text = _to_text(observation)
    m = _RE_CORRECT.search(text)
    if m:
        return (f"phase{m.group(1)}", "correct")
    m = _RE_WRONG.search(text)
    if m:
        return (f"phase{m.group(1)}", "wrong")
    return (None, None)


@dataclass
class SplitResult:
    labels: list[str]
    warnings: list[str]


def split_phases(observations: list[Any]) -> SplitResult:
    """Classify every submit observation. Emit warnings for missing
    markers (per-observation) or inconsistent marker ordering."""
    labels: list[str] = []
    warnings: list[str] = []
    saw_phase1_correct = False
    saw_phase2_marker_before_phase1 = False
    unknown_count = 0
    saw_any_marker = False

    for obs in observations:
        phase, verdict = classify_submit_observation(obs)
        if phase is None:
            # Fallback: before any phase-1 correct marker, treat as phase-1;
            # after, phase-2. If we never see any marker at all we'll
            # emit a single warning at the end.
            labels.append("phase2" if saw_phase1_correct else "phase1")
            unknown_count += 1
            continue
        saw_any_marker = True
        labels.append(phase)
        if phase == "phase2" and not saw_phase1_correct:
            saw_phase2_marker_before_phase1 = True
        if phase == "phase1" and verdict == "correct":
            saw_phase1_correct = True

    if unknown_count and not saw_any_marker:
        warnings.append(
            f"no phase markers detected across {unknown_count} submit observation(s); "
            "defaulted every submit to phase-1"
        )
    elif unknown_count:
        warnings.append(
            f"phase markers missing on {unknown_count} submit observation(s); "
            "used last-seen-correct heuristic"
        )

    if saw_phase2_marker_before_phase1:
        warnings.append(
            "phase-2 marker observed before any phase-1 success marker; "
            "this is unusual — labels follow the markers as observed"
        )

    return SplitResult(labels=labels, warnings=warnings)
