"""DEV-1515: remap legacy ``failure_classification`` values on the 53
existing submission annotations after the schema split:

* Passes (N3=1) → `primary="no_fail"` (was `other` or some flag like
  `gold_audit_quality`). When demoting a non-`other` flag, move it into
  `secondary` so the signal isn't lost.
* Legacy `grader_stability` → cascade-tier-specific bucket, picking the
  lowest-tier tolerance that fires:
    * N4 (tie order)       → `row_order`
    * N6 (numeric epsilon) → `numerical_precision`
    * N7 (whitespace)      → `trailing_whitespace`
    * N8 (column order)    → `column_order`
  When NO cascade tier fires but the subagent specifically classified as
  `grader_stability`, default to `numerical_precision` (the most common
  source of grader-stability complaints in the dataset).
* Auto-classified `other` with a cascade-tier match → upgrade to the
  specific class.

This script operates on raw JSON before Pydantic validation because the
new schema rejects legacy `grader_stability` literals on read."""
from __future__ import annotations

import json

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation


_TIER_FIELDS_TO_CLASS = (
    ("correct_up_to_tie_order", "row_order"),
    ("correct_under_numeric_epsilon", "numerical_precision"),
    ("correct_under_trailing_whitespace", "trailing_whitespace"),
    ("correct_under_column_order", "column_order"),
)


def _cascade_tier_for(ev: dict) -> str | None:
    """Tier-tolerance class (or ``None``).

    Order matches the cascade: N4 (row_order) > N5 (novel_reading_accepted)
    > N6 (numerical_precision) > N7 > N8. N5 only counts when the LLM
    judge actually returned a pass — the field can be present with a
    null/None value when the judge wasn't invoked."""
    if ev.get("phase1_against_any_audited_variant") == "pass":
        return None  # N3-strict; no tolerance needed
    if ev.get("correct_up_to_tie_order"):
        return "row_order"
    if ev.get("novel_reading_judgment") == "pass":
        return "novel_reading_accepted"
    for field, cls in _TIER_FIELDS_TO_CLASS[1:]:
        if ev.get(field):
            return cls
    return None


def _remap_raw(raw: dict) -> tuple[bool, str]:
    """Mutate ``raw`` in-place; return ``(changed, reason)``."""
    ev = raw.get("evaluation") or {}
    fc = raw.setdefault("failure_classification", {})
    primary = fc.get("primary")
    secondary = list(fc.get("secondary") or [])
    reasons: list[str] = []

    # Step 1: N3 pass → no_fail.
    if ev.get("phase1_against_any_audited_variant") == "pass":
        if primary != "no_fail":
            displaced = primary
            fc["primary"] = "no_fail"
            fc["agent_at_fault"] = False
            if displaced and displaced != "other" and displaced not in secondary:
                secondary.append(displaced)
                reasons.append(f"{displaced}→no_fail; +secondary={displaced}")
            else:
                reasons.append(f"{displaced or '<missing>'}→no_fail")
            if not fc.get("remediation_target") or fc["remediation_target"] == "other":
                fc["remediation_target"] = "other"
        # Also rewrite legacy grader_stability in secondary into a
        # cascade-tier bucket while we're here.
        new_sec: list[str] = []
        for s in secondary:
            if s == "grader_stability":
                tier = _cascade_tier_for(ev) or "numerical_precision"
                new_sec.append(tier)
                reasons.append(f"sec grader_stability→{tier}")
            else:
                new_sec.append(s)
        fc["secondary"] = new_sec
        return (bool(reasons), "; ".join(reasons))

    # Step 2: not an N3 pass. Resolve legacy grader_stability.
    tier = _cascade_tier_for(ev)
    if primary == "grader_stability":
        new = tier or "numerical_precision"  # default for sub-epsilon float noise
        fc["primary"] = new
        fc["agent_at_fault"] = False
        fc["remediation_target"] = "grader"
        reasons.append(f"grader_stability→{new}")
    elif tier and primary == "other":
        # Auto-classified `other` upgrade.
        fc["primary"] = tier
        fc["agent_at_fault"] = False
        fc["remediation_target"] = "grader"
        reasons.append(f"other→{tier}")

    # Step 3: legacy grader_stability in secondary → split.
    new_sec = []
    for s in secondary:
        if s == "grader_stability":
            sec_tier = tier or "numerical_precision"
            new_sec.append(sec_tier)
            reasons.append(f"sec grader_stability→{sec_tier}")
        else:
            new_sec.append(s)
    fc["secondary"] = new_sec

    return (bool(reasons), "; ".join(reasons))


def main() -> None:
    annroot = paths.annotations_root() / "mini-interact"
    n_seen = n_changed = 0
    print(f"Walking {annroot}")
    for p in sorted(annroot.glob("*/*.submission.*.json")):
        n_seen += 1
        raw = json.loads(p.read_text())
        changed, reason = _remap_raw(raw)
        # Always re-validate to guarantee schema conformance.
        SubmissionAnnotation.model_validate(raw)
        if changed:
            n_changed += 1
            p.write_text(json.dumps(raw, indent=2) + "\n")
            print(f"  {raw['instance_id']:40s}  {reason}")
    print(f"\nVisited {n_seen} submission annotations; rewrote {n_changed}.")


if __name__ == "__main__":
    main()
