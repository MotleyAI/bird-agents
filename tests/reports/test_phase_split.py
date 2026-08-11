"""Tests for the per-submit phase classifier.

Spec (DEV-1553):
* The bird-interact-tools submit tool emits an observation with one of:
  ``Phase 1 SQL Correct! …``, ``Phase 2 SQL Correct! …``,
  ``Submitted SQL failed test case in Phase {1|2}. …`` (authoritative
  phrasing from ``action_handler_sqlite``).
* Per-submit classification reads the verdict directly when present.
* Fallback (Codex finding #5 tightening): before the FIRST observed
  phase-1 verdict, all submits are phase-1; after, all submits are
  phase-2. Inconsistent markers (e.g. phase-2 marker before any phase-1
  marker) → manifest warning, never error. Zero markers across the run
  + non-zero submits → manifest warning, treat all as phase-1.
"""

from __future__ import annotations


def _classify(observation):
    from bird_interact_agents.reports.phase_split import classify_submit_observation

    return classify_submit_observation(observation)


# ---------------------------------------------------------------------------
# Per-observation classifier
# ---------------------------------------------------------------------------


def test_classify_phase1_correct_moving_to_phase2():
    assert _classify(
        "Phase 1 SQL Correct! (Reward: 1 points). Moving to Phase 2."
    ) == ("phase1", "correct")


def test_classify_phase1_correct_no_phase2():
    assert _classify(
        "Phase 1 SQL Correct! (Reward: 1 points). No Phase 2. Task finished."
    ) == ("phase1", "correct")


def test_classify_phase2_correct():
    assert _classify(
        "Phase 2 SQL Correct! (Reward: 1 points). Task finished."
    ) == ("phase2", "correct")


def test_classify_phase1_wrong():
    assert _classify(
        "Submitted SQL failed test case in Phase 1. Reason: row mismatch. Please try again."
    ) == ("phase1", "wrong")


def test_classify_phase2_wrong():
    assert _classify(
        "Submitted SQL failed test case in Phase 2. Reason: column mismatch."
    ) == ("phase2", "wrong")


def test_classify_unknown_observation():
    """No marker → returns (None, None)."""
    assert _classify("some unrelated observation text") == (None, None)


def test_classify_handles_dict_content():
    """Tool result content can arrive as a list of {type:text, text:…} dicts
    (Anthropic SDK shape). The classifier normalises before matching."""
    obs = [{"type": "text", "text": "Phase 2 SQL Correct! Task finished."}]
    assert _classify(obs) == ("phase2", "correct")


# ---------------------------------------------------------------------------
# Whole-trajectory walk
# ---------------------------------------------------------------------------


def test_split_phase_classifies_two_phase_pass():
    """Phase-1 right + phase-2 right → ordered phase labels."""
    from bird_interact_agents.reports.phase_split import split_phases

    observations = [
        "Phase 1 SQL Correct! Moving to Phase 2.",
        "Phase 2 SQL Correct! Task finished.",
    ]
    result = split_phases(observations)
    assert result.labels == ["phase1", "phase2"]
    assert result.warnings == []


def test_split_phase_retry_then_phase2():
    """Phase-1 wrong, phase-1 right (retry), phase-2 right."""
    from bird_interact_agents.reports.phase_split import split_phases

    observations = [
        "Submitted SQL failed test case in Phase 1. Reason: x.",
        "Phase 1 SQL Correct! Moving to Phase 2.",
        "Phase 2 SQL Correct! Task finished.",
    ]
    result = split_phases(observations)
    assert result.labels == ["phase1", "phase1", "phase2"]
    assert result.warnings == []


def test_split_phase_no_markers_anywhere_warning():
    """Submits exist but no observation carries a phase marker → warning,
    label all phase-1 (Codex finding #5 fallback)."""
    from bird_interact_agents.reports.phase_split import split_phases

    observations = ["unrelated text", "another unrelated"]
    result = split_phases(observations)
    assert result.labels == ["phase1", "phase1"]
    assert any("no phase markers" in w.lower() for w in result.warnings)


def test_split_phase_inconsistent_marker_order_warning():
    """Phase-2 marker before any phase-1 marker → warning, not error."""
    from bird_interact_agents.reports.phase_split import split_phases

    observations = [
        "Phase 2 SQL Correct! Task finished.",
        "Phase 1 SQL Correct! Moving to Phase 2.",
    ]
    result = split_phases(observations)
    # We surface the inconsistency; labels still come from the marker text.
    assert result.labels == ["phase2", "phase1"]
    assert any(
        "phase-2 marker" in w.lower() and "before" in w.lower()
        for w in result.warnings
    )


def test_split_phase_empty_input_is_clean():
    from bird_interact_agents.reports.phase_split import split_phases

    result = split_phases([])
    assert result.labels == []
    assert result.warnings == []


def test_split_phase_phase1_wrong_then_phase2_correct_warns_no_phase1_success():
    """Submits go: phase-1 wrong, phase-2 correct (skipped phase-1 success).
    Real runs can't produce this, but a corrupted trajectory could.
    Per the spec the warning is reported; never error."""
    from bird_interact_agents.reports.phase_split import split_phases

    observations = [
        "Submitted SQL failed test case in Phase 1. Reason: x.",
        "Phase 2 SQL Correct! Task finished.",
    ]
    result = split_phases(observations)
    # Labels come from the markers directly.
    assert result.labels == ["phase1", "phase2"]
    # Some warning about the missing phase-1 success.
    assert result.warnings, "expected at least one warning"
