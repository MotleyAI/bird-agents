"""Tests for split-coverage check.

Spec (DEV-1553) + Codex finding #1:
* When ``--run-id`` is used WITHOUT ``--allow-partial``, the run's
  instance set MUST equal the full benchmark split, else SystemExit.
* When ``--selection`` is used WITHOUT ``--allow-partial``, the union of
  selected instance_ids MUST equal the full benchmark split, else
  SystemExit (Codex finding #1).
* ``--allow-partial`` gates BOTH paths.
"""

from __future__ import annotations

import pytest


def _stub_split(monkeypatch, instance_ids: set[str]):
    """Stub the benchmark split lookup so we don't need real data files."""
    from bird_interact_agents.reports import coverage as _cov

    monkeypatch.setattr(
        _cov, "load_benchmark_instance_ids", lambda benchmark: instance_ids
    )


# ---------------------------------------------------------------------------
# Coverage check: matching set
# ---------------------------------------------------------------------------


def test_coverage_full_set_passes(monkeypatch):
    from bird_interact_agents.reports.coverage import (
        assert_coverage_ok,
    )

    _stub_split(monkeypatch, {"alien_1", "alien_2"})
    # No exception — full set matches.
    assert_coverage_ok(
        benchmark="bird-interact-lite-exp",
        present_instance_ids={"alien_1", "alien_2"},
        allow_partial=False,
    )


# ---------------------------------------------------------------------------
# Coverage check: missing instances
# ---------------------------------------------------------------------------


def test_coverage_smaller_set_without_allow_partial_aborts(monkeypatch):
    from bird_interact_agents.reports.coverage import (
        IncompleteCoverageError,
        assert_coverage_ok,
    )

    _stub_split(monkeypatch, {"alien_1", "alien_2", "alien_3"})
    with pytest.raises(IncompleteCoverageError) as exc_info:
        assert_coverage_ok(
            benchmark="bird-interact-lite-exp",
            present_instance_ids={"alien_1"},
            allow_partial=False,
        )
    msg = str(exc_info.value)
    # The error lists every MISSING instance and points at --allow-partial.
    assert "alien_2" in msg
    assert "alien_3" in msg
    assert "alien_1" not in msg
    assert "allow-partial" in msg.lower()


def test_coverage_smaller_set_with_allow_partial_passes(monkeypatch):
    from bird_interact_agents.reports.coverage import assert_coverage_ok

    _stub_split(monkeypatch, {"alien_1", "alien_2", "alien_3"})
    # Does not raise.
    assert_coverage_ok(
        benchmark="bird-interact-lite-exp",
        present_instance_ids={"alien_1"},
        allow_partial=True,
    )


# ---------------------------------------------------------------------------
# Coverage check: extra instances (instance_id not in the benchmark)
# ---------------------------------------------------------------------------


def test_coverage_extra_instances_are_a_hard_error_always(monkeypatch):
    """If a selection names an instance the benchmark doesn't recognise,
    abort regardless of --allow-partial — that's a typo, not a partial
    run."""
    from bird_interact_agents.reports.coverage import (
        UnknownInstanceError,
        assert_coverage_ok,
    )

    _stub_split(monkeypatch, {"alien_1", "alien_2"})
    with pytest.raises(UnknownInstanceError) as exc_info:
        assert_coverage_ok(
            benchmark="bird-interact-lite-exp",
            present_instance_ids={"alien_1", "alien_2", "alien_typo"},
            allow_partial=True,
        )
    assert "alien_typo" in str(exc_info.value)
