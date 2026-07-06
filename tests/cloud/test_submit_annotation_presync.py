"""DEV-1638 (decision 6b): cloud submit self-healing annotation gate.

The cloud image bakes ``paths.annotations_root()`` at build time and does NO
runtime GCS fetch, so a local ``bird-interact`` run and a cloud submit must
share the SAME annotation set. To guarantee that, ``submit`` calls the SAME
``local_annotations.sync_annotations`` the local run uses — pulling any missing
task annotations from GCS into ``annotations_root()`` — BEFORE the existing
``require_annotation`` gate. The gate then only fires for ids genuinely absent
locally AND in GCS.

These pin the wiring (mechanical: call order, args, error propagation) without
building an image or spinning up a cluster.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import local_annotations
from bird_interact_agents.cloud import _annotation_check, cli


def _submit_argv(extra: list[str] | None = None) -> list[str]:
    return [
        "submit",
        "--framework", "claude_sdk",
        "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--mode", "one-shot",
        "--no-subscription-auth",
        "--no-use-audited-gold-sql",  # skip the audited-gold second guard
        *(extra or []),
    ]


def test_sync_runs_before_require_annotation_gate(monkeypatch):
    order: list[str] = []

    def _sync(dataset, ids):
        order.append("sync")
        assert dataset == "livesqlbench-base-lite-sqlite"
        assert ids == ["alien_1"]
        return {"fetched": 0, "already_local": 1, "missing_in_gcs": 0}

    def _missing(ids, benchmark):
        order.append("gate")
        return []

    monkeypatch.setattr(local_annotations, "sync_annotations", _sync)
    monkeypatch.setattr(_annotation_check, "missing_annotation_ids", _missing)

    cli.parse_args(_submit_argv())
    assert order == ["sync", "gate"]


def test_gate_still_fires_for_ids_missing_after_sync(monkeypatch):
    monkeypatch.setattr(
        local_annotations, "sync_annotations",
        lambda dataset, ids: {"fetched": 0, "already_local": 0, "missing_in_gcs": 1},
    )
    monkeypatch.setattr(
        _annotation_check, "missing_annotation_ids",
        lambda ids, benchmark: ["alien_1"],
    )
    with pytest.raises(SystemExit):
        cli.parse_args(_submit_argv())


def test_sync_infra_error_propagates_not_swallowed(monkeypatch):
    """Codex #3: a broken GCS (auth/network) must FAIL the submit, not be
    misattributed to 'missing annotations'. No blanket try/except."""
    def _boom(dataset, ids):
        raise RuntimeError("GCS auth failure")

    monkeypatch.setattr(local_annotations, "sync_annotations", _boom)
    # missing_annotation_ids should never be reached.
    monkeypatch.setattr(
        _annotation_check, "missing_annotation_ids",
        lambda ids, benchmark: (_ for _ in ()).throw(
            AssertionError("gate reached despite sync failure")),
    )
    with pytest.raises(RuntimeError, match="GCS auth failure"):
        cli.parse_args(_submit_argv())


def test_no_require_annotation_skips_sync_and_gate(monkeypatch):
    """--no-require-annotation is the opt-out for BOTH the gate and the sync:
    the sync lives inside the require_annotation block, so an explicit opt-out
    performs no GCS I/O and the parse still succeeds."""
    called = []
    monkeypatch.setattr(
        local_annotations, "sync_annotations",
        lambda dataset, ids: called.append(1),
    )
    monkeypatch.setattr(
        _annotation_check, "missing_annotation_ids",
        lambda ids, benchmark: (_ for _ in ()).throw(
            AssertionError("gate reached despite --no-require-annotation")),
    )
    cli.parse_args(_submit_argv(["--no-require-annotation"]))
    assert called == []
