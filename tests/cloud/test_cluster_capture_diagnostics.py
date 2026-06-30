"""cluster.capture_diagnostics dumps head-node state (`ray status` + driver
log) for a stalled run, and must never raise so the watcher can still
fetch + kill.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bird_interact_agents.cloud import cluster


def _result(stdout="", stderr="", rc=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=rc)


def test_runs_ray_status_and_job_logs(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return _result(stdout=f"OUT::{argv[-1]}")

    monkeypatch.setattr(cluster.subprocess, "run", fake_run)
    text = cluster.capture_diagnostics(
        Path("/tmp/run.yaml"), ray_job_id="raysubmit_abc",
    )

    assert len(calls) == 2
    assert calls[0][:3] == ["ray", "exec", "/tmp/run.yaml"]
    assert calls[0][3] == "ray status"
    assert "ray job logs" in calls[1][3] and "raysubmit_abc" in calls[1][3]
    assert "ray status" in text and "raysubmit_abc" in text


def test_no_job_id_skips_logs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cluster.subprocess, "run",
        lambda argv, **kw: (calls.append(argv), _result(stdout="ok"))[1],
    )
    text = cluster.capture_diagnostics(Path("/tmp/run.yaml"))
    assert len(calls) == 1  # only ray status
    assert "SKIPPED" in text


def test_never_raises_on_exec_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ssh: head unreachable")

    monkeypatch.setattr(cluster.subprocess, "run", boom)
    text = cluster.capture_diagnostics(
        Path("/tmp/run.yaml"), ray_job_id="raysubmit_1",
    )
    assert "FAILED" in text  # annotated, not raised


def test_nonzero_exit_and_stderr_captured(monkeypatch):
    monkeypatch.setattr(
        cluster.subprocess, "run",
        lambda argv, **kw: _result(stdout="", stderr="boom", rc=1),
    )
    text = cluster.capture_diagnostics(Path("/tmp/run.yaml"))
    assert "exit 1" in text and "boom" in text
