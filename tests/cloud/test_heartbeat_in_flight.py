"""HeartbeatWriter must surface the in-flight tasks (with elapsed time) in
status.json, so a slow-but-healthy task is distinguishable from a wedged actor
— the gap that let the no-progress deadline fire on a healthy long task.
"""
from __future__ import annotations

from bird_interact_agents.cloud import ray_app
from bird_interact_agents.cloud.ray_app import HeartbeatWriter


def _capture_status(monkeypatch):
    captured: dict = {}

    def _fake_write_status(run_id, status, *, client=None):
        captured.clear()
        captured.update(status)

    monkeypatch.setattr(ray_app._gcs, "write_status", _fake_write_status)
    return captured


def test_in_flight_emitted_with_elapsed(monkeypatch):
    captured = _capture_status(monkeypatch)
    hb = HeartbeatWriter(
        run_id="r", total=3, attempt=1, ray_job_id="raysubmit_1",
        client=object(),  # truthy → skips default_gcs_client()
    )
    hb.mark_start("alpha")
    hb._write(terminal_state=None)

    assert [x["instance_id"] for x in captured["in_flight"]] == ["alpha"]
    assert captured["in_flight"][0]["elapsed_s"] >= 0.0
    assert captured["rows_done"] == 0


def test_mark_done_clears_in_flight_and_tick_counts(monkeypatch):
    captured = _capture_status(monkeypatch)
    hb = HeartbeatWriter(
        run_id="r", total=3, attempt=1, ray_job_id="raysubmit_1",
        client=object(),
    )
    hb.mark_start("alpha")
    hb.tick_done()
    hb.mark_done("alpha")
    hb._write(terminal_state=None)

    assert captured["in_flight"] == []
    assert captured["rows_done"] == 1


def test_multiple_in_flight_sorted_oldest_first(monkeypatch):
    captured = _capture_status(monkeypatch)
    hb = HeartbeatWriter(
        run_id="r", total=3, attempt=1, ray_job_id="raysubmit_1",
        client=object(),
    )
    hb.mark_start("first")
    hb.mark_start("second")
    hb._write(terminal_state=None)

    iids = [x["instance_id"] for x in captured["in_flight"]]
    assert iids == ["first", "second"]  # oldest start first
