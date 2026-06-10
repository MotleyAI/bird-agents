"""T13–T19: driver orchestration, SIGINT, stall detection, kill recovery."""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bird_interact_agents.cloud import driver  # noqa: E402


RUN_ID = "20260521T1422-pydanticai-raw-a1b2c3"


# ---------------------------------------------------------------------------
# Common fixture: a SubmitArgs that mirrors `cli.py` after argparse.
# ---------------------------------------------------------------------------


@dataclass
class FakeSubmitArgs:
    framework: str = "pydantic_ai"
    query_mode: str = "raw"
    agent_model: str = "anthropic/claude-sonnet-4-5"
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001"
    mode: str = "c-interact"
    instance_ids: tuple[str, ...] = ("alien_1", "alien_2", "alien_3")
    patience: int = 3
    strict: bool = False
    use_audited_gold_sql: bool = False
    max_depth: int = 3
    prompt_cache: bool = True
    workers: int = 2
    actors_per_worker: int = 2
    worker_type: str = "e2-standard-4"
    max_runtime_hours: int = 8
    run_id: str | None = None
    detach: bool = False
    allow_dirty: bool = False
    slayer_setup: str = "pre-encoded"
    slayer_storage_root: str = "/data/slayer_models"
    dataset: str = "mini-interact"
    # DEV-1535: flip the fake's default to the legacy API-key path so the
    # bulk of driver tests don't have to monkeypatch CLAUDE_CODE_OAUTH_TOKEN.
    # The dedicated OAuth-path tests (test_read_api_keys_oauth_*) set the
    # env explicitly and pass `no_subscription_auth=False` directly.
    no_subscription_auth: bool = True


def _patch_collaborators(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace driver's collaborators with mocks and return them."""
    mocks: dict[str, MagicMock] = {}
    for attr in (
        "prereqs",
        "image",
        "cluster",
        "gcs",
        "benchmark_data",
    ):
        m = MagicMock(name=attr)
        mocks[attr] = m
        monkeypatch.setattr(f"bird_interact_agents.cloud.driver.{attr}", m)
    # DEV-1535: the OAuth-required guard in `read_api_keys_from_local_env`
    # branches on `prereqs._is_claude_sdk_framework(framework)`. With the
    # generic MagicMock above that predicate returns a truthy Mock for
    # ANY framework, so non-claude_sdk fixture manifests (pydantic_ai,
    # annotator, ...) would incorrectly enter the OAuth branch. Wire the
    # real predicate through so the framework check actually evaluates.
    from bird_interact_agents.cloud import prereqs as _real_prereqs
    mocks["prereqs"]._is_claude_sdk_framework.side_effect = (
        _real_prereqs._is_claude_sdk_framework
    )
    mocks["image"].image_tag.return_value = "deadbeef1234-cafebabe5678"
    mocks["image"].build_and_push.return_value = (
        "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
    )
    # De-bake: submit uploads the dataset to a content-hashed GCS prefix and
    # threads it through the manifest/job-args. Mocked so submit tests don't
    # hash the real dataset dir or hit GCS.
    mocks["benchmark_data"].ensure_uploaded.return_value = (
        "benchmark-data/mini_interact/deadbeefcafe/"
    )
    mocks["cluster"].head_address.return_value = "ray://10.0.0.1:10001"
    mocks["cluster"].submit_job.return_value = "raysubmit_abc123"
    mocks["cluster"].render_from_manifest.return_value = Path("/tmp/cluster.yaml")
    return mocks


# ---------------------------------------------------------------------------
# T13 — submit() call order: build → push → up → submit_job → poll-or-detach.
# ---------------------------------------------------------------------------


def test_submit_attached_call_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a parent MagicMock with attach_mock so each collaborator call
    lands on one ordered list — enforces the prereqs→image→cluster→submit
    →wait→fetch→down sequencing."""
    mocks = _patch_collaborators(monkeypatch)
    parent = MagicMock()
    parent.attach_mock(mocks["prereqs"].check, "prereqs_check")
    parent.attach_mock(mocks["image"].build_and_push, "build_and_push")
    parent.attach_mock(mocks["gcs"].write_manifest, "write_manifest")
    parent.attach_mock(mocks["cluster"].render_from_manifest, "render")
    parent.attach_mock(mocks["cluster"].up, "up")
    parent.attach_mock(mocks["cluster"].submit_job, "submit_job")
    parent.attach_mock(mocks["cluster"].down, "down")

    wait_mock = MagicMock()
    fetch_mock = MagicMock()
    parent.attach_mock(wait_mock, "wait")
    parent.attach_mock(fetch_mock, "fetch")
    monkeypatch.setattr(driver, "wait_until_done", wait_mock)
    monkeypatch.setattr(driver, "fetch", fetch_mock)

    args = FakeSubmitArgs(detach=False)
    driver.submit(args)

    ordered_names = [c[0] for c in parent.mock_calls]
    # Required relative order (other collaborator calls may interleave).
    required = [
        "prereqs_check",
        "build_and_push",
        "write_manifest",
        "render",
        "up",
        "submit_job",
        "wait",
        "fetch",
        "down",
    ]
    positions = [ordered_names.index(n) for n in required if n in ordered_names]
    assert positions == sorted(positions), (
        f"call order violated; saw {ordered_names}"
    )
    # And every required step happened at all.
    for n in required:
        assert n in ordered_names, f"missing {n}"


# ---------------------------------------------------------------------------
# T14 — --detach returns before the Ray job completes.
# ---------------------------------------------------------------------------


def test_detach_returns_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _patch_collaborators(monkeypatch)

    wait_mock = MagicMock()
    fetch_mock = MagicMock()
    monkeypatch.setattr(driver, "wait_until_done", wait_mock)
    monkeypatch.setattr(driver, "fetch", fetch_mock)

    args = FakeSubmitArgs(detach=True)
    t0 = time.perf_counter()
    driver.submit(args)
    elapsed = time.perf_counter() - t0

    # Detached: submit_job was called (with --no-wait under the hood), but the
    # driver does NOT poll or fetch, and does NOT tear the cluster down.
    mocks["cluster"].submit_job.assert_called_once()
    wait_mock.assert_not_called()
    fetch_mock.assert_not_called()
    mocks["cluster"].down.assert_not_called()
    assert elapsed < 5  # detached return is fast


# ---------------------------------------------------------------------------
# T15 — stall detection: stale heartbeat while head is alive → non-zero hint.
# ---------------------------------------------------------------------------


def test_wait_until_done_detects_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _patch_collaborators(monkeypatch)
    now = 1_000_000.0
    stale = now - 600  # 10 min ago — over the 5-min threshold
    mocks["gcs"].read_status.return_value = {
        "ray_job_id": "raysubmit_abc",
        "last_heartbeat_ts": stale,
        "rows_done": 1,
        "rows_total": 5,
        "terminal_state": None,
    }
    mocks["gcs"].list_attempts.return_value = {"db_a_1": [1]}
    mocks["cluster"].head_is_alive.return_value = True

    monkeypatch.setattr(driver.time, "time", lambda: now)

    manifest = {
        "run_id": RUN_ID,
        "instance_ids": ["db_a_1", "db_a_2", "db_a_3", "db_a_4", "db_a_5"],
    }
    result = driver.wait_until_done(RUN_ID, manifest, poll_interval_s=0)
    assert result.terminal_state == "stalled"
    assert "resubmit" in result.hint.lower()


def test_wait_until_done_partial_retry_completes_via_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: on a partial retry, only the `missing` IIDs are
    dispatched, so previously-succeeded IIDs never receive a `next_attempt`
    row. Resubmit must pass a manifest scoped to `missing` so the row-count
    completion check (`done_count >= total`) can fire — otherwise the run
    would hang on the terminal-state write only (no row-count fallback)."""
    mocks = _patch_collaborators(monkeypatch)
    now = [1_000_000.0]
    mocks["gcs"].read_status.return_value = {
        "ray_job_id": "raysubmit_attempt2",
        "last_heartbeat_ts": None,
        "rows_done": 0,
        "rows_total": 2,
        "terminal_state": None,
    }
    # 2 missing iids; attempt-2 rows land progressively.
    rows_seq = [
        {"db_a_2": [1], "db_a_3": [1]},
        {"db_a_2": [1, 2], "db_a_3": [1]},
        {"db_a_2": [1, 2], "db_a_3": [1, 2]},
    ]
    idx = {"n": 0}

    def fake_list_attempts(_rid):
        v = rows_seq[min(idx["n"], len(rows_seq) - 1)]
        idx["n"] += 1
        return v

    mocks["gcs"].list_attempts.side_effect = fake_list_attempts
    mocks["cluster"].head_is_alive.return_value = True
    monkeypatch.setattr(driver.time, "time", lambda: now[0])
    monkeypatch.setattr(driver.time, "sleep", lambda _s: None)

    # Manifest carries ONLY the missing iids (resubmit's derived manifest).
    retry_manifest = {
        "run_id": RUN_ID,
        "instance_ids": ["db_a_2", "db_a_3"],
    }
    result = driver.wait_until_done(
        RUN_ID, retry_manifest, poll_interval_s=0.001, min_attempt=2,
    )
    assert result.terminal_state == "done"
    # Both missing iids got an attempt-2 row after exactly 3 polls.
    assert idx["n"] == 3, f"expected 3 polls, got {idx['n']}"


def test_wait_until_done_min_attempt_ignores_prior_attempt_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: on a resubmit, every iid in the manifest already has a
    row from the prior (failed) attempt, so the default `len(attempts) >=
    total` check would return `done` immediately without waiting for the
    new attempt. wait_until_done(..., min_attempt=N) must only count iids
    that have at least one row with attempt >= N."""
    mocks = _patch_collaborators(monkeypatch)
    now = [1_000_000.0]
    mocks["gcs"].read_status.return_value = {
        "ray_job_id": "raysubmit_attempt2",
        "last_heartbeat_ts": None,
        "rows_done": 0,
        "rows_total": 3,
        "terminal_state": None,
    }
    # All 3 iids have a prior-attempt row; the new attempt 2 lands rows for
    # b and c (one per poll) so the run reaches done — but never via the
    # spurious "all iids already have rows" path.
    rows_seq = [
        {"db_a_1": [1], "db_a_2": [1], "db_a_3": [1]},
        {"db_a_1": [1], "db_a_2": [1, 2], "db_a_3": [1]},
        {"db_a_1": [1], "db_a_2": [1, 2], "db_a_3": [1, 2]},
        {"db_a_1": [1, 2], "db_a_2": [1, 2], "db_a_3": [1, 2]},
    ]
    idx = {"n": 0}

    def fake_list_attempts(_rid):
        v = rows_seq[min(idx["n"], len(rows_seq) - 1)]
        idx["n"] += 1
        return v

    mocks["gcs"].list_attempts.side_effect = fake_list_attempts
    mocks["cluster"].head_is_alive.return_value = True
    monkeypatch.setattr(driver.time, "time", lambda: now[0])
    monkeypatch.setattr(driver.time, "sleep", lambda _s: None)

    manifest = {
        "run_id": RUN_ID,
        "instance_ids": ["db_a_1", "db_a_2", "db_a_3"],
    }
    result = driver.wait_until_done(
        RUN_ID, manifest, poll_interval_s=0.001, min_attempt=2,
    )
    assert result.terminal_state == "done"
    assert idx["n"] >= 4, (
        f"wait_until_done must wait until ALL 3 iids have an attempt-2 row "
        f"(saw {idx['n']} polls; would have terminated at 1 if min_attempt "
        f"was ignored)"
    )


def test_wait_until_done_skips_stall_when_heartbeat_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: before the in-job HeartbeatWriter has written its first
    row, status.last_heartbeat_ts is None (intentional — see driver.submit/
    annotate). The heartbeat-stall check must NOT fire in that window, even
    after lots of wall time elapses. The no-progress deadline is still the
    backstop. Originally bitten by livesqlbench-large (1.2 GB pg_dumps load
    on workers) where the driver pre-stamped time.time() and stalled the
    watcher at the 5-min threshold before any real heartbeat arrived."""
    mocks = _patch_collaborators(monkeypatch)
    now = [1_000_000.0]
    # last_heartbeat_ts intentionally None (matches driver.submit/annotate).
    mocks["gcs"].read_status.return_value = {
        "ray_job_id": "raysubmit_abc",
        "last_heartbeat_ts": None,
        "rows_done": 0,
        "rows_total": 5,
        "terminal_state": None,
    }
    # One row lands between polls so the no-progress deadline keeps resetting
    # (we want to prove the stall check is skipped, not the no-progress one).
    rows = [
        {},
        {"db_a_1": [1]},
        {"db_a_1": [1], "db_a_2": [1]},
        {"db_a_1": [1], "db_a_2": [1], "db_a_3": [1]},
        {"db_a_1": [1], "db_a_2": [1], "db_a_3": [1], "db_a_4": [1]},
        {"db_a_1": [1], "db_a_2": [1], "db_a_3": [1], "db_a_4": [1], "db_a_5": [1]},
    ]
    idx = {"n": 0}

    def fake_list_attempts(_rid):
        v = rows[min(idx["n"], len(rows) - 1)]
        idx["n"] += 1
        return v

    mocks["gcs"].list_attempts.side_effect = fake_list_attempts
    mocks["cluster"].head_is_alive.return_value = True
    # Advance clock far past HEARTBEAT_STALL_SECONDS (300s) on each poll —
    # the only thing keeping this run alive is `last is None` skipping the
    # stall check.
    monkeypatch.setattr(driver.time, "time", lambda: now[0])

    def fake_sleep(_s):
        now[0] += 600.0  # 10 min per poll, > HEARTBEAT_STALL_SECONDS

    monkeypatch.setattr(driver.time, "sleep", fake_sleep)

    manifest = {
        "run_id": RUN_ID,
        "instance_ids": ["db_a_1", "db_a_2", "db_a_3", "db_a_4", "db_a_5"],
    }
    result = driver.wait_until_done(RUN_ID, manifest, poll_interval_s=0.001)
    assert result.terminal_state == "done"
    assert result.hint == ""


def test_wait_until_done_no_progress_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh heartbeat with terminal_state=None forever AND zero rows (e.g.
    workers never autoscale, actor sits PENDING) must NOT poll until the VM
    self-delete timer — the no-progress deadline bounds it."""
    mocks = _patch_collaborators(monkeypatch)
    now = [1_000_000.0]
    # Heartbeat is always "fresh" (5s old) and never terminal, so the
    # stall + headless + done checks never fire.
    mocks["gcs"].read_status.return_value = {
        "ray_job_id": "raysubmit_abc",
        "last_heartbeat_ts": now[0] - 5,
        "rows_done": 0,
        "rows_total": 2,
        "terminal_state": None,
    }
    mocks["gcs"].list_attempts.return_value = {}  # NO rows ever → no progress
    mocks["cluster"].head_is_alive.return_value = True

    # Advance the clock past the deadline on each poll; keep heartbeat fresh.
    def fake_time():
        now[0] += 100
        mocks["gcs"].read_status.return_value["last_heartbeat_ts"] = now[0] - 5
        return now[0]

    monkeypatch.setattr(driver.time, "time", fake_time)

    manifest = {"run_id": RUN_ID, "instance_ids": ["db_a_1", "db_a_2"]}
    result = driver.wait_until_done(
        RUN_ID, manifest, poll_interval_s=0.001, no_progress_deadline_s=300.0
    )
    assert result.terminal_state == "timed-out"
    assert "resubmit" in result.hint.lower()


def test_wait_until_done_progress_resets_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow-but-progressing run must NOT be falsely timed out: each new row
    resets the no-progress deadline. Here rows keep landing and the wall
    clock blows past the deadline-per-poll, yet the run reaches `done`
    instead of `timed-out` (Codex finding — the old wall-clock deadline
    would have fired even though `max_runtime_hours` defaults to 8h)."""
    mocks = _patch_collaborators(monkeypatch)
    now = [1_000_000.0]
    mocks["gcs"].read_status.return_value = {
        "ray_job_id": "raysubmit_abc",
        "last_heartbeat_ts": now[0],
        "rows_done": 0,
        "rows_total": 3,
        "terminal_state": None,
    }
    mocks["cluster"].head_is_alive.return_value = True

    # Clock advances only at sleep (between polls), so time() is stable within
    # an iteration (as in reality). Each poll jumps the clock by FAR more than
    # the no-progress deadline — a wall-clock-from-start deadline would trip;
    # the no-progress one keeps resetting because rows keep landing. Heartbeat
    # kept fresh so the stall path doesn't interfere.
    monkeypatch.setattr(driver.time, "time", lambda: now[0])

    def fake_sleep(_s):
        now[0] += 10_000
        mocks["gcs"].read_status.return_value["last_heartbeat_ts"] = now[0]

    monkeypatch.setattr(driver.time, "sleep", fake_sleep)

    # One more iid completes on each successive poll → continuous progress.
    seq = [{"a": [1]}, {"a": [1], "b": [1]}, {"a": [1], "b": [1], "c": [1]}]
    idx = {"n": 0}

    def fake_list_attempts(_rid):
        v = seq[min(idx["n"], len(seq) - 1)]
        idx["n"] += 1
        return v

    mocks["gcs"].list_attempts.side_effect = fake_list_attempts

    manifest = {"run_id": RUN_ID, "instance_ids": ["a", "b", "c"]}
    result = driver.wait_until_done(
        RUN_ID, manifest, poll_interval_s=1.0, no_progress_deadline_s=400.0
    )
    assert result.terminal_state == "done"


# ---------------------------------------------------------------------------
# T16 — teardown idempotency across exit / exception / SIGINT pathways.
# ---------------------------------------------------------------------------


def test_teardown_idempotent_under_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_collaborators(monkeypatch)
    h = driver.install_signal_handlers(
        run_id=RUN_ID, yaml_path=Path("/tmp/cluster.yaml")
    )
    h.teardown(reason="exit")
    h.teardown(reason="exit")
    assert mocks["cluster"].down.call_count == 1


def test_teardown_fires_on_exception_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If submit raises *after* cluster.up, the install_signal_handlers
    `finally` path must still tear down exactly once."""
    mocks = _patch_collaborators(monkeypatch)
    mocks["cluster"].submit_job.side_effect = RuntimeError("ray job submit failed")

    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    args = FakeSubmitArgs(detach=False)
    with pytest.raises(RuntimeError):
        driver.submit(args)
    assert mocks["cluster"].down.call_count == 1


def test_sigint_path_actually_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler should be installed on signal.SIGINT and actually run
    when the signal fires (we use signal.raise_signal as §10 directs)."""
    mocks = _patch_collaborators(monkeypatch)
    fired: list[str] = []

    real_install = driver.install_signal_handlers

    def spy_install(*args, **kwargs):
        h = real_install(*args, **kwargs)
        # Wrap teardown so we can observe it firing via the signal.
        orig = h.teardown

        def wrapped(*a, **kw):
            fired.append("teardown")
            return orig(*a, **kw)

        h.teardown = wrapped  # type: ignore[method-assign]
        return h

    monkeypatch.setattr(driver, "install_signal_handlers", spy_install)
    driver.install_signal_handlers(
        run_id=RUN_ID, yaml_path=Path("/tmp/cluster.yaml")
    )
    # The installer registered an OS-level handler. Fire the signal and
    # confirm the wrapped teardown ran.
    try:
        signal.raise_signal(signal.SIGINT)
    except SystemExit:
        # First SIGINT exits 130 after teardown — that's the documented path.
        pass
    assert "teardown" in fired
    assert mocks["cluster"].down.call_count >= 1


# ---------------------------------------------------------------------------
# T17 — second SIGINT exits 130 without teardown.
# ---------------------------------------------------------------------------


def test_second_sigint_skips_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _patch_collaborators(monkeypatch)
    h = driver.install_signal_handlers(
        run_id=RUN_ID, yaml_path=Path("/tmp/cluster.yaml")
    )
    # First SIGINT → graceful teardown.
    with pytest.raises(SystemExit) as exc1:
        h._on_sigint(signal.SIGINT, None)  # type: ignore[arg-type]
    assert exc1.value.code == 130
    mocks["cluster"].down.assert_called_once()

    # Second SIGINT → exit without teardown.
    with pytest.raises(SystemExit) as exc2:
        h._on_sigint(signal.SIGINT, None)  # type: ignore[arg-type]
    assert exc2.value.code == 130
    assert mocks["cluster"].down.call_count == 1


# ---------------------------------------------------------------------------
# T18 — kill works with no cached YAML by re-rendering from manifest in GCS.
# ---------------------------------------------------------------------------


def test_kill_recovers_from_missing_yaml_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mocks = _patch_collaborators(monkeypatch)
    # Point the cache dir at an empty tmp_path so the YAML is "missing."
    monkeypatch.setattr(driver, "yaml_cache_dir", lambda: tmp_path)
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID,
        "render_inputs": {
            "workers": 1,
            "actors_per_worker": 1,
            "worker_type": "e2-standard-4",
            "zone": "us-central1-a",
            "worker_sa": "x@y.iam.gserviceaccount.com",
            "max_runtime_hours": 1,
            "image_uri": "us-central1-docker.pkg.dev/x/y/runner:tag",
        },
    }

    driver.kill(RUN_ID)
    mocks["gcs"].read_manifest.assert_called_once_with(RUN_ID)
    mocks["cluster"].render_from_manifest.assert_called_once()
    mocks["cluster"].down.assert_called_once()


# ---------------------------------------------------------------------------
# T19 — driver.kill calls cluster.fallback_delete_by_label when down fails.
#       (The argv-level assertion for `--delete-disks=all` lives in
#       test_cluster.py::test_fallback_delete_argv.)
# ---------------------------------------------------------------------------


def test_kill_invokes_fallback_when_ray_down_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mocks = _patch_collaborators(monkeypatch)
    # Seed a cached YAML so kill goes through `cluster.down` first.
    (tmp_path / f"{RUN_ID}.yaml").write_text("# fake\n")
    monkeypatch.setattr(driver, "yaml_cache_dir", lambda: tmp_path)
    mocks["cluster"].down.side_effect = RuntimeError("ray down failed")

    driver.kill(RUN_ID)
    mocks["cluster"].fallback_delete_by_label.assert_called_once()
    call_args = mocks["cluster"].fallback_delete_by_label.call_args
    # run_id must be passed (positional or kw).
    passed = list(call_args.args) + list(call_args.kwargs.values())
    assert RUN_ID in passed


# ---------------------------------------------------------------------------
# Manifest propagation: SubmitArgs → manifest.json (for T35 — strengthen
# the CLI parser test by proving values land in the manifest).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CR#4 — detach + submit failure must STILL tear down the cluster.
# ---------------------------------------------------------------------------


def test_detach_failure_tears_down_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_collaborators(monkeypatch)
    # `cluster.up` succeeds; `submit_job` raises mid-flight.
    mocks["cluster"].submit_job.side_effect = RuntimeError("submit failed")

    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    args = FakeSubmitArgs(detach=True)
    with pytest.raises(RuntimeError):
        driver.submit(args)
    # Without the CR#4 fix, this would be 0 — detached failures would
    # orphan the cluster.
    assert mocks["cluster"].down.call_count == 1


def test_detach_success_does_NOT_tear_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The successful-detach path: submit returns, leaves the cluster up
    for VM self-delete / explicit `kill` later."""
    mocks = _patch_collaborators(monkeypatch)
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    args = FakeSubmitArgs(detach=True)
    driver.submit(args)
    mocks["cluster"].down.assert_not_called()


def test_mint_run_id_is_gce_label_safe() -> None:
    """GCE label values + instance names reject uppercase letters and
    most punctuation. The run-id is used as both, so it must match
    `[a-z0-9-]+` and start with a lowercase letter or digit. Regression
    for the smoke-time `Invalid value for field 'resource.labels': '...'.
    Label value '<run-id>' violates format constraints.` failure."""
    import re
    for fw in ("pydantic_ai", "pydantic_ai_recursive", "claude_sdk"):
        for qm in ("raw", "slayer"):
            rid = driver.mint_run_id(fw, qm)
            assert re.fullmatch(r"[a-z0-9][-a-z0-9]*[a-z0-9]", rid), (
                f"run-id {rid!r} contains chars GCE rejects in label/instance names"
            )
            assert "T" not in rid, "timestamp separator must be lowercase 't'"
            assert "_" not in rid, "no underscores (instance-name regex forbids them)"


def test_mint_run_id_fits_gce_instance_name() -> None:
    """Ray composes the GCE name as ``ray-<run_id>-worker-<uuid8>-compute``
    (worst case: worker > head, compute > tpu). GCP's compute-instance regex
    rejects names > 63 chars (Ray's internal 55 assertion is loose vs that).
    The long ``pydantic_ai_otf_encode`` / ``pydantic_ai_recursive`` slugs
    tripped GCP once DEV-1468 made them cloud-submittable in slayer mode —
    the slug is now dynamically capped from GCP's 63-char limit."""
    for fw in (
        "pydantic_ai_otf_encode", "pydantic_ai_recursive", "smolagents",
        "claude_sdk", "mcp_agent", "agno", "pydantic_ai",
    ):
        for qm in ("raw", "slayer"):  # "slayer" is the longer / tighter qm
            rid = driver.mint_run_id(fw, qm)
            worst_full = f"ray-{rid}-worker-12345678-compute"
            assert len(worst_full) <= 63, (fw, qm, rid, len(worst_full))


def test_build_manifest_propagates_all_knobs() -> None:
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive",
        query_mode="slayer",
        agent_model="cerebras/zai-glm-4.7",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        mode="a-interact",
        instance_ids=("db_a_1", "db_a_2"),
        patience=4,
        strict=True,
        use_audited_gold_sql=True,
        max_depth=7,
        prompt_cache=False,
        workers=6,
        actors_per_worker=3,
        worker_type="e2-standard-8",
        max_runtime_hours=2,
        run_id=None,
        detach=True,
        allow_dirty=False,
    )
    image_uri = "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
    manifest = driver.build_manifest(args, image_uri=image_uri, run_id=RUN_ID)

    assert manifest["run_id"] == RUN_ID
    assert manifest["framework"] == "pydantic_ai_recursive"
    assert manifest["query_mode"] == "slayer"
    assert manifest["mode"] == "a-interact"
    assert manifest["agent_model"] == "cerebras/zai-glm-4.7"
    assert manifest["user_sim_model"] == "anthropic/claude-haiku-4-5-20251001"
    assert manifest["instance_ids"] == ["db_a_1", "db_a_2"]
    assert manifest["patience"] == 4
    assert manifest["strict"] is True
    assert manifest["use_audited_gold_sql"] is True
    assert manifest["max_depth"] == 7
    assert manifest["prompt_cache"] is False

    ri = manifest["render_inputs"]
    assert ri["workers"] == 6
    assert ri["actors_per_worker"] == 3
    assert ri["worker_type"] == "e2-standard-8"
    assert ri["max_runtime_hours"] == 2
    assert ri["image_uri"] == image_uri
    # zone / worker_sa / project / region come from a global config; check
    # they're present so `render_from_manifest` can round-trip them (region
    # drives provider.region + the AR docker-credential host).
    assert ri.get("zone")
    assert ri.get("worker_sa")
    assert ri.get("project")
    assert ri.get("region")


def test_resubmit_resets_heartbeat_before_submit_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: resubmit must clear last_heartbeat_ts *before* calling
    cluster.submit_job (so the in-job HeartbeatWriter — which races against
    this reset — cannot have its first real heartbeat clobbered). The
    previous attempt's timestamp (especially after a heartbeat-stall) is
    still in GCS; without this reset the new attempt's watcher reads that
    stale value and falsely returns `stalled` again. ray_job_id is None in
    this pre-submit write (the in-job HeartbeatWriter writes the real id)."""
    mocks = _patch_collaborators(monkeypatch)
    mocks["cluster"].render_from_manifest.return_value = Path("/tmp/x.yaml")
    mocks["cluster"].head_address.return_value = "http://localhost:8265"
    mocks["cluster"].submit_job.return_value = "raysubmit_attempt2"

    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "query_mode": "raw",
        "mode": "c-interact",
        "agent_model": "anthropic/claude-haiku-4-5-20251001",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1", "db_a_2", "db_a_3"],
        "patience": 3,
        "strict": False,
        "use_audited_gold_sql": False,
        "max_depth": 3,
        "prompt_cache": True,
        "render_inputs": {"workers": 1, "actors_per_worker": 2},
    }
    mocks["gcs"].list_attempts.return_value = {
        "db_a_1": [1], "db_a_2": [1], "db_a_3": [1],
    }
    # One done, two missing → resubmit proceeds with 2 missing.
    mocks["gcs"].read_row.side_effect = lambda rid, iid, n: (
        {"error": None} if iid == "db_a_1" else {"error": "boom"}
    )
    # Record the call order so we can assert write_status happens before
    # submit_job (closes the race against the in-job HeartbeatWriter).
    call_order: list[str] = []
    writes: list[tuple[str, dict]] = []
    mocks["gcs"].write_status.side_effect = lambda rid, payload, **_: (
        writes.append((rid, payload)) or call_order.append("write_status")
    )
    mocks["cluster"].submit_job.side_effect = lambda **_: (
        call_order.append("submit_job") or "raysubmit_attempt2"
    )
    wait_calls: list[tuple[dict, dict]] = []

    def fake_wait(_rid, manifest, **kwargs):
        wait_calls.append((manifest, kwargs))
        return None

    monkeypatch.setattr(driver, "wait_until_done", fake_wait)
    monkeypatch.setattr(driver, "fetch", lambda *a, **k: {})

    driver.resubmit(RUN_ID)

    reset = [p for _, p in writes if p.get("attempt") == 2]
    assert reset, f"expected an attempt=2 status write, got {writes}"
    payload = reset[-1]
    assert payload["last_heartbeat_ts"] is None
    assert payload["rows_done"] == 0
    assert payload["rows_total"] == 3
    assert payload["terminal_state"] is None
    assert payload["ray_job_id"] is None  # in-job writer fills the real id
    # The write_status reset must precede cluster.submit_job — otherwise the
    # in-job HeartbeatWriter (which starts asynchronously inside the Ray job)
    # can land its first real heartbeat before our reset overwrites it.
    assert call_order.index("write_status") < call_order.index("submit_job"), (
        f"write_status must precede submit_job; got {call_order}"
    )
    # Resubmit must pass min_attempt=next_attempt so wait_until_done doesn't
    # count prior-attempt rows toward this retry's completion.
    assert wait_calls, "expected wait_until_done to be called"
    wait_manifest, wait_kwargs = wait_calls[-1]
    assert wait_kwargs.get("min_attempt") == 2, (
        f"expected min_attempt=2 in wait_until_done kwargs, got {wait_kwargs}"
    )
    # Resubmit must scope the wait manifest to `missing` IIDs — previously-
    # succeeded IIDs won't get a next_attempt row, so leaving the full
    # manifest would prevent the row-count completion fallback from firing.
    assert wait_manifest["instance_ids"] == ["db_a_2", "db_a_3"], (
        f"expected wait_until_done's manifest to carry only the missing "
        f"iids, got {wait_manifest['instance_ids']}"
    )


def test_resubmit_passes_yaml_path_to_submit_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resubmit must submit the job via the `ray exec`/dashboard path (i.e.
    pass `yaml_path`), not the legacy unreachable `ray://:10001` path (A1)."""
    mocks = _patch_collaborators(monkeypatch)
    yaml_path = Path("/tmp/resubmit-cluster.yaml")
    mocks["cluster"].render_from_manifest.return_value = yaml_path
    mocks["cluster"].head_address.return_value = "http://localhost:8265"

    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai",
        "query_mode": "raw",
        "mode": "c-interact",
        "agent_model": "anthropic/claude-haiku-4-5-20251001",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1", "db_a_2"],
        "patience": 3,
        "strict": False,
        "use_audited_gold_sql": False,
        "max_depth": 3,
        "prompt_cache": True,
        "render_inputs": {"workers": 1, "actors_per_worker": 2},
    }
    # One task already done (no error), one still missing → resubmit proceeds.
    mocks["gcs"].list_attempts.return_value = {"db_a_1": [1], "db_a_2": [1]}
    mocks["gcs"].read_row.side_effect = lambda rid, iid, n: (
        {"error": None} if iid == "db_a_1" else {"error": "boom"}
    )
    # Don't actually poll / fetch.
    monkeypatch.setattr(driver, "wait_until_done", lambda *a, **k: None)
    monkeypatch.setattr(driver, "fetch", lambda *a, **k: {})

    driver.resubmit(RUN_ID)

    mocks["cluster"].submit_job.assert_called_once()
    assert mocks["cluster"].submit_job.call_args.kwargs["yaml_path"] == yaml_path


# ---------------------------------------------------------------------------
# DEV-1468 — cloud slayer: per-combo per-DB upload, fail-fast presence check,
# OPENAI key delivery, manifest + resubmit plumbing, no re-upload on resubmit.
# ---------------------------------------------------------------------------


def _write_dataset(tmp_path: Path, dbs: list[str]) -> Path:
    """A tiny mini_interact.jsonl mapping <db>_1 -> selected_database=<db>."""
    import json as _json
    f = tmp_path / "mini_interact.jsonl"
    f.write_text(
        "\n".join(
            _json.dumps({"instance_id": f"{db}_1", "selected_database": db})
            for db in dbs
        ) + "\n"
    )
    return f


def _setup_slayer_submit(
    monkeypatch, tmp_path, *, framework, slayer_setup, mode, dbs, lay_down=True,
):
    """Patch collaborators + dataset + the three artifact roots for a slayer
    submit. Returns (mocks, worktree, otf_cache_root, otf_ref_root)."""
    mocks = _patch_collaborators(monkeypatch)
    data_file = _write_dataset(tmp_path, dbs)
    worktree = tmp_path / "worktree"
    otf_cache = tmp_path / "main" / "slayer_otf_cache"
    otf_ref = tmp_path / "main" / "slayer_models_otf"

    monkeypatch.setattr(driver.paths, "benchmark_data_file", lambda *a, **k: data_file)
    monkeypatch.setattr(driver.paths, "benchmark_data_root", lambda *a, **k: tmp_path / "mini")
    monkeypatch.setattr(driver, "submitter_repo_root", lambda: worktree)
    monkeypatch.setattr(driver.paths, "slayer_otf_cache_root", lambda *, benchmark=None: otf_cache)
    monkeypatch.setattr(driver.paths, "slayer_models_otf_root", lambda *, benchmark=None: otf_ref)
    # read_api_keys_from_local_env now fails fast on a missing required key
    # (incl. OPENAI for slayer); set them so a successful submit doesn't raise
    # in an env without these vars.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    # download/wait/fetch never run under detach in these tests.
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    if lay_down:
        for db in dbs:
            if slayer_setup == "pre-encoded":
                d = worktree / "slayer_models" / db
                d.mkdir(parents=True)
                (d / "model.yaml").write_text("models: []\n")
            elif framework == "pydantic_ai_recursive":
                d = otf_cache / db
                d.mkdir(parents=True)
                (d / "_cache_fp.txt").write_text("fp")
            else:
                # DEV-1470: otf_encode now REQUIRES the deterministic cache
                # (`_cache_fp.txt`); the reference is an OPTIONAL seed shipped
                # only when present.
                dc = otf_cache / db
                dc.mkdir(parents=True)
                (dc / "_cache_fp.txt").write_text("fp")
                dr = otf_ref / db
                dr.mkdir(parents=True)
                (dr / "_reference_fp.txt").write_text("fp")
    return mocks, worktree, otf_cache, otf_ref


@pytest.mark.parametrize(
    "framework, slayer_setup, mode, artifact",
    [
        ("pydantic_ai_recursive", "pre-encoded", "c-interact", "slayer_models"),
        ("pydantic_ai_recursive", "on-the-fly", "a-interact", "slayer_otf_cache"),
        ("pydantic_ai_otf_encode", "on-the-fly", "a-interact", "slayer_models_otf"),
    ],
)
def test_submit_uploads_combo_dir_per_selected_db(
    monkeypatch, tmp_path, framework, slayer_setup, mode, artifact,
):
    """Submit uploads exactly the combo's local dir for each selected DB,
    under runs/<run-id>/slayer_setup/<artifact>/<db>/."""
    dbs = ["db_a", "db_b"]
    mocks, worktree, otf_cache, otf_ref = _setup_slayer_submit(
        monkeypatch, tmp_path, framework=framework, slayer_setup=slayer_setup,
        mode=mode, dbs=dbs,
    )
    src_root = {
        "slayer_models": worktree / "slayer_models",
        "slayer_otf_cache": otf_cache,
        "slayer_models_otf": otf_ref,
    }[artifact]

    args = FakeSubmitArgs(
        framework=framework, query_mode="slayer", mode=mode,
        slayer_setup=slayer_setup, instance_ids=("db_a_1", "db_b_1"),
        detach=True,
    )
    run_id = driver.submit(args)

    calls = mocks["gcs"].upload_dir_prefix.call_args_list
    uploaded = {(c.args[0], c.args[1]) for c in calls}
    expected = {
        (src_root / db,
         f"runs/{run_id}/slayer_setup/{artifact}/{db}")
        for db in dbs
    }
    if framework == "pydantic_ai_otf_encode":
        # DEV-1470: otf_encode now ships BOTH the cache (required) and the
        # reference (optional seed, present here). The parameterised artifact
        # is the reference; the cache is additional.
        expected |= {
            (otf_cache / db,
             f"runs/{run_id}/slayer_setup/slayer_otf_cache/{db}")
            for db in dbs
        }
    assert uploaded == expected, f"uploaded {uploaded}, expected {expected}"


def test_check_setup_auto_builds_missing_otf_cache(monkeypatch, tmp_path):
    """A missing REQUIRED slayer_otf_cache is built locally (no error):
    `_check_slayer_setup_present` triggers `ensure_db_cache` per missing DB and
    returns the db list instead of raising."""
    _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_otf_encode",
        slayer_setup="on-the-fly", mode="a-interact", dbs=["db_a", "db_b"],
        lay_down=False,  # cache (and optional reference) intentionally absent
    )
    built: list[str] = []
    seen_kwargs: list[dict] = []

    async def _fake_cache(db, **kw):
        built.append(db)
        seen_kwargs.append(kw)

    monkeypatch.setattr(driver, "ensure_db_cache", _fake_cache)
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1", "db_b_1"), detach=True,
    )
    dbs = driver._check_slayer_setup_present(args)
    assert dbs == ["db_a", "db_b"]
    assert sorted(built) == ["db_a", "db_b"]
    # Lock the worktree-safe contract: the cache is built under the resolved
    # roots (not a worktree-relative path) with force=False.
    assert all(
        kw["cache_root"] == driver.paths.slayer_otf_cache_root(
            benchmark="mini-interact",
        )
        for kw in seen_kwargs
    )
    assert all(
        kw["mini_interact_root"] == driver.paths.benchmark_data_root("mini-interact")
        for kw in seen_kwargs
    )
    assert all(kw["force"] is False for kw in seen_kwargs)


def test_submit_missing_otf_cache_auto_builds_then_proceeds(monkeypatch, tmp_path):
    """End-to-end: a missing deterministic cache no longer fails the submit —
    it's built locally and the submit proceeds to build/push/cluster."""
    mocks, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_recursive",
        slayer_setup="on-the-fly", mode="a-interact", dbs=["db_a"],
        lay_down=False,  # cache absent
    )
    built: list[str] = []

    async def _fake_cache(db, **kw):
        built.append(db)

    monkeypatch.setattr(driver, "ensure_db_cache", _fake_cache)
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    driver.submit(args)
    assert built == ["db_a"]
    mocks["image"].build_and_push.assert_called_once()
    mocks["cluster"].up.assert_called_once()


def test_submit_pre_encoded_missing_dir_raises(monkeypatch, tmp_path):
    """pre-encoded presence = non-empty dir; a missing slayer_models/<db>/
    raises before the cluster."""
    mocks, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_recursive",
        slayer_setup="pre-encoded", mode="c-interact", dbs=["db_a"],
        lay_down=False,
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive", query_mode="slayer", mode="c-interact",
        slayer_setup="pre-encoded", instance_ids=("db_a_1",), detach=True,
    )
    with pytest.raises(FileNotFoundError):
        driver.submit(args)
    mocks["cluster"].up.assert_not_called()


def test_submit_pre_encoded_empty_dir_raises(monkeypatch, tmp_path):
    """pre-encoded presence requires a NON-EMPTY dir — an empty
    slayer_models/<db>/ (e.g. a stale mkdir) must still fail fast."""
    mocks, worktree, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_recursive",
        slayer_setup="pre-encoded", mode="c-interact", dbs=["db_a"],
        lay_down=False,
    )
    (worktree / "slayer_models" / "db_a").mkdir(parents=True)  # exists but EMPTY
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive", query_mode="slayer", mode="c-interact",
        slayer_setup="pre-encoded", instance_ids=("db_a_1",), detach=True,
    )
    with pytest.raises(FileNotFoundError):
        driver.submit(args)
    mocks["cluster"].up.assert_not_called()


# DEV-1470 superseded the old "missing reference fails fast" contract for
# `otf_encode + on-the-fly`: the REQUIRED artifact is now the deterministic
# cache (`_cache_fp.txt`); the reference is an OPTIONAL seed. The replacement
# test is `test_otf_encode_submit_requires_cache_marker` further down.


def test_submit_dedups_uploads_when_instances_share_db(monkeypatch, tmp_path):
    """Upload is PER SELECTED DB, not per instance — two instances mapping to
    the same DB must produce exactly one upload."""
    import json as _json

    mocks, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_recursive",
        slayer_setup="on-the-fly", mode="a-interact", dbs=["db_a"],
    )
    # Two instances → one DB.
    (tmp_path / "mini_interact.jsonl").write_text(
        _json.dumps({"instance_id": "db_a_1", "selected_database": "db_a"}) + "\n"
        + _json.dumps({"instance_id": "db_a_2", "selected_database": "db_a"}) + "\n"
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1", "db_a_2"), detach=True,
    )
    driver.submit(args)
    calls = mocks["gcs"].upload_dir_prefix.call_args_list
    uploaded_dbs = [c.args[1].rsplit("/", 1)[-1] for c in calls]
    assert uploaded_dbs == ["db_a"], f"expected a single db_a upload, got {uploaded_dbs}"


def test_submit_job_args_carry_slayer_flags(monkeypatch, tmp_path):
    mocks, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_recursive",
        slayer_setup="on-the-fly", mode="a-interact", dbs=["db_a"],
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", slayer_storage_root="/data/slayer_models",
        instance_ids=("db_a_1",), detach=True,
    )
    driver.submit(args)
    job_args = mocks["cluster"].submit_job.call_args.kwargs["args"]
    assert "--slayer-setup" in job_args
    assert job_args[job_args.index("--slayer-setup") + 1] == "on-the-fly"
    assert "--slayer-storage-root" in job_args
    assert job_args[job_args.index("--slayer-storage-root") + 1] == "/data/slayer_models"


def test_read_api_keys_includes_openai_for_slayer(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5-20251001",
        query_mode="slayer",
    )
    assert keys.get("OPENAI_API_KEY") == "o"
    assert keys.get("ANTHROPIC_API_KEY") == "a"


def test_read_api_keys_excludes_openai_for_raw(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5-20251001",
        query_mode="raw",
    )
    assert "OPENAI_API_KEY" not in keys


def test_read_api_keys_raises_on_missing_required_key(monkeypatch):
    """Fail fast on a missing required key instead of silently dropping it —
    otherwise `resubmit` (no prereq check) surfaces opaque actor auth failures
    (CodeRabbit)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(driver.PrereqError) as exc:
        driver.read_api_keys_from_local_env(
            "anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5-20251001",
            query_mode="slayer",  # requires OPENAI_API_KEY
        )
    assert "OPENAI_API_KEY" in str(exc.value)
    assert "export OPENAI_API_KEY=" in exc.value.remediation


# ---------------------------------------------------------------------------
# DEV-1517 — OAuth token path for claude_sdk* frameworks.
# ---------------------------------------------------------------------------

_GOOD_TOKEN = "sk-ant-oat01-good-token"
_ANTHROPIC_KEY = "sk-ant-api-key"
_OPENAI_KEY = "sk-openai-key"


def test_read_api_keys_oauth_anthropic_usersim(monkeypatch):
    """claude_sdk + OAuth → returns CLAUDE_CODE_OAUTH_TOKEN and
    BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY; never contains ANTHROPIC_API_KEY."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
    )
    assert keys["CLAUDE_CODE_OAUTH_TOKEN"] == _GOOD_TOKEN
    assert keys["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_oauth_openai_usersim(monkeypatch):
    """claude_sdk + OAuth + openai user-sim → CLAUDE_CODE_OAUTH_TOKEN +
    OPENAI_API_KEY; no ANTHROPIC_API_KEY and no BIRD_INTERACT_LITELLM_* key."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "openai/gpt-4o",
        framework="claude_sdk",
    )
    assert keys["CLAUDE_CODE_OAUTH_TOKEN"] == _GOOD_TOKEN
    assert keys["OPENAI_API_KEY"] == _OPENAI_KEY
    assert "ANTHROPIC_API_KEY" not in keys
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_oauth_slayer_ships_openai_key(monkeypatch):
    """claude_sdk + OAuth + slayer → OPENAI_API_KEY is still shipped for
    channel-3 embeddings even though ANTHROPIC_API_KEY is omitted."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        query_mode="slayer",
    )
    assert keys["CLAUDE_CODE_OAUTH_TOKEN"] == _GOOD_TOKEN
    assert keys["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert keys["OPENAI_API_KEY"] == _OPENAI_KEY
    assert "ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_claude_sdk_no_oauth_raises(monkeypatch):
    """DEV-1535: claude_sdk + subscription auth opted-in (default
    no_subscription_auth=False) but no token → PrereqError. Replaces the
    pre-DEV-1535 silent-fallthrough-to-legacy behavior; the CLI now
    requires an explicit auth choice, and the driver mirrors that
    contract for callers (including resubmit) that don't go through the
    CLI."""
    from bird_interact_agents.cloud.prereqs import PrereqError

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    with pytest.raises(PrereqError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        driver.read_api_keys_from_local_env(
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )


def test_read_api_keys_claude_sdk_no_oauth_legacy_path_when_opted_out(
    monkeypatch,
):
    """The opt-out form: no_subscription_auth=True takes the legacy
    API-key path even without an OAuth token. Mirrors the post-DEV-1535
    CLI shape where `--no-subscription-auth` is the explicit choice."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        no_subscription_auth=True,
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_pydantic_ai_oauth_ignored(monkeypatch):
    """pydantic_ai framework + OAuth token set locally → legacy path; OAuth
    is silently ignored and ANTHROPIC_API_KEY is shipped as normal."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="pydantic_ai",
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_old_manifest_no_framework_legacy_path(monkeypatch):
    """Old manifests without a framework key default framework="" → legacy path.
    The log note fires; ANTHROPIC_API_KEY is shipped."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    # framework="" (default) → legacy
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys


def test_read_api_keys_oauth_bad_prefix_raises(monkeypatch):
    """claude_sdk + OAuth with wrong token prefix → PrereqError before any
    os.environ lookups, not a raw KeyError or silent cluster start."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api03-not-an-oauth-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    with pytest.raises(driver.PrereqError, match="sk-ant-oat01-"):
        driver.read_api_keys_from_local_env(
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )


def test_read_api_keys_oauth_missing_usersim_key_raises_prereq_error(monkeypatch):
    """claude_sdk + valid OAuth + anthropic user-sim but no ANTHROPIC_API_KEY
    → PrereqError (not KeyError) listing the missing key."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(driver.PrereqError, match="ANTHROPIC_API_KEY"):
        driver.read_api_keys_from_local_env(
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )


def test_read_api_keys_annotator_oauth_no_usersim_key_required(monkeypatch):
    """annotator framework + OAuth → no ANTHROPIC_API_KEY required for user-sim.
    Regression: resubmit() was passing agent_model as user_sim_model, which
    caused read_api_keys_from_local_env to require ANTHROPIC_API_KEY even when
    the annotator has no user simulator and only uses OAuth."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Must NOT raise — annotator has no user-sim, so no API key is needed.
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-opus-4-7", "",
        query_mode="raw", framework="annotator",
    )
    assert keys["CLAUDE_CODE_OAUTH_TOKEN"] == _GOOD_TOKEN
    assert "ANTHROPIC_API_KEY" not in keys
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in keys


def test_build_resubmit_args_old_manifest_no_framework_uses_get(monkeypatch):
    """_build_resubmit_args must use manifest.get('framework', '') so old
    manifests without the key don't raise KeyError (DEV-1517)."""
    # Old manifest without 'framework' key.
    manifest = {
        "run_id": RUN_ID,
        "query_mode": "raw",
        "mode": "c-interact",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 3,
        "max_depth": 3,
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
        "slayer_setup": "pre-encoded",
        "slayer_storage_root": "/data/slayer_models",
        "strict": False,
        "use_audited_gold_sql": False,
        "prompt_cache": True,
        "instance_ids": ["db_a_1"],
        # NOTE: no "framework" key — simulates pre-DEV-1517 manifest
    }
    # Must not raise KeyError.
    args = driver._build_resubmit_args(manifest, RUN_ID, ["db_a_1"], 2)
    # --framework must still be present in the job args (defaulting to "").
    fw_idx = args.index("--framework")
    assert args[fw_idx + 1] == ""


def test_build_manifest_carries_slayer_fields() -> None:
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", slayer_storage_root="/data/slayer_models",
        instance_ids=("db_a_1",),
    )
    manifest = driver.build_manifest(args, image_uri="x:tag", run_id=RUN_ID)
    assert manifest["slayer_setup"] == "on-the-fly"
    assert manifest["slayer_storage_root"] == "/data/slayer_models"


def test_resubmit_carries_slayer_fields_and_does_not_reupload(
    monkeypatch, tmp_path,
):
    """resubmit re-renders + re-runs the job with the slayer flags from the
    manifest, but does NOT re-upload — the setup is already in GCS under the
    run prefix and the actor downloads it."""
    mocks = _patch_collaborators(monkeypatch)
    # read_api_keys (called by resubmit) now fails fast on missing keys.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    yaml_path = Path("/tmp/resubmit-cluster.yaml")
    mocks["cluster"].render_from_manifest.return_value = yaml_path
    mocks["cluster"].head_address.return_value = "http://localhost:8265"
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai_otf_encode",
        "query_mode": "slayer",
        "mode": "a-interact",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1", "db_b_1"],
        "patience": 3,
        "strict": False,
        "use_audited_gold_sql": False,
        "max_depth": 3,
        "prompt_cache": True,
        "slayer_setup": "on-the-fly",
        "slayer_storage_root": "/data/slayer_models",
        "render_inputs": {"workers": 1, "actors_per_worker": 2},
    }
    mocks["gcs"].list_attempts.return_value = {"db_a_1": [1]}  # db_b_1 missing
    mocks["gcs"].read_row.side_effect = lambda rid, iid, n: {"error": None}
    monkeypatch.setattr(driver, "wait_until_done", lambda *a, **k: None)
    monkeypatch.setattr(driver, "fetch", lambda *a, **k: {})

    driver.resubmit(RUN_ID)

    job_args = mocks["cluster"].submit_job.call_args.kwargs["args"]
    assert "--slayer-setup" in job_args
    assert job_args[job_args.index("--slayer-setup") + 1] == "on-the-fly"
    assert "--slayer-storage-root" in job_args
    # The crux: resubmit must NOT re-upload the setup.
    mocks["gcs"].upload_dir_prefix.assert_not_called()


# ---------------------------------------------------------------------------
# DEV-1470 — otf_encode + on-the-fly: cache REQUIRED, reference OPTIONAL seed,
# instance_ids sorted by (db, iid) for dispatch grouping, fetch merge hook.
# ---------------------------------------------------------------------------


def _setup_otf_encode_submit(monkeypatch, tmp_path, *, dbs, lay_down_cache=True,
                              lay_down_reference=False):
    """Like `_setup_slayer_submit` but for the otf_encode + on-the-fly combo
    AFTER the DEV-1470 contract change: the REQUIRED artifact is the
    deterministic cache (`_cache_fp.txt`), the reference is OPTIONAL seed."""
    mocks = _patch_collaborators(monkeypatch)
    data_file = _write_dataset(tmp_path, dbs)
    worktree = tmp_path / "worktree"
    otf_cache = tmp_path / "main" / "slayer_otf_cache"
    otf_ref = tmp_path / "main" / "slayer_models_otf"

    monkeypatch.setattr(driver.paths, "benchmark_data_file", lambda *a, **k: data_file)
    monkeypatch.setattr(driver.paths, "benchmark_data_root", lambda *a, **k: tmp_path / "mini")
    monkeypatch.setattr(driver, "submitter_repo_root", lambda: worktree)
    monkeypatch.setattr(driver.paths, "slayer_otf_cache_root", lambda *, benchmark=None: otf_cache)
    monkeypatch.setattr(driver.paths, "slayer_models_otf_root", lambda *, benchmark=None: otf_ref)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    for db in dbs:
        if lay_down_cache:
            d = otf_cache / db
            d.mkdir(parents=True)
            (d / "_cache_fp.txt").write_text(f"cache-fp-{db}")
            (d / "datasources").mkdir()
            (d / "datasources" / f"{db}.yaml").write_text(
                f"connection_string: file://{db}\n"
            )
        if lay_down_reference:
            d = otf_ref / db
            d.mkdir(parents=True)
            (d / "_reference_fp.txt").write_text(f"ref-fp-{db}")
            (d / "models").mkdir()
            (d / "models" / "x.yaml").write_text(f"name: {db}\n")
    return mocks, otf_cache, otf_ref


def test_otf_encode_submit_auto_builds_missing_cache(monkeypatch, tmp_path):
    """For `otf_encode + on-the-fly` the deterministic `slayer_otf_cache` is the
    REQUIRED artifact, but a missing one is BUILT locally (no LLMs) rather than
    fail-fast — `ensure_db_cache` runs per missing DB and the submit proceeds."""
    mocks, *_ = _setup_otf_encode_submit(
        monkeypatch, tmp_path, dbs=["db_a"],
        lay_down_cache=False, lay_down_reference=True,
    )
    built: list[str] = []

    async def _fake_cache(db, **kw):
        built.append(db)

    monkeypatch.setattr(driver, "ensure_db_cache", _fake_cache)
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    driver.submit(args)
    assert built == ["db_a"]
    mocks["image"].build_and_push.assert_called_once()
    mocks["cluster"].up.assert_called_once()


def test_otf_encode_submit_accepts_missing_reference(monkeypatch, tmp_path):
    """DEV-1470: the reference is now OPTIONAL — submit must succeed when
    the cache is present but the reference is absent. The cloud will encode
    the reference for any missing db lazily."""
    mocks, _otf_cache, _otf_ref = _setup_otf_encode_submit(
        monkeypatch, tmp_path, dbs=["db_a"],
        lay_down_cache=True, lay_down_reference=False,
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    # Must NOT raise.
    driver.submit(args)
    mocks["cluster"].submit_job.assert_called_once()


def test_otf_encode_submit_uploads_cache_required_and_reference_optional(
    monkeypatch, tmp_path,
):
    """DEV-1470: when BOTH cache and reference exist locally for `otf_encode +
    on-the-fly`, the driver uploads both — cache (input) + reference (seed,
    so the cloud skips re-encoding that db)."""
    mocks, otf_cache, otf_ref = _setup_otf_encode_submit(
        monkeypatch, tmp_path, dbs=["db_a"],
        lay_down_cache=True, lay_down_reference=True,
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    run_id = driver.submit(args)

    calls = mocks["gcs"].upload_dir_prefix.call_args_list
    uploaded = {(c.args[0], c.args[1]) for c in calls}
    cache_prefix = f"runs/{run_id}/slayer_setup/slayer_otf_cache/db_a"
    ref_prefix = f"runs/{run_id}/slayer_setup/slayer_models_otf/db_a"
    assert (otf_cache / "db_a", cache_prefix) in uploaded, (
        f"cache must be uploaded; got {uploaded}"
    )
    assert (otf_ref / "db_a", ref_prefix) in uploaded, (
        f"reference seed must be uploaded when present; got {uploaded}"
    )


def test_otf_encode_submit_uploads_cache_only_when_no_local_reference(
    monkeypatch, tmp_path,
):
    """When the local reference is absent, only the cache is uploaded."""
    mocks, otf_cache, _otf_ref = _setup_otf_encode_submit(
        monkeypatch, tmp_path, dbs=["db_a"],
        lay_down_cache=True, lay_down_reference=False,
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    run_id = driver.submit(args)
    calls = mocks["gcs"].upload_dir_prefix.call_args_list
    uploaded = {(c.args[0], c.args[1]) for c in calls}
    cache_prefix = f"runs/{run_id}/slayer_setup/slayer_otf_cache/db_a"
    assert (otf_cache / "db_a", cache_prefix) in uploaded
    # No reference upload.
    assert all(not c.args[1].startswith(
        f"runs/{run_id}/slayer_setup/slayer_models_otf/"
    ) for c in calls), f"reference must not be uploaded; got {uploaded}"


def test_otf_encode_submit_passes_partial_reference_seeds(
    monkeypatch, tmp_path,
):
    """If the user has references for SOME of the selected DBs and not others,
    upload the available ones as seeds and let the cloud encode the rest."""
    mocks, _otf_cache, otf_ref = _setup_otf_encode_submit(
        monkeypatch, tmp_path, dbs=["db_a", "db_b"],
        lay_down_cache=True, lay_down_reference=False,
    )
    # Lay down a reference for db_a only.
    (otf_ref / "db_a").mkdir(parents=True)
    (otf_ref / "db_a" / "_reference_fp.txt").write_text("ref-fp-db_a")

    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1", "db_b_1"), detach=True,
    )
    run_id = driver.submit(args)
    prefixes = {c.args[1] for c in mocks["gcs"].upload_dir_prefix.call_args_list}
    # db_a reference uploaded as seed.
    assert f"runs/{run_id}/slayer_setup/slayer_models_otf/db_a" in prefixes
    # db_b reference NOT uploaded (absent locally — cloud will build it).
    assert f"runs/{run_id}/slayer_setup/slayer_models_otf/db_b" not in prefixes


def test_submit_groups_instance_ids_by_database(monkeypatch, tmp_path):
    """DEV-1470: same-db iids must be adjacent in the dispatch order so a
    single actor typically does all encoding for a given DB (reducing the
    cross-actor encode-race window). Achieved by sorting by (db, iid) in the
    `--instance-ids` arg passed to ray_app.

    Submit a permuted instance_ids list and assert the order in `--instance-ids`
    has db-runs (all db_a's, then all db_b's, etc.) regardless of input order.
    """
    import json as _json
    # Dataset with mixed iid→db mapping.
    dataset = tmp_path / "mini_interact.jsonl"
    rows = [
        {"instance_id": "db_b_2", "selected_database": "db_b"},
        {"instance_id": "db_a_2", "selected_database": "db_a"},
        {"instance_id": "db_a_1", "selected_database": "db_a"},
        {"instance_id": "db_b_1", "selected_database": "db_b"},
    ]
    dataset.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")

    mocks = _patch_collaborators(monkeypatch)
    worktree = tmp_path / "worktree"
    otf_cache = tmp_path / "main" / "slayer_otf_cache"
    otf_ref = tmp_path / "main" / "slayer_models_otf"
    monkeypatch.setattr(driver.paths, "benchmark_data_file", lambda *a, **k: dataset)
    monkeypatch.setattr(driver.paths, "benchmark_data_root", lambda *a, **k: tmp_path / "mini")
    monkeypatch.setattr(driver, "submitter_repo_root", lambda: worktree)
    monkeypatch.setattr(driver.paths, "slayer_otf_cache_root", lambda *, benchmark=None: otf_cache)
    monkeypatch.setattr(driver.paths, "slayer_models_otf_root", lambda *, benchmark=None: otf_ref)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())
    for db in ("db_a", "db_b"):
        (otf_cache / db).mkdir(parents=True)
        (otf_cache / db / "_cache_fp.txt").write_text("fp")

    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly",
        # Deliberately interleaved order.
        instance_ids=("db_b_2", "db_a_1", "db_b_1", "db_a_2"),
        detach=True,
    )
    driver.submit(args)

    job_args = mocks["cluster"].submit_job.call_args.kwargs["args"]
    raw = job_args[job_args.index("--instance-ids") + 1]
    ordered = raw.split(",")
    # L1 — exact `(selected_database, instance_id)` ordering. Within each DB,
    # iids sort ascending. Catches "grouped but unsorted within db" bugs.
    assert ordered == ["db_a_1", "db_a_2", "db_b_1", "db_b_2"], (
        f"--instance-ids must be sorted (db, iid), got {ordered}"
    )


def test_resubmit_groups_missing_instance_ids_by_database(monkeypatch, tmp_path):
    """The same db-grouping must apply on resubmit so retries don't undo the
    dispatch grouping."""
    import json as _json
    mocks = _patch_collaborators(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    dataset = tmp_path / "mini_interact.jsonl"
    rows = [
        {"instance_id": "db_b_1", "selected_database": "db_b"},
        {"instance_id": "db_a_1", "selected_database": "db_a"},
        {"instance_id": "db_a_2", "selected_database": "db_a"},
        {"instance_id": "db_b_2", "selected_database": "db_b"},
    ]
    dataset.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(driver.paths, "benchmark_data_file", lambda *a, **k: dataset)
    yaml_path = Path("/tmp/cluster.yaml")
    mocks["cluster"].render_from_manifest.return_value = yaml_path
    mocks["cluster"].head_address.return_value = "http://localhost:8265"
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID,
        "framework": "pydantic_ai_otf_encode",
        "query_mode": "slayer",
        "mode": "a-interact",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_b_1", "db_a_1", "db_a_2", "db_b_2"],
        "patience": 3,
        "strict": False,
        "use_audited_gold_sql": False,
        "max_depth": 3,
        "prompt_cache": True,
        "slayer_setup": "on-the-fly",
        "slayer_storage_root": "/data/slayer_models",
        "render_inputs": {"workers": 1, "actors_per_worker": 2},
    }
    # All four still missing (no completed rows) → all retried.
    mocks["gcs"].list_attempts.return_value = {}
    monkeypatch.setattr(driver, "wait_until_done", lambda *a, **k: None)
    monkeypatch.setattr(driver, "fetch", lambda *a, **k: {})

    driver.resubmit(RUN_ID)

    job_args = mocks["cluster"].submit_job.call_args.kwargs["args"]
    raw = job_args[job_args.index("--instance-ids") + 1]
    ordered = raw.split(",")
    # L1 — exact `(db, iid)` ordering on resubmit too.
    assert ordered == ["db_a_1", "db_a_2", "db_b_1", "db_b_2"], (
        f"resubmit --instance-ids must be sorted (db, iid), got {ordered}"
    )


def test_fetch_calls_post_run_merge_after_collation(monkeypatch, tmp_path):
    """DEV-1470: `fetch(run_id)` must run `post_run_merge.merge_post_run_into_warm_cache`
    AFTER `_collation.collate`, passing the local OTF reference root. The
    merge promotes per-DB cloud-encoded shards from
    `<run_dir>/post_run/slayer_models_otf/<shard>/<db>/` into the warm cache."""
    from bird_interact_agents.cloud import post_run_merge as _prm
    from bird_interact_agents.cloud import collation as _collation

    mocks = _patch_collaborators(monkeypatch)
    fake_results = tmp_path / "results"
    monkeypatch.setattr(driver.paths, "results_root", lambda: fake_results)
    fake_ref_root = tmp_path / "warm" / "slayer_models_otf"
    monkeypatch.setattr(driver.paths, "slayer_models_otf_root", lambda *, benchmark=None: fake_ref_root)
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID, "instance_ids": ["db_a_1"],
    }
    # The real concurrent_download_prefix creates dest when there are blobs;
    # mock it to create the empty dest so the manifest write_text below works.
    def fake_download(run_id, dest, **kw):
        Path(dest).mkdir(parents=True, exist_ok=True)
    mocks["gcs"].concurrent_download_prefix.side_effect = fake_download
    order: list[str] = []

    def fake_collate(run_dir, manifest):
        order.append("collate")
        return {"phase_passes": 1}

    def fake_merge(*, run_dir, reference_root):
        order.append("merge")
        assert reference_root == fake_ref_root, (
            f"merger called with wrong reference_root: {reference_root!r} vs "
            f"{fake_ref_root!r}"
        )
        return {"merged_dbs": [], "ignored_shards": []}

    monkeypatch.setattr(_collation, "collate", fake_collate)
    monkeypatch.setattr(_prm, "merge_post_run_into_warm_cache", fake_merge)

    metrics = driver.fetch(RUN_ID)

    assert order == ["collate", "merge"], (
        f"merge must run AFTER collate; saw {order}"
    )
    # Merge report surfaced into metrics so the CLI can summarise it.
    assert "merge_report" in metrics


def test_fetch_continues_when_merge_returns_no_shards(monkeypatch, tmp_path):
    """Most runs (raw mode, pre-encoded slayer, recursive) have no
    post_run/ shards. The merger must return cleanly (empty report) and
    `fetch` must still return the collated metrics."""
    from bird_interact_agents.cloud import post_run_merge as _prm

    mocks = _patch_collaborators(monkeypatch)
    fake_results = tmp_path / "results"
    monkeypatch.setattr(driver.paths, "results_root", lambda: fake_results)
    monkeypatch.setattr(
        driver.paths, "slayer_models_otf_root",
        lambda *, benchmark=None: tmp_path / "warm" / "slayer_models_otf",
    )
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID, "instance_ids": ["db_a_1"],
    }
    def fake_download(run_id, dest, **kw):
        Path(dest).mkdir(parents=True, exist_ok=True)
    mocks["gcs"].concurrent_download_prefix.side_effect = fake_download
    monkeypatch.setattr(
        driver._collation, "collate",
        lambda run_dir, manifest: {"phase_passes": 1},
    )
    # Empty merge report.
    monkeypatch.setattr(
        _prm, "merge_post_run_into_warm_cache",
        lambda **kw: {"merged_dbs": [], "ignored_shards": []},
    )
    metrics = driver.fetch(RUN_ID)
    assert metrics["merge_report"]["merged_dbs"] == []


# ---------------------------------------------------------------------------
# Benchmark plumbing: dataset flows through the manifest and the actor job args;
# the actor + instance→db read the benchmark's tasks file.
# ---------------------------------------------------------------------------


def test_manifest_and_job_args_carry_benchmark(monkeypatch):
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer",
        mode="one-shot", slayer_setup="on-the-fly",
    )
    args.dataset = "livesqlbench-base-lite-sqlite"

    prefix = "benchmark-data/livesqlbench/abc123/"
    m = driver.build_manifest(
        args, image_uri="img:tag", run_id="rid", benchmark_data_prefix=prefix,
    )
    assert m["dataset"] == "livesqlbench-base-lite-sqlite"
    assert m["benchmark_data_prefix"] == prefix

    # Avoid reading a real tasks file for the db-grouped sort.
    monkeypatch.setattr(
        driver, "_instance_ids_sorted_by_db",
        lambda ids, benchmark="mini-interact": list(ids),
    )
    ja = driver._build_job_args(
        args, "rid", attempt=1, benchmark_data_prefix=prefix,
    )
    assert ja[ja.index("--dataset") + 1] == "livesqlbench-base-lite-sqlite"
    assert "--gold-file" not in ja
    assert ja[ja.index("--benchmark-data-prefix") + 1] == prefix


def test_manifest_defaults_to_mini_interact_benchmark():
    args = FakeSubmitArgs()  # default dataset → mini_interact, no gold
    m = driver.build_manifest(args, image_uri="img:tag", run_id="rid")
    assert m["dataset"] == "mini-interact"
    # No prefix passed → key present but None (back-compat for direct callers).
    assert m["benchmark_data_prefix"] is None


# ---------------------------------------------------------------------------
# _validate_instance_ids: fail fast before any cloud touch
# ---------------------------------------------------------------------------


def test_validate_instance_ids_raises_for_unknown_ids(tmp_path):
    """Unknown instance_ids must raise ValueError before any cloud call."""
    data_file = tmp_path / "mini_interact.jsonl"
    data_file.write_text(
        '{"instance_id": "alien_1", "selected_database": "alien"}\n'
        '{"instance_id": "alien_2", "selected_database": "alien"}\n'
    )
    import bird_interact_agents.paths as _paths
    import unittest.mock as _mock
    with _mock.patch.object(_paths, "benchmark_data_file", return_value=data_file):
        with pytest.raises(ValueError, match="shop_1"):
            driver._validate_instance_ids(["alien_1", "shop_1"], "mini-interact")


def test_validate_instance_ids_passes_for_known_ids(tmp_path):
    """All-known instance_ids must not raise."""
    data_file = tmp_path / "mini_interact.jsonl"
    data_file.write_text(
        '{"instance_id": "alien_1", "selected_database": "alien"}\n'
    )
    import bird_interact_agents.paths as _paths
    import unittest.mock as _mock
    with _mock.patch.object(_paths, "benchmark_data_file", return_value=data_file):
        driver._validate_instance_ids(["alien_1"], "mini-interact")  # no raise


def test_validate_instance_ids_skips_when_data_file_absent(tmp_path):
    """Missing local data file → no error (resubmit on a machine without data)."""
    import bird_interact_agents.paths as _paths
    import unittest.mock as _mock
    with _mock.patch.object(
        _paths, "benchmark_data_file", return_value=tmp_path / "absent.jsonl"
    ):
        driver._validate_instance_ids(["nonexistent_1"], "mini-interact")  # no raise


def test_submit_raises_before_cloud_for_invalid_instance_ids(monkeypatch, tmp_path):
    """submit() must raise ValueError for unknown instance_ids before touching
    prereqs, image build, or any GCS/cluster call."""
    mocks = _patch_collaborators(monkeypatch)
    data_file = tmp_path / "mini_interact.jsonl"
    data_file.write_text('{"instance_id": "alien_1", "selected_database": "alien"}\n')
    monkeypatch.setattr(driver.paths, "benchmark_data_file", lambda *a, **k: data_file)

    with pytest.raises(ValueError, match="shop_1"):
        driver.submit(FakeSubmitArgs(instance_ids=("shop_1", "fake_99")))
    mocks["prereqs"].check.assert_not_called()
    mocks["image"].build_and_push.assert_not_called()
    mocks["cluster"].up.assert_not_called()


# ---------------------------------------------------------------------------
# De-bake: submit uploads the dataset to GCS (upload-once) and threads the
# returned prefix into the manifest + actor job args; the gated gold must live
# under the data root so it rides along in that upload.
# ---------------------------------------------------------------------------


def test_submit_uploads_dataset_and_threads_prefix(monkeypatch):
    """`submit` calls `benchmark_data.ensure_uploaded(benchmark)` and threads
    the returned content-hashed prefix into both the manifest and the actor
    job args (`--benchmark-data-prefix`)."""
    mocks = _patch_collaborators(monkeypatch)
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())
    mocks["benchmark_data"].ensure_uploaded.return_value = (
        "benchmark-data/mini_interact/feedface/"
    )

    args = FakeSubmitArgs(detach=True)
    driver.submit(args)

    # Uploaded the run's benchmark dataset.
    assert mocks["benchmark_data"].ensure_uploaded.call_args.args[0] == "mini-interact"
    # Manifest carries the prefix.
    manifest = mocks["gcs"].write_manifest.call_args.args[1]
    assert manifest["benchmark_data_prefix"] == "benchmark-data/mini_interact/feedface/"
    # Actor job args carry the prefix.
    job_args = mocks["cluster"].submit_job.call_args.kwargs["args"]
    assert job_args[job_args.index("--benchmark-data-prefix") + 1] == (
        "benchmark-data/mini_interact/feedface/"
    )


def test_resubmit_threads_benchmark_prefix(monkeypatch):
    """`_build_resubmit_args` re-threads the manifest's benchmark_data_prefix so
    the actor re-downloads the dataset; absent on pre-de-bake manifests."""
    monkeypatch.setattr(
        driver, "_instance_ids_sorted_by_db",
        lambda ids, benchmark="mini-interact": list(ids),
    )
    manifest = {
        "framework": "pydantic_ai", "query_mode": "raw", "mode": "c-interact",
        "dataset": "mini-interact", "agent_model": "m", "user_sim_model": "u",
        "benchmark_data_prefix": "benchmark-data/mini_interact/abc/",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }
    ja = driver._build_resubmit_args(manifest, "rid", ["db_a_1"], 2)
    assert ja[ja.index("--benchmark-data-prefix") + 1] == (
        "benchmark-data/mini_interact/abc/"
    )
    # Pre-de-bake manifest (no prefix) → flag omitted.
    manifest.pop("benchmark_data_prefix")
    ja2 = driver._build_resubmit_args(manifest, "rid", ["db_a_1"], 2)
    assert "--benchmark-data-prefix" not in ja2


def test_instance_ids_sorted_by_db_falls_back_when_data_file_absent(
    monkeypatch, tmp_path,
):
    """De-bake: `resubmit` may run on a machine without the local dataset, so a
    missing benchmark data file must NOT crash `_instance_ids_sorted_by_db` —
    it falls back to input order (DB-grouping is only a dispatch optimization,
    not a correctness gate) (Codex)."""
    monkeypatch.setattr(
        driver.paths, "benchmark_data_file",
        lambda *a, **k: tmp_path / "absent.jsonl",
    )
    ids = ["z_2", "a_1", "m_3"]
    assert driver._instance_ids_sorted_by_db(ids, "mini-interact") == ids


def test_resubmit_omits_dataset_for_pre_dataset_manifest(monkeypatch):
    """A manifest with NO 'dataset' key was written before --dataset existed,
    so its pinned image's ray_app rejects --dataset. Resubmit must OMIT both
    --dataset and --benchmark-data-prefix and let the old baked image run
    (Codex)."""
    monkeypatch.setattr(
        driver, "_instance_ids_sorted_by_db",
        lambda ids, benchmark="mini-interact": list(ids),
    )
    manifest = {
        "framework": "pydantic_ai", "query_mode": "raw", "mode": "c-interact",
        "agent_model": "m", "user_sim_model": "u",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }  # neither 'dataset' nor 'benchmark_data_prefix'
    ja = driver._build_resubmit_args(manifest, "rid", ["db_a_1"], 2)
    assert "--dataset" not in ja
    assert "--benchmark-data-prefix" not in ja


# ---------------------------------------------------------------------------
# _submit_benchmark: benchmark fallback for annotate args (args.benchmark)
# ---------------------------------------------------------------------------


def test_submit_benchmark_uses_dataset_when_present():
    """Normal submit args carry args.dataset — _submit_benchmark must use it."""
    args = FakeSubmitArgs()
    args.dataset = "livesqlbench-base-lite-sqlite"
    assert driver._submit_benchmark(args) == "livesqlbench-base-lite-sqlite"


def test_submit_benchmark_falls_back_to_benchmark_attr():
    """Annotate args carry args.benchmark but no args.dataset; _submit_benchmark
    must fall back to derive the correct benchmark."""
    ns = argparse.Namespace(benchmark="livesqlbench-base-lite-sqlite")
    assert driver._submit_benchmark(ns) == "livesqlbench-base-lite-sqlite"



# ---------------------------------------------------------------------------
# DEV-1523: BIRD_PG_* forwarding to cloud workers
# ---------------------------------------------------------------------------


def test_read_api_keys_forwards_bird_pg_vars_when_set(monkeypatch):
    """BIRD_PG_* vars that are set locally must be included in the env dict
    forwarded to cloud workers so postgres benchmarks can connect to the
    same server from the worker node."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.setenv("BIRD_PG_HOST", "pg.example.com")
    monkeypatch.setenv("BIRD_PG_PORT", "5433")
    monkeypatch.setenv("BIRD_PG_USER", "myuser")
    monkeypatch.setenv("BIRD_PG_PASSWORD", "mysecret")
    monkeypatch.setenv("BIRD_PG_STATEMENT_TIMEOUT", "60000")

    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
    )

    assert keys.get("BIRD_PG_HOST") == "pg.example.com"
    assert keys.get("BIRD_PG_PORT") == "5433"
    assert keys.get("BIRD_PG_USER") == "myuser"
    assert keys.get("BIRD_PG_PASSWORD") == "mysecret"
    assert keys.get("BIRD_PG_STATEMENT_TIMEOUT") == "60000"


def test_read_api_keys_does_not_forward_unset_bird_pg_vars(monkeypatch):
    """If BIRD_PG_* vars are not set, they must NOT appear in the forwarded
    dict (no empty-string keys that would shadow defaults on the worker)."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    for pg_var in ("BIRD_PG_HOST", "BIRD_PG_PORT", "BIRD_PG_USER",
                   "BIRD_PG_PASSWORD", "BIRD_PG_STATEMENT_TIMEOUT"):
        monkeypatch.delenv(pg_var, raising=False)

    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
    )

    for pg_var in ("BIRD_PG_HOST", "BIRD_PG_PORT", "BIRD_PG_USER",
                   "BIRD_PG_PASSWORD", "BIRD_PG_STATEMENT_TIMEOUT"):
        assert pg_var not in keys, f"{pg_var} must not appear when not set locally"


def test_read_api_keys_oauth_forwards_bird_pg_vars(monkeypatch):
    """Same forwarding must occur on the OAuth (claude_sdk) path."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.setenv("BIRD_PG_HOST", "pg.example.com")
    monkeypatch.setenv("BIRD_PG_PASSWORD", "mysecret")
    for pg_var in ("BIRD_PG_PORT", "BIRD_PG_USER", "BIRD_PG_STATEMENT_TIMEOUT"):
        monkeypatch.delenv(pg_var, raising=False)

    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
    )

    assert keys.get("BIRD_PG_HOST") == "pg.example.com"
    assert keys.get("BIRD_PG_PASSWORD") == "mysecret"
    assert "BIRD_PG_PORT" not in keys


# ---------------------------------------------------------------------------
# kill_after_fetch — auto-teardown when the cluster is still up after fetch.
# ---------------------------------------------------------------------------


def _setup_fetch_mocks(monkeypatch, tmp_path, *, head_alive, terminal_state, n_attempts, n_total):
    """Patch all collaborators needed by driver.fetch and return the mock set."""
    mocks = _patch_collaborators(monkeypatch)
    fake_results = tmp_path / "results"
    monkeypatch.setattr(driver, "local_results_root", lambda: fake_results)
    monkeypatch.setattr(
        driver.paths, "slayer_models_otf_root",
        lambda *, benchmark=None: tmp_path / "warm" / "slayer_models_otf",
    )
    monkeypatch.setattr(
        driver.paths, "annotations_root",
        lambda: tmp_path / "annotations",
    )
    mocks["gcs"].read_manifest.return_value = {
        "run_id": RUN_ID,
        "instance_ids": [f"db_a_{i}" for i in range(n_total)],
    }
    mocks["gcs"].read_status.return_value = (
        {"terminal_state": terminal_state} if terminal_state else {}
    )
    mocks["gcs"].list_attempts.return_value = {
        f"db_a_{i}": [1] for i in range(n_attempts)
    }
    mocks["cluster"].head_is_alive.return_value = head_alive

    def fake_download(run_id, dest, **kw):
        Path(dest).mkdir(parents=True, exist_ok=True)
    mocks["gcs"].concurrent_download_prefix.side_effect = fake_download

    from bird_interact_agents.cloud import post_run_merge as _prm
    monkeypatch.setattr(
        driver._collation, "collate",
        lambda run_dir, manifest: {"phase_passes": 1},
    )
    monkeypatch.setattr(
        _prm, "merge_post_run_into_warm_cache",
        lambda **kw: {"merged_dbs": [], "ignored_shards": []},
    )
    monkeypatch.setattr(
        _prm, "merge_submission_annotations",
        lambda **kw: _prm.AnnotationMergeReport(run_id=RUN_ID, benchmark="mini_interact"),
    )
    return mocks


def test_fetch_kills_cluster_when_complete_and_head_alive(monkeypatch, tmp_path):
    """fetch() with kill_after_fetch=True must call kill() when the run is
    complete (terminal_state=done) and the cluster head is still alive."""
    mocks = _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state="done", n_attempts=2, n_total=2,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID, kill_after_fetch=True)

    assert kill_calls == [RUN_ID], "kill must be called exactly once with the run_id"


def test_fetch_does_not_kill_when_head_already_dead(monkeypatch, tmp_path):
    """fetch() must not attempt kill() when head_is_alive returns False — the
    cluster is already gone."""
    mocks = _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=False, terminal_state="done", n_attempts=2, n_total=2,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID, kill_after_fetch=True)

    assert kill_calls == [], "kill must NOT be called when head is already dead"


def test_fetch_kills_even_when_run_incomplete(monkeypatch, tmp_path):
    """fetch() with kill_after_fetch=True must kill the cluster regardless of
    run completion. Previously the auto-kill guarded on
    ``terminal in ('done', 'error')`` OR ``all_attempts_present``, which
    silently leaked clusters when a waiter terminated with ``timed-out`` /
    ``stalled`` / ``headless`` (the cluster only writes ``done``/``error``
    itself; waiter-side terminals never reach ``status.json``). The CLI's
    ``--no-kill`` flag is the documented escape hatch for mid-run inspection."""
    mocks = _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state=None, n_attempts=1, n_total=3,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID, kill_after_fetch=True)

    assert kill_calls == [RUN_ID], (
        "kill MUST be called when kill_after_fetch=True and head is alive, "
        "regardless of completion — caller intent is authoritative"
    )


def test_fetch_kills_on_waiter_timeout_terminal(monkeypatch, tmp_path):
    """Regression test: a waiter that returned ``timed-out`` invokes fetch
    with kill_after_fetch=True. The cluster's own ``status.json`` still
    shows no terminal (cluster is alive and slow), but the operator has
    given up — fetch must kill the cluster anyway. The pre-fix behaviour
    leaked the cluster because ``timed-out`` isn't in ``(done, error)``
    and attempt count was below total."""
    mocks = _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state=None, n_attempts=297, n_total=298,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID, kill_after_fetch=True)

    assert kill_calls == [RUN_ID]


def test_fetch_does_not_kill_when_kill_after_fetch_false(monkeypatch, tmp_path):
    """Default behaviour (kill_after_fetch=False): cluster is never killed even
    when the run is complete and the head is alive."""
    mocks = _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state="done", n_attempts=2, n_total=2,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID)  # kill_after_fetch defaults to False

    assert kill_calls == [], "kill must NOT be called when kill_after_fetch=False"
    # head_is_alive should not even be checked when kill_after_fetch=False.
    mocks["cluster"].head_is_alive.assert_not_called()


def test_fetch_kills_cluster_when_terminal_state_is_error(monkeypatch, tmp_path):
    """fetch() with kill_after_fetch=True must call kill() when terminal_state
    is 'error' (not just 'done') — both are complete terminal states."""
    _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state="error", n_attempts=2, n_total=2,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID, kill_after_fetch=True)

    assert kill_calls == [RUN_ID], "kill must be called when terminal_state=error"


def test_fetch_kills_cluster_when_attempts_reach_total_no_terminal_state(monkeypatch, tmp_path):
    """fetch() with kill_after_fetch=True must call kill() when all attempts
    have been recorded but no terminal_state is set yet — the count-based
    completion branch."""
    _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state=None, n_attempts=2, n_total=2,
    )
    kill_calls: list[str] = []
    monkeypatch.setattr(driver, "kill", lambda rid: kill_calls.append(rid))

    driver.fetch(RUN_ID, kill_after_fetch=True)

    assert kill_calls == [RUN_ID], "kill must be called when attempts == total"


def test_fetch_surfaces_kill_error_in_metrics(monkeypatch, tmp_path):
    """When kill() raises during auto-teardown, fetch() must catch the exception,
    store it in metrics["kill_after_fetch_error"], and return normally — so
    successfully collated results are not lost."""
    _setup_fetch_mocks(
        monkeypatch, tmp_path,
        head_alive=True, terminal_state="done", n_attempts=2, n_total=2,
    )
    monkeypatch.setattr(driver, "kill", lambda _rid: (_ for _ in ()).throw(RuntimeError("cluster gone")))

    metrics = driver.fetch(RUN_ID, kill_after_fetch=True)

    assert "kill_after_fetch_error" in metrics
    assert "cluster gone" in metrics["kill_after_fetch_error"]


# ---------------------------------------------------------------------------
# DEV-1530 — --no-subscription-auth flag in read_api_keys_from_local_env
# and build_manifest / build_annotator_manifest / resubmit.
# ---------------------------------------------------------------------------


def test_read_api_keys_no_subscription_auth_flag_uses_legacy_path(monkeypatch):
    """claude_sdk + valid OAuth + no_subscription_auth=True → legacy path;
    ANTHROPIC_API_KEY shipped directly; CLAUDE_CODE_OAUTH_TOKEN not shipped."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        no_subscription_auth=True,
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_no_subscription_auth_flag_annotator(monkeypatch):
    """annotator + valid OAuth + no_subscription_auth=True → legacy path;
    ANTHROPIC_API_KEY shipped directly."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-opus-4-7", "",
        framework="annotator",
        no_subscription_auth=True,
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys
    assert "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY" not in keys


def test_read_api_keys_no_subscription_auth_flag_noop_on_pydantic_ai(monkeypatch):
    """pydantic_ai + no_subscription_auth=True → already legacy; flag is a no-op."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="pydantic_ai",
        no_subscription_auth=True,
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys


def test_read_api_keys_no_subscription_auth_bad_token_not_validated(monkeypatch):
    """claude_sdk + bad OAuth token prefix + no_subscription_auth=True
    → no error; bad token is never validated when the flag is set."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-bad-prefix-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        no_subscription_auth=True,
    )
    assert keys["ANTHROPIC_API_KEY"] == _ANTHROPIC_KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in keys


def test_build_manifest_stores_no_subscription_auth() -> None:
    """build_manifest records no_subscription_auth=True so resubmit reads it."""
    args = FakeSubmitArgs(no_subscription_auth=True)
    manifest = driver.build_manifest(args, image_uri="img:tag", run_id=RUN_ID)
    assert manifest["no_subscription_auth"] is True


def test_build_manifest_no_subscription_auth_default_false() -> None:
    """build_manifest defaults no_subscription_auth to False when absent from args."""
    # Manually create an args-like object without the attribute.
    import argparse as _ap
    args = _ap.Namespace(
        framework="pydantic_ai", query_mode="raw",
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        mode="c-interact", instance_ids=["db_a_1"],
        patience=3, strict=False, use_audited_gold_sql=False,
        max_depth=3, prompt_cache=True,
        workers=2, actors_per_worker=2,
        worker_type="e2-standard-4", max_runtime_hours=4,
        dataset="mini-interact", gold_file=None,
        slayer_setup="pre-encoded",
        slayer_storage_root="/data/slayer_models",
    )
    manifest = driver.build_manifest(args, image_uri="img:tag", run_id=RUN_ID)
    assert manifest["no_subscription_auth"] is False


def _fake_annotate_args(**overrides):
    """Minimal args object for build_annotator_manifest / submit_annotator tests."""
    ns = argparse.Namespace(
        benchmark="mini-interact",
        agent_model="anthropic/claude-opus-4-7",
        effort="medium",
        override=False,
        workers=1,
        actors_per_worker=2,
        worker_type="e2-standard-4",
        max_runtime_hours=2,
        instance_ids=["alien_1"],
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_build_annotator_manifest_stores_no_subscription_auth() -> None:
    """build_annotator_manifest records no_subscription_auth=True."""
    args = _fake_annotate_args(no_subscription_auth=True)
    manifest = driver.build_annotator_manifest(args, image_uri="img:tag", run_id=RUN_ID)
    assert manifest["no_subscription_auth"] is True


def test_build_annotator_manifest_no_subscription_auth_default_false() -> None:
    """build_annotator_manifest defaults no_subscription_auth to False."""
    args = _fake_annotate_args()
    manifest = driver.build_annotator_manifest(args, image_uri="img:tag", run_id=RUN_ID)
    assert manifest["no_subscription_auth"] is False


def test_submit_passes_no_subscription_auth_to_read_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit() must pass no_subscription_auth=True from args to
    read_api_keys_from_local_env so the correct auth path is used."""
    mocks = _patch_collaborators(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    captured: dict = {}
    _orig = driver.read_api_keys_from_local_env

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return _orig(*args, **kwargs)

    monkeypatch.setattr(driver, "read_api_keys_from_local_env", _spy)
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    args = FakeSubmitArgs(framework="claude_sdk", no_subscription_auth=True, detach=True)
    driver.submit(args)

    assert captured.get("no_subscription_auth") is True


def test_submit_annotator_passes_no_subscription_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_annotator() must add no_subscription_auth to _prereq_args and pass it
    to read_api_keys_from_local_env."""
    mocks = _patch_collaborators(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    prereqs_captured: dict = {}
    _orig_check = driver.prereqs.check

    def _spy_prereqs(args):
        prereqs_captured["no_subscription_auth"] = getattr(
            args, "no_subscription_auth", None
        )

    monkeypatch.setattr(driver.prereqs, "check", _spy_prereqs)

    keys_captured: dict = {}
    _orig_keys = driver.read_api_keys_from_local_env

    def _spy_keys(*args, **kwargs):
        keys_captured.update(kwargs)
        return _orig_keys(*args, **kwargs)

    monkeypatch.setattr(driver, "read_api_keys_from_local_env", _spy_keys)
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    ann_args = _fake_annotate_args(
        no_subscription_auth=True, run_id=None, detach=True, allow_dirty=False,
    )
    driver.submit_annotator(ann_args)

    assert prereqs_captured.get("no_subscription_auth") is True
    assert keys_captured.get("no_subscription_auth") is True


def test_resubmit_reads_no_subscription_auth_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resubmit reads no_subscription_auth from the manifest and passes it to
    read_api_keys_from_local_env so the auth path matches the original submit."""
    captured: dict = {}
    _orig = driver.read_api_keys_from_local_env

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return _orig(*args, **kwargs)

    monkeypatch.setattr(driver, "read_api_keys_from_local_env", _spy)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    manifest = {
        "run_id": RUN_ID,
        "framework": "claude_sdk",
        "mode": "c-interact",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "instance_ids": ["db_a_1"],
        "no_subscription_auth": True,
        "render_inputs": {
            "workers": 1, "actors_per_worker": 1,
            "worker_type": "e2-standard-4", "zone": "us-central1-a",
            "worker_sa": "sa@project.iam.gserviceaccount.com",
            "max_runtime_hours": 1, "image_uri": "img:tag",
            "project": "p", "region": "us-central1",
        },
    }
    mocks: dict = {}
    for attr in ("gcs", "cluster"):
        m = MagicMock(name=attr)
        mocks[attr] = m
        monkeypatch.setattr(f"bird_interact_agents.cloud.driver.{attr}", m)
    mocks["gcs"].read_manifest.return_value = manifest
    mocks["gcs"].list_attempts.return_value = {}
    mocks["cluster"].render_from_manifest.return_value = Path("/tmp/cluster.yaml")
    mocks["cluster"].head_address.return_value = "ray://10.0.0.1:10001"
    mocks["cluster"].submit_job.return_value = "raysubmit_resub"
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    driver.resubmit(RUN_ID)

    assert captured.get("no_subscription_auth") is True


def test_resubmit_annotator_reads_no_subscription_auth_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resubmit() annotator branch also forwards no_subscription_auth from manifest."""
    captured: dict = {}
    _orig = driver.read_api_keys_from_local_env

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return _orig(*args, **kwargs)

    monkeypatch.setattr(driver, "read_api_keys_from_local_env", _spy)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    manifest = {
        "run_id": RUN_ID,
        "framework": "annotator",
        "mode": "annotate",
        "query_mode": "raw",
        "agent_model": "anthropic/claude-opus-4-7",
        "instance_ids": ["db_a_1"],
        "no_subscription_auth": True,
        "render_inputs": {
            "workers": 1, "actors_per_worker": 1,
            "worker_type": "e2-standard-4", "zone": "us-central1-a",
            "worker_sa": "sa@project.iam.gserviceaccount.com",
            "max_runtime_hours": 1, "image_uri": "img:tag",
            "project": "p", "region": "us-central1",
        },
    }
    mocks: dict = {}
    for attr in ("gcs", "cluster"):
        m = MagicMock(name=attr)
        mocks[attr] = m
        monkeypatch.setattr(f"bird_interact_agents.cloud.driver.{attr}", m)
    mocks["gcs"].read_manifest.return_value = manifest
    mocks["gcs"].list_attempts.return_value = {}
    mocks["cluster"].render_from_manifest.return_value = Path("/tmp/cluster.yaml")
    mocks["cluster"].head_address.return_value = "ray://10.0.0.1:10001"
    mocks["cluster"].submit_job.return_value = "raysubmit_resub_ann"
    monkeypatch.setattr(driver, "wait_until_done", MagicMock())
    monkeypatch.setattr(driver, "fetch", MagicMock())

    driver.resubmit(RUN_ID)

    assert captured.get("no_subscription_auth") is True


# ---------------------------------------------------------------------------
# Postgres benchmark: read_api_keys_from_local_env must NOT forward BIRD_PG_*
# ---------------------------------------------------------------------------


def test_read_api_keys_no_pg_vars_for_postgres_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BIRD_PG_* env vars must NOT be forwarded when the dataset is a postgres
    benchmark (livesqlbench-base-lite) — the worker runs its own bundled server
    and forwarding an external address would override localhost."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("BIRD_PG_HOST", "external.db.host")
    monkeypatch.setenv("BIRD_PG_PORT", "5432")
    monkeypatch.setenv("BIRD_PG_USER", "pguser")
    monkeypatch.setenv("BIRD_PG_PASSWORD", "pgpass")

    result = driver.read_api_keys_from_local_env(
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-haiku-4-5-20251001",
        dataset="livesqlbench-base-lite",
    )

    for pg_key in ("BIRD_PG_HOST", "BIRD_PG_PORT", "BIRD_PG_USER",
                   "BIRD_PG_PASSWORD", "BIRD_PG_STATEMENT_TIMEOUT"):
        assert pg_key not in result, (
            f"{pg_key} must not be forwarded for postgres benchmarks"
        )


def test_read_api_keys_forwards_pg_vars_for_sqlite_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BIRD_PG_* env vars ARE forwarded for non-postgres (sqlite) benchmarks."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("BIRD_PG_HOST", "external.db.host")
    monkeypatch.setenv("BIRD_PG_PORT", "5432")

    result = driver.read_api_keys_from_local_env(
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-haiku-4-5-20251001",
        dataset="mini-interact",
    )

    assert result.get("BIRD_PG_HOST") == "external.db.host"
    assert result.get("BIRD_PG_PORT") == "5432"


def test_read_api_keys_pg_vars_forwarded_for_empty_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When dataset is empty (legacy call), BIRD_PG_* are still forwarded."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("BIRD_PG_HOST", "external.db.host")

    result = driver.read_api_keys_from_local_env(
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-haiku-4-5-20251001",
        dataset="",
    )

    assert result.get("BIRD_PG_HOST") == "external.db.host"
