"""T13–T19: driver orchestration, SIGINT, stall detection, kill recovery."""

from __future__ import annotations

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
    instance_ids: tuple[str, ...] = ("db_a_1", "db_a_2", "db_a_3")
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
    dataset: str = "mini_interact"
    gold_file: str | None = None


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
            benchmark="mini_interact",
        )
        for kw in seen_kwargs
    )
    assert all(
        kw["mini_interact_root"] == driver.paths.mini_interact_root()
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


def test_read_api_keys_claude_sdk_no_oauth_legacy_path(monkeypatch):
    """claude_sdk + no OAuth token → legacy path; ANTHROPIC_API_KEY shipped."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    keys = driver.read_api_keys_from_local_env(
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
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
    with pytest.raises(driver.PrereqError, match="BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"):
        driver.read_api_keys_from_local_env(
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )


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
    monkeypatch.setattr(driver, "local_results_root", lambda: fake_results)
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
    monkeypatch.setattr(driver, "local_results_root", lambda: fake_results)
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
# Benchmark plumbing: dataset + gold_file flow through the manifest and the
# actor job args; the actor + instance→db read the benchmark's tasks file.
# ---------------------------------------------------------------------------


def test_manifest_and_job_args_carry_benchmark(monkeypatch):
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer",
        mode="one-shot", slayer_setup="on-the-fly",
    )
    # FakeSubmitArgs predates --dataset/--gold-file; set them as the cli would.
    args.dataset = "livesqlbench"
    args.gold_file = "/abs/gold.jsonl"

    prefix = "benchmark-data/livesqlbench/abc123/"
    m = driver.build_manifest(
        args, image_uri="img:tag", run_id="rid", benchmark_data_prefix=prefix,
    )
    assert m["dataset"] == "livesqlbench"
    # De-bake: the gold rides along in the GCS dataset upload, so the manifest
    # stores the IN-CLUSTER path (under the benchmark's container_data_dir),
    # NOT the submitter's local path. `/abs/gold.jsonl` isn't under the data
    # root → basename fallback under /data/livesqlbench.
    assert m["gold_file"] == "/data/livesqlbench/gold.jsonl"
    assert m["benchmark_data_prefix"] == prefix

    # Avoid reading a real tasks file for the db-grouped sort.
    monkeypatch.setattr(
        driver, "_instance_ids_sorted_by_db",
        lambda ids, benchmark="mini_interact": list(ids),
    )
    ja = driver._build_job_args(
        args, "rid", attempt=1, benchmark_data_prefix=prefix,
    )
    assert ja[ja.index("--dataset") + 1] == "livesqlbench"
    assert ja[ja.index("--gold-file") + 1] == "/data/livesqlbench/gold.jsonl"
    assert ja[ja.index("--benchmark-data-prefix") + 1] == prefix


def test_manifest_defaults_to_mini_interact_benchmark():
    args = FakeSubmitArgs()  # default dataset → mini_interact, no gold
    m = driver.build_manifest(args, image_uri="img:tag", run_id="rid")
    assert m["dataset"] == "mini_interact"
    assert m["gold_file"] is None
    # No prefix passed → key present but None (back-compat for direct callers).
    assert m["benchmark_data_prefix"] is None


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
    assert mocks["benchmark_data"].ensure_uploaded.call_args.args[0] == "mini_interact"
    # Manifest carries the prefix.
    manifest = mocks["gcs"].write_manifest.call_args.args[1]
    assert manifest["benchmark_data_prefix"] == "benchmark-data/mini_interact/feedface/"
    # Actor job args carry the prefix.
    job_args = mocks["cluster"].submit_job.call_args.kwargs["args"]
    assert job_args[job_args.index("--benchmark-data-prefix") + 1] == (
        "benchmark-data/mini_interact/feedface/"
    )


def test_validate_gold_under_data_root_rejects_outside(monkeypatch, tmp_path):
    """A `--gold-file` outside the benchmark data root fails fast at submit —
    it would otherwise be silently absent in-cluster (the gold rides along in
    the GCS dataset upload, which only covers the data root)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(driver.paths, "benchmark_data_root", lambda *a, **k: data_root)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n")

    args = FakeSubmitArgs()
    args.dataset = "livesqlbench"
    args.gold_file = str(outside)
    with pytest.raises(ValueError, match="must live under the benchmark data root"):
        driver._validate_gold_under_data_root(args)


def test_validate_gold_under_data_root_accepts_inside(monkeypatch, tmp_path):
    """A `--gold-file` inside the data root passes the guard, and
    `_in_cluster_gold_file` maps it to its container path preserving the
    relative location."""
    data_root = tmp_path / "data"
    (data_root / "sub").mkdir(parents=True)
    gold = data_root / "sub" / "gold.jsonl"
    gold.write_text("{}\n")
    monkeypatch.setattr(driver.paths, "benchmark_data_root", lambda *a, **k: data_root)

    args = FakeSubmitArgs()
    args.dataset = "livesqlbench"
    args.gold_file = str(gold)
    driver._validate_gold_under_data_root(args)  # no raise
    assert driver._in_cluster_gold_file(args) == "/data/livesqlbench/sub/gold.jsonl"


def test_resubmit_threads_benchmark_prefix(monkeypatch):
    """`_build_resubmit_args` re-threads the manifest's benchmark_data_prefix so
    the actor re-downloads the dataset; absent on pre-de-bake manifests."""
    monkeypatch.setattr(
        driver, "_instance_ids_sorted_by_db",
        lambda ids, benchmark="mini_interact": list(ids),
    )
    manifest = {
        "framework": "pydantic_ai", "query_mode": "raw", "mode": "c-interact",
        "dataset": "mini_interact", "agent_model": "m", "user_sim_model": "u",
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
    assert driver._instance_ids_sorted_by_db(ids, "mini_interact") == ids


def test_resubmit_omits_dataset_for_pre_dataset_manifest(monkeypatch):
    """A manifest with NO 'dataset' key was written before --dataset existed,
    so its pinned image's ray_app rejects --dataset. Resubmit must OMIT both
    --dataset and --benchmark-data-prefix and let the old baked image run
    (Codex)."""
    monkeypatch.setattr(
        driver, "_instance_ids_sorted_by_db",
        lambda ids, benchmark="mini_interact": list(ids),
    )
    manifest = {
        "framework": "pydantic_ai", "query_mode": "raw", "mode": "c-interact",
        "agent_model": "m", "user_sim_model": "u",
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }  # neither 'dataset' nor 'benchmark_data_prefix'
    ja = driver._build_resubmit_args(manifest, "rid", ["db_a_1"], 2)
    assert "--dataset" not in ja
    assert "--benchmark-data-prefix" not in ja
