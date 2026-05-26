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


def _patch_collaborators(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace driver's collaborators with mocks and return them."""
    mocks: dict[str, MagicMock] = {}
    for attr in (
        "prereqs",
        "image",
        "cluster",
        "gcs",
    ):
        m = MagicMock(name=attr)
        mocks[attr] = m
        monkeypatch.setattr(f"bird_interact_agents.cloud.driver.{attr}", m)
    mocks["image"].image_tag.return_value = "deadbeef1234-cafebabe5678"
    mocks["image"].build_and_push.return_value = (
        "us-central1-docker.pkg.dev/motley-team-475011/x/runner:tag"
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

    monkeypatch.setattr(driver.paths, "mini_interact_data_file", lambda: data_file)
    monkeypatch.setattr(driver.paths, "mini_interact_root", lambda: tmp_path / "mini")
    monkeypatch.setattr(driver, "submitter_repo_root", lambda: worktree)
    monkeypatch.setattr(driver.paths, "slayer_otf_cache_root", lambda: otf_cache)
    monkeypatch.setattr(driver.paths, "slayer_models_otf_root", lambda: otf_ref)
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
                d = otf_ref / db
                d.mkdir(parents=True)
                (d / "_reference_fp.txt").write_text("fp")
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
    assert uploaded == expected, f"uploaded {uploaded}, expected {expected}"


def test_submit_missing_local_artifact_raises_before_cluster(
    monkeypatch, tmp_path,
):
    """Fail-fast: an on-the-fly DB with no local artifact must raise at submit
    BEFORE building/pushing the image or bringing up a cluster."""
    mocks, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_recursive",
        slayer_setup="on-the-fly", mode="a-interact", dbs=["db_a"],
        lay_down=False,  # artifact intentionally absent
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_recursive", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    with pytest.raises(FileNotFoundError):
        driver.submit(args)
    mocks["image"].build_and_push.assert_not_called()
    mocks["cluster"].up.assert_not_called()
    mocks["gcs"].upload_dir_prefix.assert_not_called()


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


def test_submit_otf_encode_missing_reference_raises(monkeypatch, tmp_path):
    """otf_encode presence = slayer_models_otf/<db>/_reference_fp.txt marker;
    a missing reference must fail fast before the cluster."""
    mocks, *_ = _setup_slayer_submit(
        monkeypatch, tmp_path, framework="pydantic_ai_otf_encode",
        slayer_setup="on-the-fly", mode="a-interact", dbs=["db_a"],
        lay_down=False,
    )
    args = FakeSubmitArgs(
        framework="pydantic_ai_otf_encode", query_mode="slayer", mode="a-interact",
        slayer_setup="on-the-fly", instance_ids=("db_a_1",), detach=True,
    )
    with pytest.raises(FileNotFoundError):
        driver.submit(args)
    mocks["cluster"].up.assert_not_called()


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
