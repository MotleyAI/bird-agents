"""T24–T30: in-cluster actor pool, RayActorError handling, SLayer per-task,
fd-level log capture, HeartbeatWriter, env-var wiring.

Important: the cloud import comes BEFORE `pytest.importorskip("ray")` so a
machine without the cloud package fails-for-the-right-reason (cloud
ModuleNotFoundError) rather than masquerading as a skip from a missing Ray.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from bird_interact_agents.cloud import ray_app  # noqa: E402  (must precede ray-skip)

ray = pytest.importorskip("ray")  # noqa: E402 — local_only=True tests don't strictly need ray but the drain_pool import path does


RUN_ID = "20260521T1422-pydanticai-raw-a1b2c3"


@pytest.fixture(scope="module")
def ray_local():
    """Boot Ray once per module. The spec called for `local_mode=True` but
    Ray dropped it; we use `num_cpus=1` and an isolated tmp dir as the
    fallback the spec anticipated."""
    try:
        ray.init(local_mode=True, ignore_reinit_error=True,
                 include_dashboard=False)
    except (TypeError, RuntimeError):
        ray.init(num_cpus=2, ignore_reinit_error=True,
                 include_dashboard=False,
                 _temp_dir=None,
                 logging_level="ERROR")
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# T24 — ActorPool dispatches each iid exactly once.
# ---------------------------------------------------------------------------


def test_actor_pool_dispatches_each_id_once(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    """Each iid processed exactly once AND K actors are constructed.

    Uses `local_only=True` + an injected actor class so the dispatch logic
    is exercised in-process — Ray remote actors run in subprocesses that
    don't inherit pytest monkeypatches, so the real-Ray dispatch path is
    covered by integration probes (gated, manual-only)."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    # Bypass the cached-runner path (CR#14) so monkeypatching
    # `run_one_task` is honoured. The dedicated runner-cache tests
    # (`test_actor_caches_runner_for_raw_mode`) cover that path.
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    seen: list[str] = []

    async def fake_run_one_task(task_data, **_kwargs):
        seen.append(task_data["instance_id"])
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a",
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    # Record actor constructions.
    constructed: list[tuple] = []

    class RecordingActor(ray_app._LocalActor):
        def __init__(self, cfg, run_id, attempt):
            constructed.append((run_id, attempt))
            super().__init__(cfg, run_id, attempt)

    instance_ids = [f"db_a_{i}" for i in range(7)]
    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=instance_ids,
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=3,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in instance_ids
        },
        local_only=True,
        actor_cls=RecordingActor,
    )

    # Each id processed exactly once.
    assert sorted(seen) == sorted(instance_ids)
    # Exactly num_actors actors constructed.
    assert len(constructed) == 3


# ---------------------------------------------------------------------------
# T25 — one task raising doesn't corrupt others; the failure lands in GCS.
# ---------------------------------------------------------------------------


def test_failing_task_isolated(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    async def fake_run_one_task(task_data, **_kwargs):
        if task_data["instance_id"] == "db_a_2":
            raise RuntimeError("synthetic explosion")
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a",
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    instance_ids = ["db_a_1", "db_a_2", "db_a_3"]
    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=instance_ids,
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=2,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in instance_ids
        },
        local_only=True,
    )

    # All three rows present in GCS; the failed one has error set.
    # Path is runs/<run_id>/rows/<iid>/attempt-1.json → split[-2] is the iid.
    rows = {
        k.split("/")[-2]: json.loads(v)
        for k, v in store.items()
        if k.endswith("attempt-1.json") and "/rows/" in k
    }
    assert set(rows) == set(instance_ids)
    assert rows["db_a_2"]["error"]
    assert rows["db_a_1"]["error"] is None
    assert rows["db_a_3"]["error"] is None


# ---------------------------------------------------------------------------
# T26 — RayActorError in the drain loop becomes an `actor-lost` synthetic row.
# ---------------------------------------------------------------------------


def test_drain_loop_handles_actor_error(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    """When a RayActorError fires inside the drain loop, the iid in flight
    on the dying actor is recovered from drain_pool's own bookkeeping (NOT
    from a synthetic `.iid` attribute on the exception — real RayActorError
    carries no task arg). Remaining work continues."""
    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)

    # FakePool: yields two successes (db_a_1, db_a_3) and one RayActorError
    # for db_a_2. After the error, has_next() still reports True until the
    # remaining successes are drained.
    class FakePool:
        def __init__(self, plan):
            # plan is a list of ("ok", iid) or ("dead", iid) markers
            self._plan = list(plan)
            self.last_failed_iid: str | None = None
            self.submitted: list[str] = []

        def has_next(self) -> bool:
            return bool(self._plan)

        def get_next_unordered(self):
            kind, iid = self._plan.pop(0)
            if kind == "dead":
                self.last_failed_iid = iid
                # Ray's RayActorError signature varies across versions —
                # use the bare init that all versions support.
                raise ray.exceptions.RayActorError()
            return iid

        def submit(self, _fn, iid: str) -> None:
            self.submitted.append(iid)

    pool = FakePool([("ok", "db_a_1"), ("dead", "db_a_2"), ("ok", "db_a_3")])
    ray_app.drain_pool(
        pool=pool,
        run_id=RUN_ID,
        attempt=1,
        gcs_client=client,
    )

    # db_a_2's actor-lost row lands.
    err_blob = next(
        v for k, v in store.items()
        if "db_a_2" in k and k.endswith("attempt-1.json")
    )
    err_row = json.loads(err_blob)
    assert err_row["error"] == "actor-lost"
    assert err_row["instance_id"] == "db_a_2"

    # The drain loop did NOT abort: it consumed both healthy iids from
    # the pool. (Their per-task rows are written by the actor — outside
    # drain_pool's scope; drain_pool only writes synthetic actor-lost rows.)
    assert not pool.has_next(), "drain_pool exited with work remaining"


# ---------------------------------------------------------------------------
# T27 — SLayer mode: per-task ephemeral server (NOT one per actor).
# ---------------------------------------------------------------------------


def test_slayer_server_is_per_task(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, baked_slayer_dbs,
):
    """Each task gets a *unique* per-task SQLite copy AND a fresh server.
    A per-actor server pointed at the shared baked DB would silently pass
    a boot-count assertion; the per-task path uniqueness is what rules it out.
    """
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    baked_slayer_dbs("db_a")

    boots: list[str] = []
    teardowns: list[str] = []
    seen_storage_roots: list[str] = []

    class SlayerStub:
        def __init__(self, sqlite_path: str):
            boots.append(sqlite_path)
            self.sqlite_path = sqlite_path

        def close(self):
            teardowns.append(self.sqlite_path)

    monkeypatch.setattr(ray_app, "EphemeralSlayerServer", SlayerStub)

    async def fake_run_one_task(task_data, slayer_storage_root=None, **_kw):
        # The per-task path must be set AND unique per task.
        assert slayer_storage_root is not None
        seen_storage_roots.append(slayer_storage_root)
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a",
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    instance_ids = ["db_a_1", "db_a_2", "db_a_3"]
    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=instance_ids,
        framework="pydantic_ai",
        query_mode="slayer",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,  # single actor → per-actor would give 1 boot
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in instance_ids
        },
        local_only=True,
    )

    assert len(boots) == len(instance_ids)
    assert len(teardowns) == len(instance_ids)
    # Per-task: each boot must point at a distinct sqlite path.
    assert len(set(boots)) == len(instance_ids), (
        f"slayer boots reused the same sqlite path: {boots}"
    )
    # And the per-task storage root threaded into run_one_task is unique too.
    assert len(set(seen_storage_roots)) == len(instance_ids)


def test_slayer_server_teardown_on_task_exception(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, baked_slayer_dbs
):
    """Even if a task raises, the per-task SLayer server is torn down."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    baked_slayer_dbs("db_a")

    boots: list[str] = []
    teardowns: list[str] = []

    class SlayerStub:
        def __init__(self, sqlite_path: str):
            boots.append(sqlite_path)
            self.sqlite_path = sqlite_path

        def close(self):
            teardowns.append(self.sqlite_path)

    monkeypatch.setattr(ray_app, "EphemeralSlayerServer", SlayerStub)

    async def fake_run_one_task(task_data, **_kwargs):
        raise RuntimeError(f"boom on {task_data['instance_id']}")

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    instance_ids = ["db_a_1", "db_a_2"]
    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=instance_ids,
        framework="pydantic_ai",
        query_mode="slayer",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in instance_ids
        },
        local_only=True,
    )
    assert len(teardowns) == len(boots) == len(instance_ids)


def test_setup_per_task_slayer_fails_loudly_when_db_missing(
    baked_slayer_dbs,
) -> None:
    """A missing baked DB means the image build / ingest is broken. Booting
    a server against an empty stand-in would make every task silently fail
    its queries, so `_setup_per_task_slayer` must raise instead (D1a)."""
    baked_slayer_dbs("db_a")  # only db_a is baked
    with pytest.raises(FileNotFoundError, match="db_missing"):
        ray_app._setup_per_task_slayer("db_missing")


def test_setup_per_task_slayer_copies_baked_db(
    monkeypatch: pytest.MonkeyPatch, baked_slayer_dbs,
) -> None:
    """Happy path: the baked DB is copied into a fresh per-task tmp dir and a
    server is booted against the copy (not the shared baked original)."""
    dbs_dir = baked_slayer_dbs("db_a")

    booted: list[str] = []

    class _Stub:
        def __init__(self, sqlite_path: str):
            booted.append(sqlite_path)

        def close(self):
            pass

    monkeypatch.setattr(ray_app, "EphemeralSlayerServer", _Stub)

    server, storage_root = ray_app._setup_per_task_slayer("db_a")
    try:
        copied = Path(booted[0])
        assert copied.exists()
        # Copied to a per-task tmp dir, not served from the shared baked dir.
        assert copied.parent != dbs_dir
        assert copied.parent == Path(storage_root)
        assert copied.read_bytes() == (dbs_dir / "db_a.sqlite").read_bytes()
    finally:
        server.close()


def test_local_actor_uses_injected_gcs_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_LocalActor` accepts an injected client so in-process tests can pass
    a fake directly (no pickling locally), while the real Ray `WorkerActor`
    keeps self-building. Injection must win over `default_gcs_client`."""
    sentinel = object()

    def _boom():
        raise AssertionError("default_gcs_client must not be called when a "
                             "client is injected")

    monkeypatch.setattr(ray_app, "default_gcs_client", _boom)

    cfg = {"query_mode": "raw", "mode": "oracle"}  # no cached runner built
    actor = ray_app._LocalActor(cfg, RUN_ID, 1, gcs_client=sentinel)
    assert actor.gcs_client is sentinel


def test_local_actor_self_builds_client_without_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no injected client, `_LocalActor` falls back to
    `default_gcs_client` (the production path)."""
    sentinel = object()
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: sentinel)

    cfg = {"query_mode": "raw", "mode": "oracle"}
    actor = ray_app._LocalActor(cfg, RUN_ID, 1)
    assert actor.gcs_client is sentinel


def test_run_pool_threads_injected_client_into_local_actors(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
) -> None:
    """CR thread #3 — in local mode, `run_pool(gcs_client=...)` must reach
    the actors' row/log writes, not just the heartbeat. Otherwise the actor
    self-builds a `default_gcs_client()` and a run's artifacts split across
    backends. We make injected != default and assert the actors end up on
    the injected one."""
    injected, _store = fake_gcs_bucket
    wrong = object()
    # default_gcs_client returns the WRONG client; only the injected one
    # (passed to run_pool) should win for the actors.
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: wrong)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    seen_clients: list[object] = []

    class _CaptureActor(ray_app._LocalActor):
        def run_one(self, task_data: dict) -> str:
            seen_clients.append(self.gcs_client)
            return "ok"

    async def fake_run_one_task(task_data, **_kwargs):
        return {"instance_id": task_data["instance_id"], "database": "db_a",
                "phase1_passed": True, "phase2_passed": True,
                "total_reward": 1.0, "duration_s": 0.01, "error": None}

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1", "db_a_2"],
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=2,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2")
        },
        local_only=True,
        gcs_client=injected,
        actor_cls=_CaptureActor,
    )

    assert seen_clients, "no actor ran"
    assert all(c is injected for c in seen_clients), (
        f"actors used a client other than the injected one: {seen_clients}"
    )
    assert wrong not in seen_clients


def test_with_actor_env_applies_per_actor_runtime_env() -> None:
    """Secrets reach worker actors via a PER-ACTOR runtime_env (not the job
    runtime_env, which `ray job list` echoes)."""
    from unittest.mock import MagicMock

    cls = MagicMock()
    out = ray_app._with_actor_env(cls, {"ANTHROPIC_API_KEY": "sk"})
    cls.options.assert_called_once_with(
        runtime_env={"env_vars": {"ANTHROPIC_API_KEY": "sk"}}
    )
    assert out is cls.options.return_value


def test_with_actor_env_is_noop_without_env() -> None:
    """No env vars → return the class untouched (so a custom actor_cls
    without `.options()` isn't broken)."""
    from unittest.mock import MagicMock

    cls = MagicMock()
    assert ray_app._with_actor_env(cls, None) is cls
    assert ray_app._with_actor_env(cls, {}) is cls
    cls.options.assert_not_called()


def test_load_secrets_file_loads_and_deletes(tmp_path: Path) -> None:
    """The secrets file is read then deleted (minimise secret-at-rest), and
    values are coerced to str."""
    f = tmp_path / "secrets.json"
    f.write_text(json.dumps({"ANTHROPIC_API_KEY": "sk", "N": 1}))
    out = ray_app._load_secrets_file(str(f))
    assert out == {"ANTHROPIC_API_KEY": "sk", "N": "1"}
    assert not f.exists(), "secrets file must be deleted after loading"


def test_load_secrets_file_none_returns_none() -> None:
    assert ray_app._load_secrets_file(None) is None


def test_load_secrets_file_deletes_even_on_bad_json(tmp_path: Path) -> None:
    """A garbage file still gets deleted (it holds secrets) and the error
    surfaces rather than silently yielding an empty env."""
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        ray_app._load_secrets_file(str(f))
    assert not f.exists()


def test_run_pool_local_applies_actor_env_to_os_environ(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
) -> None:
    """In local mode the in-process actors share os.environ, so run_pool must
    apply `actor_env_vars` there for the agent to see the API keys."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    # setenv so monkeypatch restores/removes it at teardown even though
    # run_pool overwrites it via os.environ.update (no env pollution).
    monkeypatch.setenv("BIRD_TEST_SECRET", "placeholder")

    seen: dict[str, str | None] = {}

    async def fake_run_one_task(task_data, **_kwargs):
        seen["val"] = os.environ.get("BIRD_TEST_SECRET")
        return {"instance_id": task_data["instance_id"], "database": "db_a",
                "phase1_passed": True, "phase2_passed": True,
                "total_reward": 1.0, "duration_s": 0.01, "error": None}

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1"],
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={"db_a_1": {"instance_id": "db_a_1",
                                    "selected_database": "db_a"}},
        local_only=True,
        actor_env_vars={"BIRD_TEST_SECRET": "hunter2"},
    )

    assert seen.get("val") == "hunter2"


def test_run_pool_local_default_actor_skips_default_gcs_client(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
) -> None:
    """Codex #3: with an injected client, the default `_LocalActor` is built
    WITH it and must never call `default_gcs_client()` (which can fail with no
    creds). We make `default_gcs_client` raise; reaching the end proves it was
    never called — at run_pool top (client injected) nor in actor init."""
    injected, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    def _boom():
        raise AssertionError("default_gcs_client must not be called")

    monkeypatch.setattr(ray_app, "default_gcs_client", _boom)

    async def fake_run_one_task(task_data, **_kwargs):
        return {"instance_id": task_data["instance_id"], "database": "db_a",
                "phase1_passed": True, "phase2_passed": True,
                "total_reward": 1.0, "duration_s": 0.01, "error": None}

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1"],
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={"db_a_1": {"instance_id": "db_a_1",
                                    "selected_database": "db_a"}},
        local_only=True,
        gcs_client=injected,  # no actor_cls → default _LocalActor path
    )


# ---------------------------------------------------------------------------
# T28 — fd-level log capture catches subprocess output too.
# ---------------------------------------------------------------------------


def test_fd_capture_catches_subprocess_output(tmp_path: Path) -> None:
    """The critical thing fd_capture must do that `redirect_stdout` doesn't:
    catch subprocess output (which inherits fd 1/2 from the parent).
    Direct-fd writes are also caught. pytest's `sys.stdout` swap interferes
    with `print()`, so we avoid the Python stream and write directly via fd."""
    log_path = tmp_path / "task.log"
    with ray_app.fd_capture(log_path):
        # Direct fd write — proves the fd was redirected.
        os.write(1, b"from-direct-fd\n")
        os.write(2, b"from-direct-fd-stderr\n")
        # Subprocess inherits the redirected fds.
        import subprocess
        subprocess.run(
            ["sh", "-c", "echo from-subprocess; echo from-subprocess-stderr 1>&2"],
            check=True,
        )

    text = log_path.read_text()
    assert "from-direct-fd" in text
    assert "from-direct-fd-stderr" in text
    assert "from-subprocess" in text
    assert "from-subprocess-stderr" in text


# ---------------------------------------------------------------------------
# T29 — HeartbeatWriter writes status.json on schedule + terminal_state.
# ---------------------------------------------------------------------------


def test_heartbeat_writer_writes_status(fake_gcs_bucket):
    client, store = fake_gcs_bucket
    hb = ray_app.HeartbeatWriter(
        run_id=RUN_ID, total=5, attempt=1, ray_job_id="raysubmit_abc",
        client=client, interval_s=0.01,
    )
    hb.start()
    try:
        hb.tick_done()
        hb.tick_done()
        time.sleep(0.05)  # let at least one heartbeat fire
    finally:
        hb.stop_and_flush(terminal_state="done")

    status_path = f"runs/{RUN_ID}/status.json"
    assert status_path in store, "status.json was never written"
    status = json.loads(store[status_path])
    assert status["terminal_state"] == "done"
    assert status["rows_total"] == 5
    assert status["rows_done"] == 2
    assert status["ray_job_id"] == "raysubmit_abc"
    assert "last_heartbeat_ts" in status


def test_heartbeat_marks_error_state_on_raise(fake_gcs_bucket):
    client, store = fake_gcs_bucket
    hb = ray_app.HeartbeatWriter(
        run_id=RUN_ID, total=3, attempt=1, ray_job_id="raysubmit_xyz",
        client=client, interval_s=0.01,
    )
    hb.start()
    try:
        raise RuntimeError("simulated cluster-side blow-up")
    except RuntimeError:
        hb.stop_and_flush(terminal_state="error")

    status = json.loads(store[f"runs/{RUN_ID}/status.json"])
    assert status["terminal_state"] == "error"


# ---------------------------------------------------------------------------
# Cx3 — `_load_task_data(use_audited_gold_sql=True)` applies the audited
# overlay, so cloud actors evaluate against audited gold SQL.
# ---------------------------------------------------------------------------


def test_load_task_data_applies_audited_gold_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Without this wiring the cloud manifest claims `use_audited_gold_sql=true`
    but actors read raw gold SQL — silent contract violation."""
    from bird_interact_agents import paths as _paths

    # Stub the dataset file with one row.
    mi = tmp_path / "mi"
    mi.mkdir()
    dataset_file = mi / "mini_interact.jsonl"
    dataset_file.write_text(
        json.dumps({
            "instance_id": "db_a_1",
            "selected_database": "db_a",
            "sol_sql": "SELECT raw",
        }) + "\n"
    )
    monkeypatch.setattr(_paths, "mini_interact_data_file", lambda: dataset_file)

    overlay_calls: list[tuple] = []

    def fake_overlay(rows, audited_root):
        overlay_calls.append((list(rows), audited_root))
        # Mutate the row's sol_sql to mimic the real overlay.
        for r in rows:
            if r["instance_id"] == "db_a_1":
                r["sol_sql"] = "SELECT audited"

    monkeypatch.setattr(
        "bird_interact_agents.harness.apply_audited_gold_overlay",
        fake_overlay,
    )

    # Off → overlay not called.
    out = ray_app._load_task_data(["db_a_1"], use_audited_gold_sql=False)
    assert out["db_a_1"]["sol_sql"] == "SELECT raw"
    assert overlay_calls == []

    # On → overlay called and applied.
    out = ray_app._load_task_data(["db_a_1"], use_audited_gold_sql=True)
    assert overlay_calls, "overlay was not applied"
    assert out["db_a_1"]["sol_sql"] == "SELECT audited"


# ---------------------------------------------------------------------------
# CR#14 — actor caches the framework runner for raw mode (1 build, N tasks).
# ---------------------------------------------------------------------------


def test_undispatched_pending_iids_get_error_rows(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    """If every actor dies AND `actor_factory()` keeps failing, pending
    iids would never get dispatched. Without the post-loop drain they'd
    be silently missing from GCS and the run would still mark itself
    `done`. The drain writes synthetic `undispatched` error rows so the
    spec's record-and-move-on contract holds."""
    client, store = fake_gcs_bucket

    class _DeadActor:
        class _Method:
            def remote(self, task_data):
                raise RuntimeError("dead handle")

        def __getattr__(self, name):
            return _DeadActor._Method()

    def factory_that_also_fails():
        raise RuntimeError("provisioning quota exhausted")

    hb = ray_app.HeartbeatWriter(
        run_id=RUN_ID, total=3, attempt=1, ray_job_id="x",
        client=client, interval_s=99,
    )
    hb.start()
    try:
        ray_app._run_with_actors(
            actors=[_DeadActor()],
            instance_ids=["db_a_1", "db_a_2", "db_a_3"],
            task_data_by_id={
                iid: {"instance_id": iid, "selected_database": "db_a"}
                for iid in ("db_a_1", "db_a_2", "db_a_3")
            },
            run_id=RUN_ID,
            attempt=1,
            gcs_client=client,
            heartbeat=hb,
            actor_factory=factory_that_also_fails,
        )
    finally:
        hb.stop_and_flush(terminal_state="done")

    # Every iid landed as an error row — none silently dropped.
    import json as _json
    iids_with_rows = set()
    for k, v in store.items():
        if "/rows/" in k and k.endswith("attempt-1.json"):
            row = _json.loads(v)
            iids_with_rows.add(row["instance_id"])
    assert iids_with_rows == {"db_a_1", "db_a_2", "db_a_3"}


def test_dispatch_failure_mints_replacement_actor(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    """On dispatch failure, `_run_with_actors` must NOT recycle the same
    failing actor back into the free pool — otherwise one broken handle
    fails every remaining iid. It should mint a replacement via
    actor_factory and continue with that."""
    client, store = fake_gcs_bucket

    class _DeadActor:
        class _Method:
            def remote(self, task_data):
                raise RuntimeError("dead actor handle")

        def __getattr__(self, name):
            return _DeadActor._Method()

    class _HealthyActor:
        class _Method:
            def remote(self, task_data):
                return _FakeFuture(task_data["instance_id"])

        def __getattr__(self, name):
            return _HealthyActor._Method()

    class _FakeFuture:
        def __init__(self, iid: str):
            self.iid = iid

    import ray as _real_ray

    def fake_wait(futures, num_returns=1):
        return [futures[0]], futures[1:]

    def fake_get(future):
        return future.iid

    monkeypatch.setattr(_real_ray, "wait", fake_wait)
    monkeypatch.setattr(_real_ray, "get", fake_get)

    factory_calls = [0]

    def factory():
        factory_calls[0] += 1
        return _HealthyActor()

    hb = ray_app.HeartbeatWriter(
        run_id=RUN_ID, total=3, attempt=1, ray_job_id="x",
        client=client, interval_s=99,
    )
    hb.start()
    try:
        ray_app._run_with_actors(
            actors=[_DeadActor()],  # dead from the start
            instance_ids=["db_a_1", "db_a_2", "db_a_3"],
            task_data_by_id={
                iid: {"instance_id": iid, "selected_database": "db_a"}
                for iid in ("db_a_1", "db_a_2", "db_a_3")
            },
            run_id=RUN_ID,
            attempt=1,
            gcs_client=client,
            heartbeat=hb,
            actor_factory=factory,
        )
    finally:
        hb.stop_and_flush(terminal_state="done")

    # First iid failed (dead actor), the rest succeeded on replacement.
    assert factory_calls[0] >= 1, "no replacement actor minted"
    # db_a_1's error row was written, db_a_2 + db_a_3 did NOT all land
    # as dispatch-failures.
    import json as _json
    errors = []
    for k, v in store.items():
        if "/rows/" in k and k.endswith("attempt-1.json"):
            row = _json.loads(v)
            if row.get("error") and "dispatch-failure" in row["error"]:
                errors.append(row["instance_id"])
    # At most one iid should be a dispatch-failure (the one that hit the
    # dead actor); the others should be healthy.
    assert len(errors) <= 1, (
        f"replacement actor wasn't minted in time; dispatch-failures: {errors}"
    )


def test_actor_caches_runner_for_raw_mode(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)

    build_count = [0]
    runner_call_count = [0]

    async def fake_runner(td, data_dir, patience, user_sim_model):
        runner_call_count[0] += 1
        return {
            "instance_id": td["instance_id"],
            "database": td.get("selected_database", ""),
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    def fake_make_runner(**_kwargs):
        build_count[0] += 1
        return fake_runner

    monkeypatch.setattr(
        "bird_interact_agents.run.make_runner", fake_make_runner
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1", "db_a_2", "db_a_3"],
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2", "db_a_3")
        },
        local_only=True,
    )

    # ONE build (cached), THREE runner invocations.
    assert build_count[0] == 1
    assert runner_call_count[0] == 3


def test_actor_no_runner_cache_for_slayer_mode(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, baked_slayer_dbs
):
    """Slayer mode rotates `slayer_storage_root` per task — caching the
    runner would freeze it to one task's storage. The actor must skip
    the cache."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    baked_slayer_dbs("db_a")

    build_count = [0]

    def fake_make_runner(**_kwargs):
        build_count[0] += 1

        async def _runner(td, data_dir, patience, user_sim_model):
            return {
                "instance_id": td["instance_id"],
                "database": td.get("selected_database", ""),
                "phase1_passed": True,
                "phase2_passed": True,
                "total_reward": 1.0,
                "duration_s": 0.01,
                "error": None,
            }
        return _runner

    monkeypatch.setattr(
        "bird_interact_agents.run.make_runner", fake_make_runner
    )

    class _NoopSlayer:
        def __init__(self, sqlite_path: str):
            pass

        def close(self):
            pass

    monkeypatch.setattr(ray_app, "EphemeralSlayerServer", _NoopSlayer)

    async def fake_run_one_task(task_data, **_kwargs):
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a",
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1", "db_a_2"],
        framework="pydantic_ai",
        query_mode="slayer",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2")
        },
        local_only=True,
    )

    # No runner cached for slayer mode — `make_runner` not invoked at all.
    assert build_count[0] == 0


# ---------------------------------------------------------------------------
# T30 — env-var wiring: the actor (not just paths.py directly) honours
#       BIRD_DB_PATH / BIRD_DATA_PATH / BIRD_RESULTS_ROOT.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CR#11 — per-task tmp dirs are cleaned up.
# ---------------------------------------------------------------------------


def test_per_task_tmp_dirs_are_cleaned(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
    baked_slayer_dbs,
):
    """After a task returns, the per-task SLayer SQLite tmp dir and log
    tmp dir must be removed — without cleanup they accumulate and fill
    worker disks on big runs (CR#11)."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    baked_slayer_dbs("db_a")

    boots: list[str] = []
    teardowns: list[str] = []

    class SlayerStub:
        def __init__(self, sqlite_path: str):
            boots.append(sqlite_path)
            self.sqlite_path = sqlite_path

        def close(self):
            teardowns.append(self.sqlite_path)

    monkeypatch.setattr(ray_app, "EphemeralSlayerServer", SlayerStub)

    # Force tmp dirs into a single tmp root so we can inspect what's left.
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    async def fake_run_one_task(task_data, **_kwargs):
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a",
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1", "db_a_2"],
        framework="pydantic_ai",
        query_mode="slayer",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2")
        },
        local_only=True,
    )

    # No leftover per-task tmp dirs under $TMPDIR.
    leftover = [
        d for d in tmp_path.iterdir()
        if d.is_dir() and (
            d.name.startswith("slayer_") or d.name.startswith("cloud_log_")
        )
    ]
    assert not leftover, f"per-task tmp dirs leaked: {leftover}"


# ---------------------------------------------------------------------------
# CR#12 / Codex #4 — actor death in the real Ray path produces an
# `actor-lost` row for the exact iid that was on the dying actor.
# ---------------------------------------------------------------------------


def test_run_with_actors_writes_actor_lost_row_for_dying_actor(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    """The replacement for `drain_pool` — `_run_with_actors` — tracks
    `future → (actor, iid)` so when `ray.get(future)` raises
    `RayActorError` we know which iid was on the dying actor."""
    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)

    # Fake ray module-level surface: ray.wait + ray.get + RayActorError.
    import ray as _real_ray
    real_ray_get = _real_ray.get
    real_ray_wait = _real_ray.wait

    class _FakeFuture:
        def __init__(self, iid: str, will_fail: bool):
            self.iid = iid
            self.will_fail = will_fail

    class _FakeActor:
        def __init__(self):
            self._name = "actor"

        class _Method:
            def __init__(self, parent, will_fail_set):
                self.parent = parent
                self.will_fail_set = will_fail_set

            def remote(self, task_data):
                iid = task_data["instance_id"]
                return _FakeFuture(iid, iid in self.will_fail_set)

        def __getattr__(self, name):
            return _FakeActor._Method(self, {"db_a_2"})  # db_a_2 always dies

    actors = [_FakeActor()]
    dispatched_iids: list[str] = []

    def fake_wait(futures, num_returns=1):
        return [futures[0]], futures[1:]

    def fake_get(future):
        dispatched_iids.append(future.iid)
        if future.will_fail:
            raise _real_ray.exceptions.RayActorError()
        return future.iid

    monkeypatch.setattr(_real_ray, "wait", fake_wait)
    monkeypatch.setattr(_real_ray, "get", fake_get)

    hb = ray_app.HeartbeatWriter(
        run_id=RUN_ID, total=3, attempt=1, ray_job_id="x",
        client=client, interval_s=99,
    )
    hb.start()
    try:
        ray_app._run_with_actors(
            actors=list(actors),
            instance_ids=["db_a_1", "db_a_2", "db_a_3"],
            task_data_by_id={
                iid: {"instance_id": iid, "selected_database": "db_a"}
                for iid in ("db_a_1", "db_a_2", "db_a_3")
            },
            run_id=RUN_ID,
            attempt=1,
            gcs_client=client,
            heartbeat=hb,
            actor_factory=lambda: _FakeActor(),
        )
    finally:
        hb.stop_and_flush(terminal_state="done")
        monkeypatch.setattr(_real_ray, "wait", real_ray_wait)
        monkeypatch.setattr(_real_ray, "get", real_ray_get)

    # db_a_2 lands as an actor-lost row.
    iid_to_row_path = {
        k.split("/")[-2]: k
        for k in store
        if k.endswith("attempt-1.json") and "/rows/" in k
    }
    assert "db_a_2" in iid_to_row_path
    import json as _json
    err_row = _json.loads(store[iid_to_row_path["db_a_2"]])
    assert err_row["error"] == "actor-lost"
    assert err_row["instance_id"] == "db_a_2"


# ---------------------------------------------------------------------------
# CR#13 — dispatch failures don't silently disappear.
# ---------------------------------------------------------------------------


def test_dispatch_failure_writes_error_row(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    client, store = fake_gcs_bucket

    class _BoomActor:
        class _Method:
            def remote(self, task_data):
                raise RuntimeError("boom on dispatch")

        def __getattr__(self, name):
            return _BoomActor._Method()

    import ray as _real_ray

    def fake_wait(futures, num_returns=1):
        return [], futures

    monkeypatch.setattr(_real_ray, "wait", fake_wait)

    hb = ray_app.HeartbeatWriter(
        run_id=RUN_ID, total=1, attempt=1, ray_job_id="x",
        client=client, interval_s=99,
    )
    hb.start()
    try:
        ray_app._run_with_actors(
            actors=[_BoomActor()],
            instance_ids=["db_a_1"],
            task_data_by_id={
                "db_a_1": {"instance_id": "db_a_1", "selected_database": "db_a"},
            },
            run_id=RUN_ID,
            attempt=1,
            gcs_client=client,
            heartbeat=hb,
            actor_factory=lambda: _BoomActor(),
        )
    finally:
        hb.stop_and_flush(terminal_state="done")

    # The dispatch failure landed as an `error` row, not silently dropped.
    iid_path = next(
        k for k in store
        if "db_a_1" in k and k.endswith("attempt-1.json") and "/rows/" in k
    )
    import json as _json
    row = _json.loads(store[iid_path])
    assert row["error"]
    assert "dispatch-failure" in row["error"]


def test_actor_uses_env_resolved_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """Route a task through run_pool with env patched, and verify the
    actor passes the env-resolved paths down to `run_one_task`. This
    catches the bug where the actor would compute paths at import time
    and ignore later env changes."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    mi = tmp_path / "mi"
    mi.mkdir()
    (mi / "mini_interact.jsonl").write_text("")
    results = tmp_path / "results"
    monkeypatch.setenv("BIRD_DB_PATH", str(mi))
    monkeypatch.setenv("BIRD_DATA_PATH", str(mi / "mini_interact.jsonl"))
    monkeypatch.setenv("BIRD_RESULTS_ROOT", str(results))

    seen: list[tuple[str, str | None]] = []

    async def fake_run_one_task(task_data, data_dir=None, **_kwargs):
        seen.append((task_data["instance_id"], data_dir))
        return {
            "instance_id": task_data["instance_id"],
            "database": "db_a",
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "duration_s": 0.01,
            "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1"],
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            "db_a_1": {"instance_id": "db_a_1", "selected_database": "db_a"}
        },
        local_only=True,
    )

    assert seen, "actor never invoked run_one_task"
    _iid, data_dir = seen[0]
    # The actor passes the env-resolved BIRD_DB_PATH as `data_dir`.
    assert data_dir == str(mi)
