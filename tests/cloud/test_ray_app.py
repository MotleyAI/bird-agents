"""T24–T30: in-cluster actor pool, RayActorError handling, SLayer per-task,
fd-level log capture, HeartbeatWriter, env-var wiring.

Important: the cloud import comes BEFORE `pytest.importorskip("ray")` so a
machine without the cloud package fails-for-the-right-reason (cloud
ModuleNotFoundError) rather than masquerading as a skip from a missing Ray.
"""

from __future__ import annotations

import json
import os
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
# T27 (DEV-1468) — SLayer mode mirrors local: no ephemeral server, no baked
# DBs. The actor downloads the uploaded setup once per process; run_one_task
# resolves per-task storage exactly as it does locally.
# ---------------------------------------------------------------------------


def test_ephemeral_slayer_symbols_removed() -> None:
    """The stubbed per-task SLayer server model is deleted — the cloud path
    now mirrors local (per-task storage via run_one_task)."""
    for sym in (
        "EphemeralSlayerServer", "_setup_per_task_slayer", "SLAYER_DBS_DIR",
    ):
        assert not hasattr(ray_app, sym), f"{sym} must be removed (mirror local)"


@pytest.mark.parametrize(
    "framework, slayer_setup, mode",
    [
        ("pydantic_ai_recursive", "pre-encoded", "c-interact"),   # combo 1
        ("pydantic_ai_recursive", "on-the-fly", "a-interact"),    # combo 2
        ("pydantic_ai_otf_encode", "on-the-fly", "a-interact"),   # combo 3
    ],
)
def test_slayer_combo_threads_kwargs_into_run_one_task(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket,
    framework: str, slayer_setup: str, mode: str,
):
    """All 3 slayer combos thread query_mode='slayer', slayer_setup, and
    slayer_storage_root into run_one_task — the cloud path is just local
    run_one_task with the right kwargs (slayer_setup was the missing one)."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    # The actor downloads setup in __init__; stub it (covered separately).
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    seen: list[dict] = []

    async def fake_run_one_task(task_data, **kw):
        seen.append(kw)
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0,
            "duration_s": 0.01, "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1"],
        framework=framework,
        query_mode="slayer",
        mode=mode,
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            "db_a_1": {"instance_id": "db_a_1", "selected_database": "db_a"}
        },
        slayer_setup=slayer_setup,
        slayer_storage_root="/data/slayer_models",
        local_only=True,
    )

    assert len(seen) == 1
    kw = seen[0]
    assert kw["query_mode"] == "slayer"
    assert kw["slayer_setup"] == slayer_setup
    assert kw["slayer_storage_root"] == "/data/slayer_models"


def test_actor_downloads_slayer_setup_once_per_process(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket,
):
    """The actor must call download_slayer_setup in __init__ for slayer mode
    (once per worker process), not per task."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)

    calls: list[tuple] = []

    def fake_download(run_id, cfg, *, client):  # noqa: ARG001
        calls.append((run_id, cfg.get("query_mode")))

    monkeypatch.setattr(ray_app, "download_slayer_setup", fake_download)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0,
            "duration_s": 0.01, "error": None,
        }

    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["db_a_1", "db_a_2", "db_a_3"],
        framework="pydantic_ai_recursive",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,  # one process → exactly one download
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2", "db_a_3")
        },
        slayer_setup="on-the-fly",
        slayer_storage_root="/data/slayer_models",
        local_only=True,
    )

    assert calls == [(RUN_ID, "slayer")], (
        f"download must run once per process for slayer; got {calls}"
    )


def test_actor_does_not_download_in_raw_mode(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket,
):
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    calls: list = []
    monkeypatch.setattr(
        ray_app, "download_slayer_setup",
        lambda *a, **k: calls.append(1),
    )

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0,
            "duration_s": 0.01, "error": None,
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
        task_data_by_id={"db_a_1": {"instance_id": "db_a_1", "selected_database": "db_a"}},
        local_only=True,
    )
    # The actor gates the download call on query_mode=="slayer", so raw mode
    # must not invoke it at all (defence-in-depth: the helper also no-ops in
    # raw — see test_download_slayer_setup_noop_in_raw).
    assert calls == [], "raw mode must not download any slayer setup"


# --- download_slayer_setup: lands at the per-combo env root, idempotent ----


@pytest.mark.parametrize(
    "framework, slayer_setup, artifact, env_var",
    [
        ("pydantic_ai_recursive", "pre-encoded", "slayer_models",
         "BIRD_SLAYER_MODELS_ROOT"),
        ("pydantic_ai_recursive", "on-the-fly", "slayer_otf_cache",
         "BIRD_OTF_CACHE_ROOT"),
        ("pydantic_ai_otf_encode", "on-the-fly", "slayer_models_otf",
         "BIRD_SLAYER_MODELS_OTF_ROOT"),
    ],
)
def test_download_slayer_setup_lands_at_combo_root(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
    framework: str, slayer_setup: str, artifact: str, env_var: str,
):
    """download_slayer_setup pulls the combo's uploaded dir to the env-override
    root the local readers use, preserving nested paths + binary bytes, and
    writes the .download_complete marker."""
    client, store = fake_gcs_bucket
    dest_root = tmp_path / "data" / artifact
    monkeypatch.setenv(env_var, str(dest_root))

    pfx = f"runs/{RUN_ID}/slayer_setup/{artifact}"
    store[f"{pfx}/db_a/_marker"] = b"m"
    store[f"{pfx}/db_a/models/db_a/x.yaml"] = b"name: x\n"
    store[f"{pfx}/db_a/embeddings.db"] = b"SQLite format 3\x00\xff"
    store[f"{pfx}/db_b/_marker"] = b"m2"

    # DEV-1470: otf_encode now ALSO downloads the REQUIRED cache. Lay down a
    # minimal cache prefix + point its env-override root, so the otf_encode
    # case doesn't fail on the missing-cache fail-fast.
    if framework == "pydantic_ai_otf_encode":
        cache_root = tmp_path / "data" / "slayer_otf_cache"
        monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
        cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
        store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"

    cfg = {
        "query_mode": "slayer", "slayer_setup": slayer_setup,
        "framework": framework,
    }
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    assert (dest_root / "db_a" / "models" / "db_a" / "x.yaml").read_text() == "name: x\n"
    assert (dest_root / "db_a" / "embeddings.db").read_bytes() == b"SQLite format 3\x00\xff"
    assert (dest_root / "db_b" / "_marker").read_bytes() == b"m2"
    # DEV-1470: required artifacts use `.download_complete`; the optional
    # `slayer_models_otf` seed uses `.optional_seed_download_complete` instead.
    expected_marker = (
        ".optional_seed_download_complete" if artifact == "slayer_models_otf"
        else ".download_complete"
    )
    assert (dest_root / expected_marker).is_file(), (
        f"expected {expected_marker} marker under {dest_root}"
    )


def test_download_slayer_setup_idempotent_under_two_actors(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """Two racing in-process actors must not crash or corrupt the dest — the
    root-level .download_complete marker (REQUIRED) and
    .optional_seed_download_complete (OPTIONAL) make the second call a no-op."""
    client, store = fake_gcs_bucket
    # DEV-1470: otf_encode now has TWO artifacts. Set up both.
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))
    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"fp"
    store[f"{rpfx}/db_a/_kb_rows.json"] = b"[]"

    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "pydantic_ai_otf_encode",
    }
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)  # must no-op

    assert (ref_root / "db_a" / "_reference_fp.txt").read_bytes() == b"fp"
    assert (cache_root / ".download_complete").is_file()
    assert (ref_root / ".optional_seed_download_complete").is_file()


def test_download_slayer_setup_noop_in_raw(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """In raw mode the helper does nothing — it never touches the slayer
    roots."""
    client, store = fake_gcs_bucket
    dest_root = tmp_path / "data" / "slayer_models"
    monkeypatch.setenv("BIRD_SLAYER_MODELS_ROOT", str(dest_root))
    store[f"runs/{RUN_ID}/slayer_setup/slayer_models/db_a/x"] = b"x"

    ray_app.download_slayer_setup(
        RUN_ID, {"query_mode": "raw", "slayer_setup": "pre-encoded",
                 "framework": "pydantic_ai"},
        client=client,
    )
    assert not dest_root.exists()


def test_download_slayer_setup_rename_race_is_success(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """If a peer process won the rename onto dest_root while we downloaded
    (dest_root now exists + marked), our os.rename raises OSError; the helper
    must treat that as success, not crash."""
    client, store = fake_gcs_bucket
    dest_root = tmp_path / "data" / "slayer_otf_cache"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(dest_root))
    store[f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache/db_a/_cache_fp.txt"] = b"fp"

    real_rename = os.rename

    def racing_rename(src, dst):
        if Path(dst) == dest_root and not dest_root.exists():
            dest_root.mkdir(parents=True)
            (dest_root / ".download_complete").write_text("peer")
            raise OSError("Directory not empty")
        return real_rename(src, dst)

    monkeypatch.setattr(ray_app.os, "rename", racing_rename)
    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "pydantic_ai_recursive",
    }
    # Must not raise.
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)
    assert (dest_root / ".download_complete").is_file()


def test_download_slayer_setup_rename_error_without_marker_reraises(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """A genuine os.rename failure (dest_root absent + no marker) must
    propagate — not be silently swallowed as a race win."""
    client, store = fake_gcs_bucket
    dest_root = tmp_path / "data" / "slayer_otf_cache"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(dest_root))
    store[f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache/db_a/_cache_fp.txt"] = b"fp"

    def boom_rename(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(ray_app.os, "rename", boom_rename)
    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "pydantic_ai_recursive",
    }
    with pytest.raises(OSError, match="disk on fire"):
        ray_app.download_slayer_setup(RUN_ID, cfg, client=client)
    assert not (dest_root / ".download_complete").exists()


def test_download_slayer_setup_empty_prefix_raises(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """A missing/empty GCS prefix (failed/partial upload, wrong prefix) must
    NOT be cached as a complete download — it must raise so the bad upload
    surfaces, never permanently wedging tasks against an empty setup (Codex)."""
    client, _store = fake_gcs_bucket  # empty store → no blobs under the prefix
    dest_root = tmp_path / "data" / "slayer_otf_cache"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(dest_root))
    cfg = {
        "query_mode": "slayer", "slayer_setup": "on-the-fly",
        "framework": "pydantic_ai_recursive",
    }
    with pytest.raises(FileNotFoundError):
        ray_app.download_slayer_setup(RUN_ID, cfg, client=client)
    # Nothing cached — a retry (after fixing the upload) can still run.
    assert not (dest_root / ".download_complete").exists()
    assert not dest_root.exists()


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
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket
):
    """Slayer mode rotates per-task storage — caching the runner would
    freeze it. The actor must skip the cache (make_runner never called)."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

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
        framework="pydantic_ai_recursive",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2")
        },
        slayer_setup="on-the-fly",
        slayer_storage_root="/data/slayer_models",
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
):
    """After a task returns, the per-task log tmp dir must be removed —
    without cleanup they accumulate and fill worker disks on big runs (CR#11).
    DEV-1468: the per-task SLayer SQLite tmp dir is gone (no ephemeral
    server), so the only per-task tmp is the `cloud_log_` dir; assert it (and
    any stray `slayer_` dirs) are cleaned in slayer mode too."""
    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

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
        framework="pydantic_ai_recursive",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            iid: {"instance_id": iid, "selected_database": "db_a"}
            for iid in ("db_a_1", "db_a_2")
        },
        slayer_setup="on-the-fly",
        slayer_storage_root="/data/slayer_models",
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


# ---------------------------------------------------------------------------
# DEV-1470 — multi-artifact setup download for otf_encode + on-the-fly:
# REQUIRED cache + OPTIONAL reference (no-op when empty, file-merge when present).
# ---------------------------------------------------------------------------


def test_download_slayer_setup_otf_encode_downloads_both_cache_and_reference(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """For `otf_encode + on-the-fly` the actor must download BOTH the required
    deterministic cache (`slayer_otf_cache/`) AND the optional reference seed
    (`slayer_models_otf/`) to their respective env-override roots."""
    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    store[f"{cpfx}/db_a/_kb_rows.json"] = b"[]"
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"ref-fp"
    store[f"{rpfx}/db_a/models/x.yaml"] = b"name: x\n"

    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    # Both roots populated, both markers written.
    assert (cache_root / "db_a" / "_cache_fp.txt").read_bytes() == b"cache-fp"
    assert (cache_root / ".download_complete").is_file()
    assert (ref_root / "db_a" / "_reference_fp.txt").read_bytes() == b"ref-fp"
    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"name: x\n"


def test_download_slayer_setup_otf_encode_optional_reference_empty_is_noop(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """M2/H3 — for `otf_encode + on-the-fly` an empty `slayer_models_otf/`
    prefix means "no local seed, cloud will encode" and must NO-OP (no
    marker, no raise, no rmtree). The REQUIRED cache prefix still raises
    when empty (existing semantics)."""
    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    # Cache prefix present; reference prefix is EMPTY (no seed uploaded).
    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"

    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    # MUST NOT raise — the optional artifact is allowed to be absent.
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    # Cache landed.
    assert (cache_root / "db_a" / "_cache_fp.txt").read_bytes() == b"cache-fp"
    assert (cache_root / ".download_complete").is_file()
    # Reference root either absent or empty — must NOT carry the marker
    # (the cloud will encode references into this root; a marker would
    # prevent retries).
    assert not (ref_root / ".download_complete").exists()


def test_download_slayer_setup_otf_encode_optional_reference_skipped_on_restart(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """M3 — after an actor restart, `download_slayer_setup` MUST NOT clobber
    cloud-built references that already exist under `ref_root/<db>/`. The
    OPTIONAL download is gated by its own `.optional_seed_download_complete`
    marker; once written it's idempotent. If the marker is absent but the
    root has cloud-built content, re-download merges file-by-file (newest
    mtime wins) instead of rmtree+rename."""
    import os

    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    # Seed for db_a present in GCS.
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"seed-fp"
    store[f"{rpfx}/db_a/models/x.yaml"] = b"name: SEED-OLD\n"

    # Simulate a prior actor having built a CLOUD reference for db_b
    # locally — must not be touched.
    (ref_root / "db_b" / "models").mkdir(parents=True)
    (ref_root / "db_b" / "models" / "y.yaml").write_bytes(b"CLOUD-BUILT\n")
    (ref_root / "db_b" / "_reference_fp.txt").write_bytes(b"cloud-fp-b")
    cloud_built_mtime = time.time() + 100
    for p in (ref_root / "db_b").rglob("*"):
        if p.is_file():
            os.utime(p, (cloud_built_mtime, cloud_built_mtime))

    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    # MUST NOT raise; cloud-built db_b must survive.
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    # db_b cloud-built reference UNTOUCHED.
    assert (ref_root / "db_b" / "models" / "y.yaml").read_bytes() == b"CLOUD-BUILT\n"
    assert (ref_root / "db_b" / "_reference_fp.txt").read_bytes() == b"cloud-fp-b"
    # db_a SEED arrived (no prior local content for db_a).
    assert (ref_root / "db_a" / "_reference_fp.txt").read_bytes() == b"seed-fp"


def test_download_slayer_setup_optional_seed_merges_per_file_newest_mtime(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """M1 — when optional seed exists in GCS AND the local reference root
    already has files for the same db (from a prior actor's cloud build), the
    download must MERGE file-by-file (newest mtime wins), NOT atomic-replace
    the existing dir."""
    import os

    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    # Seed in GCS — but OLD.
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"seed-old-fp"
    store[f"{rpfx}/db_a/models/x.yaml"] = b"SEED-OLD\n"
    store[f"{rpfx}/db_a/models/y.yaml"] = b"SEED-Y\n"

    # Local already has a NEWER cloud-built x.yaml.
    local_db = ref_root / "db_a"
    (local_db / "models").mkdir(parents=True)
    (local_db / "models" / "x.yaml").write_bytes(b"LOCAL-NEW\n")
    future_mtime = time.time() + 10_000
    os.utime(local_db / "models" / "x.yaml", (future_mtime, future_mtime))

    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    # Local-newer x.yaml survives; seed's y.yaml lands fresh.
    assert (local_db / "models" / "x.yaml").read_bytes() == b"LOCAL-NEW\n", (
        "older seed must NOT overwrite newer local file (per-file mtime-wins)"
    )
    assert (local_db / "models" / "y.yaml").read_bytes() == b"SEED-Y\n", (
        "seed file absent locally must be downloaded"
    )


def test_optional_seed_merge_preserves_src_mtime(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """CodeRabbit r2 — when the optional seed is merged file-by-file into an
    existing root, the local dst MUST inherit the source's mtime, not the
    download/replace time. Without this, a later actor restart's mtime
    comparison reads the local-write time and can wrongly block a genuinely
    newer seed/local reference from winning."""
    import os

    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"seed"
    store[f"{rpfx}/db_a/models/x.yaml"] = b"seed-content\n"

    # Capture the src mtime that the download will see (download_prefix
    # writes the bytes with whatever mtime the OS sets at write-time; we
    # just want to verify dst's mtime matches src's after merge).
    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    # Spy on os.utime to assert the merge calls it before os.replace.
    real_utime = os.utime
    utime_calls: list[tuple[str, tuple[float, float]]] = []

    def spy_utime(path, times):
        utime_calls.append((str(path), times))
        return real_utime(path, times)

    monkeypatch.setattr(os, "utime", spy_utime)
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    # The merge path must have stamped the tmp file with src_mtime.
    seed_writes = [
        c for c in utime_calls
        if "seed-" in c[0] and c[1][0] == c[1][1]
    ]
    assert seed_writes, (
        "optional seed merge did not call os.utime on the tmp file before "
        f"os.replace — dst will inherit fetch-time mtime instead of src "
        f"mtime. All utime calls: {utime_calls}"
    )


def test_download_slayer_setup_optional_seed_marker_makes_rerun_noop(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """M1 — after a successful optional-seed merge, the helper must write a
    `.optional_seed_download_complete` marker in the reference root. A second
    call must observe the marker and no-op (no re-download, no re-merge)."""
    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"seed"
    store[f"{rpfx}/db_a/models/x.yaml"] = b"seed-content\n"

    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)

    marker = ref_root / ".optional_seed_download_complete"
    assert marker.is_file(), (
        "optional seed download must drop `.optional_seed_download_complete` "
        "after a successful merge"
    )

    # Mutate the GCS seed content — a second call MUST NOT re-download it.
    store[f"{rpfx}/db_a/models/x.yaml"] = b"NEW-SEED-MUST-NOT-LAND\n"
    ray_app.download_slayer_setup(RUN_ID, cfg, client=client)
    assert (ref_root / "db_a" / "models" / "x.yaml").read_bytes() == b"seed-content\n", (
        "second call must no-op when the marker is present"
    )


def _otf_seed_lock_holder(args):
    """Worker that takes `<ref_root>/<db>.build.lock` and holds it briefly."""
    import fcntl
    ref_root_str, db, hold_s, sentinel_path = args
    from pathlib import Path as _P
    import time as _t
    ref_root = _P(ref_root_str)
    ref_root.mkdir(parents=True, exist_ok=True)
    lock_path = ref_root / f"{db}.build.lock"
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        _P(sentinel_path).write_text("locked")
        _t.sleep(hold_s)
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def test_optional_seed_download_blocks_on_per_db_build_lock(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """H4 — the optional reference seed download must take the SAME per-DB
    `fcntl.flock` on `<ref_root>/<db>.build.lock` that `_build_reference`
    holds, so a download cannot land on top of an in-progress cloud encoder
    (which would clobber its writes). A peer process holding the lock must
    cause the download to block until release."""
    import multiprocessing
    import time as _t
    import fcntl  # noqa: F401 — fail loudly if unavailable

    client, store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cpfx = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache"
    rpfx = f"runs/{RUN_ID}/slayer_setup/slayer_models_otf"
    store[f"{cpfx}/db_a/_cache_fp.txt"] = b"cache-fp"
    store[f"{rpfx}/db_a/_reference_fp.txt"] = b"seed"
    store[f"{rpfx}/db_a/models/x.yaml"] = b"seed-content\n"

    sentinel = tmp_path / "holder.txt"
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=_otf_seed_lock_holder,
        args=((str(ref_root), "db_a", 2.0, str(sentinel)),),
    )
    p.start()
    try:
        deadline = _t.time() + 5.0
        while not sentinel.exists() and _t.time() < deadline:
            _t.sleep(0.05)
        assert sentinel.exists(), "holder failed to acquire the lock"

        cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
               "framework": "pydantic_ai_otf_encode"}
        t0 = _t.time()
        ray_app.download_slayer_setup(RUN_ID, cfg, client=client)
        elapsed = _t.time() - t0
        assert elapsed >= 0.5, (
            f"optional seed download did not block on the per-DB build lock "
            f"(elapsed={elapsed:.3f}s) — H4 race against in-flight encoder remains open"
        )
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()


def test_download_slayer_setup_otf_encode_missing_cache_still_raises(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """An empty REQUIRED cache prefix must still raise FileNotFoundError —
    the change in DEV-1470 makes only the reference optional; the cache
    remains required."""
    client, _store = fake_gcs_bucket
    cache_root = tmp_path / "data" / "slayer_otf_cache"
    ref_root = tmp_path / "data" / "slayer_models_otf"
    monkeypatch.setenv("BIRD_OTF_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(ref_root))

    cfg = {"query_mode": "slayer", "slayer_setup": "on-the-fly",
           "framework": "pydantic_ai_otf_encode"}
    with pytest.raises(FileNotFoundError):
        ray_app.download_slayer_setup(RUN_ID, cfg, client=client)


# ---------------------------------------------------------------------------
# DEV-1470 — _run_one_in_actor wires upload-back AFTER row/log writes
# ---------------------------------------------------------------------------


def test_run_one_in_actor_invokes_upload_back_after_row_and_log(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket,
):
    """After the row + log writes, the actor must call (in order):
      1. _gcs.write_row
      2. _gcs.write_log
      3. upload_back.upload_per_task_debug
      4. upload_back.upload_per_task_setup_sessions
      5. upload_back.upload_otf_reference_delta
    BEFORE wiping the per-task log tmp dir. (H2: previously the test only
    asserted the upload-back order, leaving the row-before-upload invariant
    unproven.)"""
    from bird_interact_agents.cloud import upload_back, gcs as _gcs

    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    # Stub setup download — these tests exercise upload-back wiring, not the
    # download path.
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0,
            "duration_s": 0.01, "error": None,
        }
    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )

    order: list[str] = []
    # Instrument BOTH the row/log writes and the upload-back calls so the
    # full sequence is observable.
    monkeypatch.setattr(
        ray_app._gcs, "write_row",
        lambda *a, **kw: order.append("write_row"),
    )
    monkeypatch.setattr(
        ray_app._gcs, "write_log",
        lambda *a, **kw: order.append("write_log"),
    )
    monkeypatch.setattr(
        upload_back, "upload_per_task_debug",
        lambda **kw: order.append("debug"),
    )
    monkeypatch.setattr(
        upload_back, "upload_per_task_setup_sessions",
        lambda **kw: order.append("setup_sessions"),
    )
    monkeypatch.setattr(
        upload_back, "upload_otf_reference_delta",
        lambda **kw: order.append("ref_delta"),
    )

    actor = ray_app._LocalActor(
        {"framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
         "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
         "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
         "patience": 3, "strict": False, "use_audited_gold_sql": False,
         "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
         "slayer_storage_root": "/data/slayer_models"},
        RUN_ID, 1, gcs_client=client,
    )
    actor.run_one({"instance_id": "db_a_1", "selected_database": "db_a"})

    # write_log may be skipped when log_tmp is empty; require row+log if log
    # is non-empty, but in all cases write_row precedes any upload-back.
    assert "write_row" in order, f"row never written: {order}"
    row_idx = order.index("write_row")
    debug_idx = order.index("debug")
    setup_idx = order.index("setup_sessions")
    ref_idx = order.index("ref_delta")
    assert row_idx < debug_idx, (
        f"upload-back ran before row write: {order}"
    )
    assert debug_idx < setup_idx < ref_idx, (
        f"upload-back functions called in wrong order: {order}"
    )
    if "write_log" in order:
        log_idx = order.index("write_log")
        assert row_idx < log_idx < debug_idx, (
            f"write_log out of order vs write_row/upload-back: {order}"
        )


def test_run_one_in_actor_swallows_upload_back_exceptions(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket,
):
    """An upload-back exception MUST NOT propagate — the per-task row already
    landed before the hook; failing the actor here would corrupt the
    in-flight ActorPool slot for no logging gain."""
    from bird_interact_agents.cloud import upload_back

    client, store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    # Stub setup download — this test exercises upload-back error handling,
    # not the download path.
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True, "total_reward": 1.0,
            "duration_s": 0.01, "error": None,
        }
    monkeypatch.setattr(
        "bird_interact_agents.run.run_one_task", fake_run_one_task
    )
    monkeypatch.setattr(
        upload_back, "upload_per_task_debug",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("simulated upload boom")),
    )
    monkeypatch.setattr(
        upload_back, "upload_per_task_setup_sessions",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        upload_back, "upload_otf_reference_delta",
        lambda **kw: None,
    )

    actor = ray_app._LocalActor(
        {"framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
         "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
         "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
         "patience": 3, "strict": False, "use_audited_gold_sql": False,
         "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
         "slayer_storage_root": "/data/slayer_models"},
        RUN_ID, 1, gcs_client=client,
    )
    # Must NOT raise.
    actor.run_one({"instance_id": "db_a_1", "selected_database": "db_a"})
    # Row still landed.
    assert f"runs/{RUN_ID}/rows/db_a_1/attempt-1.json" in store


def test_actor_captures_uploaded_dbs_and_initial_seed_fps(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """H2/H3 — the actor must, at init, snapshot the FINGERPRINTS of every
    `_reference_fp.txt` present in the seed under `slayer_models_otf_root()`
    AFTER `download_slayer_setup` has run (snapshotting BEFORE would miss
    everything in the seed and incorrectly mark cloud-built references as
    pre-existing — and then never upload them back). Test this by having the
    fake `download_slayer_setup` CREATE the seed files: a snapshot taken
    before would observe an empty root and assert against `{}`."""
    from bird_interact_agents import paths as _paths

    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    ref_root = tmp_path / "slayer_models_otf"
    monkeypatch.setattr(_paths, "slayer_models_otf_root", lambda: ref_root)

    def fake_download(run_id, cfg, *, client):  # noqa: ARG001
        # The download is what populates the seed; the snapshot MUST observe
        # what download just landed.
        (ref_root / "db_a").mkdir(parents=True, exist_ok=True)
        (ref_root / "db_a" / "_reference_fp.txt").write_text("seed-fp-A")
        (ref_root / "db_b").mkdir(parents=True, exist_ok=True)
        (ref_root / "db_b" / "_reference_fp.txt").write_text("seed-fp-B")

    monkeypatch.setattr(ray_app, "download_slayer_setup", fake_download)

    cfg = {
        "framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
        "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 3, "strict": False, "use_audited_gold_sql": False,
        "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
        "slayer_storage_root": "/data/slayer_models",
    }
    actor = ray_app._LocalActor(cfg, RUN_ID, 1, gcs_client=client)

    assert hasattr(actor, "uploaded_dbs"), (
        "actor must expose `uploaded_dbs` (set[str]) for per-actor retry tracking"
    )
    assert actor.uploaded_dbs == set()
    assert hasattr(actor, "initial_seed_fp_by_db"), (
        "actor must expose `initial_seed_fp_by_db` (dict[str, str]) snapshot"
    )
    # Snapshot must see what `download_slayer_setup` JUST laid down — proves
    # the snapshot ran AFTER, not before.
    assert actor.initial_seed_fp_by_db == {"db_a": "seed-fp-A", "db_b": "seed-fp-B"}


def test_actor_initial_seed_fps_empty_when_no_seed(
    monkeypatch: pytest.MonkeyPatch, fake_gcs_bucket, tmp_path: Path,
):
    """When the reference root is empty (no seeds uploaded), the snapshot is
    an empty dict — every db the cloud encodes will get uploaded."""
    from bird_interact_agents import paths as _paths

    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)
    monkeypatch.setattr(ray_app, "download_slayer_setup", lambda *a, **k: None)

    ref_root = tmp_path / "slayer_models_otf"  # absent
    monkeypatch.setattr(_paths, "slayer_models_otf_root", lambda: ref_root)

    cfg = {
        "framework": "pydantic_ai_otf_encode", "query_mode": "slayer",
        "mode": "a-interact", "agent_model": "anthropic/claude-sonnet-4-5",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 3, "strict": False, "use_audited_gold_sql": False,
        "prompt_cache": True, "max_depth": 3, "slayer_setup": "on-the-fly",
        "slayer_storage_root": "/data/slayer_models",
    }
    actor = ray_app._LocalActor(cfg, RUN_ID, 1, gcs_client=client)
    assert actor.initial_seed_fp_by_db == {}
    assert actor.uploaded_dbs == set()
