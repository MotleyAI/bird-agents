"""DEV-1470: per-task debug bundle + setup-encoder session + OTF reference
delta upload-back from cloud workers.

Three upload entry points called from `_run_one_in_actor` after the row+log
write, before the per-task tmp-dir cleanup:

* `upload_per_task_debug(run_id, iid, attempt, work_root, client)` — ships
  `<work_root>/<iid>/` (sessions/<sid>.{json,md} + INDEX.md + the per-task
  HARD-8 SLayer scratch storage) to `runs/<run_id>/sessions/<iid>/attempt-<n>/`.
* `upload_per_task_setup_sessions(run_id, iid, attempt, setup_sessions_root,
  task_start_ts, client)` — for each `<setup_sessions_root>/<db>/` with any
  file mtime >= task_start_ts, uploads to
  `runs/<run_id>/sessions/_setup_sessions/<db>/`.
* `upload_otf_reference_delta(run_id, cfg, shard, uploaded_dbs,
  initial_seed_fp_by_db, client)` — for each `<db>/` under
  `paths.slayer_models_otf_root()` whose `_reference_fp.txt` exists AND
  `<db>` is not in `uploaded_dbs` AND fingerprint differs from
  `initial_seed_fp_by_db.get(<db>)`, uploads the ENTIRE `<db>/` to
  `runs/<run_id>/post_run/slayer_models_otf/<shard>/<db>/`, then a sidecar
  `_source_mtimes.json` (per-file local mtimes), then a final
  `_upload_complete` marker — in that order. Adds `<db>` to `uploaded_dbs`
  ONLY on success.

All three are best-effort: any exception is swallowed + logged to stderr so
the per-task row already-landed before this hook is never lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_interact_agents.cloud import upload_back


RUN_ID = "20260521t1422-pydanticaiotf-slayer-deadbe"


# ---------------------------------------------------------------------------
# upload_per_task_debug
# ---------------------------------------------------------------------------


def test_upload_per_task_debug_ships_sessions_and_scratch(
    tmp_path: Path, fake_gcs_bucket,
):
    """The whole `<work_root>/<iid>/` tree lands under
    `runs/<run_id>/sessions/<iid>/attempt-<n>/`. Binary bytes preserved."""
    client, store = fake_gcs_bucket
    work_root = tmp_path / "bird_interact_slayer_otf"
    iid_dir = work_root / "db_a_1"
    (iid_dir / "sessions").mkdir(parents=True)
    (iid_dir / "sessions" / "agent__top.json").write_text(
        json.dumps([{"role": "user", "content": "x"}])
    )
    (iid_dir / "sessions" / "agent__top.md").write_text("# session\n")
    (iid_dir / "sessions" / "INDEX.md").write_text("| session | role |\n")
    (iid_dir / "scratch_storage" / "models").mkdir(parents=True)
    (iid_dir / "scratch_storage" / "models" / "x.yaml").write_text("name: x\n")
    (iid_dir / "scratch_storage" / "embeddings.db").write_bytes(b"\x00\xff\x01")

    upload_back.upload_per_task_debug(
        run_id=RUN_ID, iid="db_a_1", attempt=2,
        work_root=work_root, client=client,
    )

    base = f"runs/{RUN_ID}/sessions/db_a_1/attempt-2"
    assert store[f"{base}/sessions/agent__top.json"] == (
        json.dumps([{"role": "user", "content": "x"}]).encode()
    )
    assert store[f"{base}/sessions/agent__top.md"] == b"# session\n"
    assert store[f"{base}/sessions/INDEX.md"].startswith(b"| session")
    assert store[f"{base}/scratch_storage/models/x.yaml"] == b"name: x\n"
    assert store[f"{base}/scratch_storage/embeddings.db"] == b"\x00\xff\x01"


def test_upload_per_task_debug_noop_when_iid_dir_absent(
    tmp_path: Path, fake_gcs_bucket,
):
    """If `<work_root>/<iid>/` doesn't exist (non-OTF agent, or the agent
    never wrote anything), the helper must no-op silently."""
    client, store = fake_gcs_bucket
    work_root = tmp_path / "bird_interact_slayer_otf"
    work_root.mkdir()

    upload_back.upload_per_task_debug(
        run_id=RUN_ID, iid="db_a_1", attempt=1,
        work_root=work_root, client=client,
    )
    assert store == {}


def test_upload_per_task_debug_swallows_exceptions(
    tmp_path: Path, fake_gcs_bucket, capsys: pytest.CaptureFixture,
):
    """Upload failure must NOT propagate — the per-task row already landed
    and a logging-side failure must never poison the run."""
    client, _store = fake_gcs_bucket
    work_root = tmp_path / "bird_interact_slayer_otf"
    iid_dir = work_root / "db_a_1"
    iid_dir.mkdir(parents=True)
    (iid_dir / "x.json").write_text("{}")

    # Make every upload raise.
    def boom_blob(_name):
        class _B:
            def upload_from_string(self, *a, **kw):
                raise RuntimeError("simulated gcs failure")
        return _B()

    client.bucket = lambda _name: type("B", (), {"blob": staticmethod(boom_blob)})()

    # Must not raise.
    upload_back.upload_per_task_debug(
        run_id=RUN_ID, iid="db_a_1", attempt=1,
        work_root=work_root, client=client,
    )
    captured = capsys.readouterr()
    assert "upload_back" in captured.err.lower() or "db_a_1" in captured.err


# ---------------------------------------------------------------------------
# upload_per_task_setup_sessions
# ---------------------------------------------------------------------------


def test_upload_per_task_setup_sessions_ships_dbs_newer_than_start(
    tmp_path: Path, fake_gcs_bucket,
):
    """Only `<setup_sessions_root>/<db>/` trees with at least one file mtime
    >= task_start_ts get shipped. Pre-existing setup-session dirs from earlier
    tasks must NOT be re-uploaded."""
    import os

    client, store = fake_gcs_bucket
    sroot = tmp_path / "_setup_sessions"
    fresh = sroot / "db_a"
    stale = sroot / "db_b"
    for d in (fresh, stale):
        d.mkdir(parents=True)
        (d / "setup__kb_001.md").write_text("# x\n")

    # Stamp stale db_b BEFORE task_start; fresh db_a AFTER.
    task_start_ts = 1_700_000_000.0
    os.utime(stale / "setup__kb_001.md",
             (task_start_ts - 100, task_start_ts - 100))
    os.utime(fresh / "setup__kb_001.md",
             (task_start_ts + 50, task_start_ts + 50))

    upload_back.upload_per_task_setup_sessions(
        run_id=RUN_ID, iid="db_a_1", attempt=1,
        setup_sessions_root=sroot, task_start_ts=task_start_ts,
        client=client,
    )

    base = f"runs/{RUN_ID}/sessions/_setup_sessions"
    assert f"{base}/db_a/setup__kb_001.md" in store
    assert all(not k.startswith(f"{base}/db_b/") for k in store), (
        f"stale db_b sessions were re-uploaded: {sorted(store)}"
    )


def test_upload_per_task_setup_sessions_noop_when_root_absent(
    tmp_path: Path, fake_gcs_bucket,
):
    client, store = fake_gcs_bucket
    upload_back.upload_per_task_setup_sessions(
        run_id=RUN_ID, iid="x", attempt=1,
        setup_sessions_root=tmp_path / "_setup_sessions",  # absent
        task_start_ts=0.0, client=client,
    )
    assert store == {}


def test_upload_per_task_setup_sessions_swallows_exceptions(
    tmp_path: Path, fake_gcs_bucket, capsys: pytest.CaptureFixture,
):
    """M2 — best-effort: an upload failure must NOT propagate. Symmetric to
    `upload_per_task_debug` and `upload_otf_reference_delta`."""
    import os

    client, _store = fake_gcs_bucket
    sroot = tmp_path / "_setup_sessions"
    fresh = sroot / "db_a"
    fresh.mkdir(parents=True)
    (fresh / "setup__kb_001.md").write_text("# x\n")
    task_start_ts = 0.0
    os.utime(fresh / "setup__kb_001.md", (task_start_ts + 50, task_start_ts + 50))

    def boom_blob(_name):
        class _B:
            def upload_from_string(self, *a, **kw):
                raise RuntimeError("simulated gcs failure")
        return _B()

    client.bucket = lambda _name: type("B", (), {"blob": staticmethod(boom_blob)})()

    # Must NOT raise.
    upload_back.upload_per_task_setup_sessions(
        run_id=RUN_ID, iid="db_a_1", attempt=1,
        setup_sessions_root=sroot, task_start_ts=task_start_ts,
        client=client,
    )
    captured = capsys.readouterr()
    assert "upload_back" in captured.err.lower() or "_setup_sessions" in captured.err


# ---------------------------------------------------------------------------
# upload_otf_reference_delta — H2 (per-actor retry tracker, not mtime threshold)
# ---------------------------------------------------------------------------


def _make_otf_ref_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    dbs: dict[str, str],
) -> Path:
    """Build a `slayer_models_otf/<db>/` tree with each db carrying
    `_reference_fp.txt = <fp>` plus a couple of payload files. Returns the root
    and monkeypatches `paths.slayer_models_otf_root` to point at it."""
    from bird_interact_agents import paths as _paths

    root = tmp_path / "slayer_models_otf"
    for db, fp in dbs.items():
        d = root / db
        (d / "models").mkdir(parents=True)
        (d / "models" / "x.yaml").write_text(f"name: {db}\n")
        (d / "memories.yaml").write_text("[]\n")
        (d / "_reference_fp.txt").write_text(fp)
    monkeypatch.setattr(_paths, "slayer_models_otf_root", lambda: root)
    return root


def _cfg_for_otf_encode() -> dict:
    return {
        "query_mode": "slayer",
        "slayer_setup": "on-the-fly",
        "framework": "pydantic_ai_otf_encode",
    }


def test_upload_otf_reference_delta_ships_new_db_whole_subtree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """A db with a present `_reference_fp.txt` whose fingerprint differs from
    the initial seed (or wasn't seeded at all) must ship its ENTIRE subtree
    to `post_run/slayer_models_otf/<shard>/<db>/` — whole-dir, not per-file,
    so the marker moves atomically with its content."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-cloud-new"})
    uploaded: set[str] = set()
    initial_seed_fp: dict[str, str] = {}  # nothing seeded — db_a is brand new

    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host1-12345",
        uploaded_dbs=uploaded,
        initial_seed_fp_by_db=initial_seed_fp,
        client=client,
    )

    base = f"runs/{RUN_ID}/post_run/slayer_models_otf/host1-12345/db_a"
    assert store[f"{base}/_reference_fp.txt"] == b"fp-cloud-new"
    assert store[f"{base}/models/x.yaml"] == b"name: db_a\n"
    assert store[f"{base}/memories.yaml"] == b"[]\n"
    assert "db_a" in uploaded, "db_a must be recorded as uploaded"


def test_upload_otf_reference_delta_writes_sidecar_and_marker_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Upload ordering (M1): every content file is uploaded BEFORE the sidecar
    `_source_mtimes.json`, and the sidecar BEFORE the final `_upload_complete`.
    Otherwise a concurrent fetch could merge a half-uploaded shard."""
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-x"})
    upload_order: list[str] = []

    class RecordingClient:
        def bucket(self, _name):
            return self

        def blob(self, name):
            client_self = self

            class _B:
                def upload_from_string(self, *a, **kw):
                    upload_order.append(name)
            return _B()

    client = RecordingClient()
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host1-12345",
        uploaded_dbs=set(),
        initial_seed_fp_by_db={},
        client=client,
    )

    base = f"runs/{RUN_ID}/post_run/slayer_models_otf/host1-12345/db_a"
    # Find the sidecar and marker positions.
    sidecar_idx = upload_order.index(f"{base}/_source_mtimes.json")
    marker_idx = upload_order.index(f"{base}/_upload_complete")
    content_indices = [
        i for i, n in enumerate(upload_order)
        if n.startswith(f"{base}/") and n not in (
            f"{base}/_source_mtimes.json", f"{base}/_upload_complete",
        )
    ]
    assert content_indices, "no content uploads recorded"
    assert max(content_indices) < sidecar_idx, (
        f"sidecar uploaded before some content; order={upload_order}"
    )
    assert sidecar_idx < marker_idx, (
        f"_upload_complete must be LAST; order={upload_order}"
    )


def test_upload_otf_reference_delta_sidecar_carries_per_file_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """The sidecar maps each relative path to its FLOAT local mtime so the
    laptop merger can apply newest-mtime-wins."""
    import os

    client, store = fake_gcs_bucket
    root = _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-x"})
    target_mtime = 1_725_000_000.5
    for p in (root / "db_a").rglob("*"):
        if p.is_file():
            os.utime(p, (target_mtime, target_mtime))

    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=set(),
        initial_seed_fp_by_db={}, client=client,
    )

    sidecar = json.loads(
        store[f"runs/{RUN_ID}/post_run/slayer_models_otf/host-1/db_a/_source_mtimes.json"]
    )
    assert "_reference_fp.txt" in sidecar
    assert "models/x.yaml" in sidecar
    for v in sidecar.values():
        assert isinstance(v, float)
        assert abs(v - target_mtime) < 1.0


def test_upload_otf_reference_delta_skips_dbs_in_uploaded_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """A db already in `uploaded_dbs` (uploaded on a prior task) must be
    skipped on subsequent tasks — no duplicate uploads."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-cloud-built"})
    uploaded = {"db_a"}

    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=uploaded,
        initial_seed_fp_by_db={}, client=client,
    )
    assert all(not k.startswith(
        f"runs/{RUN_ID}/post_run/slayer_models_otf/host-1/db_a/"
    ) for k in store)


def test_upload_otf_reference_delta_skips_unchanged_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """If the on-disk `_reference_fp.txt` matches the initial seed fingerprint,
    the cloud didn't rebuild — nothing new to ship. Skip."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-seeded"})
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=set(),
        initial_seed_fp_by_db={"db_a": "fp-seeded"},
        client=client,
    )
    assert all(not k.startswith(
        f"runs/{RUN_ID}/post_run/slayer_models_otf/host-1/db_a/"
    ) for k in store)


def test_upload_otf_reference_delta_uploads_when_fp_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """Local on-disk fp differs from the initial seed fp → upload."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-cloud-new"})
    uploaded: set[str] = set()
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=uploaded,
        initial_seed_fp_by_db={"db_a": "fp-seeded-OLD"},
        client=client,
    )
    assert f"runs/{RUN_ID}/post_run/slayer_models_otf/host-1/db_a/_reference_fp.txt" in store
    assert uploaded == {"db_a"}


def test_upload_otf_reference_delta_retries_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """H2 / M6 — upload failure must NOT mark the db as uploaded; the next
    task gets another chance. Otherwise an unlucky first-attempt failure
    permanently loses the warm-cache artifact."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp-cloud"})
    uploaded: set[str] = set()

    # First attempt: every blob upload raises. Save the underlying bucket's
    # `.blob` so we can restore it (the fake bucket is a single SimpleNamespace
    # shared across `client.bucket(...)` calls — mutating `.blob` persists).
    bucket = client.bucket("ignored")
    real_blob = bucket.blob

    def boom_blob(_n):
        class _B:
            def upload_from_string(self, *a, **kw):
                raise RuntimeError("simulated upload failure")
        return _B()

    bucket.blob = boom_blob
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=uploaded,
        initial_seed_fp_by_db={}, client=client,
    )
    # Failure was best-effort: didn't raise, didn't add to uploaded_dbs.
    assert uploaded == set(), (
        f"db_a must NOT be in uploaded_dbs after failure: {uploaded}"
    )

    # Second attempt: restore real blob factory. The db must still be eligible.
    bucket.blob = real_blob
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=uploaded,
        initial_seed_fp_by_db={}, client=client,
    )
    assert "db_a" in uploaded
    assert f"runs/{RUN_ID}/post_run/slayer_models_otf/host-1/db_a/_upload_complete" in store


def test_upload_otf_reference_delta_noop_when_query_mode_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """raw mode never produces an OTF reference — the helper must no-op
    without touching paths or making any GCS call."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "fp"})

    raw_cfg = {"query_mode": "raw", "slayer_setup": "pre-encoded",
               "framework": "pydantic_ai"}
    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=raw_cfg,
        shard="host-1", uploaded_dbs=set(),
        initial_seed_fp_by_db={}, client=client,
    )
    assert store == {}


def test_upload_otf_reference_delta_noop_for_non_otf_encode_slayer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """CodeRabbit r2 — the helper must no-op for any slayer combo OTHER than
    `otf_encode + on-the-fly`. Under recursive+on-the-fly or pre-encoded
    slayer, `initial_seed_fp_by_db` is `{}`, so stale `slayer_models_otf/`
    content on a shared worker FS would pass the fp-mismatch check and
    pollute the laptop warm cache."""
    client, store = fake_gcs_bucket
    _make_otf_ref_root(monkeypatch, tmp_path, {"db_a": "stale-cloud-fp"})

    for cfg in (
        # recursive + on-the-fly
        {"query_mode": "slayer", "slayer_setup": "on-the-fly",
         "framework": "pydantic_ai_recursive"},
        # pre-encoded (any framework)
        {"query_mode": "slayer", "slayer_setup": "pre-encoded",
         "framework": "pydantic_ai_otf_encode"},
        {"query_mode": "slayer", "slayer_setup": "pre-encoded",
         "framework": "pydantic_ai_recursive"},
    ):
        upload_back.upload_otf_reference_delta(
            run_id=RUN_ID, cfg=cfg,
            shard="host-1", uploaded_dbs=set(),
            initial_seed_fp_by_db={}, client=client,
        )
    assert store == {}, (
        f"OTF reference delta must not upload from any non-otf_encode slayer "
        f"combo (stale content would pollute the warm cache); saw {sorted(store)}"
    )


def test_upload_otf_reference_delta_skips_incomplete_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_gcs_bucket,
):
    """A `<db>/` without `_reference_fp.txt` is an in-progress build (or
    leftover scratch) — never upload, completeness invariant requires the
    marker."""
    from bird_interact_agents import paths as _paths
    client, store = fake_gcs_bucket
    root = tmp_path / "slayer_models_otf"
    (root / "db_a" / "models").mkdir(parents=True)
    (root / "db_a" / "models" / "x.yaml").write_text("name: x\n")
    # NO _reference_fp.txt
    monkeypatch.setattr(_paths, "slayer_models_otf_root", lambda: root)

    upload_back.upload_otf_reference_delta(
        run_id=RUN_ID, cfg=_cfg_for_otf_encode(),
        shard="host-1", uploaded_dbs=set(),
        initial_seed_fp_by_db={}, client=client,
    )
    assert store == {}
