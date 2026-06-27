"""T9–T12: attempt-aware GCS sink, SDK-only (no gsutil)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from bird_interact_agents.cloud import gcs  # noqa: E402


RUN_ID = "20260521T1422-pydanticai-raw-a1b2c3"


# ---------------------------------------------------------------------------
# T9 — round-trip write_row + list_attempts + read_row.
# ---------------------------------------------------------------------------


def test_write_list_read_round_trip(fake_gcs_bucket, sample_task_result_row):
    client, store = fake_gcs_bucket
    gcs.write_row(RUN_ID, "db_a_1", attempt=1,
                  row=sample_task_result_row, client=client)
    gcs.write_row(RUN_ID, "db_a_2", attempt=1,
                  row={**sample_task_result_row, "instance_id": "db_a_2"},
                  client=client)

    attempts = gcs.list_attempts(RUN_ID, client=client)
    assert attempts == {"db_a_1": [1], "db_a_2": [1]}

    back = gcs.read_row(RUN_ID, "db_a_1", attempt=1, client=client)
    assert back["instance_id"] == "db_a_1"
    assert back["phase1_passed"] is True


# ---------------------------------------------------------------------------
# T10 — concurrent writes don't lose objects.
# ---------------------------------------------------------------------------


def test_concurrent_writes_preserve_all(fake_gcs_bucket,
                                         sample_task_result_row):
    """Mix attempts (1 and 2 for some iids) so the test exercises the
    real-world resubmit pattern, not only the attempt=1 happy path."""
    client, _store = fake_gcs_bucket
    iids = [f"db_a_{i}" for i in range(32)]

    # Iids 0–7 also get an attempt=2 row written concurrently.
    second_attempt_ids = iids[:8]

    async def _write(iid: str, attempt: int):
        row = {**sample_task_result_row, "instance_id": iid}
        gcs.write_row(RUN_ID, iid, attempt=attempt, row=row, client=client)

    async def _drive() -> None:
        jobs = [_write(i, 1) for i in iids] + [
            _write(i, 2) for i in second_attempt_ids
        ]
        await asyncio.gather(*jobs)

    asyncio.run(_drive())
    attempts = gcs.list_attempts(RUN_ID, client=client)
    assert set(attempts) == set(iids)
    for iid in iids:
        expected = [1, 2] if iid in second_attempt_ids else [1]
        assert attempts[iid] == expected, f"iid {iid}: got {attempts[iid]}"


# ---------------------------------------------------------------------------
# T11 — attempt-aware naming: second attempt sits next to first.
# ---------------------------------------------------------------------------


def test_attempt_naming_keeps_history(fake_gcs_bucket, sample_task_result_row):
    client, store = fake_gcs_bucket
    err_row = {
        **sample_task_result_row,
        "error": "boom",
        "phase1_passed": False,
        "phase2_passed": False,
    }
    gcs.write_row(RUN_ID, "db_a_1", attempt=1, row=err_row, client=client)
    ok_row = {**sample_task_result_row, "error": None}
    gcs.write_row(RUN_ID, "db_a_1", attempt=2, row=ok_row, client=client)

    attempts = gcs.list_attempts(RUN_ID, client=client)
    assert attempts == {"db_a_1": [1, 2]}
    assert gcs.latest_attempt(RUN_ID, "db_a_1", client=client) == 2

    a1 = gcs.read_row(RUN_ID, "db_a_1", attempt=1, client=client)
    a2 = gcs.read_row(RUN_ID, "db_a_1", attempt=2, client=client)
    assert a1["error"] == "boom"
    assert a2["error"] is None


# ---------------------------------------------------------------------------
# T12 — SDK-only: concurrent_download_prefix uses the SDK, NOT gsutil.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CR#6 — bucket lifecycle is enforced on EXISTING buckets, not just on create.
# ---------------------------------------------------------------------------


def test_ensure_bucket_enforces_lifecycle_on_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the bucket already exists without the 30-day `runs/` rule,
    ensure_bucket must patch it onto the existing bucket — not skip
    because the bucket isn't being created."""
    patched: list[list] = []
    set_iam: list[list] = []

    class _FakePolicy:
        def __init__(self):
            self.bindings: list[dict] = []

    class _FakeBucket:
        def __init__(self):
            self.lifecycle_rules: list = []  # existing bucket, no rules
            self._policy = _FakePolicy()

        def patch(self):
            patched.append(list(self.lifecycle_rules))

        def get_iam_policy(self, requested_policy_version=3):
            return self._policy

        def set_iam_policy(self, p):
            set_iam.append(list(p.bindings))

    class _FakeClient:
        def __init__(self):
            self.bucket_obj = _FakeBucket()

        def get_bucket(self, _name):
            return self.bucket_obj

        def create_bucket(self, *_args, **_kwargs):
            raise AssertionError("should not create — bucket exists")

    client = _FakeClient()
    gcs.ensure_bucket(
        worker_sa_email="bird-interact-runner@motley-team-475011.iam.gserviceaccount.com",
        client=client,
    )

    # Lifecycle was patched in even though the bucket already existed.
    assert patched, "ensure_bucket didn't patch existing bucket"
    assert any(
        rule.get("action", {}).get("type") == "Delete"
        and rule.get("condition", {}).get("age") == 30
        for rule in patched[-1]
    )
    # Worker SA's objectUser binding was applied to the IAM policy.
    assert set_iam, "no IAM policy applied"


def test_ensure_bucket_idempotent_when_lifecycle_already_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desired = {
        "action": {"type": "Delete"},
        "condition": {"age": 30, "matchesPrefix": ["runs/"]},
    }

    class _FakePolicy:
        def __init__(self):
            self.bindings: list[dict] = [
                {
                    "role": "roles/storage.objectUser",
                    "members": {
                        "serviceAccount:bird-interact-runner@motley-team-475011.iam.gserviceaccount.com",
                    },
                },
            ]

    patched: list = []
    set_iam: list = []

    class _FakeBucket:
        def __init__(self):
            self.lifecycle_rules: list = [desired]
            self._policy = _FakePolicy()

        def patch(self):
            patched.append("patched")

        def get_iam_policy(self, requested_policy_version=3):
            return self._policy

        def set_iam_policy(self, p):
            set_iam.append("set")

    class _FakeClient:
        def get_bucket(self, _name):
            return _FakeBucket()

        def create_bucket(self, *_a, **_k):
            raise AssertionError

    gcs.ensure_bucket(
        worker_sa_email="bird-interact-runner@motley-team-475011.iam.gserviceaccount.com",
        client=_FakeClient(),
    )
    # Nothing was changed — already converged.
    assert not patched
    assert not set_iam


def test_concurrent_download_prefix_no_gsutil(
    monkeypatch: pytest.MonkeyPatch,
    fake_gcs_bucket,
    sample_task_result_row,
    tmp_path: Path,
) -> None:
    client, store = fake_gcs_bucket
    # Seed two row objects + a manifest.
    for iid in ("db_a_1", "db_a_2"):
        path = f"runs/{RUN_ID}/rows/{iid}/attempt-1.json"
        store[path] = json.dumps(
            {**sample_task_result_row, "instance_id": iid}
        ).encode()
    store[f"runs/{RUN_ID}/manifest.json"] = json.dumps({"run_id": RUN_ID}).encode()

    # Trip-wire: if anyone tries to shell out to gsutil, we fail loudly.
    def boom(*a, **kw):
        raise AssertionError("gcs.py shelled out — expected SDK-only")

    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("subprocess.check_call", boom)
    monkeypatch.setattr("subprocess.check_output", boom)

    gcs.concurrent_download_prefix(
        RUN_ID, dest=tmp_path, max_workers=4, client=client
    )

    # Files should now exist locally with their original bytes.
    assert (tmp_path / "rows" / "db_a_1" / "attempt-1.json").exists()
    assert (tmp_path / "rows" / "db_a_2" / "attempt-1.json").exists()
    assert (tmp_path / "manifest.json").exists()
    back = json.loads(
        (tmp_path / "rows" / "db_a_1" / "attempt-1.json").read_text()
    )
    assert back["instance_id"] == "db_a_1"


# ---------------------------------------------------------------------------
# DEV-1468 — upload_dir_prefix + download_prefix (slayer-setup delivery).
# ---------------------------------------------------------------------------


def _make_artifact_dir(root: Path) -> dict[str, bytes]:
    """Lay down a believable per-DB SLayer artifact dir (markers, nested
    YAML, and a BINARY embeddings.db). Returns {relpath: bytes} for asserts."""
    files = {
        "_cache_fp.txt": b"deadbeefcafe1234",
        "_kb_rows.json": b'[{"id": 1}]',
        "memories.yaml": b"- id: x\n",
        "datasources/db.yaml": b"name: db\n",
        "models/db/households.yaml": b"name: households\n",
        # Binary blob with NUL bytes + a sqlite-ish header — must survive intact.
        "embeddings.db": b"SQLite format 3\x00\x01\x02\x03\xff\xfe",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return files


def test_upload_dir_prefix_round_trip(fake_gcs_bucket, tmp_path: Path) -> None:
    """upload_dir_prefix ships every file (incl. dot-free markers + binary
    embeddings.db) under <prefix>/<relpath>; download_prefix restores them
    byte-for-byte under dest/<relpath>."""
    client, store = fake_gcs_bucket
    src = tmp_path / "slayer_otf_cache" / "db"
    src.mkdir(parents=True)
    expected = _make_artifact_dir(src)

    prefix = f"runs/{RUN_ID}/slayer_setup/slayer_otf_cache/db"
    gcs.upload_dir_prefix(src, prefix, client=client)

    # Every file landed under the prefix with the right relative key.
    for rel, data in expected.items():
        assert store[f"{prefix}/{rel}"] == data

    dest = tmp_path / "down" / "db"
    gcs.download_prefix(prefix, dest, client=client)
    for rel, data in expected.items():
        assert (dest / rel).read_bytes() == data, f"{rel} round-trip mismatch"
    # Binary integrity (NUL bytes preserved).
    assert (dest / "embeddings.db").read_bytes() == expected["embeddings.db"]


def test_upload_dir_prefix_tolerates_file_vanishing_mid_walk(
    fake_gcs_bucket, tmp_path: Path, monkeypatch,
) -> None:
    """A file enumerated by the dir-walk can disappear before it is read — e.g.
    a transient SQLite WAL sidecar removed when its DB closes. The upload must
    skip the vanished file, not abort the whole transfer with FileNotFoundError."""
    client, store = fake_gcs_bucket
    src = tmp_path / "ds"
    src.mkdir()
    (src / "keep.txt").write_text("keep\n")
    vanishing = src / "db.sqlite-shm"
    vanishing.write_bytes(b"\x00")

    real_read = Path.read_bytes

    def _flaky_read(self):
        if self.name == "db.sqlite-shm":
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _flaky_read)
    gcs.upload_dir_prefix(src, "runs/r/ds", client=client)
    # The surviving file uploaded; the vanished one was skipped (no crash).
    assert store["runs/r/ds/keep.txt"] == b"keep\n"
    assert "runs/r/ds/db.sqlite-shm" not in store


def test_upload_dir_prefix_preserves_nested_rel_paths(
    fake_gcs_bucket, tmp_path: Path,
) -> None:
    client, store = fake_gcs_bucket
    src = tmp_path / "ref" / "db"
    (src / "models" / "db").mkdir(parents=True)
    (src / "models" / "db" / "a.yaml").write_text("a\n")
    (src / "top.txt").write_text("t\n")

    prefix = "runs/r/slayer_setup/slayer_models_otf/db"
    gcs.upload_dir_prefix(src, prefix, client=client)
    assert f"{prefix}/models/db/a.yaml" in store
    assert f"{prefix}/top.txt" in store
    # No directory entries, only files.
    assert all(not k.endswith("/") for k in store)


def test_download_prefix_strips_prefix(fake_gcs_bucket, tmp_path: Path) -> None:
    """download_prefix writes each blob to dest/<path-after-prefix>, dropping
    the GCS prefix itself."""
    client, store = fake_gcs_bucket
    prefix = "runs/r/slayer_setup/slayer_models/db"
    store[f"{prefix}/households.yaml"] = b"name: households\n"
    store[f"{prefix}/sub/x.yaml"] = b"x\n"
    # A sibling blob OUTSIDE the prefix must not be downloaded.
    store["runs/r/slayer_setup/slayer_models/other_db/y.yaml"] = b"y\n"

    dest = tmp_path / "out"
    gcs.download_prefix(prefix, dest, client=client)
    assert (dest / "households.yaml").read_bytes() == b"name: households\n"
    assert (dest / "sub" / "x.yaml").read_bytes() == b"x\n"
    assert not (dest / "y.yaml").exists()
    assert not (dest.parent / "other_db").exists()


def test_concurrent_download_prefix_is_a_thin_wrapper(
    fake_gcs_bucket, tmp_path: Path,
) -> None:
    """concurrent_download_prefix(run_id, dest) must delegate to
    download_prefix(f'runs/<run_id>/', dest) so the two share one code path."""
    client, store = fake_gcs_bucket
    store[f"runs/{RUN_ID}/manifest.json"] = b'{"run_id": "x"}'
    store[f"runs/{RUN_ID}/rows/db_a_1/attempt-1.json"] = b'{"instance_id": "db_a_1"}'

    dest = tmp_path / "dl"
    gcs.concurrent_download_prefix(RUN_ID, dest, client=client)
    assert (dest / "manifest.json").exists()
    assert (dest / "rows" / "db_a_1" / "attempt-1.json").exists()


def test_download_prefix_empty_is_noop(fake_gcs_bucket, tmp_path: Path) -> None:
    """No blobs under the prefix → nothing downloaded (no crash)."""
    client, _store = fake_gcs_bucket
    dest = tmp_path / "out"
    gcs.download_prefix("runs/does-not-exist/", dest, client=client)
    assert not dest.exists() or not any(dest.iterdir())


def _fake_bucket_with_one_racing_blob(
    racing_name: str,
    *,
    prefix_root: str = "runs/r",
    racing_exc: Exception | None = None,
):
    """Helper: build an in-memory bucket where ``racing_name`` raises on
    download but every other listed blob succeeds. ``racing_exc``
    defaults to the SDK's ``google.api_core.exceptions.NotFound`` —
    that's what the production code path actually raises when the
    generation-pinned URL hits a rewritten blob; tests must mirror it
    so the narrowed ``except NotFound`` is exercised."""
    from types import SimpleNamespace

    if racing_exc is None:
        from google.api_core.exceptions import NotFound
        racing_exc = NotFound("simulated 404 (object replaced)")

    payloads = {
        f"{prefix_root}/manifest.json": b"manifest\n",
        racing_name: b"will race",
        f"{prefix_root}/rows/db_a/attempt-1.json": b"row\n",
    }

    def _blob(name: str):
        def _download():
            if name == racing_name:
                raise racing_exc
            return payloads[name]

        return SimpleNamespace(name=name, download_as_bytes=_download)

    def _list_blobs(*, prefix: str = "", **_kw):
        return [_blob(k) for k in sorted(payloads) if k.startswith(prefix)]

    bucket = SimpleNamespace(blob=_blob, list_blobs=_list_blobs,
                              name="motley-team-birdbench")
    return SimpleNamespace(bucket=lambda _name: bucket)


def test_download_prefix_skip_missing_blobs_true_keeps_going(tmp_path: Path) -> None:
    """Regression test for the fetch-vs-live-cluster race: when
    ``skip_missing_blobs=True`` is set (the canonical caller is
    ``driver.fetch`` via ``concurrent_download_prefix``), one blob's
    download raising must NOT crash the entire fetch — otherwise the
    auto-kill step at the end of fetch never runs and the cluster leaks."""
    client = _fake_bucket_with_one_racing_blob("runs/r/status.json")
    dest = tmp_path / "out"
    gcs.download_prefix("runs/r/", dest, client=client, skip_missing_blobs=True)
    # Successful blobs still landed on disk.
    assert (dest / "manifest.json").read_bytes() == b"manifest\n"
    assert (dest / "rows" / "db_a" / "attempt-1.json").read_bytes() == b"row\n"
    # The racing blob was skipped, not corruptly written.
    assert not (dest / "status.json").exists()


def test_download_prefix_default_propagates_failures(tmp_path: Path) -> None:
    """Default ``skip_missing_blobs=False`` MUST propagate per-blob failures
    so strict callers (slayer-setup at ``ray_app.py:303,361``, benchmark-data
    at ``benchmark_data.py:181``) don't silently land a partial cache and
    let the cluster proceed with bad inputs. The fetch-side tolerance is
    opt-in only."""
    from google.api_core.exceptions import NotFound

    client = _fake_bucket_with_one_racing_blob("runs/r/status.json")
    dest = tmp_path / "out"
    with pytest.raises(NotFound):
        gcs.download_prefix("runs/r/", dest, client=client)


def test_download_prefix_skip_missing_propagates_non_notfound(tmp_path: Path) -> None:
    """Regression for the Codex follow-up: even with ``skip_missing_blobs=True``,
    non-``NotFound`` errors (transient network / 5xx / auth) MUST propagate.
    Swallowing them would let ``driver.fetch`` log+ignore a transient failure
    and then auto-kill the cluster — losing the only source of the missing
    rows. The narrowed ``except NotFound`` keeps the race-tolerance scoped
    to the actual race-class error."""
    from google.api_core.exceptions import ServiceUnavailable

    transient = ServiceUnavailable("simulated 503")
    client = _fake_bucket_with_one_racing_blob(
        "runs/r/status.json", racing_exc=transient,
    )
    dest = tmp_path / "out"
    with pytest.raises(ServiceUnavailable):
        gcs.download_prefix(
            "runs/r/", dest, client=client, skip_missing_blobs=True,
        )


def test_concurrent_download_prefix_skips_by_default(tmp_path: Path) -> None:
    """``concurrent_download_prefix`` is the fetch path — it must default
    ``skip_missing_blobs=True`` so the documented race-tolerance is the
    out-of-box behaviour for ``driver.fetch`` callers."""
    client = _fake_bucket_with_one_racing_blob(
        "runs/test-id/status.json", prefix_root="runs/test-id",
    )
    dest = tmp_path / "dl"
    # No explicit skip flag — the wrapper's default must already be True.
    gcs.concurrent_download_prefix("test-id", dest, client=client)
    assert (dest / "manifest.json").read_bytes() == b"manifest\n"


@pytest.mark.parametrize(
    "slayer_setup, framework, expected",
    [
        ("pre-encoded", "pydantic_ai_recursive", "slayer_models"),
        ("pre-encoded", "claude_sdk", "slayer_models"),
        ("on-the-fly", "pydantic_ai_recursive", "slayer_otf_cache"),
        ("on-the-fly", "pydantic_ai_otf_encode", "slayer_models_otf"),
    ],
)
def test_slayer_artifact_name(slayer_setup, framework, expected) -> None:
    """The combo → GCS artifact-dir-name mapping is the single source of truth
    shared by the driver (upload) and the actor (download)."""
    assert gcs.slayer_artifact_name(slayer_setup, framework) == expected
