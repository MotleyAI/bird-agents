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
