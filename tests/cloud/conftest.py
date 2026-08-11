"""Shared fixtures for tests/cloud/.

The cloud package itself is built behind an optional `cloud` extra (ray,
google-cloud-storage, jinja2). When those aren't installed locally these
tests collect-error on import, which is the right behaviour: a developer
who hasn't installed the cloud extra shouldn't see cloud-test failures
masquerading as something else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_repo_root(tmp_path: Path) -> Path:
    """A minimal repo layout that the image-hash helpers can walk.

    Lays down:
      <tmp>/uv.lock
      <tmp>/pyproject.toml
      <tmp>/src/bird_interact_agents/__init__.py
      <tmp>/src/bird_interact_agents/run.py
      <tmp>/slayer_models/db_a/model.yaml
      <tmp>/audited_gold/db_a/db_a_audited.jsonl
      <tmp>/Dockerfile.cloud   (with sentinel-delimited DATA/CODE sections)
    """
    (tmp_path / "src" / "bird_interact_agents").mkdir(parents=True)
    (tmp_path / "src" / "bird_interact_agents" / "__init__.py").write_text(
        "VERSION = '0.1.0'\n"
    )
    (tmp_path / "src" / "bird_interact_agents" / "run.py").write_text(
        "def main(): pass\n"
    )
    (tmp_path / "uv.lock").write_text("# lock file\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    (tmp_path / "slayer_models" / "db_a").mkdir(parents=True)
    (tmp_path / "slayer_models" / "db_a" / "model.yaml").write_text(
        "models: []\n"
    )

    (tmp_path / "audited_gold" / "db_a").mkdir(parents=True)
    (tmp_path / "audited_gold" / "db_a" / "db_a_audited.jsonl").write_text(
        '{"instance_id":"db_a_1"}\n'
    )

    # DEV-1468: slayer_models/ is uploaded to GCS at submit, no longer baked.
    # De-bake (this chunk): the benchmark dataset is no longer baked either
    # (delivered via GCS), so the DATA section no longer COPYs mini-interact —
    # only the code-like audited_gold/ corrections stay baked.
    (tmp_path / "Dockerfile.cloud").write_text(
        "FROM python:3.11-slim\n"
        "RUN pip install uv\n"
        "# DATA-LAYERS\n"
        "COPY audited_gold/  /app/bird-interact-agents/audited_gold/\n"
        "# CODE-LAYERS\n"
        "COPY pyproject.toml uv.lock /app/bird-interact-agents/\n"
        "RUN uv sync --extra all --extra dev\n"
        "COPY src/ /app/bird-interact-agents/src/\n"
    )
    return tmp_path


@pytest.fixture
def fake_mini_interact(tmp_path: Path) -> Path:
    """A minimal mini-interact sibling data dir."""
    root = tmp_path / "mini-interact"
    root.mkdir()
    (root / "mini_interact.jsonl").write_text(
        '{"instance_id":"db_a_1","selected_database":"db_a"}\n'
    )
    (root / "db_a").mkdir()
    (root / "db_a" / "schema.sql").write_text("CREATE TABLE t (x INT);\n")
    return root


@pytest.fixture
def make_git(monkeypatch: pytest.MonkeyPatch):
    """Stub `subprocess.run` for `git status` / `git rev-parse` calls.

    Returns a callable so each test can configure the canned `git status`
    output and any other git invocations the function-under-test needs.
    """

    def _set(*, status_porcelain: str = "", commit: str = "deadbeef"):
        import subprocess as _sp

        real_run = _sp.run

        def fake_run(argv, *args, **kwargs):
            if argv[:2] == ["git", "status"]:
                kwargs.setdefault("capture_output", False)
                return _sp.CompletedProcess(
                    argv, 0, stdout=status_porcelain, stderr=""
                )
            if argv[:3] == ["git", "rev-parse", "HEAD"]:
                return _sp.CompletedProcess(
                    argv, 0, stdout=f"{commit}\n", stderr=""
                )
            return real_run(argv, *args, **kwargs)

        monkeypatch.setattr(_sp, "run", fake_run)

    return _set


@pytest.fixture
def fake_gcs_bucket():
    """In-memory stand-in for the subset of google.cloud.storage we use.

    Returns a tuple `(client, store)` where `store` is a plain
    `dict[str, bytes]` keyed by full blob path. The `client` quacks like
    `google.cloud.storage.Client` for the methods the cloud package
    actually calls (`bucket`, `bucket().blob`, `blob.upload_from_string`,
    `blob.download_as_bytes`, `bucket.list_blobs`).
    """
    from pathlib import Path
    from types import SimpleNamespace

    store: dict[str, bytes] = {}
    # DEV-1653: capture every upload_from_filename call as
    # (name, timeout, retry) so tests can assert which blobs were (not) sent
    # and with which timeout/retry — skip-existing must NOT re-send matches.
    upload_calls: list = []

    def _blob(name: str):
        def _upload(data, content_type=None, **_kw):
            store[name] = data if isinstance(data, bytes) else data.encode()

        def _upload_from_filename(filename, *, timeout=None, retry=None,
                                  content_type=None, **_kw):
            # DEV-1653: real Blob.upload_from_filename streams a file from disk.
            # The fake reads it eagerly and records the call for assertions.
            upload_calls.append((name, timeout, retry))
            store[name] = Path(filename).read_bytes()

        def _download():
            if name not in store:
                raise KeyError(name)
            return store[name]

        return SimpleNamespace(
            name=name,
            # DEV-1653: real listed blobs carry `.size` (object byte length);
            # None for a blob that isn't in the store yet.
            size=len(store[name]) if name in store else None,
            upload_from_string=_upload,
            upload_from_filename=_upload_from_filename,
            download_as_bytes=_download,
            exists=lambda: name in store,
        )

    def _list_blobs(*, prefix: str = "", **_kw):
        # Match real google.cloud.storage.Client.list_blobs(prefix=...):
        # returned objects have `.name`, `.size`, and `.download_as_bytes()`.
        return [_blob(k) for k in sorted(store) if k.startswith(prefix)]

    bucket = SimpleNamespace(
        blob=_blob, list_blobs=_list_blobs, name="motley-team-birdbench",
        upload_calls=upload_calls,
    )
    client = SimpleNamespace(bucket=lambda _name: bucket)
    return client, store


@pytest.fixture
def sample_task_result_row() -> dict:
    """A canonical task-result row matching `TaskResultRow` from
    bird_interact_agents.results_db."""
    return {
        "run_id": "20260521T1422-pydanticai-raw-a1b2c3",
        "framework": "pydantic_ai",
        "mode": "c-interact",
        "query_mode": "raw",
        "instance_id": "db_a_1",
        "database": "db_a",
        "started_at": 1747825320.0,
        "duration_s": 42.5,
        "phase1_passed": True,
        "phase2_passed": True,
        "total_reward": 1.0,
        "submitted_sql": "SELECT 1;",
        "submitted_query": None,
        "ground_truth_sql": "SELECT 1;",
        "error": None,
        "usage_json": "{}",
        "user_query": "Find 1.",
        "submission_status": "submitted_correct",
        "phase1_observation": "OK",
        "phase2_observation": "OK",
        "predicted_result_json": "[[1]]",
        "gold_result_json": "[[1]]",
        "n_agent_turns": 3,
        "tool_call_stats_json": None,
    }


def write_row_object(store: dict[str, bytes], run_id: str, iid: str,
                     attempt: int, row: dict) -> None:
    """Helper: place a row directly into the fake-GCS store."""
    path = f"runs/{run_id}/rows/{iid}/attempt-{attempt}.json"
    store[path] = json.dumps(row).encode()
