"""DEV-1640: the persistence-backend seam shared by the cloud actor and
the local process pool.

The per-task body (:func:`ray_app._run_one_in_actor`) is identical for
cloud and local; only WHERE its artifacts land differs. That difference
is captured by a :class:`PersistenceStore`:

* :class:`GcsStore` — the cloud backend. Every method delegates to the
  module-level ``gcs.*`` / ``upload_back.*`` functions, so it is
  behaviourally identical to the old inline ``_gcs.*`` calls (and stays
  transparent to the existing monkeypatch-based cloud tests).
* :class:`LocalFsStore` — the local backend. Writes the SAME on-disk
  layout the cloud row blobs use (``rows/<iid>/attempt-<n>.json`` etc.)
  under the run's output dir, so :func:`cloud.collation.collate` can build
  ``results.db`` + ``eval.json`` from it unchanged. The DEV-1470
  upload-back triple is a GCS-merge-home concern and is a no-op locally
  (on-the-fly OTF artifacts already build in place under the shared
  ``paths.slayer_models_otf_root()``).

Both talk to module objects (not captured function references) so a test
monkeypatching ``gcs.write_row`` still intercepts ``GcsStore.write_row``.
"""

from __future__ import annotations

import abc
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

from bird_interact_agents.cloud import gcs as _gcs
from bird_interact_agents.cloud import upload_back as _upload_back


class PersistenceStore(abc.ABC):
    """Backend the per-task body writes its artifacts through."""

    @abc.abstractmethod
    def write_row(self, run_id: str, iid: str, attempt: int, row: dict) -> None:
        ...

    @abc.abstractmethod
    def write_submission_annotation(
        self, run_id: str, iid: str, annotation: dict,
    ) -> None:
        ...

    @abc.abstractmethod
    def write_log(self, run_id: str, iid: str, attempt: int, log_bytes: bytes) -> None:
        ...

    @abc.abstractmethod
    def write_partial_transcript(self, run_id: str, iid: str, data: str) -> None:
        ...

    @abc.abstractmethod
    def upload_back(
        self, run_id: str, cfg: dict, iid: str, attempt: int, *,
        task_start_ts: float,
        uploaded_dbs: set[str],
        initial_seed_fp_by_db: dict[str, str],
    ) -> None:
        ...


class GcsStore(PersistenceStore):
    """Cloud backend — delegates to the module-level ``gcs.*`` /
    ``upload_back.*`` functions using the actor's own GCS client."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def write_row(self, run_id, iid, attempt, row):
        _gcs.write_row(run_id, iid, attempt, row, client=self.client)

    def write_submission_annotation(self, run_id, iid, annotation):
        _gcs.write_submission_annotation(run_id, iid, annotation, client=self.client)

    def write_log(self, run_id, iid, attempt, log_bytes):
        _gcs.write_log(run_id, iid, attempt, log_bytes, client=self.client)

    def write_partial_transcript(self, run_id, iid, data):
        _gcs.write_partial_transcript(run_id, iid, data, client=self.client)

    def upload_back(self, run_id, cfg, iid, attempt, *, task_start_ts,
                    uploaded_dbs, initial_seed_fp_by_db):
        # DEV-1470 upload-back triple (moved out of _run_one_in_actor
        # verbatim). Ordering is load-bearing: debug -> setup-sessions ->
        # reference-delta.
        work_root = Path(tempfile.gettempdir()) / "bird_interact_slayer_otf"
        _upload_back.upload_per_task_debug(
            run_id=run_id, iid=iid, attempt=attempt,
            work_root=work_root, client=self.client,
        )
        _upload_back.upload_per_task_setup_sessions(
            run_id=run_id, iid=iid, attempt=attempt,
            setup_sessions_root=work_root / "_setup_sessions",
            task_start_ts=task_start_ts, client=self.client,
        )
        _upload_back.upload_otf_reference_delta(
            run_id=run_id, cfg=cfg,
            shard=f"{socket.gethostname()}-{os.getpid()}",
            uploaded_dbs=uploaded_dbs,
            initial_seed_fp_by_db=initial_seed_fp_by_db,
            client=self.client,
        )


class LocalFsStore(PersistenceStore):
    """Local backend — writes the GCS-mirroring row-blob layout under the
    run's output dir. ``upload_back`` is a no-op (see module docstring)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def _row_dir(self, iid: str) -> Path:
        d = self.run_dir / "rows" / iid
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _atomic_write_text(dest: Path, text: str) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(dest)

    @staticmethod
    def _atomic_write_bytes(dest: Path, data: bytes) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)

    def write_row(self, run_id, iid, attempt, row):
        self._atomic_write_text(
            self._row_dir(iid) / f"attempt-{attempt}.json",
            json.dumps(row, default=str),
        )

    def write_submission_annotation(self, run_id, iid, annotation):
        self._atomic_write_text(
            self._row_dir(iid) / "submission_annotation.json",
            json.dumps(annotation, default=str),
        )

    def write_log(self, run_id, iid, attempt, log_bytes):
        self._atomic_write_bytes(
            self._row_dir(iid) / f"task-{attempt}.log", log_bytes,
        )

    def write_partial_transcript(self, run_id, iid, data):
        # Full-snapshot semantics mirror gcs.write_partial_transcript (the
        # uploader passes the whole accumulated JSONL each time).
        self._atomic_write_text(
            self._row_dir(iid) / "partial_transcript.jsonl", data,
        )

    def upload_back(self, run_id, cfg, iid, attempt, *, task_start_ts,
                    uploaded_dbs, initial_seed_fp_by_db):
        # No-op locally: OTF artifacts already build in place under the
        # shared roots; the triple only exists to merge cloud shards home.
        return None
