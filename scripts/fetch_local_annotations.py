"""Thin CLI wrapper around ``bird_interact_agents.local_annotations``.

DEV-1638: the sync logic moved into the package so both the local
``bird-interact`` run and the cloud ``submit`` pre-build call the SAME
``sync_annotations``. This script is retained only as a standalone admin entry
point (e.g. to pre-populate a checkout's annotations before an image build).

Usage:
    uv run python scripts/fetch_local_annotations.py \
        --benchmark livesqlbench-large --instance-ids id1,id2,...
    uv run python scripts/fetch_local_annotations.py \
        --benchmark livesqlbench-large            # all tasks in the benchmark
"""
from __future__ import annotations

from bird_interact_agents.local_annotations import main

if __name__ == "__main__":
    raise SystemExit(main())
