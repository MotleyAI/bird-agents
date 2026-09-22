"""Split-coverage check.

Without ``--allow-partial``, the present instance set MUST equal the
benchmark's full instance set. Extra instances (typo'd instance_ids
that aren't in the benchmark) are always a hard error — they signal a
selection-file mistake, not a partial run.
"""

from __future__ import annotations

import json

from bird_interact_agents import paths


class IncompleteCoverageError(ValueError):
    pass


class UnknownInstanceError(ValueError):
    pass


def load_benchmark_instance_ids(benchmark: str) -> set[str]:
    """Return every ``instance_id`` declared in the benchmark's task data
    JSONL."""
    path = paths.benchmark_data_file(benchmark)
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj.get("instance_id")
            if iid:
                ids.add(str(iid))
    return ids


def assert_coverage_ok(
    *,
    benchmark: str,
    present_instance_ids: set[str],
    allow_partial: bool,
) -> None:
    """Raise if the present set isn't equal to the full benchmark split.

    * Missing instances → ``IncompleteCoverageError`` unless ``allow_partial``.
    * Extra instances → ``UnknownInstanceError`` regardless.
    """
    full = load_benchmark_instance_ids(benchmark)
    extra = sorted(present_instance_ids - full)
    if extra:
        raise UnknownInstanceError(
            f"instance_id(s) not in benchmark {benchmark!r}: "
            f"{', '.join(extra)}. Check for typos in your selection file."
        )
    missing = sorted(full - present_instance_ids)
    if missing and not allow_partial:
        raise IncompleteCoverageError(
            f"benchmark {benchmark!r} has {len(full)} instances; "
            f"submission covers {len(present_instance_ids)}. Missing: "
            f"{', '.join(missing)}. Pass --allow-partial to override."
        )
