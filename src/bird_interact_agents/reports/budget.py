"""Budget calculation + replay.

* ``calculate_total_budget(task_data, patience)`` mirrors
  ``harness.calculate_budget(task_data, patience, mode="a-interact")`` =
  ``6 + 2*ambiguity_count + 2*patience``. We re-implement here so the
  reports package has no runtime dependency on the harness import chain
  (which pulls in heavy adapters); the parity is pinned by tests.
* ``replay_remaining_budget(total, costs)`` returns the per-step
  ``remaining_budget`` (clipped at 0).
* ``lookup_task_data(benchmark, instance_id)`` joins the benchmark's
  task data file on ``instance_id``.
"""

from __future__ import annotations


def _ambiguity_count(task_data: dict) -> int:
    n = 0
    user_query_ambiguity = task_data.get("user_query_ambiguity", {}) or {}
    if "critical_ambiguity" in user_query_ambiguity:
        n += len(user_query_ambiguity["critical_ambiguity"])
    kb_amb = task_data.get("knowledge_ambiguity") or []
    n += len(kb_amb)
    return n


def calculate_total_budget(task_data: dict, *, patience: int) -> float:
    return 6.0 + 2.0 * _ambiguity_count(task_data) + 2.0 * patience


def replay_remaining_budget(
    *, total_budget: float, action_costs: list[float]
) -> list[float]:
    cum = 0.0
    out: list[float] = []
    for c in action_costs:
        cum += c
        out.append(max(0.0, total_budget - cum))
    return out


# ---------------------------------------------------------------------------
# task_data lookup
# ---------------------------------------------------------------------------

# Cache by (benchmark, instance_id) so a 270-instance run doesn't re-parse
# the gold JSONL 270 times.
_TASK_DATA_CACHE: dict[tuple[str, str], dict] = {}


def lookup_task_data(benchmark: str, instance_id: str) -> dict:
    """Load the benchmark's task data file and return the row whose
    ``instance_id`` matches. Raises ``KeyError`` when not found.

    Uses ``bird_interact_agents.benchmark`` resolution; data lives at
    ``paths.benchmark_data_file(benchmark)``.
    """
    key = (benchmark, instance_id)
    if key in _TASK_DATA_CACHE:
        return _TASK_DATA_CACHE[key]

    import json

    from bird_interact_agents import paths

    path = paths.benchmark_data_file(benchmark)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("instance_id") == instance_id:
                _TASK_DATA_CACHE[key] = row
                return row
    raise KeyError(
        f"instance_id={instance_id!r} not found in {path} (benchmark={benchmark!r})"
    )
