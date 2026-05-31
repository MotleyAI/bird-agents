"""DEV-1515: grep-sweep regression — no production code emits the legacy
``phase1_passed_audited`` / ``phase1_passed_original`` raw fields.

The fields are replaced by per-row ``submission_annotation.json`` files
produced by tolerant_grader. Every place that previously wrote the raw
bools must now reference the new path instead.

This test grep-scans the source tree at runtime so adding a new agent
that re-introduces the legacy fields trips immediately.
"""
from __future__ import annotations

import re
from pathlib import Path

import bird_interact_agents


def _bird_src_root() -> Path:
    return Path(bird_interact_agents.__file__).resolve().parent


_LEGACY_PATTERN = re.compile(
    r"\bphase1_passed_(?:audited|original)\b"
)

# Files allowed to mention the legacy fields purely in a comment / changelog
# context (no executable use). Keep this list intentionally tight; if a
# production path needs an exception, the test should be the gate.
_ALLOWED_FILES_WITH_MENTION: set[str] = set()
# Future comment-only entries (e.g. a CHANGELOG note in code) belong here.


def test_no_production_code_references_legacy_dual_eval_fields():
    src_root = _bird_src_root()
    offenders: list[tuple[str, int, str]] = []
    for py in src_root.rglob("*.py"):
        if py.name == "__pycache__":
            continue
        rel = str(py.relative_to(src_root))
        if rel in _ALLOWED_FILES_WITH_MENTION:
            continue
        for i, line in enumerate(py.read_text().splitlines(), start=1):
            if _LEGACY_PATTERN.search(line):
                offenders.append((rel, i, line.strip()))
    assert not offenders, (
        "DEV-1515: the following lines still reference the legacy "
        "phase1_passed_audited / phase1_passed_original fields. "
        "Replace with submission_annotation.json plumbing:\n"
        + "\n".join(f"  {f}:{i}: {ln}" for f, i, ln in offenders)
    )


def test_results_db_schema_does_not_define_legacy_columns():
    """Explicit catch for the SQLite CREATE TABLE / dataclass row."""
    from bird_interact_agents import results_db

    src = Path(results_db.__file__).read_text()
    assert "phase1_passed_audited" not in src
    assert "phase1_passed_original" not in src


def test_agents_submit_helpers_drop_legacy_kwargs():
    """The shared submit-helper signatures must NOT accept the legacy
    kwargs any more — callers should rely on the worker-side grader."""
    import inspect
    from bird_interact_agents.agents import _submit

    for name in (
        "_diagnostic_payload",
        "submit_raw_sql",
        "submit_slayer_query",
    ):
        fn = getattr(_submit, name, None)
        if fn is None:
            continue
        sig = inspect.signature(fn)
        for forbidden in ("phase1_passed_audited", "phase1_passed_original"):
            assert forbidden not in sig.parameters, (
                f"{name}: legacy kwarg {forbidden!r} must be removed"
            )


def test_build_aggregate_eval_does_not_query_legacy_columns():
    """``run.py::build_aggregate_eval`` previously SELECTed
    ``phase1_passed_audited`` / ``phase1_passed_original``. Whatever
    replacement helper it now provides must NOT name those columns."""
    import inspect
    from bird_interact_agents.run import build_aggregate_eval

    src = inspect.getsource(build_aggregate_eval)
    assert "phase1_passed_audited" not in src
    assert "phase1_passed_original" not in src
