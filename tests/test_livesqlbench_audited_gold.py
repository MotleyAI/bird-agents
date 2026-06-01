"""DEV-1510 contract tests for `audited_gold/livesqlbench_audited.jsonl`.

The file is the deliverable of this issue: one row per livesqlbench museum
SELECT task (museum_1..10), each row attaching to the canonical gold for
that instance. Tests here pin three layers of contract:

* **Schema** — every row parses, has the required keys + types, and is
  unique by `instance_id`. Coverage = exactly {museum_1..10}; the 5
  Management tasks (museum_M_1..5) are explicitly out of scope per the
  Linear issue.
* **Status-claim consistency** — `clean` rows have `audited_sol_sql ==
  original_sol_sql` and `changes == []`; `edited` / `unrecoverable` have
  `audited_sol_sql != original_sol_sql` and at least one change entry
  carrying `clause_kind`, `original`, `replacement`, `why_unjustified`,
  and a non-empty `justified_by` list.
* **Pinned decisions** — museum_7 and museum_9 are the issue's worked
  examples, locked in the spec:
  - museum_7 (`edited`): KB-canonical rewrite using a NULL-safe
    `CASE WHEN ... THEN 1 ELSE 0 END + ... >= 3` predicate; cites
    `kb:16` AND `external_knowledge:16` (16 is in the task's
    `external_knowledge` anchor list).
  - museum_9 (`clean`): gold's `ConditionAssessments→LightAndRadiationReadings`
    join chain IS justified by `column_meaning:ConditionAssessments|LightReadRefObserved`
    + the schema's single-hop declared FK. `reasoning_summary` documents
    the KB-alone underspec AND mentions both join chains by name.

Tests that need the real upstream data (`museum_kb.jsonl`,
`museum_column_meaning_base.json`, the gated gold sidecar) skip when the
livesqlbench data root is absent (CI doesn't ship the gitignored data).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import pytest

from bird_interact_agents import paths


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


EXPECTED_INSTANCE_IDS = {f"museum_{i}" for i in range(1, 11)}
"""The 10 museum SELECT tasks. museum_M_1..5 (management) are out of scope."""

REQUIRED_ROW_KEYS = {
    "instance_id",
    "selected_database",
    "benchmark",
    "audit_status",
    "original_sol_sql",
    "audited_sol_sql",
    "audited_sample_row",
    "changes",
    "reasoning_summary",
    "skill_version",
    "audited_at",
}

VALID_STATUSES = {"clean", "edited", "unrecoverable"}

REQUIRED_CHANGE_KEYS = {
    "clause_kind", "original", "replacement",
    "why_unjustified", "justified_by",
}

CITATION_REGEX = re.compile(
    r"\b("
    r"kb:\d+"
    r"|external_knowledge:\d+"
    r"|column_meaning:[^\s,\]\"']+\|[^\s,\]\"']+"
    r"|primitive"
    r"|dialect:[a-z_]+:[a-z_]+"
    r")\b"
)


def _audit_file_or_skip() -> Path:
    """Return the audited-gold file path, skipping the test if absent.

    The file ships in the repo (it's the deliverable of this PR), so
    absence in a normal checkout means the deliverable was rolled back
    or never authored — skip rather than fail so this test module can
    still run on a stub branch.
    """
    path = paths.audited_gold_root() / "livesqlbench_audited.jsonl"
    if not path.exists():
        pytest.skip(
            f"audited-gold deliverable not present: {path}. "
            "This test pins the file's contract; it has no meaning when "
            "the file is absent."
        )
    return path


def _iter_audit_rows() -> Iterator[dict]:
    path = _audit_file_or_skip()
    with path.open() as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover
                pytest.fail(f"audit row {n} is not valid JSON: {e}")


def _load_audit_rows() -> dict[str, dict]:
    """Return PRIMARY audit rows keyed by instance_id.

    Multi-variant pattern (DEV-1515): a task may carry one primary row plus
    N non-primary alternates sharing the same instance_id. Tests that pin a
    specific reading (e.g. museum_7 edited, museum_9 clean) should reach the
    primary; iterating tests (citation resolvability, status consistency)
    should use ``_iter_audit_rows`` to see every variant.
    """
    out: dict[str, dict] = {}
    for r in _iter_audit_rows():
        if r.get("primary", True):
            out[r["instance_id"]] = r
    return out


def _livesqlbench_data_or_skip() -> Path:
    """Return the livesqlbench data root, skipping the test if absent.

    Used by the citation-resolvability + original-sol-sql tests, which
    need the gitignored upstream data (`museum_kb.jsonl`,
    `museum_column_meaning_base.json`, the gated gold sidecar). CI runs
    without those files; local dev has them.
    """
    root = paths.livesqlbench_root()
    if not root.exists():
        pytest.skip(
            f"livesqlbench data root not present: {root}. "
            "Citation-resolvability + original-gold tests need the gitignored "
            "upstream data; skip cleanly in CI."
        )
    return root


def _load_museum_kb_ids() -> set[int]:
    root = _livesqlbench_data_or_skip()
    kb_path = root / "museum" / "museum_kb.jsonl"
    if not kb_path.exists():
        pytest.skip(f"museum_kb.jsonl missing at {kb_path}")
    ids: set[int] = set()
    with kb_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "id" in d:
                ids.add(int(d["id"]))
    return ids


def _load_museum_column_meaning_keys() -> set[str]:
    root = _livesqlbench_data_or_skip()
    cm_path = root / "museum" / "museum_column_meaning_base.json"
    if not cm_path.exists():
        pytest.skip(f"museum_column_meaning_base.json missing at {cm_path}")
    with cm_path.open() as f:
        return set(json.load(f).keys())


def _load_task_external_knowledge() -> dict[str, list[int]]:
    """Map `instance_id` -> task's `external_knowledge` list.

    Pulled from the GATED GOLD sidecar (where `external_knowledge` lives in
    the livesqlbench layout — the public data file ships an empty list and
    the sidecar merges the real one at task-load time, per
    `livesqlbench_loader`).
    """
    root = _livesqlbench_data_or_skip()
    gold = root / "livesqlbench_sqlite_gt_kg_testcases_0528.jsonl"
    if not gold.exists():
        pytest.skip(f"livesqlbench gold sidecar missing at {gold}")
    out: dict[str, list[int]] = {}
    with gold.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iid = d.get("instance_id")
            ek = d.get("external_knowledge") or []
            if iid:
                out[iid] = [int(x) for x in ek]
    return out


def _load_task_canonical_gold() -> dict[str, list[str]]:
    """Map `instance_id` -> canonical `sol_sql` from the gated gold sidecar.

    For dual-eval to be meaningful the audit row's `original_sol_sql` MUST
    equal the canonical gold the loader merges in at task time. A drift
    here means we'd be scoring against a fictional 'original'.
    """
    root = _livesqlbench_data_or_skip()
    gold = root / "livesqlbench_sqlite_gt_kg_testcases_0528.jsonl"
    if not gold.exists():
        pytest.skip(f"livesqlbench gold sidecar missing at {gold}")
    out: dict[str, list[str]] = {}
    with gold.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iid = d.get("instance_id")
            sql = d.get("sol_sql")
            if iid and isinstance(sql, list):
                out[iid] = list(sql)
    return out


# ---------------------------------------------------------------------------
# Schema + coverage
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_main_checkout_cache():
    """Defensive: another test (e.g. in test_paths.py) may have populated
    `_main_checkout_root_cached` with a tmp value while monkey-patching
    `_LOOKUP_DIR`. Even though monkeypatch restored `_LOOKUP_DIR`, the
    lru_cache holds the tmp result until cleared. Clear before AND after
    so this module's path resolutions always anchor at the real main
    checkout, and we don't leak our reset to the next module.
    """
    paths._main_checkout_root_cached.cache_clear()
    yield
    paths._main_checkout_root_cached.cache_clear()


def test_audit_file_exists_when_other_audits_are_present():
    """The audited_gold/ dir is gitignored (rides into the cloud image via
    the BuildKit `audited-gold=` context, not git — see the
    `project_gold_delivery_split` memory), so absence on a fresh checkout
    is normal and we skip.

    BUT: if mini-interact audit dirs are present (i.e. this dev has authored
    audit work), then the livesqlbench single-file MUST also be present —
    that's DEV-1510's deliverable. Catches the case where someone re-runs
    the audit pipeline and forgets to regenerate the livesqlbench file."""
    root = paths.audited_gold_root()
    if not root.exists():
        pytest.skip(f"audited_gold root absent ({root}); fresh checkout")
    has_other_audits = any(
        p.is_dir() and (p / f"{p.name}_audited.jsonl").exists()
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not has_other_audits:
        pytest.skip(
            f"no audited_gold/<db>/<db>_audited.jsonl present in {root}; "
            "no audit work in this checkout, so no DEV-1510 deliverable expected"
        )
    path = root / "livesqlbench_audited.jsonl"
    assert path.exists(), (
        f"expected audited-gold deliverable at {path}; mini-interact audits "
        "are present but the livesqlbench single-file (DEV-1510) is not. "
        "Re-author via the `audit-gold-sql-livesqlbench` skill."
    )


def test_audit_rows_cover_exactly_museum_1_through_10():
    rows = _load_audit_rows()
    assert set(rows.keys()) == EXPECTED_INSTANCE_IDS, (
        f"audit must cover exactly museum_1..10; missing="
        f"{sorted(EXPECTED_INSTANCE_IDS - rows.keys())}; "
        f"extra={sorted(rows.keys() - EXPECTED_INSTANCE_IDS)}"
    )


def test_audit_rows_have_required_keys_and_types():
    for row in _iter_audit_rows():
        iid = row.get("instance_id", "<missing>")
        missing = REQUIRED_ROW_KEYS - row.keys()
        assert not missing, f"{iid}: missing required keys {sorted(missing)}"
        assert isinstance(row["instance_id"], str), iid
        assert isinstance(row["selected_database"], str), iid
        assert isinstance(row["benchmark"], str), iid
        assert isinstance(row["audit_status"], str), iid
        assert isinstance(row["original_sol_sql"], list), iid
        assert isinstance(row["audited_sol_sql"], list), iid
        assert isinstance(row["audited_sample_row"], list), iid
        assert isinstance(row["changes"], list), iid
        assert isinstance(row["reasoning_summary"], str), iid
        assert isinstance(row["skill_version"], str), iid
        assert isinstance(row["audited_at"], str), iid
        # SQL list elements are all strings.
        for s in row["original_sol_sql"]:
            assert isinstance(s, str), f"{iid}: non-str in original_sol_sql"
        for s in row["audited_sol_sql"]:
            assert isinstance(s, str), f"{iid}: non-str in audited_sol_sql"


def test_audit_rows_use_valid_audit_status():
    for row in _iter_audit_rows():
        assert row["audit_status"] in VALID_STATUSES, (
            f"{row['instance_id']}: audit_status={row['audit_status']!r} "
            f"not in {sorted(VALID_STATUSES)}"
        )


def test_audit_rows_tag_benchmark_and_database():
    for row in _iter_audit_rows():
        assert row["benchmark"] == "livesqlbench", (
            f"{row['instance_id']}: benchmark={row['benchmark']!r} (expected 'livesqlbench')"
        )
        assert row["selected_database"] == "museum", (
            f"{row['instance_id']}: selected_database={row['selected_database']!r} "
            "(expected 'museum')"
        )


def test_audit_rows_skill_version_pinned():
    for row in _iter_audit_rows():
        assert row["skill_version"].startswith("audit-gold-sql-livesqlbench/"), (
            f"{row['instance_id']}: unexpected skill_version="
            f"{row['skill_version']!r}"
        )


def test_audit_rows_audited_at_is_iso8601():
    """ISO-8601 with timezone — same convention as the existing mini-interact
    audited_gold rows. Catches accidental Excel-style timestamps."""
    iso_re = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
    )
    for row in _iter_audit_rows():
        assert iso_re.match(row["audited_at"]), (
            f"{row['instance_id']}: audited_at={row['audited_at']!r} "
            "is not ISO-8601 with timezone"
        )


def test_no_duplicate_instance_id_variant_pairs():
    """The dedup contract is on the (instance_id, variant_id) pair, not on
    instance_id alone — DEV-1515 multi-variant audits ship N rows per task
    (one primary + alternates). Also: each instance_id MUST have exactly one
    primary row."""
    seen_pairs: list[tuple[str, str]] = []
    primaries_per_iid: dict[str, int] = {}
    for row in _iter_audit_rows():
        iid = row["instance_id"]
        vid = row.get("variant_id", "primary")
        seen_pairs.append((iid, vid))
        if row.get("primary", True):
            primaries_per_iid[iid] = primaries_per_iid.get(iid, 0) + 1
    dupes = [p for p in seen_pairs if seen_pairs.count(p) > 1]
    assert not dupes, (
        f"duplicate (instance_id, variant_id) pairs in audit file: {dupes}"
    )
    over_primaries = {
        iid: n for iid, n in primaries_per_iid.items() if n != 1
    }
    assert not over_primaries, (
        f"each instance_id must have exactly one primary row; "
        f"got counts: {over_primaries}"
    )


# ---------------------------------------------------------------------------
# Status-claim consistency
# ---------------------------------------------------------------------------


def test_clean_rows_have_audited_equal_original_and_no_changes():
    for row in _iter_audit_rows():
        if row["audit_status"] != "clean":
            continue
        iid = row["instance_id"]
        assert row["audited_sol_sql"] == row["original_sol_sql"], (
            f"{iid}: clean row must have audited_sol_sql == original_sol_sql; "
            f"audited={row['audited_sol_sql']!r}, original={row['original_sol_sql']!r}"
        )
        assert row["changes"] == [], (
            f"{iid}: clean row must have empty changes; got {row['changes']!r}"
        )


def test_edited_and_unrecoverable_rows_have_changes_and_differ():
    for row in _iter_audit_rows():
        if row["audit_status"] not in {"edited", "unrecoverable"}:
            continue
        iid = row["instance_id"]
        assert row["audited_sol_sql"] != row["original_sol_sql"], (
            f"{iid}: {row['audit_status']} row must differ from original_sol_sql"
        )
        assert row["changes"], (
            f"{iid}: {row['audit_status']} row must have non-empty changes"
        )
        for j, change in enumerate(row["changes"]):
            missing = REQUIRED_CHANGE_KEYS - change.keys()
            assert not missing, (
                f"{iid}: changes[{j}] missing keys {sorted(missing)}"
            )
            assert isinstance(change["clause_kind"], str) and change["clause_kind"], iid
            assert isinstance(change["why_unjustified"], str) and change["why_unjustified"], iid
            assert isinstance(change["justified_by"], list) and change["justified_by"], (
                f"{iid}: changes[{j}].justified_by must be a non-empty list"
            )
            # Every justified_by token must look like a citation. (The
            # resolvability tests below confirm the tokens actually resolve.)
            for token in change["justified_by"]:
                assert CITATION_REGEX.fullmatch(token), (
                    f"{iid}: changes[{j}].justified_by carries unrecognised "
                    f"token {token!r}"
                )


def test_clean_rows_have_substantive_reasoning_summary():
    """Without `changes` entries, `reasoning_summary` is the ONLY audit-trail
    artifact a reviewer (or test) can inspect to know whether the auditor
    actually walked each clause. Require it to be substantive: at least
    200 characters AND carry at least one citation token of the form
    `kb:N` / `column_meaning:T|C` / `external_knowledge:N`. The 200-char
    bar comes from inspecting the existing households audit rows — every
    `clean`/`edited` reasoning_summary there clears it comfortably."""
    for row in _iter_audit_rows():
        if row["audit_status"] != "clean":
            continue
        iid = row["instance_id"]
        rs = row["reasoning_summary"]
        assert len(rs) >= 200, (
            f"{iid}: clean reasoning_summary is too short "
            f"({len(rs)} chars); minimum is 200 so a reviewer can see the "
            f"per-clause justification walk."
        )
        assert CITATION_REGEX.search(rs), (
            f"{iid}: clean reasoning_summary must cite at least one source "
            f"(kb:N / column_meaning:T|C / external_knowledge:N); none found"
        )


# ---------------------------------------------------------------------------
# Pinned decisions: museum_7 (edited) + museum_9 (clean)
# ---------------------------------------------------------------------------


def test_museum_7_is_edited_with_null_safe_three_of_four_predicate():
    """DEV-1510 locked decision: museum_7 gold treated 4 flags as independently
    sufficient; KB 16 says SESR<4 OR at-least-three-of-the-4-flags. Audited
    SQL must rewrite to a NULL-safe CASE-based count (the naive boolean-
    arithmetic form `(seal='Poor')+(maint='Overdue')+... >= 3` is NULL-
    fragile in SQLite: any one nullable flag turns the sum to NULL, masking
    three positive flags)."""
    rows = _load_audit_rows()
    row = rows.get("museum_7")
    assert row is not None, "museum_7 must be in the audit file"
    assert row["audit_status"] == "edited", (
        f"museum_7 must be 'edited'; got {row['audit_status']!r}"
    )

    sql_blob = " ".join(row["audited_sol_sql"]).lower()
    flags = (
        ("sealcondition", "poor"),
        ("maintstatus", "overdue"),
        ("filterstatus", "replace now"),
        ("silicagelstatus", "replace now"),
    )

    # The four flag predicates must appear in CASE form (NULL-safe).
    for col, lit in flags:
        case_pat = re.compile(
            rf"case\s+when\s+[a-z0-9_.]*{col}\s*=\s*'{re.escape(lit)}'\s+then\s+1\s+else\s+0\s+end",
        )
        assert case_pat.search(sql_blob), (
            f"museum_7 audited_sol_sql missing NULL-safe CASE for "
            f"{col}='{lit}'; got SQL: {sql_blob!r}"
        )

    # The four CASE expressions must be summed AND the sum compared >= 3.
    # A loose `>= 3` somewhere in the SQL is not enough — pin the sum-of-
    # four-CASEs >= 3 shape so a future regression (e.g. dropping one CASE,
    # or comparing against the wrong total) is caught. The column names
    # are not all `*status` — sealcondition ends in `condition` — so the
    # CASE term matches any identifier (table-qualified optional).
    case_term = (
        r"case\s+when\s+[a-z0-9_.]+\s*=\s*'[^']+'"
        r"\s+then\s+1\s+else\s+0\s+end"
    )
    sum_of_four_geq_three = re.compile(
        rf"(?:{case_term})(?:\s*\+\s*(?:{case_term})){{3}}\s*\)?\s*>=\s*3\b"
    )
    assert sum_of_four_geq_three.search(sql_blob), (
        f"museum_7 audited_sol_sql must compare a 4-CASE sum to '>= 3' "
        f"(KB 16's at-least-three-of-the-4-flags); got SQL: {sql_blob!r}"
    )

    # Naive boolean-arithmetic form must NOT appear for ANY of the four
    # flags — the pattern `(<col>='<lit>') +` is NULL-fragile in SQLite.
    for col, lit in flags:
        # Boolean addition in either direction: `(col='lit') +` or
        # `+ (col='lit')`. Reject both.
        naive_left = re.compile(
            rf"\(\s*[a-z0-9_.]*{col}\s*=\s*'{re.escape(lit)}'\s*\)\s*\+",
        )
        naive_right = re.compile(
            rf"\+\s*\(\s*[a-z0-9_.]*{col}\s*=\s*'{re.escape(lit)}'\s*\)",
        )
        assert not naive_left.search(sql_blob), (
            f"museum_7 audited_sol_sql uses NULL-fragile boolean-arithmetic "
            f"for {col}='{lit}'; must use CASE WHEN ... THEN 1 ELSE 0 END. "
            f"Got SQL: {sql_blob!r}"
        )
        assert not naive_right.search(sql_blob), (
            f"museum_7 audited_sol_sql uses NULL-fragile boolean-arithmetic "
            f"for {col}='{lit}'; must use CASE WHEN ... THEN 1 ELSE 0 END. "
            f"Got SQL: {sql_blob!r}"
        )

    # And the OR-each-flag form (the original bug) must also be gone for
    # EVERY pair — not just sealcondition/maintstatus. Build a list of
    # (col, lit) pair regexes and assert none of them appears as `<a> OR <b>`.
    for (col_a, lit_a), (col_b, lit_b) in (
        (flags[0], flags[1]),
        (flags[1], flags[2]),
        (flags[2], flags[3]),
        (flags[0], flags[3]),
    ):
        or_pair = re.compile(
            rf"[a-z0-9_.]*{col_a}\s*=\s*'{re.escape(lit_a)}'"
            rf"\s+or\s+[a-z0-9_.]*{col_b}\s*=\s*'{re.escape(lit_b)}'",
        )
        assert not or_pair.search(sql_blob), (
            f"museum_7 audited_sol_sql still has the original any-one-flag "
            f"OR chain ({col_a}='{lit_a}' OR {col_b}='{lit_b}'); "
            f"contradicts KB 16. Got SQL: {sql_blob!r}"
        )

    # Citations: must cite kb:16 AND external_knowledge:16 (16 is in the
    # task's external_knowledge anchor list).
    tokens = {
        t for change in row["changes"] for t in change["justified_by"]
    }
    assert "kb:16" in tokens, (
        f"museum_7 changes must cite kb:16; got tokens={sorted(tokens)}"
    )
    assert "external_knowledge:16" in tokens, (
        f"museum_7 changes must cite external_knowledge:16 (KB 16 is in the "
        f"task's anchor list); got tokens={sorted(tokens)}"
    )


def test_museum_9_is_clean_with_column_meaning_justification():
    """DEV-1510 locked decision: museum_9 gold uses
    ConditionAssessments.LightReadRefObserved (single-hop declared FK with
    column-meaning text 'Associates the assessment with relevant light
    data'). Audit status is 'clean'; reasoning_summary must name BOTH
    candidate join chains (so the audit trail explains WHY the agent's
    UsageRecords reading is also defensible from KB-alone) and cite the
    column-meaning that resolves the disambiguation."""
    rows = _load_audit_rows()
    row = rows.get("museum_9")
    assert row is not None, "museum_9 must be in the audit file"
    assert row["audit_status"] == "clean", (
        f"museum_9 must be 'clean'; got {row['audit_status']!r}"
    )
    assert row["audited_sol_sql"] == row["original_sol_sql"], (
        "museum_9 is 'clean' — audited_sol_sql must equal original_sol_sql"
    )
    assert row["changes"] == [], (
        f"museum_9 is 'clean' — changes must be empty; got {row['changes']!r}"
    )

    rs = row["reasoning_summary"]
    rs_lower = rs.lower()
    # Both candidate join chains named, AND the discriminating endpoints
    # cited. A bare "usagerecords" or "conditionassessments" mention
    # would let the reasoning slip past with no actual explanation of
    # the underspec — pin the FK column and the alternative chain's
    # discriminator so the audit trail is meaningful.
    assert "usagerecords" in rs_lower, (
        f"museum_9 reasoning_summary must name the alternative UsageRecords "
        f"chain (the one the agent picked) so the audit explains the "
        f"disambiguation; got: {rs!r}"
    )
    # The agent's chain pivots through Showcases / EnvironmentalReadingsCore
    # to find a light reading — naming at least one of those endpoints
    # demonstrates the audit understood the 3-hop chain.
    assert (
        "environmentalreadingscore" in rs_lower
        or "showcaseref" in rs_lower
        or "showcases" in rs_lower
    ), (
        f"museum_9 reasoning_summary must name an endpoint of the "
        f"UsageRecords→Showcases→EnvironmentalReadingsCore→LightAndRadiationReadings "
        f"chain (Showcases / EnvironmentalReadingsCore / ShowcaseRef) so "
        f"the audit shows what makes that chain weaker than gold's. "
        f"Got: {rs!r}"
    )
    # Gold's chain is single-hop via LightReadRefObserved; the audit
    # MUST name that FK column explicitly (it's the discriminator that
    # resolves the KB-alone underspec).
    assert "lightreadrefobserved" in rs_lower, (
        f"museum_9 reasoning_summary must name "
        f"ConditionAssessments.LightReadRefObserved — the single-hop FK "
        f"that resolves the KB underspec; got: {rs!r}"
    )
    # And the column-meaning citation MUST appear in the exact token form
    # the verifier resolves against `museum_column_meaning_base.json`. A
    # loose substring match would let `LightReadRefObserved` mentioned in
    # prose-only count, which weakens the resolvability guarantee.
    assert "column_meaning:ConditionAssessments|LightReadRefObserved" in rs, (
        f"museum_9 reasoning_summary must contain the EXACT citation token "
        f"'column_meaning:ConditionAssessments|LightReadRefObserved' so "
        f"the resolvability test catches typos; got: {rs!r}"
    )


# ---------------------------------------------------------------------------
# Citation resolvability (requires upstream livesqlbench data)
# ---------------------------------------------------------------------------


def test_every_kb_citation_resolves_to_a_museum_kb_id():
    """Every `kb:N` citation MUST resolve to a row in museum_kb.jsonl —
    catches typos (kb:116 instead of kb:16) and id-drift after upstream
    KB renumbers. Same posture as the mini-interact resolvability tests."""
    kb_ids = _load_museum_kb_ids()
    for row in _iter_audit_rows():
        iid = row["instance_id"]
        tokens = _collect_citation_tokens(row)
        for tok in tokens:
            if tok.startswith("kb:"):
                kb_id = int(tok.split(":", 1)[1])
                assert kb_id in kb_ids, (
                    f"{iid}: kb:{kb_id} does not resolve in museum_kb.jsonl"
                )


def test_every_external_knowledge_citation_is_in_task_anchor_list():
    """`external_knowledge:N` is a strictly STRONGER signal than `kb:N`:
    it says the KB id is in the TASK's `external_knowledge` anchor list.
    A citation against an id NOT in that list is wrong: the auditor is
    claiming the task anchored on a KB it didn't actually anchor on."""
    task_ek = _load_task_external_knowledge()
    for row in _iter_audit_rows():
        iid = row["instance_id"]
        anchored = set(task_ek.get(iid, []))
        for tok in _collect_citation_tokens(row):
            if tok.startswith("external_knowledge:"):
                kb_id = int(tok.split(":", 1)[1])
                assert kb_id in anchored, (
                    f"{iid}: external_knowledge:{kb_id} cited but KB id "
                    f"{kb_id} is NOT in the task's external_knowledge "
                    f"anchor list ({sorted(anchored)})"
                )


def test_every_column_meaning_citation_resolves():
    """Catches case-typos in column-meaning citations — keys are
    case-sensitive in the JSON (e.g. `museum|ConditionAssessments|LightReadRefObserved`,
    NOT lowercase)."""
    keys = _load_museum_column_meaning_keys()
    for row in _iter_audit_rows():
        iid = row["instance_id"]
        for tok in _collect_citation_tokens(row):
            if tok.startswith("column_meaning:"):
                table_col = tok.split(":", 1)[1]
                # Citations use the `Table|Column[|SubField]` form; the
                # JSON key is `museum|Table|Column[|SubField]`. Both shapes
                # accepted.
                candidates = [f"museum|{table_col}", table_col]
                assert any(c in keys for c in candidates), (
                    f"{iid}: column_meaning:{table_col} does not resolve in "
                    f"museum_column_meaning_base.json (tried: {candidates})"
                )


def test_original_sol_sql_matches_canonical_gold_for_each_task():
    """`original_sol_sql` must equal the canonical gold from the
    livesqlbench gated sidecar — otherwise the dual-eval would score
    against a fictional 'original' that nothing else in the pipeline
    knows about, and the `phase1_passed_audited > phase1_passed_original`
    acceptance signal would be meaningless."""
    canonical = _load_task_canonical_gold()
    for row in _iter_audit_rows():
        iid = row["instance_id"]
        expected = canonical.get(iid)
        assert expected is not None, (
            f"{iid}: no canonical gold found in livesqlbench gold sidecar"
        )
        # Compare after normalising whitespace, since the sidecar's
        # `sol_sql` strings sometimes have collapsed whitespace from the
        # upstream re-export pipeline (cf. the SQLite gold file note in
        # the `reference_livesqlbench_sqlite_gold` memory).
        assert _normalise_sql_list(row["original_sol_sql"]) == _normalise_sql_list(expected), (
            f"{iid}: original_sol_sql in audit does NOT match the canonical "
            f"livesqlbench gold. audit={row['original_sol_sql']!r}; "
            f"canonical={expected!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_citation_tokens(row: dict) -> set[str]:
    """Citations live in `changes[].justified_by[]` and (loosely) inside
    `reasoning_summary`. Tests that care about resolvability scan both."""
    out: set[str] = set()
    for change in row.get("changes", []):
        for t in change.get("justified_by", []) or []:
            out.add(t)
    for m in CITATION_REGEX.finditer(row.get("reasoning_summary", "")):
        out.add(m.group(1))
    return out


def _normalise_sql_list(sqls: list[str]) -> list[str]:
    return [" ".join(s.split()).strip() for s in sqls]
