"""Shared prompt string constants for the claude_sdk OTF agent family.

These constants are used verbatim (format params substituted at compose
time) by both the SLayer OTF agents and the raw OTF agents to keep prompts
aligned wherever SLayer is not involved.

Constraint: after the SLayer prompt files are refactored to import from
here, the rendered values of SLAYER_OTF_ONE_SHOT and SLAYER_OTF_AINTERACT
must remain byte-for-byte identical except for deliberate prompt-content
changes that bump the SHA-256 snapshot constants in
tests/test_shared_otf_prompts.py.

Format param conventions:
  {db_name}           — datasource id, surfaced inside the SLayer-tools block
  {sources_desc}      — phrase describing the knowledge sources available
  {action_label}      — upper-case verb for Rule 0 heading ("ENCODE"/"SUBMIT")
  {action_context}    — Rule 0 first sentence opener
  {submit_tool}       — the submission tool name
  {knowledge_source}  — "a memory" (slayer) / "a knowledge definition" (raw)
  {clause_b}          — "(b) required by an ___" clause in one-shot check
  {clause_c}          — "(c) required by an ___" clause in ainteract check
"""

# ---------------------------------------------------------------------------
# Shared constants — format params noted per variable
# ---------------------------------------------------------------------------

# Format params: {sources_desc}
_NO_USER_TO_CONSULT = """\
There is NO user to consult — for every operationalisation choice (numeric
threshold, value list, aggregation operator, case-sensitivity, grouping,
unit, rounding, sort direction, LIMIT) pick the most conservative,
defensible interpretation supported by {sources_desc}, and proceed autonomously."""

# No format params.
_DECOMPOSE_DISCIPLINE = """\
1. DECOMPOSE the question into logical blocks. Every qualifier
   (e.g. "premium", "highly-rated", "nearby", "active"), every projected
   column, filter, grouping, unit, rounding and ordering hint is a
   separate block that MUST be represented. Write the list out before
   encoding."""

# Format params: {action_label}, {action_context}, {submit_tool}
_RULE_0_ASK_BEFORE = """\
RULE 0 — ASK BEFORE YOU {action_label}.
{action_context} identify the single operationalisation
choice you are LEAST certain about — a numeric threshold, a value list /
IN-set, an aggregation operator, a case-sensitivity choice, a grouping
or standardisation, a unit (fraction vs percent), an output rounding, a
sort direction, or a LIMIT — and call `ask_user` on it ONCE. The user
holds masked knowledge-base ground-truth that is unrecoverable from the
visible KB alone. The submit gate will REFUSE `{submit_tool}` until you
have called `ask_user` at least once. Propose your best guess and ask
for the EXACT predicate / value / formula — never "what does X mean?"."""

# Format params: {knowledge_source}
_ASK_AGAIN_RULE = """\
4. ASK AGAIN IF NEEDED. Rule 0 covers the FIRST ask; for any further
   operationalisation choice not pinned by {knowledge_source} or column
   description, call `ask_user` again. If a reply lists multiple criteria
   joined by "and", apply EACH as its own filter."""

# Format params: {submit_tool}, {clause_b}
_PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT = """\
5. PRE-SUBMIT MUTATION CHECK. Before calling `{submit_tool}`, audit every
   TRIM, LOWER, UPPER, ROUND, CAST, dedup, canonicalize-via-CASE, and
   output-shape choice in the FINAL query. Each one MUST be either
   (a) explicitly named in the user's question or (b) required by an
   {clause_b}. If neither holds, DROP the mutation and submit the raw
   form. "Defensive" normalisation of an output column, a join key, a
   JSON key, or a CHAR-padded literal silently corrupts the rowset —
   never apply one without an explicit source. There is no user to
   second-guess this on your behalf."""

# Format params: {submit_tool}, {clause_c}
_PRE_SUBMIT_MUTATION_CHECK_AINTERACT = """\
6. PRE-SUBMIT MUTATION CHECK. Before calling `{submit_tool}`, audit every
   TRIM, LOWER, UPPER, ROUND, CAST, dedup, canonicalize-via-CASE, and
   output-shape choice in the FINAL query. Each one MUST be either
   (a) explicitly named in the user's question, (b) explicitly named OR
   authorized in a reply to one of your `ask_user` calls in this
   session, or (c) required by an {clause_c}. If none of (a-b-c) hold,
   DROP the mutation and submit the raw form. Particularly: when an
   `ask_user` reply said "use exact values", "don't normalize", "use
   this output shape / columns / sort axis", or named a specific format
   (date, label casing, JSON shape), DO NOT silently override that on
   final-assembly. Conversely, when an `ask_user` reply DID name a
   specific transformation (e.g. "lowercase the bracket labels",
   "round to 2 decimals", "TRIM the keys"), that reply IS the
   authorization for that mutation — apply it."""

# DEV-1534 Fix B — applies to all 4 OTF flavors (grader behaviour, not
# mode-specific). No format params.
# DEV-1535 Fix: added cheap-experiments-first guidance for opaque grader
# failures (cheaper than adding logic — fixes the `output_schema_misread`
# class of failures observed in production).
_COLUMN_NAMES_DONT_AFFECT_GRADING = """\
COLUMN HEADERS DO NOT AFFECT GRADING. The grader compares value tuples
POSITIONALLY — column COUNT, positional ORDER, value TYPES, and VALUES
matter; column NAMES do not. Do not waste turns renaming projection
aliases to match the user's wording or any reference/gold labels.

RE-READ THE QUESTION FOR SHAPE CUES before submit. Explicit ordering
language ("list X, then Y, then Z" or "show NAICS, percentage, count")
pins the projection ORDER positionally. Bare-quoted IDs like "541511" or
"00123" can be string OR numeric — try the literal form as it appears in
the question first. References to flags / "is_X" / boolean predicates
mean RAW BOOLEAN output unless the question says "as 0/1"; do NOT wrap
in `CAST AS INTEGER` defensively.

WHEN THE GRADER RETURNS AN OPAQUE FAILURE (e.g. "ex_base returned 0 but
expected 1") AND YOUR VALUES LOOK RIGHT, try the cheap shape experiments
FIRST — before adding logic, changing formulas, or asking the user:
  * column-order permutations (the gold may order them differently from
    your alphabetical / "as-they-appeared" sequence),
  * bare-type variants (raw BOOLEAN vs CAST INT; string vs numeric for
    ID-like columns; date vs ISO-string),
  * column-count variants (drop a derived column that isn't named in the
    question; add an obvious ID that the user implied but didn't list).
These swap the row tuple structure without changing what you computed —
much cheaper than a new join or KB re-read."""

# DEV-1534 Fix D — slayer-mode only (one-shot + a-interact). No format
# params. The `normalize_filters` opt-out applies to `query`,
# `query_nested`, AND `submit_query` — all served by our bird-interact-
# tools wrappers in the OTF agents.
_SLAYER_SQL_ARTIFACT_CHECK = """\
SANITY-CHECK THE GENERATED SQL FOR SLAYER ARTIFACTS. After `query` /
`query_nested` returns, inspect the rendered SQL for these patterns
before submitting:

  1. GROUP BY on every projected column with NO aggregate functions —
     silently deduplicates rows. Fix: add the table's primary-key column
     as a dimension, or restructure as a nested-DAG stage with a
     `*:count` measure.
  2. `lower(trim(col)) = '<lowercase literal>'` on string equality
     filters — wrapped automatically by default. When the gold answer
     requires exact-case equality (proper-noun categories with
     known-fixed casing), pass `normalize_filters=false` as a SEPARATE
     parameter on the offending `query` / `query_nested` / `submit_query`
     call (the flag lives OUTSIDE the JSON DSL).
  3. Broken operator precedence on WHERE arithmetic:
     `expr1*w1 + expr2*w2 > threshold` without outer parens — the
     comparator binds only to the last additive term. Fix: push the
     score into a HAVING on a nested-stage measure rather than a WHERE
     on a raw formula."""

# DEV-1534 Fix E — a-interact only (one-shot variants get exactly ONE
# submission, so the "after 3 failed" trigger never fires). Format
# params:
#   {artifact_inspect_step} — slayer prompts pass the SLayer-artifact
#       inspection text + cross-ref to the artifact-check rule below;
#       raw prompts pass a mode-agnostic "inspect the SQL you submitted"
#       paragraph instead.
#   {extra_hypothesis_axes} — slayer prompts add the
#       `normalize_filters=false` axis to the structural-hypothesis
#       list; raw prompts pass an empty string.
_PIVOT_AFTER_REPEATED_FAILURES = """\
PIVOT AFTER 3 FAILED SUBMISSIONS WITH THE SAME OPAQUE ERROR. Stop
varying surface parameters and:

  1. {artifact_inspect_step}
  2. Enumerate ≤4 structurally different hypotheses not yet tested:
     different row grain, formula kernel, join path, or output column
     count/type{extra_hypothesis_axes}.
  3. Call `ask_user` ONCE with those hypotheses as concrete options to
     get directional guidance (cheaper than many resubmissions of
     near-identical queries).
  4. Test each surviving hypothesis exactly once — never re-submit a
     query structurally identical to a prior attempt."""

# DEV-1534 Fix F — a-interact only (one-shot variants have no user
# simulator). No format params.
_USER_SIM_TRUST_CALIBRATION = """\
USER-SIM ANSWERS ARE CLARIFICATIONS, NOT GROUND TRUTH.

  - Cross-check user-sim formulas against the {knowledge_label} before
    submitting. If they contradict (e.g. the user-sim denies a column
    the {knowledge_label} explicitly names), try the
    {knowledge_label}-grounded interpretation first.
  - After ≥2 failures with a user-sim-confirmed interpretation, try
    the {knowledge_label}-literal / schema-literal interpretation as a
    fallback submission.
  - When a user-sim constraint makes the required output cardinality
    impossible (e.g. "use only table X" but X has 4 distinct values
    and the task needs top-5), call `ask_user` again to flag the
    contradiction explicitly rather than submitting an impossible
    query."""


# DEV-1550 A3: shared "SLAYER TOOLS" block — extracted byte-for-byte
# from the previously-duplicated `_AINTERACT_SLAYER_TOOLS` /
# `_ENCODE_CORE_HEAD` (verified identical at extraction time), with
# the new "READ A KNOWN MEMORY'S FULL BODY" drill-in paragraph
# inserted as a sibling between the existing column-drill-in
# paragraph and the `ENCODE-THEN-QUERY DISCIPLINE:` header. The
# memory-drill-in nudge documents the compact-mode opt-out introduced
# by SLayer 0.7.3 (DEV-1549): `search` now defaults to `compact=True`
# and renders one-line `Memory.description` summaries; agents need
# `compact=False` (plus a tight `max_memories`) to get the full
# `learning` body for a memory id they've already identified.
#
# Format params: {db_name}
_SLAYER_TOOLS_BLOCK = """\
The database's domain knowledge is pre-loaded as SLayer MEMORIES — one per
knowledge-base (KB) item, with ids like `{db_name}_kb_<n>` whose body
starts `KB <n> —`. The base tables are already ingested as SLayer models,
but NOTHING is encoded yet: you encode exactly what THIS question needs,
on the fly.

SLAYER TOOLS (read their own descriptions). Call `help` FIRST to learn the
query syntax — the colon-aggregation form (`revenue:sum`, `*:count`) and
the `source_model` / `dimensions` / `measures` / `filters` schema. Use
`search` to find relevant memories and existing entities; `inspect_model`
to see a model's columns / measures / joins; `create_model` / `edit_model`
to add columns and measures; `query` / `query_nested` to test.

READ A KNOWN COLUMN'S FULL DESCRIPTION before committing to it as a
filter, projection, or join key — `search` with `entities=[
"<db>.<model>.<col>"]`, `max_memories=0`, `max_example_queries=0`. The
returned `EntityHit.text` carries `Description:` and `Sample values:`
inline. The truncated `Sample values:` line is your authoritative source
of which literal forms actually occur in this column — case variants,
whitespace forms, abbreviations, alternate phrasings of the same concept.
Use it BEFORE writing any IN-set (see rule 3 below).

READ A KNOWN MEMORY'S FULL BODY when you need the verbatim KB content for
a memory id you've already identified — `search` with `entities=[
"memory:<id>"]`, `max_memories=1`, `compact=False`. By default `search`
is compact (one-line `description` summary per hit); `compact=False`
returns the full `learning` body, and `max_memories=1` keeps adjacent
matches out of the response.

ENCODE-THEN-QUERY DISCIPLINE:"""
