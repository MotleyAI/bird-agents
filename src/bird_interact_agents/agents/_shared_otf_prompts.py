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

# DEV-1546 — slayer-mode only (one-shot + a-interact). No format params.
# Proactive guidance on the SLayer 0.7.2 `distinct_dimension_values`
# field: the agent decides BEFORE writing the JSON whether the question
# wants raw rows or distinct dimension tuples. Synthetic example uses
# fabricated names (per project convention) but mirrors the structural
# shape of real overaggregation misses — dim-only ask, optional LIMIT,
# no measures.
_DEDUP_VS_RAW_ROWS = """\
DEDUP vs RAW ROWS. By default SLayer auto-DEDUPLICATES dimension-only
queries: when `measures` is empty, it wraps every projected column in a
top-level `GROUP BY`, collapsing rows that share the same dimension
tuple. To emit raw per-record rows instead, set
`distinct_dimension_values: false` inside the query JSON — flat
`SELECT <dims/td> FROM ... WHERE ... ORDER BY ... LIMIT`, no top-level
`GROUP BY`. The field lives INSIDE the SlayerQuery JSON (alongside
`source_model`, `dimensions`, etc.), same shape for `query` / `submit_query`.

Decide BEFORE writing the query:

  * Use `distinct_dimension_values: false` when the question asks for a
    PER-RECORD listing (e.g. "list each <X>'s <a> and <b>; if two
    <X>s share the same <a>, <b>, return BOTH rows").
  * Keep the default `true` when the question asks for the distinct
    <a>, <b> COMBINATIONS (or for an aggregation grouped by them).
  * If you need BOTH a per-record listing and a count, keep the
    default and add `*:count` as a measure (or restructure as a
    nested-DAG stage with `*:count`).

Validation: `distinct_dimension_values: false` requires
  * `measures` empty (the flag asks for raw rows, not aggregations),
  * at least one of `dimensions` / `time_dimensions` non-empty
    (something must be projected),
  * no measure reference in `filters` / `order`.
SLayer raises a `DistinctDimensionValuesError` otherwise.

Synthetic example (fabricated names — your DB uses different
identifiers):

  Q: "List the first 10 (workshop_id, district) pairs in the workshops
     table. If two workshops share the same (workshop_id, district),
     return BOTH rows."

  WRONG (default — silently dedups when two workshops share the
  tuple):
    {{"source_model": "workshops",
     "dimensions": ["workshop_id", "district"],
     "limit": 10}}

  RIGHT (raw rows):
    {{"source_model": "workshops",
     "dimensions": ["workshop_id", "district"],
     "limit": 10,
     "distinct_dimension_values": false}}
"""

# DEV-1534 Fix D / DEV-1546 — slayer-mode only (one-shot + a-interact).
# No format params. The `normalize_filters` opt-out applies to `query`,
# AND `submit_query`; the `distinct_dimension_values`
# field lives INSIDE the JSON DSL (see _DEDUP_VS_RAW_ROWS above for
# the proactive rule + synthetic example).
_SLAYER_SQL_ARTIFACT_CHECK = """\
SANITY-CHECK THE GENERATED SQL FOR SLAYER ARTIFACTS. After `query` returns, inspect the rendered SQL for these patterns
before submitting:

  1. GROUP BY on every projected column with NO aggregate functions —
     SLayer's default dim-only auto-dedup. If the question asks for
     raw per-record rows, fix by setting
     `distinct_dimension_values: false` INSIDE the query JSON (see
     the DEDUP vs RAW ROWS rule above). If you DO need a count
     alongside the rows, keep the default and add `*:count` as a
     measure — or restructure as a nested-DAG stage with a `*:count`
     measure.
  2. `lower(trim(col)) = '<lowercase literal>'` on string equality
     filters — wrapped automatically by default. When the gold answer
     requires exact-case equality (proper-noun categories with
     known-fixed casing), pass `normalize_filters=false` as a SEPARATE
     parameter on the offending `query` / `submit_query`
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


# DEV-1555 r6-diagnosis fix: after-rejection discipline. Failure pattern
# observed: agent reads "wrong_result" as "the WHOLE model design is
# wrong, start over from scratch" — picks a different grouping column,
# renames every entity, picks a different quality formula in the SAME
# attempt. Burns 15+ turns per rebuild and makes triangulation
# impossible. The fix is mode-agnostic, applies on EVERY rejection (not
# just after 3 consecutive failures), and lives in BOTH one-shot and
# a-interact prompts.
_AFTER_REJECTED_DISCIPLINE = """\
AFTER A REJECTED SUBMIT: CHANGE ONE VARIABLE. When a submit comes back
"wrong_result" or "phase1 failed", do NOT throw out the source-model
design. The submit succeeded structurally (no dry-run error, rows came
back), so the encoding + join + grouping are likely fine — the
mismatch is in a single dimension.

Enumerate the possible variables in priority order, change exactly ONE,
and resubmit:

  1. Output COLUMN SET / COUNT — the rejected output had the right
     values but the wrong column tuple (extra ID column, missing
     summary column, two columns transposed).
  2. SORT direction / column — flip ascending↔descending; try a
     different sort key (e.g. count instead of avg).
  3. AGGREGATION SCOPE — switch between "over all rows in group" vs
     "over usable subset" for ONE measure, keep the others.
  4. FORMULA TWEAK — same kernel, different rounding/cast/precision.
  5. ONLY IF (1-4) all fail across separate attempts: revisit the
     source-model design (different grouping column, different join
     path, different formula). This is the "pivot" step above —
     reserve it for the third or fourth rejection, NOT the second.

Anti-patterns to avoid:

  * Renaming the source model between attempts (e.g.
    `signal_quality_by_weather` → `signals_atmo_enriched` →
    `signals_snr_enriched`) — the rename invalidates the diff so you
    cannot tell what actually changed.
  * Switching the grouping column AND the quality formula in the same
    attempt — two changes at once means a pass tells you nothing about
    which one was wrong, a fail tells you nothing either.
  * Re-asking `ask_user` to re-confirm something already answered in
    this session. Re-read your existing Q→A pairs first.

Keep a one-line mental log: "attempt N changed <variable X> from <a>
to <b>; everything else identical to attempt N-1". If you cannot state
that sentence, you are changing too many things at once."""


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


# DEV-1545: targets `wrong_join_path` (autopsies: polar_4, museum_9,
# cross_db_10). Shared across one-shot AND a-interact: the structural-
# pivot half does not need a user-sim. Per Codex review #7, the wording
# explicitly discourages "submit one variant per candidate path" — that
# would worsen `exhausted_budget_guessing`. The agent enumerates
# internally, narrows by evidence, asks ONE discriminating question if
# evidence remains thin, and submits ONLY when one path is selected.
_TABLE_SET_PROBE_HEAD = """\
ALTERNATIVE-JOIN-PATH PROBE.

When {schema_source} reveals a foreign-key path through a table
your current query does NOT use — or when grader diagnostics include
`wrong_table_set` after a submission — the gold likely flows through a
different host or bridge table than the one you picked. Do NOT brute-
force-submit one variant per candidate path; that burns budget.
Instead:

  1. Enumerate the alternative paths INTERNALLY (read columns +
     {knowledge_label} for each candidate bridge). Pick the most
     evidence-supported one."""

# a-interact: a user-sim is available, so the disambiguation step asks it.
_TABLE_SET_PROBE = _TABLE_SET_PROBE_HEAD + """
  2. If two paths remain equally plausible, ask the user-sim ONE
     discriminating question — name both paths and ask which provides
     the canonical link. Do not ask vague "which table?" questions.
  3. Submit ONLY after step 2 distinguishes them. If the user-sim
     refuses to choose, fall back to the {knowledge_label}-grounded
     candidate."""

# one-shot: NO user-sim — disambiguate from evidence and submit the
# best-grounded candidate (DEV-1545: the structural-pivot half needs no
# user). Shared by the slayer + raw one-shot prompts.
_TABLE_SET_PROBE_ONESHOT = _TABLE_SET_PROBE_HEAD + """
  2. If two paths remain equally plausible, re-read the column
     descriptions and {knowledge_label} for each candidate bridge to
     find the discriminating detail — do not submit one variant per
     path.
  3. Submit ONLY the {knowledge_label}-grounded candidate; if the
     descriptions still tie, prefer the shortest declared join path."""


# DEV-1545: targets `never_asked_key_question` (autopsies: robot_9,
# organ_transplant_16). A-interact only — the diagnostic action is a
# user-sim question. Per Codex review #8, the trigger fires on the
# FIRST zero-result mismatch (not the second) because by the second
# attempt the agent has already burned budget; the failure mode is
# failure-to-ask-early.
_GRADER_ZERO_VS_ONE_DIAGNOSTIC = """\
GRADER ZERO-VS-EXPECTED-ONE — ASK A STRUCTURAL QUESTION IMMEDIATELY.

When the grader returns "ex_base 0 vs expected 1" (the gold expects a
row your query is not producing) on a submission AND your query
SHAPE was stable across recent {attempt_noun} attempts, this is almost never
a formula tweak — it is a structural mismatch. Do NOT iterate on
threshold / sort / CAST permutations. Instead, immediately call
`ask_user` with these two specific structural questions, in this order:

  (a) "Is the criteria a FILTER on a pre-classified status column
      (e.g. a `level_val='Marginal'` / `risk_level='High'` column that
      already exists in the schema), or only a REASON label assigned
      to rows the WHERE clause already selected?"
  (b) "Is the rank / window function computed BEFORE the WHERE
      filter is applied (over the full population), or AFTER (over
      the filtered subset)?"

If your task does not involve a ranking / window, skip (b). After
the answer, {apply_verb} the implied filter / pre-classified column /
window scope and resubmit. The point is to flip a single structural
bit, not to keep submitting near-identical queries."""


# DEV-1591 — search-vs-inspect discipline. Shared across every SLayer
# prompt (v0 snapshots, v1 head blocks, the v1 main workflow note, and
# the claude_sdk a/c-interact prompts). A glm-5.2 households run issued a
# broad `search(question=..., compact=False)` that returned 10 full entity
# renders (~55K chars) which then rode in cached context every turn
# (~6x cache-read, ~3.8x prompt tokens vs raw mode). The opus cea364 run
# repeated the pattern despite a prose rule forbidding it, so the
# claude_sdk agents now HARDWIRE `search` to compact=True via a PreToolUse
# hook (`_force_compact_search_hook`): broad discovery returns one-line
# `description`s only, and every targeted detail read moves to the
# `inspect` point-lookup (`entity_type=…`, `compact=False`) — which has no
# RRF fusion, so it needs no `cypher_filter`. This prose teaches the same
# split the hook enforces.
#
# DEV-1629 merge: `search` / `inspect` now live on the v1 slayer MAIN loop
# too (no longer discovery-only), so this block is spliced into the main
# workflow note as well. It therefore frames "discovery" as the ACTIVITY of
# finding candidate ids (not the discovery subagent), and defers root / host
# selection to `recommend_root_model` (Codex review of the merge).
#
# This constant is param-free AND brace-free (no `{`/`}`): several consumers
# (SLAYER_A_INTERACT / SLAYER_C_INTERACT and the `.format()`-rendered v0/v1
# templates) would break on a stray brace. All examples are synthetic — no
# real eval-set DB / model / column names.
_COMPACT_SEARCH_DISCIPLINE = """\
SEARCH-vs-INSPECT DISCIPLINE. Two different jobs; keeping them separate is
what keeps context cheap. `search` is for broad DISCOVERY — finding candidate
entity ids by relevance. `inspect` is for DETAIL — reading the full body of
ids you have already pinned down. ("Discovery" here is the activity of finding
candidates, not any particular agent.) Choosing the query root / `source_model`
/ encoding host is a SEPARATE decision — where your tool surface offers
`recommend_root_model`, use that for it. `search` and `inspect` here are only
for finding candidate ids and reading their details, never for selecting the
root or host.

  * DISCOVER with `search(question="…")`. It returns each hit's one-line
    `description` — enough to choose candidates. Treat broad search as
    compact-only: never pass `compact`, and don't try to pull per-entity
    Type / Description / Sample values / SQL out of `search`. (A broad
    `compact=False` search drags tens of thousands of characters of full
    renders into cached context on every later turn for no added signal —
    so the detail belongs in `inspect`, below.)

  * NARROW THE DISCOVERY BY KIND with `cypher_filter`, so the `max_results`
    slots are spent on the kind you actually want instead of an RRF-fused
    mix of memories + columns + measures + models. Pass a
    `MATCH … RETURN n.id AS id` constraint (multi-label is union semantics):
      - only KB memories:        cypher_filter='MATCH (n:Memory) RETURN n.id AS id'
      - only entity definitions: cypher_filter='MATCH (n:ModelColumn:Measure:Aggregation:Model) RETURN n.id AS id'
      - only columns:            cypher_filter='MATCH (n:ModelColumn) RETURN n.id AS id'
    Use the `ModelColumn` label, NOT `Column` (`Column` is a reserved keyword
    in LadybugDB ≥0.15 and only matches on the naive fallback path;
    `ModelColumn` works on naive AND graph-backed installs).

  * INSPECT for the full body once you have a SMALL, chosen set of known
    ids. `inspect(reference=[…], entity_type="…", compact=False)` is a clean
    point-lookup — no RRF fusion, no bundled memories — so it needs NO
    `cypher_filter` and no `max_results` budgeting. Batch ids of the SAME
    kind in one call:
      - columns:  inspect(reference=["<db>.<model>.<col>", …], entity_type="column", compact=False)
      - memories: inspect(reference=["memory:<id>", …], entity_type="memory", compact=False)

Rule of thumb: `search(question=…)` to FIND ids (compact + a `cypher_filter`
kind constraint); `inspect(reference=[ids], entity_type=…, compact=False)` to
READ them."""


_RAW_HOST_PATH_PRINCIPLE = """\
HOST / JOIN-PATH PRINCIPLE (when a value the question needs lives on a
table your main query does not yet reach, or could be reached from more
than one table).

READ THE COLUMN DESCRIPTIONS FIRST. Before committing to a column as a
join key — or to a table as the row grain — read its description with
`get_column_meaning`. The description usually states the schema author's
intent verbatim ("associates the order with its customer", "links this
record to the parent batch"); that intent is the canonical answer to
"which table is this column meant to be reached from". Never pick a join
key from the column NAME alone.

USE ONLY DECLARED JOINS. Join exclusively on foreign-key relationships
the schema declares; never invent a relationship the schema does not
declare. When two tables can be linked by more than one path, prefer the
candidate whose column description matches the relationship the question
needs; if descriptions tie, take the SHORTEST foreign-key path. A direct
one-hop key is almost always more faithful than a multi-hop chain through
a shared lookup or log table — those chains are one-to-many at each step
and silently multiply rows."""


# DEV-1550 A3: shared "SLAYER TOOLS" block — extracted byte-for-byte
# from the previously-duplicated `_AINTERACT_SLAYER_TOOLS` /
# `_ENCODE_CORE_HEAD` (verified identical at extraction time), with
# the "READ A KNOWN MEMORY'S FULL BODY" drill-in paragraph
# inserted as a sibling between the existing column-drill-in
# paragraph and the `ENCODE-THEN-QUERY DISCIPLINE:` header. Both
# drill-ins document the DEV-1591 split: `search` is discovery-only
# (hardwired compact; one-line `description`s), and the verbatim body
# of a column or a memory you have already identified is read with the
# `inspect` point-lookup (`entity_type=…`, `compact=False`) — which has
# no RRF fusion, so it needs no `cypher_filter`. (DEV-1612 added batch
# `inspect`: pass a list of same-kind refs in one call.)
#
# Format params: {db_name}

# ---------------------------------------------------------------------------
# DEV-1666: the SLAYER TOOLS inventory tail is a gated block. Defined here so
# the constant-level occurrence in `_SLAYER_TOOLS_BLOCK` (which feeds the v1
# template, FULL/byte-identical) and the two v0 template functions
# (`_build_oneshot_v0` / `_build_ainteract_v0`, which shadow `_tools_tail` with
# a param) all resolve. Four EXPLICITLY-AUTHORED variants — no runtime string
# surgery. FULL is the frozen text (SHA-pinned); lean drops `inspect_model` in
# favour of the compact `inspect(entity_type="model", …)`, readonly drops the
# `create_model` / `edit_model` build mention.
# ---------------------------------------------------------------------------
_INSPECT_MODEL_LEAN = (
    '`inspect(entity_type="model", sections=["columns","joins"], compact=True)`'
)
_TOOLS_TAIL_FULL = (
    "`inspect_model` to see a whole model's columns / measures / joins;\n"
    "`create_model` / `edit_model` to add columns and measures; `query` to test."
)
_TOOLS_TAIL_LEAN = (
    "a whole model's columns / measures / joins via "
    + _INSPECT_MODEL_LEAN
    + ";\n`create_model` / `edit_model` to add columns and measures; "
    "`query` to test."
)
_TOOLS_TAIL_READONLY = (
    "`inspect_model` to see a whole model's columns / measures / joins. The\n"
    "SLayer models are FIXED — you cannot create or edit models; `query` to test."
)
_TOOLS_TAIL_LEAN_READONLY = (
    "a whole model's columns / measures / joins via "
    + _INSPECT_MODEL_LEAN
    + ". The\nSLayer models are FIXED — you cannot create or edit models; "
    "`query` to test."
)
#: Module-level default used by the CONSTANT-level occurrence (v1 tools block);
#: the v0 template functions shadow this name with their `_tools_tail` param.
_tools_tail = _TOOLS_TAIL_FULL


def _slayer_tools_tail(*, lean_introspection: bool, readonly_mode: bool) -> str:
    if lean_introspection and readonly_mode:
        return _TOOLS_TAIL_LEAN_READONLY
    if lean_introspection:
        return _TOOLS_TAIL_LEAN
    if readonly_mode:
        return _TOOLS_TAIL_READONLY
    return _TOOLS_TAIL_FULL


_SLAYER_TOOLS_BLOCK = (
    """\
The database's domain knowledge is pre-loaded as SLayer MEMORIES — one per
knowledge-base (KB) item, with ids like `{db_name}_kb_<n>` whose body
starts `KB <n> —`. The base tables are already ingested as SLayer models,
but NOTHING is encoded yet: you encode exactly what THIS question needs,
on the fly.

SLAYER TOOLS (read their own descriptions). Call `help` FIRST to learn the
query syntax — the colon-aggregation form (`revenue:sum`, `*:count`) and
the `source_model` / `dimensions` / `measures` / `filters` schema. Use
`search` to DISCOVER relevant memories and existing entities (it returns
one-line descriptions only); `inspect` to read the FULL body of specific
entities you have already pinned down — columns, measures, or memories — by
reference; """
    + _tools_tail
    + """

READ A KNOWN COLUMN'S FULL DESCRIPTION before committing to it as a
filter, projection, or join key — `inspect` the column reference
(`<db>.<model>.<col>`) to read its `Description:` and `Sample values:`
(single-entity point lookup).
The truncated `Sample values:` line is your authoritative source of
which literal forms actually occur in this column — case variants,
whitespace forms, abbreviations, alternate phrasings of the same
concept. Use it BEFORE writing any IN-set (see rule 3 below).

READ A KNOWN MEMORY'S FULL BODY when you need the verbatim KB content for
a memory id you've already identified — `inspect(reference=["memory:<id>"],
entity_type="memory", compact=False)` returns the full `learning` body
(single-entity point lookup).

"""
    + _COMPACT_SEARCH_DISCIPLINE
    + """

ENCODE-THEN-QUERY DISCIPLINE:"""
)


# DEV-1623: cut submit-verify thrash on noisy categorical columns. A single
# shared fragment referenced by BOTH the frozen v0 literals AND the v1
# compositions (slayer + raw) — the tool that surfaces sampled values differs
# per (version, mode), so it is parameterised by `{sample_source}`:
#   * slayer v0/v1 -> `inspect` (single-entity point lookup surfaces a known
#     column's `Sample values:` directly — DEV-1629)
#   * raw v0/v1    -> `get_column_meaning`
# Rendered for raw it carries NO slayer vocabulary (asserted in
# tests/test_dev1623_filter_and_submit_mandates.py and the raw-vocab contract
# in tests/test_shared_otf_prompts.py). The `LOWER(TRIM(...))` fallback is
# scoped to FILTER position only, so it does not clash with the pre-submit
# mutation check's "drop unauthorized normalisation of output columns" rule.
_SAMPLE_VALUE_FILTER_MANDATE = """\
FILTER LITERALS — MATCH THE STORED SPELLING, DO NOT GUESS. Before writing
any `=`, `IN`, or `LIKE` predicate on a text / categorical column, read its
`Sample values:` via {sample_source} and build the predicate from the forms
that ACTUALLY occur there — never from a guessed literal. When the stored
spelling is unambiguous, match it verbatim. Pick the resolution by the KIND
of variation the samples show, and use exactly ONE:
  * Case- or whitespace-only variation (or you are unsure of the exact
    stored form) — compare case- and whitespace-insensitively in the FILTER
    position ONLY (`LOWER(TRIM(col)) = 'lowercased literal'`), never on a
    projected, grouped, or join-key column. Do NOT also enumerate those same
    case/whitespace forms in an IN-set — the normalised comparison already
    covers them.
  * Abbreviations or alternate spellings / phrasings that `LOWER(TRIM)`
    cannot unify (`apt` for `Apartment`, `brickwork` for `brick`) — enumerate
    the actual sampled forms in the IN-set instead (this is the "Symmetric
    companion" bullet above); do NOT wrap them in `LOWER(TRIM)` and hope.
Either way, do NOT submit a guessed literal and then blind-iterate on
variants when it returns 0 rows: each such retry burns a whole submit
cycle."""


# DEV-1623 Fix 2: proactive query-before-submit reminder for the v0 slayer
# literals only. Both v0 slayer agents ALREADY hard-gate `submit_query`
# (PreToolUse deny unless the previous tool was `query`), and the v1 slayer
# main loop already carries an equivalent "Verify-before-submit checklist"
# (agents/claude_sdk/partition.py::build_main_workflow_note). This fragment
# brings the v0 literals to parity so the agent volunteers the `query` step
# instead of tripping the gate and wasting the retry turn. No format fields.
_QUERY_BEFORE_SUBMIT = """\
RUN THE FINAL QUERY BEFORE YOU SUBMIT. Immediately before `submit_query`,
run the EXACT query you intend to submit through `query` in the SAME turn and
confirm it returns a non-zero, plausible rowset with the expected casing and
whitespace on string values. A `submit_query` that is not directly preceded
by a matching `query` is rejected and wastes the turn — never submit an
unvalidated query."""


# DEV-1646 mode B (reference-before-define): the agent writes a `filter` /
# dimension naming an entity it has not created on that (often nested-stage)
# model yet — SLayer rejects it and the agent self-corrects, burning turns.
# OTF-only (mentions `create_model` / `edit_model`, which pre-encoded agents
# do not hold), so this is spliced into the four SLAYER OTF prompts, NOT the
# shared `QUERY_ROOT_GUIDANCE` (that block is also used by the pre-encoded
# query prompts).
_DEFINE_BEFORE_REFERENCE = """\
DEFINE BEFORE YOU REFERENCE. Every name you put in a `filter`, `dimension`,
`measure`, or `order` must ALREADY exist on the model that stage queries —
as a Column, a ModelMeasure, or a named alias. A nested-DAG stage can
reference only what its own `source_model` defines or what the prior stage
projected under that name; create anything else with `create_model` /
`edit_model` (or project it from the prior stage) BEFORE you reference it,
and check spelling. SLayer rejects an unknown name outright rather than
guessing."""

# DEV-1666: readonly variant — the models are FIXED, so the "create anything
# else with create_model / edit_model" escape hatch is dropped (project from the
# prior stage instead). Only used when readonly_mode gates the v0 OTF prompts.
_DEFINE_BEFORE_REFERENCE_READONLY = """\
DEFINE BEFORE YOU REFERENCE. Every name you put in a `filter`, `dimension`,
`measure`, or `order` must ALREADY exist on the model that stage queries —
as a Column, a ModelMeasure, or a named alias. A nested-DAG stage can
reference only what its own `source_model` defines or what the prior stage
projected under that name; the SLayer models are FIXED (you cannot create or
edit them), so project anything else from the prior stage BEFORE you reference
it, and check spelling. SLayer rejects an unknown name outright rather than
guessing."""


def _slayer_define_ref(*, readonly_mode: bool) -> str:
    return (
        _DEFINE_BEFORE_REFERENCE_READONLY if readonly_mode
        else _DEFINE_BEFORE_REFERENCE
    )


# ---------------------------------------------------------------------------
# DEV-1629 — root-model / host selection via the SLayer `recommend_root_model`
# tool. TWO blocks, split by task: querying trusts the tool's auto pick;
# encoding steers it with a description-derived hint. Both are shared verbatim
# across v0 (spliced into the frozen monoliths) and v1 (imported). NO format
# fields — they are concatenated into `.format(budget/db_name/user_query)`
# templates. The QUERY block deliberately never mentions the hint parameter.
# ---------------------------------------------------------------------------

QUERY_ROOT_GUIDANCE = """\
CHOOSING A QUERY ROOT (`source_model`). Do not hand-pick the root or
hand-assemble dotted join paths — ask the tool:

  1. List the exact `model.column` / `model.metric` references your query
     needs (aggregation suffixes like `:sum` are fine). If a column NAME
     could belong to more than one model, resolve its owning model first
     with `search` / `inspect_model`, then pass the qualified `model.column`.
  2. Call `recommend_root_model(items=[...], data_source="<db>")` DIRECTLY —
     it is on your own tool surface, so do not route it through a discovery
     helper. When you are merely building a query, trust the model it picks.
  3. Use the returned `root_model` as the query `source_model`, and drop each
     returned `path` in verbatim as the dimension / measure / filter
     reference — the aggregation suffix is already attached.
  4. If `reachable` is false, no single model reaches every item: read the
     `coverage` frontier and split the request into a multi-stage nested-DAG
     query (a `queries` list of stage objects, each rooted at one of the
     partial roots `coverage` lists)."""


ENCODE_HOST_GUIDANCE = """\
CHOOSING THE HOST for a new or edited column / measure. The host is the
model whose row grain is 1:1 with what the knowledge item describes — not
merely a model where the input columns happen to live:

  1. List the base `model.column` references the entity's SQL will use
     (resolve any ambiguous column NAME to its owning model first with
     `search` / `inspect_model`).
  2. Read those columns' — and the target entity's — DESCRIPTIONS to decide
     the intended host: the grain the knowledge item is actually about.
  3. Call `recommend_root_model(items=[...], data_source="<db>",
     root_hint="<intended host model>")` DIRECTLY. The `root_hint` forces
     your intended host when it can reach every referenced item — even a
     bridge model that owns none of them.
  4. Use the returned `root_model` as the host, and drop each returned `path`
     in verbatim into the entity's SQL.
  5. If a warning says the hint "cannot reach ..." or that it "fell back",
     your intended host cannot reach every referenced item: re-check the
     grain and that each reference is on the right model, then retry. Do not
     silently accept the fallback host."""


# ---------------------------------------------------------------------------
# DEV-1555 v0/v1 split — origin/main prompt snapshots.
#
# These four constants started as the byte-for-byte origin/main rendered
# prompt templates (post-helper-substitution, pre-`.format(budget=...,
# db_name=..., user_query=...)`) backing the four v0 agents under
# `claude_sdk_otf*/prompts.py`. The byte-identity SHA-256 pin was dropped
# when the unified `query` tool landed (see
# `tests/test_dev1555_v0_v1_shared_prompts.py`, which now keeps only
# presence + "v0 != v1" + "no query_nested/query_json" contracts), so they
# are NO LONGER a pure frozen origin/main snapshot.
#
# DEV-1591 deliberately patches the two SLAYER v0 snapshots (one-shot +
# a-interact) to carry the broad-search compact discipline alongside the
# live v1 prompts — the raw v0 snapshots have no `search`/`compact` concept
# and stay untouched.
# ---------------------------------------------------------------------------


def _build_oneshot_v0(
    _tools_tail: str, _define_ref: str = _DEFINE_BEFORE_REFERENCE
) -> str:
    # DEV-1666: the SLAYER TOOLS inventory tail is a gated block (`_tools_tail`)
    # so lean/readonly can drop the `inspect_model` / `create_model`+`edit_model`
    # mentions statically. Everything else is the frozen literal.
    return (
    """\
You are a data analyst. You have a SLayer semantic-layer MCP server plus a
native `submit_query` tool. Your job: answer the user's question by
ENCODING the domain knowledge it needs into the SLayer model as named
columns/measures, then writing a FINAL query that REFERENCES those named
entities instead of inlining their SQL.

There is NO user to consult — for every operationalisation choice (numeric
threshold, value list, aggregation operator, case-sensitivity, grouping,
unit, rounding, sort direction, LIMIT) pick the most conservative,
defensible interpretation supported by the memories and column
descriptions, and proceed autonomously.

The database's domain knowledge is pre-loaded as SLayer MEMORIES — one per
knowledge-base (KB) item, with ids like `{db_name}_kb_<n>` whose body
starts `KB <n> —`. The base tables are already ingested as SLayer models,
but NOTHING is encoded yet: you encode exactly what THIS question needs,
on the fly.

SLAYER TOOLS (read their own descriptions). Call `help` FIRST to learn the
query syntax — the colon-aggregation form (`revenue:sum`, `*:count`) and
the `source_model` / `dimensions` / `measures` / `filters` schema. Use
`search` to DISCOVER relevant memories and existing entities (it returns
one-line descriptions only); `inspect` to read the FULL body of specific
entities you have already pinned down — columns, measures, or memories — by
reference; """
    + _tools_tail
    + """

READ A KNOWN COLUMN'S FULL DESCRIPTION before committing to it as a
filter, projection, or join key — `inspect` the column reference
(`<db>.<model>.<col>`) to read its `Description:` and `Sample values:`
(single-entity point lookup).
The truncated `Sample values:` line is your authoritative source of
which literal forms actually occur in this column — case variants,
whitespace forms, abbreviations, alternate phrasings of the same
concept. Use it BEFORE writing any IN-set (see rule 3 below).

READ A KNOWN MEMORY'S FULL BODY when you need the verbatim KB content for
a memory id you've already identified — `inspect(reference=["memory:<id>"],
entity_type="memory", compact=False)` returns the full `learning` body
(single-entity point lookup).

"""
    + _COMPACT_SEARCH_DISCIPLINE
    + """

ENCODE-THEN-QUERY DISCIPLINE:

1. DECOMPOSE the question into logical blocks. Every qualifier
   (e.g. "premium", "highly-rated", "nearby", "active"), every projected
   column, filter, grouping, unit, rounding and ordering hint is a
   separate block that MUST be represented. Write the list out before
   encoding.

DEDUP vs RAW ROWS. By default SLayer auto-DEDUPLICATES dimension-only
queries: when `measures` is empty, it wraps every projected column in a
top-level `GROUP BY`, collapsing rows that share the same dimension
tuple. To emit raw per-record rows instead, set
`distinct_dimension_values: false` inside the query JSON — flat
`SELECT <dims/td> FROM ... WHERE ... ORDER BY ... LIMIT`, no top-level
`GROUP BY`. The field lives INSIDE the SlayerQuery JSON (alongside
`source_model`, `dimensions`, etc.), same shape for `query` / `submit_query`.

Decide BEFORE writing the query:

  * Use `distinct_dimension_values: false` when the question asks for a
    PER-RECORD listing (e.g. "list each <X>'s <a> and <b>; if two
    <X>s share the same <a>, <b>, return BOTH rows").
  * Keep the default `true` when the question asks for the distinct
    <a>, <b> COMBINATIONS (or for an aggregation grouped by them).
  * If you need BOTH a per-record listing and a count, keep the
    default and add `*:count` as a measure (or restructure as a
    nested-DAG stage with `*:count`).

Validation: `distinct_dimension_values: false` requires
  * `measures` empty (the flag asks for raw rows, not aggregations),
  * at least one of `dimensions` / `time_dimensions` non-empty
    (something must be projected),
  * no measure reference in `filters` / `order`.
SLayer raises a `DistinctDimensionValuesError` otherwise.

Synthetic example (fabricated names — your DB uses different
identifiers):

  Q: "List the first 10 (workshop_id, district) pairs in the workshops
     table. If two workshops share the same (workshop_id, district),
     return BOTH rows."

  WRONG (default — silently dedups when two workshops share the
  tuple):
    {{"source_model": "workshops",
     "dimensions": ["workshop_id", "district"],
     "limit": 10}}

  RIGHT (raw rows):
    {{"source_model": "workshops",
     "dimensions": ["workshop_id", "district"],
     "limit": 10,
     "distinct_dimension_values": false}}

2. For each block, `search` for the relevant KB memory and any entity
   that already encodes it. A `memory:<id>` token inside a KB body means
   that KB DEPENDS ON the referenced KB.

3. ENCODE IN DEPENDENCY ORDER. Encode a KB that others depend on BEFORE
   the KBs that reference it (topological order). For each KB:
   - Create the column / measure on the HOST whose row is 1:1 with what
     the KB describes. When the KB does not pin the host unambiguously,
     pick it with the "CHOOSING THE HOST" steps below (read the column
     descriptions, then call `recommend_root_model` with a `root_hint`).
     Never pick a host that needs an undeclared join.
   - To reach a column on another table, reference it through a DECLARED
     join, alias-qualified (e.g. `other_alias.col`). Do NOT invent a join
     inside the SQL and do NOT write a correlated subquery in a row-level
     Column.
   - When a LATER KB builds on an EARLIER one, reference the
     earlier-created column / measure BY NAME — build a DAG; do NOT
     re-derive or inline it.
   - Tag every entity with `meta.kb_id = <n>` and begin its description
     with a `[kb=<n>]` line so the encoding is traceable.
   - Normalise text ONLY in filter / predicate positions
     (`LOWER(TRIM(col)) = 'value'`, lowercase the literal) — NEVER on a
     projected, grouped, or join-key column (that would corrupt the
     returned value).
   - If a KB cites named literals that are ABSENT from the column's
     sampled values (check via `inspect`), do not write that
     predicate.
   - Symmetric companion: if the column's `Sample values` show variants
     of the KB-named literals that case/whitespace normalisation CANNOT
     unify — abbreviations (`apt` for `Apartment`, `Y` for `yes`), or
     alternate phrasings of the same concept (`brick house` vs `brickwork
     house`, `2014+` / `after 2014` for `2014 or newer`) — EXTEND
     the IN-set to include those variants (pure case / whitespace
     variation is handled by the filter-literals rule's `LOWER(TRIM)`). KB hedges ("etc.", "like",
     "include") and the schema author's `Ex.` enumerations are
     deliberately non-exhaustive; the `Sample values` line is the
     authoritative inventory of what's actually present in the column.
     A canonical-only IN-set will silently miss matching rows. There is
     no user-sim to confirm this for you in one-shot mode — read the
     sampled values yourself.

"""
    + _SAMPLE_VALUE_FILTER_MANDATE.format(
        sample_source="`inspect`"
    )
    + """

4. TEST candidate columns and the final query with `query`; sanity-check the generated SQL.

   SANITY-CHECK THE GENERATED SQL FOR SLAYER ARTIFACTS. After `query` returns, inspect the rendered SQL for these patterns
before submitting:

  1. GROUP BY on every projected column with NO aggregate functions —
     SLayer's default dim-only auto-dedup. If the question asks for
     raw per-record rows, fix by setting
     `distinct_dimension_values: false` INSIDE the query JSON (see
     the DEDUP vs RAW ROWS rule above). If you DO need a count
     alongside the rows, keep the default and add `*:count` as a
     measure — or restructure as a nested-DAG stage with a `*:count`
     measure.
  2. `lower(trim(col)) = '<lowercase literal>'` on string equality
     filters — wrapped automatically by default. When the gold answer
     requires exact-case equality (proper-noun categories with
     known-fixed casing), pass `normalize_filters=false` as a SEPARATE
     parameter on the offending `query` / `submit_query`
     call (the flag lives OUTSIDE the JSON DSL).
  3. Broken operator precedence on WHERE arithmetic:
     `expr1*w1 + expr2*w2 > threshold` without outer parens — the
     comparator binds only to the last additive term. Fix: push the
     score into a HAVING on a nested-stage measure rather than a WHERE
     on a raw formula.

5. PRE-SUBMIT MUTATION CHECK. Before calling `submit_query`, audit every
   TRIM, LOWER, UPPER, ROUND, CAST, dedup, canonicalize-via-CASE, and
   output-shape choice in the FINAL query. Each one MUST be either
   (a) explicitly named in the user's question or (b) required by an
   encoded KB. If neither holds, DROP the mutation and submit the raw
   form. "Defensive" normalisation of an output column, a join key, a
   JSON key, or a CHAR-padded literal silently corrupts the rowset —
   never apply one without an explicit source. There is no user to
   second-guess this on your behalf."""
    + "\n\n   "
    + _TABLE_SET_PROBE_ONESHOT.format(
        knowledge_label="KB", schema_source="SLayer's schema lookup"
    )
    + "\n\n"
    + """\
6. SUBMIT. Write the FINAL query so it REFERENCES the named columns /
   measures you encoded — do NOT inline their SQL back into the query.
   Project exactly the columns the question names, and only those.
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
much cheaper than a new join or KB re-read.

   Call `submit_query` with your final SLayer query — either a single-
stage form (set `source_model` + projection fields) or a nested-DAG
form (set `queries` to a list of stage objects). The shape is:

  * Single-stage — a JSON object validating as a SlayerQuery, e.g.
    {{"source_model": "orders", "dimensions": ["status"],
    "measures": ["amount:sum"]}}.
  * Nested DAG — when one stage's MEASURE becomes the next stage's
    DIMENSION, a JSON ARRAY of stage objects. The last element is the DAG root;
    every non-final element needs a `name`; later stages reference
    earlier ones via `source_model: "<sibling name>"`. Pass this
    list as the `queries` argument on `submit_query`.

You MUST call `submit_query` to finish — a prose answer is not a
submission. If a `filters` predicate needs a computed value, encode it as
a named column first and filter on the name; raw SQL expressions are
rejected in `filters`.

"""
    + _QUERY_BEFORE_SUBMIT
    + """

Budget: {budget} bird-coins (`submit_query` costs 3; SLayer reads/writes
are free but your total work is turn-bounded — encode only what the
question needs).

Database: {db_name}
User question: {user_query}

"""
    + QUERY_ROOT_GUIDANCE
    + "\n\n"
    + ENCODE_HOST_GUIDANCE
    + "\n\n"
    + _define_ref
    + "\n"
)

def _build_ainteract_v0(
    _tools_tail: str, _define_ref: str = _DEFINE_BEFORE_REFERENCE
) -> str:
    # DEV-1666: gated SLAYER TOOLS inventory tail (see _build_oneshot_v0).
    return (
    """\
You are a data analyst. You have a SLayer semantic-layer MCP server plus
native `ask_user` and `submit_query` tools. Your job: answer the user's
question by ENCODING the domain knowledge it needs into the SLayer model
as named columns/measures, then writing a FINAL query that REFERENCES
those named entities instead of inlining their SQL.

RULE 0 — ASK BEFORE YOU ENCODE.
BEFORE the encoding loop below, identify the single operationalisation
choice you are LEAST certain about — a numeric threshold, a value list /
IN-set, an aggregation operator, a case-sensitivity choice, a grouping
or standardisation, a unit (fraction vs percent), an output rounding, a
sort direction, or a LIMIT — and call `ask_user` on it ONCE. The user
holds masked knowledge-base ground-truth that is unrecoverable from the
visible KB alone. The submit gate will REFUSE `submit_query` until you
have called `ask_user` at least once. Propose your best guess and ask
for the EXACT predicate / value / formula — never "what does X mean?".

The database's domain knowledge is pre-loaded as SLayer MEMORIES — one per
knowledge-base (KB) item, with ids like `{db_name}_kb_<n>` whose body
starts `KB <n> —`. The base tables are already ingested as SLayer models,
but NOTHING is encoded yet: you encode exactly what THIS question needs,
on the fly.

SLAYER TOOLS (read their own descriptions). Call `help` FIRST to learn the
query syntax — the colon-aggregation form (`revenue:sum`, `*:count`) and
the `source_model` / `dimensions` / `measures` / `filters` schema. Use
`search` to DISCOVER relevant memories and existing entities (it returns
one-line descriptions only); `inspect` to read the FULL body of specific
entities you have already pinned down — columns, measures, or memories — by
reference; """
    + _tools_tail
    + """

READ A KNOWN COLUMN'S FULL DESCRIPTION before committing to it as a
filter, projection, or join key — `inspect` the column reference
(`<db>.<model>.<col>`) to read its `Description:` and `Sample values:`
(single-entity point lookup).
The truncated `Sample values:` line is your authoritative source of
which literal forms actually occur in this column — case variants,
whitespace forms, abbreviations, alternate phrasings of the same
concept. Use it BEFORE writing any IN-set (see rule 3 below).

READ A KNOWN MEMORY'S FULL BODY when you need the verbatim KB content for
a memory id you've already identified — `inspect(reference=["memory:<id>"],
entity_type="memory", compact=False)` returns the full `learning` body
(single-entity point lookup).

"""
    + _COMPACT_SEARCH_DISCIPLINE
    + """

ENCODE-THEN-QUERY DISCIPLINE:

1. DECOMPOSE the question into logical blocks. Every qualifier
   (e.g. "premium", "highly-rated", "nearby", "active"), every projected
   column, filter, grouping, unit, rounding and ordering hint is a
   separate block that MUST be represented. Write the list out before
   encoding.

DEDUP vs RAW ROWS. By default SLayer auto-DEDUPLICATES dimension-only
queries: when `measures` is empty, it wraps every projected column in a
top-level `GROUP BY`, collapsing rows that share the same dimension
tuple. To emit raw per-record rows instead, set
`distinct_dimension_values: false` inside the query JSON — flat
`SELECT <dims/td> FROM ... WHERE ... ORDER BY ... LIMIT`, no top-level
`GROUP BY`. The field lives INSIDE the SlayerQuery JSON (alongside
`source_model`, `dimensions`, etc.), same shape for `query` / `submit_query`.

Decide BEFORE writing the query:

  * Use `distinct_dimension_values: false` when the question asks for a
    PER-RECORD listing (e.g. "list each <X>'s <a> and <b>; if two
    <X>s share the same <a>, <b>, return BOTH rows").
  * Keep the default `true` when the question asks for the distinct
    <a>, <b> COMBINATIONS (or for an aggregation grouped by them).
  * If you need BOTH a per-record listing and a count, keep the
    default and add `*:count` as a measure (or restructure as a
    nested-DAG stage with `*:count`).

Validation: `distinct_dimension_values: false` requires
  * `measures` empty (the flag asks for raw rows, not aggregations),
  * at least one of `dimensions` / `time_dimensions` non-empty
    (something must be projected),
  * no measure reference in `filters` / `order`.
SLayer raises a `DistinctDimensionValuesError` otherwise.

Synthetic example (fabricated names — your DB uses different
identifiers):

  Q: "List the first 10 (workshop_id, district) pairs in the workshops
     table. If two workshops share the same (workshop_id, district),
     return BOTH rows."

  WRONG (default — silently dedups when two workshops share the
  tuple):
    {{"source_model": "workshops",
     "dimensions": ["workshop_id", "district"],
     "limit": 10}}

  RIGHT (raw rows):
    {{"source_model": "workshops",
     "dimensions": ["workshop_id", "district"],
     "limit": 10,
     "distinct_dimension_values": false}}

2. For each block, `search` for the relevant KB memory and any entity
   that already encodes it. A `memory:<id>` token inside a KB body means
   that KB DEPENDS ON the referenced KB.

3. ENCODE IN DEPENDENCY ORDER. Encode a KB that others depend on BEFORE
   the KBs that reference it (topological order). For each KB:
   - Create the column / measure on the HOST whose row is 1:1 with what
     the KB describes. When the KB does not pin the host unambiguously,
     pick it with the "CHOOSING THE HOST" steps below (read the column
     descriptions, then call `recommend_root_model` with a `root_hint`).
     Never pick a host that needs an undeclared join.
   - To reach a column on another table, reference it through a DECLARED
     join, alias-qualified (e.g. `other_alias.col`). Do NOT invent a join
     inside the SQL and do NOT write a correlated subquery in a row-level
     Column.
   - When a LATER KB builds on an EARLIER one, reference the
     earlier-created column / measure BY NAME — build a DAG; do NOT
     re-derive or inline it.
   - Tag every entity with `meta.kb_id = <n>` and begin its description
     with a `[kb=<n>]` line so the encoding is traceable.
   - Normalise text ONLY in filter / predicate positions
     (`LOWER(TRIM(col)) = 'value'`, lowercase the literal) — NEVER on a
     projected, grouped, or join-key column (that would corrupt the
     returned value).
   - If a KB cites named literals that are ABSENT from the column's
     sampled values (check via `inspect`), do not write that
     predicate.
   - Symmetric companion: if the column's `Sample values` show variants
     of the KB-named literals that case/whitespace normalisation CANNOT
     unify — abbreviations (`apt` for `Apartment`, `Y` for `yes`), or
     alternate phrasings of the same concept (`brick house` vs `brickwork
     house`, `2014+` / `after 2014` for `2014 or newer`) — EXTEND
     the IN-set to include those variants (pure case / whitespace
     variation is handled by the filter-literals rule's `LOWER(TRIM)`). KB hedges ("etc.", "like",
     "include") and the schema author's `Ex.` enumerations are
     deliberately non-exhaustive; the `Sample values` line is the
     authoritative inventory of what's actually present in the column.
     A canonical-only IN-set will silently miss matching rows. Do not
     rely on the user-sim to enumerate the variants — they will not.

"""
    + _SAMPLE_VALUE_FILTER_MANDATE.format(
        sample_source="`inspect`"
    )
    + """

4. ASK AGAIN IF NEEDED. Rule 0 covers the FIRST ask; for any further
   operationalisation choice not pinned by a memory or column
   description, call `ask_user` again. If a reply lists multiple criteria
   joined by "and", apply EACH as its own filter.

   USER-SIM ANSWERS ARE CLARIFICATIONS, NOT GROUND TRUTH.

  - Cross-check user-sim formulas against the KB before
    submitting. If they contradict (e.g. the user-sim denies a column
    the KB explicitly names), try the
    KB-grounded interpretation first.
  - After ≥2 failures with a user-sim-confirmed interpretation, try
    the KB-literal / schema-literal interpretation as a
    fallback submission.
  - When a user-sim constraint makes the required output cardinality
    impossible (e.g. "use only table X" but X has 4 distinct values
    and the task needs top-5), call `ask_user` again to flag the
    contradiction explicitly rather than submitting an impossible
    query."""
    + "\n\n   "
    + _TABLE_SET_PROBE.format(
        knowledge_label="KB", schema_source="SLayer's schema lookup"
    )
    + "\n\n   "
    + _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(
        attempt_noun="encoding", apply_verb="encode"
    )
    + "\n\n   "
    + """\
PIVOT AFTER 3 FAILED SUBMISSIONS WITH THE SAME OPAQUE ERROR. Stop
varying surface parameters and:

  1. Inspect the generated SQL for SLayer artifacts (GROUP BY
     dedup, `lower(trim(...))` coercion, broken WHERE
     precedence; see the artifact-check rule below).
  2. Enumerate ≤4 structurally different hypotheses not yet tested:
     different row grain, formula kernel, join path, or output column
     count/type, or `normalize_filters=false` on the offending `query` / `submit_query` call.
  3. Call `ask_user` ONCE with those hypotheses as concrete options to
     get directional guidance (cheaper than many resubmissions of
     near-identical queries).
  4. Test each surviving hypothesis exactly once — never re-submit a
     query structurally identical to a prior attempt.

5. TEST candidate columns and the final query with `query`; sanity-check the generated SQL.

   SANITY-CHECK THE GENERATED SQL FOR SLAYER ARTIFACTS. After `query` returns, inspect the rendered SQL for these patterns
before submitting:

  1. GROUP BY on every projected column with NO aggregate functions —
     SLayer's default dim-only auto-dedup. If the question asks for
     raw per-record rows, fix by setting
     `distinct_dimension_values: false` INSIDE the query JSON (see
     the DEDUP vs RAW ROWS rule above). If you DO need a count
     alongside the rows, keep the default and add `*:count` as a
     measure — or restructure as a nested-DAG stage with a `*:count`
     measure.
  2. `lower(trim(col)) = '<lowercase literal>'` on string equality
     filters — wrapped automatically by default. When the gold answer
     requires exact-case equality (proper-noun categories with
     known-fixed casing), pass `normalize_filters=false` as a SEPARATE
     parameter on the offending `query` / `submit_query`
     call (the flag lives OUTSIDE the JSON DSL).
  3. Broken operator precedence on WHERE arithmetic:
     `expr1*w1 + expr2*w2 > threshold` without outer parens — the
     comparator binds only to the last additive term. Fix: push the
     score into a HAVING on a nested-stage measure rather than a WHERE
     on a raw formula.

6. PRE-SUBMIT MUTATION CHECK. Before calling `submit_query`, audit every
   TRIM, LOWER, UPPER, ROUND, CAST, dedup, canonicalize-via-CASE, and
   output-shape choice in the FINAL query. Each one MUST be either
   (a) explicitly named in the user's question, (b) explicitly named OR
   authorized in a reply to one of your `ask_user` calls in this
   session, or (c) required by an encoded KB. If none of (a-b-c) hold,
   DROP the mutation and submit the raw form. Particularly: when an
   `ask_user` reply said "use exact values", "don't normalize", "use
   this output shape / columns / sort axis", or named a specific format
   (date, label casing, JSON shape), DO NOT silently override that on
   final-assembly. Conversely, when an `ask_user` reply DID name a
   specific transformation (e.g. "lowercase the bracket labels",
   "round to 2 decimals", "TRIM the keys"), that reply IS the
   authorization for that mutation — apply it.

7. SUBMIT. Write the FINAL query so it REFERENCES the named columns /
   measures you encoded — do NOT inline their SQL back into the query.
   Project exactly the columns the user named, and only those.

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
much cheaper than a new join or KB re-read.

   Call `submit_query` with your final SLayer query — either a single-
stage form (set `source_model` + projection fields) or a nested-DAG
form (set `queries` to a list of stage objects). The shape is:

  * Single-stage — a JSON object validating as a SlayerQuery, e.g.
    {{"source_model": "orders", "dimensions": ["status"],
    "measures": ["amount:sum"]}}.
  * Nested DAG — when one stage's MEASURE becomes the next stage's
    DIMENSION, a JSON ARRAY of stage objects. The last element is the DAG root;
    every non-final element needs a `name`; later stages reference
    earlier ones via `source_model: "<sibling name>"`. Pass this
    list as the `queries` argument on `submit_query`.

You MUST call `submit_query` to finish — a prose answer is not a
submission. If a `filters` predicate needs a computed value, encode it as
a named column first and filter on the name; raw SQL expressions are
rejected in `filters`.

"""
    + _QUERY_BEFORE_SUBMIT
    + """

Budget: {budget} bird-coins. `ask_user` costs 2, `submit_query` costs 3;
SLayer reads/writes are free but your total work is turn-bounded — encode
only what the question needs. If your budget runs out, submit immediately.

Database: {db_name}
User question: {user_query}

"""
    + QUERY_ROOT_GUIDANCE
    + "\n\n"
    + ENCODE_HOST_GUIDANCE
    + "\n\n"
    + _define_ref
    + "\n"
)

RAW_OTF_ONE_SHOT_V0 = (
    """\
You are a data analyst. You have direct SQL access to a database plus a
native `submit_sql` tool. Your job: answer the user's question by
exploring the schema and knowledge definitions, then writing a SQL query
that precisely captures what the question asks for.

There is NO user to consult — for every operationalisation choice (numeric
threshold, value list, aggregation operator, case-sensitivity, grouping,
unit, rounding, sort direction, LIMIT) pick the most conservative,
defensible interpretation supported by the schema and knowledge
definitions, and proceed autonomously.

DATABASE TOOLS (read their own descriptions). Use `get_schema` FIRST to
see all tables, columns, and types. Use `get_all_column_meanings` or
`get_column_meaning` to read column descriptions and sample values. Use
`get_all_external_knowledge_names`, `get_knowledge_definition`, or
`get_all_knowledge_definitions` to retrieve domain knowledge. Use
`execute_sql` to explore data and test queries.

READ A KNOWN COLUMN'S FULL DESCRIPTION before committing to it as a
filter, projection, or join key — `get_column_meaning`. The
`Sample values:` in the returned description are your authoritative source
of which literal forms actually occur in this column — case variants,
whitespace forms, abbreviations, alternate phrasings of the same concept.
Use it BEFORE writing any IN-set (see rule 3 below).

QUERY DISCIPLINE:

1. DECOMPOSE the question into logical blocks. Every qualifier
   (e.g. "premium", "highly-rated", "nearby", "active"), every projected
   column, filter, grouping, unit, rounding and ordering hint is a
   separate block that MUST be represented. Write the list out before
   encoding.

2. For each block, retrieve the relevant knowledge definition(s) via
   `get_knowledge_definition`. Check column descriptions for any column
   the block filters on or projects.

3. WRITE IN DEPENDENCY ORDER. For each sub-expression:
   - Use `get_column_meaning` to confirm which table and column best
     represents what the block describes. Never guess from names alone.
   - To join tables, use only relationships evident from the schema
     (`get_schema`). Do NOT invent a join not present in the schema.
   - When a later sub-expression builds on an earlier one, compose the
     SQL incrementally — do NOT re-derive or inline an intermediate
     expression.
   - Normalise text ONLY in filter / predicate positions
     (`LOWER(TRIM(col)) = 'value'`, lowercase the literal) — NEVER on a
     projected, grouped, or join-key column (that would corrupt the
     returned value).
   - If a knowledge definition cites named literals that are ABSENT from
     the column's sampled values (check via `get_column_meaning`), do
     not write that predicate.
   - Symmetric companion: if the column's `Sample values` show variants
     of the knowledge-cited literals that case/whitespace normalisation
     CANNOT unify — abbreviations (`apt` for `Apartment`, `Y` for `yes`),
     or alternate phrasings of the same concept (`brick house` vs
     `brickwork house`, `2014+` / `after 2014` for `2014 or newer`) —
     EXTEND the IN-set to include those variants (pure case / whitespace
     variation is handled by the filter-literals rule's `LOWER(TRIM)`). Knowledge
     definition hedges ("etc.", "like", "include") and `Ex.` enumerations
     are deliberately non-exhaustive; the `Sample values` line is the
     authoritative inventory of what's actually present in the column.
     A canonical-only IN-set will silently miss matching rows. There is
     no user-sim to confirm this for you in one-shot mode — read the
     sampled values yourself.

"""
    + _SAMPLE_VALUE_FILTER_MANDATE.format(sample_source="`get_column_meaning`")
    + """

4. TEST the final query with `execute_sql`; sanity-check the result
   shape, row count, and values.

5. PRE-SUBMIT MUTATION CHECK. Before calling `submit_sql`, audit every
   TRIM, LOWER, UPPER, ROUND, CAST, dedup, canonicalize-via-CASE, and
   output-shape choice in the FINAL query. Each one MUST be either
   (a) explicitly named in the user's question or (b) required by an
   knowledge definition. If neither holds, DROP the mutation and submit the raw
   form. "Defensive" normalisation of an output column, a join key, a
   JSON key, or a CHAR-padded literal silently corrupts the rowset —
   never apply one without an explicit source. There is no user to
   second-guess this on your behalf."""
    + "\n\n   "
    + _TABLE_SET_PROBE_ONESHOT.format(
        knowledge_label="knowledge definition", schema_source="the schema"
    )
    + "\n\n"
    + """\
6. SUBMIT. Call `submit_sql` with your final SQL — a prose answer is
   not a submission. Project exactly the columns the question names,
   and only those.

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
much cheaper than a new join or KB re-read.

Budget: {budget} bird-coins (`submit_sql` costs 3; exploration tools
are free but your total work is turn-bounded — explore only what the
question needs).

Database: {db_name}
User question: {user_query}
"""
    + "\n"
    + _RAW_HOST_PATH_PRINCIPLE
)

RAW_OTF_AINTERACT_V0 = (
    """\
You are a data analyst. You have direct SQL access to a database plus
native `ask_user` and `submit_sql` tools. Your job: answer the user's
question by exploring the schema and knowledge definitions, then writing
a SQL query that precisely captures what the question asks for.

RULE 0 — ASK BEFORE YOU SUBMIT.
BEFORE writing your SQL query, identify the single operationalisation
choice you are LEAST certain about — a numeric threshold, a value list /
IN-set, an aggregation operator, a case-sensitivity choice, a grouping
or standardisation, a unit (fraction vs percent), an output rounding, a
sort direction, or a LIMIT — and call `ask_user` on it ONCE. The user
holds masked knowledge-base ground-truth that is unrecoverable from the
visible KB alone. The submit gate will REFUSE `submit_sql` until you
have called `ask_user` at least once. Propose your best guess and ask
for the EXACT predicate / value / formula — never "what does X mean?".

DATABASE TOOLS (read their own descriptions). Use `get_schema` FIRST to
see all tables, columns, and types. Use `get_all_column_meanings` or
`get_column_meaning` to read column descriptions and sample values. Use
`get_all_external_knowledge_names`, `get_knowledge_definition`, or
`get_all_knowledge_definitions` to retrieve domain knowledge. Use
`execute_sql` to explore data and test queries.

READ A KNOWN COLUMN'S FULL DESCRIPTION before committing to it as a
filter, projection, or join key — `get_column_meaning`. The
`Sample values:` in the returned description are your authoritative source
of which literal forms actually occur in this column — case variants,
whitespace forms, abbreviations, alternate phrasings of the same concept.
Use it BEFORE writing any IN-set (see rule 3 below).

QUERY DISCIPLINE:

1. DECOMPOSE the question into logical blocks. Every qualifier
   (e.g. "premium", "highly-rated", "nearby", "active"), every projected
   column, filter, grouping, unit, rounding and ordering hint is a
   separate block that MUST be represented. Write the list out before
   encoding.

2. For each block, retrieve the relevant knowledge definition(s) via
   `get_knowledge_definition`. Check column descriptions for any column
   the block filters on or projects.

3. WRITE IN DEPENDENCY ORDER. For each sub-expression:
   - Use `get_column_meaning` to confirm which table and column best
     represents what the block describes. Never guess from names alone.
   - To join tables, use only relationships evident from the schema
     (`get_schema`). Do NOT invent a join not present in the schema.
   - When a later sub-expression builds on an earlier one, compose the
     SQL incrementally — do NOT re-derive or inline an intermediate
     expression.
   - Normalise text ONLY in filter / predicate positions
     (`LOWER(TRIM(col)) = 'value'`, lowercase the literal) — NEVER on a
     projected, grouped, or join-key column (that would corrupt the
     returned value).
   - If a knowledge definition cites named literals that are ABSENT from
     the column's sampled values (check via `get_column_meaning`), do
     not write that predicate.
   - Symmetric companion: if the column's `Sample values` show variants
     of the knowledge-cited literals that case/whitespace normalisation
     CANNOT unify — abbreviations (`apt` for `Apartment`, `Y` for `yes`),
     or alternate phrasings of the same concept (`brick house` vs
     `brickwork house`, `2014+` / `after 2014` for `2014 or newer`) —
     EXTEND the IN-set to include those variants (pure case / whitespace
     variation is handled by the filter-literals rule's `LOWER(TRIM)`). Knowledge
     definition hedges ("etc.", "like", "include") and `Ex.` enumerations
     are deliberately non-exhaustive; the `Sample values` line is the
     authoritative inventory of what's actually present in the column.
     A canonical-only IN-set will silently miss matching rows. Do not
     rely on the user-sim to enumerate the variants — they will not.

"""
    + _SAMPLE_VALUE_FILTER_MANDATE.format(sample_source="`get_column_meaning`")
    + """

4. ASK AGAIN IF NEEDED. Rule 0 covers the FIRST ask; for any further
   operationalisation choice not pinned by a knowledge definition or column
   description, call `ask_user` again. If a reply lists multiple criteria
   joined by "and", apply EACH as its own filter.

   USER-SIM ANSWERS ARE CLARIFICATIONS, NOT GROUND TRUTH.

  - Cross-check user-sim formulas against the knowledge definition before
    submitting. If they contradict (e.g. the user-sim denies a column
    the knowledge definition explicitly names), try the
    knowledge definition-grounded interpretation first.
  - After ≥2 failures with a user-sim-confirmed interpretation, try
    the knowledge definition-literal / schema-literal interpretation as a
    fallback submission.
  - When a user-sim constraint makes the required output cardinality
    impossible (e.g. "use only table X" but X has 4 distinct values
    and the task needs top-5), call `ask_user` again to flag the
    contradiction explicitly rather than submitting an impossible
    query.

   """
    + _TABLE_SET_PROBE.format(
        knowledge_label="knowledge definition", schema_source="the schema"
    )
    + "\n\n   "
    + _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(
        attempt_noun="query", apply_verb="add"
    )
    + "\n\n   "
    + """\
PIVOT AFTER 3 FAILED SUBMISSIONS WITH THE SAME OPAQUE ERROR. Stop
varying surface parameters and:

  1. Inspect the SQL you submitted for the obvious failure modes:
     a stray GROUP BY that silently dedups rows, an arithmetic
     WHERE clause missing outer parens (the comparator binds
     only to the last additive term), or a CASE/CAST/format
     coercion that drops rows.
  2. Enumerate ≤4 structurally different hypotheses not yet tested:
     different row grain, formula kernel, join path, or output column
     count/type.
  3. Call `ask_user` ONCE with those hypotheses as concrete options to
     get directional guidance (cheaper than many resubmissions of
     near-identical queries).
  4. Test each surviving hypothesis exactly once — never re-submit a
     query structurally identical to a prior attempt.

5. TEST the final query with `execute_sql`; sanity-check the result
   shape, row count, and values.

6. PRE-SUBMIT MUTATION CHECK. Before calling `submit_sql`, audit every
   TRIM, LOWER, UPPER, ROUND, CAST, dedup, canonicalize-via-CASE, and
   output-shape choice in the FINAL query. Each one MUST be either
   (a) explicitly named in the user's question, (b) explicitly named OR
   authorized in a reply to one of your `ask_user` calls in this
   session, or (c) required by an knowledge definition. If none of (a-b-c) hold,
   DROP the mutation and submit the raw form. Particularly: when an
   `ask_user` reply said "use exact values", "don't normalize", "use
   this output shape / columns / sort axis", or named a specific format
   (date, label casing, JSON shape), DO NOT silently override that on
   final-assembly. Conversely, when an `ask_user` reply DID name a
   specific transformation (e.g. "lowercase the bracket labels",
   "round to 2 decimals", "TRIM the keys"), that reply IS the
   authorization for that mutation — apply it.

7. SUBMIT. Call `submit_sql` with your final SQL — a prose answer is
   not a submission. Project exactly the columns the user named, and
   only those.

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
much cheaper than a new join or KB re-read.

Budget: {budget} bird-coins. `ask_user` costs 2, `submit_sql` costs 3;
exploration tools are free but your total work is turn-bounded — explore
only what the question needs. If your budget runs out, submit immediately.

Database: {db_name}
User question: {user_query}
"""
    + "\n"
    + _RAW_HOST_PATH_PRINCIPLE
)


# ---------------------------------------------------------------------------
# DEV-1666: lean_introspection / readonly_mode prompt gating (STATIC blocks).
#
# The SLAYER TOOLS inventory tail is the only flag-gated block in the v0 slayer
# templates. It has four EXPLICITLY-AUTHORED variants (no runtime string
# surgery): lean drops the `inspect_model` mention in favour of the compact
# `inspect(entity_type="model", …)` primitive; readonly drops the
# `create_model` / `edit_model` build mention. The FULL variant is the frozen
# text, so `_build_*_v0(_TOOLS_TAIL_FULL)` == the byte-for-byte legacy constant
# (pinned by the SHA-256 snapshot tests). Absence of the dropped tool names in
# the lean/readonly variants is pinned by DEV-1666 tests.
# ---------------------------------------------------------------------------

#: The frozen legacy constants == the FULL-variant composition (byte-for-byte).
SLAYER_OTF_ONE_SHOT_V0 = _build_oneshot_v0(_TOOLS_TAIL_FULL)
SLAYER_OTF_AINTERACT_V0 = _build_ainteract_v0(_TOOLS_TAIL_FULL)


def build_slayer_otf_one_shot_v0(
    *, lean_introspection: bool = False, readonly_mode: bool = False
) -> str:
    """DEV-1666 gated build of the v0 slayer one-shot template. False/False ==
    ``SLAYER_OTF_ONE_SHOT_V0`` (byte-for-byte)."""
    return _build_oneshot_v0(
        _slayer_tools_tail(
            lean_introspection=lean_introspection, readonly_mode=readonly_mode),
        _slayer_define_ref(readonly_mode=readonly_mode),
    )


def build_slayer_otf_ainteract_v0(
    *, lean_introspection: bool = False, readonly_mode: bool = False
) -> str:
    """DEV-1666 gated build of the v0 slayer a-interact template. False/False ==
    ``SLAYER_OTF_AINTERACT_V0`` (byte-for-byte)."""
    return _build_ainteract_v0(
        _slayer_tools_tail(
            lean_introspection=lean_introspection, readonly_mode=readonly_mode),
        _slayer_define_ref(readonly_mode=readonly_mode),
    )
