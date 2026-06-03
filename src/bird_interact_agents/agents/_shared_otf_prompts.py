"""Shared prompt string constants for the claude_sdk OTF agent family.

These constants are used verbatim (format params substituted at compose
time) by both the SLayer OTF agents and the raw OTF agents to keep prompts
aligned wherever SLayer is not involved.

Constraint: after the SLayer prompt files are refactored to import from
here, the rendered values of SLAYER_OTF_ONE_SHOT and SLAYER_OTF_AINTERACT
must remain byte-for-byte identical. SHA-256 snapshot tests in
tests/test_shared_otf_prompts.py enforce this.

Format param conventions:
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
