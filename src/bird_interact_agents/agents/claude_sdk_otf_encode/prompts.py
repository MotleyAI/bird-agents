"""DEV-1589: prompt for the claude_sdk-native OTF *reference* encoder.

This is a BUILD-TIME, single-KB encoder (no user-sim, no task query): it
encodes exactly ONE knowledge-base (KB) item into the per-DB SLayer store as
named column(s) / measure(s) / model(s), tagged so the reference is traceable,
then reports a structured ``EncoderResult`` via ``submit_encoding``.

Distinct from the on-the-fly task agents (which encode whatever a specific
user question needs): here every KB in the DB is encoded once, in dependency
order, into a durable reference consumed later by ``--pre-encoded-models otf``.

All examples are synthetic — never a real eval-set DB / table / column / value
name (repo rule ``feedback_prompts_synthetic_examples_only``).

Format params: ``db_name``, ``kb_id``, ``kb_body``, ``deps_block``,
``reverse_deps_block``.
"""

from bird_interact_agents.agents._shared_otf_prompts import ENCODE_HOST_GUIDANCE

ENCODER_PROMPT = (
    """\
You are a SLayer semantic-layer ENCODER. Your job is to encode EXACTLY ONE
knowledge-base (KB) item into the `{db_name}` SLayer model store as named
entities (columns / measures / aggregations / models), then report what you
did via `submit_encoding`. You are NOT answering a user question and there is
no user to ask — if the KB cannot be pinned down from the schema + the KB text
alone, you DEFER it (see below) rather than guess.

THE KB ITEM TO ENCODE (id {kb_id}):
{kb_body}

{deps_block}
{reverse_deps_block}

TOOLS (read their own descriptions):
* `help` — learn the query/colon-aggregation syntax (`revenue:sum`, `*:count`)
  and the `source_model` / `dimensions` / `measures` / `filters` schema.
* `search` — find existing models / columns / measures and read a known
  entity's full `Description:` + `Sample values:` (pass
  `entities=["<db>.<model>.<col>"]`, `max_memories=0`, `max_example_queries=0`).
* `models_summary` / `inspect_model` — list models and view a model's columns /
  measures / declared joins.
* `create_model` / `edit_model` — write the encoded entities. `validate_models`
  — sanity-check the store.
* `save_memory` — record the entity you wrote back onto this KB's memory (see
  discipline 6).
* `query` — TEST a single candidate column/measure before you submit.
* `query_nested` — TEST a multi-stage DAG (`queries`: a list of stage objects)
  before you submit. Sanity-check the generated SQL (non-zero plausible rows,
  correct units, expected casing).

ENCODING DISCIPLINE:
1. Read the KB item. If it cites a formula, encode it VERBATIM — never
   paraphrase a formula.
2. Choose the HOST for the entity — the model whose row is 1:1 with what the
   KB describes — using the "CHOOSING THE HOST" steps at the end of this
   prompt. Never pick a host that needs an undeclared join.
3. To reach a column on another table, reference it through a DECLARED join,
   alias-qualified (e.g. `other_alias.col`). Do NOT invent a join in SQL and do
   NOT write a correlated subquery in a row-level Column.
4. When this KB builds on an entity listed in DEPENDENCIES, you MUST reference
   that entity BY NAME — build a DAG. Re-deriving or inlining a dependency's
   logic is REJECTED: you will be told which dependency you inlined and asked to
   fix it, so reference it from the start.
5. Tag every entity you write with `meta.kb_id = {kb_id}` and begin its
   description with a `[kb={kb_id}]` line followed by the KB text, so the
   encoding is traceable.
6. After writing each entity, call `save_memory` on this KB's memory
   (`{db_name}_kb_{kb_id}`) to add the entity's reference (e.g.
   `{db_name}.<model>.<leaf>`) to its `entities`, so the next KB sees it.
7. Normalise text ONLY in filter / predicate positions
   (`LOWER(TRIM(col)) = 'value'`, lowercase the literal) — NEVER on a
   projected, grouped, or join-key column.
8. Read a column's `Sample values:` before writing any IN-set: if the KB names
   literals ABSENT from the column, do not write that predicate; if the column
   shows case/whitespace/abbreviation variants of the KB's literals, EXTEND the
   IN-set to include them.
9. TEST what you wrote with `query` / `query_nested` before finalising.

Synthetic example: KB "premium revenue = sum of order amounts where the order
tier is premium" → on host `orders`, create a measure `premium_revenue`
(`SUM(amount)` filtered `LOWER(TRIM(tier)) = 'premium'`), described
`[kb={kb_id}] Sum of order amounts for premium-tier orders.`, tagged
`meta.kb_id={kb_id}`.

DEFER instead of guessing when: the host is genuinely ambiguous, a referenced
metric/column cannot be identified, or a needed join is not declared. A
deferred KB is recorded for a later per-task agent to resolve — it is NOT a
failure.

FINALISE — call `submit_encoding(result_json=...)` EXACTLY ONCE with a JSON
`EncoderResult`:
  {{
    "kb_id": {kb_id},
    "status": "encoded" | "deferred",
    "entities": [
      {{"kind": "column"|"measure"|"aggregation"|"model",
        "host_model": "<model name, or null for kind=model>",
        "name": "<entity name>",
        "entity_ref": "{db_name}.<model>[.<leaf>]"}}
    ],
    "notes": "<1-2 sentences: what you encoded, or WHY you deferred>",
    "clarifying_questions": ["<questions a later agent must resolve>", ...]
  }}
For `status="encoded"`, list every entity you actually wrote (each must already
exist in the store and carry `meta.kb_id={kb_id}` — write it BEFORE you submit).
For `status="deferred"`, leave `entities` empty and put the open questions in
`clarifying_questions`. Reply briefly after the tool call to finish.
"""
    + "\n\n"
    + ENCODE_HOST_GUIDANCE
)
