"""System prompt for the on-the-fly KB-encoding Claude SDK agent
(livesqlbench / one-shot flavor).

A single agent (no forced stages, no recursion, no `kb_to_slayer` tool):
it ENCODES the relevant knowledge-base (KB) items into the per-task SLayer
store as named columns/measures — in dependency order, referencing earlier
entities through declared joins — then writes a FINAL query that references
those named entities instead of inlining their SQL.

After DEV-1507 this module carries the **one-shot only** template; the
sibling ``claude_sdk_otf_ainteract`` package owns the a-interact template
plus the hard ``ask_user``-before-submit discipline.

The encoding discipline is distilled from
``pydantic_ai_otf_encode/prompts.py::_STYLE_GUIDE`` (string normalisation,
cross-model access via declared joins, no invented joins, host choice,
``[kb=N]`` self-annotation). All examples are synthetic — never reference
a real dataset's table/column/value names.

Format params: ``budget``, ``db_name``, ``user_query``.
"""

from bird_interact_agents.agents._host_discovery_playbook import (
    HOST_DISCOVERY_PLAYBOOK as _HOST_DISCOVERY_PLAYBOOK,
)
from bird_interact_agents.agents._shared_otf_prompts import (
    _COLUMN_NAMES_DONT_AFFECT_GRADING,
    _DECOMPOSE_DISCIPLINE,
    _DEDUP_VS_RAW_ROWS,
    _NO_USER_TO_CONSULT,
    _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    _SLAYER_SQL_ARTIFACT_CHECK,
)

# Shared submission contract (single-stage or nested-DAG). Literal JSON
# braces are doubled because the prompt is consumed via ``str.format``.
_SUBMIT_CONTRACT = """\
Call `submit_query` with your final SLayer query JSON. `query_json` is one
of two top-level shapes:

  * Single-stage — a JSON object validating as a SlayerQuery, e.g.
    {{"source_model": "orders", "dimensions": ["status"],
    "measures": ["amount:sum"]}}.
  * Nested DAG — when one stage's MEASURE becomes the next stage's
    DIMENSION, a JSON ARRAY of stage objects (the shape `query_nested`
    accepts). The last element is the DAG root; every non-final element
    needs a `name`; later stages reference earlier ones via
    `source_model: "<sibling name>"`. Do NOT wrap the array in
    {{"queries": ...}} — that shape is rejected.

You MUST call `submit_query` to finish — a prose answer is not a
submission. If a `filters` predicate needs a computed value, encode it as
a named column first and filter on the name; raw SQL expressions are
rejected in `filters`."""

_ENCODE_CORE_HEAD = """\
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
"<db>.<model>.<col>"]`, `datasource="<db>"`, `max_results=10`. The
returned `SearchHit(kind="entity").text` carries `Description:` and
`Sample values:` inline (skim past any interleaved memory hits to
the entity hits). The truncated `Sample values:` line is your
authoritative source of which literal forms actually occur in this
column — case variants, whitespace forms, abbreviations, alternate
phrasings of the same concept. Use it BEFORE writing any IN-set
(see rule 3 below).

ENCODE-THEN-QUERY DISCIPLINE:"""

_ENCODE_CORE_TAIL = """\
2. For each block, `search` for the relevant KB memory and any entity
   that already encodes it. A `memory:<id>` token inside a KB body means
   that KB DEPENDS ON the referenced KB.

3. ENCODE IN DEPENDENCY ORDER. Encode a KB that others depend on BEFORE
   the KBs that reference it (topological order). For each KB:
   - Create the column / measure on the HOST whose row is 1:1 with what
     the KB describes. When the KB does not pin the host unambiguously,
     follow the HOST DISCOVERY playbook below to pick it — description
     match first, then shortest declared-join path. Never pick a host
     that needs an undeclared join.
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
     sampled values (check via `inspect_model`), do not write that
     predicate.
   - Symmetric companion: if the column's `Sample values` show variants
     of the KB-named literals — case differences, internal whitespace,
     abbreviations (`apt` for `Apartment`, `Y` for `yes`), or alternate
     phrasings of the same concept (`brick house` vs `brickwork house`,
     `2014+` / `after 2014` for `2014 or newer`) — NORMALISE and EXTEND
     the IN-set to include those variants. KB hedges ("etc.", "like",
     "include") and the schema author's `Ex.` enumerations are
     deliberately non-exhaustive; the `Sample values` line is the
     authoritative inventory of what's actually present in the column.
     A canonical-only IN-set will silently miss matching rows. There is
     no user-sim to confirm this for you in one-shot mode — read the
     sampled values yourself.
"""

_ENCODE_CORE = (
    _ENCODE_CORE_HEAD + "\n\n"
    + _DECOMPOSE_DISCIPLINE + "\n\n"
    # DEV-1546: dim-only auto-dedup vs raw-rows decision — taught BEFORE
    # the encoding rules so the agent decides on `distinct_dimension_values`
    # while writing the final query, not after seeing the artifact.
    + _DEDUP_VS_RAW_ROWS + "\n"
    + _ENCODE_CORE_TAIL
)

SLAYER_OTF_ONE_SHOT = (
    "You are a data analyst. You have a SLayer semantic-layer MCP server plus a\n"
    "native `submit_query` tool. Your job: answer the user's question by\n"
    "ENCODING the domain knowledge it needs into the SLayer model as named\n"
    "columns/measures, then writing a FINAL query that REFERENCES those named\n"
    "entities instead of inlining their SQL.\n\n"
    + _NO_USER_TO_CONSULT.format(sources_desc="the memories and column\ndescriptions")
    + "\n\n"
    + _ENCODE_CORE
    + "\n4. TEST candidate columns and the final query with `query` /\n"
      "   `query_nested`; sanity-check the generated SQL.\n\n"
      "   "
    + _SLAYER_SQL_ARTIFACT_CHECK
    + "\n\n"
    + _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT.format(
        submit_tool="submit_query",
        clause_b="encoded KB",
    )
    + "\n\n6. SUBMIT. Write the FINAL query so it REFERENCES the named columns /\n"
      "   measures you encoded — do NOT inline their SQL back into the query.\n"
      "   Project exactly the columns the question names, and only those.\n"
      "   "
    + _COLUMN_NAMES_DONT_AFFECT_GRADING
    + "\n\n   "
    + _SUBMIT_CONTRACT
    + "\n\nBudget: {budget} bird-coins (`submit_query` costs 3; SLayer reads/writes\n"
      "are free but your total work is turn-bounded — encode only what the\n"
      "question needs).\n\n"
      "Database: {db_name}\n"
      "User question: {user_query}\n"
    + "\n"
    + _HOST_DISCOVERY_PLAYBOOK
)
