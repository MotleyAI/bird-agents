"""Single source of truth for the HOST DISCOVERY playbook (DEV-1512).

Imported by every OTF-encoder / query-constructor prompt that needs to
teach the agent how to pick a root model or join host when a KB
underspecifies the join graph. Used by:

* ``claude_sdk_otf.prompts.SLAYER_OTF_ONE_SHOT``
* ``pydantic_ai_otf_encode.prompts._STYLE_GUIDE`` (which is appended to
  ``KB_ENCODER_PROMPT`` / ``KB_ENCODER_ONESHOT_PROMPT`` /
  ``SETUP_ENCODER_PROMPT``)
* ``pydantic_ai_otf_encode.prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT``
* ``pydantic_ai_recursive.prompts.QUERY_CONSTRUCTOR_PROMPT``
* ``pydantic_ai_recursive.prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT``

The text has no ``str.format`` placeholders so it can be concatenated
into any prompt body without breaking the call site's ``.format(...)``.

DEV-1591: the search-vs-inspect discipline (``_COMPACT_SEARCH_DISCIPLINE``)
is imported from ``_shared_otf_prompts`` and embedded inline so the rule
lives in exactly one place. `search` is discovery-only (hardwired compact);
every targeted detail read in this playbook uses the ``inspect`` point-lookup
(``entity_type=…``, ``compact=False``), which has no RRF fusion and so needs
no ``cypher_filter``. The constant is param-free and brace-free, so the
no-``.format()``-placeholder contract still holds.
"""

from bird_interact_agents.agents._shared_otf_prompts import (
    _COMPACT_SEARCH_DISCIPLINE,
)

HOST_DISCOVERY_PLAYBOOK = (
    """\
HOST DISCOVERY (when picking a root model [`source_model`] for a query or
a host for an encoded entity, AND the KB body does not pin the host
unambiguously — i.e. the KB names columns/fields that exist on more than
one candidate table, or the formula's columns live on a table T that's
not the natural per-entity grain of the question).

HARD REQUIREMENT — READ COLUMN DESCRIPTIONS BEFORE COMMITTING. For every
candidate (root model OR join-key column) you are about to use, read its
description first. Column descriptions are the PRIMARY signal — they
often state the schema author's semantic intent verbatim ("Associates
the inspection with the relevant sensor reading", "Links this audit to
the underlying measurement", "References the parent batch's sample
set"). That intent is the canonical answer to "which table is this
column meant to be reached from". Skip this only when the host is
trivially pinned (single candidate table, KB explicitly names it, OR no
joins involved at all).

NEVER read raw `*_column_meaning_base.json` / `*_schema.txt` /
`*_kb.jsonl` files. They are already projected into the SLayer surface:
column meanings live in `Column.description`, schema FKs are ModelJoin
entries on the source-side table only (the FK side, not both sides),
and KB items are memories you `search` for.

HOW TO READ COLUMN DESCRIPTIONS via SLayer MCP:

  * Known column(s), want their descriptions by canonical ref:

        inspect(
            reference=["<db>.<model>.<col>", ...],
            entity_type="column",
            compact=False,
        )

    `inspect` is a clean point-lookup — no RRF fusion, no bundled
    memories — so it returns exactly the columns you name and needs no
    `cypher_filter` or `max_results` budgeting. Pass a list to read
    several columns of the same kind in one call (per-id resolution
    errors are isolated and don't sink the batch). Each column comes back
    as a multi-line block — `Column: <ds>.<model>.<col> / Type: <type> /
    Description: <intent text> / Sample values: ...`. (The default
    `compact=True` returns description-only; pass `compact=False` for the
    full body with Sample values.)
  * Whole-model bulk read (every column at once): `inspect_model(<model>,
    sections=["columns"], data_source="<db>")` — Column.name + .type +
    .description for every column. Use when scanning a model end-to-end.
  * Discover columns whose descriptions match a phrase:
    `search(question="<one-sentence paraphrase>", max_results=10,
    datasource="<db>", cypher_filter='MATCH (n:ModelColumn:Measure:Aggregation:Model) RETURN n.id AS id')`.
    The cypher filter pins the result list to entity hits (multi-label
    is union semantics); the tantivy + dense-embedding channels then
    rank all column / model / measure descriptions and return a unified
    `results` list of entity hits. This is a BROAD search — it returns
    one-line descriptions only; you never pass `compact` (see
    SEARCH-vs-INSPECT DISCIPLINE below).

"""
    + _COMPACT_SEARCH_DISCIPLINE
    + """

HOW TO FIND CANDIDATE HOSTS for a target table T (the table whose
columns the KB references):

  1. Call `models_summary(datasource_name="<db>", format="json")` once.
     Every model's `joins_to` list is its OUTGOING declared joins.
     Models whose `joins_to` contains T are the candidate 1-hop hosts.
     (SLayer ingestion only emits FKs on the source side, so calling
     `inspect_model(T, sections=["joins"])` would only list joins FROM
     T and miss inbound candidates — the inverse direction.)

  2. For each candidate host H, call `inspect_model(H,
     sections=["columns", "joins"], data_source="<db>")` to see (a)
     the join_pairs naming the FK columns that connect H to T, and (b)
     the descriptions of those FK columns on H's side.

  3. If no candidate has a 1-hop join to T, widen the search: for each
     candidate H, call `inspect_model(H, sections=["reachable_fields"],
     reachable_fields_depth=3, data_source="<db>")` and grep the
     returned dotted paths for those ending at one of T's columns —
     path depth equals hop count.

DECISION ORDER:

  PRIMARY (description match). Pick the candidate whose FK column's
  description literally states the relationship the KB needs
  ("associates X with Y" / "links X to Y" / "references X's Y data").
  This is the canonical schema-author signal.

  TIE-BREAKER 1 (shortest declared path). If multiple candidates match
  on description (or none does), prefer the candidate with the SHORTEST
  declared-join path to T. A 1-hop FK is almost always more canonical
  than a 3-hop chain through shared-infrastructure tables (e.g. tables
  named "events", "log", "sites", "registry", or any table with
  many-to-many roles in the schema) — those chains are one-to-many at
  every step and silently multiply rows even when structurally valid.

  TIE-BREAKER 2 (KB body re-read). If still tied, re-read the KB body
  via `inspect(reference=["memory:<id>"], entity_type="memory",
  compact=False)` — KB definitions occasionally include parenthetical
  hints ("per entity", "per inspection") that pin the grain. `inspect`
  with `compact=False` returns the full `learning` body, whereas a
  `search` would only return the one-line `description`.

  LAST RESORT (deterministic). If still ambiguous, pick the candidate
  whose model name sorts FIRST lexicographically and record the choice
  in your notes / final answer so a reviewer can flag it. Never
  silently guess.

WORKED EXAMPLE (fictional names — invent your own).
KB defines `quality_risk = sp * sw * ml / 100` on columns of
`sensor_readings`. Call `models_summary(datasource_name="demo",
format="json")` and scan each model's `joins_to`:
`asset_inspections.joins_to` includes `sensor_readings` (1 hop);
`service_log.joins_to` includes `equipment` only;
`equipment.joins_to` includes `maintenance_log`;
`maintenance_log.joins_to` includes `sensor_readings`. So
`asset_inspections` is a 1-hop candidate and `service_log` is a 3-hop
candidate.

Description read (PRIMARY):
  inspect(
    reference=["demo.asset_inspections.sensor_read_ref",
               "demo.maintenance_log.sensor_read_ref"],
    entity_type="column", compact=False,
  )
returns the column bodies:
  * `asset_inspections.sensor_read_ref` — "Associates the inspection
    with the relevant sensor reading."
  * `maintenance_log.sensor_read_ref` — "Associates maintenance data
    with an existing sensor measurement."
Only `asset_inspections.sensor_read_ref` directly states an
inspection-to-sensor relationship; `maintenance_log` talks about
maintenance data, not inspection. PICK `asset_inspections` on the
description signal alone — tie-breakers are unnecessary. If both
descriptions had been equally on-intent, the 1-hop tiebreaker would
have picked `asset_inspections` anyway.
"""
)
