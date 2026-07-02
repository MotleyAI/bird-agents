"""DEV-1555 Stage 1 / DEV-1581 R2: discovery/main partition helpers.

The four ``claude_sdk_otf*`` agents split each task between a long-lived
*discovery* assistant (introspection + user clarification, its own context
window) and the main loop (encode / query / submit on a slim context). The
split keeps peak context inside open-weight model windows (~260K): the
dominant context consumers are introspection tool outputs, which now stay in
the discovery context and reach the main loop only through ``ask_discovery``
answers.

R2 enforcement model (verified against the SDK semantics):

* The two halves are SEPARATE persistent ``ClaudeSDKClient`` sessions, so the
  main client's per-turn tool schema NEVER contains the introspection tools —
  the partition is a HARD boundary (no shared schema, no ``disallowed_tools``
  global, no per-call deny hook). Main reaches discovery ONLY through the
  in-process ``ask_discovery`` tool, which forwards to the warm discovery
  client and returns its text.
* The discovery client is warm: its context accumulates across ``ask_discovery``
  calls, so follow-ups are cheap (no cold re-introspection — the root cause of
  the DEV-1581 v1 turn blow-up). Its own ``max_turns`` is ``DISCOVERY_MAX_TURNS``
  per ask; the per-task number of asks is capped by the ``DiscoveryChannel``.
"""

from __future__ import annotations

from bird_interact_agents.harness import MAX_MODEL_TURNS

# Hard cap for ONE discovery answer (one ``ask_discovery`` round). The main
# loop keeps the existing 2x cap (`_MAX_TURNS`); a single focused introspection
# sweep needs far less.
DISCOVERY_MAX_TURNS = MAX_MODEL_TURNS


_DISCOVERY_PROMPT_COMMON = """\
You are the discovery assistant for a data-analysis task. Your ONLY job is
to gather everything the main agent needs to answer the user's question and
report it back. You cannot submit answers and you must not try to solve the
task yourself. You are long-lived: the main agent will ask you a series of
focused questions and your context carries over between them, so build on
what you already found instead of re-introspecting from scratch.

For the FIRST, broad request, produce a structured report with EXACTLY the
sections below. For a NARROW follow-up, do NOT re-emit the whole report —
answer just what was asked (you may cite the relevant section by name).

Report sections (broad request only):

1. RELEVANT ENTITIES — every table/model plausibly needed, with a one-line
   purpose each.
2. COLUMNS — for each relevant column: name, description, observed sample
   values (quote them verbatim, preserving the EXACT stored spelling — never
   canonicalise). For text / categorical columns, explicitly flag any case,
   whitespace, or spelling variation across the sampled values, and any
   mismatch between documented enums and the actual values, so the main agent
   can build filter predicates that match what is really stored.
3. JOIN PATHS — how the relevant entities connect (direction and keys).
4. KNOWLEDGE-BASE ITEMS — every KB item that bears on the question, with
   its definition quoted VERBATIM (formulas must not be paraphrased).
5. USER CLARIFICATIONS — every question you asked and the answer, quoted
   verbatim as Q→A pairs.
6. OPEN AMBIGUITIES — anything you could not resolve. This section MUST
   be empty in the common case; only put something here if the user-sim
   explicitly declined to answer a clarifying question. Do NOT defer
   resolvable ambiguities to the main agent — see the resolution
   checklist below.

Be exhaustive in coverage but terse in prose: the report replaces raw tool
output in the main agent's context, so include facts, not narration.
"""

_DISCOVERY_PROMPT_ASK_USER = """
The user simulator holds masked ground truth that is unrecoverable from
the KB alone. You MUST resolve every load-bearing ambiguity with
ask_user BEFORE writing the report — anything left in "Open ambiguities"
is a failure of this subagent, not the main agent's problem to clean up.

Resolution checklist — ask explicitly about each of these when the
request leaves them open (one ask_user can bundle several as a numbered
list; the user-sim answers point-by-point):

* GROUPING / SCOPING — which column defines the bucket? (e.g.
  "atmospheric condition" could map to several columns.)
* METRIC IDENTITY — when the request names a quantity ("signal
  quality", "score", "X factor"), which exact column or formula is it?
  If a KB item names a formula, quote it verbatim and confirm "is THIS
  the formula?".
* AGGREGATION SCOPE — for each aggregate (avg/median/sum/count): is it
  computed over ALL rows in the group, or only over a filtered subset
  (e.g. usable / non-null / passing a threshold)?
* SORT ORDER — which column orders the output rows? Direction
  (ascending / descending)?
* TIE-BREAKING — secondary sort, deterministic order for equal keys.
* NULL HANDLING — drop, treat as zero, treat as a bucket, propagate.
* UNITS / ROUNDING — explicit units when the column allows ambiguity
  (seconds vs minutes; °C vs °F); rounding rules for displayed numbers.
* FILTER LITERALS — when the request references a labeled value
  ("High Income", "Severe"), quote the user's label back and confirm
  the exact stored literal it maps to.

Record each Q and verbatim A in section 5. If the user-sim refuses or
returns no answer, record the refusal verbatim and put the unresolved
choice in section 6 with the two or three candidate interpretations.
"""


def build_discovery_prompt(*, with_ask_user: bool) -> str:
    """Compose the discovery subagent system prompt."""
    if with_ask_user:
        return _DISCOVERY_PROMPT_COMMON + _DISCOVERY_PROMPT_ASK_USER
    return _DISCOVERY_PROMPT_COMMON


_VERIFY_TOOL_BY_MODE = {
    "slayer": ("query", "submit_query"),
    "raw": ("execute_sql", "submit_sql"),
}

# The raw-mode introspection tools that live only on the discovery client, named
# in the bridging clause so any task-guidance that says "call <tool>" is rerouted
# through ``ask_discovery`` for the two-stage v1 main loop. DEV-1629: the slayer
# main loop now HOLDS ``search`` / ``inspect_model`` directly, so slayer no longer
# needs this reroute clause (its ``discovery_section`` is written directly); only
# raw still routes introspection through ``ask_discovery``.
_INTROSPECTION_TOOLS_BY_MODE = {
    "raw": "`get_schema` / `get_all_column_meanings`",
}


def build_main_workflow_note(*, query_mode: str) -> str:
    """Compose the main-agent workflow note.

    Codex r5: the verify-before-submit step must reference the tool
    the agent's tool surface actually exposes — ``query`` for slayer
    mode, ``execute_sql`` for raw mode. The shared discovery-subagent
    block stays identical across modes.
    """
    try:
        verify_tool, submit_tool = _VERIFY_TOOL_BY_MODE[query_mode]
    except KeyError as exc:
        raise ValueError(
            f"build_main_workflow_note: unknown query_mode {query_mode!r}; "
            f"expected one of {sorted(_VERIFY_TOOL_BY_MODE)}"
        ) from exc
    if query_mode == "slayer":
        # DEV-1629: the slayer main loop now HOLDS `search` / `inspect_model`
        # directly; the warm discovery assistant is for grilling the user and
        # broad whole-schema questions, not the primary introspection path.
        discovery_section = """
## Discovery workflow

You HOLD the schema-introspection tools `search`, `inspect_model`, and
`inspect` (single-entity point lookup — use it for a known column's sample
values) on your own surface — call them DIRECTLY for model/column/join
schemas, sample values, and descriptions. A long-lived warm 'discovery'
assistant is ALSO available through the `ask_discovery` tool for delegating
broad whole-schema questions (it owns `models_summary` and accumulates context
across questions). Knowledge-base item definitions are on your surface too —
read them verbatim with `get_knowledge_definition` /
`get_all_external_knowledge_names`."""
        post_fail = f"""

After a FAILED `{submit_tool}` (any non-pass status), do NOT go back to
re-introspect the schema — the new evidence is in the grader's miss
diagnostics and in your candidate query's output, not in the schema. Re-run
the candidate through `{verify_tool}`, pivot your operationalisation, or ask
the user. Re-introspection is for schema/KB gaps you discover while building,
not for grader misses."""
    else:
        introspection_tools = _INTROSPECTION_TOOLS_BY_MODE[query_mode]
        discovery_section = f"""
## Discovery workflow (mandatory)

Schema/data introspection tools (model/table schemas, sample values, join
paths, knowledge-base definitions) are NOT on your tool surface — you
physically cannot call them. A long-lived 'discovery' assistant holds them.
Reach it ONLY through the `ask_discovery` tool: ask a focused question and it
returns its findings. Start by asking discovery for everything you need to
answer the user's question, then work from its answer.

This OVERRIDES any instruction elsewhere in this prompt: wherever the task
guidance tells you to call an introspection tool directly
({introspection_tools}), you do NOT have it — ask `ask_discovery` for that
exact information instead. (Knowledge-base item definitions ARE on your
surface: read those verbatim with `get_knowledge_definition` /
`get_all_external_knowledge_names`, not through discovery.)

Discovery is WARM — it remembers your earlier questions, so follow-ups are
cheap. But before asking, check whether discovery ALREADY told you the
answer in a previous reply; do not re-ask for facts you already have."""
        post_fail = f"""

After a FAILED `{submit_tool}` (any non-pass status), do NOT go back to
`ask_discovery` to re-introspect the schema — the new evidence is in the
grader's miss diagnostics and in your candidate query's output, not in the
schema. Re-run the candidate through `{verify_tool}`, pivot your
operationalisation, or ask the user. `ask_discovery` is for schema/KB gaps
you discover while reading discovery's answers, not for grader misses."""
    return f"""
{discovery_section}{post_fail}

## Verify-before-submit checklist (mandatory)

Before EVERY {submit_tool}, run the exact query you intend to submit
through the `{verify_tool}` tool and verify, against the user-sim if
uncertain:

* SORT ORDER — column AND direction match what the user wants. When the
  request says "rank/top/bottom/most/least", confirm the direction; when
  it's silent, ask once. Never assume.
* AGGREGATION SCOPE — each aggregate is computed over the right subset
  (all rows in the group vs filtered subset). A discrepancy here is the
  most common silent miss.
* NULL HANDLING — every aggregate's behavior on NULLs matches the
  request.
* COLUMN COUNT and POSITIONAL ORDER — match the request's stated
  output shape (column NAMES are not graded; counts and positions are).
* ROW COUNT — non-zero and plausible given the data you saw.
"""


# Back-compat for callers that still import the legacy constant —
# defaults to the slayer-mode wording (mirrors prior content).
MAIN_WORKFLOW_NOTE = build_main_workflow_note(query_mode="slayer")
