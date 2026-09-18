## Purpose

Keeps the slayer-mode `claude_sdk` prompts free of guidance the SLayer tool descriptions
already carry, while retaining the benchmark-, grader-, user-sim-, and harness-specific
discipline the tools cannot know, so per-turn context stays lean without losing signal.

## ADDED Requirements

### Requirement: Prompts do not restate tool-covered guidance

The slayer-mode `claude_sdk` prompts (the v0 and v1 OTF one-shot and a-interact builders,
the pre-encoded and readonly variants, the reference encoder, and the shared
`SLAYER_A_INTERACT` / `SLAYER_C_INTERACT`) SHALL NOT contain guidance the pinned SLayer
`query` / `search` / `inspect` / `create_model` / `edit_model` / `recommend_root_model`
tool descriptions, the `SlayerQuery` schema, or the `help.intro` memory already carry. In
particular they MUST NOT mention a `query_nested` or `help` tool, teach colon-aggregation
syntax, or restate the dedup-vs-raw-rows, search-vs-inspect, sample-value-filter, host /
join-key, or generated-SQL-artifact rules verbatim.

#### Scenario: No dead tool names

- **WHEN** any slayer-mode `claude_sdk` prompt is rendered
- **THEN** it contains neither `query_nested` nor a SLayer `help` tool reference

#### Scenario: No restated tool-doc blocks

- **WHEN** any slayer-mode `claude_sdk` prompt is rendered
- **THEN** it does not contain the removed tool-covered blocks (dedup-vs-raw-rows walkthrough, search-vs-inspect discipline, generated-SQL-artifact checklist, colon-aggregation teaching)

### Requirement: Prompts teach the single-query capability

Each slayer-mode `claude_sdk` prompt SHALL state that SLayer expresses multistage
aggregations, computed dimensions, ranking, and cross-model joins in one query, reserving a
multi-stage list for when a whole result set must be re-queried or joined.

#### Scenario: Single-query nudge present

- **WHEN** any slayer-mode `claude_sdk` prompt is rendered
- **THEN** it contains the single-query nudge naming the multi-stage list as the exception

### Requirement: Harness-specific discipline retained

The prompts SHALL retain the guidance the SLayer tools cannot know: the positional
grading rule, the pre-submit mutation check, the query-before-submit gate, the ask-first /
user-sim-trust / grader-diagnostic / pivot / after-rejected blocks (interactive variants),
the automatic filter-normalisation note describing the actual normaliser behaviour, the
`[kb=N]` / `meta.kb_id` traceability rule, and the readonly no-rederive rule.

#### Scenario: Grading and gate discipline kept

- **WHEN** a slayer-mode OTF prompt is rendered
- **THEN** it still tells the model that column names do not affect grading and that a `query` must immediately precede `submit_query`

### Requirement: Raw-mode prompts unchanged

The raw-mode (non-slayer) `claude_sdk` prompts SHALL be byte-identical to before this
change.

#### Scenario: Raw prompts untouched

- **WHEN** the raw one-shot and a-interact OTF prompts are rendered
- **THEN** they are byte-for-byte identical to the pre-change output
