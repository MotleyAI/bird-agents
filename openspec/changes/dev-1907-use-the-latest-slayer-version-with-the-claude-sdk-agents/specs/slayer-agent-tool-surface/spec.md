## Purpose

Defines the SLayer query and submission tool contract the `claude_sdk` agents expose to
the model, so it tracks the pinned SLayer version's shape and no dead or duplicate tool
surface leaks into the model's per-turn context.

## ADDED Requirements

### Requirement: Unified query argument

The in-process `query` tool the `claude_sdk` agents register SHALL accept the SLayer
query as a single `query` argument whose value is one query object, a list of query-stage
objects, or a run-by-name model-name string, plus the execution wrappers `variables`,
`show_sql`, `dry_run`, `explain`, `format`. It MUST NOT expose separate top-level
`source_model` / `measures` / `dimensions` / `filters` / … arguments, and MUST forward
the value to the pinned SLayer `query` tool function unchanged in shape.

#### Scenario: Single query object

- **WHEN** the model calls `query` with `query` set to a query object such as `{"source_model": "orders", "dimensions": ["status"], "measures": [{"formula": "count(*)"}]}`
- **THEN** the tool executes it through SLayer and returns the formatted result

#### Scenario: Multi-stage list

- **WHEN** the model calls `query` with `query` set to a list of stage objects whose last element is the DAG root
- **THEN** the tool executes the multi-stage query and returns the root stage's rows without needing any separate nested-query tool

#### Scenario: Computed-dimension query in one call

- **WHEN** the model calls `query` with a single query object whose `dimensions` contains a computed expression that re-aggregates at a coarser grain (a `partition_by=` aggregate inside the expression)
- **THEN** the tool executes that single query rather than requiring a multi-stage list

### Requirement: No query_nested or help tool

The `claude_sdk` agents MUST NOT register or allow-list a `query_nested` tool or a SLayer
`help` tool, and MUST NOT name either in any tool description, allow-list, disallow-list
derivation, or action-cost table.

#### Scenario: query_nested absent from the advertised surface

- **WHEN** the agent's SLayer tool allow-list is derived against the pinned SLayer MCP surface
- **THEN** the derivation succeeds and neither `mcp__slayer__query_nested` nor `mcp__slayer__help` appears in the allow-list or its complement without error

### Requirement: Submission accepts the unified query shape

The in-process `submit_query` tool SHALL accept the final query as the same shape the
`query` tool's `query` argument takes — one query object or a multi-stage list — compile
it deterministically to SQL, and record the agent's original query payload on the result.
A payload that is neither an object nor a list MUST be rejected with a message naming the
two valid shapes.

#### Scenario: Submit a single-stage query

- **WHEN** the model calls `submit_query` with a query object
- **THEN** the tool compiles it to SQL, grades it, and stores the original payload as the submitted query

#### Scenario: Submit a multi-stage list

- **WHEN** the model calls `submit_query` with a list of stage objects
- **THEN** the tool compiles the multi-stage query to SQL and grades it

### Requirement: Filter normalisation is a sibling argument

`normalize_filters` SHALL remain a sibling tool argument (default true) on both `query`
and `submit_query`, never a field inside the query object. When a tool-level option
(`show_sql`, `dry_run`, `explain`, `format`, `normalize_filters`) is misplaced inside a
single query object it is lifted out into the corresponding argument; when the query is a
multi-stage list, such a misplaced key is stripped from each stage and never lifted.

#### Scenario: normalize_filters opt-out

- **WHEN** the model calls `submit_query` with `normalize_filters=false`
- **THEN** the recorded submitted query is byte-identical to the agent's payload and the compiled SQL keeps exact-case equality on text filters

#### Scenario: Misplaced tool-level key in a list stage

- **WHEN** the model calls `query` with a multi-stage list and one stage object carries a stray `show_sql` key
- **THEN** the tool strips that key from the stage and does not treat differing per-stage values as a top-level option
