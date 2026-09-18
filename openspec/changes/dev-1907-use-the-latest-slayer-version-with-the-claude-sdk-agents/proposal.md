## Why

The repo pins `motley-slayer[advanced-search]>=0.9.6` (lock resolves 0.9.6). SLayer
0.10.2 is out and (a) deletes the `query_nested` MCP tool and reshapes `query` to a
single `query` argument, (b) removes the `help` tool and several private helpers the
harness imports, (c) tightens `ModelExtension` and query-backed-model validation so
the committed reference no longer loads, and (d) moves a large amount of query-language
and discipline guidance into the tool descriptions / `SlayerQuery` schema /
`help.intro` memory. The `claude_sdk` agents break on import against 0.10.2, and their
prompts now duplicate 2k+ tokens per turn of guidance the tools already carry.

## What Changes

- Bump `motley-slayer[advanced-search]` to `>=0.10.2` in both extras; relock.
- **BREAKING (agent tool surface)** — the in-process `query` / `submit_query` tools
  adopt SLayer 0.10.2's shape: one `query` argument (a query object, a multi-stage
  list, or a run-by-name string) plus the execution wrappers; the `query_nested` tool
  is deleted; `normalize_filters` stays a sibling tool argument, never a query field.
- Restore import compatibility: the removed `slayer.core.formula._preprocess_sql_operators`
  / `parse_filter` and `slayer.engine.column_dependency._is_trivial_base` /
  `_resolve_target_for_ref` / `_root_scope_column_ids` are replaced with their 0.10.2
  homes in `slayer.engine.syntax` and `slayer.sql.column_expansion`.
- Migrate the committed `slayer_models/` reference to the current SLayer storage schema
  once, drop three dead top-level `measures` blocks on query-backed models, and fix one
  `ModelExtension` that carried a now-rejected `filters` key (silently ignored before,
  so this restores the model's stated 180-day-window semantics).
- Remove from every slayer-mode `claude_sdk` prompt the guidance the 0.10.2 tool
  descriptions / `SlayerQuery` schema / `help.intro` already carry; add a one-line
  single-query nudge; keep only benchmark-, grader-, user-sim-, and
  harness-mechanics-specific text.
- Reword the two eval remediation strings that name `query_nested`; drop `help` from
  the action-cost table; update the shared filter-normalisation facade to the new
  `query` payload shape and remove its `query_nested` branch.

## Capabilities

### New Capabilities

- `slayer-agent-tool-surface`: the SLayer query/submit tool contract the `claude_sdk`
  agents expose to the model — the unified `query` shape, the absence of `query_nested`
  and `help`, `normalize_filters` as a sibling argument, and the advertised tool set
  tracking the pinned SLayer version.
- `slayer-reference-store`: the committed `slayer_models/<db>` reference round-trips
  through the pinned SLayer storage schema without rewriting tracked files, and every
  committed model validates under the pinned SLayer version.
- `slayer-agent-prompts`: the slayer-mode `claude_sdk` prompts do not restate guidance
  the SLayer tool descriptions carry, do teach the single-query capability, and retain
  the benchmark/grader/user-sim/harness-specific discipline.

### Modified Capabilities

<!-- none: the repo has no pre-existing specs corpus (fresh openspec init). -->

## Impact

- Dependency: `pyproject.toml` (two pins), `uv.lock` (motley-slayer 0.10.2, sqlglot
  30.6.0 → 30.18.0).
- Code: `agents/_query.py`, `agents/claude_sdk/agent.py`, `agents/_submit.py`,
  `agents/_tool_specs.py`, `agents/claude_sdk/partition.py`, `agents/_shared_otf_prompts.py`,
  `agents/_pre_encoded_prompts.py`, `agents/claude_sdk/prompts.py`,
  `agents/claude_sdk_otf*/{agent,prompts,setup_encoder}.py`,
  `slayer_pipeline/filter_normalization.py`, `slayer_otf/store_kb_checker.py`,
  `slayer_otf/store_cleanup.py`, `harness.py` (ACTION_COSTS), `eval/annotation_schema.py`,
  `eval/autopsy.py`.
- Data: `slayer_models/**` (one-off storage-schema migration + three measure drops + one
  ModelExtension fix).
- Tests: ~35 existing test modules re-blessed + several new modules; `tests/data/dev1603/`
  goldens re-captured.
- Deferred adapters (pydantic_ai*, agno, smolagents, mcp_agent): touched only where a
  shared code path or a failing test forces it.
