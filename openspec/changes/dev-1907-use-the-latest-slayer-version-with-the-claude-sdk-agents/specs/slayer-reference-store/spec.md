## Purpose

Ensures the committed hand-curated `slayer_models/` reference loads cleanly under the
pinned SLayer version and stays at that storage schema, so tests, prompt builders, and the
pre-encoded "custom" source never rewrite tracked data on load.

## ADDED Requirements

### Requirement: Committed reference validates under the pinned SLayer version

Every committed `slayer_models/<db>` reference SHALL load through the pinned SLayer
`YAMLStorage` without a validation error. Query-backed models MUST NOT declare a top-level
`measures` block, and inline `ModelExtension` sources MUST NOT carry keys the pinned SLayer
version rejects.

#### Scenario: Every committed DB loads

- **WHEN** each `slayer_models/<db>` is loaded through the pinned SLayer `YAMLStorage` (against an empty scratch database)
- **THEN** every model in every DB validates, with no `measures`-on-query-backed-model or `extra_forbidden` ModelExtension error

#### Scenario: A previously silent extension filter is preserved as behaviour

- **WHEN** the `fake` DB's trusted-account model is loaded
- **THEN** its 180-day recency window is expressed as an entity the pinned SLayer version accepts (a column plus a stage filter), not as a rejected `ModelExtension.filters` key

### Requirement: Loading the reference does not rewrite tracked files

Loading every committed `slayer_models/<db>` model through the pinned SLayer storage SHALL
leave each committed `models/**.yaml` byte-identical, so no test run, prompt build, or
pre-encoded run dirties the worktree.

#### Scenario: Round-trip is a no-op

- **WHEN** each committed DB is copied to a scratch dir and every model is loaded through the pinned SLayer `YAMLStorage`
- **THEN** the committed YAML on disk is unchanged (a guard test fails loudly if a future SLayer bump would rewrite it)
