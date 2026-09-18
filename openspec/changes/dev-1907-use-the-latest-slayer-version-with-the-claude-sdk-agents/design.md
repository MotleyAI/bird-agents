## Context

See proposal.md (Why). The suite was run against 0.10.2 with two temporary import shims:
22 failures + 1 collection error, all with known causes. SLayer 0.10.2 source is read-only
at `/home/james/GitHub/SLayer/slayer`. The `claude_sdk` agents reach SLayer two ways: the
v0 agents via the `slayer mcp` stdio server, the v1 agents and the encoder via in-process
natives that bridge SLayer's own tool descriptions/schemas from a storage-less
`create_mcp_server`. The base `SLAYER_A/C_INTERACT` prompts are consumed only by the
deprecated adapters but live in the `claude_sdk` package.

## Goals / Non-Goals

**Goals:**
- One SLayer query shape everywhere (SLayer's own), taught by the tool docs, not by us.
- Prompts carry only what the tools cannot: grader, user-sim, gates, `[kb=N]` tags.
- The committed reference is valid and stable at the pinned storage schema.

**Non-Goals:**
- Deep rewrite of the deprecated adapters' own prompt bodies or the HOST DISCOVERY
  playbook (touched only where a shared path or a failing test forces it).
- Retiring `--no-lean` (its FULL surface advertises the now-deprecated `inspect_model`).

## Decisions

- **D-tool-shape**: our `query`/`submit_query` natives advertise SLayer's own `query`
  docstring + JSON schema (via `_slayer_tool_metadata`) plus a `normalize_filters` arg;
  `query_nested` is deleted; `_query.py` drops its hand-maintained field allow-list. Rationale:
  the brief's premise is that the tool docs teach the shape — two shapes would defeat it and
  drift on the next release. Alternative (keep structured args, adapt internals) rejected: it
  keeps a parallel schema and teaches the model a different shape than SLayer's docs.
- **D-key-lift (Codex 5)**: tool-level keys are lifted out of a query only in the
  single-object form (the existing DEV-1577 scope); for a multi-stage list each stage is
  stripped of misplaced tool-level keys and none is lifted — no undefined last-stage-wins
  precedence.
- **D-kb-checker (Codex 2)**: the filter-ref helper walks `AggCall.source` recursively (and
  kwarg refs), so an aggregate over an expression source (`sum(amount - cost)`) still
  contributes its column dependencies; a test pins this.
- **D-prompt-A/C-interact (Codex 1)**: `SLAYER_A_INTERACT` / `SLAYER_C_INTERACT` are cleaned
  by the same rules as the other slayer prompts (they live in the `claude_sdk` package and
  still name the removed `help` tool). The deprecated adapters' own prompt bodies stay.
- **D-normalize-note (Codex 4)**: `_NORMALIZE_FILTERS_NOTE` describes the actual normaliser
  subset (a plain-column LHS compared `==`/`!=`/`in`/`not in` against a string literal or an
  all-string list), not "every filter".
- **D-define-ref (Codex 3)**: the standalone `_DEFINE_BEFORE_REFERENCE` block is dropped; only
  its "create it with create_model/edit_model first" clause survives inside the encode step.
  The single-query nudge is kept (v1/encoder never see SLayer's server-level instructions).
- **D-schema-validation (Codex 8)**: a test validates both native tool schemas as JSON Schema
  (reused `$defs` resolvable) and validates representative payloads — computed dimension,
  inline ModelExtension, multi-stage list, `month(col)` functional granularity — against the
  `query` property.
- **D-reference-migration**: a one-off, no-new-script migration round-trips each
  `slayer_models/<db>` through 0.10.2 `YAMLStorage` against an empty scratch sqlite (so
  on-load type refinement is a no-op), copies `models/` back, and commits the schema-10 YAML.
  A guard test asserts the round-trip is now a no-op. Tests that open the committed dir in
  place get a shared conftest fixture that copies to tmp first.
- **D-version-pin (Codex 7)**: keep `>=0.10.2` + `uv.lock`, matching the repo's documented
  floor-plus-lock invariant; the surface test is the alarm that fires when the lock moves.

## Risks / Trade-offs

- Large mechanical YAML diff from the storage migration → reviewed to be versioning +
  `unique: false` only; commands recorded in the PR.
- Broad prompt diff re-blesses SHA-pinned prompt tests and the `tests/data/dev1603` goldens →
  the new tests assert kept/removed markers per prompt family, not one universal set
  (Codex 6), so an accidental over-trim still fails.
- v0 stdio startup / help.intro seeding on 0.10.2 is not provable by unit tests → covered by
  the live smoke.

## Migration Plan

Merge `origin/main` (2 non-overlapping commits) → bump pins + relock → import-compat fixes →
reference migration + fixture → prompt cleanup → tests (spec-tests stage first) → full
non-integration suite green → docs → live smoke → PR. Rollback: revert the branch; the lock
pin is the only externally visible change.

## Open Questions

None blocking. Follow-ups (out of scope): retire `--no-lean`; clean the deprecated adapters'
prompt bodies and the HOST DISCOVERY playbook's non-existent `reachable_fields` argument.
