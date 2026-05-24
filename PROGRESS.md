# BIRD-Interact main task — progress + remaining plan

Status snapshot as of 2026-05-05. The work below covers everything from
the initial KB audit through landing the slayer-mode harness wiring,
plus what's still pending. Treat this as a living document — update it
when sections complete.

## Linked artifacts

- **Linear**: parent issue [DEV-1316](https://linear.app/motley-ai/issue/DEV-1316);
  this branch's issue [DEV-1318](https://linear.app/motley-ai/issue/DEV-1318);
  benchmark-discovered SLayer bugs under
  [DEV-1329](https://linear.app/motley-ai/issue/DEV-1329) (12 children
  filed: DEV-1330–1341).
- **PR**: [MotleyAI/bird-interact-agents#10](https://github.com/MotleyAI/bird-interact-agents/pull/10).
- **Branch**: `egor/dev-1318-bird-interact-translate-kb-to-slayer-skill-pilot-on-one-db`.
- **Plan file** (private): `~/.claude/plans/look-at-the-bird-interact-agents-serialized-treasure.md`.

---

## What's done

### W-prep / W0 / W-cleanup

- **DEV-1319 doc fixes** shipped in SLayer 0.4.0.
- **DEV-1317 math/stat UDFs** (`ln`, `log10`, `exp`, `sqrt`, `pow`,
  `power`, `stddev_*`, `var_*`, `corr`, `covar_*`) shipped in SLayer
  0.4.1 (PR #82).
- **W0**: `models_summary` docstring fix shipped in PR #81.
- **`meta` on ModelMeasure + Aggregation** shipped in SLayer 0.4.2 (PR
  #83). Lets the verifier match encoded entities to KB ids uniformly.
- **W-cleanup**: deleted the broken 0.3.1-era `credit` datasource that
  had a v1 dim/measure name collision; re-piloted in W4b.

### W1 — split skill + verifier

- New skill `bird-interact-agents/.claude/skills/kb-to-slayer-models/`
  — domain-agnostic recipe book covering R-DESCRIBE, R-JOIN, R-COL,
  R-CASE, R-FILTER, R-MEASURE, R-AGG, R-RESOLVE, R-MULTISTAGE,
  R-WINDOW, R-EXISTS, R-VAR, R-HOST, R-PROSE. v0.4.x idioms throughout
  (`columns=[…]`, `measures=[{name, formula}]`, `aggregations=[…]`).
- New skill
  `bird-interact-agents/.claude/skills/translate-mini-interact-kb/` —
  mini-interact wrapper that mandates `meta={"kb_id": <id>}` (or
  `meta={"kb_ids": [...]}` for multi-KB entities) on every encoded
  entity. References the generic skill for recipes.
- New verifier `bird-interact-agents/scripts/verify_kb_coverage.py` —
  partitions every KB id into `encoded ∪ documented`; exits 0 only
  when both `unaccounted` and `overlap` are empty.

### W4a — pilot on 2 simple DBs

- `households` and `polar` verifier-clean against SLayer 0.4.2.
- Skill + verifier reached steady state (zero changes between the
  second pilot's first attempt and final state).
- Surfaced the **peer-join limitation** (auto-FK joins only go
  child → parent, not peer ↔ peer through a shared parent), documented
  in `slayer_models/_notes/polar.md`.

### W4b — fan-out across 25 remaining DBs

- All 27 mini-interact DBs verifier-clean. 5 partial-DB stub-refinement
  passes ran for DBs that hit usage limits, plus 2 fresh dispatches
  (`sports_events`, `virtual`).
- 312 KB entries deferred to per-DB
  `bird-interact-agents/slayer_models/_notes/<db>.md` files; 1,285 KB
  entries encoded.
- 12 distinct SLayer bugs surfaced and filed as DEV-1330 through
  DEV-1341 under DEV-1329. Each has Symptom, Reproduction, Where
  Surfaced, Impact, Root Cause Hypothesis, Suggested Fix, References.

### W5 — HARD-8 preprocessor (this PR)

- New `src/bird_interact_agents/hard8_preprocessor.py`:
  - `extract_deleted_kb_ids(task_data)` — flattens
    `knowledge_ambiguity[*].deleted_knowledge` (int-or-list) into a
    `set[int]`.
  - `build_task_variant_storage(...)` — async; loads a per-DB
    `YAMLStorage`, drops models / columns / measures / aggregations
    whose `meta.kb_id` (or any `meta.kb_ids`) is in the deletion set,
    writes the survivors to a task-scoped scratch dir, returns its
    path. Empty deletion set short-circuits to the canonical path with
    no copy.
- New helpers in `src/bird_interact_agents/harness.py`:
  - `_task_variant_workdir(instance_id)` — scratch under
    `$TMPDIR/bird_interact_w5_variants/<instance_id>/`.
  - `resolve_task_storage_dir(...)` — single per-task entry point that
    handles raw vs slayer mode, missing storage root, and the variant
    branch. Returns `(slayer_storage_dir, deleted_kb_ids)`.
  - `finalize_result_row(row, *, deleted_kb_ids, slayer_storage_dir)`
    — stamps the two new fields onto every result row.
- All 5 framework adapters
  (`agents/{claude_sdk,agno,mcp_agent,pydantic_ai,smolagents}/agent.py`)
  now call `resolve_task_storage_dir` instead of computing
  `slayer_storage_dir` inline; their 3 result-row return points (skip,
  error, success) are wrapped through `finalize_result_row`.
- `src/bird_interact_agents/run.py` wraps the oracle and
  error-fallback rows the same way.
- 8 new unit tests in `tests/test_hard8_preprocessor.py` — all passing.

### W6 — slayer-mode wiring end-to-end (this PR)

- The harness side was 80% wired before W5: `--query-mode slayer` /
  `--slayer-storage-root` flags, per-task `slayer_storage_dir`,
  per-task slayer MCP launch via `slayer_mcp_stdio_config`, slayer-only
  tool whitelist. The W5 helper plugs the variant path through
  transparently — no further adapter changes needed.
- New `tests/test_slayer_models_loadable.py` — 28 tests parameterized
  over all 27 DBs + a sentinel; every `slayer_models/<db>/` round-trips
  through `YAMLStorage` (datasource present, ≥1 model, every model
  Pydantic-validates against the current `SlayerModel` schema).
- New `tests/test_w6_resolve_and_load.py` — 5 integration tests against
  the real `slayer_models/households/` tree using KB id 15
  (`properties.bathroom_ratio`), the same KB id `households_1` deletes
  in `mini_interact.jsonl`. Confirms the variant has the column
  dropped, the canonical YAML still has it, raw mode + missing
  storage_root short-circuit correctly.
- Manual end-to-end smoke (`bird-interact --framework claude_sdk
  --query-mode slayer --limit 5` against a slice including a task with
  non-empty deletion list) is **deferred until W7 unfreezes** —
  documented as a one-shot verification step rather than a recurring
  pytest test.

---

## Failure-mode analysis — all 312 deferred KB sections

Source: regex pass over the 26 non-empty
`slayer_models/_notes/<db>.md` files (4 DBs have **zero** deferrals:
`disaster`, `exchange_traded_funds`, `hulushows`, `planets_data`),
followed by manual reclassification of the OTHER bucket (~72 entries
that the regex missed because they used phrasings like `Schema gap.`
or `cascades on top of prose-only`).

### Distribution

| Bucket             | Count | %    | Unblocked by SLayer bug fix? |
|--------------------|------:|-----:|------------------------------|
| **PEER-JOIN**      |  115  | 37%  | **Yes** — DEV-1338, 1340, 1341, 1339, 1330 |
| **CASCADE**        |  119  | 38%  | **Yes** if their parent unblocks |
| **SCHEMA-GAP**     |   37  | 12%  | No — KB describes data not in schema |
| **AMBIGUOUS-PROSE**|   19  |  6%  | No — KB underspecifies thresholds/weights |
| **R-WINDOW**       |   11  |  4%  | **Yes** — DEV-1336 |
| **TIME-ANCHOR**    |    6  |  2%  | No — needs harness-side anchor |
| **DML**            |    3  |  1%  | No — not modelable |
| **CROSS-DB**       |    1  |  0.3%| No |

### Bucket explanations + per-bucket bug-fix mapping

**PEER-JOIN (115 — 37%).** Composite metric spans 2+ peer tables that
all FK to a common parent (e.g. polar's RSSI = 0.6×REC[`thermalsolarwindandgrid`]
+ 0.4×WRMI[`waterandwaste`]). Auto-FK joins go child → parent only;
two children of the same parent can't reach each other through a bare
`Column.sql`. Natural encoding is R-MULTISTAGE — a query-backed model
that joins each peer to the parent in its own stage and composes in a
final stage. **Unblocked by**:
[DEV-1340](https://linear.app/motley-ai/issue/DEV-1340) (inline named
stages can't be `joins.target_model`),
[DEV-1341](https://linear.app/motley-ai/issue/DEV-1341) (final-stage
`*:count` + composite measure → `__agg3__` placeholder leak),
[DEV-1338](https://linear.app/motley-ai/issue/DEV-1338) (no auto
reverse joins),
[DEV-1339](https://linear.app/motley-ai/issue/DEV-1339) (multi-hop
alias path doesn't qualify joined-model derived columns),
[DEV-1330](https://linear.app/motley-ai/issue/DEV-1330) (global model
namespace collision). Estimate ~90% become encodable.

**CASCADE (119 — 38%).** KB-N depends on KB-M which is itself
deferred (e.g. polar KB 40 = "EOC depends on KB #36 ECAC, deferred").
Auto-unblock when their parent does. Most parents are PEER-JOIN, so
most cascades follow. Estimate ~80% unblock.

**R-WINDOW (11 — 4%).** Window function (e.g. `row_number() over (...)
<= 3`) inlined in WHERE — rejected by SQLite. Unblocked by
[DEV-1336](https://linear.app/motley-ai/issue/DEV-1336). Near-100%
unblock.

**SCHEMA-GAP (37 — 12%).** KB describes data the schema doesn't
carry. Concentrated in:
- `sports_events` (~10 of 21 deferred): no race-results / per-driver
  finish-times / fastest-lap data. About half the file's deferrals.
- `cold_chain_pharma_compliance` (~7 of 9): explicit `Schema gap.`
  markers — temperature accuracy / packaging cost data.
- `crypto` (~3 confirmed + many regex-flagged): no momentum
  indicators, no intraday breakout data.
- `solar` (~4): missing `POAIrradianceWM2`, warranty-curve data.
- `households` (3 stale: KB 34 zone types, 35 utility access, 38
  dwelling condition — all describe label sets the schema doesn't
  carry).

These are real data gaps. SLayer fixes don't help; they stay
deferred.

**AMBIGUOUS-PROSE (19 — 6%).** KB pins concept but not numbers.
`households` dominates (Crowded Household = "Household Density greater
than a threshold" — threshold unspecified; Modern Dwelling = "specific
Dwelling Type and active Cable TV Status" — neither pinned). The
agent at query time can ad-hoc compose against helper columns, so the
encoding deferral is the right call.

**TIME-ANCHOR (6 — 2%).** KB anchors to "current period" /
"year-over-year" / "last 30 days" but the model has no `now()` anchor
(e.g. `labor_certification_applications` KB 36 Wage Growth Rate;
`credit` KB 47 Declining Credit Health = "trend over last N months").
Out of scope until the harness or the KB itself supplies an anchor.

**DML (3 — 1%).** `households` KB 31/32/33: INSERT/UPDATE/DELETE
workflows. SLayer models describe queryable shape, not mutations.
Permanently not-applicable.

**CROSS-DB (1 — 0.3%).** `cross_db.md` KB 78. By construction.

### Top contributors (where to send W4c agents)

- **PEER-JOIN re-encoding**: `mental` (27), `archeology` (16),
  `crypto` (12), `fake` (12), `vaccine` (11), `polar` (9), `robot`
  (8) → ~95 of 115 entries.
- **CASCADE re-encoding** (auto-resolves when parent does): `vaccine`
  (21), `fake` (18), `crypto` (15), `archeology` (9), `news` (8),
  `polar` (8), `sports_events` (~10).
- **R-WINDOW re-encoding**: `robot` (4), `fake` (2), 5 others with 1
  each.

### Predicted post-W4c state

After W4c re-encoding pass: ~211 of 312 entries become encoded, ~100
stay genuinely deferred (37 SCHEMA-GAP + 19 AMBIGUOUS + 6 TIME-ANCHOR
+ 3 DML + 1 CROSS-DB + ~30 stubborn PEER-JOIN/CASCADE residuals that
even the bug fixes don't reach).

---

## What's left

### Section 2 — gates on the SLayer release

#### W4c — Re-encoding pass

After the 12 bug fixes from DEV-1330–1341 ship as a SLayer release and
the user reconnects the MCP, run a parallel fan-out over the 23 DBs
with deferred entries.

**Workflow per DB (one parallel agent per DB):**

1. Re-read `slayer_models/_notes/<db>.md`. Each `## KB <id> — <name>`
   section is a candidate.
2. Classify the Reason against the bug-fix mapping (above table) and
   either re-encode or update the section's `Status:` line to
   `Status: deferred — genuine <bucket>`.
3. For each successfully re-encoded entry: remove the section from the
   notes file, attach `meta.kb_id` (or `meta.kb_ids`) on the new
   entity, re-export YAML via `scripts/export_slayer_models.py --db
   <db>`.
4. Run `scripts/verify_kb_coverage.py --db <db>`. Must still exit 0.

**Sequencing — 23 DBs in parallel, in priority order:**

- **Tier 1** (high-yield, ~150 of ~211 unblockables): `mental`,
  `vaccine`, `crypto`, `fake`, `archeology`, `polar`, `news`,
  `cybermarket`, `robot`.
- **Tier 2** (small mixed-bucket): `solar`, `households`,
  `cold_chain_pharma_compliance`, `museum`, `organ_transplant`,
  `gaming`, `reverse_logistics`, `credit`.
- **Tier 3** (dominantly schema-gap, low yield): `sports_events`,
  `labor_certification_applications`, `virtual`, `cross_db`, `alien`,
  `insider`. Mostly Status-line triage.

**Verification gate:** for every DB with deferred entries, verifier
exits 0 and remaining notes-file sections are all in {SCHEMA-GAP,
AMBIGUOUS, TIME-ANCHOR, DML, CROSS-DB}. No remaining PEER-JOIN /
CASCADE / R-WINDOW sections.

#### W7 — Benchmark + iteration (HOLD)

**Do not start until explicit go-ahead.** When green-lit:

1. Manual end-to-end smoke (the W6 deferred step): `bird-interact
   --framework claude_sdk --query-mode slayer --limit 5` against a
   slice including at least one task with non-empty
   `deleted_knowledge`. Confirm:
   - The agent sees only SLayer MCP tools, no raw-SQL fallback.
   - `mcp__slayer__list_datasources()` returns the per-task DB.
   - `mcp__slayer__models_summary()` returns the encoded models.
   - Result rows record `deleted_kb_ids` + `variant_storage_path`.
2. `scripts/run_three_way.sh --mode a-interact --limit 30` for a pilot
   slice; triage failures.
3. Full 300 with per-task results persisted to the results DB.
   Compare slayer vs raw vs original phase1/phase2 rates side-by-side
   (`comparison.json`).

---

## Sequencing summary

```
Section 1 (LANDED — PR #10)              Section 2 (gates on SLayer release)
─────────────────────────                ───────────────────────────────────
W5 HARD-8 preprocessor              ──→  (W4c uses W5's preprocessor)
W6 slayer-mode wiring + tests            W4c re-encoding pass
                                         W7 benchmark (HOLD until explicit go)
```

## Verification gates

| Gate | Pass condition |
|---|---|
| W5  | A task with `deleted_knowledge: [N]` cannot see KB id N's encoded entity in the variant storage via `mcp__slayer__inspect_model`. Variant path + `deleted_kb_ids` logged in result rows. |
| W6  | YAMLStorage-loadability test green for all 27 DBs. Resolver pipeline passes against real households KB id 15. (Manual smoke deferred to W7 kickoff.) |
| W4c | Per-DB verifier still exits 0; remaining notes-file sections are all SCHEMA-GAP / AMBIGUOUS / TIME-ANCHOR / DML / CROSS-DB. |
| W7 (gated) | Full 300-task run completes; comparison shows slayer vs raw vs original side-by-side; per-task results in results DB. |

## Critical files

Section 1 (landed in PR #10):

- `src/bird_interact_agents/hard8_preprocessor.py` — new (W5)
- `src/bird_interact_agents/harness.py` — modified (shared
  `extract_deleted_kb_ids` / `resolve_task_storage_dir` /
  `finalize_result_row` helpers)
- `src/bird_interact_agents/agents/{claude_sdk,agno,mcp_agent,pydantic_ai,smolagents}/agent.py`
  — modified (replaced inline `slayer_storage_dir` with the harness
  helper; wrapped result rows)
- `src/bird_interact_agents/run.py` — modified (wrapped oracle +
  error-fallback rows)
- `tests/test_hard8_preprocessor.py` — new (8 unit tests)
- `tests/test_slayer_models_loadable.py` — new (28 parametrized
  loadability tests)
- `tests/test_w6_resolve_and_load.py` — new (5 real-data integration
  tests)

Section 2 (W4c, after SLayer release):

- `slayer_models/<db>/*.yaml` — modified (more entries gain
  `meta.kb_id`)
- `slayer_models/_notes/<db>.md` — modified (PEER-JOIN / CASCADE /
  R-WINDOW sections removed; remaining sections updated)
