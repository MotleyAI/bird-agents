# W4c — KB-to-SLayer enrichment summary

Snapshot of the slayer-mode model state as of **2026-05-06**, after
the W4c re-encoding pass and one follow-up. This is the reference
for what's already done; ongoing pre-W7 work is tracked under
[DEV-1361](https://linear.app/motley-ai/issue/DEV-1361) (type-aware
CAST emission) and
[DEV-1362](https://linear.app/motley-ai/issue/DEV-1362) (KB
self-annotation + multi-KB-entity splits + description refresh +
c-interact prompt render).

## What was built

### Skills (the load-bearing artefact)

`bird-interact-agents/.claude/skills/`:

- **`kb-to-slayer-models/SKILL.md`** — domain-agnostic recipe book
  for translating a structured knowledge base into edits on a
  SLayer datasource via the v0.4.x MCP. Recipes:
  R-DESCRIBE, R-JOIN, R-COL, R-CASE, R-FILTER, R-MEASURE, R-AGG,
  R-RESOLVE, R-MULTISTAGE, R-PEER-JOIN (default for shared-parent
  composites), R-WINDOW, R-EXISTS, R-VAR, R-HOST, R-PROSE. Includes
  a "Working from a partially-encoded model" subsection for the
  W4c verify-then-fill workflow and a "Notes-file regeneration"
  subsection with canonical Status values.
- **`translate-mini-interact-kb/SKILL.md`** — mini-interact
  wrapper: mandates `meta.kb_id` on every encoded entity, defines
  the per-DB notes-file format, points at the W4c override at the
  top of its Workflow section.

The skills tell agents: use the SLayer MCP for all model edits
(never edit YAML directly); discard the existing notes file under
the W4c override; encode by reading the KB JSONL + schema +
column-meanings JSON (never gold SQL); regenerate notes with one
of the canonical Status values for unencodable entries.

### Per-DB instruction files (transient)

`bird-interact-agents/_w4c_instructions/`:

- `_README.md` — shared workflow for every agent.
- One `<db>.md` per task-relevant database (13 files;
  `archeology, crypto, cybermarket, fake, households,
  labor_certification_applications, mental, museum, news, polar,
  reverse_logistics, vaccine, virtual`). Each file is ~20-40
  lines: workflow pointer + DB-specific quirks the prior W4b pass
  got wrong or non-obvious structural points the agent should
  know. Robot intentionally absent — its KBs depend on transforms
  that ship with [DEV-1353](https://linear.app/motley-ai/issue/DEV-1353).

### Tooling

`bird-interact-agents/scripts/`:

- **`verify_kb_coverage.py`** (W1; pre-existing) — partitions every
  KB id into `encoded ∪ documented`; exits 0 only when both
  `unaccounted` and `overlap` are empty. Reads `meta.kb_id` and
  `meta.kb_ids` across all entities + the per-DB notes file headers.
- **`export_slayer_models.py`** (pre-existing) — exports YAML from
  the live MCP storage to `slayer_models/<db>/`. Patched during W4c
  by the `archeology` agent to pass `data_source=db` into
  `list_models` / `get_model` so multi-datasource storage
  disambiguates correctly.

### Result: 13 datasources refreshed

Verifier exits 0 across all 27 DBs after the W4c pass. Per-DB
outcomes (verified / fixed / newly-encoded / deferred KB id counts,
from the agent reports):

| DB | verified | fixed | newly encoded | deferred (Status) |
|---|---:|---:|---:|---|
| `archeology` | 50 | 0 | 0 (already done in prior pass) | 2 (SCHEMA-GAP: 16, 41) |
| `crypto` | 33 | 0 | 4 (KB 46, 47, 49, 51) | 6 (AMBIG/TIME/SCHEMA: 14, 17, 19, 42, 44, 48) |
| `cybermarket` | 41 | 5 (broken multistage rebuilt) | 0 | 3 (SCHEMA-GAP / AMBIG: 27, 34, 44) |
| `fake` | 68 | 0 | 0 | 19 (all SCHEMA-GAP — schema lacks pairwise / time-series data) |
| `households` (W4c) | 25 | 0 | 0 | 20 |
| `households` follow-up | — | 0 | 3 (KB 19, 26, 42 with explicit hints) | 17 (KB 32 stays DML; 16, 22, 23, 25, 27, 28, 30, 31, 33, 34, 35, 38, 39, 40, 41, 42, 43) |
| `labor_certification_applications` | 47 | 1 (`cases.has_attorney` SQL fix) | 6 (KB 32, 36, 42, 49, 53, 54) | 3 (SCHEMA-GAP / AMBIG: 22, 38, 55) |
| `mental` (W4c) | many (all `meta.kb_id` stamps verified) | several `meta.kb_id` stamps added; joins added | 1 big `facility_metrics` query-backed model + 3 query-backed predicates (KBs 12, 56, 57) | 1 (KB 61 — agent flagged scalar-broadcast difficulty) |
| `mental` follow-up | — | 0 | 1 (KB 61 — `facility_performance_quadrant` 4-stage scalar-broadcast multistage; 206/147/149/93 facilities across the 4 quadrants) | 0 |
| `museum` | 51 | 0 | 4 (KB 9, 17, 36, 45 — including `BudgetAllocStatus`-driven CBE) | 2 (AMBIG: 39, 47) |
| `news` | 49 | 2 (added missing kb_id stamps) | 2 (KB 31, 37 — query-backed multistage) | 14 (mostly AMBIG; KB 38 SCHEMA-GAP) |
| `polar` | 22 | 9 (formula corrections — KB 1 ORS, 2 ESI, 12 priority bands, 25 water bands, 31 PTEC rerouted as multistage, 46 CIPL rerouted, 52 ESCS split off, 53 WRMSC split off; recreated `equipment` and `communication` models that had vanished from storage) | 17 (R-MULTISTAGE peer-joins via `equipment`: KB 10, 33, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 49, 50, 51) | 1 (KB 41 SCHEMA-GAP — `emergencycommunicationstatus` not in schema) |
| `reverse_logistics` | 27 | 1 (KB 6 WCR added `auth_status='Approved'` filter) | 1 (KB 9 RCCI — multistage row-vs-channel-avg) | 2 (SCHEMA-GAP: 3, 18 — `orders.txndate` is null for all rows) |
| `vaccine` | 33 | several (cross-cutting: every model's auto-FK join re-pointed from the renamed `shipments` to `vaccine_shipments`) | 16 (R-MULTISTAGE peer-joins via `container` / shipment / vehicle: KB 3, 10, 12, 18, 19, 32, 33, 34/45, 35/40, 36/41/48, 37/42/46, 38/43/47, 39/44, 49, 54, 65) | 8 (AMBIGUOUS-PROSE: free α/β/γ parameters or unspecified scope: 50, 51, 52, 55, 56, 57, 58, 59) |
| `virtual` | 53 | 0 | 1 (KB 29 — `multi_idol_supporters` 3-stage multistage) | 1 (SCHEMA-GAP: 43 — no per-content-piece grain) |

Every DB ends the W4c pass with `verify_kb_coverage.py --db <db>`
exiting 0.

### Result: critical Linear issues filed

- [DEV-1350](https://linear.app/motley-ai/issue/DEV-1350) — design
  note on peer-join encoding; documents why R-MULTISTAGE is the
  default (cardinality-safe regardless of m:N, no manual reverse
  joins required, no dependency on DEV-1338).
- [DEV-1353](https://linear.app/motley-ai/issue/DEV-1353) — adding
  `percent_rank` / `dense_rank` / `row_number` / `ntile` as
  first-class SLayer transforms. Robot's task-relevant KBs depend
  on `percent_rank` so robot is deferred until this lands.

## State of the storage

**Storage location:** `~/.local/share/slayer/` (the live MCP
storage; the `slayer_models/<db>/` directories under
`bird-interact-agents/` are exports of this storage).

**Datasources:** all 28 mini-interact DBs auto-ingested into the
storage. (28 because the `demo` Jaffle Shop datasource also lives
there; 27 are mini-interact DBs.)

**Coverage:**

- 27 DBs × `verify_kb_coverage.py --db <db>` exits 0.
- Total KB ids encoded: ~1300+; deferred: ~150 (mostly
  AMBIGUOUS-PROSE, SCHEMA-GAP, and the robot pending block).
- Robot's notes file is unchanged from W4b state; its 13
  task-relevant deferrals stay deferred until DEV-1353.

**Diff scope (vs the start of W4c):** ~13 notes files rewritten,
many YAML files across 13 DBs updated, 13 instruction files in
`_w4c_instructions/` added, two skill files updated, one harness
script (`export_slayer_models.py`) patched.

## Known caveats / unfinished work (handed off to DEV-1361 / DEV-1362)

### Type-mismatch failures in result tuples — DEV-1361

The benchmark harness compares submitted-SQL result tuples to gold
SQL result tuples after modest normalisation that doesn't coerce
integers to floats or `json_extract`-text to numbers. SLayer's
`Column.type` field exists but doesn't drive CAST emission today.
Several patterns will fail at scale (json_extract values,
integer-division ratios, COUNT vs SUM(CASE WHEN)). Fix:
type-aware CAST emission per DEV-1361.

### Multi-KB-entity prevalence and kb_id misattribution — DEV-1362

A storage walk (`/tmp/claude/multi_kb_audit.py`, in this branch)
found **96 entities across 22 datasources** carrying
`meta.kb_ids` (plural). Six pattern buckets — A (calc+threshold),
B (value-illustration JSON blob), C (illustration+calc+threshold
trinity), D (multi-formula model), E (over-grouping monsters), F
(source-duplicate KBs). Almost all are encoding shortcuts that
should be split into single-KB entities; only Bucket F (~5 cases)
is a genuine source-duplicate situation that the notes file can
absorb. Fix: splitting pass per DEV-1362.

Related: many entities have `meta.kb_id` stamped on a *data
carrier* (a JSON blob column that holds raw values) rather than
on the entity that *operationalizes* the KB (the helper column
that extracts and computes). The skill currently doesn't enforce
this rule; DEV-1362 adds it.

### Description inconsistency — DEV-1362

W4c agents wrote varying descriptions: some rich (e.g.
`polar.cabinenvironment.is_habitable` reads "Cabin Habitability
Standard (KB #17): 18–24°C cabin temp AND O₂ > 19.5% AND CO₂ <
1000 ppm"), some empty, some paraphrased. The benchmark agent at
runtime relies on these descriptions via `models_summary` /
`inspect_model` to find the right entity for a user query. Fix:
description-refresh script per DEV-1362, populating
`label = KB.knowledge` and
`description = KB.definition + " — " + KB.description` verbatim,
wrapped in `[kb=<id>]…[/kb=<id>]` markers for idempotency.

### Robot pending — DEV-1353

13 task-relevant deferrals in robot. Re-dispatch one agent for
robot once `percent_rank` / `dense_rank` / `row_number` / `ntile`
ship in a SLayer release. Notes file unchanged.

### W7 itself

Gated benchmark run not started. Preconditions: DEV-1361 (type
casts) and DEV-1362 (description refresh + multi-KB splits)
should land first. Robot can be brought back in once DEV-1353
ships.

## Cross-references

- `bird-interact-agents/PROGRESS.md` — broader project state
  (W-prep / W0 / W1 / W4a / W4b / W5 / W6 / what's left).
- `bird-interact-agents/.claude/skills/kb-to-slayer-models/SKILL.md`
- `bird-interact-agents/.claude/skills/translate-mini-interact-kb/SKILL.md`
- `bird-interact-agents/_w4c_instructions/` — per-DB instruction
  files used by the W4c agents.
- `bird-interact-agents/scripts/verify_kb_coverage.py`
- `bird-interact-agents/scripts/export_slayer_models.py`
- `/tmp/claude/multi_kb_audit.py` — the audit script that produced
  the 96-entity inventory; lives at the path during this session,
  worth keeping under `bird-interact-agents/scripts/` long-term.
- DEV-1316, DEV-1318, DEV-1329, DEV-1338, DEV-1339, DEV-1340,
  DEV-1341, DEV-1350, DEV-1353, DEV-1361, DEV-1362 — Linear
  issues covering parent initiative, the current branch, the
  benchmark-bugs umbrella, the original peer-join SLayer
  fix-tickets, the design note, and the four ongoing pre-W7
  fix-tickets.
