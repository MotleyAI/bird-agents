# gaming — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/gaming/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — …` headers to
distinguish "skipped on purpose" from "missed".

The schema is a star around `deviceidentity` (each device has at most
one row in each per-aspect table: `mechanical`, `audioandmedia`,
`interactionandcontrol`, `physicaldurability`, `rgb`, `performance`,
`testsessions`). Auto-FK joins go child → parent (the per-aspect table
holds the FK to `deviceidentity`). To make every composite KB metric
reachable from a single host, a few **reverse joins** were added on
`deviceidentity` (→ `mechanical`, → `audioandmedia`,
→ `interactionandcontrol`); from there `physicaldurability` and `rgb`
are reachable via `interactionandcontrol`. `mechanical.ci`, `mechanical.spr`,
`physicaldurability.ds` etc. were inlined as natural-column expansions
in the composite formulas because SLayer's `Column.sql` substitution
does not propagate **synthetic** columns across joined tables — only
natural columns work in cross-table refs.

All 51 single- or multi-table-reachable KB entries are encoded on the
gaming models. The 3 entries below are deferred because they require
either (a) a setup-level entity that does not exist in the schema, or
(b) a window function that the project notes flag for R-WINDOW
deferral.

## KB 19 — Full-Featured Gaming Setup

Reason: the predicate is "average GDVI > 8.5 across at least three
different device categories (Mouse, Keyboard, Headset, etc.)". This
is a property of a **setup** — a multi-device collection — but the
schema has no setup / collection / owner table grouping multiple
`deviceidentity` rows. The `testsessions.devscope` enum carries a
single device category per session (Keyboard, Headset, Gamepad, Mouse,
Controller), but devices are not grouped into setups, so we cannot
compute "average GDVI across categories within a setup". Status:
deferred — would need either a synthetic setup entity injected into
the schema or a query-time aggregation across all devices that the
agent can compose ad-hoc using `deviceidentity.gdvi` grouped by
`testsessions.devscope`.

## KB 49 — Elite Gaming Ecosystem

Reason: same shape as KB #19 — "a multi-device setup where each
component achieves VPI > 8.5, with an average GDVI > 9.0 across at
least three different device categories". No setup-grouping entity in
the schema. The per-device pieces (`vpi`, `gdvi` on `deviceidentity`)
are encoded; the cross-device aggregation is not. Status: deferred —
agent can filter on `deviceidentity.vpi > 8.5` and compute average
`gdvi` grouped by `testsessions.devscope` ad-hoc.

## KB 53 — Global Efficiency Percentile (GEP)

Reason: defined as `PERCENT_RANK() OVER (ORDER BY BER) * 100`. This
is a window function whose output is a row-level dimension —
R-WINDOW per the kb-to-slayer-models recipe set, which the project
brief flags for deferral when not inlinable. PERCENT_RANK in a
`Column.sql` would change the cardinality of any aggregating query
(window evaluation depends on the surrounding query's frame), so it
needs a query-backed multistage model. The base metric BER is
encoded on `deviceidentity` (KB #1), so the agent can compute GEP at
query time as `rank(ber)` or via a custom multistage model when needed.
Status: deferred — R-WINDOW (multistage) per project brief.
