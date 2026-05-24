# fake — KB coverage notes

KB entries that could not be encoded as SLayer entities. Section
headers are parsed by the verifier (`## KB <id> — …`); the body is
free-form Markdown.

## KB 50 — Temporal Pattern Deviation Score (TPDS)

Reason: Definition is `sqrt(Σᵢ₌₁²⁴ ((obsfreq_i − expfreq_i) / expfreq_i)²)`,
i.e. a chi-square-like deviation across 24 hourly buckets between an
observed frequency and an expected one. Neither `obsfreq` nor
`expfreq` exists anywhere in the schema (no per-account hourly
activity histogram, no baseline), and the JSON `acttimedist` column is
empty in sample data. There is no reasonable column-meaning resolution
for the two operands.

Status: deferred — SCHEMA-GAP

## KB 52 — Multi-Account Correlation Index (MACI)

Reason: Definition averages a behavioural correlation `corr(i,j)` over
all linked-account pairs in a cluster. The schema has no per-account
behaviour vector or pairwise-correlation column (`linkacctnum` only
counts linked accounts; it doesn't tell us *which* accounts or expose
their per-feature time series). Expressing pairwise Pearson
correlation requires a feature matrix that the schema doesn't carry.

Status: deferred — SCHEMA-GAP

## KB 53 — Reputation Volatility Index (RVI)

Reason: Definition is `σ(reputscore)/μ(reputscore) × (1 + |Δreputscore|/Δt)`.
The schema stores `reputscore` only on `moderationaction`, with no
per-account historical series — moderation actions don't carry an
"as-of" reputscore snapshot. Computing a per-account σ/μ and a
reputscore time-derivative both require a longitudinal dimension that
isn't in the data.

Status: deferred — SCHEMA-GAP

## KB 54 — Content Distribution Pattern Score (CDPS)

Reason: Definition is `0.4 × entropy(posttimes) + 0.3 × burstiness +
0.3 × (1 − periodicity)`. None of `posttimes` (a per-post timestamp
sequence), `burstiness`, or `periodicity` exists in the schema. The
content tables expose aggregates (`postnum`, `postfreq`, `postintvar`)
but not the per-post timestamp series the entropy / burstiness /
periodicity terms require.

Status: deferred — SCHEMA-GAP

## KB 55 — Behavioral Consistency Score (BCS)

Reason: Composite of `(1 − TPDS) × (1 − RVI) × (1 − patterndev/100)`.
TPDS (KB 50) and RVI (KB 53) are themselves SCHEMA-GAP. `patterndev`
also has no schema column. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 56 — Network Synchronization Index (NSI)

Reason: Definition is `(Σᵢ Σⱼ sync(i,j) / clustsize) × MACI` with a
pairwise `sync(i,j)` between cluster members and MACI from KB 52.
Neither the per-pair sync function nor MACI's underlying correlation
matrix is expressible from the available columns.

Status: deferred — SCHEMA-GAP

## KB 58 — Authentication Pattern Score (APS)

Reason: Composite of `(1 − TEI) × BCS × (1 − authanom/100)`. BCS
(KB 55) cascades from TPDS / RVI. `authanom` also has no schema
column. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 59 — Cross-Platform Correlation Score (CPCS)

Reason: `CPRI × MACI × (1 + platformlinks/10)`. MACI (KB 52) is
SCHEMA-GAP and `platformlinks` has no schema column. Cascading
defer.

Status: deferred — SCHEMA-GAP

## KB 60 — Coordinated Influence Operation

Reason: Cluster-level predicate requiring `NSI > 0.8 AND CAE > 0.7
AND contains ≥1 Content Manipulation Ring`. NSI (KB 56) is
SCHEMA-GAP. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 61 — Behavioral Pattern Anomaly

Reason: Account predicate `BCS < 0.3 AND TPDS > 0.7 AND not Trusted
Account`. Both BCS and TPDS are SCHEMA-GAP. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 62 — Cross-Platform Bot Network

Reason: A Bot Network with `CPCS > 0.8 AND all accounts have similar
MACI patterns`. CPCS (KB 59) and MACI (KB 52) are both SCHEMA-GAP.
Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 63 — Authentication Anomaly Cluster

Reason: Cluster with `avg APS < 0.3 AND contains ≥1 Authentication
Risk Account`. APS (KB 58) is SCHEMA-GAP. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 64 — Network Influence Hub

Reason: Account with `NIC > 0.8 AND CAE > 0.7 AND part of a
Coordinated Influence Operation`. The CIO membership (KB 60) is
SCHEMA-GAP, so the third clause can't be evaluated. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 65 — Reputation Manipulation Ring

Reason: A Content Manipulation Ring where every member has
`RVI > 0.7 AND similar CDPS patterns`. RVI (KB 53) and CDPS (KB 54)
are both SCHEMA-GAP. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 66 — Synchronized Behavior Cluster

Reason: Cluster with `NSI > 0.9 AND all members have similar BCS
patterns`. NSI (KB 56) and BCS (KB 55) are both SCHEMA-GAP.
Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 67 — Multi-Platform Threat Network

Reason: Cross-Platform Threat where `CPCS > 0.8 AND every member is
in a Synchronized Behavior Cluster`. Depends on KB 59 (CPCS) and
KB 66 (Synchronized Behavior Cluster), both SCHEMA-GAP.

Status: deferred — SCHEMA-GAP

## KB 68 — Advanced Influence Campaign

Reason: A Mass Manipulation Campaign that contains ≥1 Network
Influence Hub and has high NSI. KB 64 (Network Influence Hub) and
KB 56 (NSI) are both SCHEMA-GAP. Cascading defer.

Status: deferred — SCHEMA-GAP

## KB 69 — Persistent Pattern Anomaly

Reason: A Behavioral Pattern Anomaly that "persists for over 30 days
and maintains high TPDS". The base predicate (KB 61) is SCHEMA-GAP,
and the persistence clause needs a per-account TPDS time series the
schema doesn't carry. Combined SCHEMA-GAP + TIME-ANCHOR.

Status: deferred — SCHEMA-GAP

## KB 85 — review priority

Reason: Definition states this is "a field in the `account` table set
to a specific value like `'Review_Inactive_Trusted'`". The
`account` schema (`accindex`, `acctident`, `platident`, `plattype`,
`acctcreatedate`, `acctagespan`, `acctstatus`, `acctcategory`,
`authstatus`) has no such column — the closest enum is `acctstatus`,
whose documented values are Active / Deleted / Suspended / Dormant
(no `Review_*` value). This is a workflow artifact the schema
doesn't actually expose, not data we can model.

Status: deferred — SCHEMA-GAP

## KB 77 — High-Activity Account

Reason: Verbatim restatement of KB 74; encoded entity is `moderationaction.is_high_activity_account` with `meta.kb_id = 74`.

Status: not-applicable — duplicate of KB 74
