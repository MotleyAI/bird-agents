# cybermarket — KB entries not encoded as model entities

## KB 27 — Market Migration Indicator

The KB describes "a pattern where multiple vendors (vendregistry) and
buyers (buyregistry) associated with one market (mktregistry) begin
appearing on another market within a short timeframe (less than 30
days)". This requires a stable cross-market identity for both
vendors and buyers — but in this schema `vendors.vendregistry` is the
PK of a (vendor, market) row (vendors FK to markets via mktref) and
`buyers.buyregistry` similarly (buyers FK to markets via mktref).
There is no cross-market vendor or buyer identity column, no name /
key / fingerprint that would let us say "the same vendor on two
different markets". The "begin appearing on another market" temporal
half also has no column to anchor it (vendors has `vendlastmoment`
but no first-appearance column on a per-market basis that could be
compared across markets).

This is the same identity-mapping shortfall that affects KB #29
(Cross-Platform Operator), KB #31 (VNC), KB #38 (CPRA), KB #41
(Market Kingpin), and KB #48 (Multi-Platform Threat Entity) — those
sibling entries each carry a literal-fit encoding documenting the
shortfall in the entity description. KB #27, however, has no scalar
score or row-level boolean shape that would let a literal-fit
produce *anything* meaningful: the predicate is intrinsically a
cross-market, cross-row temporal pattern (Σ_vendors and Σ_buyers
appearing on a *second* market within 30 days of leaving a *first*).
A literal-fit would collapse to a constant 0 or 1 with no semantic
content.

Status: deferred — SCHEMA-GAP

## KB 34 — Transaction Velocity Metric (TVM)

The KB definition is

```
TVM = COUNT(txregistry) / (MAX(eventstamp) - MIN(eventstamp))
      * payamtusd/500
      * (1 + paymethod_weight * 0.1)
```

which mixes per-source aggregates (`COUNT(txregistry)`,
`MAX(eventstamp)`, `MIN(eventstamp)`) with row-level columns
(`payamtusd`, `paymethod_weight`). The "single source" the KB hints
at is undefined: the transaction table has FKs to market, product,
and buyer, but the KB's prose ("rapidity and volume of transactions
from a single source") doesn't specify which grouping key. Different
choices (per-buyer, per-vendor, per-market, per-wallet via
riskanalysis) give materially different metric values, and the KB's
downstream user (KB #44 Flash Transaction Cluster) requires a
"related-sources" cluster which is itself unspecified.

Beyond the grouping ambiguity, the time arithmetic
`MAX(eventstamp) - MIN(eventstamp)` is unitless in the formula —
days, hours, seconds all give different scales, and the SLayer
engine does not currently expose a portable `julianday`/`epoch`
difference for `text(6)` timestamps without a SQLite-specific
expression.

Encoding helper columns (per-buyer counts, payamtusd/500,
paymethod_weight) lets an agent compose the metric ad-hoc once a
specific source grain is supplied at query time — but the metric
itself can't be saved as a fixed ModelMeasure without picking one of
several incompatible groupings.

Status: deferred — AMBIGUOUS-PROSE

## KB 44 — Flash Transaction Cluster

Builds on KB #34 (TVM) with TVM > 50, plus "from related sources",
"completed within < 24 hours", and `escrowhrs < 12`. The "related
sources" qualifier is undefined (see KB #34 above) and cascades the
ambiguity: there's no per-cluster identity column and no clustering
algorithm specified. Even discarding the cluster-identity
requirement, TVM itself isn't encodable (KB #34 deferral), so any
predicate of shape "TVM > 50" cannot be expressed as a row-level
boolean against a fixed model.

`paymethod = 'Crypto_B'` (privacy coin) and `escrowhrs < 12` are
already directly available as row-level filters on transactions; an
agent can apply them ad-hoc without a saved entity.

Status: deferred — AMBIGUOUS-PROSE
