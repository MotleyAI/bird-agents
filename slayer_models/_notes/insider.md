# insider — KB entries not encoded as model entities

The insider DB FK graph is a tree rooted at `compliancecase` with
`transactionrecord` (which joins to `trader`), `advancedbehavior`,
`sentimentandfundamentals`, `investigationdetails`, and
`enforcementactions` all reachable through child→parent FKs. There
are no peer-only joins to defer here. Where a KB entry crosses an
aggregation boundary (per-transaction child rows aggregated up to a
per-trader parent), the recipe is R-MULTISTAGE; those live in the
five `trader_*` query-backed models the export now contains.

The remaining KB id below requires a window function over a
`tradekind` partition that can't be inlined as a `Column.sql` and
is intentionally deferred from this pass.

## KB 73 — Peer Correlation Z-Score

Reason: the formula is

    Z = (peercorr - AVG(peercorr) OVER (PARTITION BY tradekind))
        / STDDEV_SAMP(peercorr) OVER (PARTITION BY tradekind),
        with Z = 0 when STDDEV_SAMP is 0 or NULL.

The expression is a windowed AVG/STDDEV partitioned by
`trader.tradekind`, applied to `advancedbehavior.peercorr` joined
through `transactionrecord` and `trader`. Two independent reasons
to defer:

1. **R-WINDOW / R-MULTISTAGE**: a window aggregate cannot be inlined
   inside a row-level `Column.sql` on `advancedbehavior` — it has to
   run after the per-row JOIN materialises, which means a backing
   query (multistage model). SQLite supports window functions, so
   the multistage encoding is feasible, but it crosses an aggregation
   boundary in the same way the current `trader_*` models do.
2. **STDDEV_SAMP availability**: SQLite's stock build doesn't ship
   `STDDEV_SAMP` as a window-aggregate UDF; SLayer registers the
   math UDFs needed for `log10` (KB #35) but the verifier-tested
   surface here didn't include `STDDEV_SAMP` for the SQLite backend
   in this run. Encoding KB #73 cleanly needs either a custom UDF
   registration or a manual two-pass query (compute mean/stddev per
   `tradekind`, then join back).

Status: deferred — multistage model with a window aggregate over
`tradekind`; defer to a follow-up pass that either provides a
`STDDEV_SAMP` UDF binding or a two-stage backing query that
materialises the per-`tradekind` mean + stddev separately.
