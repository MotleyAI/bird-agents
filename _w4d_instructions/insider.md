# W4d: insider

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### trader_cancel_mod_profile.trader_cancel_mod_profile  (model)

- Current `meta.kb_ids`: `[18, 47]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 18 [domain_knowledge] "High Cancellation/Modification Trader"
    def: A trader is flagged if their average cancelpct > 0.5 OR their average OMI > 1.5 across their transactions.
  - KB 47 [domain_knowledge] "Potentially Evasive Order Modifier"
    def: A trader identified as a High Cancellation/Modification Trader  AND whose transaction records show Dark Pool Usage  in more than 50% of instances.

### trader_event_speculator.trader_event_speculator  (model)

- Current `meta.kb_ids`: `[41, 46, 64]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 41 [domain_knowledge] "Suspected Event-Driven Insider"
    def: A trader who meets the criteria for Event-Driven Trader  AND for whom the Potential Insider Trading Flag  is True.
  - KB 46 [domain_knowledge] "Aggressive Event Speculator"
    def: A trader classified as an Event-Driven Trader  AND whose Trader Risk Appetite  is 'Aggressive'.
  - KB 64 [domain_knowledge] "Volatile Event Speculator"
    def: A trader identified as an Aggressive Event Speculator  AND associated with a high Sentiment Divergence Factor (e.g., SDF > 1.0).

### trader_intensity_metrics.trader_intensity_metrics  (model)

- Current `meta.kb_ids`: `[36, 37, 54, 66]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 4 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 36 [calculation_knowledge] "Aggressive Trading Intensity (ATI)"
    def: ATI = \text{DTR} \times \text{TLE} \times \text{OMI}.
  - KB 37 [calculation_knowledge] "Suspicion-Weighted Turnover (SWT)"
    def: SWT = \text{SAI} \times \text{DTR} \\ \text{where SAI is Suspicious Activity Index  and DTR is Daily Turnover Rate .}
  - KB 54 [calculation_knowledge] "Aggressive Suspicion Score (ASS)"
    def: ASS = \text{SAI} \times \text{ATI}
  - KB 66 [domain_knowledge] "High Velocity Suspicion Trader"
    def: A trader with a high Risk-Adjusted Turnover (RAT)  (e.g., > 1.0) AND a high Suspicious Activity Index (SAI)  (e.g., > 0.6).

### transactionrecord.risk_indicators  (column)

- Current `meta.kb_ids`: `[24, 25, 30]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 24 [value_illustration] "Momentum Ignition Signals"
    def: `risk_indicators.momentignit`: 'Strong' indicates patterns consistent with attempts to trigger momentum algorithms or attract other traders by creating a false sense of rapid price movement (e.g., through rapid successive trades). 'Weak' suggests such patterns are less evident or absent. This is a p…
  - KB 25 [value_illustration] "Marking the Close Patterns"
    def: `risk_indicators.markclosepat`: 'Frequent' indicates repeated trading activity near the market close, potentially intended to manipulate the closing price (e.g., to affect margin calls, NAV calculations, or settlement prices). 'Occasional' suggests such activity is infrequent or less patterned. Mark…
  - KB 30 [calculation_knowledge] "Risk-Adjusted Turnover (RAT)"
    def: RAT = \text{DTR} \times \text{TLE} \\ \text{where DTR is Daily Turnover Rate  and TLE is Trader Leverage Exposure .}
