# cross_db — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/cross_db/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — …` headers to
distinguish "skipped on purpose" from "missed".

## KB 78 — High Vendor Risk Concentration

Reason: source-data inconsistency. The KB title is "High Vendor Risk
Concentration", but the `definition` field is verbatim
`"A data flow where CURRENT_DATE - RemedDue > 0"` — that's identical
to KB #76 (Slow Remediation Timeline) and unrelated to vendor risk.
The expected definition (something like `VRA > 3 AND VRI < 1` or a
threshold over `vrc`) is not provided. Encoding the literal definition
would just duplicate KB #76's `is_slow_remediation` and tag it with
the wrong KB id.

Status: deferred — bad source data. Agent has `vendormanagement.vrc`,
`vendormanagement.vra`, `vendormanagement.vri`, and
`vendormanagement.vendor_risk_tier` available to filter on at query
time once the intended threshold is clarified.
