# W4c: mental

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- This DB has the only true **1:N peer relationship** in
  mini-interact: `facilities` → `encounters` (1 facility, many
  encounters; up to 6). For any KB that composes per-facility
  measures across `encounters`, `treatmentbasics`,
  `treatmentoutcomes`, `assessmentbasics`,
  `assessmentsocialanddiagnosis`, `assessmentsymptomsandrisk`,
  `facilities`, etc. — **R-MULTISTAGE is mandatory, not optional**.
  Each peer must be aggregated to the parent's grain in its own
  stage before the final composition.
- KBs that ask for cross-row correlation (e.g. "correlation between
  resource adequacy score and adherence rate, across facilities")
  use the built-in `corr` aggregation in the final stage. The
  aggregation slot accepts two-column kwargs:
  `corr(x, y) → "x:corr(other=y)"` per the colon-syntax docs.
- KBs framed in terms of quadrants or buckets ("median TAR vs
  median PSM") are typically encoded with a median aggregation and
  CASE-WHEN labels in the final stage — not NTILE.
- The `treatmentbasics.therapy_details` and
  `facilities.support_and_resources` JSON columns drive several
  enum-band KBs; encode via `json_extract` + R-CASE helper columns.

### Specific KB to (re-)encode this round

- **KB 61 — Facility Performance Quadrant (FPQ).** The KB itself
  pins the encoding fully — four quadrants based on TAR vs the
  median TAR across facilities and PSM vs the median PSM across
  facilities. Encode as a 4-stage multistage:
  1. Stage 1 — scalar `median_tar` over the per-facility TAR
     distribution (single-row).
  2. Stage 2 — scalar `median_psm` over the per-facility PSM
     distribution (single-row).
  3. Stage 3 — per-facility `(facid, tar, psm)` from the existing
     `facility_metrics` model (or its inputs), joined via degenerate
     cross-join to stages 1 and 2 to broadcast the two scalars onto
     each facility row.
  4. Stage 4 — emit `quadrant` as a CASE-WHEN over the four
     conditions verbatim from the KB:
     - `TAR ≥ median_tar AND PSM ≥ median_psm` → `'High Adherence, High Stability'`
     - `TAR ≥ median_tar AND PSM < median_psm` → `'High Adherence, Low Stability'`
     - `TAR < median_tar AND PSM ≥ median_psm` → `'Low Adherence, High Stability'`
     - else → `'Low Adherence, Low Stability'`

  Same scalar-broadcast pattern as the no-shared-parent multistage
  encoding documented in the generic skill. Stamp `meta.kb_id=61`
  on the resulting query-backed model.
