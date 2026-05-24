# W4c: labor_certification_applications

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- Some KBs reference quantities the schema doesn't carry directly —
  e.g. a "visa-dependent industry" predicate that wants
  visa-applications-as-fraction-of-workforce, but no workforce
  column exists. The schema does carry application counts per
  industry, which is a usable proxy denominator. Encode the proxy
  and document the substitution in the entity's `description`
  (not in the notes); the KB id is still considered encoded.
- For KBs that genuinely cannot be expressed against the schema
  (no proxy available), defer as `SCHEMA-GAP` after verifying
  against the schema and column meanings (including any JSON
  subfields).
