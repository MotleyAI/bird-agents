# W4c: cybermarket

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- A KB about "vendor network centrality" wants a per-vendor count
  of distinct markets, but `vendregistry` is unique per (vendor,
  market) — there's no cross-market vendor identity column in the
  schema. Encode the formula literally as the schema permits
  (treating each vendor row as single-market) and document the
  semantic shortfall in the entity's `description` field — *not*
  in the notes file. The KB ids of these "literal-fit" encodings
  are still encoded for verifier purposes.
- Cascading composites resolve once their parents do; encode the
  parent first.
