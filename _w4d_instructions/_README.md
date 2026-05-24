# W4d splitting-pass instructions

One file per DB in this directory describes the W4d refresh — turning
multi-KB entities (entities with `meta.kb_ids` plural) into N single-KB
entities, each carrying its own `meta.kb_id`. Each file is the contract
between the planner and the parallel agent assigned to that DB.

W4d differs from W4c (the prior re-encoding pass) in that **coverage is
already complete**: the verifier (`scripts/verify_kb_coverage.py`) is
green for every DB, every KB id is accounted for. W4d just splits the
multi-KB encoding shortcuts so each entity carries exactly one id.

## What every agent does

Per `.claude/skills/translate-mini-interact-kb/SKILL.md` (the W4d
refresh override) and `.claude/skills/kb-to-slayer-models/SKILL.md`
("Splitting multi-KB entities" section + the existing recipes):

1. Read this `<db>.md` file. Each entity listed has a heuristic
   bucket label (A/B/C/D/E/?) and the matching `R-SPLIT-*` recipe
   from the generic skill.
2. Connect to the SLayer MCP and inspect each affected model via
   `mcp__slayer__models_summary` + `inspect_model` to confirm the
   current state.
3. For each multi-KB entity, apply the matching `R-SPLIT-*` recipe.
   End state: N single-KB entities each carrying its own
   `meta.kb_id`. The KB texts (excerpts in this file, full text in
   `mini-interact/<db>/<db>_kb.jsonl`) drive the per-entity name,
   host, and recipe choice (R-COL / R-CASE / R-FILTER / R-MEASURE
   / R-AGG / R-MULTISTAGE).
4. After splitting, run `python scripts/export_slayer_models.py
   --db <db>` to refresh the YAML, then BOTH gates:
   - `python scripts/verify_kb_coverage.py --db <db>` — exit 0
     (coverage unchanged).
   - `python scripts/multi_kb_audit.py --db <db>` — exit 0 (no
     multi-KB entities remain).
5. Confirm `tests/test_slayer_models_loadable.py` still passes
   for the DB.

## Hard constraints

- **Never read gold SQL.** No `mini_interact.jsonl`, no `sol_sql`
  fields, no instance task records. Encoding decisions come from the
  KB text + schema + column meanings + skill recipes.
- **Use the MCP for all model edits.** Don't open YAML files for
  editing.
- **Stamp `meta.kb_id` (singular) on every produced entity.** The
  verifier requires it, the multi-KB audit requires no `meta.kb_ids`
  plural to remain.
- **Don't disturb existing single-KB encodings on the affected
  models.** Other entities (columns/measures with `meta.kb_id`
  already singular) are load-bearing and should be left alone unless
  the split inherently rehosts them.
- **Don't touch other DBs' files.**

## Bucket labels

- **A — calc + threshold/classification.** Produce a calc Column
  with one kb_id and a CASE-WHEN sibling with the other.
- **B — value-illustration JSON blob.** One helper column per
  illustrated sub-field, each carrying the matching kb_id. The JSON
  blob column itself stays untagged (per kb_id placement rule).
- **C — illustration + calc + threshold trinity.** 3-way split:
  helper column + calc + classification.
- **D — multi-formula model.** One entity per formula.
- **E — over-grouping monster.** Aggressive split; the agent must
  decide per-kb_id where each lives.
- **? — bucket unclear.** Read the KB texts and pick. Default to A
  if it's a calc + threshold pair, B if illustration-shaped, D if
  multi-formula.

The bucket label is heuristic — the agent should read the KB
excerpts before applying the recipe, especially for `?` cases.
