# W4c re-encoding instructions

One file per DB in this directory describes the W4c refresh for that
database. Each file is the contract between the planner and the
parallel agent assigned to that DB.

## What every agent does

Per `.claude/skills/translate-mini-interact-kb/SKILL.md` (the W4c
refresh override) and `.claude/skills/kb-to-slayer-models/SKILL.md`
(the recipes):

1. Connect to the SLayer MCP. Get the current state via
   `mcp__slayer__models_summary(datasource_name="<db>")` and
   `inspect_model(...)` on each model.
2. **Verify-then-fill.** Walk the KB jsonl. For each id with an
   existing entity-with-`meta.kb_id`, verify the encoding matches
   the KB's `definition`; fix or replace if not. For ids without
   coverage, encode under the recipes.
3. **Discard the existing notes file** at
   `slayer_models/_notes/<db>.md` and regenerate from scratch with
   only the KB ids you couldn't encode (using the canonical Status
   values from the skill).
4. Run `python scripts/export_slayer_models.py --db <db>` to refresh
   the YAML, then `python scripts/verify_kb_coverage.py --db <db>`
   until exit 0.
5. Confirm `tests/test_slayer_models_loadable.py` still passes for
   the DB.

## Hard constraints

- **Never read gold SQL.** No `mini_interact.jsonl`, no `sol_sql`
  fields, no instance task records. Encoding decisions come from the
  KB text + schema + column meanings + skill recipes.
- **Use the MCP for all model edits.** Don't open YAML files for
  editing.
- **Stamp `meta.kb_id` (or `meta.kb_ids`) on every encoded entity.**
  The verifier requires it.
- **Don't touch other DBs' files.**

## Per-DB hints

Each `<db>.md` file lists DB-specific quirks the prior W4b pass got
wrong, or non-obvious structural points the agent should know. They
are *warnings*, not encoding recipes — recipes live in the generic
skill. Verify everything against the schema before trusting any hint.
