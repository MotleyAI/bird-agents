# Cross-cutting observations (post-2026-05-31 failure analysis)

These are patterns that span multiple tasks and motivate the infra plan in the Linear issue.

## 1. SLayer encoder pre-bakes wrong answers (museum_2)

The on-the-fly encoder writes pre-computed measures into the slayer model based on KB-enumeration during ingest. For museum_2's ERF, this committed to a 4-column reading at encode time (`slayer_models_otf_livesqlbench/museum/models/museum/sensitivitydata.yaml:311-326` defines `erf = (env_sens_weight + light_sens_weight + temp_sens_weight + humid_sens_weight) / 4.0`). An agent that *trusts* the slayer model would still get the wrong answer. Upstream of any prompt change.

## 2. Trajectory truncation hides inspect-tool responses

`src/bird_interact_agents/agents/claude_sdk_otf_ainteract/agent.py:~370` (and the `claude_sdk_otf` sibling) clips each saved trajectory message to `str(msg)[:500]`. For failure forensics this hides the inspect-model / models-summary / search responses we need to verify what the agent actually saw. Cross-cuts both `claude_sdk_otf` and `claude_sdk_otf_ainteract`. Worth raising the clip (or moving inspect responses to a separate per-task log file).

## 3. Single-bit grader feedback is the largest aggravator (museum side)

museum_2 and museum_5 both had the correct hypothesis on the table at one point and were unable to choose between candidates given `"ex_base returned 0 but expected 1. Please try again."`. A row-count/columns/sample-diff would have flipped both. Independent of any model improvement.

## 4. User-sim under-disclosure is the largest aggravator (households side)

Three of four households failures involve the user-sim being **too literal** (households_10 — answered the canonical-label question correctly but never volunteered synonym coverage), **too vague** (households_12 — "use the service criteria" instead of naming `domestichelp` + `total_vehicles>1`), or **too slow** (households_15 — leaked formula only on 5th of 5 asks).

When the user-sim holds a `critical_ambiguity.sql_snippet` with multiple AND clauses and the agent asks an open-ended "what counts as X?", the sim paraphrases instead of disclosing each clause. Reads like a human deliberately holding back — appropriate for benchmark fidelity, but masks metadata-completeness issues that would otherwise be visible.

## 5. Slayer search rendering ≠ structured data

Both motley-slayer 0.6.10 and 0.7.0 have identical `render_column_text` (`slayer/search/render.py:166-192`). It emits only the truncated `Sample values:` string (~210-330 chars per column, top-20ish high-frequency values), NOT the structured `column.sampled_values: List[str]` field (50 items, full noisy list).

For households_10 specifically, the truncated string DOES contain `Y`, `have`, `brick house`, `apt`, `2014+`, `10-13`, `05-09` (high-frequency). But `after 2014` and `2014 and newer` (lower-frequency at 11-12 rows each) are structured-only — they don't survive truncation.

If we want full sampled-value visibility for agents, that's a slayer upstream change — or wrappers around `search`/`inspect_model` that expand the structured field. Worth a slayer-side issue.

## 6. KB hedging vs strict reading

KB 6 ends with "etc.", KB 10 says "like", KB 7 enumerates only 3 specific values (no hedge). The agent and the audit can both legitimately read these differently:

- Strict reading (canonical literals only) — KB 7 invites this; KB 6/10 less so
- Extensible reading (canonical + observed synonyms) — KB 6/10 invite this; KB 7 doesn't

The audit picks extensible; the agent (lacking a prompt rule to extend) picks strict. Both internally consistent. Phase1 fails because they're not the same extension.

The prompt-level fix (rule-3 symmetric companion) directly addresses this: tell the agent to extend the IN-set when sampled values show variants of KB-named literals.

## 7. The audit-gold-sql contract is not enforced

households_10's audit recorded `clause_kind="manual_data_align_reaudit"`, `source="manual_dev1478"`, `justified_by: []`. Per `.claude/skills/_shared/audit-gold-sql.contract.md`, such clauses should be DEFERRED, not encoded. The auditor self-flagged the change as unjustified but still committed it.

This is an audit-quality regression worth fixing upstream: any non-empty `justified_by: []` should bubble to the audit-time validator and either defer the clause or fail the audit.

## 8. Quarantine-vs-score multidimensionality

households_15 was stamped `audit_status="unrecoverable"` because no minimal rewrite makes the gold KB-derivable. Counting it in the headline pass rate misrepresents the agent (no agent could win). But it also isn't useful to filter it out wholesale — it tells us something about the gold-audit pipeline's ability to handle "the original gold is fundamentally underspecified".

Per the agreed annotation schema, quarantine isn't a separate flag — it's derived from the multi-dimensional scoring (`original_gold_passes`, `audited_gold_passes`, `matches_vagueness`). The reporting layer can choose to exclude or include based on which dimension matters for a given audience.
