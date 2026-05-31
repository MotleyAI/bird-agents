# Households (mini-interact / a-interact) failure analysis

**Run ID:** `20260531t1008-claudes-slayer-890419`
**Date:** 2026-05-31
**Branch / commit:** `main @ af58457` (post-README defaults bake)
**Compared against prior:** `20260529t2103-claudes-slayer-ca9536` (11/15 today vs 11/15 prior; flip: `_14` now passes, `_15` now fails)

## Configuration

| field | value |
| -- | -- |
| framework | `claude_sdk_otf_ainteract` |
| dataset / mode | `mini_interact` / `a-interact` |
| agent model | `anthropic/claude-opus-4-7` |
| user-sim model | `anthropic/claude-sonnet-4-6` |
| reasoning effort | `high` |
| slayer setup | `on-the-fly` |
| audited gold | on (`use_audited_gold_sql=True`) |
| patience | 500 |
| instance ids | all 15 households (`_1.._19` minus skipped) |
| cluster | 1 worker × 1 actor, `e2-standard-4` |

## Headline result

| metric | value |
| -- | -- |
| phase1 pass rate (raw) | **11 / 15 (73%)** |
| phase1 pass rate (after accountability adjustment) | **14 / 15 (93%)** |
| run cost | **$57.19** (agent $55.94, user-sim $1.25) |
| longest task | households_12 (1238 s, $12.05) |
| shortest pass | households_16 (33 s, $0.41) |

Adjustment rationale: of the 4 failures, **1 is a genuine agent miss**, **1 is a gold/audit bug that violates the audit-gold-sql contract**, **1 is a gold-spec gap compounded by user-sim under-disclosure**, and **1 is a known-`unrecoverable` audit instance that should be quarantined from the eval set**.

## Per-task verdicts

| inst | dur | cost | category | root cause |
| -- | --: | --: | -- | -- |
| households_1 | 75 s | $1.16 | PASS (audited only) | — |
| households_2 | 908 s | $7.78 | FAIL — **agent miss** | User-sim at idx 100 told the agent "strict exact matches, no normalization"; slayer model exposed `Riacho  Fundo I` (two spaces) as a distinct sampled_value; agent's idx 125 raw-locregion query returned the gold answer `Riacho  Fundo I \| 0.500`; agent then ran a normalized variant at idx 127, saw `riacho fundo i \| 0.008`, and **submitted the normalized one anyway**. Final assistant text at idx 205 explicitly compares the two and picks the wrong one. |
| households_3 | 537 s | $4.32 | PASS (audited + original) | — |
| households_4 | 109 s | $0.93 | PASS (audited only) | — |
| households_5 | 265 s | $1.60 | PASS (audited only) | — |
| households_6 | 61 s | $0.75 | PASS (audited only) | — |
| households_7 | 111 s | $1.20 | PASS (audited only) | — |
| households_10 | 926 s | $11.68 | FAIL — **gold/audit bug + user-sim under-disclosure** | The DEV-1478 re-audit expands KB labels with synonym sets (`'y'`, `'have'`, `'apt'`, `'brick house'`, `'2014+'`, `'after 2014'`, `'10-13'`) marked `clause_kind="manual_data_align_reaudit"`, `justified_by: []` — i.e. the auditor itself admits there is no KB / column-meaning / schema justification. KBs 6, 7, 10 list only canonical labels. Agent's submission is KB-faithful. User-sim refused to disclose synonyms (idx 35 listed canonicals only) or row-counts (idx 105 said *"I don't have access to expected row counts"*). |
| households_11 | 236 s | $1.74 | PASS (audited only) | — |
| households_12 | 1238 s | $12.05 | FAIL — **gold-spec gap + user-sim failure** | "Independent households" mask requires `domestichelp='no domestic workers' AND total_vehicles>1` — present nowhere in KBs (KB 42 only mentions `socsupport='No'` + "high income"), column meanings, or schema. Lives only inside `critical_ambiguity.sql_snippet`. User-sim, holding that snippet, told the agent at idx 34 *"don't have a specific cutoff for high income — just use all households that qualify as independent based on the service criteria"* — vague, no clause naming. Agent correctly normalised `Dwelling_Class` aliases (KB-encoded in slayer's pre-baked `dwelling_type_clean`) — that part isn't the failure. |
| households_14 | 291 s | $2.34 | PASS (audited only) | — |
| households_15 | 618 s | $6.53 | FAIL — **gold/audit bug, known `unrecoverable`** | The audit itself stamps `audit_status="unrecoverable"`. The Socioeconomic Index coefficients (0.4 / 0.4 / 0.2), threshold (0.7), ratio cap (1.5), and income 1-6 ordinal mapping live only in the labeled `sql_snippet`. KB 28 ("Economically Stable Household") describes the concept narratively but isn't even in this task's `external_knowledge=[9,11,24,25]`. Auditor kept the unjustified scaffolding intentionally because removing it would over-rewrite. User-sim eventually leaked the formula on idx 136, after 4 prior asks failed — agent didn't pivot in time. |
| households_16 | 33 s | $0.41 | PASS (audited + original) | — |
| households_17 | 181 s | $1.37 | PASS (audited only) | — |
| households_19 | 417 s | $3.34 | PASS (audited + original) | — |

## Was the audited gold derivable from data the agent had?

The audit pipeline (`audit-gold-sql` skill, DEV-1478) re-authors original golds so each clause is justified by `<db>_kb.jsonl` + `<db>_column_meaning_base.json` + the task's labeled ambiguity sources. Per-task:

### households_2 — yes, fully derivable. Agent ignored the evidence.

- **KB / metadata:** KB 9 (high-risk family) + KB 10 (mobile household) + KB 14 (vehicle count) cover the conjunctive predicate. The slayer model's `households.yaml` `locregion.sampled_values` list contains `Riacho  Fundo I` (two spaces) as a *distinct* sampled value next to `Riacho Fundo I` (one space) — direct evidence that normalising would collapse meaningful groups.
- **User-sim:** unambiguous and correct. idx 19 gave the full predicate; idx 100 said *"Please use strict exact matches without normalizing case or trimming for the filtering conditions"*.
- **What the agent did:** At idx 125 it ran a raw-locregion grouping and saw the gold output `Riacho  Fundo I \| 0.500`. At idx 127 it then ran a normalized variant. At idx 130 it submitted the normalized one. Final text (idx 205) explicitly weighs both candidates and picks wrong.
- **Why:** the agent's data-hygiene prior ("dirty data should be normalised") overrode (a) an explicit user-sim instruction and (b) the sampled-values caveat. This is the only failure in this run where the agent had everything it needed and still submitted wrong.

### households_10 — no, the audit itself is contract-violating.

- **KB / metadata:** KB 6 lists only `'Brickwork house'`, `'Apartment'`, `'Condominium'`; KB 7 lists only `'avail'`, `'available'`, `'yes'`; KB 10 lists only canonical Newest_Year buckets. None of `'y'`, `'have'`, `'apt'`, `'brick house'`, `'2014+'`, `'after 2014'`, `'2010-2013'`, `'10-13'` appear anywhere in the KB or column-meaning corpus.
- **Audit provenance:** the change is marked `clause_kind="manual_data_align_reaudit"` with `justified_by: []` — the auditor explicitly admits no source justification. Per the audit-gold-sql skill's own contract (`.claude/skills/_shared/audit-gold-sql.contract.md`) such clauses should be DEFERRED, not encoded. This is a regression in audit quality on this instance.
- **User-sim:** idx 35 stated the canonical Newest_Year list verbatim; never volunteered synonyms. idx 105 declined to disclose row counts when the agent specifically asked for debugging help.
- **What the agent did:** submitted the KB-faithful canonical literal set. Returned 153 housenums vs gold's 233.
- **Net:** the agent did what the supplied corpus told it to do. The gold drifted from the corpus.

### households_12 — no, gold-spec gap; user-sim compounded.

- **KB / metadata:** KB 42 ("Economically Independent Household") says "high Income Classification + social support status of 'No'" — neither `domestichelp='no domestic workers'` nor `total_vehicles>1` is mentioned anywhere in KBs, column descriptions, or the slayer model. Both constraints live only inside `critical_ambiguity.sql_snippet`.
- **Slayer pre-baking helped on the dwelling part:** `properties.yaml` exposes `dwelling_type_clean` with sampled values `'Brickwork house', 'BRICKWORK HOUSE', 'brick house', 'BRICK HOUSE', "brickwork\thouse", 'Unfinished brickwork', 'Shack'`. Agent correctly handled this — alias folding is not the failure.
- **User-sim:** idx 20 leaked the prosperity-score formula correctly; idx 106/147/233 confirmed the dwelling-class alias merge. But on the critical question at idx 34 — "what exactly counts as independent?" — sim said *"just use all households that qualify as independent based on the service criteria"*, naming neither `domestichelp` nor `total_vehicles>1`.
- **What the agent did:** built `socsupport='No' AND dwelling_class<>'other'`. Predicted `('brickwork house', 2106)` vs gold `('brickwork house', 1373)` — same dwelling class (alias merge worked), but 53% over-count because single-vehicle and `domestichelp='yes domestic workers'` households leaked in.
- **Net:** primary fault is gold-spec gap (KB 42 incomplete); user-sim's vague non-answer at idx 34 made it irrecoverable.

### households_15 — no, audit-stamped unrecoverable.

- **KB / metadata:** original `external_knowledge=[9, 11, 24, 25]`. KB 28 (Economically Stable Household) is conceptually relevant but *not in this task's KB list*. KBs 9, 11, 24, 25 cover Social Support / Household Density / Crowded Household — irrelevant. The SEI coefficients 0.4/0.4/0.2, threshold 0.7, ratio cap 1.5, and income 1-6 ordinal mapping exist only in the gold `sql_snippet`.
- **Auditor's own assessment** (verbatim): *"Notes: (a) labeled 'financially secure' references income_score/tenure_score variables; we use the same 1-6 income mapping as in households_11 as scaffolding — not in any source but is required scaffolding."* The audit kept the unjustified clauses to preserve the labeled `sql_snippet` rather than rewrite to something derivable.
- **User-sim:** finally leaked the SEI formula on idx 136 (turn 5 of 5 asks) — too late. Earlier asks elicited the OR-of-(domestichelp, socsupport) reading the agent took.
- **What the agent did:** invented its own tenure filter (`OWNED` only) and an Income_Bracket ordinal that doesn't match any KB-mentioned label; never built the weighted SEI.
- **Net:** the instance is structurally unrecoverable — the labeled `sql_snippet` is its own source of truth. No agent can reproduce these specific numerical constants from the supplied metadata.

## Cross-cutting observations

### 1. User-sim under-disclosure is now the dominant failure aggravator

Three of four failures involve the user-sim either being **too literal** (households_10 — answered the canonical-label question correctly but never volunteered the synonym coverage the gold actually needed), **too vague** (households_12 — "use the service criteria" instead of naming `domestichelp` + `total_vehicles>1`), or **too slow** (households_15 — leaked the formula only on the 5th of 5 asks). This is a separate failure mode from the museum-side single-bit grader feedback: here the agent has a channel to ask, and the channel is leaking signal at far below its theoretical capacity.

A targeted change: when the user-sim is asked an open-ended "what exactly counts as X" question and X corresponds to a `critical_ambiguity` it holds, it should disclose **every clause** of `sql_snippet` rather than paraphrase. The current behaviour reads like a human deliberately holding back — appropriate for benchmark fidelity, but it's masking metadata-completeness issues we'd otherwise see clearly.

### 2. The audit-gold-sql contract was violated on households_10

`audit_status="edited"` is supposed to mean "every changed clause is justified by KB / column_meaning / labeled ambiguity". households_10's audit explicitly records `clause_kind="manual_data_align_reaudit"`, `justified_by: []` — i.e. the auditor self-flagged that the expanded synonym sets have no source justification, then encoded them anyway. Per the skill's contract those clauses should have been DEFERRED. This is an audit-quality regression worth fixing upstream: any non-empty `justified_by: []` should bubble to the audit-time validator.

### 3. Slayer's sampled_values caveats are *working* (and the agent ignored them)

households_2 is direct evidence that the post-S1-S5 sampled-values caveat work has shipped correctly: `Riacho  Fundo I` (two spaces) was visible in the slayer model alongside the clean variant. The agent didn't act on this signal *and* didn't act on the user-sim's explicit "don't normalise" instruction. This is purely an agent miss — the metadata side did its job.

### 4. households_15 should be quarantined from the eval set

`audit_status="unrecoverable"` is the auditor's signal that no minimal rewrite makes the gold KB-derivable. Counting these instances in the headline pass rate misrepresents the agent. Suggest filtering them out of `--instance-ids` for measurement runs, or partitioning the `eval.json` into `pass / fail / unrecoverable_gold` buckets.

### 5. Same trajectory-truncation issue as museum

`src/bird_interact_agents/agents/claude_sdk_otf_ainteract/agent.py:~370` clips each saved message to `str(msg)[:500]`. For the long households runs (households_12 at 1238 s, households_10 at 926 s) this is hiding most of the agent's inspect/search responses and the entire user-sim disclosure pattern. Cross-cuts both `claude_sdk_otf` and `claude_sdk_otf_ainteract`.

## Cost breakdown

Sum across the 15 instances:

| component | USD |
| -- | --: |
| Opus agent | **$55.94** |
| Sonnet user-sim | $1.25 |
| **total** | **$57.19** |

Per-task: households_12 ($12.05), households_10 ($11.68), households_2 ($7.78), households_15 ($6.53), households_3 ($4.32), households_19 ($3.34), households_14 ($2.34), households_11 ($1.74), households_5 ($1.60), households_17 ($1.37), households_7 ($1.20), households_1 ($1.16), households_4 ($0.93), households_6 ($0.75), households_16 ($0.41).

Of the $57.19, **$38.04 (66%)** went to the 4 failing tasks. Of *that*, **$30.26 (53% of run total)** is on three failures where the agent is not at fault (gold/audit bug, gold-spec gap, unrecoverable). Same lesson as museum: the failure-cost concentration is dominated by tasks the model can't win on the current spec.

## Suggested follow-ups

1. **Quarantine households_15** (audit-stamped `unrecoverable`) from default `--instance-ids` for measurement runs; partition `eval.json` summary by audit_status.
2. **Re-audit households_10** under the audit-gold-sql contract. The current re-audit's `manual_data_align_reaudit` + `justified_by: []` clauses should be DEFERRED, or the audit should add a per-column KB entry that enumerates the synonym set with a justification anchor.
3. **Backfill KB 42** with the full "independent household" predicate (`domestichelp='no domestic workers' AND total_vehicles > 1`), or remove the masked clauses from `critical_ambiguity` so the user-sim has nothing to hold back.
4. **Tighten the user-sim disclosure rule** for `critical_ambiguity.sql_snippet`-anchored questions: when asked an open-ended definitional question and the held `sql_snippet` has multiple AND clauses, disclose each one (not a paraphrase). Lives in the user-sim prompt / sim policy.
5. **Add an agent-side prompt nudge** for the households_2 class: "If the user explicitly says 'use strict matches / do not normalize', that overrides your dirty-data priors and the slayer model's normalised columns. Quote the user instruction in your final reasoning."
6. **Raise the trajectory truncation cap** (same as museum). `agent.py:~370`.

## Provenance

- Per-task attempt artifacts: `/home/james/Dropbox/SLayer/bird-agents/results/cloud/20260531t1008-claudes-slayer-890419/rows/<instance_id>/attempt-1.json`
- Manifest: `/home/james/Dropbox/SLayer/bird-agents/results/cloud/20260531t1008-claudes-slayer-890419/manifest.json`
- Eval summary: `…/eval.json` (`p1=11/15 (0.733)`)
- Task data: `/home/james/Dropbox/SLayer/mini-interact/mini_interact.jsonl`
- Audited gold: `/home/james/Dropbox/SLayer/bird-agents/audited_gold/households/households_audited.jsonl`
- Households metadata: `/home/james/Dropbox/SLayer/mini-interact/households/{households_kb.jsonl, households_column_meaning_base.json, households_schema.txt}`
- Slayer model: `/home/james/Dropbox/SLayer/bird-agents/slayer_models_otf/households/models/households/` (`households.yaml`, `transportation_assets.yaml`, `properties.yaml` — sampled_values blocks especially)
- Audit contract: `.claude/skills/_shared/audit-gold-sql.contract.md`
