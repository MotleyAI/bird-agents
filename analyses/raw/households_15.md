# households_15 — failure trace + metadata audit

Run: `20260531t1008-claudes-slayer-890419`. Duration 618s. submission_status=wrong_result.

audit_status: **`unrecoverable`** — auditor explicitly stamped this.

## User query + masked terms

> "Find all highly supported homes that are also financially secure, listing their unique house codes."

Three flagged ambiguities (mini_interact.jsonl, instance `households_15`):
- `"highly supported homes"` (knowledge_linking, mask=true) → gold snippet: `((CASE WHEN domestichelp='none' THEN 0 ELSE 3 END)+(CASE WHEN socsupport='No' THEN 0 ELSE 4 END))>2`
- `"financially secure"` (knowledge_linking, mask=true) → gold snippet: weighted index `0.4*income_score + 0.4*(Expend_Coeff/income_score) + 0.2*tenure_score > 0.7 AND Expend_Coeff/income_score < 1.5`
- `"unique house codes"` (intent, mask=false) → `SELECT DISTINCT housenum`

External KBs declared: `[9, 11, 24, 25]` — Social Support Status, Household Density, Supported Household, Crowded Household. **None define "financially secure"**, and KB 11/25 (density) are entirely unused by the audited gold.

## Auditor's view — why unrecoverable

`reasoning_summary`:
> *"Gold's predicate uses an unjustified density>4 'AtRisk' bucketing and ignores the labeled 'highly supported' + 'financially secure' definitions entirely; output is wrong (returns one locregion, user asked for DISTINCT housenum). Audited SQL uses the two labeled-ambiguity sql_snippets verbatim and projects DISTINCT housenum. Notes: (a) labeled 'financially secure' references income_score/tenure_score variables; we use the same 1-6 income mapping as in households_11 as scaffolding, (b) actual result is 0 rows…"*

Two `changes`:
1. Whole outer SELECT replaced because original gold returns `locregion` (regional ratio) rather than `housenum` — output-shape bug.
2. The 1-6 `income_score` mapping kept as "scaffolding" but explicitly flagged as **not in any source** — same caveat as households_11/12. The "financially secure" coefficients (0.4 / 0.4 / 0.2), threshold 0.7, ratio cap 1.5, tenure 3/1/0 mapping, and "highly supported" weights 3/4 with threshold >2 are pulled directly from the labeled_ambiguity `sql_snippet` — they exist nowhere in the KB.

## ask_user

5 calls (turns 36, 39, 42, 100, 135). Sim never gave numeric thresholds for the support score. Only on the last ask (turn 136) leaked the exact financially-secure formula:
> *"0.4 * income_score + 0.4 * (Expend_Coeff / income_score) + 0.2 * tenure_score, threshold > 0.7, ratio < 1.5, income_score > 0"*

For "highly supported" sim gave the OR-rule (`domestic help present OR socsupport='Yes'`) — which is what the agent encoded — not the gold's weighted-score `(3 or 0) + (4 or 0) > 2`, which happens to be ≡ OR-of-the-two only in binary sense. Gold also uses `LOWER(COALESCE(domestichelp,''))='none'` rather than the agent's `LIKE 'yes%'`.

## Submitted vs audited gold

Agent: `LEFT JOIN service_types`, filters `Tenure_Type IN ('owned','own')` AND `Income_Bracket` ordinal ≥5 AND `Expend_Coeff/income ≤12`, support=OR-of-(domestichelp LIKE 'yes%', socsupport='yes'), `GROUP BY housenum` → 122 rows.

Audited gold: weighted SEI > 0.7 AND ratio < 1.5 AND income_score > 0 AND `((dh!='none')?3:0) + ((soc!='No')?4:0) > 2` → 0 rows.

Agent never built the weighted index; invented its own tenure filter (`OWNED` only) and an Income_Bracket ordinal mapping with R$ bands that don't match any KB-mentioned label.

## Decision-point lock-in

Agent fixated on KB 24 (binary socsupport='Yes') interpretation early, then on the last attempt swapped to `domestichelp LIKE 'yes%' OR socsupport='yes'` per sim's OR-confirmation. For "financially secure" never adopted the user-sim's leaked SEI formula at turn 136 — submitted SQL still uses an income-only conjunction.

## Metadata sufficiency

**Resolving information NOT present.** KB 28 ("Economically Stable Household") defines the concept narratively — *"high Socioeconomic Index AND low Expenditure Ratio"* — but the **coefficients (0.4/0.4/0.2), threshold (0.7), ratio cap (1.5), income 1-6 ordinal, and tenure 3/1/0 mapping exist only in the gold `sql_snippet`**. The KB chain for "Socioeconomic Index" (KB 12) and "Expenditure Ratio" (KB 19) is not in `external_knowledge=[9,11,24,25]`. KB 16 ("Service Support Score") says *"weighted score combining domestic help availability and social assistance participation"* with no weights.

Auditor explicitly flags income mapping as "not in any source but is required scaffolding." Same pattern the project memory `households_gold_reaudit` calls out as deliberately noisy.

## Audit isn't a real repair

It's a **partial rewrite**:
- Kept gold's literal `sql_snippet` for both masked terms (verbatim, including unsourced coefficients)
- Fixed output shape (DISTINCT housenum vs locregion)
- Replaced unrelated density>4 AtRisk machinery

Audit doesn't repair derivability — it just makes gold match labeled `sql_snippet`, which itself isn't derivable from KB. Audited gold returns 0 rows on live DB ("few households simultaneously match both predicates").

## Verdict

**Gold/audit bug — known unrecoverable.** Auditor itself stamped `audit_status="unrecoverable"` and acknowledged the kept scaffolding has no KB justification. No reasoning agent can reproduce the exact coefficients 0.4/0.4/0.2/0.7/1.5 or the 1-6 income mapping from KBs 9/11/24/25 — and KB 28 (the actual relevant concept) isn't even in `external_knowledge`. User-sim did leak the formula on turn 136, but only after 4 failed asks, and the agent didn't pivot.

Remediation: remove `households_15` from default eval set (or quarantine into a separate "known-unrecoverable" bucket). The labeled `sql_snippet` is the only source-of-truth for the SEI weights/thresholds and is not reachable from any KB or column metadata. Same class as `households_11/12` per the audit's own cross-reference.

Provenance: `results/cloud/20260531t1008-claudes-slayer-890419/rows/households_15/attempt-1.json`; audited gold (`audit_status: unrecoverable`); `households_kb.jsonl` (KBs 9/11/16/21/24/25/28/42).
