# museum_4 — failure trace + metadata audit

Run: `20260531t1013-claudes-slayer-48eb0f`. Duration 608s. submission_status=wrong_result.

## Task

Per-dynasty budget report. Gold returns 7 columns (`dynasty, artifact_count, total_cpi, artifacts_with_adequate_budget, artifacts_with_insufficient_budget, CBE, budget_status`). Agent returned 6 (dropped `artifact_count`).

CBE in gold: `SUM(CPI × adequate_budget/total_records) / artifact_count` — weighted average; `BudgetRatio_i = adequate_records/total_records` derived from `conservationandmaintenance.budgetallocstatus` (Adequate / Insufficient / Review Required categorical enum).

CBE in agent: `SUM(cpi) / COUNT²` — uniform 1/N reading of "BudgetRatio".

## Was CBE a KB entry?

Yes. `museum_kb.jsonl` id=9:
```text
"CBE = \frac{\sum_{i \in artifacts} (CPI_i \times BudgetRatio_i)}{|artifacts|}, where BudgetRatio is the proportion of total conservation budget allocated to each artifact"
```

Agent retrieved at `trajectory[8 → 9]` via `get_knowledge_definition("Conservation Budget Efficiency (CBE)")`. Also pulled KB 17 (Crisis), KB 0 (CPI), KB 21 (ConserveStatus enum), KB 22 (HistSignRating). KB coverage was complete.

## Agent's reasoning

ThinkingBlocks encrypted (`thinking=''`), but `create_model` description at `trajectory[33]`:
```text
"CBE = sum(CPI*1/N)/N = sum(CPI)/N^2 assuming uniform BudgetRatio"
```

Final TextBlock at `trajectory[96]`:
> "the KB definition is ambiguous about what `BudgetRatio` means when no explicit per-artifact budget column exists … My final encoded model uses the most literal reading of KB 9: BudgetRatio_i = 1/N (uniform distribution of the total budget across artifacts), so CBE = SUM(CPI)/N²"

Agent read KB correctly; resolved ambiguity by assuming uniform allocation. Gold instead reads `BudgetRatio_i = adequate/total` from `conservationandmaintenance.budgetallocstatus`. Both KB 9 and the schema are silent on this.

## artifact_count column

Never mentioned by the agent. Prompt: "show the dynasty name, their total conservation priority index, the count of artifacts with adequate funding, the count with insufficient funding, the calculated CBE value, and a budget status" — 5 named outputs + dynasty = 6. Gold reads the same prompt and adds a 7th unstated column. Agent's parse matched the prompt literally; missing column is a gold-side under-specification.

## Decision-point lock-in

`trajectory[33]` (initial `create_model`). Description already contains `"CBE = sum(CPI*1/N)/N = sum(CPI)/N^2"`. Subsequent `edit_model` calls (49, ~64, ~76, ~90) only changed scaling/rebasing (cbe drifted 0.0107 → 10.14 → 0.0108 → 0.487) — the structural choice `BudgetRatio = 1/N` was never revisited. Even after 4 submit_query rejections with generic Phase-1 failure, the agent never reached for `adequate_budget / total_records`.

## Metadata audit

`museum_column_meaning_base.json:128-147` — `ConservationAndMaintenance` column block (20 columns). Every budget-adjacent description is purely categorical:

- L141 `BudgetAllocStatus`: *"A VARCHAR(50) describing budget allocation status (possible values: 'Review Required', 'Insufficient', 'Adequate')."*
- L142 `MaintBudgetStatus`: *"...'Limited', 'Depleted', 'Available'..."*

Full scan: NO column meaning uses "ratio", "proportion", "share", "allocation share", or "adequate / total".

Full scan of `museum_kb.jsonl` (56 entries) for "Ratio", "Allocation", "Budget", "proportion", "share", "adequate":
- KB 9 (the under-specified CBE itself)
- KB 17 (Crisis) — uses `BudgetAllocStatus='Insufficient'` as categorical threshold, not ratio
- KB 36 (CRAE) — references CBE but doesn't redefine BudgetRatio
- Other matches in KB 22, 28 are unrelated

No value-illustration KB sibling pinning `BudgetAllocStatus` semantics.

Agent's slayer exploration: `models_summary` at turn 16 (exposed BudgetAllocStatus description from L141); `query(conservationandmaintenance, n_cm, n_distinct_art)` → 951/951; `query(conservationandmaintenance, dim=budgetallocstatus, count)` → Adequate 314 / Insufficient 326 / Review Required 311. Three `search(...)` calls returned only KBs 9 and 17 (already known).

## Verdict

**KB coverage under-specified.** KB 9's `BudgetRatio` is undefined for a schema with no per-artifact budget *amount* column. Gold silently expects `adequate/total` derived from `BudgetAllocStatus`. Missing `artifact_count` is a secondary prompt-vs-gold gap. Agent reasoning sound given the inputs.

Remediation target: KB (anchor BudgetRatio to BudgetAllocStatus with an explicit value-illustration sibling KB).

Provenance: trajectory items 9, 33, 96; `museum_kb.jsonl#9,17`; `museum_schema.txt:122` (`budgetallocstatus`, no amount column).
