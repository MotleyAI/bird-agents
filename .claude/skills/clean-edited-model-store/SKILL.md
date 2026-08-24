---
name: clean-edited-model-store
description: >-
  Clean up ONE saved edited-model store (runs/<benchmark>/<db>/<iid>/edited_models.tar.gz)
  so the reused (apply) store contains ONLY legitimate reusable definitions — KB-derived
  entities and reusable concepts — not one-off inline query scaffolding and not broken/unused
  leftovers. Use when asked to clean/prune/hoist a saved slayer edited-model store, or to make
  a store reach the "passing state" reported by the store_kb_checker. Mutates gitignored data
  in the MAIN checkout; always backs up first.
---

# Clean up a saved edited-model store (DEV-1671)

Post-hoc, behaviour-preserving cleanup of a single saved store. Everything goes through a live
SLayer instance (`YAMLStorage` + the MCP `query`/`edit_model`/`delete_model` tools) — **never
hand-edit YAML and never hand-roll a SQL/formula parser.**

## HARD CONSTRAINT — do NOT cheat
Do NOT encapsulate the winning query as a model. Baking *the answer* (the specific
aggregation / filter / threshold that produces the graded result) into a stored model or
measure is forbidden. What MAY be stored: **KB definitions** (`[kb=N]`-tagged, tracing to a
knowledge-base entry) and **reusable concepts** (`[concept]`-tagged — e.g. "roomy", "well-off",
"apartment" — a general building block that is not itself the answer). The aggregation/threshold
that turns those building blocks into the answer stays in the query.

## The "passing state" the checker nudges toward
1. every surviving agent-added entity is USED by the final query;
2. the reformulated query returns the IDENTICAL result to the original winning query;
3. relevant KB items are encoded as entities (advisory — `external_knowledge` is unreliable);
4. entities refer to each other (clean DAG — KB defs referenced by name, not inlined).

## Inputs (resolve first)
- `benchmark`, `instance_id`; `db = "_".join(iid.split("_")[:-1]) + "_large"` (livesqlbench-large).
- **Store archive:** `edited_models.run_edited_models_archive(benchmark=…, selected_database=db, instance_id=iid)`.
- **Winning query:** `submitted_query` of the latest successful attempt under
  `results/<benchmark>/*/rows/<iid>/attempt-*.json` (a saved store that already reformulated its
  query carries `_winning_query.json` inside the archive — prefer it).
- **Baseline (advisory):** OTF cache `paths.slayer_otf_cache_root(benchmark=…)/db`, guarded by
  `cache_fp` == the store's `_edited_models_meta.json`.
- **KB rows:** `_kb_rows.json` inside the archive (carries `children_knowledge`).
- Local postgres must be up (the store's datasource connection_string points at it) for the
  identical-result check.

## Recipe (per store) — REFERENCE-FIRST, DELETE-LAST

**Guiding principle (do NOT get this backwards):** the goal is a store whose entities are a good,
reusable set of KB/concept definitions, and whose winning query REFERENCES those definitions
instead of doing the work inline. So you **build the good query first, add/reference the
encodings it needs, and only then delete what is genuinely left over.** Never delete a
kb/concept encoding just because the *recorded* (often inline-heavy) query didn't use it — that
query may be bad; a cleaner one would use it.

0. **Back up first (non-negotiable):** copy the archive to
   `runs/_edited_models_backups/<date>_<label>/<benchmark>/<db>/<iid>/edited_models.tar.gz`.
1. **Materialise** (untar) and get the baseline result of the recorded winning query:
   `store_cleanup.run_store_query(scratch, winning_query)`. Read the checker findings
   (`store_kb_checker.check_store(...)`) and the KB rows (`_kb_rows.json`).
2. **ADD missing encodings** (do this BEFORE deleting anything). Using `create_model` /
   `edit_model` on the store's SLayer MCP server, encode the KB items / complex concepts the
   query needs but the store lacks — driven by:
   - `INLINE_QUERY_WORK`: the query computes a KB item / reusable concept inline (JSON extraction,
     CASE, multi-column arithmetic, an inline model/column). Encode it as a `[kb=N]` column/measure
     (KB-derived) or a `[concept]`-tagged column (reusable, non-KB), then reference it.
   - `EXPECTED_KB_NOT_MATERIALIZED` / `DEFERRED_RELEVANT_KB`: if the KB is genuinely relevant to
     this question, encode it (see the `kb-to-slayer-models` / `translate-mini-interact-kb`
     skills for KB→SLayer encoding recipes). Waive if the anchor is mislabeled for this task.
   - `NON_KB_ENTITY`: a used derived column without provenance — if it's a real concept, tag it
     `[concept]`; if it *is* the answer aggregation/threshold, leave it inline in the query.
3. **Build the NICE query** that REFERENCES the encodings (existing + newly added) by name instead
   of inlining, and returns the identical result. This is the query the cleaned store supports.
   - `INLINED_KB_DEF`: rewrite the consumer to reference the stored KB column by name.
4. **Identical-result gate (behaviour preservation):** run the nice query and assert
   `store_cleanup.results_identical(baseline_result, nice_result)` (positional value tuples,
   headers ignored, Decimal-precision; for a top-N with tied ORDER BY add a deterministic tiebreak
   or use `order_sensitive=False`). If it fails, the encoding/reference is wrong — fix it; do not
   repack a store whose result changed.
5. **DELETE what's genuinely left over** — only now, and only: untagged unused scratch, broken
   siblings (e.g. non-queryable `source_queries` models), duplicate encodings, and any kb/concept
   encoding you judged NOT useful and chose not to reference. Re-run the identical-result check.
6. **Re-verify:** `check_store` reports a **clean** store (`.ok` True: no substantive
   recommendation; advisory `EXPECTED_KB`/`ORPHAN`/`DEFERRED` may remain if waived) **AND**
   identical-result holds.
7. **Record + repack:** write the nice query to `<scratch>/_winning_query.json` (so a future scan
   validates against it), then repack with `edited_models.save_edited_store(...)`. Re-run
   `scripts/check_edited_model_stores.py --instance-ids <iid>` to confirm `-> OK`.

**Drive the REAL SLayer tools directly** — do NOT hand-edit YAML, hand-roll SQL, or wrap the tools
in a bespoke ops layer. Build the store's MCP server on the unpacked scratch and call the honest
tool surface exactly as the query agent would:

```python
from slayer.mcp.server import create_mcp_server
from slayer.storage.yaml_storage import YAMLStorage
mcp = create_mcp_server(YAMLStorage(base_dir=scratch))
tools = mcp._tool_manager._tools           # 'inspect','search','recommend_root_model',
                                           # 'create_model','edit_model','delete_model',
                                           # 'query','query_nested','validate_models', …
await tools["inspect"].fn(reference="<db>.<model>.<col>")     # read a def
await tools["create_model"].fn(name=..., data_source="<db>", sql=..., columns=[...])  # ADD encoding
await tools["edit_model"].fn(model_name=..., data_source="<db>", columns=[...], remove={...})
await tools["query"].fn(source_model=..., measures=[...], filters=[...], format="json")  # test
# dispose the engine when done: eng=getattr(mcp,'_slayer_engine',None); await eng.aclose()
```

**Two thin (non-MCP) helpers do the scaffolding — NOT the edits:**
- `store_cleanup.materialize_store(benchmark, db, instance_id, work_dir=…)` → backs up (once) and
  untars; returns `{scratch, meta, kb_rows}`. Capture the baseline result with
  `run_store_query(scratch, winning_query)` **before** you edit.
- `store_cleanup.verify_and_repack(benchmark, db, instance_id, work_dir, scratch, meta, kb_rows,
  relevant_kb_ids, baseline_result, nice_query, record_query=<nice≠original>)` → runs the nice
  query, GATES on identical-result vs `baseline_result`, runs the checker, and repacks in place
  (or leaves the archive untouched and returns the reason). YOU do the edits with the real tools
  in between; this only backs the correctness gate + repack.

## Worked exemplar — `reverse_logistics_4`
- `delete_model("return_sal_native")` (broken `source_queries` sibling; UNUSED).
- `edit_model("return_sal", columns=[{name:"sal", sql:"trc + 0.5*carbon_footprint -
  recovery_value", …}])` — reference `trc [kb=0]` instead of inlining it (clears INLINED_KB_DEF).
- `edit_model("return_sal", remove={"measures":["avg_sal"]})` — `avg_sal = round(sal:avg,2)` is a
  pure aggregation of KB quantity `sal`, i.e. THE ANSWER → demote. Reformulated query:
  `{source_model:"return_sal", measures:[{formula:"round(sal:avg, 2)"}]}`; recorded in
  `_winning_query.json`.
- Identical result: `-256.47` before and after. Checker → OK (only advisory EXPECTED_KB waived).

## Gotchas (each cost real debugging)
- **Demotion changes the supported query.** After removing an answer measure, the store no
  longer answers the *original* attempt query — record the reformulated query in
  `_winning_query.json`, or a later scan flags every column UNUSED.
- **`results_identical` ignores column headers** (grader is positional): `return_sal.avg_sal`
  vs `return_sal.round(sal_avg,_2)` with the same value ARE identical.
- **`external_knowledge` is unreliable** — its transitive closure can be large and the anchors
  can be mislabeled (e.g. residential_4's `[27,28,29]` are household-quality composites while the
  question is pure roomy-apartment arithmetic). Waive `EXPECTED_KB_NOT_MATERIALIZED` freely when
  the anchors don't match the actual question.
- **`_column_dependencies` drops trivial-base refs**, so INLINED_KB_DEF uses the inbound-edge
  rule (a stored KB column nobody references, whose KB is a child of a used KB's KB). PK and
  JOIN-key columns are structural scaffolding and are never flagged unused.
