## 1. Setup

- [x] 1.1 Merge `origin/main` into the branch (commit local first, never rebase/stash) and verify the tree is clean and the two upstream commits are present in `git log`.
- [x] 1.2 Bump both `motley-slayer[advanced-search]>=0.9.6` pins to `>=0.10.2` in `pyproject.toml`, `uv lock`, `uv sync --extra all --extra dev`; verify `uv.lock` resolves motley-slayer 0.10.2 and `python -c "import slayer; print(slayer.__version__)"` prints 0.10.2.

## 2. Import compatibility

- [x] 2.1 `slayer_pipeline/filter_normalization.py`: import `_normalize_sql_filter_operators` from `slayer.engine.syntax`; drop the `query_nested` branch of `normalize_tool_filters` and normalise `args["query"]` (dict → its filters, list → each stage). Verify `tests/slayer_pipeline/test_filter_normalization.py` collects and its normalisation cases pass.
- [x] 2.2 `slayer_otf/store_kb_checker.py`: replace the removed `parse_filter` with a helper on `parse_filter_expr` + `walk_parsed_refs` that recurses into `AggCall.source` (and kwarg refs); rehome `is_trivial_base` / `root_scope_column_ids` / `resolve_ref_target` from `slayer.sql.column_expansion`. Verify `tests/test_store_kb_checker.py` passes, including a new case for an aggregate-over-expression filter source.

## 3. Tool surface (query / submit_query)

- [x] 3.1 `agents/_query.py`: rewrite `query_impl(query, *, variables, show_sql, dry_run, explain, format, normalize_filters)` to accept dict|list|str, lift tool-level keys only from the single-object form and strip (not lift) them from list stages, normalise, and forward `fn(query=…, …)`; delete `query_nested_impl`, `_processed_nested_stages`, `_QUERY_JSON_ALLOWLIST`. Verify the rewritten wrapper tests pass.
- [x] 3.2 `agents/claude_sdk/agent.py`: `query` native = SLayer's own `query` description+schema (via `_slayer_tool_metadata`) + `normalize_filters`; `submit_query` native accepts `{query: <object|array>, normalize_filters}` only; delete the `query_nested` native, `"query_nested"` from `_STATIC_NATIVE_TOOLS`, `"help"` from `_SLAYER_NATIVE_NAMES`. Verify the natives build and advertise the SLayer `query` schema.
- [x] 3.3 `agents/_submit.py` + `agents/_tool_specs.py`: `_compile_sql` → `client.sql_sync(parsed)` only; drop `query_nested` wording; point the dry-run error at `inspect(entity_type="model")`. Verify `tests/test_submit_helpers.py` / `tests/test_slayer_tools.py` pass.
- [x] 3.4 Drop `query_nested` from `claude_sdk_otf_v1/agent.py::_MAIN_NATIVE_BARE` and `setup_encoder.py::SLAYER_MCP_TOOLS`; drop `help` from `harness.ACTION_COSTS`; collapse `store_cleanup.run_store_query` to the single `query.fn(query=…)` path; reword the `query_nested` remediation strings in `eval/annotation_schema.py` + `eval/autopsy.py`. Verify `tests/test_slayer_tool_surface.py` pins the 0.10.2 surface (20 tools, no `query_nested`/`help`).

## 4. Committed reference migration

- [x] 4.1 Round-trip every `slayer_models/<db>` through 0.10.2 `YAMLStorage` against an empty scratch sqlite and commit the schema-10 YAML; record the exact commands in the PR. Verify `git diff --stat slayer_models` is versioning + `unique: false` only.
- [x] 4.2 Drop the dead top-level `measures:` block from `crypto/smart_money_accuracy`, `mental/facility_metrics`, `robot/program_efficiency`; fix `fake/trusted_account` so its stage's 180-day window is a column + stage filter, not a rejected `ModelExtension.filters` key. Verify all four DBs load with no validation error.
- [x] 4.3 Add a shared conftest fixture that copies a committed DB to tmp before loading, and adopt it in every test that opens `slayer_models/` in place. Verify a full test run leaves `git status slayer_models` clean.
- [x] 4.4 Add the round-trip guard test (loading every committed DB leaves each `models/**.yaml` byte-identical). Verify it passes now and would fail on a schema drift.

## 5. Prompt cleanup

- [ ] 5.1 `_shared_otf_prompts.py`: remove the tool-covered blocks; add `_SINGLE_QUERY_NUDGE`, `_SLAYER_ONBOARDING` (gated tail), `_NORMALIZE_FILTERS_NOTE` (actual-subset wording), `_QUERY_ROOT_POINTER`, `_ENCODE_HOST_POINTER`, the one-clause define-before-reference, `_SUBMIT_CONTRACT`; recompose the four v0 builders. Verify the re-blessed SHA/presence tests and `tests/data/dev1603` goldens pass.
- [ ] 5.2 `claude_sdk_otf_v1` + `claude_sdk_otf_ainteract_v1` prompts, `_pre_encoded_prompts.py`, `claude_sdk_otf_encode/prompts.py`, `claude_sdk/partition.py`: apply the same removals/additions, keeping the v1-only and pre-encoded-only lines. Verify the v1/encoder/pre-encoded prompt tests pass.
- [ ] 5.3 Clean `SLAYER_A_INTERACT` / `SLAYER_C_INTERACT` in `claude_sdk/prompts.py` by the same rules (they still name the removed `help` tool). Verify `tests/test_prompt_parity.py` / `tests/test_prompt_kb_descriptions.py` pass.

## 6. Tests (new coverage, written in the spec-tests stage)

- [ ] 6.1 Add: dependency floor is 0.10.2; advertised surface == the 20 0.10.2 tools; `query`/`submit_query` native schemas validate as JSON Schema and accept representative payloads (computed dimension, inline ModelExtension, multi-stage list, `month(col)`); the KB-checker aggregate-source case; the reference round-trip guard; the three fixed models carry no top-level measures and `fake/trusted_account` applies its window.
- [ ] 6.2 Add per-prompt-family assertions: each family contains its kept blocks and the nudge/onboarding/note, and none of the removed markers (`query_nested`, `help`, colon syntax, dedup/search-vs-inspect/artifact blocks); raw prompts byte-identical. Verify all new tests fail before implementation and pass after.

## 7. Suite, docs, smoke

- [ ] 7.1 Run the full non-integration suite via the CLAUDE.md command and verify 0 failures / 0 errors.
- [ ] 7.2 Add a "SLayer 0.10.2" note to the project `CLAUDE.md` (one `query` argument, no `query_nested`/`help`, prompts must not restate tool docs, reference at schema 10 + migration procedure). Verify the note is present.
- [ ] 7.3 Run the two opus-4.8 local smokes (`claude_sdk_otf` and `claude_sdk_otf_v1`, one mini-interact task each, creds from `/home/james/GitHub/SLayer/.env.agents`); verify each reaches evaluation (a `Generated SQL:` + `Result:` observation, `submission_status` not a json/translation/dry-run/infrastructure error) with no tool-not-found / schema / `HermeticEnvError` entry, and record the outcome in the PR.
