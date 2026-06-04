"""Tests for agents/annotator/agent.py run_task (DEV-1518).

Contract:
* run_task happy path returns a valid AnnotatorResult with task_annotation set.
* submit_annotation tool validation error does NOT set _submission_done;
  agent can retry with corrected JSON.
* Turn cap without submit_annotation → AnnotatorResult(error=..., task_annotation=None).
* Mini-interact benchmark: get_ambiguity_resolutions tool is in allowed_tools.
* LiveSQLBench benchmark: get_ambiguity_resolutions tool is NOT in allowed_tools.
* AuditedGoldRef.file is filled by _fill_audited_gold_ref_files() (harness helper),
  not left as the agent-supplied sentinel.
* Non-Anthropic model produces an error result without calling the SDK.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Task data fixtures
# ---------------------------------------------------------------------------

def _task_mini(instance_id: str = "shop_1") -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": "shop",
        "amb_user_query": "How many premium orders?",
        "sol_sql": ["SELECT COUNT(*) FROM orders WHERE tier='Premium';"],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "[MASKED_TIER]",
                    "type": "value",
                    "is_mask": True,
                    "metadata_evidence": "KB 3",
                    "sql_snippet": "tier='Premium'",
                }
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [
            {
                "term": "premium tier",
                "definition": "Orders where tier='Premium'.",
                "deleted_knowledge": False,
            }
        ],
        "external_knowledge": [3],
    }


def _task_livesqlbench(instance_id: str = "flight_1") -> dict:
    return {
        "instance_id": instance_id,
        "selected_database": "flight",
        "amb_user_query": "Count all delayed flights.",
        "sol_sql": ["SELECT COUNT(*) FROM flights WHERE delayed=1;"],
        "external_knowledge": [1, 2],
    }


# ---------------------------------------------------------------------------
# Valid JSON payloads for submit_annotation
# ---------------------------------------------------------------------------

def _valid_task_annotation_json(instance_id: str = "shop_1", db: str = "shop") -> str:
    return json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "How many premium orders?",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 3 directly pins the tier.",
            "evidence_sources_consulted": ["kb:3"],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": instance_id,
        },
    })


# ---------------------------------------------------------------------------
# Fake SDK helpers
# ---------------------------------------------------------------------------

def _make_fake_sdk(monkeypatch, ann_agent, *, submit_calls=None, n_turns: int = 2):
    """Wire a fake ClaudeSDKClient into ann_agent.

    `submit_calls`: list of (task_annotation_json, variants_json) the fake
    client will invoke submit_annotation with (one per iteration, in order).
    If the list is exhausted, subsequent turns yield only an AssistantMessage
    without a tool call — simulating the agent stalling.
    """
    class _AssistantMsg:
        pass
    _AssistantMsg.__name__ = "AssistantMessage"

    submit_iter = iter(submit_calls or [])

    class FakeClient:
        def __init__(self, options):
            self._options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def query(self, *a, **kw):
            pass

        async def receive_response(self):
            for _ in range(n_turns):
                # Try to simulate a submit_annotation tool call this turn
                try:
                    ta_json, variants_json = next(submit_iter)
                    await ann_agent.submit_annotation(
                        {
                            "task_annotation_json": ta_json,
                            "audited_gold_variants_json": variants_json,
                        }
                    )
                except StopIteration:
                    pass
                msg = _AssistantMsg()
                yield msg
                if ann_agent._ctx.get("_submission_done"):
                    break

    monkeypatch.setattr(ann_agent, "ClaudeSDKClient", FakeClient)
    monkeypatch.setattr(ann_agent, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(ann_agent, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(ann_agent, "materialize_task_db", lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# run_task — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_happy_path_returns_valid_result(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _make_fake_sdk(
        monkeypatch, ann_agent,
        submit_calls=[(_valid_task_annotation_json(), "[]")],
    )

    result = await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="anthropic/claude-opus-4-7",
        effort="medium",
    )

    assert result.error is None
    assert result.task_annotation is not None
    assert result.task_annotation.instance_id == "shop_1"
    assert result.instance_id == "shop_1"
    assert isinstance(result.audited_gold_variants, list)
    assert result.duration_s >= 0


@pytest.mark.asyncio
async def test_run_task_happy_path_parses_audited_gold_variants(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    # original_gold_is_correct=True rejects non-empty variants; use False + ref.
    annotation_with_variant = json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "shop",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "How many premium orders?",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 3 directly pins the tier.",
            "evidence_sources_consulted": ["kb:3"],
        },
        "original_gold_is_correct": False,
        "gold_variants": [{
            "variant_id": "primary",
            "interpretation": "KB-anchored",
            "primary": True,
            "anchored_in": [],
            "audited_gold_ref": {
                "file": "__HARNESS_FILLS__",
                "instance_id": "shop_1",
                "variant_id": "primary",
            },
        }],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })
    variants = [
        {
            "instance_id": "shop_1",
            "variant_id": "primary",
            "primary": True,
            "selected_database": "shop",
            "benchmark": "mini-interact",
            "audit_status": "edited",
            "original_sol_sql": ["SELECT COUNT(*) FROM orders;"],
            "audited_sol_sql": ["SELECT COUNT(*) FROM orders WHERE active = 1;"],
            "audited_sample_row": [],
            "changes": [],
            "reasoning_summary": "Gold is correct.",
            "skill_version": "annotator-agent/1.0",
            "audited_at": "2026-06-02",
        }
    ]
    _make_fake_sdk(
        monkeypatch, ann_agent,
        submit_calls=[(annotation_with_variant, json.dumps(variants))],
    )

    result = await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="anthropic/claude-opus-4-7",
        effort="medium",
    )

    assert result.error is None
    assert len(result.audited_gold_variants) == 1
    assert result.audited_gold_variants[0]["variant_id"] == "primary"


# ---------------------------------------------------------------------------
# run_task — turn cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_turn_cap_returns_error_result(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _make_fake_sdk(monkeypatch, ann_agent, submit_calls=None, n_turns=1)

    result = await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="anthropic/claude-opus-4-7",
        effort="medium",
    )

    assert result.task_annotation is None
    assert result.error is not None


# ---------------------------------------------------------------------------
# run_task — submit_annotation validation error → retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_bad_json_then_good_json_succeeds(monkeypatch):
    """submit_annotation with bad JSON returns error string; agent can retry."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _make_fake_sdk(
        monkeypatch, ann_agent,
        submit_calls=[
            ("not valid json{{{", "[]"),           # first: bad → error to agent
            (_valid_task_annotation_json(), "[]"), # second: good → success
        ],
        n_turns=4,
    )

    result = await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="anthropic/claude-opus-4-7",
        effort="medium",
    )

    assert result.error is None
    assert result.task_annotation is not None


# ---------------------------------------------------------------------------
# run_task — benchmark-specific tool sets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mini_interact_includes_get_ambiguity_resolutions(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    allowed_tools_seen: list[list[str]] = []

    class CapturingClient:
        def __init__(self, options):
            allowed_tools_seen.append(list(options.allowed_tools or []))

        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def query(self, *a, **kw): pass

        async def receive_response(self):
            ann_agent._ctx.update({"_submission_done": True})
            return
            yield  # make it an async generator

    monkeypatch.setattr(ann_agent, "ClaudeSDKClient", CapturingClient)
    monkeypatch.setattr(ann_agent, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(ann_agent, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(ann_agent, "materialize_task_db", lambda *a, **kw: None)

    await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="anthropic/claude-opus-4-7",
        effort="medium",
    )

    all_tools = [t for tools in allowed_tools_seen for t in tools]
    assert any("get_ambiguity_resolutions" in t for t in all_tools)


@pytest.mark.asyncio
async def test_livesqlbench_excludes_get_ambiguity_resolutions(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    allowed_tools_seen: list[list[str]] = []

    class CapturingClient:
        def __init__(self, options):
            allowed_tools_seen.append(list(options.allowed_tools or []))

        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def query(self, *a, **kw): pass

        async def receive_response(self):
            ann_agent._ctx.update({"_submission_done": True})
            return
            yield

    monkeypatch.setattr(ann_agent, "ClaudeSDKClient", CapturingClient)
    monkeypatch.setattr(ann_agent, "create_sdk_mcp_server", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(ann_agent, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(ann_agent, "materialize_task_db", lambda *a, **kw: None)

    await ann_agent.run_task(
        task_data=_task_livesqlbench(),
        data_path_base="/tmp/data",
        benchmark="livesqlbench-base-lite-sqlite",
        model="anthropic/claude-opus-4-7",
        effort="medium",
    )

    all_tools = [t for tools in allowed_tools_seen for t in tools]
    assert not any("get_ambiguity_resolutions" in t for t in all_tools)


# ---------------------------------------------------------------------------
# AuditedGoldRef.file filled by harness
# ---------------------------------------------------------------------------

def test_fill_audited_gold_ref_files_replaces_sentinel():
    """_fill_audited_gold_ref_files must set AuditedGoldRef.file to the
    canonical consolidated JSONL path, replacing any sentinel value the
    agent may have supplied."""
    from bird_interact_agents.agents.annotator.agent import _fill_audited_gold_ref_files
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "shop",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "r",
            "evidence_sources_consulted": [],
        },
        "original_gold_is_correct": False,
        "gold_variants": [
            {
                "variant_id": "primary",
                "interpretation": "The canonical reading.",
                "primary": True,
                "anchored_in": ["kb:1"],
                "audited_gold_ref": {
                    "file": "__HARNESS_FILLS__",
                    "instance_id": "shop_1",
                    "variant_id": "primary",
                },
            }
        ],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })

    filled = _fill_audited_gold_ref_files(ann, benchmark="mini-interact")

    assert filled.gold_variants[0].audited_gold_ref.file == \
        "audited_gold/mini-interact_audited.jsonl"


def test_fill_audited_gold_ref_files_livesqlbench():
    from bird_interact_agents.agents.annotator.agent import _fill_audited_gold_ref_files
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "flight_1",
        "selected_database": "flight",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "r",
            "evidence_sources_consulted": [],
        },
        "original_gold_is_correct": False,
        "gold_variants": [
            {
                "variant_id": "primary",
                "interpretation": "The canonical reading.",
                "primary": True,
                "anchored_in": [],
                "audited_gold_ref": {
                    "file": "__HARNESS_FILLS__",
                    "instance_id": "flight_1",
                    "variant_id": "primary",
                },
            }
        ],
        "provenance": {
            "task_jsonl_path": "livesqlbench.jsonl",
            "task_jsonl_instance_id": "flight_1",
        },
    })

    filled = _fill_audited_gold_ref_files(ann, benchmark="livesqlbench-base-lite-sqlite")

    assert filled.gold_variants[0].audited_gold_ref.file == \
        "audited_gold/livesqlbench-base-lite-sqlite_audited.jsonl"


def test_fill_audited_gold_ref_files_noop_when_no_variants():
    """Tasks with original_gold_is_correct=True and no gold_variants:
    _fill_audited_gold_ref_files must not raise and must return the annotation unchanged."""
    from bird_interact_agents.agents.annotator.agent import _fill_audited_gold_ref_files
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "shop",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "r",
            "evidence_sources_consulted": [],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })

    filled = _fill_audited_gold_ref_files(ann, benchmark="mini-interact")
    assert filled.gold_variants == []


# ---------------------------------------------------------------------------
# Non-Anthropic model short-circuits
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _fill_deterministic_fields
# ---------------------------------------------------------------------------

def _minimal_annotation(instance_id: str = "shop_1", db: str = "shop") -> dict:
    return {
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "r",
            "evidence_sources_consulted": [],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "PLACEHOLDER",
            "task_jsonl_instance_id": "PLACEHOLDER",
        },
    }


def test_fill_deterministic_fields_overwrites_provenance_and_external_knowledge():
    """_fill_deterministic_fields must overwrite provenance.* and external_knowledge
    from task_data regardless of what the agent placed in those fields."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate(_minimal_annotation())
    task_data = {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "external_knowledge": [3, 7],
    }
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="mini-interact")

    assert filled.external_knowledge == [3, 7]
    assert filled.provenance.task_jsonl_path == "mini_interact.jsonl"
    assert filled.provenance.task_jsonl_instance_id == "shop_1"


def test_fill_deterministic_fields_livesqlbench_provenance():
    """livesqlbench benchmark produces the correct task_jsonl_path from the registry."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation
    from bird_interact_agents.eval.implicit_annotation import _benchmark_task_jsonl_name

    base = _minimal_annotation("flight_1", "flight")
    base["provenance"]["task_jsonl_path"] = "WRONG"
    ann = TaskAnnotation.model_validate(base)
    task_data = {"instance_id": "flight_1", "selected_database": "flight"}
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="livesqlbench-base-lite-sqlite")

    assert filled.provenance.task_jsonl_path == _benchmark_task_jsonl_name("livesqlbench-base-lite-sqlite")
    assert filled.provenance.task_jsonl_instance_id == "flight_1"


def test_fill_deterministic_fields_merges_masked_terms_from_critical_ambiguity():
    """For mini_interact, _fill_deterministic_fields must merge critical_ambiguity
    entries as is_mask=True MaskedTerm objects without duplicating existing entries."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import MaskedTerm, TaskAnnotation

    base = _minimal_annotation()
    # Agent already found one schema-linking term (is_mask=False)
    base["masked_terms"] = [
        {"term": "area", "type": "schema_linking_ambiguity", "is_mask": False, "metadata_evidence": []}
    ]
    ann = TaskAnnotation.model_validate(base)
    task_data = {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "[MASKED_TIER]",
                    "type": "knowledge_linking_ambiguity",
                    "is_mask": True,
                    "metadata_evidence": "KB 3",
                    "sql_snippet": "tier='Premium'",
                }
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
    }
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="mini-interact")

    terms_by_name = {mt.term: mt for mt in filled.masked_terms}
    assert "[MASKED_TIER]" in terms_by_name
    assert terms_by_name["[MASKED_TIER]"].is_mask is True
    assert terms_by_name["[MASKED_TIER]"].type == "knowledge_linking_ambiguity"
    # Agent's is_mask=False entry must be preserved
    assert "area" in terms_by_name
    assert terms_by_name["area"].is_mask is False


def test_fill_deterministic_fields_string_metadata_evidence_wrapped_in_list():
    """metadata_evidence that is a string (e.g. 'KB 3') must be wrapped in a
    single-element list, not silently dropped."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate(_minimal_annotation())
    task_data = {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "[MASKED_TIER]",
                    "type": "knowledge_linking_ambiguity",
                    "is_mask": True,
                    "metadata_evidence": "KB 3",
                }
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
    }
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="mini-interact")

    mt = next(mt for mt in filled.masked_terms if mt.term == "[MASKED_TIER]")
    assert mt.metadata_evidence == ["KB 3"]


def test_fill_deterministic_fields_is_mask_false_same_term_does_not_block_authoritative_entry():
    """If the agent submitted an is_mask=False entry with the same term as a critical_ambiguity
    item, the harness must still insert the authoritative is_mask=True entry — deduplicate only
    against existing is_mask=True entries."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    base = _minimal_annotation()
    # Agent found [MASKED_TIER] as a schema-linking ambiguity (is_mask=False).
    base["masked_terms"] = [
        {"term": "[MASKED_TIER]", "type": "schema_linking_ambiguity", "is_mask": False, "metadata_evidence": []}
    ]
    ann = TaskAnnotation.model_validate(base)
    task_data = {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "[MASKED_TIER]",
                    "type": "knowledge_linking_ambiguity",
                    "is_mask": True,
                    "metadata_evidence": "KB 3",
                }
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
    }
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="mini-interact")

    mask_true_entries = [mt for mt in filled.masked_terms if mt.term == "[MASKED_TIER]" and mt.is_mask]
    assert len(mask_true_entries) == 1, (
        "Authoritative is_mask=True entry must be present even when agent submitted "
        "an is_mask=False entry with the same term"
    )


def test_fill_deterministic_fields_no_duplicate_masked_terms():
    """If critical_ambiguity contains a term already in masked_terms, it must NOT be duplicated."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    base = _minimal_annotation()
    base["masked_terms"] = [
        {"term": "[MASKED_TIER]", "type": "knowledge_linking_ambiguity", "is_mask": True, "metadata_evidence": []}
    ]
    ann = TaskAnnotation.model_validate(base)
    task_data = {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {"term": "[MASKED_TIER]", "type": "knowledge_linking_ambiguity", "is_mask": True, "metadata_evidence": "KB 3"}
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
    }
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="mini-interact")

    assert sum(1 for mt in filled.masked_terms if mt.term == "[MASKED_TIER]") == 1


def test_fill_deterministic_fields_authoritative_overwrites_stale_is_mask_true():
    """If the agent already submitted an is_mask=True entry for a term that appears
    in critical_ambiguity, the harness must REPLACE it with the authoritative
    metadata_evidence — not skip the authoritative entry."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    base = _minimal_annotation()
    # Agent submitted stale is_mask=True entry with wrong metadata_evidence.
    base["masked_terms"] = [
        {"term": "[MASKED_TIER]", "type": "knowledge_linking_ambiguity",
         "is_mask": True, "metadata_evidence": ["stale_source"]}
    ]
    ann = TaskAnnotation.model_validate(base)
    task_data = {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {"term": "[MASKED_TIER]", "type": "knowledge_linking_ambiguity",
                 "is_mask": True, "metadata_evidence": ["authoritative_source"]}
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
    }
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="mini-interact")

    mask_entries = [mt for mt in filled.masked_terms if mt.term == "[MASKED_TIER]" and mt.is_mask]
    assert len(mask_entries) == 1, "Must have exactly one is_mask=True entry (no duplicates)"
    assert mask_entries[0].metadata_evidence == ["authoritative_source"], (
        "Harness-authoritative metadata_evidence must replace the stale agent-submitted value"
    )


def test_fill_deterministic_fields_livesqlbench_skips_masked_terms():
    """For livesqlbench (no user_query_ambiguity), masked_terms must be left untouched."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    base = _minimal_annotation("flight_1", "flight")
    ann = TaskAnnotation.model_validate(base)
    task_data = {"instance_id": "flight_1", "selected_database": "flight"}
    filled = _fill_deterministic_fields(ann, task_data=task_data, benchmark="livesqlbench-base-lite-sqlite")

    assert filled.masked_terms == []


def test_fill_deterministic_fields_does_not_mutate_original():
    """_fill_deterministic_fields must return a new object; the input annotation is unchanged."""
    from bird_interact_agents.agents.annotator.agent import _fill_deterministic_fields
    from bird_interact_agents.eval.annotation_schema import TaskAnnotation

    ann = TaskAnnotation.model_validate(_minimal_annotation())
    original_path = ann.provenance.task_jsonl_path

    _fill_deterministic_fields(
        ann,
        task_data={"instance_id": "shop_1", "external_knowledge": [99]},
        benchmark="mini-interact",
    )

    assert ann.provenance.task_jsonl_path == original_path


# ---------------------------------------------------------------------------
# Non-Anthropic model short-circuits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_anthropic_model_returns_error_without_sdk_call(monkeypatch):
    from bird_interact_agents.agents.annotator import agent as ann_agent

    sdk_called = []

    class NeverCalledClient:
        def __init__(self, *a, **kw):
            sdk_called.append(1)

        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def query(self, *a, **kw): pass
        async def receive_response(self):
            return
            yield

    monkeypatch.setattr(ann_agent, "ClaudeSDKClient", NeverCalledClient)
    monkeypatch.setattr(ann_agent, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(ann_agent, "materialize_task_db", lambda *a, **kw: None)

    result = await ann_agent.run_task(
        task_data=_task_mini(),
        data_path_base="/tmp/data",
        benchmark="mini-interact",
        model="openai/gpt-4o",  # not an Anthropic model
        effort="medium",
    )

    assert sdk_called == []
    assert result.task_annotation is None
    assert result.error is not None
    assert "anthropic" in result.error.lower() or "model" in result.error.lower()
