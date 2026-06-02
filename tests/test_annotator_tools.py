"""Unit tests for annotator tool implementations (DEV-1518).

Contract:
* get_ambiguity_resolutions reads critical_ambiguity from the NESTED path
  task_data["user_query_ambiguity"]["critical_ambiguity"], not top-level.
* knowledge_ambiguity is read from the TOP-LEVEL task_data["knowledge_ambiguity"].
* Both sources are surfaced in the tool's text output.
* submit_annotation with bad task_annotation_json returns error text and
  does NOT set _submission_done.
* submit_annotation with valid JSON sets _submission_done and stores
  annotation_result in the contextvar.
* submit_annotation with invalid TaskAnnotation schema returns error and
  does NOT set _submission_done.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Context setup helper
# ---------------------------------------------------------------------------

def _setup_ctx(task_data: dict, benchmark: str = "mini_interact") -> None:
    from bird_interact_agents.agents.annotator import agent as ann_agent

    ctx: dict = {
        "task_data": task_data,
        "benchmark": benchmark,
        "data_path_base": "/tmp/data",
        "annotation_result": None,
        "_submission_done": False,
    }
    ann_agent._ctx_var.set(ctx)


def _task_with_ambiguities() -> dict:
    return {
        "instance_id": "shop_1",
        "selected_database": "shop",
        "amb_user_query": "Count [MASKED_TIER] orders.",
        "sol_sql": ["SELECT COUNT(*) FROM orders WHERE tier='Premium';"],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "[MASKED_TIER]",
                    "type": "value",
                    "is_mask": True,
                    "metadata_evidence": "KB 3",
                    "sql_snippet": "tier='Premium'",
                },
                {
                    "term": "[MASKED_CLASS]",
                    "type": "category",
                    "is_mask": True,
                    "metadata_evidence": "KB 5",
                    "sql_snippet": "class IN ('A','B')",
                },
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


def _valid_task_annotation_json(instance_id: str = "shop_1") -> str:
    return json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": "shop",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 3 pins the tier.",
            "consulted_sources": ["kb:3"],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": instance_id,
        },
    })


# ---------------------------------------------------------------------------
# get_ambiguity_resolutions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_ambiguity_resolutions_returns_critical_snippets():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx(_task_with_ambiguities())
    result = await ann_agent.get_ambiguity_resolutions({})

    text = result["content"][0]["text"]
    assert "[MASKED_TIER]" in text
    assert "tier='Premium'" in text
    assert "[MASKED_CLASS]" in text
    assert "class IN ('A','B')" in text


@pytest.mark.asyncio
async def test_get_ambiguity_resolutions_includes_knowledge_ambiguity():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx(_task_with_ambiguities())
    result = await ann_agent.get_ambiguity_resolutions({})

    text = result["content"][0]["text"]
    assert "premium tier" in text


@pytest.mark.asyncio
async def test_get_ambiguity_resolutions_reads_nested_critical_ambiguity_path():
    """critical_ambiguity lives at task_data['user_query_ambiguity']['critical_ambiguity'].
    A stray top-level 'critical_ambiguity' key must NOT be read."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    task_data = {
        "instance_id": "x_1",
        "selected_database": "x",
        "amb_user_query": "q",
        "sol_sql": ["SELECT 1;"],
        "user_query_ambiguity": {
            "critical_ambiguity": [
                {
                    "term": "[CORRECT_TERM]",
                    "sql_snippet": "correct_value=1",
                    "type": "value",
                    "is_mask": True,
                    "metadata_evidence": "KB 1",
                }
            ],
            "non_critical_ambiguity": [],
        },
        "knowledge_ambiguity": [],
        # Deliberate trap: top-level key that should NOT be read
        "critical_ambiguity": [
            {"term": "WRONG_TERM", "sql_snippet": "WRONG_VALUE"}
        ],
    }
    _setup_ctx(task_data)
    result = await ann_agent.get_ambiguity_resolutions({})
    text = result["content"][0]["text"]

    assert "CORRECT_TERM" in text
    assert "correct_value=1" in text
    assert "WRONG_TERM" not in text


@pytest.mark.asyncio
async def test_get_ambiguity_resolutions_empty_when_no_ambiguities():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    task_data = {
        "instance_id": "x_1",
        "selected_database": "x",
        "amb_user_query": "q",
        "sol_sql": ["SELECT 1;"],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
        "knowledge_ambiguity": [],
    }
    _setup_ctx(task_data)
    result = await ann_agent.get_ambiguity_resolutions({})

    # Must return something (not crash), content can be empty/acknowledgement
    assert "content" in result


# ---------------------------------------------------------------------------
# submit_annotation — validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_annotation_bad_json_returns_error():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    result = await ann_agent.submit_annotation({
        "task_annotation_json": "not valid json{{{",
        "audited_gold_variants_json": "[]",
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "invalid" in text


@pytest.mark.asyncio
async def test_submit_annotation_bad_json_does_not_set_submission_done():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    await ann_agent.submit_annotation({
        "task_annotation_json": "not valid json{{{",
        "audited_gold_variants_json": "[]",
    })

    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_schema_violation_returns_error():
    """Valid JSON but fails TaskAnnotation schema validation → error, no done flag."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    # Missing required fields (metadata_sufficiency, provenance, etc.)
    bad_schema = json.dumps({"instance_id": "shop_1", "kind": "task_annotation"})
    result = await ann_agent.submit_annotation({
        "task_annotation_json": bad_schema,
        "audited_gold_variants_json": "[]",
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "invalid" in text or "validation" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_valid_sets_done_flag():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": "[]",
    })

    assert ann_agent._ctx.get("_submission_done") is True
    assert ann_agent._ctx.get("annotation_result") is not None


@pytest.mark.asyncio
async def test_submit_annotation_stores_annotation_result():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": "[]",
    })

    stored = ann_agent._ctx.get("annotation_result")
    assert stored is not None
    assert stored["task_annotation"].instance_id == "shop_1"
    assert stored["audited_gold_variants"] == []


@pytest.mark.asyncio
async def test_submit_annotation_parses_audited_gold_variants():
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    variants = json.dumps([
        {
            "instance_id": "shop_1",
            "variant_id": "primary",
            "selected_database": "shop",
            "benchmark": "mini_interact",
            "audit_status": "clean",
            "original_sol_sql": ["SELECT 1;"],
            "audited_sol_sql": ["SELECT 1;"],
            "audited_sample_row": [],
            "changes": [],
            "reasoning_summary": "correct",
            "skill_version": "annotator-agent/1.0",
            "audited_at": "2026-06-02",
        }
    ])
    await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variants,
    })

    stored = ann_agent._ctx.get("annotation_result")
    assert len(stored["audited_gold_variants"]) == 1
    assert stored["audited_gold_variants"][0]["variant_id"] == "primary"
