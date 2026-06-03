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

def _setup_ctx(task_data: dict, benchmark: str = "mini-interact") -> None:
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
async def test_submit_annotation_zero_primary_variants_rejected():
    """gold_variants with no primary=True must be rejected — downstream grading
    uses the primary variant for N2 and a zero-primary annotation would always
    produce N2=fail regardless of answer quality."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    annotation_no_primary = json.dumps({
        "instance_id": "shop_1",
        "selected_database": "shop",
        "benchmark": "mini_interact",
        "kind": "task_annotation",
        "original_gold_is_correct": False,
        "metadata_sufficiency": {"verdict": "sufficient", "reason": "r"},
        "gold_variants": [
            {
                "variant_id": "canonical_only",
                "interpretation": "The canonical reading.",
                "primary": False,
                "audited_gold_ref": {
                    "file": "audited_gold/mini_interact_audited.jsonl",
                    "instance_id": "shop_1",
                    "variant_id": "canonical_only",
                },
            }
        ],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })
    result = await ann_agent.submit_annotation({
        "task_annotation_json": annotation_no_primary,
        "audited_gold_variants_json": "[]",
    })
    text = result["content"][0]["text"].lower()
    assert "error" in text or "primary" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_wrong_benchmark_returns_error():
    """A variant whose benchmark doesn't match the current run benchmark must
    return an error so GCS routing uses the correct benchmark path."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"}, benchmark="mini_interact")
    variant_wrong_benchmark = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "variant_id": "primary",
            "benchmark": "livesqlbench",  # wrong
            "audit_status": "clean",
            "audited_sol_sql": ["SELECT 1;"],
        }
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variant_wrong_benchmark,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "benchmark" in text
    assert not ann_agent._ctx.get("_submission_done")


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
    # Must use original_gold_is_correct=False with a gold_variants reference,
    # since original_gold_is_correct=True now rejects any non-empty variants.
    annotation_json = json.dumps({
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
    variants = json.dumps([
        {
            "instance_id": "shop_1",
            "variant_id": "primary",
            "primary": True,
            "selected_database": "shop",
            "benchmark": "mini-interact",
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
        "task_annotation_json": annotation_json,
        "audited_gold_variants_json": variants,
    })

    stored = ann_agent._ctx.get("annotation_result")
    assert len(stored["audited_gold_variants"]) == 1
    assert stored["audited_gold_variants"][0]["variant_id"] == "primary"


@pytest.mark.asyncio
async def test_submit_annotation_wrong_instance_id_returns_error():
    """A TaskAnnotation with an instance_id that doesn't match the current
    task context must return an error and NOT set _submission_done."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("alien_99"),
        "audited_gold_variants_json": "[]",
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "validation" in text or "instance_id" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_wrong_database_returns_error():
    """A TaskAnnotation with a selected_database that doesn't match the
    current task context must return an error and NOT set _submission_done."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    # Build a valid annotation JSON but swap the database to "alien".
    wrong_db = json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "alien",  # wrong DB
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
    result = await ann_agent.submit_annotation({
        "task_annotation_json": wrong_db,
        "audited_gold_variants_json": "[]",
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "validation" in text or "selected_database" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_missing_required_field_returns_error():
    """A variant dict that omits a required field (e.g. variant_id) must
    return an error and NOT set _submission_done — so the malformed row
    is never written to GCS stable blobs."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    # Missing 'variant_id' — a required field.
    bad_variant = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "benchmark": "mini-interact",
            "audit_status": "clean",
            "audited_sol_sql": ["SELECT 1;"],
            # variant_id intentionally omitted
        }
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": bad_variant,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "missing" in text or "required" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_not_a_dict_returns_error():
    """If audited_gold_variants_json is an array but contains a non-dict
    element, return an error and do NOT set _submission_done."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": json.dumps(["not_a_dict"]),
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "invalid" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_wrong_instance_id_returns_error():
    """A variant whose instance_id doesn't match the current task must
    return an error so cross-task contamination is caught before GCS write."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    variant_wrong_iid = json.dumps([
        {
            "instance_id": "alien_99",  # wrong
            "selected_database": "shop",
            "variant_id": "primary",
            "benchmark": "mini-interact",
            "audit_status": "clean",
            "audited_sol_sql": ["SELECT 1;"],
        }
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variant_wrong_iid,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "instance_id" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_wrong_database_returns_error():
    """A variant whose selected_database doesn't match the current task must
    return an error."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    variant_wrong_db = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "alien",  # wrong
            "variant_id": "primary",
            "benchmark": "mini-interact",
            "audit_status": "clean",
            "audited_sol_sql": ["SELECT 1;"],
        }
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variant_wrong_db,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "selected_database" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_audited_sol_sql_not_a_list_returns_error():
    """audited_sol_sql must be a list; a bare string silently breaks tolerant_grader
    which does list(v.get("audited_sol_sql") or []) and iterates characters."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    variant_string_sql = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "variant_id": "primary",
            "benchmark": "mini_interact",
            "audit_status": "clean",
            "audited_sol_sql": "SELECT 1;",  # string instead of list
        }
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variant_string_sql,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "audited_sol_sql" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_variant_invalid_audit_status_returns_error():
    """audit_status must be one of the known valid values; an unknown value is
    silently ignored by the harness, so reject it at submit time."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    variant_bad_status = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "variant_id": "primary",
            "benchmark": "mini_interact",
            "audit_status": "unknown_status",  # not a valid value
            "audited_sol_sql": ["SELECT 1;"],
        }
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variant_bad_status,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "audit_status" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_gold_variant_ref_missing_audited_row_returns_error():
    """A TaskAnnotation that references a variant_id in gold_variants but
    doesn't include that variant_id in audited_gold_variants_json must return
    an error — prevents the grader from falling back to wrong gold."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    annotation_with_variant = json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "shop",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 3 pins the tier.",
            "evidence_sources_consulted": ["kb:3"],
        },
        "original_gold_is_correct": False,
        "gold_variants": [
            {
                "variant_id": "canonical_only",
                "interpretation": "KB-anchored",
                "primary": True,
                "anchored_in": ["kb:3"],
                "audited_gold_ref": {
                    "file": "__HARNESS_FILLS__",
                    "instance_id": "shop_1",
                    "variant_id": "canonical_only",
                },
            }
        ],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })
    # Audited variants list is empty — no row for "canonical_only".
    result = await ann_agent.submit_annotation({
        "task_annotation_json": annotation_with_variant,
        "audited_gold_variants_json": "[]",
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "validation" in text or "canonical_only" in text.lower()
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_gold_variant_ref_with_matching_row_succeeds():
    """When gold_variants refs are all satisfied by audited_gold_variants_json,
    submission must succeed."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    annotation_with_variant = json.dumps({
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
                "variant_id": "canonical_only",
                "interpretation": "KB-anchored",
                "primary": True,
                "anchored_in": [],
                "audited_gold_ref": {
                    "file": "__HARNESS_FILLS__",
                    "instance_id": "shop_1",
                    "variant_id": "canonical_only",
                },
            }
        ],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })
    matching_variant = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "variant_id": "canonical_only",
            "primary": True,
            "benchmark": "mini-interact",
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
    result = await ann_agent.submit_annotation({
        "task_annotation_json": annotation_with_variant,
        "audited_gold_variants_json": matching_variant,
    })

    assert ann_agent._ctx.get("_submission_done") is True


@pytest.mark.asyncio
async def test_submit_annotation_original_gold_correct_with_variants_rejected():
    """original_gold_is_correct=True with a non-empty audited_gold_variants_json
    must be rejected — a stray audited row could make future submissions pass
    against a task marked as requiring the original gold."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    correct_gold_annotation = json.dumps({
        "instance_id": "shop_1",
        "selected_database": "shop",
        "schema_version": 1,
        "kind": "task_annotation",
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
    stray_variant = json.dumps([{
        "instance_id": "shop_1",
        "selected_database": "shop",
        "variant_id": "v0",
        "benchmark": "mini_interact",
        "audit_status": "clean",
        "audited_sol_sql": ["SELECT 1;"],
    }])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": correct_gold_annotation,
        "audited_gold_variants_json": stray_variant,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "validation" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_empty_audited_sol_sql_non_unrecoverable_rejected():
    """audited_sol_sql=[] must be rejected for non-unrecoverable audit_status —
    a variant with empty SQL would silently behave as if no audited row exists
    and cause wrong N2/N3 grading."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    variant_empty_sql = json.dumps([{
        "instance_id": "shop_1",
        "selected_database": "shop",
        "variant_id": "v0",
        "benchmark": "mini_interact",
        "audit_status": "clean",
        "audited_sol_sql": [],  # empty — should be rejected for audit_status=clean
    }])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": _valid_task_annotation_json("shop_1"),
        "audited_gold_variants_json": variant_empty_sql,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "sql" in text
    assert not ann_agent._ctx.get("_submission_done")


@pytest.mark.asyncio
async def test_submit_annotation_multiple_primary_variants_rejected():
    """Multiple audited_gold_variants rows with primary=True must be rejected —
    the grader picks the first primary it sees, so multiple primaries produce
    unstable N2 grading across different consolidated file orderings."""
    from bird_interact_agents.agents.annotator import agent as ann_agent

    _setup_ctx({"instance_id": "shop_1", "selected_database": "shop"})
    annotation_with_two_variants = json.dumps({
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": "shop_1",
        "selected_database": "shop",
        "annotated_by": "annotator-agent",
        "annotated_at": "2026-06-02",
        "amb_user_query": "q",
        "metadata_sufficiency": {
            "verdict": "ambiguous",
            "rationale": "r",
            "evidence_sources_consulted": [],
        },
        "original_gold_is_correct": False,
        "gold_variants": [
            {
                "variant_id": "v1",
                "interpretation": "a",
                "primary": True,
                "anchored_in": [],
                "audited_gold_ref": {
                    "file": "__HARNESS_FILLS__",
                    "instance_id": "shop_1",
                    "variant_id": "v1",
                },
            },
            {
                "variant_id": "v2",
                "interpretation": "b",
                "primary": False,
                "anchored_in": [],
                "audited_gold_ref": {
                    "file": "__HARNESS_FILLS__",
                    "instance_id": "shop_1",
                    "variant_id": "v2",
                },
            },
        ],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": "shop_1",
        },
    })
    two_primaries = json.dumps([
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "variant_id": "v1",
            "primary": True,
            "benchmark": "mini_interact",
            "audit_status": "clean",
            "audited_sol_sql": ["SELECT 1;"],
        },
        {
            "instance_id": "shop_1",
            "selected_database": "shop",
            "variant_id": "v2",
            "primary": True,  # second primary — should be rejected
            "benchmark": "mini_interact",
            "audit_status": "clean",
            "audited_sol_sql": ["SELECT 2;"],
        },
    ])
    result = await ann_agent.submit_annotation({
        "task_annotation_json": annotation_with_two_variants,
        "audited_gold_variants_json": two_primaries,
    })

    text = result["content"][0]["text"].lower()
    assert "error" in text or "primary" in text
    assert not ann_agent._ctx.get("_submission_done")
