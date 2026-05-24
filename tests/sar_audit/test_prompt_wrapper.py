"""Wrapper renders upstream BIRD prompt + BIRD-Interact KB sections."""

from __future__ import annotations

import re

from bird_interact_agents.sar_audit import prompt_wrapper


def _render(task, kb, column_meanings):
    return prompt_wrapper.render_prompt(
        task=task,
        full_kb=kb,
        full_column_meanings=column_meanings,
        schema_str="CREATE TABLE t (x INTEGER);",
    )


def test_upstream_prompt_body_appears_verbatim(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    out = _render(fake_task, fake_kb, fake_column_meanings)
    assert "<<UPSTREAM_BIRD_PROMPT_BEGIN" in out
    assert "UPSTREAM_BIRD_PROMPT_END>>" in out


def test_bird_interact_section_precedes_upstream_body(
    stub_upstream, fake_task, fake_kb, fake_column_meanings
):
    """The wrapper function is `prepend_bird_interact_section` — KB section
    comes BEFORE the upstream BIRD prompt body."""
    out = _render(fake_task, fake_kb, fake_column_meanings)
    kb_idx = out.find("## BIRD-Interact knowledge")
    upstream_idx = out.find("<<UPSTREAM_BIRD_PROMPT_BEGIN")
    assert kb_idx >= 0
    assert upstream_idx >= 0
    assert kb_idx < upstream_idx, (
        f"BIRD-Interact knowledge section (idx {kb_idx}) must precede upstream "
        f"prompt body (idx {upstream_idx})"
    )


def test_upstream_template_formatted_with_amb_user_query_and_sol_sql(
    stub_upstream, fake_task, fake_kb, fake_column_meanings
):
    out = _render(fake_task, fake_kb, fake_column_meanings)
    # Loaded the upstream template exactly once.
    assert len(stub_upstream.calls) == 1
    # The formatted template carries the right values for amb_user_query
    # and sol_sql, with an empty evidence section (we feed KB via the
    # wrapper, not via upstream's external_knowledge slot).
    assert f"Question: {fake_task['amb_user_query']}" in out
    assert f"Annotated Query: {fake_task['sol_sql'][0]}" in out
    assert "External Knowledge: " in out
    assert "CREATE TABLE t" in out


def test_bird_interact_section_header_present_exactly_once(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    out = _render(fake_task, fake_kb, fake_column_meanings)
    assert out.count("## BIRD-Interact knowledge") == 1


def test_full_kb_included_not_external_knowledge_subset(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    # task.external_knowledge == [2], but the wrapper must include ALL kb entries.
    out = _render(fake_task, fake_kb, fake_column_meanings)
    for entry in fake_kb:
        assert f"KB {entry['id']}" in out or f"kb:{entry['id']}" in out or f"id={entry['id']}" in out, (
            f"missing id marker for kb entry {entry['id']}"
        )
        assert entry["knowledge"] in out, f"missing verbatim knowledge text for kb entry {entry['id']}"


def test_task_external_knowledge_called_out_as_hint(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    # external_knowledge=[2] should appear under a hint sub-section,
    # distinct from (but in addition to) the full-KB section.
    out = _render(fake_task, fake_kb, fake_column_meanings)
    hint_header_pattern = re.compile(r"(task[-_ ]flagged|external[_ ]knowledge|hints)", re.IGNORECASE)
    match = hint_header_pattern.search(out)
    assert match is not None, "no sub-header calling out task-level external_knowledge hints"
    assert "2" in out[match.start() : match.start() + 500]


def test_full_column_meanings_included(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    out = _render(fake_task, fake_kb, fake_column_meanings)
    # Top-level key
    assert "t|x" in out
    assert "the integer payload column" in out
    # Nested fields_meaning
    assert "t|extra" in out
    assert "JSON blob" in out
    assert "the key field" in out


def test_labeled_ambiguities_rendered(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    out = _render(fake_task, fake_kb, fake_column_meanings)
    # critical
    assert "smallest" in out
    assert "ORDER BY x ASC LIMIT 1" in out
    # non_critical
    assert "show me" in out
    assert "SELECT x" in out


def test_knowledge_ambiguities_rendered(stub_upstream, fake_task, fake_kb, fake_column_meanings):
    out = _render(fake_task, fake_kb, fake_column_meanings)
    assert "x value" in out


def test_empty_labeled_ambiguities_does_not_break(stub_upstream, fake_kb, fake_column_meanings):
    task = {
        "instance_id": "fake_2",
        "selected_database": "fake",
        "sol_sql": ["SELECT 1"],
        "amb_user_query": "trivial",
        "external_knowledge": [],
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
        "knowledge_ambiguity": [],
    }
    out = prompt_wrapper.render_prompt(
        task=task,
        full_kb=fake_kb,
        full_column_meanings=fake_column_meanings,
        schema_str="CREATE TABLE t (x INTEGER);",
    )
    # Still renders without error; KB section is present even if task has no labels.
    assert "## BIRD-Interact knowledge" in out
    assert "KB ONE" in out or "rows are integers" in out
