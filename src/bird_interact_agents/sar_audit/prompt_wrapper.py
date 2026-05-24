"""Render the SAR-audit prompt: BIRD-Interact KB section first, then upstream BIRD template formatted with our data."""

from __future__ import annotations

from . import _upstream_import


def render_prompt(
    *,
    task: dict,
    full_kb: list[dict],
    full_column_meanings: dict,
    schema_str: str,
) -> str:
    """Return the wrapped prompt for one mini-interact task.

    Loads upstream's `prompts/prompt_user_bird.txt` template and formats it
    with `question=amb_user_query`, `schema=schema_str`, `external_knowledge=""`,
    `gold_query=sol_sql[0]`. Then prepends a `## BIRD-Interact knowledge`
    section with the full KB, full column meanings, the task's
    `external_knowledge` hint ids, labeled ambiguities, and knowledge
    ambiguities.
    """
    template = _upstream_import.load_prompt_template()
    base = _format_template(
        template,
        question=task["amb_user_query"],
        schema=schema_str,
        external_knowledge="",
        gold_query=task["sol_sql"][0],
    )

    section = _render_bird_interact_section(
        task=task,
        full_kb=full_kb,
        full_column_meanings=full_column_meanings,
    )
    return section + "\n\n" + base


def _format_template(template: str, **kwargs) -> str:
    """Format upstream's template via .format(). Escaping `{` / `}` in our
    KB-section content is handled separately — we never .format() the
    BIRD-Interact section."""
    return template.format(**kwargs)


def _render_bird_interact_section(
    *,
    task: dict,
    full_kb: list[dict],
    full_column_meanings: dict,
) -> str:
    lines: list[str] = ["## BIRD-Interact knowledge", ""]

    # Full KB
    lines.append("### Full knowledge base")
    for entry in full_kb:
        lines.append(f"- KB {entry['id']}: {entry.get('knowledge', '')}")
    lines.append("")

    # Task-flagged external_knowledge hints
    ext_ids = task.get("external_knowledge") or []
    lines.append("### Task-flagged external_knowledge hints")
    if ext_ids:
        lines.append(f"The task itself flagged these KB ids as most relevant: {ext_ids}.")
    else:
        lines.append("(none)")
    lines.append("")

    # Full column meanings
    lines.append("### Full column meanings")
    for key, val in full_column_meanings.items():
        if isinstance(val, str):
            lines.append(f"- {key}: {val}")
        elif isinstance(val, dict):
            top = val.get("column_meaning")
            if top:
                lines.append(f"- {key}: {top}")
            fields = val.get("fields_meaning") or {}
            for sub_key, sub_val in _walk_fields(key, fields):
                lines.append(f"- {sub_key}: {sub_val}")
    lines.append("")

    # Labeled ambiguities
    uqa = task.get("user_query_ambiguity") or {}
    crit = uqa.get("critical_ambiguity") or []
    non_crit = uqa.get("non_critical_ambiguity") or []
    lines.append("### Labeled ambiguities")
    if crit:
        lines.append("Critical:")
        for entry in crit:
            term = entry.get("term", "")
            snippet = entry.get("sql_snippet", "")
            lines.append(f"- {term}: {snippet}")
    if non_crit:
        lines.append("Non-critical:")
        for entry in non_crit:
            term = entry.get("term", "")
            snippet = entry.get("sql_snippet", "")
            lines.append(f"- {term}: {snippet}")
    if not crit and not non_crit:
        lines.append("(none)")
    lines.append("")

    # Knowledge ambiguities
    ka = task.get("knowledge_ambiguity") or []
    lines.append("### Knowledge ambiguities")
    if ka:
        for entry in ka:
            lines.append(f"- {entry.get('term', '')}")
    else:
        lines.append("(none)")

    return "\n".join(lines)


def _walk_fields(prefix: str, blob) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(blob, dict):
        return out
    for k, v in blob.items():
        composite = f"{prefix}|{k}"
        if isinstance(v, str):
            out.append((composite, v))
        elif isinstance(v, dict):
            top = v.get("column_meaning")
            if isinstance(top, str) and top:
                out.append((composite, top))
            nested = v.get("fields_meaning")
            if isinstance(nested, dict):
                out.extend(_walk_fields(composite, nested))
    return out
