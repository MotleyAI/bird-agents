"""Single-source-of-truth tool registry — the per-adapter raw-mode
discovery tools all collapse to one specification list."""

from __future__ import annotations

import pytest


def test_bird_interact_tools_covers_all_seven_discovery_names():
    from bird_interact_agents.agents._tool_specs import BIRD_INTERACT_TOOLS

    names = {t.name for t in BIRD_INTERACT_TOOLS}
    assert names == {
        "execute_sql",
        "get_schema",
        "get_all_column_meanings",
        "get_column_meaning",
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
    }


def test_render_action_no_params():
    from bird_interact_agents.agents._tool_specs import (
        BIRD_INTERACT_TOOLS, render_action,
    )

    by_name = {t.name: t for t in BIRD_INTERACT_TOOLS}
    assert render_action(by_name["get_schema"]) == "get_schema()"
    assert (
        render_action(by_name["get_all_column_meanings"])
        == "get_all_column_meanings()"
    )
    assert (
        render_action(by_name["get_all_external_knowledge_names"])
        == "get_all_external_knowledge_names()"
    )
    assert (
        render_action(by_name["get_all_knowledge_definitions"])
        == "get_all_knowledge_definitions()"
    )


def test_render_action_unquoted_sql_param():
    from bird_interact_agents.agents._tool_specs import (
        BIRD_INTERACT_TOOLS, render_action,
    )

    by_name = {t.name: t for t in BIRD_INTERACT_TOOLS}
    sql = "SELECT * FROM telescopes WHERE id = 1"
    assert render_action(by_name["execute_sql"], sql=sql) == f"execute({sql})"


def test_render_action_quoted_string_params():
    from bird_interact_agents.agents._tool_specs import (
        BIRD_INTERACT_TOOLS, render_action,
    )

    by_name = {t.name: t for t in BIRD_INTERACT_TOOLS}
    assert (
        render_action(
            by_name["get_column_meaning"],
            table_name="telescopes", column_name="bandusagepct",
        )
        == "get_column_meaning('telescopes', 'bandusagepct')"
    )
    assert (
        render_action(
            by_name["get_knowledge_definition"], knowledge_name="some_kb_term",
        )
        == "get_knowledge_definition('some_kb_term')"
    )


def test_each_spec_carries_description_and_typed_params():
    from bird_interact_agents.agents._tool_specs import BIRD_INTERACT_TOOLS

    for spec in BIRD_INTERACT_TOOLS:
        assert spec.name
        assert spec.description, f"missing description on {spec.name}"
        for p in spec.parameters:
            assert p.name
            assert p.type_name == "str", (
                f"only str params expected on bird-interact discovery tools; "
                f"got {p.type_name} on {spec.name}.{p.name}"
            )


def test_submit_specs_have_expected_signatures():
    """submit_sql takes one `sql: str`; submit_query takes one
    `query_json: str`."""
    from bird_interact_agents.agents._tool_specs import (
        SUBMIT_SQL_SPEC, SUBMIT_QUERY_SPEC,
    )

    assert SUBMIT_SQL_SPEC.name == "submit_sql"
    assert [p.name for p in SUBMIT_SQL_SPEC.parameters] == ["sql"]

    assert SUBMIT_QUERY_SPEC.name == "submit_query"
    assert [p.name for p in SUBMIT_QUERY_SPEC.parameters] == ["query_json"]


def test_ask_user_spec():
    from bird_interact_agents.agents._tool_specs import ASK_USER_SPEC

    assert ASK_USER_SPEC.name == "ask_user"
    assert [p.name for p in ASK_USER_SPEC.parameters] == ["question"]


def test_render_action_unknown_kwarg_raises():
    from bird_interact_agents.agents._tool_specs import (
        BIRD_INTERACT_TOOLS, render_action,
    )

    by_name = {t.name: t for t in BIRD_INTERACT_TOOLS}
    with pytest.raises((KeyError, ValueError)):
        # missing required kwarg
        render_action(by_name["execute_sql"])
