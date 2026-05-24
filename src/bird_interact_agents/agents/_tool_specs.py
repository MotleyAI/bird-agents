"""Single-source-of-truth registry for the bird-interact and submit tools.

Every framework adapter (`claude_sdk`/`pydantic_ai`/`smolagents`/`agno`/
`mcp_agent`) used to carry its own copy of the seven raw-mode discovery
tools plus `submit_sql` / `submit_query` / `ask_user` — same body,
different decoration. This module collapses those into one declarative
registry so per-adapter files become a small loop that turns each
`ToolSpec` into the framework's tool object.

`render_action(spec, **kwargs)` builds the action string consumed by
`bird_interact_agents.harness.execute_env_action`, e.g.
`get_column_meaning('telescopes', 'bandusagepct')`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolParam(BaseModel):
    """A single tool parameter — name and Python type only.

    Type is intentionally restricted to a small set of strings; per-adapter
    tool builders look this up to construct framework-specific signatures.
    """

    name: str
    type_name: str = "str"
    description: str = ""


class ToolSpec(BaseModel):
    """Declarative tool definition shared across framework adapters."""

    name: str
    description: str
    parameters: list[ToolParam] = Field(default_factory=list)
    # Format string passed to `str.format(**kwargs)` to build the action
    # string consumed by `execute_env_action`. Examples:
    #   "execute({sql})"
    #   "get_column_meaning('{table_name}', '{column_name}')"
    action_template: str = ""


def render_action(spec: ToolSpec, **kwargs: str) -> str:
    """Render `spec.action_template` with the provided kwargs."""
    return spec.action_template.format(**kwargs)


# ---------------------------------------------------------------------------
# bird-interact raw-mode discovery tools
# ---------------------------------------------------------------------------

BIRD_INTERACT_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="execute_sql",
        description="Execute a SQL query against the database and return results.",
        parameters=[ToolParam(name="sql")],
        action_template="execute({sql})",
    ),
    ToolSpec(
        name="get_schema",
        description="Get the database schema (CREATE TABLE statements with sample data).",
        action_template="get_schema()",
    ),
    ToolSpec(
        name="get_all_column_meanings",
        description="Get the meanings/descriptions of all columns in the database.",
        action_template="get_all_column_meanings()",
    ),
    ToolSpec(
        name="get_column_meaning",
        description="Get the meaning of a specific column in a table.",
        parameters=[
            ToolParam(name="table_name"),
            ToolParam(name="column_name"),
        ],
        action_template="get_column_meaning('{table_name}', '{column_name}')",
    ),
    ToolSpec(
        name="get_all_external_knowledge_names",
        description="List all available external knowledge entry names for this database.",
        action_template="get_all_external_knowledge_names()",
    ),
    ToolSpec(
        name="get_knowledge_definition",
        description="Get the definition of a specific external knowledge entry.",
        parameters=[ToolParam(name="knowledge_name")],
        action_template="get_knowledge_definition('{knowledge_name}')",
    ),
    ToolSpec(
        name="get_all_knowledge_definitions",
        description="Get all external knowledge definitions for this database.",
        action_template="get_all_knowledge_definitions()",
    ),
]


# ---------------------------------------------------------------------------
# Submission + user-sim tool specs
# ---------------------------------------------------------------------------

SUBMIT_SQL_SPEC = ToolSpec(
    name="submit_sql",
    description=(
        "Submit your final SQL query for evaluation. Only submit when "
        "confident — submission ends the task."
    ),
    parameters=[ToolParam(name="sql")],
)

SUBMIT_QUERY_SPEC = ToolSpec(
    name="submit_query",
    description=(
        "Submit your final SLayer query for evaluation. `query_json` is a "
        "JSON string whose top-level value is EITHER (a) a single "
        'SlayerQuery object, e.g. {"source_model": "orders", '
        '"dimensions": ["status"], "measures": ["amount:sum"]}, OR '
        "(b) a nested-DAG array of stage objects — the same shape "
        '`query_nested` accepts — e.g. [{"name": "monthly", '
        '"source_model": "orders", "measures": ["amount:sum"], '
        '"time_dimensions": [{"dimension": "created_at", '
        '"granularity": "month"}]}, {"source_model": "monthly", '
        '"measures": ["amount:avg"]}]. In the array form the last '
        "element is the DAG root and every non-final element must have "
        '`name`. Do NOT wrap the array in `{"queries": ...}` — that '
        "shape is rejected. The chosen query is translated to SQL "
        "deterministically and tested against ground truth."
    ),
    parameters=[ToolParam(name="query_json")],
)

ASK_USER_SPEC = ToolSpec(
    name="ask_user",
    description="Ask the user a clarification question about their query.",
    parameters=[ToolParam(name="question")],
)
