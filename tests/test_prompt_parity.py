"""Cross-framework prompt parity for c-interact mode.

All five adapters (claude_sdk, pydantic_ai, agno, mcp_agent, smolagents) must
emit byte-identical c-interact prompts so eval comparisons are meaningful.
The shared builders in `agents/_prompt_builders.py` are the single source of
truth; this test locks the contract in CI.
"""

import json

import pytest

from bird_interact_agents.config import settings
from bird_interact_agents.harness import SampleStatus, load_db_data_if_needed


_TASK = {
    "selected_database": "alien",
    "amb_user_query": "How many calibrated signals were observed last quarter?",
    "knowledge_ambiguity": [],
    "instance_id": "alien_1",
}
_BUDGET = 12.0
_SLAYER_DIR = "./slayer_models/alien"


@pytest.fixture(scope="module", autouse=True)
def _load_alien():
    load_db_data_if_needed("alien", settings.db_path)


async def _claude_sdk_prompt(query_mode: str, eval_mode: str) -> str:
    from bird_interact_agents.agents.claude_sdk import agent as cs

    cs._ctx_var.set({"slayer_storage_dir": _SLAYER_DIR})
    return await cs._build_prompt(query_mode, eval_mode, _TASK, _BUDGET)


async def _pydantic_ai_prompt(query_mode: str, eval_mode: str) -> str:
    from bird_interact_agents.agents.pydantic_ai.agent import (
        PydanticAIAgent,
        TaskDeps,
    )

    pa = PydanticAIAgent(slayer_storage_root="./slayer_models")
    deps = TaskDeps(
        status=SampleStatus(idx=0, original_data=_TASK),
        data_path_base=settings.db_path,
        slayer_storage_dir=_SLAYER_DIR,
    )
    return await pa._build_prompt(query_mode, eval_mode, _TASK, _BUDGET, deps)


def _make_state(module):
    return module.TaskState(
        status=SampleStatus(idx=0, original_data=_TASK),
        data_path_base=settings.db_path,
        slayer_storage_dir=_SLAYER_DIR,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )


async def _agno_prompt(query_mode: str, eval_mode: str) -> str:
    from bird_interact_agents.agents.agno import agent as ag

    return await ag._build_prompt(
        query_mode, eval_mode, _TASK, _BUDGET, _make_state(ag)
    )


async def _mcp_agent_prompt(query_mode: str, eval_mode: str) -> str:
    from bird_interact_agents.agents.mcp_agent import agent as mc

    return await mc._build_prompt(
        query_mode, eval_mode, _TASK, _BUDGET, _make_state(mc)
    )


async def _smolagents_prompt(query_mode: str, eval_mode: str) -> str:
    from bird_interact_agents.agents.smolagents import agent as sm

    return await sm._build_prompt(
        query_mode, eval_mode, _TASK, _BUDGET, _make_state(sm)
    )


_BUILDERS = [
    ("claude_sdk", _claude_sdk_prompt),
    ("pydantic_ai", _pydantic_ai_prompt),
    ("agno", _agno_prompt),
    ("mcp_agent", _mcp_agent_prompt),
    ("smolagents", _smolagents_prompt),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("eval_mode", ["c-interact"])
@pytest.mark.parametrize("query_mode", ["raw", "slayer"])
async def test_c_interact_prompts_are_byte_identical_across_adapters(
    query_mode: str, eval_mode: str
):
    prompts = {}
    for name, fn in _BUILDERS:
        prompts[name] = await fn(query_mode, eval_mode)

    reference = prompts["claude_sdk"]
    for name, p in prompts.items():
        assert p == reference, (
            f"{name} {query_mode}/{eval_mode} prompt diverges from claude_sdk"
        )


@pytest.mark.asyncio
async def test_raw_c_interact_knowledge_block_is_valid_json():
    """The knowledge section must be parseable JSON, not a markdown bullet list."""
    prompt = await _claude_sdk_prompt("raw", "c-interact")
    block = prompt.split("# External Knowledge\n", 1)[1]
    json_text, _, _ = block.partition("\nUser question:")
    json_text = json_text.strip()
    if json_text == "(no external knowledge available)":
        return
    parsed = json.loads(json_text)
    assert isinstance(parsed, list)


@pytest.mark.asyncio
async def test_slayer_c_interact_no_8_item_truncation():
    """Models summary must include all of a model's columns, not [:8]."""
    prompt = await _claude_sdk_prompt("slayer", "c-interact")
    assert "# Available models" in prompt
    summary_block = prompt.split("# Available models\n", 1)[1]
    summary_block = summary_block.split("\n# External knowledge", 1)[0]

    # New multi-line format from DEV-1362 Step 7:
    #   - signals
    #       columns:
    #         - <name>: <description>
    #         - …
    lines = summary_block.splitlines()
    try:
        signals_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "- signals")
    except StopIteration:
        pytest.fail("expected 'signals' model in summary")

    columns_idx = next(
        (i for i in range(signals_idx + 1, len(lines))
         if lines[i].strip() == "columns:"),
        None,
    )
    assert columns_idx is not None, "signals model has no `columns:` sub-section"

    column_count = 0
    for ln in lines[columns_idx + 1:]:
        stripped = ln.lstrip()
        if not stripped.startswith("- "):
            break  # left the columns block
        column_count += 1
    assert column_count > 8, (
        f"signals model rendered {column_count} columns — [:8] cap is back"
    )


@pytest.mark.asyncio
async def test_slayer_c_interact_includes_knowledge_section():
    """Slayer c-interact must include the {knowledge} placeholder rendering."""
    prompt = await _claude_sdk_prompt("slayer", "c-interact")
    assert "# External knowledge" in prompt
