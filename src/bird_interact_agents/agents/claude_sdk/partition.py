"""DEV-1555 Stage 1: discovery/main subagent partition helpers.

The four ``claude_sdk_otf*`` agents split each task into a ``discovery``
subagent (introspection + user clarification, its own context window) and
the main loop (encode / query / submit on a slim context). The split keeps
peak context inside open-weight model windows (~260K): the dominant context
consumers are introspection tool outputs, which now stay in the subagent's
context and reach the main loop only as a handoff report.

Enforcement model (verified against the Claude Code permission docs):

* ``ClaudeAgentOptions.disallowed_tools`` is GLOBAL — it blocks a tool
  inside subagents too, so it cannot express "main loop may not, subagent
  may". Both partitions therefore stay in ``allowed_tools``.
* ``AgentDefinition.tools`` restricts what the subagent sees.
* The main-loop block is the ``partition_deny`` PreToolUse hook below:
  hook inputs carry ``agent_id`` only when the call originates inside a
  subagent, so a missing ``agent_id`` means "main loop" and the call is
  denied with a redirect to the discovery subagent.
* ``ClaudeAgentOptions.max_turns`` caps only the MAIN loop; the discovery
  subagent gets its own ``maxTurns`` (``DISCOVERY_MAX_TURNS``). The
  combined soft turn budget is still tracked by the turn-budget warning
  hook, which fires for subagent calls as well.
"""

from __future__ import annotations

from bird_interact_agents.harness import MAX_MODEL_TURNS

DISCOVERY_AGENT_NAME = "discovery"

# Hard cap for one discovery invocation. The main loop keeps the existing
# 2x cap (`_MAX_TURNS`); a single introspection sweep needs far less.
DISCOVERY_MAX_TURNS = MAX_MODEL_TURNS


def make_partition_deny_hook(discovery_only_tools):
    """Build the PreToolUse hook enforcing the main-loop side of the split.

    ``discovery_only_tools`` is the set of full tool names available ONLY
    inside the discovery subagent. Calls without ``agent_id`` in the hook
    input originate in the main loop and are denied with a redirect.
    """
    blocked = frozenset(discovery_only_tools)

    async def partition_deny(input_data, tool_use_id, context):
        if input_data.get("agent_id"):
            return {}
        tool_name = input_data.get("tool_name") or ""
        if tool_name not in blocked:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"{tool_name} is reserved for the '{DISCOVERY_AGENT_NAME}' "
                    "subagent. Delegate introspection to it via the Task tool "
                    f"(subagent_type='{DISCOVERY_AGENT_NAME}') and work from "
                    "its handoff report instead of calling this tool directly."
                ),
            }
        }

    return partition_deny


_DISCOVERY_PROMPT_COMMON = """\
You are the discovery subagent for a data-analysis task. Your ONLY job is
to gather everything the main agent needs to answer the user's question,
then hand it back as a single structured report. You cannot submit answers
and you must not try to solve the task yourself.

Produce a handoff report with EXACTLY these sections:

1. RELEVANT ENTITIES — every table/model plausibly needed, with a one-line
   purpose each.
2. COLUMNS — for each relevant column: name, description, observed sample
   values (quote them verbatim; flag mismatches between documented enums
   and actual values).
3. JOIN PATHS — how the relevant entities connect (direction and keys).
4. KNOWLEDGE-BASE ITEMS — every KB item that bears on the question, with
   its definition quoted VERBATIM (formulas must not be paraphrased).
5. USER CLARIFICATIONS — every question you asked and the answer, quoted
   verbatim as Q→A pairs.
6. OPEN AMBIGUITIES — anything you could not resolve, with the candidate
   interpretations.

Be exhaustive in coverage but terse in prose: the report replaces raw tool
output in the main agent's context, so include facts, not narration.
"""

_DISCOVERY_PROMPT_ASK_USER = """
The user simulator holds masked ground truth that is unrecoverable from
the KB alone. Identify the most load-bearing ambiguities in the request
and resolve them with ask_user BEFORE writing the report; record each
exchange verbatim in section 5.
"""


def build_discovery_prompt(*, with_ask_user: bool) -> str:
    """Compose the discovery subagent system prompt."""
    if with_ask_user:
        return _DISCOVERY_PROMPT_COMMON + _DISCOVERY_PROMPT_ASK_USER
    return _DISCOVERY_PROMPT_COMMON


MAIN_WORKFLOW_NOTE = f"""

## Subagent workflow (mandatory)

Schema/data introspection tools are NOT available to you directly — they
live in the '{DISCOVERY_AGENT_NAME}' subagent. Start by delegating to it
via the Task tool (subagent_type='{DISCOVERY_AGENT_NAME}') and wait for
its handoff report; work from that report. If you later need more
introspection or another user clarification you cannot ask yourself,
spawn '{DISCOVERY_AGENT_NAME}' again with a focused follow-up request
rather than guessing.
"""
