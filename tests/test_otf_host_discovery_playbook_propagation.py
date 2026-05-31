"""DEV-1512 — mechanical propagation tests for the HOST DISCOVERY playbook.

Per `feedback_no_prompt_content_tests.md`: NO substring / anchor-phrase /
coverage assertions on prompt natural-language content. These tests only
verify mechanical contracts:

* There is a SINGLE canonical `HOST_DISCOVERY_PLAYBOOK` constant in
  `bird_interact_agents.agents._host_discovery_playbook`; the three
  prompts modules that need it import the same object (`is`-equal — no
  drift possible by construction).
* The canonical constant contains no Python `.format()` placeholders.
* The constant is injected into every render site that should carry it.

Behavioural validation that the playbook actually moves the agent's
host-choice toward the canonical 1-hop FK lives in the manual cloud
smoke on `museum_9`, per the DEV-1512 plan §C. Stub-LLM behaviour
tests are explicitly disallowed by the same memory.
"""

from __future__ import annotations

import string

import yaml

from bird_interact_agents.agents._host_discovery_playbook import (
    HOST_DISCOVERY_PLAYBOOK,
)


def _claude_sdk_otf_args() -> dict:
    return dict(budget=100, db_name="tinydb", user_query="?")


def _kb_encoder_args() -> dict:
    return dict(
        db_name="tinydb",
        kb_id=5,
        kb_row_yaml=yaml.safe_dump(
            {"id": 5, "knowledge": "test", "definition": "x"},
            sort_keys=False,
        ),
        deps_block="(none)",
        budget=100.0,
        existing_kb_tagged_entities_block="(none)",
    )


def _setup_encoder_args() -> dict:
    return dict(
        db_name="tinydb",
        kb_id=6,
        kb_body="KB 6 — Dwelling Type\n\nKB item (verbatim):\nid: 6",
        deps_block="(none)",
        reverse_deps_block="(none)",
        existing_kb_tagged_entities_block="(none)",
    )


def _query_constructor_args() -> dict:
    return dict(
        amb_user_query="how many?",
        spec="(spec)",
        confirmed_projection="  1. col_a\n  2. col_b",
        budget=100.0,
        db_name="tinydb",
    )


# ---------------------------------------------------------------------------
# 1. Single canonical source + drift-impossible identity.
# ---------------------------------------------------------------------------


def test_canonical_playbook_is_substantive():
    assert isinstance(HOST_DISCOVERY_PLAYBOOK, str)
    assert len(HOST_DISCOVERY_PLAYBOOK) > 100, (
        f"HOST_DISCOVERY_PLAYBOOK is too short ({len(HOST_DISCOVERY_PLAYBOOK)} "
        "chars) — expected a substantive playbook (>100)."
    )


def test_canonical_playbook_has_no_format_fields():
    """The playbook is concatenated into prompts that go through
    .format() with caller-supplied kwargs. Any {key} in the playbook
    would expose a KeyError to call sites that don't pass it."""
    fields = {
        fname
        for _, fname, _, _ in string.Formatter().parse(HOST_DISCOVERY_PLAYBOOK)
        if fname
    }
    assert fields == set(), (
        f"HOST_DISCOVERY_PLAYBOOK must contain no format fields; got {fields}"
    )


def test_three_prompts_modules_import_the_same_object():
    """Each of the three OTF prompts modules re-exports the canonical
    playbook as a private alias `_HOST_DISCOVERY_PLAYBOOK`. They MUST
    be the same object — drift is impossible by construction."""
    from bird_interact_agents.agents.claude_sdk_otf import prompts as cs_otf
    from bird_interact_agents.agents.pydantic_ai_otf_encode import (
        prompts as otf_encode,
    )
    from bird_interact_agents.agents.pydantic_ai_recursive import (
        prompts as recursive,
    )

    assert cs_otf._HOST_DISCOVERY_PLAYBOOK is HOST_DISCOVERY_PLAYBOOK
    assert otf_encode._HOST_DISCOVERY_PLAYBOOK is HOST_DISCOVERY_PLAYBOOK
    assert recursive._HOST_DISCOVERY_PLAYBOOK is HOST_DISCOVERY_PLAYBOOK


# ---------------------------------------------------------------------------
# 2. Playbook is injected into every named render site.
# ---------------------------------------------------------------------------


def test_claude_sdk_otf_one_shot_includes_playbook():
    from bird_interact_agents.agents.claude_sdk_otf import prompts

    rendered = prompts.SLAYER_OTF_ONE_SHOT.format(**_claude_sdk_otf_args())
    assert HOST_DISCOVERY_PLAYBOOK in rendered


def test_kb_encoder_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.KB_ENCODER_PROMPT.format(**_kb_encoder_args())
    assert HOST_DISCOVERY_PLAYBOOK in rendered


def test_kb_encoder_oneshot_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.KB_ENCODER_ONESHOT_PROMPT.format(**_kb_encoder_args())
    assert HOST_DISCOVERY_PLAYBOOK in rendered


def test_setup_encoder_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.SETUP_ENCODER_PROMPT.format(**_setup_encoder_args())
    assert HOST_DISCOVERY_PLAYBOOK in rendered


def test_query_constructor_oneshot_prompt_otf_encode_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT.format(
        **_query_constructor_args()
    )
    assert HOST_DISCOVERY_PLAYBOOK in rendered


def test_query_constructor_prompt_recursive_includes_playbook():
    """The a-interact constructor, owned by pydantic_ai_recursive.prompts
    and re-exported via pydantic_ai_otf_encode.prompts (used at
    pydantic_ai_otf_encode/agent.py:1301 when not one-shot)."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    rendered = prompts.QUERY_CONSTRUCTOR_PROMPT.format(
        **_query_constructor_args()
    )
    assert HOST_DISCOVERY_PLAYBOOK in rendered


def test_query_constructor_oneshot_prompt_recursive_includes_playbook():
    """The recursive one-shot constructor — selected by
    pydantic_ai_recursive.agent:614 when eval_mode == 'one-shot'. Must
    also carry the playbook so the autonomous-decision path gets the
    same host-discovery guidance as the a-interact path."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    rendered = prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT.format(
        **_query_constructor_args()
    )
    assert HOST_DISCOVERY_PLAYBOOK in rendered


# ---------------------------------------------------------------------------
# 3. Format-placeholder integrity (existing rule, extended to new sites
#    if any new placeholders were introduced — the plan introduces none).
# ---------------------------------------------------------------------------


def test_no_leftover_braces_slayer_otf_one_shot():
    from bird_interact_agents.agents.claude_sdk_otf import prompts

    args = _claude_sdk_otf_args()
    out = prompts.SLAYER_OTF_ONE_SHOT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out


def test_no_leftover_braces_kb_encoder_oneshot():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    args = _kb_encoder_args()
    out = prompts.KB_ENCODER_ONESHOT_PROMPT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out


def test_no_leftover_braces_query_constructor_oneshot_otf_encode():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    args = _query_constructor_args()
    out = prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out


def test_no_leftover_braces_query_constructor_recursive():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    args = _query_constructor_args()
    out = prompts.QUERY_CONSTRUCTOR_PROMPT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out


def test_no_leftover_braces_query_constructor_oneshot_recursive():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    args = _query_constructor_args()
    out = prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out
