"""DEV-1512 — mechanical propagation tests for the HOST DISCOVERY playbook.

Per `feedback_no_prompt_content_tests.md`: NO substring / anchor-phrase /
coverage assertions on prompt natural-language content. These tests only
verify mechanical contracts:

* Each prompts module defines a module-level `_HOST_DISCOVERY_PLAYBOOK`
  string constant that is substantive (non-stub).
* The constant's exact value is injected verbatim into every prompt that
  is supposed to carry it (claude_sdk_otf one-shot; pydantic_ai_otf_encode
  setup/kb/oneshot-constructor; pydantic_ai_recursive query-constructor).
* The constant contains no Python `.format()` placeholders (so call sites
  that don't supply playbook-specific kwargs cannot KeyError).

Behavioural validation that the playbook actually moves the agent's
host-choice toward the canonical 1-hop FK lives in the manual cloud
smoke on `museum_9` (claude_sdk_otf + pydantic_ai_otf_encode), per the
DEV-1512 plan §C. Stub-LLM behaviour tests are explicitly disallowed
by the same memory.
"""

from __future__ import annotations

import string

import yaml


_PLAYBOOK_ATTR = "_HOST_DISCOVERY_PLAYBOOK"
_SUBSTANTIVE_MIN_LEN = 100  # smoke check; not pinning a specific length


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
# 1. Per-file _HOST_DISCOVERY_PLAYBOOK constant exists and is substantive.
# ---------------------------------------------------------------------------


def test_claude_sdk_otf_prompts_defines_host_discovery_playbook():
    from bird_interact_agents.agents.claude_sdk_otf import prompts

    assert hasattr(prompts, _PLAYBOOK_ATTR), (
        f"claude_sdk_otf.prompts must define `{_PLAYBOOK_ATTR}` constant"
    )
    text = getattr(prompts, _PLAYBOOK_ATTR)
    assert isinstance(text, str)
    assert len(text) > _SUBSTANTIVE_MIN_LEN, (
        f"`{_PLAYBOOK_ATTR}` is too short ({len(text)} chars) — expected "
        f"a substantive playbook (>{_SUBSTANTIVE_MIN_LEN})."
    )


def test_pydantic_ai_otf_encode_prompts_defines_host_discovery_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    assert hasattr(prompts, _PLAYBOOK_ATTR)
    text = getattr(prompts, _PLAYBOOK_ATTR)
    assert isinstance(text, str)
    assert len(text) > _SUBSTANTIVE_MIN_LEN


def test_pydantic_ai_recursive_prompts_defines_host_discovery_playbook():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    assert hasattr(prompts, _PLAYBOOK_ATTR)
    text = getattr(prompts, _PLAYBOOK_ATTR)
    assert isinstance(text, str)
    assert len(text) > _SUBSTANTIVE_MIN_LEN


# ---------------------------------------------------------------------------
# 2. Playbook constant propagates verbatim into every render site.
#
# Each rendered prompt MUST contain the per-file _HOST_DISCOVERY_PLAYBOOK
# constant value as a literal substring. The TEXT of the constant isn't
# pinned anywhere — only its propagation. Rewriting the playbook
# automatically updates every render site; forgetting one site fails the
# test.
# ---------------------------------------------------------------------------


def test_claude_sdk_otf_one_shot_includes_playbook():
    from bird_interact_agents.agents.claude_sdk_otf import prompts

    rendered = prompts.SLAYER_OTF_ONE_SHOT.format(**_claude_sdk_otf_args())
    assert prompts._HOST_DISCOVERY_PLAYBOOK in rendered, (
        "SLAYER_OTF_ONE_SHOT does not include _HOST_DISCOVERY_PLAYBOOK"
    )


def test_kb_encoder_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.KB_ENCODER_PROMPT.format(**_kb_encoder_args())
    assert prompts._HOST_DISCOVERY_PLAYBOOK in rendered


def test_kb_encoder_oneshot_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.KB_ENCODER_ONESHOT_PROMPT.format(**_kb_encoder_args())
    assert prompts._HOST_DISCOVERY_PLAYBOOK in rendered


def test_setup_encoder_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.SETUP_ENCODER_PROMPT.format(**_setup_encoder_args())
    assert prompts._HOST_DISCOVERY_PLAYBOOK in rendered


def test_query_constructor_oneshot_prompt_includes_playbook():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    rendered = prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT.format(
        **_query_constructor_args()
    )
    assert prompts._HOST_DISCOVERY_PLAYBOOK in rendered


def test_query_constructor_prompt_includes_playbook():
    """The a-interact constructor — owned by pydantic_ai_recursive.prompts and
    re-exported via pydantic_ai_otf_encode.prompts (and used at agent.py:1301
    when not one-shot). Test against the source-of-truth module so the test
    passes regardless of re-export path."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    rendered = prompts.QUERY_CONSTRUCTOR_PROMPT.format(
        **_query_constructor_args()
    )
    assert prompts._HOST_DISCOVERY_PLAYBOOK in rendered


# ---------------------------------------------------------------------------
# 3. Playbook constant contains no Python format placeholders.
#
# The playbook is pure prose + literal SLayer tool-call syntax. Any
# accidental `{key}` token would expose a KeyError to call sites that
# don't supply that key. Literal curly braces in tool-call examples
# must be doubled (`{{ }}`) so they survive .format().
# ---------------------------------------------------------------------------


def _format_fields(text: str) -> set[str]:
    return {
        fname
        for _, fname, _, _ in string.Formatter().parse(text)
        if fname
    }


def test_claude_sdk_otf_playbook_has_no_format_fields():
    from bird_interact_agents.agents.claude_sdk_otf import prompts

    fields = _format_fields(prompts._HOST_DISCOVERY_PLAYBOOK)
    assert fields == set(), (
        f"_HOST_DISCOVERY_PLAYBOOK must contain no format fields; got {fields}"
    )


def test_pydantic_ai_otf_encode_playbook_has_no_format_fields():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    fields = _format_fields(prompts._HOST_DISCOVERY_PLAYBOOK)
    assert fields == set()


def test_pydantic_ai_recursive_playbook_has_no_format_fields():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    fields = _format_fields(prompts._HOST_DISCOVERY_PLAYBOOK)
    assert fields == set()


# ---------------------------------------------------------------------------
# 4. Format-placeholder integrity (existing rule, extended to new sites
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


def test_no_leftover_braces_query_constructor_oneshot():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import prompts

    args = _query_constructor_args()
    out = prompts.QUERY_CONSTRUCTOR_ONESHOT_PROMPT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out


def test_no_leftover_braces_query_constructor():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    args = _query_constructor_args()
    out = prompts.QUERY_CONSTRUCTOR_PROMPT.format(**args)
    for key in args:
        assert "{" + key + "}" not in out
