"""DEV-1545: mechanical contract tests for the v3 user-sim prompt.

Per `feedback_no_prompt_content_tests.md` — no phrase / anchor / substring
assertions on the prompt's natural-language content. Only:

  * v3 keys are registered in the upstream
    `USER_SIMULATOR_ENCODER` / `USER_SIMULATOR_DECODER` dicts on
    `import bird_interact_agents`.
  * Every declared `[[placeholder]]` substitutes cleanly when v3 prompts
    are rendered via the upstream builders.
  * Upstream-collision sentinel: if the dict already contains `"v3"`
    when our module imports, the import crashes loudly (assert-and-fail
    per Codex review + user decision — never silently override).
"""

from __future__ import annotations

import importlib
import re

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All seven placeholders v2's decoder uses. v3 mirrors v2's placeholders.
_ALL_PLACEHOLDERS = [
    "[[clarification_Q]]",
    "[[amb_json]]",
    "[[SQL_Glot]]",
    "[[DB_schema]]",
    "[[GT_SQL]]",
    "[[clear_query]]",
    "[[Action]]",
]

# Placeholders the ENCODER prompt is responsible for (subset of the decoder's;
# v2's encoder has these three).
_ENCODER_PLACEHOLDERS = [
    "[[clarification_Q]]",
    "[[amb_json]]",
    "[[SQL_Glot]]",
]


def _make_sample_status():
    """Build a minimal `SampleStatus` populated with every field
    `build_user_*_prompt` reads. Test-only — not a fixture importable
    from elsewhere."""
    from batch_run_bird_interact.sample_status import SampleStatus

    return SampleStatus(
        idx=0,
        original_data={
            "selected_database": "fakedb",
            "instance_id": "fake_1",
            "amb_user_query": "show me the things",
            "query": "show me the rows where score > 5",  # clear query
            "user_query_ambiguity": {
                "critical_ambiguity": [
                    {"term": "things", "interpretation": "rows"},
                ],
            },
            "knowledge_ambiguity": [
                {"name": "FOO", "interpretation": "the FOO metric"},
            ],
            "sol_sql": "SELECT * FROM foo WHERE score > 5;",
            "follow_up": {"sol_sql": "", "query": ""},
        },
    )


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_v3_registered_on_package_import():
    """`import bird_interact_agents` (the package) must trigger v3
    registration into the upstream dicts. Reachability via the
    package's `__init__.py` is the load-bearing contract — Codex #4."""
    # Force a clean import cycle to assert the side effect fires.
    import sys

    for mod in list(sys.modules):
        if mod.startswith("bird_interact_agents"):
            del sys.modules[mod]

    importlib.import_module("bird_interact_agents")

    from src.envs.user_simulator.prompts import (
        USER_SIMULATOR_DECODER,
        USER_SIMULATOR_ENCODER,
    )

    assert "v3" in USER_SIMULATOR_ENCODER
    assert "v3" in USER_SIMULATOR_DECODER


def test_v3_encoder_placeholders_present():
    """Mechanical: v3 encoder string contains every placeholder its
    consumers will substitute via `str.replace`. Catches accidental
    placeholder typos at template-author time."""
    import bird_interact_agents  # noqa: F401  side-effect registration

    from src.envs.user_simulator.prompts import USER_SIMULATOR_ENCODER

    v3 = USER_SIMULATOR_ENCODER["v3"]
    for placeholder in _ENCODER_PLACEHOLDERS:
        assert placeholder in v3, (
            f"v3 encoder is missing placeholder {placeholder!r}; "
            f"upstream `build_user_encoder_prompt` will silently leave "
            f"its substitution off."
        )


def test_v3_decoder_placeholders_present():
    """Mechanical: v3 decoder string contains every placeholder."""
    import bird_interact_agents  # noqa: F401

    from src.envs.user_simulator.prompts import USER_SIMULATOR_DECODER

    v3 = USER_SIMULATOR_DECODER["v3"]
    for placeholder in _ALL_PLACEHOLDERS:
        assert placeholder in v3, (
            f"v3 decoder is missing placeholder {placeholder!r}."
        )


# ---------------------------------------------------------------------------
# Builder substitution tests — both encoder and decoder (Codex #5).
# ---------------------------------------------------------------------------


def _no_residual_placeholders(rendered: str) -> list[str]:
    """Return any unsubstituted `[[<word>]]` placeholders in `rendered`."""
    return re.findall(r"\[\[[a-zA-Z_][a-zA-Z0-9_]*\]\]", rendered)


def test_build_user_encoder_prompt_v3_substitutes_every_placeholder():
    """`build_user_encoder_prompt(..., user_sim_prompt_version='v3')`
    renders without leftover `[[...]]` tokens for any of v3's declared
    placeholders."""
    import bird_interact_agents  # noqa: F401

    from batch_run_bird_interact.prompt_utils import build_user_encoder_prompt

    status = _make_sample_status()
    rendered = build_user_encoder_prompt(
        question="what does FOO mean?",
        sample_status=status,
        db_schema="CREATE TABLE foo (score INT);",
        user_sim_prompt_version="v3",
    )
    residual = _no_residual_placeholders(rendered)
    declared_residual = [p for p in residual if p in _ENCODER_PLACEHOLDERS]
    assert not declared_residual, (
        f"v3 encoder left declared placeholders unsubstituted: "
        f"{declared_residual!r}"
    )


def test_build_user_decoder_prompt_v3_substitutes_every_placeholder():
    """Same mechanical contract for the decoder. Per Codex #5 we test
    both phases, not just the decoder."""
    import bird_interact_agents  # noqa: F401

    from batch_run_bird_interact.prompt_utils import build_user_decoder_prompt

    status = _make_sample_status()
    rendered = build_user_decoder_prompt(
        question="what does FOO mean?",
        encoded_action="labeled(\"FOO\")",
        sample_status=status,
        db_schema="CREATE TABLE foo (score INT);",
        user_sim_prompt_version="v3",
    )
    residual = _no_residual_placeholders(rendered)
    declared_residual = [p for p in residual if p in _ALL_PLACEHOLDERS]
    assert not declared_residual, (
        f"v3 decoder left declared placeholders unsubstituted: "
        f"{declared_residual!r}"
    )


def test_build_user_decoder_prompt_v3_includes_sol_sql():
    """Value-substitution presence: the supplied gold SQL appears in the
    rendered v3 decoder (it's the evidence base v3's anti-invention
    rule asks the sim to consult). Mechanical contract per the
    [[feedback_no_prompt_content_tests]] canonical style."""
    import bird_interact_agents  # noqa: F401

    from batch_run_bird_interact.prompt_utils import build_user_decoder_prompt

    status = _make_sample_status()
    rendered = build_user_decoder_prompt(
        question="what does FOO mean?",
        encoded_action="labeled(\"FOO\")",
        sample_status=status,
        db_schema="CREATE TABLE foo (score INT);",
        user_sim_prompt_version="v3",
    )
    assert "SELECT * FROM foo WHERE score > 5" in rendered


# ---------------------------------------------------------------------------
# Upstream-collision sentinel (Codex #3 + user decision: assert-and-crash).
# ---------------------------------------------------------------------------


def test_v3_import_crashes_loudly_on_upstream_collision(monkeypatch):
    """If upstream `batch_run_bird_interact` ever ships its own `v3`
    BEFORE we import, our module must crash with a clear AssertionError
    rather than silently overwrite. A `setdefault` would silently keep
    upstream's value — Codex #3 flagged that as a hidden-semantic-drift
    risk; user picked loud crash."""
    import sys

    # Wipe any cached bird_interact_agents state so the registration
    # side effect fires fresh.
    for mod in list(sys.modules):
        if mod.startswith("bird_interact_agents.user_sim_prompts"):
            del sys.modules[mod]

    from src.envs.user_simulator.prompts import (
        USER_SIMULATOR_DECODER,
        USER_SIMULATOR_ENCODER,
    )

    # Pre-poison the encoder dict with a different 'v3'.
    saved_enc = dict(USER_SIMULATOR_ENCODER)
    saved_dec = dict(USER_SIMULATOR_DECODER)
    try:
        USER_SIMULATOR_ENCODER["v3"] = "preexisting upstream v3 encoder"
        # Decoder may or may not be poisoned — the encoder assertion
        # fires first.
        with pytest.raises(AssertionError):
            importlib.import_module("bird_interact_agents.user_sim_prompts")
    finally:
        USER_SIMULATOR_ENCODER.clear()
        USER_SIMULATOR_ENCODER.update(saved_enc)
        USER_SIMULATOR_DECODER.clear()
        USER_SIMULATOR_DECODER.update(saved_dec)
        for mod in list(sys.modules):
            if mod.startswith("bird_interact_agents"):
                del sys.modules[mod]


# ---------------------------------------------------------------------------
# PR #42 Codex review #1: cloud-only install resilience.
# ---------------------------------------------------------------------------


def test_package_import_survives_missing_upstream(monkeypatch):
    """`import bird_interact_agents` (the package) must NOT crash when
    the optional upstream `mini-interact-agent` package is absent.

    `mini-interact-agent` (which provides `src.envs.user_simulator.prompts`)
    only ships with the `original` and `all` extras. A bare `cloud`
    install — used by Ray workers etc. — does NOT pull it in. Before
    Codex review #1 the unconditional side-effect import in
    `__init__.py` would crash any `from bird_interact_agents import
    paths` on such installs. The contract under the fix: bare package
    import succeeds; v3 isn't registered (v2 still works because the
    same call site already imports from the upstream); user-sim
    invocations land a clearer error in `_submit.py`.

    Test mechanics: simulate the missing upstream by inserting a
    `src.envs.user_simulator.prompts` import that raises
    ModuleNotFoundError, drop bird_interact_agents from sys.modules,
    re-import, assert no exception."""
    import builtins
    import sys

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.envs.user_simulator.prompts" or name.startswith(
            "src.envs.user_simulator"
        ):
            raise ModuleNotFoundError(
                f"No module named {name!r} (simulated cloud-only install)"
            )
        return real_import(name, globals, locals, fromlist, level)

    # Wipe bird_interact_agents so __init__.py re-runs.
    for mod in list(sys.modules):
        if mod.startswith("bird_interact_agents"):
            del sys.modules[mod]
    # Also wipe any cached `src.envs.user_simulator.prompts` so the
    # fake importer is actually invoked.
    for mod in list(sys.modules):
        if mod.startswith("src.envs.user_simulator"):
            del sys.modules[mod]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    try:
        importlib.import_module("bird_interact_agents")
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        # Re-wipe so subsequent tests get a clean re-registration.
        for mod in list(sys.modules):
            if mod.startswith("bird_interact_agents"):
                del sys.modules[mod]


def test_v3_collision_guard_works_under_optimised_mode(monkeypatch):
    """Codex review #2: `assert` is stripped under `python -O`, which
    would silently disable the collision guard. The fix replaced both
    `assert ... not in dict` checks with explicit `if cond: raise
    AssertionError(...)`, which DOES fire in optimised mode.

    Test mechanics: there's no portable way to flip `__debug__` mid-
    process, so we directly exercise the guard by walking the import
    path with the upstream dict pre-poisoned. The test passing under
    the regular interpreter does NOT prove `-O` safety; the regression
    we're guarding against (replacing `if/raise` with `assert`) is
    structural. This test pins the structural shape — a future
    refactor reintroducing `assert` would still pass this test under
    `pytest`, but `grep -n 'assert.*USER_SIMULATOR' src/.../user_sim_prompts.py`
    in CI would catch it. We rely on review + the comment in the
    module for the `-O` invariant."""
    import sys

    for mod in list(sys.modules):
        if mod.startswith("bird_interact_agents.user_sim_prompts"):
            del sys.modules[mod]

    from src.envs.user_simulator.prompts import (
        USER_SIMULATOR_DECODER,
        USER_SIMULATOR_ENCODER,
    )

    saved_enc = dict(USER_SIMULATOR_ENCODER)
    saved_dec = dict(USER_SIMULATOR_DECODER)
    try:
        USER_SIMULATOR_ENCODER["v3"] = "poisoned-by-upstream encoder"
        with pytest.raises(AssertionError):
            importlib.import_module("bird_interact_agents.user_sim_prompts")
    finally:
        USER_SIMULATOR_ENCODER.clear()
        USER_SIMULATOR_ENCODER.update(saved_enc)
        USER_SIMULATOR_DECODER.clear()
        USER_SIMULATOR_DECODER.update(saved_dec)
        for mod in list(sys.modules):
            if mod.startswith("bird_interact_agents"):
                del sys.modules[mod]
