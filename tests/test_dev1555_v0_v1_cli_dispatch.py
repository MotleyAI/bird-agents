"""DEV-1555 v0/v1 split — CLI token acceptance + dispatch routing.

Asserts:

1. The ten v0+v1 framework tokens are accepted by both
   ``bird-interact run`` (``run.main`` argparse) and
   ``bird-interact-cloud submit`` (``cloud.cli.parse_args``).
2. The dispatcher in ``run.make_runner`` routes each token to the
   correct agent class — v0 tokens to the unsuffixed v0 packages, v1
   tokens to the ``_v1`` packages.
3. The aggregator tokens (``claude_sdk`` → v0, ``claude_sdk_v1`` → v1)
   pick the right variant for each
   ``(benchmark.one_shot × query_mode)`` cell.
"""

from __future__ import annotations

from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# All ten framework tokens (4 v0 variants + v0 aggregator + 4 v1 variants
# + v1 aggregator).
# ---------------------------------------------------------------------------

_V0_VARIANT_TOKENS = (
    "claude_sdk_otf",
    "claude_sdk_otf_raw",
    "claude_sdk_otf_ainteract",
    "claude_sdk_otf_ainteract_raw",
)

_V1_VARIANT_TOKENS = (
    "claude_sdk_otf_v1",
    "claude_sdk_otf_raw_v1",
    "claude_sdk_otf_ainteract_v1",
    "claude_sdk_otf_ainteract_raw_v1",
)

_AGGREGATOR_TOKENS = ("claude_sdk", "claude_sdk_v1")

_ALL_TOKENS = _V0_VARIANT_TOKENS + _V1_VARIANT_TOKENS + _AGGREGATOR_TOKENS


# ---------------------------------------------------------------------------
# (1a) `bird-interact run` (run.main argparse) accepts every token.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", _ALL_TOKENS)
def test_run_cli_accepts_framework_token(token: str):
    """``run.main`` --framework choices include this token."""
    # Build a parser by introspecting the same choices the code uses.
    # We hit `_validate_framework_mode`'s underlying argparse via a
    # source-level introspection — the cleanest probe without invoking
    # the full main(): the choices list must include the token.
    import bird_interact_agents.run as run_module

    # Find the line `choices=[...]` for the --framework arg, eval it.
    import inspect

    src = inspect.getsource(run_module)
    # Locate the --framework block and extract its `choices=[...]`
    # list literal.
    import re

    m = re.search(
        r'add_argument\(\s*"--framework"[^)]*?choices\s*=\s*(\[[^\]]+\])',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "run.main --framework argparse block not found or choices= missing."
    )
    choices = eval(m.group(1))
    assert token in choices, (
        f"--framework choices in run.main is missing '{token}'. "
        f"Current choices: {choices}"
    )


# ---------------------------------------------------------------------------
# (1b) `bird-interact-cloud submit` accepts every token — exercised via the
# real `cli.parse_args` so a parser-construction change is caught directly.
# ---------------------------------------------------------------------------


def _cloud_submit_argv_for(framework: str) -> list[str]:
    return [
        "submit",
        "--framework", framework,
        "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1",
        "--mode", "one-shot",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--no-require-annotation",
        "--no-subscription-auth",
    ]


@pytest.mark.parametrize("token", _ALL_TOKENS)
def test_cloud_cli_accepts_framework_token(token: str):
    """``cloud.cli.parse_args(['submit', '--framework', token, …])`` accepts every token."""
    from bird_interact_agents.cloud import cli as cli_mod

    ns = cli_mod.parse_args(_cloud_submit_argv_for(token))
    assert ns.framework == token, (
        f"cloud.cli parse_args returned framework={ns.framework!r}, "
        f"expected {token!r}"
    )


# ---------------------------------------------------------------------------
# (2) Per-variant token dispatch: each v0/v1 token routes to the right class.
# ---------------------------------------------------------------------------


def _common_kwargs() -> dict:
    """Shared kwargs for `make_runner` that aren't varied by these tests."""
    return dict(
        dataset="livesqlbench-base-lite-sqlite",
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=True,
        max_depth=3,
        slayer_storage_root="/tmp/test-slayer-storage",
        slayer_setup="on-the-fly",
        reasoning_effort=None,
    )


def _spy_class_for(module_path: str, attr: str) -> mock.MagicMock:
    """Build a MagicMock that can be patched in as the agent class."""
    spy = mock.MagicMock(spec_set=())
    # Make the spy instance look enough like an agent to capture the run_task
    # closure later — but the tests don't actually invoke run_task.
    spy.return_value.run_task = mock.AsyncMock()
    return spy


@pytest.mark.parametrize(
    "framework,target_module,target_class",
    [
        # v0 single-variant tokens → unsuffixed v0 packages
        (
            "claude_sdk_otf",
            "bird_interact_agents.agents.claude_sdk_otf",
            "ClaudeSDKOtfAgent",
        ),
        (
            "claude_sdk_otf_raw",
            "bird_interact_agents.agents.claude_sdk_otf_raw",
            "ClaudeSDKOtfRawAgent",
        ),
        (
            "claude_sdk_otf_ainteract",
            "bird_interact_agents.agents.claude_sdk_otf_ainteract",
            "ClaudeSDKOtfAInteractAgent",
        ),
        (
            "claude_sdk_otf_ainteract_raw",
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
            "ClaudeSDKOtfAInteractRawAgent",
        ),
        # v1 single-variant tokens → _v1 packages
        (
            "claude_sdk_otf_v1",
            "bird_interact_agents.agents.claude_sdk_otf_v1",
            "ClaudeSDKOtfAgent",
        ),
        (
            "claude_sdk_otf_raw_v1",
            "bird_interact_agents.agents.claude_sdk_otf_raw_v1",
            "ClaudeSDKOtfRawAgent",
        ),
        (
            "claude_sdk_otf_ainteract_v1",
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1",
            "ClaudeSDKOtfAInteractAgent",
        ),
        (
            "claude_sdk_otf_ainteract_raw_v1",
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1",
            "ClaudeSDKOtfAInteractRawAgent",
        ),
    ],
)
def test_per_variant_dispatch(framework, target_module, target_class):
    """Framework token routes to the expected class in the expected package."""
    import bird_interact_agents.run as run_module

    spy = _spy_class_for(target_module, target_class)
    # Mode and query-mode chosen to match the variant: ainteract → a-interact;
    # one-shot variants → one-shot; raw vs slayer query mode follows the name.
    mode = "a-interact" if "ainteract" in framework else "one-shot"
    query_mode = "raw" if framework.endswith("_raw") or framework.endswith(
        "_raw_v1"
    ) else "slayer"

    with mock.patch(f"{target_module}.{target_class}", spy):
        run_module.make_runner(
            framework=framework, mode=mode, query_mode=query_mode,
            **_common_kwargs(),
        )

    assert spy.called, (
        f"--framework {framework} did not instantiate {target_module}.{target_class}."
    )


# ---------------------------------------------------------------------------
# (3) Aggregator tokens route to the right variant for each
# `(one_shot × query_mode)` cell.
# ---------------------------------------------------------------------------


# `one_shot=True` is signalled by `mode != "a-interact"` AND by
# `benchmark.one_shot`. The aggregator branch reads `b.one_shot` from
# `get_benchmark(dataset)` — so we need a dataset whose .one_shot matches.
# We pick: livesqlbench-base-lite-sqlite (one_shot=True) for the one_shot
# rows, and mini-interact (one_shot=False) for the a-interact rows.


_AGGREGATOR_CELLS = [
    # (framework_aggregator, dataset, mode, query_mode,
    #  expected_target_module, expected_class)
    (
        "claude_sdk", "livesqlbench-base-lite-sqlite", "one-shot", "slayer",
        "bird_interact_agents.agents.claude_sdk_otf", "ClaudeSDKOtfAgent",
    ),
    (
        "claude_sdk", "livesqlbench-base-lite-sqlite", "one-shot", "raw",
        "bird_interact_agents.agents.claude_sdk_otf_raw",
        "ClaudeSDKOtfRawAgent",
    ),
    (
        "claude_sdk", "mini-interact", "a-interact", "slayer",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract",
        "ClaudeSDKOtfAInteractAgent",
    ),
    (
        "claude_sdk", "mini-interact", "a-interact", "raw",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
        "ClaudeSDKOtfAInteractRawAgent",
    ),
    (
        "claude_sdk_v1", "livesqlbench-base-lite-sqlite", "one-shot", "slayer",
        "bird_interact_agents.agents.claude_sdk_otf_v1", "ClaudeSDKOtfAgent",
    ),
    (
        "claude_sdk_v1", "livesqlbench-base-lite-sqlite", "one-shot", "raw",
        "bird_interact_agents.agents.claude_sdk_otf_raw_v1",
        "ClaudeSDKOtfRawAgent",
    ),
    (
        "claude_sdk_v1", "mini-interact", "a-interact", "slayer",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1",
        "ClaudeSDKOtfAInteractAgent",
    ),
    (
        "claude_sdk_v1", "mini-interact", "a-interact", "raw",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1",
        "ClaudeSDKOtfAInteractRawAgent",
    ),
]


@pytest.mark.parametrize(
    "framework,dataset,mode,query_mode,target_module,target_class",
    _AGGREGATOR_CELLS,
)
def test_aggregator_dispatch_routes_to_right_variant(
    framework, dataset, mode, query_mode, target_module, target_class
):
    """``claude_sdk`` and ``claude_sdk_v1`` route correctly for each cell.

    Without this test, an aggregator wired to the wrong sibling dir would
    pass parser-acceptance tests but silently run the wrong agent.
    """
    import bird_interact_agents.run as run_module

    spy = _spy_class_for(target_module, target_class)
    kwargs = _common_kwargs()
    kwargs["dataset"] = dataset

    with mock.patch(f"{target_module}.{target_class}", spy):
        run_module.make_runner(
            framework=framework, mode=mode, query_mode=query_mode, **kwargs,
        )

    assert spy.called, (
        f"--framework {framework} --mode {mode} --query-mode {query_mode} "
        f"--dataset {dataset} did not instantiate "
        f"{target_module}.{target_class}."
    )


# ---------------------------------------------------------------------------
# (3b) Negative cell: claude_sdk_v1 must NOT instantiate any v0 class.
# Catches the "copy-paste forgot to repoint" bug in the aggregator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "v0_module,v0_class",
    [
        (
            "bird_interact_agents.agents.claude_sdk_otf",
            "ClaudeSDKOtfAgent",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_raw",
            "ClaudeSDKOtfRawAgent",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract",
            "ClaudeSDKOtfAInteractAgent",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
            "ClaudeSDKOtfAInteractRawAgent",
        ),
    ],
)
def test_claude_sdk_v1_aggregator_does_not_instantiate_v0(v0_module, v0_class):
    """The v1 aggregator must never reach into the v0 packages."""
    import bird_interact_agents.run as run_module

    spy = _spy_class_for(v0_module, v0_class)
    # Walk every (mode, query_mode, dataset) cell of the v1 aggregator.
    cells = [
        ("one-shot", "slayer", "livesqlbench-base-lite-sqlite"),
        ("one-shot", "raw", "livesqlbench-base-lite-sqlite"),
        ("a-interact", "slayer", "mini-interact"),
        ("a-interact", "raw", "mini-interact"),
    ]
    for mode, qm, ds in cells:
        spy.reset_mock()
        kwargs = _common_kwargs()
        kwargs["dataset"] = ds
        with mock.patch(f"{v0_module}.{v0_class}", spy):
            run_module.make_runner(
                framework="claude_sdk_v1", mode=mode, query_mode=qm, **kwargs,
            )
        assert not spy.called, (
            f"--framework claude_sdk_v1 --mode {mode} --query-mode {qm} "
            f"--dataset {ds} reached into v0 class {v0_module}.{v0_class}."
        )
