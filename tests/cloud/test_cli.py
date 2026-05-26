"""T35: CLI argument parsing.

Asserts that `bird-interact-cloud submit` exposes all new flags, that the
mode values match the local `bird-interact` CLI, and that mutually exclusive
combinations are rejected.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import cli  # noqa: E402


# ---------------------------------------------------------------------------
# Mode names must match the local CLI (Codex MAJOR #10).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["a-interact", "c-interact", "oracle"])
def test_mode_values_accepted(mode: str) -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", mode,
        ]
    )
    assert ns.mode == mode


def test_unknown_mode_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "interactive",  # not a real local mode
            ]
        )


# ---------------------------------------------------------------------------
# All pass-through flags parse.
# ---------------------------------------------------------------------------


def test_pass_through_flags_parse() -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--framework", "pydantic_ai_recursive",
            "--query-mode", "raw",  # slayer is guarded for cloud (see below)
            "--agent-model", "cerebras/zai-glm-4.7",
            "--instance-ids", "db_a_1,db_a_2,db_a_3",
            "--mode", "c-interact",
            "--use-audited-gold-sql",
            "--max-depth", "5",
            "--no-prompt-cache",
            "--workers", "8",
            "--actors-per-worker", "2",
            "--worker-type", "e2-standard-8",
            "--max-runtime-hours", "12",
            "--patience", "4",
            "--strict",
            "--detach",
        ]
    )
    assert ns.use_audited_gold_sql is True
    assert ns.max_depth == 5
    assert ns.prompt_cache is False
    assert ns.workers == 8
    assert ns.actors_per_worker == 2
    assert ns.worker_type == "e2-standard-8"
    assert ns.max_runtime_hours == 12
    assert ns.patience == 4
    assert ns.strict is True
    assert ns.detach is True
    assert ns.instance_ids == ["db_a_1", "db_a_2", "db_a_3"]


def test_prompt_cache_default_on() -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
        ]
    )
    assert ns.prompt_cache is True


# ---------------------------------------------------------------------------
# --detach --allow-dirty is mutually exclusive.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DEV-1468 — cloud slayer mode is now wired. The submit parser accepts slayer
# and the new --slayer-setup / --slayer-storage-root flags, and validates the
# (framework, query_mode, mode, slayer_setup) tuple via run._validate_slayer_setup.
# ---------------------------------------------------------------------------


def _slayer_argv(**over) -> list[str]:
    base = {
        "framework": "pydantic_ai_recursive",
        "query_mode": "slayer",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "instance_ids": "db_a_1",
        "mode": "c-interact",
    }
    base.update(over)
    argv = [
        "submit",
        "--framework", base["framework"],
        "--query-mode", base["query_mode"],
        "--agent-model", base["agent_model"],
        "--instance-ids", base["instance_ids"],
        "--mode", base["mode"],
    ]
    if "slayer_setup" in over:
        argv += ["--slayer-setup", over["slayer_setup"]]
    if "slayer_storage_root" in over:
        argv += ["--slayer-storage-root", over["slayer_storage_root"]]
    return argv


def test_slayer_pre_encoded_accepted() -> None:
    """pre-encoded (default) + slayer + any supported framework is accepted —
    the old hard rejection is gone."""
    ns = cli.parse_args(_slayer_argv(mode="c-interact"))
    assert ns.query_mode == "slayer"
    assert ns.slayer_setup == "pre-encoded"  # default
    assert ns.slayer_storage_root == "/data/slayer_models"  # default


def test_slayer_on_the_fly_recursive_accepted() -> None:
    ns = cli.parse_args(_slayer_argv(
        framework="pydantic_ai_recursive", mode="a-interact",
        slayer_setup="on-the-fly",
    ))
    assert ns.slayer_setup == "on-the-fly"


def test_slayer_on_the_fly_otf_encode_accepted() -> None:
    ns = cli.parse_args(_slayer_argv(
        framework="pydantic_ai_otf_encode", mode="a-interact",
        slayer_setup="on-the-fly",
    ))
    assert ns.framework == "pydantic_ai_otf_encode"
    assert ns.slayer_setup == "on-the-fly"


def test_slayer_storage_root_override_parsed() -> None:
    ns = cli.parse_args(_slayer_argv(slayer_storage_root="/data/custom_models"))
    assert ns.slayer_storage_root == "/data/custom_models"


def test_on_the_fly_wrong_framework_rejected() -> None:
    """on-the-fly only with pydantic_ai_recursive / pydantic_ai_otf_encode."""
    with pytest.raises(SystemExit):
        cli.parse_args(_slayer_argv(
            framework="pydantic_ai", mode="a-interact", slayer_setup="on-the-fly",
        ))


def test_on_the_fly_wrong_mode_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(_slayer_argv(
            framework="pydantic_ai_recursive", mode="c-interact",
            slayer_setup="on-the-fly",
        ))


def test_otf_encode_requires_on_the_fly() -> None:
    """pydantic_ai_otf_encode is on-the-fly-only; pre-encoded (default) must
    be rejected at submit."""
    with pytest.raises(SystemExit):
        cli.parse_args(_slayer_argv(
            framework="pydantic_ai_otf_encode", mode="a-interact",
        ))


def test_detach_and_allow_dirty_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
                "--detach",
                "--allow-dirty",
            ]
        )


# ---------------------------------------------------------------------------
# CR#2 — empty `--instance-ids` rejected at argparse time.
# ---------------------------------------------------------------------------


def test_empty_instance_ids_string_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "",
                "--mode", "c-interact",
            ]
        )


def test_empty_instance_ids_file_rejected(tmp_path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n  \n")  # whitespace only
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids-file", str(empty),
                "--mode", "c-interact",
            ]
        )


# ---------------------------------------------------------------------------
# Sub-commands present.
# ---------------------------------------------------------------------------


def test_build_subcommand_passes_audited_gold_root_to_image_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """DEV-1470 round 2 — `cli.main(["build"])` must pass `audited_gold_root`
    to both `image.image_tag` and `image.build_and_push`. The driver path
    was updated when the signatures became 3-positional, but the `build`
    subcommand is a SECOND production caller and was missed — invoking it
    pre-fix raises TypeError, which CI/users would hit immediately.
    """
    from bird_interact_agents.cloud import image as _image
    from bird_interact_agents.cloud import driver as _driver
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(
        _driver, "submitter_repo_root", lambda: tmp_path / "repo",
    )
    fake_mi = tmp_path / "mini-interact"
    fake_ag = tmp_path / "main-checkout" / "audited_gold"
    monkeypatch.setattr(_paths, "mini_interact_root", lambda: fake_mi)
    monkeypatch.setattr(_paths, "audited_gold_root", lambda: fake_ag)

    tag_calls: list[tuple] = []
    push_calls: list[tuple] = []

    def fake_image_tag(repo_root, mini_interact_root, audited_gold_root,
                       *, allow_dirty):
        tag_calls.append((repo_root, mini_interact_root, audited_gold_root,
                          allow_dirty))
        return "deadbeef-cafebabe"

    def fake_build_and_push(tag, repo_root, *, audited_gold_root=None,
                            force=False, **_kw):
        push_calls.append((tag, repo_root, audited_gold_root, force))
        return f"registry.example/x/runner:{tag}"

    monkeypatch.setattr(_image, "image_tag", fake_image_tag)
    monkeypatch.setattr(_image, "build_and_push", fake_build_and_push)

    rc = cli.main(["build"])
    assert rc == 0
    assert len(tag_calls) == 1, "image_tag was not called by the build subcommand"
    assert tag_calls[0][2] == fake_ag, (
        f"image_tag missing audited_gold_root positional arg; got {tag_calls[0]}"
    )
    assert len(push_calls) == 1
    assert push_calls[0][2] == fake_ag, (
        f"build_and_push missing audited_gold_root kwarg; got {push_calls[0]}"
    )


@pytest.mark.parametrize("sub", ["submit", "fetch", "kill", "list", "build", "resubmit"])
def test_subcommand_registered(sub: str) -> None:
    # All sub-commands at least parse to a known namespace. `fetch` / `kill`
    # / `resubmit` take a run-id positional; `list` / `build` are flagless.
    if sub == "list":
        ns = cli.parse_args(["list"])
    elif sub == "build":
        ns = cli.parse_args(["build"])
    elif sub == "submit":
        ns = cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
            ]
        )
    else:
        ns = cli.parse_args([sub, "some-run-id"])
    assert ns.subcommand == sub
