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


def _lsb_argv(extra: list[str]) -> list[str]:
    # DEV-1510: the audited-gold guard now fires for livesqlbench too;
    # fake `alien_1` has no audit row in CI checkouts, so default-skip
    # the guard here. The DEV-1510-specific tests (further down) test
    # the guard's livesqlbench behaviour with `--no-require-audited-gold`
    # toggled deliberately.
    return [
        "submit",
        "--framework", "pydantic_ai_otf_encode",
        "--query-mode", "slayer",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1",
        "--slayer-setup", "on-the-fly",
        "--dataset", "livesqlbench",
        "--no-require-audited-gold",
        *extra,
    ]


def test_livesqlbench_one_shot_accepted_with_gold():
    ns = cli.parse_args(
        _lsb_argv(["--mode", "one-shot", "--gold-file", "/abs/gold.jsonl"])
    )
    assert ns.dataset == "livesqlbench"
    assert ns.gold_file == "/abs/gold.jsonl"
    assert ns.mode == "one-shot"


def test_livesqlbench_without_gold_file_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(_lsb_argv(["--mode", "one-shot"]))  # no --gold-file


def test_livesqlbench_rejects_unsupported_mode():
    # a-interact is not in livesqlbench's supported modes → gate rejects.
    with pytest.raises(SystemExit):
        cli.parse_args(_lsb_argv(["--mode", "a-interact", "--gold-file", "/g.jsonl"]))


def test_dataset_hyphen_alias_normalized_to_canonical():
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai", "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1", "--mode", "a-interact",
            "--dataset", "mini-interact",  # hyphen alias
            "--no-require-audited-gold",
        ]
    )
    assert ns.dataset == "mini_interact"  # normalized to canonical


def test_dataset_is_required():
    """--dataset has no default — omitting it must be rejected. Prevents
    silently running mini-interact when --mode/--instance-ids are consistent
    with both benchmarks."""
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--framework", "pydantic_ai", "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1", "--mode", "a-interact",
                "--no-require-audited-gold",
            ]
        )


@pytest.mark.parametrize("mode", ["a-interact", "c-interact", "oracle"])
def test_mode_values_accepted(mode: str) -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", mode,
            "--no-require-audited-gold",
        ]
    )
    assert ns.mode == mode


def test_unknown_mode_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--dataset", "mini_interact",
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
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai_recursive",
            "--query-mode", "raw",  # slayer is guarded for cloud (see below)
            "--agent-model", "cerebras/zai-glm-4.7",
            "--instance-ids", "db_a_1,db_a_2,db_a_3",
            "--mode", "c-interact",
            "--use-audited-gold-sql",
            "--no-require-audited-gold",  # fake ids, skip the audit-gold guard
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
    assert ns.require_audited_gold is False
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


def test_default_patience_is_500() -> None:
    """DEV-1478 follow-up: bird-interact-cloud default patience flipped
    from 3 → 500. Patience 3 was producing apparent OTF regressions
    that were actually just early-termination, not encoder failures."""
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
            "--no-require-audited-gold",
        ]
    )
    assert ns.patience == 500


def test_default_use_audited_gold_sql_is_true() -> None:
    """DEV-1478 follow-up: default flipped from False → True. Eval
    against the original un-audited gold is unsound — the un-audited
    rows include KB-described predicates the agent has no way to
    ground."""
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
            "--no-require-audited-gold",
        ]
    )
    assert ns.use_audited_gold_sql is True


def test_no_use_audited_gold_sql_opt_out() -> None:
    """The BooleanOptionalAction emits a paired `--no-` form."""
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
            "--no-use-audited-gold-sql",
        ]
    )
    assert ns.use_audited_gold_sql is False


def test_require_audited_gold_default_on() -> None:
    """When --use-audited-gold-sql is on (default), the parser also
    requires every instance_id to have an audited-gold entry; the
    `db_a_1` fixture id has no entry, so the parser must reject."""
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--dataset", "mini_interact",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
            ]
        )


def test_require_audited_gold_disabled_when_use_audited_gold_off() -> None:
    """If the user opts out of audited gold entirely, the
    require-audited-gold guard is moot — accept any id."""
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
            "--no-use-audited-gold-sql",
        ]
    )
    assert ns.use_audited_gold_sql is False
    assert ns.instance_ids == ["db_a_1"]


# ---------------------------------------------------------------------------
# DEV-1510: the `require_audited_gold` guard ALSO fires for livesqlbench now
# that the benchmark has its own audited-gold sidecar
# (`audited_gold/livesqlbench_audited.jsonl`). Pre-fix, cli.py skipped the
# guard with `not get_benchmark(ns.dataset).gold_required` — the safety net
# was disabled for the only gold_required benchmark, so a livesqlbench submit
# could ship with NO audited rows and silently fall back to original gold.
# ---------------------------------------------------------------------------


def _lsb_audit_argv(extra: list[str] | None = None) -> list[str]:
    """Like `_lsb_argv`, but with `museum_7` (the locked DEV-1510 audit
    subject), `--mode one-shot`, `--gold-file`, and `--require-audited-gold`
    forced back ON (overrides `_lsb_argv`'s default `--no-require-audited-gold`)
    so the argv reaches the audited-gold guard."""
    return _lsb_argv([
        "--mode", "one-shot",
        "--gold-file", "/tmp/fake-livesqlbench-gold.jsonl",
        "--instance-ids", "museum_7",
        "--require-audited-gold",
        *(extra or []),
    ])


def _stub_lsb_dataset_file(tmp_path, monkeypatch, *, instance_id: str = "museum_7") -> None:
    """Point `paths.benchmark_data_file` at a tmp livesqlbench JSONL so the
    audited-gold guard's instance_id → selected_database lookup resolves
    (without depending on the gitignored real data root)."""
    from bird_interact_agents import paths as _paths

    lsb_root = tmp_path / "livesqlbench-base-lite-sqlite"
    lsb_root.mkdir(exist_ok=True)
    data_file = lsb_root / "livesqlbench_data_sqlite.jsonl"
    data_file.write_text(
        f'{{"instance_id":"{instance_id}","selected_database":"museum"}}\n'
    )
    monkeypatch.setattr(
        _paths, "benchmark_data_file", lambda *a, **k: data_file,
    )


def _stub_empty_audited_gold_root(tmp_path, monkeypatch) -> None:
    """Point `paths.audited_gold_root` at an EMPTY tmp dir so the
    livesqlbench guard sees `missing-file` (no `livesqlbench_audited.jsonl`)
    rather than reading whatever the dev's main checkout contains."""
    from bird_interact_agents import paths as _paths

    audited_root = tmp_path / "audited_gold"
    audited_root.mkdir(exist_ok=True)
    monkeypatch.setattr(_paths, "audited_gold_root", lambda: audited_root)


def test_require_audited_gold_default_on_for_livesqlbench(
    tmp_path, monkeypatch,
) -> None:
    """DEV-1510: pre-fix, cli.py skipped this guard for any benchmark with
    `gold_required=True` — i.e. livesqlbench. A submit with `museum_7`
    and no audit row would silently fall back. Post-fix the guard fires
    for livesqlbench too, so the submit must reject."""
    _stub_empty_audited_gold_root(tmp_path, monkeypatch)
    _stub_lsb_dataset_file(tmp_path, monkeypatch, instance_id="museum_7")

    with pytest.raises(SystemExit):
        cli.parse_args(_lsb_audit_argv())


def test_require_audited_gold_can_be_disabled_for_livesqlbench(
    tmp_path, monkeypatch,
) -> None:
    """Same opt-out shape as mini-interact: `--no-require-audited-gold`
    lets the submit proceed without an audit row."""
    _stub_empty_audited_gold_root(tmp_path, monkeypatch)
    _stub_lsb_dataset_file(tmp_path, monkeypatch, instance_id="museum_7")

    ns = cli.parse_args(_lsb_audit_argv(["--no-require-audited-gold"]))
    assert ns.dataset == "livesqlbench"
    assert ns.require_audited_gold is False
    assert ns.use_audited_gold_sql is True  # default-on stays on


def test_require_audited_gold_passes_for_livesqlbench_when_row_present(
    tmp_path, monkeypatch,
) -> None:
    """The guard accepts a livesqlbench submit whose instance_ids ARE in
    the audit file — proves the symmetry is bidirectional (not just
    `always reject livesqlbench`).

    Uses an `edited` row with `audited_sol_sql` for `museum_7` — matches
    the locked DEV-1510 decision (museum_7 is `edited`, not `clean`)."""
    from bird_interact_agents import paths as _paths

    audited_root = tmp_path / "audited_gold"
    audited_root.mkdir()
    (audited_root / "livesqlbench_audited.jsonl").write_text(
        '{"instance_id":"museum_7","selected_database":"museum",'
        '"benchmark":"livesqlbench",'
        '"audit_status":"edited","audited_sol_sql":["SELECT 1"]}\n'
    )
    monkeypatch.setattr(_paths, "audited_gold_root", lambda: audited_root)
    _stub_lsb_dataset_file(tmp_path, monkeypatch, instance_id="museum_7")

    ns = cli.parse_args(_lsb_audit_argv())
    assert ns.dataset == "livesqlbench"
    assert ns.instance_ids == ["museum_7"]


def test_prompt_cache_default_on() -> None:
    ns = cli.parse_args(
        [
            "submit",
            "--dataset", "mini_interact",
            "--framework", "pydantic_ai",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "c-interact",
            "--no-require-audited-gold",
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
        "--dataset", base.get("dataset", "mini_interact"),
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
    # fake instance ids in fixture argv — skip the audited-gold guard.
    argv += ["--no-require-audited-gold"]
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


# ---------------------------------------------------------------------------
# DEV-1507 — claude_sdk_otf split: ainteract framework + dataset×framework
# gate (Codex HIGH#1 / MED#6).
# ---------------------------------------------------------------------------


def _ainteract_argv(**over) -> list[str]:
    base = {
        "framework": "claude_sdk_otf_ainteract",
        "query_mode": "slayer",
        "agent_model": "anthropic/claude-opus-4-7",
        "instance_ids": "shop_1",
        "mode": "a-interact",
        "dataset": "mini_interact",
        "slayer_setup": "on-the-fly",
    }
    base.update(over)
    argv = [
        "submit",
        "--framework", base["framework"],
        "--query-mode", base["query_mode"],
        "--agent-model", base["agent_model"],
        "--instance-ids", base["instance_ids"],
        "--mode", base["mode"],
        "--dataset", base["dataset"],
        "--slayer-setup", base["slayer_setup"],
        "--no-require-audited-gold",
    ]
    if "gold_file" in over:
        argv += ["--gold-file", over["gold_file"]]
    return argv


def test_cloud_ainteract_with_mini_interact_a_interact_on_the_fly_accepted():
    ns = cli.parse_args(_ainteract_argv())
    assert ns.framework == "claude_sdk_otf_ainteract"
    assert ns.dataset == "mini_interact"
    assert ns.mode == "a-interact"
    assert ns.slayer_setup == "on-the-fly"


def test_cloud_ainteract_with_pre_encoded_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(_ainteract_argv(slayer_setup="pre-encoded"))


def test_cloud_ainteract_with_livesqlbench_rejected():
    """Dataset×framework gate: ainteract is bound to mini_interact even with
    a gold-file present."""
    with pytest.raises(SystemExit):
        cli.parse_args(_ainteract_argv(
            dataset="livesqlbench", gold_file="/abs/gold.jsonl",
        ))


def test_cloud_ainteract_with_one_shot_rejected():
    """one-shot is rejected by _validate_one_shot_framework (only
    claude_sdk_otf is whitelisted on the one-shot path)."""
    with pytest.raises(SystemExit):
        cli.parse_args(_ainteract_argv(
            mode="one-shot", dataset="livesqlbench",
            gold_file="/abs/gold.jsonl",
        ))


def test_cloud_claude_sdk_otf_with_mini_interact_oracle_rejected():
    """Codex MED#6 / HIGH#1: claude_sdk_otf + mini_interact + oracle would
    pass `_validate_dataset_mode` (mini_interact supports oracle) — the new
    `_validate_framework_dataset_mode` is the only gate that rejects it."""
    with pytest.raises(SystemExit):
        cli.parse_args([
            "submit",
            "--framework", "claude_sdk_otf",
            "--query-mode", "slayer",
            "--agent-model", "anthropic/claude-opus-4-7",
            "--instance-ids", "alien_1",
            "--mode", "oracle",
            "--dataset", "mini_interact",
            "--no-require-audited-gold",
        ])


def test_cloud_ainteract_with_livesqlbench_oracle_rejected():
    """Symmetric oracle case for the new flavor."""
    with pytest.raises(SystemExit):
        cli.parse_args([
            "submit",
            "--framework", "claude_sdk_otf_ainteract",
            "--query-mode", "slayer",
            "--agent-model", "anthropic/claude-opus-4-7",
            "--instance-ids", "alien_1",
            "--mode", "oracle",
            "--dataset", "livesqlbench",
            "--gold-file", "/abs/gold.jsonl",
            "--no-require-audited-gold",
        ])


def test_detach_and_allow_dirty_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "submit",
                "--dataset", "mini_interact",
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
                "--dataset", "mini_interact",
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
                "--dataset", "mini_interact",
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
    """`cli.main(["build"])` must pass `audited_gold_root` to both
    `image.image_tag` and `image.build_and_push`. The `build` subcommand is a
    SECOND production caller (besides the driver) — invoking it with a stale
    signature raises TypeError, which CI/users would hit immediately.

    De-bake: `image_tag`/`build_and_push` no longer take `mini_interact_root`
    (the dataset is delivered via GCS, not baked), so `audited_gold_root` is
    now the 2nd positional arg to `image_tag`.
    """
    from bird_interact_agents.cloud import image as _image
    from bird_interact_agents.cloud import driver as _driver
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(
        _driver, "submitter_repo_root", lambda: tmp_path / "repo",
    )
    fake_ag = tmp_path / "main-checkout" / "audited_gold"
    monkeypatch.setattr(_paths, "audited_gold_root", lambda: fake_ag)
    fake_ann = tmp_path / "main-checkout" / "annotations"
    monkeypatch.setattr(_paths, "annotations_root", lambda: fake_ann)

    tag_calls: list[tuple] = []
    push_calls: list[tuple] = []

    def fake_image_tag(repo_root, audited_gold_root, *, allow_dirty,
                       annotations_root=None):
        tag_calls.append((repo_root, audited_gold_root, allow_dirty,
                          annotations_root))
        return "deadbeef-cafebabe"

    def fake_build_and_push(tag, repo_root, *, audited_gold_root=None,
                            annotations_root=None,
                            force=False, **_kw):
        push_calls.append((tag, repo_root, audited_gold_root,
                           annotations_root, force))
        return f"registry.example/x/runner:{tag}"

    monkeypatch.setattr(_image, "image_tag", fake_image_tag)
    monkeypatch.setattr(_image, "build_and_push", fake_build_and_push)

    rc = cli.main(["build"])
    assert rc == 0
    assert len(tag_calls) == 1, "image_tag was not called by the build subcommand"
    assert tag_calls[0][1] == fake_ag, (
        f"image_tag missing audited_gold_root positional arg; got {tag_calls[0]}"
    )
    assert tag_calls[0][3] == fake_ann, (
        f"image_tag missing annotations_root kwarg; got {tag_calls[0]}"
    )
    assert len(push_calls) == 1
    assert push_calls[0][2] == fake_ag, (
        f"build_and_push missing audited_gold_root kwarg; got {push_calls[0]}"
    )
    assert push_calls[0][3] == fake_ann, (
        f"build_and_push missing annotations_root kwarg; got {push_calls[0]}"
    )


def test_annotate_gold_file_arg_accepted() -> None:
    """The `annotate` subcommand must expose `--gold-file` so users can
    provide an explicit gold sidecar path for LiveSQLBench annotation."""
    ns = cli.parse_args([
        "annotate",
        "--benchmark", "mini_interact",
        "--agent-model", "anthropic/claude-opus-4-7",
        "--instance-ids", "db_a_1",
        "--gold-file", "/data/livesqlbench/gold.jsonl",
    ])
    assert ns.gold_file == "/data/livesqlbench/gold.jsonl"


def test_annotate_gold_file_defaults_to_none() -> None:
    """When `--gold-file` is omitted, `ns.gold_file` must be None so the
    in-cluster fallback path can take over."""
    ns = cli.parse_args([
        "annotate",
        "--benchmark", "mini_interact",
        "--agent-model", "anthropic/claude-opus-4-7",
        "--instance-ids", "db_a_1",
    ])
    assert ns.gold_file is None


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
                "--dataset", "mini_interact",
                "--framework", "pydantic_ai",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
                "--no-require-audited-gold",
            ]
        )
    else:
        ns = cli.parse_args([sub, "some-run-id"])
    assert ns.subcommand == sub
