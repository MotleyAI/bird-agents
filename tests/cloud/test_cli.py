"""T35: CLI argument parsing.

Asserts that `bird-interact-cloud submit` exposes all new flags, that the
mode values match the local `bird-interact` CLI, and that mutually exclusive
combinations are rejected.
"""

from __future__ import annotations

import argparse

import pytest

from bird_interact_agents.cloud import cli  # noqa: E402


def _parse(argv: list[str]) -> argparse.Namespace:
    """`cli.parse_args` wrapper that auto-adds `--no-subscription-auth` for
    `submit` / `annotate` argvs when neither explicit form is already present.

    `--subscription-auth` / `--no-subscription-auth` is REQUIRED at the CLI
    (DEV-1535: no default, to prevent silent fall-back to the API-key path).
    The bulk of the parse-time tests below don't care which auth path was
    chosen — they exercise other flags / validations — so the wrapper picks
    the API-key path by default (no env dependency in CI). The dedicated
    DEV-1535 tests further down toggle the flag deliberately.
    """
    has_auth = bool(argv) and any(
        a in ("--subscription-auth", "--no-subscription-auth") for a in argv
    )
    needs_auth = bool(argv) and argv[0] in ("submit", "annotate")
    if needs_auth and not has_auth:
        argv = [*argv, "--no-subscription-auth"]
    return cli.parse_args(argv)


# ---------------------------------------------------------------------------
# Mode names must match the local CLI (Codex MAJOR #10).
# ---------------------------------------------------------------------------


def _lsb_argv(extra: list[str]) -> list[str]:
    # The annotation guard fires for livesqlbench too; fake `alien_1`
    # has no annotation in CI checkouts, so default-skip the guard
    # here. The annotation-presence tests further down toggle
    # `--no-require-annotation` deliberately.
    return [
        "submit",
        "--framework", "claude_sdk",
        "--query-mode", "slayer",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1",
        "--slayer-setup", "on-the-fly",
        "--dataset", "livesqlbench-base-lite-sqlite",
        "--no-require-annotation",
        *extra,
    ]


def test_livesqlbench_one_shot_accepted():
    ns = _parse(
        _lsb_argv(["--mode", "one-shot"])
    )
    assert ns.dataset == "livesqlbench-base-lite-sqlite"
    assert ns.mode == "one-shot"


def test_livesqlbench_rejects_unsupported_mode():
    # a-interact is not in livesqlbench's supported modes → gate rejects.
    with pytest.raises(SystemExit):
        _parse(_lsb_argv(["--mode", "a-interact"]))


def test_dataset_hyphen_alias_normalized_to_canonical():
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk", "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1", "--mode", "a-interact",
            "--dataset", "mini-interact",  # hyphen alias
            "--no-require-annotation",
        ]
    )
    assert ns.dataset == "mini-interact"  # normalized to canonical


def test_dataset_is_required():
    """--dataset has no default — omitting it must be rejected. Prevents
    silently running mini-interact when --mode/--instance-ids are consistent
    with both benchmarks."""
    with pytest.raises(SystemExit):
        _parse(
            [
                "submit",
                "--framework", "claude_sdk", "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1", "--mode", "a-interact",
                "--no-require-annotation",
            ]
        )


@pytest.mark.parametrize("mode", ["a-interact", "oracle"])
def test_mode_values_accepted(mode: str) -> None:
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", mode,
            "--no-require-annotation",
        ]
    )
    assert ns.mode == mode


def test_unknown_mode_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
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
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",  # slayer is guarded for cloud (see below)
            "--agent-model", "cerebras/zai-glm-4.7",
            "--instance-ids", "db_a_1,db_a_2,db_a_3",
            "--mode", "a-interact",
            "--use-audited-gold-sql",
            "--no-require-annotation",  # fake ids, skip the audit-gold guard
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
    assert ns.require_annotation is False
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


def test_default_patience_is_250() -> None:
    """DEV-1535: default patience dropped 500 → 250. The 500 was
    historical paranoia from DEV-1478 (avoiding apparent OTF regressions
    from early termination); a sweep of 76 mini-interact retries showed
    the max trajectory among `correct` runs was 175 steps, so the 500
    budget was never close to binding. The 250 ceiling still gives
    correct/valid runs ~30%+ headroom while squeezing thrashing
    agent_miss runs that were burning past 200 steps."""
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "a-interact",
            "--no-require-annotation",
        ]
    )
    assert ns.patience == 250


def test_default_use_audited_gold_sql_is_true() -> None:
    """DEV-1478 follow-up: default flipped from False → True. Eval
    against the original un-audited gold is unsound — the un-audited
    rows include KB-described predicates the agent has no way to
    ground."""
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "a-interact",
            "--no-require-annotation",
        ]
    )
    assert ns.use_audited_gold_sql is True


def test_no_use_audited_gold_sql_opt_out() -> None:
    """The BooleanOptionalAction emits a paired `--no-` form. The
    annotation guard is decoupled from `--use-audited-gold-sql` (see
    `test_require_annotation_decoupled_from_use_audited_gold_sql`), so
    the fixture id needs `--no-require-annotation` too."""
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "a-interact",
            "--no-use-audited-gold-sql",
            "--no-require-annotation",
        ]
    )
    assert ns.use_audited_gold_sql is False


def test_require_annotation_default_on() -> None:
    """The annotation guard is default-on. Fixture id `db_a_1` is not in
    the mini-interact data file and has no annotation, so the parser
    must reject."""
    with pytest.raises(SystemExit):
        _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "c-interact",
            ]
        )


def test_require_annotation_decoupled_from_use_audited_gold_sql() -> None:
    """`--no-use-audited-gold-sql` no longer disables the annotation guard.
    Annotations are needed for grading (evaluator_prompt, novel-reading
    judgment, masked-term anchors) regardless of whether the audited-gold
    overlay is applied."""
    with pytest.raises(SystemExit):
        _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "a-interact",
                "--no-use-audited-gold-sql",
            ]
        )


def test_require_annotation_opt_out_accepts_unannotated_id() -> None:
    """`--no-require-annotation` is the bypass — accept any id for smoke
    runs without committing an annotation first."""
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "a-interact",
            "--no-use-audited-gold-sql",
            "--no-require-annotation",
        ]
    )
    assert ns.use_audited_gold_sql is False
    assert ns.require_annotation is False
    assert ns.instance_ids == ["db_a_1"]


# ---------------------------------------------------------------------------
# The annotation guard fires for livesqlbench too (same code path as
# mini-interact; the only delta is the benchmark argument and the
# annotations sub-tree). Pre-DEV-1515 the equivalent audited-gold check
# also flipped to per-benchmark dispatch in DEV-1510; the annotation
# check inherits the same symmetry — these tests pin that contract.
# ---------------------------------------------------------------------------


def _lsb_ann_argv(extra: list[str] | None = None) -> list[str]:
    """Like `_lsb_argv`, but with `museum_7`, `--mode one-shot`, and
    `--require-annotation` forced back ON (overrides `_lsb_argv`'s default
    `--no-require-annotation`) so the argv reaches the annotation guard."""
    return _lsb_argv([
        "--mode", "one-shot",
        "--instance-ids", "museum_7",
        "--require-annotation",
        *(extra or []),
    ])


def _stub_lsb_dataset_file(tmp_path, monkeypatch, *, instance_id: str = "museum_7") -> None:
    """Point `paths.benchmark_data_file` at a tmp livesqlbench JSONL so the
    annotation guard's instance_id → selected_database lookup resolves
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


def _stub_empty_annotations_root(tmp_path, monkeypatch) -> None:
    """Point `paths.annotations_root` at an EMPTY tmp dir so the guard sees
    `missing-file` rather than reading whatever the dev's checkout has."""
    from bird_interact_agents import paths as _paths

    annotations_root = tmp_path / "annotations"
    annotations_root.mkdir(exist_ok=True)
    monkeypatch.setattr(_paths, "annotations_root", lambda: annotations_root)


def test_require_annotation_default_on_for_livesqlbench(
    tmp_path, monkeypatch,
) -> None:
    """Annotation guard fires for livesqlbench too. `museum_7` with no
    annotation must reject."""
    _stub_empty_annotations_root(tmp_path, monkeypatch)
    _stub_lsb_dataset_file(tmp_path, monkeypatch, instance_id="museum_7")

    with pytest.raises(SystemExit):
        _parse(_lsb_ann_argv())


def test_require_annotation_can_be_disabled_for_livesqlbench(
    tmp_path, monkeypatch,
) -> None:
    """Same opt-out shape as mini-interact: `--no-require-annotation`
    lets the submit proceed without an annotation file."""
    _stub_empty_annotations_root(tmp_path, monkeypatch)
    _stub_lsb_dataset_file(tmp_path, monkeypatch, instance_id="museum_7")

    ns = _parse(_lsb_ann_argv(["--no-require-annotation"]))
    assert ns.dataset == "livesqlbench-base-lite-sqlite"
    assert ns.require_annotation is False
    assert ns.use_audited_gold_sql is True  # default-on stays on


def test_require_annotation_passes_for_livesqlbench_when_file_present(
    tmp_path, monkeypatch,
) -> None:
    """The guard accepts a livesqlbench submit whose instance_ids DO have
    annotation files — proves the symmetry is bidirectional."""
    from bird_interact_agents import paths as _paths

    annotations_root = tmp_path / "annotations"
    ann_dir = annotations_root / "livesqlbench-base-lite-sqlite" / "museum"
    ann_dir.mkdir(parents=True)
    (ann_dir / "museum_7.task.json").write_text("{}")  # presence-only check
    monkeypatch.setattr(_paths, "annotations_root", lambda: annotations_root)
    _stub_lsb_dataset_file(tmp_path, monkeypatch, instance_id="museum_7")

    ns = _parse(_lsb_ann_argv())
    assert ns.dataset == "livesqlbench-base-lite-sqlite"
    assert ns.instance_ids == ["museum_7"]


def test_prompt_cache_default_on() -> None:
    ns = _parse(
        [
            "submit",
            "--dataset", "mini-interact",
            "--framework", "claude_sdk",
            "--query-mode", "raw",
            "--agent-model", "anthropic/claude-sonnet-4-5",
            "--instance-ids", "db_a_1",
            "--mode", "a-interact",
            "--no-require-annotation",
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
        "framework": "claude_sdk",
        "query_mode": "slayer",
        "agent_model": "anthropic/claude-sonnet-4-5",
        "instance_ids": "db_a_1",
        "mode": "a-interact",
    }
    base.update(over)
    argv = [
        "submit",
        "--dataset", base.get("dataset", "mini-interact"),
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
    argv += ["--no-require-annotation"]
    return argv


def test_slayer_pre_encoded_rejected() -> None:
    """pre-encoded + query-mode=slayer is always rejected — on-the-fly is
    the only valid setup for slayer mode."""
    with pytest.raises(SystemExit):
        _parse(_slayer_argv(mode="a-interact"))


def test_slayer_on_the_fly_recursive_accepted() -> None:
    ns = _parse(_slayer_argv(
        framework="claude_sdk", mode="a-interact",
        slayer_setup="on-the-fly",
    ))
    assert ns.slayer_setup == "on-the-fly"


def test_slayer_on_the_fly_otf_encode_accepted() -> None:
    ns = _parse(_slayer_argv(
        framework="claude_sdk", mode="a-interact",
        slayer_setup="on-the-fly",
    ))
    assert ns.framework == "claude_sdk"
    assert ns.slayer_setup == "on-the-fly"


def test_slayer_storage_root_override_parsed() -> None:
    ns = _parse(_slayer_argv(
        slayer_setup="on-the-fly", mode="a-interact",
        slayer_storage_root="/data/custom_models",
    ))
    assert ns.slayer_storage_root == "/data/custom_models"


def test_on_the_fly_accepted_any_framework() -> None:
    """on-the-fly + any supported framework + a-interact is accepted —
    framework-specific validation was removed in DEV-1525."""
    ns = _parse(_slayer_argv(
        framework="claude_sdk", mode="a-interact", slayer_setup="on-the-fly",
    ))
    assert ns.slayer_setup == "on-the-fly"


def test_otf_encode_requires_on_the_fly() -> None:
    """pydantic_ai_otf_encode is on-the-fly-only; pre-encoded (default) must
    be rejected at submit."""
    with pytest.raises(SystemExit):
        _parse(_slayer_argv(
            framework="claude_sdk", mode="a-interact",
        ))


# ---------------------------------------------------------------------------
# DEV-1507 — claude_sdk_otf split: ainteract framework + dataset×framework
# gate (Codex HIGH#1 / MED#6).
# ---------------------------------------------------------------------------


def _ainteract_argv(**over) -> list[str]:
    base = {
        "framework": "claude_sdk",
        "query_mode": "slayer",
        "agent_model": "anthropic/claude-opus-4-7",
        "instance_ids": "shop_1",
        "mode": "a-interact",
        "dataset": "mini-interact",
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
        "--no-require-annotation",
    ]
    return argv


def test_cloud_ainteract_with_mini_interact_a_interact_on_the_fly_accepted():
    ns = _parse(_ainteract_argv())
    assert ns.framework == "claude_sdk"
    assert ns.dataset == "mini-interact"
    assert ns.mode == "a-interact"
    assert ns.slayer_setup == "on-the-fly"


def test_cloud_ainteract_with_pre_encoded_rejected():
    with pytest.raises(SystemExit):
        _parse(_ainteract_argv(slayer_setup="pre-encoded"))


def test_cloud_ainteract_with_livesqlbench_rejected():
    """Dataset×framework gate: ainteract is bound to mini_interact."""
    with pytest.raises(SystemExit):
        _parse(_ainteract_argv(
            dataset="livesqlbench-base-lite-sqlite",
        ))


def test_cloud_ainteract_one_shot_livesqlbench_accepted():
    """one-shot + livesqlbench is accepted — framework-specific validation
    was removed in DEV-1525, and livesqlbench supports one-shot mode."""
    ns = _parse(_ainteract_argv(
        mode="one-shot", dataset="livesqlbench-base-lite-sqlite",
    ))
    assert ns.mode == "one-shot"


def test_cloud_claude_sdk_otf_with_mini_interact_oracle_rejected():
    """Codex MED#6 / HIGH#1: claude_sdk_otf + mini_interact + oracle would
    pass `_validate_dataset_mode` (mini_interact supports oracle) — the new
    `_validate_framework_dataset_mode` is the only gate that rejects it."""
    with pytest.raises(SystemExit):
        _parse([
            "submit",
            "--framework", "claude_sdk_otf",
            "--query-mode", "slayer",
            "--agent-model", "anthropic/claude-opus-4-7",
            "--instance-ids", "alien_1",
            "--mode", "oracle",
            "--dataset", "mini-interact",
            "--no-require-annotation",
        ])


def test_cloud_ainteract_with_livesqlbench_oracle_rejected():
    """Symmetric oracle case for the new flavor."""
    with pytest.raises(SystemExit):
        _parse([
            "submit",
            "--framework", "claude_sdk_otf_ainteract",
            "--query-mode", "slayer",
            "--agent-model", "anthropic/claude-opus-4-7",
            "--instance-ids", "alien_1",
            "--mode", "oracle",
            "--dataset", "livesqlbench-base-lite-sqlite",
            "--no-require-annotation",
        ])


def test_detach_and_allow_dirty_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
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
        _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
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
        _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
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


@pytest.mark.parametrize("sub", ["submit", "fetch", "kill", "list", "build", "resubmit"])
def test_subcommand_registered(sub: str) -> None:
    # All sub-commands at least parse to a known namespace. `fetch` / `kill`
    # / `resubmit` take a run-id positional; `list` / `build` are flagless.
    if sub == "list":
        ns = _parse(["list"])
    elif sub == "build":
        ns = _parse(["build"])
    elif sub == "submit":
        ns = _parse(
            [
                "submit",
                "--dataset", "mini-interact",
                "--framework", "claude_sdk",
                "--query-mode", "raw",
                "--agent-model", "anthropic/claude-sonnet-4-5",
                "--instance-ids", "db_a_1",
                "--mode", "a-interact",
                "--no-require-annotation",
            ]
        )
    else:
        ns = _parse([sub, "some-run-id"])
    assert ns.subcommand == sub


# ---------------------------------------------------------------------------
# DEV-1535 — --subscription-auth is REQUIRED on submit + annotate; the silent
# fall-back to the API-key path (the failure mode that burned credits and
# turned 20 tasks into eval_failed mid-run) is gone.
# ---------------------------------------------------------------------------


def _minimal_submit_argv(extra: list[str] | None = None) -> list[str]:
    return [
        "submit",
        "--dataset", "mini-interact",
        "--framework", "claude_sdk",
        "--query-mode", "raw",
        "--agent-model", "anthropic/claude-sonnet-4-5",
        "--instance-ids", "db_a_1",
        "--mode", "a-interact",
        "--no-require-annotation",
        *(extra or []),
    ]


def _minimal_annotate_argv(extra: list[str] | None = None) -> list[str]:
    return [
        "annotate",
        "--benchmark", "mini-interact",
        "--agent-model", "anthropic/claude-opus-4-7",
        "--instance-ids", "db_a_1",
        *(extra or []),
    ]


def test_subscription_auth_required_on_submit() -> None:
    """Omitting both `--subscription-auth` and `--no-subscription-auth` on
    submit is rejected at parse time — no default."""
    with pytest.raises(SystemExit):
        # Bypass the `_parse` wrapper that would auto-inject the flag.
        cli.parse_args(_minimal_submit_argv())


def test_subscription_auth_required_on_annotate() -> None:
    """Same required-arg shape for annotate."""
    with pytest.raises(SystemExit):
        cli.parse_args(_minimal_annotate_argv())


def test_no_subscription_auth_submit_sets_legacy_path() -> None:
    """`--no-subscription-auth` on submit chooses the legacy API-key path."""
    ns = cli.parse_args(_minimal_submit_argv(["--no-subscription-auth"]))
    assert ns.subscription_auth is False
    assert ns.no_subscription_auth is True


def test_subscription_auth_submit_with_valid_token(monkeypatch) -> None:
    """`--subscription-auth` with a valid-prefix OAuth token in the env
    parses to subscription_auth=True / no_subscription_auth=False."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake-token")
    ns = cli.parse_args(_minimal_submit_argv(["--subscription-auth"]))
    assert ns.subscription_auth is True
    assert ns.no_subscription_auth is False


def test_subscription_auth_submit_without_token_rejected(monkeypatch) -> None:
    """`--subscription-auth` without CLAUDE_CODE_OAUTH_TOKEN must fail at
    parse — the exact silent-fall-back gap DEV-1535 closes."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.parse_args(_minimal_submit_argv(["--subscription-auth"]))


def test_subscription_auth_submit_with_bad_prefix_rejected(monkeypatch) -> None:
    """A malformed token (wrong prefix) is rejected too — caught at parse
    time before the cluster warms up."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api03-not-an-oat")
    with pytest.raises(SystemExit):
        cli.parse_args(_minimal_submit_argv(["--subscription-auth"]))


def test_no_subscription_auth_annotate_sets_legacy_path() -> None:
    ns = cli.parse_args(_minimal_annotate_argv(["--no-subscription-auth"]))
    assert ns.subscription_auth is False
    assert ns.no_subscription_auth is True


def test_subscription_auth_annotate_without_token_rejected(monkeypatch) -> None:
    """Annotate enforces the same token requirement as submit."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cli.parse_args(_minimal_annotate_argv(["--subscription-auth"]))
