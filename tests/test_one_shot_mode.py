"""DEV-1462 — `--mode one-shot`, `--dataset`, `--gold-file`, and the new
validation guards.

All guards must fail fast BEFORE any task starts — and they must fire on
BOTH the CLI path (`run.main` → `parser.error`) AND the programmatic
path (`run_evaluation` / `make_runner` / `run_one_task` → `ValueError`),
matching the dual-path style of the existing `_validate_slayer_setup`
checks.

Tests drive `run.main` directly with synthesised argv (same idiom as
`test_slayer_setup_flag.py`) so argparse + the validation hook are both
exercised in one shot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bird_interact_agents import run as run_module


def _argv_base(tmp_path: Path) -> list[str]:
    data = tmp_path / "data.jsonl"
    data.write_text("")
    db_path = tmp_path / "ds"
    db_path.mkdir()
    return [
        "bird-interact",
        "--data", str(data),
        "--db-path", str(db_path),
        "--output", str(tmp_path / "out.json"),
        "--limit", "0",
    ]


def _drive_main(monkeypatch, argv: list[str]) -> dict:
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {"metrics": "fake"}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", argv)
    run_module.main()
    return captured


def _assert_failed_validation(capsys, exc, *, must_contain: list[str]):
    if isinstance(exc, SystemExit):
        cap = capsys.readouterr()
        msg = (cap.err or "") + (cap.out or "")
    else:
        msg = str(exc)
    for substr in must_contain:
        assert substr in msg, (
            f"validation error must mention {substr!r}; got: {msg!r}"
        )


# ---------------------------------------------------------------------------
# Mode + budget
# ---------------------------------------------------------------------------


def test_one_shot_in_mode_choices(monkeypatch, tmp_path):
    """argparse must accept `--mode one-shot` without erroring."""
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(tmp_path / "g.jsonl"),
        "--mode", "one-shot",
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
        "--slayer-setup", "on-the-fly",
    ]
    (tmp_path / "g.jsonl").write_text("")
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("mode") == "one-shot"


def test_calculate_budget_one_shot_is_30():
    from bird_interact_agents.harness import calculate_budget

    # The budget is dataset-independent; pass a minimal task dict.
    task = {"amb_user_query": "x"}
    assert calculate_budget(task, patience=3, mode="one-shot") == 30.0
    # Patience must not affect one-shot (only a-interact / c-interact use it).
    assert calculate_budget(task, patience=99, mode="one-shot") == 30.0


# ---------------------------------------------------------------------------
# Flag wiring
# ---------------------------------------------------------------------------


def test_dataset_flag_default_is_mini_interact(monkeypatch, tmp_path):
    argv = _argv_base(tmp_path) + [
        "--framework", "pydantic_ai", "--query-mode", "raw",
        "--mode", "a-interact",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("dataset") == "mini-interact"


def test_dataset_flag_livesqlbench_is_plumbed(monkeypatch, tmp_path):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", "oracle",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("dataset") == "livesqlbench"
    assert kwargs.get("gold_file") == str(gold)


def test_livesqlbench_requires_gold_file(monkeypatch, tmp_path, capsys):
    """`--dataset livesqlbench` without `--gold-file` MUST fail fast."""
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--mode", "oracle",
    ]
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        run_module.main()
    assert not captured
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["gold-file"],
    )


# ---------------------------------------------------------------------------
# Guards — one-shot ⟺ livesqlbench / livesqlbench ⟹ {one-shot, oracle}
# ---------------------------------------------------------------------------


def test_one_shot_requires_livesqlbench_dataset(monkeypatch, tmp_path, capsys):
    argv = _argv_base(tmp_path) + [
        "--mode", "one-shot",
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
        "--slayer-setup", "on-the-fly",
    ]
    captured: dict = {}

    async def fake_run_evaluation(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(run_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        run_module.main()
    assert not captured
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["one-shot", "livesqlbench"],
    )


@pytest.mark.parametrize("bad_mode", ["a-interact", "c-interact"])
def test_livesqlbench_rejects_interactive_modes(
    monkeypatch, tmp_path, capsys, bad_mode,
):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", bad_mode,
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
    ]
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        monkeypatch.setattr(sys, "argv", argv)
        run_module.main()
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["livesqlbench", "mode"],
    )


def test_livesqlbench_oracle_is_accepted(monkeypatch, tmp_path):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", "oracle",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("mode") == "oracle"
    assert kwargs.get("dataset") == "livesqlbench"


# ---------------------------------------------------------------------------
# Guards — one-shot ⟹ on-the-fly, slayer, recursive|otf_encode
# ---------------------------------------------------------------------------


def test_one_shot_requires_on_the_fly(monkeypatch, tmp_path, capsys):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", "one-shot",
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "slayer",
        "--slayer-setup", "pre-encoded",
    ]
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        monkeypatch.setattr(sys, "argv", argv)
        run_module.main()
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["one-shot", "on-the-fly"],
    )


def test_one_shot_requires_slayer_query_mode(monkeypatch, tmp_path, capsys):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", "one-shot",
        "--framework", "pydantic_ai_recursive",
        "--query-mode", "raw",
        "--slayer-setup", "on-the-fly",
    ]
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        monkeypatch.setattr(sys, "argv", argv)
        run_module.main()
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["one-shot", "query-mode"],
    )


@pytest.mark.parametrize(
    "framework", ["pydantic_ai", "claude_sdk", "agno", "smolagents", "mcp_agent"],
)
def test_one_shot_rejects_non_slayer_frameworks(
    monkeypatch, tmp_path, capsys, framework,
):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", "one-shot",
        "--framework", framework,
        "--query-mode", "slayer",
        "--slayer-setup", "on-the-fly",
    ]
    with pytest.raises((SystemExit, ValueError)) as exc_info:
        monkeypatch.setattr(sys, "argv", argv)
        run_module.main()
    _assert_failed_validation(
        capsys, exc_info.value, must_contain=["one-shot", "framework"],
    )


@pytest.mark.parametrize(
    "framework", ["pydantic_ai_recursive", "pydantic_ai_otf_encode"],
)
def test_one_shot_accepts_both_slayer_frameworks(
    monkeypatch, tmp_path, framework,
):
    gold = tmp_path / "g.jsonl"
    gold.write_text("")
    argv = _argv_base(tmp_path) + [
        "--dataset", "livesqlbench",
        "--gold-file", str(gold),
        "--mode", "one-shot",
        "--framework", framework,
        "--query-mode", "slayer",
        "--slayer-setup", "on-the-fly",
    ]
    kwargs = _drive_main(monkeypatch, argv)
    assert kwargs.get("framework") == framework
    assert kwargs.get("mode") == "one-shot"


# ---------------------------------------------------------------------------
# Programmatic bypass close (Codex #1) — run_evaluation / make_runner /
# run_one_task must also reject invalid combos.
# ---------------------------------------------------------------------------


# Parametrized invalid (mode, dataset, query_mode, framework, slayer_setup,
# expected-substr) combos — each one MUST be rejected by EACH of the three
# programmatic-entry-point validators (run_evaluation / make_runner /
# run_one_task) per Codex's "validate everywhere or someone bypasses you"
# rule.

_BAD_COMBOS = [
    pytest.param(
        # one-shot + mini-interact dataset
        dict(mode="one-shot", query_mode="slayer",
             framework="pydantic_ai_recursive",
             slayer_setup="on-the-fly", dataset="mini-interact"),
        ["one-shot"],
        id="one-shot+mini-interact",
    ),
    pytest.param(
        # one-shot + wrong framework
        dict(mode="one-shot", query_mode="slayer",
             framework="pydantic_ai",
             slayer_setup="on-the-fly", dataset="livesqlbench"),
        ["one-shot", "framework"],
        id="one-shot+wrong-framework",
    ),
    pytest.param(
        # one-shot + pre-encoded
        dict(mode="one-shot", query_mode="slayer",
             framework="pydantic_ai_recursive",
             slayer_setup="pre-encoded", dataset="livesqlbench"),
        ["one-shot", "on-the-fly"],
        id="one-shot+pre-encoded",
    ),
    pytest.param(
        # one-shot + raw
        dict(mode="one-shot", query_mode="raw",
             framework="pydantic_ai_recursive",
             slayer_setup="on-the-fly", dataset="livesqlbench"),
        ["one-shot"],
        id="one-shot+raw",
    ),
    pytest.param(
        # livesqlbench + a-interact
        dict(mode="a-interact", query_mode="slayer",
             framework="pydantic_ai_recursive",
             slayer_setup="on-the-fly", dataset="livesqlbench"),
        ["livesqlbench"],
        id="livesqlbench+a-interact",
    ),
]


@pytest.mark.parametrize("combo,must_contain", _BAD_COMBOS)
async def test_run_evaluation_rejects_invalid_combos(tmp_path, combo, must_contain):
    data = tmp_path / "data.jsonl"; data.write_text("")
    db_path = tmp_path / "ds"; db_path.mkdir()
    with pytest.raises(ValueError) as exc_info:
        await run_module.run_evaluation(
            data_path=str(data), data_dir=str(db_path),
            output_path=str(tmp_path / "out.json"),
            limit=0,
            **combo,
        )
    msg = str(exc_info.value).lower()
    for substr in must_contain:
        assert substr in msg, (
            f"run_evaluation rejection MUST mention {substr!r}; got: {msg!r}"
        )


@pytest.mark.parametrize("combo,must_contain", _BAD_COMBOS)
def test_make_runner_rejects_invalid_combos(combo, must_contain):
    # make_runner's signature differs from run_evaluation: it has no
    # `dataset` (it's a per-task runner factory). The "one-shot + mini-
    # interact" case is therefore enforced inside run_task (B3), not
    # here — skip combos that key only on `dataset`.
    if combo.get("dataset") == "mini-interact" and combo.get("mode") == "one-shot":
        pytest.skip("dataset-vs-mode guard lives in run_task, not make_runner")
    if combo.get("dataset") == "livesqlbench" and combo.get("mode") != "one-shot":
        pytest.skip("dataset gate handled by run_evaluation, not make_runner")
    with pytest.raises(ValueError) as exc_info:
        run_module.make_runner(
            agent_model="anthropic/claude-sonnet-4-5",
            strict=False, prompt_cache=True, max_depth=3,
            slayer_storage_root=None,
            mode=combo["mode"], query_mode=combo["query_mode"],
            framework=combo["framework"], slayer_setup=combo["slayer_setup"],
        )
    msg = str(exc_info.value).lower()
    for substr in must_contain:
        if substr in {"livesqlbench"}:  # only enforceable via dataset
            continue
        assert substr in msg


async def test_run_evaluation_rejects_one_shot_with_mini_interact(tmp_path):
    data = tmp_path / "data.jsonl"; data.write_text("")
    db_path = tmp_path / "ds"; db_path.mkdir()
    with pytest.raises(ValueError) as exc_info:
        await run_module.run_evaluation(
            data_path=str(data), data_dir=str(db_path),
            output_path=str(tmp_path / "out.json"),
            mode="one-shot", query_mode="slayer",
            framework="pydantic_ai_recursive",
            slayer_setup="on-the-fly",
            dataset="mini-interact",
            limit=0,
        )
    msg = str(exc_info.value)
    assert "one-shot" in msg and "livesqlbench" in msg


def test_make_runner_rejects_one_shot_with_wrong_framework():
    with pytest.raises(ValueError) as exc_info:
        run_module.make_runner(
            framework="pydantic_ai",  # not a slayer framework
            query_mode="slayer", mode="one-shot",
            agent_model="anthropic/claude-sonnet-4-5",
            strict=False, prompt_cache=True, max_depth=3,
            slayer_storage_root=None,
            slayer_setup="on-the-fly",
        )
    msg = str(exc_info.value)
    assert "one-shot" in msg and "framework" in msg


async def test_run_one_task_one_shot_rejects_missing_livesqlbench_marker(tmp_path):
    """The loader stamps `task["dataset"]="livesqlbench"`. A programmatic
    caller bypassing the loader must NOT silently get a one-shot run on
    an un-marked task (Codex #1 — programmatic-bypass close)."""
    db_path = tmp_path / "ds"; db_path.mkdir()
    # Task is missing the dataset marker.
    task = {
        "instance_id": "x1", "selected_database": "alien",
        "amb_user_query": "x", "sol_sql": ["SELECT 1"],
    }
    with pytest.raises(ValueError) as exc_info:
        await run_module.run_one_task(
            task_data=task, data_dir=str(db_path),
            framework="pydantic_ai_recursive",
            query_mode="slayer", mode="one-shot",
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            patience=3, strict=False, use_audited_gold_sql=False,
            prompt_cache=True, max_depth=3, slayer_storage_root=None,
            slayer_setup="on-the-fly",
        )
    msg = str(exc_info.value)
    assert "livesqlbench" in msg.lower() or "dataset" in msg.lower()


# ---------------------------------------------------------------------------
# Empty --filter-ids footgun (B3 hardening)
# ---------------------------------------------------------------------------


async def test_empty_filter_ids_fails_instead_of_running_full_set(tmp_path):
    """`run_evaluation(filter_ids=[])` with the empty-list footgun fix
    MUST raise, not silently expand scope to the full task set."""
    data = tmp_path / "data.jsonl"; data.write_text("")
    db_path = tmp_path / "ds"; db_path.mkdir()
    with pytest.raises(ValueError) as exc_info:
        await run_module.run_evaluation(
            data_path=str(data), data_dir=str(db_path),
            output_path=str(tmp_path / "out.json"),
            mode="a-interact", query_mode="raw",
            framework="pydantic_ai", limit=0,
            filter_ids=[],
        )
    assert "filter" in str(exc_info.value).lower()
