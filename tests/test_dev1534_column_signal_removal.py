"""DEV-1534 Fix A: column-NAMING signal removal (ORDER signal stays).

The pre-DEV-1534 schema carried both column-NAMING signals
(`column_name_match_case_insensitive`, `agent_columns`,
`best_variant_columns`) AND a column-ORDER signal
(`column_order_match` + the `column_order_mismatch` MissPattern).

Autopsy agents misattributed cascade-fails to "column naming mismatch"
when the real cause was value differences. The grader's correctness
tiers (N1-N3) use `_set_equal` on value tuples — column names are
irrelevant. N8 column-order tolerance IS a genuine tier and column
ORDER divergence IS a real cause.

Post-DEV-1534:
- `column_name_match_case_insensitive` removed from both
  `VariantInformational` and `MissDiagnostics`.
- `agent_columns` / `best_variant_columns` removed from
  `MissDiagnostics` (they're column-NAME lists).
- `column_order_match` stays on both models (positional ORDER signal).
- `column_order_mismatch` stays in `MissPattern` Literal.
- `_column_diff` / `_column_match_signals` return 2-tuple
  `(count_match, order_match)` — name_match dropped.

A backward-compat migration shim drops the named-legacy fields on
`model_validate` for both models, scoped per-model:
- `VariantInformational` strips only
  `column_name_match_case_insensitive`.
- `MissDiagnostics` strips all three named legacy keys.

Other unknown fields (typos, future renames) still raise
`ValidationError` under the unchanged `extra="forbid"`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Schema field absence
# ---------------------------------------------------------------------------


def test_variant_informational_drops_naming_signal():
    """`column_name_match_case_insensitive` is NOT a field on
    `VariantInformational` post-Fix-A."""
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    assert "column_name_match_case_insensitive" not in VariantInformational.model_fields


def test_variant_informational_keeps_column_order_match():
    """`column_order_match` is preserved (ORDER signal stays)."""
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    assert "column_order_match" in VariantInformational.model_fields
    assert "column_count_match" in VariantInformational.model_fields


def test_miss_diagnostics_drops_three_legacy_fields():
    """`column_name_match_case_insensitive`, `agent_columns`,
    `best_variant_columns` are removed from `MissDiagnostics`."""
    from bird_interact_agents.eval.annotation_schema import MissDiagnostics

    for f in (
        "column_name_match_case_insensitive",
        "agent_columns",
        "best_variant_columns",
    ):
        assert f not in MissDiagnostics.model_fields, (
            f"{f} should be removed from MissDiagnostics"
        )


def test_miss_diagnostics_keeps_column_order_match():
    from bird_interact_agents.eval.annotation_schema import MissDiagnostics

    assert "column_order_match" in MissDiagnostics.model_fields
    assert "column_count_match" in MissDiagnostics.model_fields


def test_miss_pattern_keeps_column_order_mismatch():
    """`column_order_mismatch` stays a valid MissPattern (ORDER is a
    real cause; the issue body said remove but the interview reversed
    this)."""
    from bird_interact_agents.eval.annotation_schema import MissPattern

    # `MissPattern = Literal[...]` — its `__args__` is the tuple of allowed values.
    allowed = set(getattr(MissPattern, "__args__", ()))
    assert "column_order_mismatch" in allowed
    assert "column_count_mismatch" in allowed


# ---------------------------------------------------------------------------
# Constructor / kwarg behaviour (after the migration shim runs)
# ---------------------------------------------------------------------------


def _variant_informational_kwargs(**overrides):
    """Minimal valid kwargs for VariantInformational construction."""
    base = dict(
        rowset_relation="equal_rowset",
        column_count_match=True,
        column_order_match=True,
        first_divergent_row_index=None,
        first_divergent_cell_diff=None,
    )
    base.update(overrides)
    return base


def _miss_diagnostics_kwargs(**overrides):
    """Minimal valid kwargs for MissDiagnostics construction."""
    base = dict(
        best_variant_id="primary",
        agent_row_count=0,
        best_variant_row_count=0,
        original_gold_row_count=None,
        overlap_with_best=0,
        rowset_relation_to_best="disjoint",
        agent_column_count=0,
        best_variant_column_count=0,
        column_count_match=True,
        column_order_match=True,
        first_divergent_cell_diff=None,
        agent_sql_parse_ok=True,
        best_variant_sql_parse_ok=True,
        agent_sql_parse_error=None,
        best_variant_sql_parse_error=None,
        agent_sql_executed_ok=True,
        agent_sql_error_excerpt=None,
        user_sim_n_asks=None,
        miss_patterns=[],
    )
    base.update(overrides)
    return base


def test_variant_informational_constructible_without_naming_field():
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    # Constructing without column_name_match_case_insensitive works.
    obj = VariantInformational(**_variant_informational_kwargs())
    assert obj.column_order_match is True
    assert not hasattr(obj, "column_name_match_case_insensitive")


def test_miss_diagnostics_constructible_without_legacy_fields():
    from bird_interact_agents.eval.annotation_schema import MissDiagnostics

    obj = MissDiagnostics(**_miss_diagnostics_kwargs())
    assert obj.column_order_match is True
    for f in (
        "column_name_match_case_insensitive",
        "agent_columns",
        "best_variant_columns",
    ):
        assert not hasattr(obj, f), f"{f} should not exist on MissDiagnostics"


# ---------------------------------------------------------------------------
# Migration shim — strip ONLY the named legacy keys, per model
# ---------------------------------------------------------------------------


def test_variant_informational_legacy_json_loads_dropping_naming():
    """Legacy JSON with `column_name_match_case_insensitive` loads OK;
    the field is silently stripped by the migration shim."""
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    legacy = dict(
        rowset_relation="equal_rowset",
        column_count_match=True,
        column_name_match_case_insensitive=True,  # legacy
        column_order_match=True,
        first_divergent_row_index=None,
        first_divergent_cell_diff=None,
    )
    obj = VariantInformational.model_validate(legacy)
    assert obj.column_order_match is True
    assert not hasattr(obj, "column_name_match_case_insensitive")


def test_miss_diagnostics_legacy_json_loads_dropping_three_legacy_keys():
    """Legacy JSON with all three named keys loads OK."""
    from bird_interact_agents.eval.annotation_schema import MissDiagnostics

    legacy = _miss_diagnostics_kwargs(
        column_name_match_case_insensitive=False,
        agent_columns=["a", "b"],
        best_variant_columns=["a", "b"],
    )
    obj = MissDiagnostics.model_validate(legacy)
    assert obj.column_order_match is True
    for f in (
        "column_name_match_case_insensitive",
        "agent_columns",
        "best_variant_columns",
    ):
        assert not hasattr(obj, f)


def test_variant_informational_shim_does_not_silently_accept_miss_diagnostics_keys():
    """Per-model allowlist regression: `VariantInformational.model_validate(
    {"agent_columns": []})` still raises because `agent_columns` was never a
    field on this model. Stripping it would mask a real shape error."""
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    legacy = _variant_informational_kwargs(agent_columns=[])
    with pytest.raises(ValidationError):
        VariantInformational.model_validate(legacy)


def test_variant_informational_shim_does_not_silently_accept_best_variant_columns():
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    legacy = _variant_informational_kwargs(best_variant_columns=[])
    with pytest.raises(ValidationError):
        VariantInformational.model_validate(legacy)


def test_miss_diagnostics_novel_unknown_still_rejected():
    """Per-model allowlist regression: a typo'd / future unknown field
    that ISN'T on the legacy strip list still raises ValidationError."""
    from bird_interact_agents.eval.annotation_schema import MissDiagnostics

    payload = _miss_diagnostics_kwargs(qux_typo_field=1)
    with pytest.raises(ValidationError):
        MissDiagnostics.model_validate(payload)


def test_variant_informational_novel_unknown_still_rejected():
    from bird_interact_agents.eval.annotation_schema import VariantInformational

    payload = _variant_informational_kwargs(some_typo_field=1)
    with pytest.raises(ValidationError):
        VariantInformational.model_validate(payload)


# ---------------------------------------------------------------------------
# Grader internal API — _column_diff / _column_match_signals return 2-tuple
# ---------------------------------------------------------------------------


def test_column_diff_returns_count_and_order_only():
    """Post-Fix-A `_column_diff` returns `(count_match, order_match)` —
    the middle `name_match_ci` element is dropped."""
    from bird_interact_agents.eval.tolerant_grader import _column_diff

    # Same names, same order → both True.
    cm, om = _column_diff(pred_cols=["a", "b"], gold_cols=["a", "b"])
    assert (cm, om) == (True, True)


def test_column_diff_count_mismatch_order_false():
    from bird_interact_agents.eval.tolerant_grader import _column_diff

    cm, om = _column_diff(pred_cols=["a"], gold_cols=["a", "b"])
    assert (cm, om) == (False, False)


def test_column_diff_same_set_different_order():
    from bird_interact_agents.eval.tolerant_grader import _column_diff

    cm, om = _column_diff(pred_cols=["b", "a"], gold_cols=["a", "b"])
    assert (cm, om) == (True, False)


def test_column_match_signals_returns_count_and_order_only():
    """Post-Fix-A `_column_match_signals` returns
    `(count_match, order_match)` — name_match dropped."""
    from bird_interact_agents.eval.tolerant_grader import _column_match_signals

    cm, om = _column_match_signals(agent_cols=["a", "b"], gold_cols=["a", "b"])
    assert (cm, om) == (True, True)


def test_column_match_signals_same_set_different_order():
    from bird_interact_agents.eval.tolerant_grader import _column_match_signals

    cm, om = _column_match_signals(agent_cols=["b", "a"], gold_cols=["a", "b"])
    assert (cm, om) == (True, False)


# ---------------------------------------------------------------------------
# Codex post-merge: column-order signal must not leak NAMING differences
# (e.g. SLayer's `<db>.<table>.<col>` namespacing vs gold's bare column
# names). Both informational helpers normalize via `_normalize_col` (lowercase
# + strip the longest dot-prefix) before comparing — otherwise `order_match`
# re-introduces the very naming signal Fix A removed.
# ---------------------------------------------------------------------------


def test_column_diff_namespaced_pred_vs_bare_gold_order_true():
    from bird_interact_agents.eval.tolerant_grader import _column_diff

    cm, om = _column_diff(
        pred_cols=["orders.status", "orders.amount"],
        gold_cols=["status", "amount"],
    )
    assert (cm, om) == (True, True)


def test_column_diff_namespaced_pred_vs_bare_gold_order_false_when_swapped():
    """Same SLayer/bare prefix difference, but the agent ALSO swapped
    column positions — `order_match` correctly reports False."""
    from bird_interact_agents.eval.tolerant_grader import _column_diff

    cm, om = _column_diff(
        pred_cols=["orders.amount", "orders.status"],
        gold_cols=["status", "amount"],
    )
    assert (cm, om) == (True, False)


def test_column_match_signals_namespaced_pred_vs_bare_gold_order_true():
    from bird_interact_agents.eval.tolerant_grader import _column_match_signals

    cm, om = _column_match_signals(
        agent_cols=["households.housenum", "households.income"],
        gold_cols=["housenum", "income"],
    )
    assert (cm, om) == (True, True)


def test_column_match_signals_case_only_difference_order_true():
    from bird_interact_agents.eval.tolerant_grader import _column_match_signals

    cm, om = _column_match_signals(
        agent_cols=["Status", "Amount"],
        gold_cols=["status", "amount"],
    )
    assert (cm, om) == (True, True)
