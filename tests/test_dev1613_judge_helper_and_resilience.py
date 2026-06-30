"""DEV-1613: the shared ``run_novel_reading_judge`` helper, the
``build_inline_judge`` factory, and ``LiteLLMJudge`` 429-resilience.

These are mechanical contract tests (stub judges, mocked litellm) per the
project convention — no prompt-substring assertions.
"""
from __future__ import annotations

import sys
import types
from typing import List, Optional

from tests._dev1613_helpers import _make_task_annotation


class _RecordingJudge:
    def __init__(self, accept: Optional[bool]):
        self.accept = accept
        self.calls: List[dict] = []

    def judge(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return self.accept


# ---------------------------------------------------------------------------
# run_novel_reading_judge — gating matrix
# ---------------------------------------------------------------------------


def test_helper_fires_for_insufficient_and_returns_accept():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    ann = _make_task_annotation(verdict="insufficient", evaluator_prompt="rules")
    judge = _RecordingJudge(accept=True)
    out = run_novel_reading_judge(
        task_annotation=ann,
        audited_gold_rows=[],
        submitted_sql="SELECT 1",
        predicted_rows_head=[(1,)],
        llm_judge=judge,
    )
    assert out is True
    assert len(judge.calls) == 1


def test_helper_returns_false_when_judge_rejects():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    ann = _make_task_annotation(verdict="insufficient")
    judge = _RecordingJudge(accept=False)
    out = run_novel_reading_judge(
        task_annotation=ann, audited_gold_rows=[], submitted_sql="S",
        predicted_rows_head=[], llm_judge=judge,
    )
    assert out is False


def test_helper_skips_for_sufficient_verdict_without_calling_judge():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    ann = _make_task_annotation(verdict="sufficient")
    judge = _RecordingJudge(accept=True)
    out = run_novel_reading_judge(
        task_annotation=ann, audited_gold_rows=[], submitted_sql="S",
        predicted_rows_head=[], llm_judge=judge,
    )
    assert out is None
    assert judge.calls == []  # gate held — no paid call


def test_helper_skips_when_no_evaluator_prompt():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    ann = _make_task_annotation(verdict="insufficient", evaluator_prompt=None)
    judge = _RecordingJudge(accept=True)
    out = run_novel_reading_judge(
        task_annotation=ann, audited_gold_rows=[], submitted_sql="S",
        predicted_rows_head=[], llm_judge=judge,
    )
    assert out is None
    assert judge.calls == []


def test_helper_returns_none_when_no_judge():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    ann = _make_task_annotation(verdict="insufficient")
    out = run_novel_reading_judge(
        task_annotation=ann, audited_gold_rows=[], submitted_sql="S",
        predicted_rows_head=[], llm_judge=None,
    )
    assert out is None


def test_helper_swallows_judge_exception_to_none():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    class _BoomJudge:
        def judge(self, **kwargs):  # noqa: ANN003
            raise RuntimeError("429 boom")

    ann = _make_task_annotation(verdict="insufficient")
    out = run_novel_reading_judge(
        task_annotation=ann, audited_gold_rows=[], submitted_sql="S",
        predicted_rows_head=[], llm_judge=_BoomJudge(),
    )
    assert out is None  # never raises; falls through to deterministic tiers


def test_helper_forwards_predicted_rows_to_judge():
    from bird_interact_agents.eval.tolerant_grader import run_novel_reading_judge

    ann = _make_task_annotation(verdict="insufficient")
    judge = _RecordingJudge(accept=True)
    rows = [(1, "a"), (2, "b")]
    run_novel_reading_judge(
        task_annotation=ann, audited_gold_rows=[], submitted_sql="S",
        predicted_rows_head=rows, llm_judge=judge,
    )
    assert judge.calls[0]["predicted_rows_head"] == rows


# ---------------------------------------------------------------------------
# build_inline_judge — bare LiteLLMJudge factory for cloud + local inline
# ---------------------------------------------------------------------------


def test_build_inline_judge_returns_bare_litellm_for_model():
    from bird_interact_agents.eval.grade_in_place import build_inline_judge
    from bird_interact_agents.eval.tolerant_grader import (
        CachedLLMJudge,
        LiteLLMJudge,
    )

    judge = build_inline_judge("anthropic/claude-opus-4-7")
    assert isinstance(judge, LiteLLMJudge)
    # Bare, NOT cache-wrapped (cloud worker has no stable per-run cache dir).
    assert not isinstance(judge, CachedLLMJudge)
    assert judge.model_name == "anthropic/claude-opus-4-7"


def test_build_inline_judge_none_for_empty_model():
    from bird_interact_agents.eval.grade_in_place import build_inline_judge

    assert build_inline_judge(None) is None
    assert build_inline_judge("") is None


# ---------------------------------------------------------------------------
# LiteLLMJudge — 429 resilience via litellm built-in retries
# ---------------------------------------------------------------------------


def test_litellm_judge_passes_num_retries(monkeypatch):
    """The judge must hand litellm a num_retries so transient 429s retry
    with litellm's built-in backoff rather than immediately falling
    through to None."""
    from bird_interact_agents.eval.tolerant_grader import LiteLLMJudge

    recorded: dict = {}

    def _fake_completion(**kwargs):  # noqa: ANN003
        recorded.update(kwargs)
        return {"choices": [{"message": {"content": "reasoning...\nACCEPT"}}]}

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = _fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    judge = LiteLLMJudge(model="anthropic/claude-opus-4-7")
    out = judge.judge(
        evaluator_prompt="rules", gold_variants_summary=[],
        metadata_anchors=[], submitted_sql="S", predicted_rows_head=[],
    )
    assert out is True
    assert recorded.get("num_retries") is not None
    assert int(recorded["num_retries"]) >= 1
