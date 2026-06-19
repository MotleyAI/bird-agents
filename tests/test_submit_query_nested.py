"""Nested-query support for `submit_query` (DEV-1435).

`submit_slayer_query` must accept two top-level JSON shapes:

* Single-stage: a JSON object validating as a `SlayerQuery`.
* Nested DAG: a JSON array of stage dicts; last element is the DAG root
  (same shape `query_nested` MCP tool accepts).

Anything else — top-level scalar, wrapped `{queries: [...]}`,
`{nested_queries: [...]}`, `{source_model: ..., queries: [...]}` — is
rejected up-front with a sharp error message naming the two valid
shapes, before `sql_sync` is ever called.

The brace for slayer's HTTP path (and any future slayer refactor): when
`client.sql_sync(list)` raises, fall back to
`client._engine.execute_sync(query=list, dry_run=True).sql` — the engine
has always accepted lists. The exception classes the brace must catch are
the union of:

* `AttributeError` — current slayer 0.6.8's HTTP path reaches
  `query.model_dump(...)` on a list and crashes with
  `AttributeError: 'list' object has no attribute 'model_dump'`. The
  in-process path on 0.6.8 actually works (engine handles list natively),
  but we cannot rely on that across slayer refactors.
* `pydantic.ValidationError` — what a refactored `query_sync` is likely
  to raise if it adds an explicit list check via Pydantic.
* `TypeError` — what an explicit `if not isinstance(query, (dict,
  SlayerQuery)): raise TypeError(...)` would emit.

The brace is removed once `pyproject.toml`'s slayer floor bumps past
DEV-1437.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from bird_interact_agents.harness import ACTION_COSTS


# ---------------------------------------------------------------------------
# Test fixtures: a `state` shaped like the real one (mirrors the helpers in
# tests/test_submit_helpers.py without importing private bits), and a
# selection of fake SlayerClients that mimic new-slayer, old-slayer, and
# old-slayer-without-engine behaviours.
# ---------------------------------------------------------------------------


class _FakeState(SimpleNamespace):
    def __init__(self, **kw):
        defaults = dict(
            status=SimpleNamespace(
                original_data={"selected_database": "fake_db"},
                remaining_budget=100.0,
                total_budget=100.0,
                force_submit=False,
            ),
            data_path_base="/tmp/ignored",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            user_sim_prompt_version="v2",
            slayer_storage_dir="",
            result=None,
        )
        defaults.update(kw)
        super().__init__(**defaults)


class _NewSlayerClient:
    """Slayer client whose `sql_sync` already accepts dict OR list — the
    post-DEV-1437 behaviour. The brace must NOT fire."""

    def __init__(self):
        self.calls: list = []
        # `_engine` is set so test failures don't masquerade as "brace fired
        # successfully": if the implementation accidentally falls into the
        # brace, this attribute lets us detect that the engine was poked
        # when it shouldn't have been.
        self._engine = SimpleNamespace(
            execute_sync=self._engine_should_not_be_called,
        )

    @staticmethod
    def _engine_should_not_be_called(*a, **kw):
        raise AssertionError(
            "Brace fired on new-slayer client; sql_sync handled the input."
        )

    def sql_sync(self, query):
        self.calls.append(query)
        if isinstance(query, list):
            return f"-- nested SQL from {len(query)} stages"
        return f"-- single-stage SQL from {query.get('source_model', '?')}"


class _OldSlayerClient:
    """Slayer client whose `sql_sync` rejects lists. Three plausible
    rejection modes the brace must cover:

    * `AttributeError` — slayer 0.6.8's HTTP path raises
      `'list' object has no attribute 'model_dump'` when it tries to
      serialise the list before posting.
    * `pydantic.ValidationError` — what a refactored `query_sync` would
      throw if it added an explicit list check via Pydantic.
    * `TypeError` — what an explicit `isinstance` check would raise.

    Defaults to `AttributeError` because that's what current slayer
    actually emits in the wild (HTTP path); the other two are
    forward-looking.
    """

    def __init__(self, *, raise_exc: Exception | None = None):
        self.calls: list = []
        self.engine_calls: list = []
        self._engine = SimpleNamespace(execute_sync=self._engine_execute_sync)
        self._raise_exc = raise_exc or AttributeError(
            "'list' object has no attribute 'model_dump'"
        )

    def sql_sync(self, query):
        self.calls.append(query)
        if isinstance(query, list):
            raise self._raise_exc
        return f"-- single-stage SQL from {query.get('source_model', '?')}"

    def _engine_execute_sync(self, *, query, dry_run: bool = False, explain: bool = False):
        self.engine_calls.append((query, dry_run, explain))
        assert dry_run is True, (
            "submit_query must request dry_run when reaching past sql_sync — "
            "we only want SQL, not execution."
        )
        # Mirror real slayer's engine: empty lists are rejected with a
        # clear ValueError. This is what bubbles up if the bird-interact
        # layer lets `[]` through to the engine.
        if isinstance(query, list) and not query:
            raise ValueError(
                "'query' must be a non-empty list when passing staged queries."
            )
        return SimpleNamespace(sql=f"-- engine SQL from {len(query)} stages")


class _OldSlayerClientWithoutEngine:
    """Hypothetical HTTP-mode client — `_engine is None`, so the brace
    cannot fire. submit_query must surface a clean translation error
    rather than crashing with AttributeError."""

    _engine = None

    def sql_sync(self, query):
        if isinstance(query, list):
            raise _make_validation_error()
        return "-- single-stage SQL"


class _Tiny(BaseModel):
    x: int


def _make_validation_error() -> ValidationError:
    """Build a real `pydantic.ValidationError`. Constructing one directly
    is awkward in pydantic v2, so we trigger a validation failure on a
    throwaway model."""
    try:
        _Tiny.model_validate({"x": "not an int"})
    except ValidationError as e:
        return e
    raise AssertionError("pydantic.model_validate failed to raise")


@pytest.fixture(autouse=True)
def _patch_side_effects(monkeypatch):
    """Stub out the BIRD-Interact evaluator + snapshot side effects so
    these tests stay hermetic. Each test gets a fresh patch — never any
    real DB access."""
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: (f"obs of {sql}", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)
    yield


# ---------------------------------------------------------------------------
# Happy path — single-stage dict (regression) + nested list (new).
# ---------------------------------------------------------------------------


def test_single_stage_dict_unaffected():
    """Existing dict-shape submissions keep working exactly the same way."""
    from bird_interact_agents.agents import _submit

    client = _NewSlayerClient()
    state = _FakeState()
    payload = '{"source_model": "orders", "measures": ["amount:sum"]}'

    out = _submit.submit_slayer_query(
        state, payload, slayer_client_factory=lambda _s: client,
    )

    assert "Generated SQL" in out
    assert "single-stage SQL from orders" in state.result["submitted_sql"]
    # submitted_query stays the raw JSON string — diagnostic schema unchanged.
    assert state.result["submitted_query"] == payload
    assert state.result["submission_status"] == "passed_phase2"
    # sql_sync called exactly once, with the parsed dict.
    assert len(client.calls) == 1
    assert client.calls[0] == {"source_model": "orders", "measures": ["amount:sum"]}


def test_nested_list_happy_path_new_slayer():
    """Top-level JSON array → sql_sync receives the parsed list. Brace
    must not fire; submitted_query is preserved as the raw string."""
    from bird_interact_agents.agents import _submit

    client = _NewSlayerClient()
    state = _FakeState()
    payload = json.dumps([
        {"name": "stage1", "source_model": "encounters",
         "measures": ["crisisint:sum"], "dimensions": ["clinid"]},
        {"source_model": "stage1", "measures": ["crisisint:sum"]},
    ])

    out = _submit.submit_slayer_query(
        state, payload, slayer_client_factory=lambda _s: client,
    )

    assert "Generated SQL" in out
    assert "nested SQL from 2 stages" in state.result["submitted_sql"]
    assert state.result["submitted_query"] == payload  # raw string preserved
    assert state.result["submission_status"] == "passed_phase2"
    assert len(client.calls) == 1
    assert isinstance(client.calls[0], list)
    assert len(client.calls[0]) == 2


def test_nested_list_charges_submit_budget_once():
    from bird_interact_agents.agents import _submit

    client = _NewSlayerClient()
    state = _FakeState()
    start = state.status.remaining_budget
    payload = json.dumps([{"name": "a", "source_model": "x"}, {"source_model": "a"}])

    out = _submit.submit_slayer_query(
        state, payload, slayer_client_factory=lambda _s: client,
    )

    assert state.status.remaining_budget == start - ACTION_COSTS["submit_query"]
    # One — and only one — budget note in the response (no double-charge).
    assert out.count("[Remaining budget:") == 1


# ---------------------------------------------------------------------------
# The brace: old-slayer fallback to engine.execute_sync when sql_sync rejects.
# ---------------------------------------------------------------------------


def test_brace_fires_on_attributeerror_for_list():
    """Current slayer 0.6.8's HTTP path raises AttributeError when it
    reaches `query.model_dump(...)` on a list. The brace must catch
    that exception class — this is the real-world rejection mode."""
    from bird_interact_agents.agents import _submit

    client = _OldSlayerClient()  # default rejection = AttributeError
    state = _FakeState()
    payload = json.dumps([
        {"name": "stage1", "source_model": "encounters"},
        {"source_model": "stage1"},
    ])

    _submit.submit_slayer_query(
        state, payload, slayer_client_factory=lambda _s: client,
    )

    assert len(client.engine_calls) == 1
    query_passed, dry_run_flag, _ = client.engine_calls[0]
    assert dry_run_flag is True
    assert isinstance(query_passed, list) and len(query_passed) == 2
    assert "engine SQL from 2 stages" in state.result["submitted_sql"]
    assert state.result["submission_status"] == "passed_phase2"


def test_brace_fires_on_validationerror_for_list():
    """Forward-looking: a refactored slayer may raise ValidationError
    instead of AttributeError. The brace must cover that class too."""
    from bird_interact_agents.agents import _submit

    client = _OldSlayerClient(raise_exc=_make_validation_error())
    state = _FakeState()
    payload = json.dumps([
        {"name": "stage1", "source_model": "encounters"},
        {"source_model": "stage1"},
    ])

    out = _submit.submit_slayer_query(
        state, payload, slayer_client_factory=lambda _s: client,
    )

    # Brace fired — engine was called exactly once with dry_run=True.
    assert len(client.engine_calls) == 1
    query_passed, dry_run_flag, _ = client.engine_calls[0]
    assert dry_run_flag is True
    assert isinstance(query_passed, list) and len(query_passed) == 2
    # And the SQL it returned landed on state.result.
    assert "engine SQL from 2 stages" in state.result["submitted_sql"]
    assert state.result["submission_status"] == "passed_phase2"
    assert "Generated SQL" in out


def test_brace_fires_on_typeerror_for_list():
    """Same as above, but the old slayer raises TypeError instead. Both
    must trigger the brace — we don't lock the precise exception class."""
    from bird_interact_agents.agents import _submit

    client = _OldSlayerClient(raise_exc=TypeError("expected SlayerQuery or dict"))
    state = _FakeState()
    payload = json.dumps([{"name": "a", "source_model": "x"}, {"source_model": "a"}])

    _submit.submit_slayer_query(
        state, payload, slayer_client_factory=lambda _s: client,
    )

    assert len(client.engine_calls) == 1
    assert "engine SQL" in state.result["submitted_sql"]


def test_brace_does_not_fire_for_dict_validationerror():
    """A dict that's not a valid SlayerQuery (real user typo / unrelated
    error) must continue to land in `translation_error` — the brace is
    only for the nested-list workaround, not a catch-all."""
    from bird_interact_agents.agents import _submit

    class _ClientRejectsAllDicts:
        _engine = SimpleNamespace(
            execute_sync=lambda **_: pytest.fail(
                "engine fallback fired for dict path — brace scope leaked"
            )
        )

        def sql_sync(self, _q):
            raise _make_validation_error()

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state, '{"source_model": "orders", "measures": "bad-type"}',
        slayer_client_factory=lambda _s: _ClientRejectsAllDicts(),
    )

    assert state.result["submission_status"] == "translation_error"
    assert state.result["submitted_sql"] is None
    assert "Could not generate SQL" in out
    # Per DEV-1432: deterministic pre-eval failures are FREE.
    # `execute_submit_action` was never reached.
    assert state.status.remaining_budget == start


@pytest.mark.parametrize(
    "raise_exc",
    [
        AttributeError("'list' object has no attribute 'model_dump'"),
        _make_validation_error(),
        TypeError("expected SlayerQuery or dict"),
    ],
    ids=["attributeerror", "validationerror", "typeerror"],
)
def test_brace_with_no_engine_returns_translation_error(raise_exc):
    """HTTP-mode-style client where `_engine is None`. The brace cannot
    reach the engine — we must surface a clean translation error, not
    crash with the underlying exception. Parameterised over every
    exception class the brace catches so an over-narrow `except` is
    caught here."""
    from bird_interact_agents.agents import _submit

    class _NoEngineClient(_OldSlayerClientWithoutEngine):
        def sql_sync(self, query):
            if isinstance(query, list):
                raise raise_exc
            return "-- single-stage SQL"

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state,
        json.dumps([{"name": "a", "source_model": "x"}, {"source_model": "a"}]),
        slayer_client_factory=lambda _s: _NoEngineClient(),
    )

    assert state.result["submission_status"] == "translation_error"
    assert state.result["submitted_sql"] is None
    assert "Could not generate SQL" in out
    # Per DEV-1432: deterministic pre-eval failures are FREE. The
    # translation never reached `execute_submit_action`, so budget
    # is untouched even after multiple failed submission attempts.
    assert state.status.remaining_budget == start


def test_brace_engine_failure_propagates_as_translation_error():
    """If the engine ALSO fails (e.g., cycle in the DAG), the user gets
    a translation_error with the engine's message — not an opaque crash."""
    from bird_interact_agents.agents import _submit

    class _EngineThatFails:
        @staticmethod
        def execute_sync(*, query, dry_run, explain=False):
            raise ValueError("cycle detected in nested DAG: stage1 -> stage1")

    class _ClientBraceAndFail:
        _engine = _EngineThatFails

        def sql_sync(self, _q):
            raise _make_validation_error()

    state = _FakeState()
    out = _submit.submit_slayer_query(
        state,
        json.dumps([{"name": "stage1", "source_model": "stage1"}]),
        slayer_client_factory=lambda _s: _ClientBraceAndFail(),
    )

    assert state.result["submission_status"] == "translation_error"
    assert state.result["submitted_sql"] is None
    assert "cycle detected" in out


# ---------------------------------------------------------------------------
# Sharp shape-error rejection — wrapped variants + non-object/non-array.
# ---------------------------------------------------------------------------


WRAPPED_SHAPES = [
    pytest.param(
        {"queries": [{"source_model": "x"}]},
        id="wrapped_queries",
    ),
    pytest.param(
        {"nested_queries": [{"source_model": "x"}]},
        id="wrapped_nested_queries",
    ),
    pytest.param(
        {"source_model": "nested", "queries": [{"source_model": "x"}]},
        id="wrapped_with_source_model_and_queries",
    ),
]


@pytest.mark.parametrize("wrapped", WRAPPED_SHAPES)
def test_wrapped_shapes_rejected_with_sharp_message(wrapped):
    """The four shapes the mental_4 agent burned bird-coins on are all
    detected before sql_sync sees them. The response must name both
    valid top-level shapes AND show a concrete shape hint so the agent
    self-corrects without external prompt help."""
    from bird_interact_agents.agents import _submit

    def _factory(_s):
        pytest.fail("sql_sync must not be called for wrapped-shape inputs")

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state, json.dumps(wrapped), slayer_client_factory=_factory,
    )

    # Both valid top-level shapes named in the error.
    lo = out.lower()
    assert "object" in lo, f"sharp error must say 'object' for single-stage shape; got: {out!r}"
    assert "array" in lo or "list" in lo, (
        f"sharp error must say 'array' or 'list' for nested shape; got: {out!r}"
    )
    # Concrete shape hint — the message must show a SlayerQuery field
    # name so the agent sees an example, not just two abstract terms.
    assert "source_model" in lo, (
        f"sharp error must include the literal `source_model` so the "
        f"agent has a concrete shape hint; got: {out!r}"
    )
    # Sharp message reuses the json_error bucket (per Q4 decision).
    assert state.result["submission_status"] == "json_error"
    assert state.result["submitted_sql"] is None
    # Per DEV-1432: deterministic pre-eval failures are FREE. The wrapped-
    # shape rejection never reaches `execute_submit_action`, so budget is
    # untouched.
    assert state.status.remaining_budget == start


@pytest.mark.parametrize("scalar_json", ['"oops"', "42", "true", "null"])
def test_top_level_scalars_rejected_with_sharp_message(scalar_json):
    from bird_interact_agents.agents import _submit

    def _factory(_s):
        pytest.fail("sql_sync must not be called for scalar inputs")

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state, scalar_json, slayer_client_factory=_factory,
    )

    lo = out.lower()
    assert "object" in lo
    assert "array" in lo or "list" in lo
    assert "source_model" in lo
    assert state.result["submission_status"] == "json_error"
    # Per DEV-1432: deterministic pre-eval failures are FREE.
    assert state.status.remaining_budget == start


def test_empty_list_rejected_or_translation_error():
    """An empty JSON array `[]` is a degenerate DAG. Implementations
    may reject it up-front as a shape error OR let the engine surface
    its own "non-empty list required" message — either way the submit
    must not succeed AND no budget is charged (pre-eval failure)."""
    from bird_interact_agents.agents import _submit

    state = _FakeState()
    start = state.status.remaining_budget
    client = _OldSlayerClient()  # has a working engine
    out = _submit.submit_slayer_query(
        state, "[]", slayer_client_factory=lambda _s: client,
    )

    assert state.result["submitted_sql"] is None
    assert state.result["submission_status"] in {"json_error", "translation_error"}
    # The response must NOT claim "Generated SQL" — that header is only
    # for a successful translation.
    assert "Generated SQL" not in out
    # Per DEV-1432: pre-eval failures are FREE.
    assert state.status.remaining_budget == start


def test_malformed_json_still_records_json_error():
    """Regression: the existing bad-JSON path must not change. A literal
    `{not json` was always a json_error and still is."""
    from bird_interact_agents.agents import _submit

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state, "{not json",
        slayer_client_factory=lambda _s: pytest.fail("must not be called"),
    )
    assert state.result["submission_status"] == "json_error"
    assert state.result["submitted_sql"] is None
    assert "Invalid JSON" in out
    # Per DEV-1432: deterministic pre-eval failures (including this
    # JSON-parse error) are FREE. `execute_submit_action` never ran.
    assert state.status.remaining_budget == start


# ---------------------------------------------------------------------------
# Tool spec + prompt teaching surface.
# ---------------------------------------------------------------------------


def test_submit_query_spec_description_teaches_both_shapes():
    """The agent-facing tool description must name both valid input
    shapes — single-stage object and nested array."""
    from bird_interact_agents.agents._tool_specs import SUBMIT_QUERY_SPEC

    desc = SUBMIT_QUERY_SPEC.description.lower()
    # "object" / "single" for the single-stage shape.
    assert "single" in desc or "object" in desc
    # "array" / "list" / "nested" for the DAG shape.
    assert "array" in desc or "list" in desc or "nested" in desc


def _assert_prompt_teaches_nested_shape(body: str, *, prompt_name: str) -> None:
    """Shared assertion: the prompt must (a) name the nested shape and
    (b) show a concrete example near *some* `submit_query` mention so
    the agent has a literal pattern to copy, not just two abstract
    terms.

    A "concrete example" means: within 600 chars after at least one
    `submit_query` mention, the prompt contains BOTH `[` (the start of
    a JSON array) AND `source_model` (a SlayerQuery field name). That's
    what closes the gap from "you can submit nested" to "here is what
    nested looks like" — the latter is what prevents an agent from
    guessing wrapped shapes the way mental_4 did. Scanning every
    mention (not just the first) avoids locking the example to a
    specific spot in the prompt."""
    lo = body.lower()
    assert "nested" in lo, f"{prompt_name} never mentions 'nested'"
    assert "array" in lo or "list" in lo, (
        f"{prompt_name} doesn't link nested to an array/list shape"
    )

    found_example = False
    cursor = 0
    while True:
        idx = lo.find("submit_query", cursor)
        if idx < 0:
            break
        near = body[idx : idx + 600]
        if "[" in near and "source_model" in near:
            found_example = True
            break
        cursor = idx + 1

    assert found_example, (
        f"{prompt_name} doesn't show BOTH `[` (JSON array opener) and "
        f"`source_model` within 600 chars after any `submit_query` "
        f"mention — agent has no literal pattern to copy. The prompt "
        f"must include a concrete nested-array example near at least "
        f"one submit_query mention."
    )


def test_pydantic_ai_recursive_constructor_prompt_teaches_nested_shape():
    """The query-constructor prompt must explicitly teach that a JSON
    array is the way to submit nested queries — otherwise the agent
    falls back to guessing wrapped shapes (the mental_4 failure mode)."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts as p

    _assert_prompt_teaches_nested_shape(
        p.QUERY_CONSTRUCTOR_PROMPT, prompt_name="QUERY_CONSTRUCTOR_PROMPT",
    )


def test_claude_sdk_slayer_prompt_teaches_nested_shape():
    """Same teaching directive on the claude_sdk side."""
    from bird_interact_agents.agents.claude_sdk import prompts as p

    _assert_prompt_teaches_nested_shape(
        p.SLAYER_A_INTERACT, prompt_name="SLAYER_A_INTERACT",
    )


# ---------------------------------------------------------------------------
# Adapter integration smoke — confirm the per-adapter `submit_query` wrapper
# passes a nested JSON string through to `submit_slayer_query` unchanged.
# The bulk of behaviour is tested at the helper level above; this guards
# against a wrapper-side parameter-binding regression (the claude_sdk path is
# the most fragile because it unpacks `args["query_json"]` from a dict).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_sdk_submit_query_routes_nested_through_helper(monkeypatch):
    """The claude_sdk `submit_query` tool receives a dict with key
    `query_json` and must pass the nested-array string straight into
    `submit_slayer_query`. Mocking the slayer client lets us assert
    the helper got the raw payload."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.agents import _submit
    from bird_interact_agents.harness import SampleStatus

    seen: dict = {}
    sentinel_sql = "-- nested SQL via claude_sdk wrapper"

    class _Client:
        _engine = None

        def sql_sync(self, q):
            seen["query"] = q
            return sentinel_sql

    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: (f"obs of {sql}", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0,
            original_data={
                "selected_database": "alien",
                "knowledge_ambiguity": [],
                "instance_id": "alien_1",
            },
            remaining_budget=20.0,
            total_budget=20.0,
        ),
        "data_path_base": "/tmp/ignored",
        "slayer_storage_dir": "",
        "_slayer_client": _Client(),
        "_slayer_storage": None,
        "result": None,
    })

    queries = [
        {"name": "stage1", "source_model": "encounters"},
        {"source_model": "stage1", "measures": ["crisisint:sum"]},
    ]
    # DEV-1555 CR r1 unification: pass nested-DAG via the structured
    # `queries` arg (wrapper builds the JSON internally).
    result = await agent_mod.submit_query.handler({"queries": queries})

    # The wrapper built a JSON array from `queries` and forwarded it
    # to submit_slayer_query, which parsed it as a list.
    assert isinstance(seen.get("query"), list)
    assert len(seen["query"]) == 2
    text = result["content"][0]["text"]
    assert sentinel_sql in text
