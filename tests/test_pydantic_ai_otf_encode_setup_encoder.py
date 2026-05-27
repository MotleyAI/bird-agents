"""Tests for the setup encoder ``agents.pydantic_ai_otf_encode.setup_encoder``
(DEV-1454).

The setup encoder is the per-DB, build-time encoder. Unlike the task-time
``KB_ENCODER`` it has **no `ask_user`** (there's no task to clarify against):
it encodes when confident and DEFERS otherwise. Its results are verified to
(a) actually exist in storage and (b) carry ``meta.kb_id`` — the sole key
HARD-8 deletion uses; an untagged or missing entity downgrades the KB to
``deferred``.
"""

from __future__ import annotations

from slayer.core.models import Column, DatasourceConfig, SlayerModel
from slayer.storage.yaml_storage import YAMLStorage


DB = "tinydb"


def test_setup_encoder_has_no_ask_user_or_clarifier_tools():
    """Tool surface: SLayer MCP read/write tools come via the toolset; the
    native function-toolset must NOT carry ask_user / submit_query /
    spawn_subagent / kb_to_slayer."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        _build_setup_encoder,
    )

    agent = _build_setup_encoder(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    tool_names = set(dict(agent._function_toolset.tools).keys())
    assert "ask_user" not in tool_names
    assert "submit_query" not in tool_names
    assert "spawn_subagent" not in tool_names
    assert "kb_to_slayer" not in tool_names


async def _storage_with_model(tmp_path, *, tagged_kb_id=None, col_name="c"):
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    cols = [Column(name="id", primary_key=True)]
    meta = {"kb_id": tagged_kb_id} if tagged_kb_id is not None else {}
    cols.append(Column(name=col_name, sql="1", meta=meta))
    await storage.save_model(SlayerModel(
        name="households", data_source=DB, sql_table="households", columns=cols,
    ))
    return storage


async def test_verify_downgrades_when_claimed_entity_missing(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        _verify_setup_result,
    )

    storage = await _storage_with_model(tmp_path, tagged_kb_id=None, col_name="c")
    result = EncoderResult(
        kb_id=5, status="encoded",
        entities=[EncodedEntity(kind="column", host_model="households",
                                name="ghost", entity_ref=f"{DB}.households.ghost")],
        notes="claimed but never written",
    )
    out = await _verify_setup_result(result, storage, DB)
    assert out.status == "deferred"


async def test_verify_downgrades_when_entity_untagged(tmp_path):
    """The entity exists but carries no ``meta.kb_id`` — HARD-8 deletion
    could never mask it, so the KB is downgraded to deferred."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        _verify_setup_result,
    )

    storage = await _storage_with_model(tmp_path, tagged_kb_id=None, col_name="c")
    result = EncoderResult(
        kb_id=5, status="encoded",
        entities=[EncodedEntity(kind="column", host_model="households",
                                name="c", entity_ref=f"{DB}.households.c")],
        notes="written but untagged",
    )
    out = await _verify_setup_result(result, storage, DB)
    assert out.status == "deferred"


async def test_verify_passes_when_tagged_and_present(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        _verify_setup_result,
    )

    storage = await _storage_with_model(tmp_path, tagged_kb_id=5, col_name="c")
    result = EncoderResult(
        kb_id=5, status="encoded",
        entities=[EncodedEntity(kind="column", host_model="households",
                                name="c", entity_ref=f"{DB}.households.c")],
        notes="properly tagged",
    )
    out = await _verify_setup_result(result, storage, DB)
    assert out.status == "encoded"


async def test_verify_leaves_deferred_results_untouched(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        _verify_setup_result,
    )

    storage = await _storage_with_model(tmp_path, tagged_kb_id=5)
    result = EncoderResult(
        kb_id=5, status="deferred", entities=[], notes="ambiguous",
        clarifying_questions=["q?"],
    )
    out = await _verify_setup_result(result, storage, DB)
    assert out.status == "deferred"
    assert out.clarifying_questions == ["q?"]


# ---------------------------------------------------------------------------
# Shared-MCP-server lifecycle (regression: anyio cancel scope)
# ---------------------------------------------------------------------------


async def test_shared_server_entered_and_exited_in_same_task(monkeypatch):
    """Regression for "Attempted to exit cancel scope in a different task than
    it was entered in".

    ``MCPServerStdio.__aenter__`` opens an anyio ``stdio_client`` whose cancel
    scope is bound to the entering task. The old lazy-entry ``__aenter__``'d the
    shared server inside whichever ``gather``'d child task won an entry lock,
    while ``aclose()`` ``__aexit__``'d from the parent — different tasks, so
    anyio raised at teardown and the whole reference build crashed (losing the
    encode). The fix: ``aopen()`` enters in the parent BEFORE the fan-out, and
    ``aclose()`` exits in the same parent task.

    This drives ``run_one`` through ``asyncio.gather`` (real child tasks) and
    asserts the server is entered exactly once and that the enter/exit happen in
    the SAME task — and that that task is the parent, not a child runner.
    """
    import asyncio

    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    events: list[tuple] = []

    class _FakeServer:
        async def __aenter__(self):
            events.append(("enter", id(asyncio.current_task())))
            return self

        async def __aexit__(self, *exc):
            events.append(("exit", id(asyncio.current_task())))
            return False

    server = _FakeServer()
    # Avoid building a real pydantic-ai Agent / hitting an LLM.
    monkeypatch.setattr(
        setup_encoder, "_build_setup_encoder", lambda **kw: object(),
    )

    async def _fake_run(*, agent, kb_id, kb_body, deps_results, storage, db,
                        self_model_id="unknown", sessions_dir=None,
                        index_rows=None, reverse_deps=None):
        events.append(("run", kb_id, id(asyncio.current_task())))
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    monkeypatch.setattr(setup_encoder, "_run_setup_encoder", _fake_run)

    build_encoder = setup_encoder.make_setup_build_encoder(
        model=object(), model_settings=None, self_model_id="m",
        build_shared_slayer_server=lambda _dir: server,
    )
    run_one = build_encoder(storage=object(), build_dir="/x", db=DB)

    parent_id = id(asyncio.current_task())
    # Parent enters the server, fans out run_one in child tasks, then exits —
    # mirroring reference_build._build_reference.
    await run_one.aopen()
    await asyncio.gather(
        run_one(1, {"id": 1, "knowledge": "k1"}, []),
        run_one(2, {"id": 2, "knowledge": "k2"}, []),
    )
    await run_one.aclose()

    enters = [e for e in events if e[0] == "enter"]
    exits = [e for e in events if e[0] == "exit"]
    runs = [e for e in events if e[0] == "run"]

    assert len(enters) == 1, "shared server must be __aenter__'d exactly once"
    assert len(exits) == 1
    # The bug was enters[0] task != exits[0] task. Both must be the parent.
    assert enters[0][1] == exits[0][1] == parent_id
    # run_one bodies ran in child tasks (gather), distinct from the parent —
    # the structural condition that made the old code crash at teardown.
    assert len(runs) == 2
    assert all(r[2] != parent_id for r in runs)


# ---------------------------------------------------------------------------
# Capture-on-failure (the load-bearing reason for agent.iter() over .run())
# ---------------------------------------------------------------------------


async def test_run_setup_encoder_captures_partial_trajectory_on_failure(tmp_path):
    """When the run raises mid-flight (e.g. UsageLimitExceeded), `.run()` would
    discard the trajectory; `agent.iter()` keeps it. `_run_setup_encoder` must
    (a) never raise — return EncoderResult(status="error", error=...), and
    (b) still write the per-kb session file with the PARTIAL trace + the error,
    because those are exactly the sessions worth inspecting (KB 10/34/35)."""
    from pydantic_ai.messages import (
        ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart,
    )

    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder

    class _FakeUsageLimitExceeded(Exception):
        pass

    # A realistic partial history: the user prompt + one model turn that issued
    # a search tool call, then the run blew the request budget.
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Encode KB 7 per the recipe.")]),
        ModelResponse(parts=[
            TextPart(content="searching the KB for prior art"),
            ToolCallPart(tool_name="search", args="{}"),
        ]),
    ]

    class _Run:
        def __init__(self):
            self.result = None  # no valid EncoderResult was produced

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise _FakeUsageLimitExceeded(
                "The next request would exceed the request_limit of 120"
            )

        def all_messages(self):
            return messages

        def usage(self):
            return None

    class _CM:
        async def __aenter__(self):
            return _Run()

        async def __aexit__(self, *exc):
            return False

    class _Agent:
        def iter(self, **kw):
            return _CM()

    final = await setup_encoder._run_setup_encoder(
        agent=_Agent(), kb_id=7, kb_body="KB 7 — body", deps_results=[],
        storage=object(), db="households", sessions_dir=tmp_path,
    )

    # (a) never raised; surfaced as a non-fatal error result.
    assert final.status == "error"
    assert "request_limit" in (final.error or "")

    # (b) the session file exists with the error in the HEADER + the partial
    # trace in the step section.
    md = (tmp_path / "setup__kb_007.md").read_text()
    header, sep, trace = md.partition("## Step trace")
    assert sep, "session md must have a step-trace section"
    assert "status: error" in header
    assert "- error:" in header
    assert "request_limit" in header
    assert "search" in trace  # the partial trajectory was captured, not discarded


# ---------------------------------------------------------------------------
# submit_encoding result capture (DEV-1454): the encoder now reasons in text and
# delivers its EncoderResult via the submit_encoding tool into per-run deps,
# instead of pydantic-ai structured output. The runner reads deps, not run.output.
# ---------------------------------------------------------------------------


class _SubmittingRun:
    """A fake AgentRun that simulates the model calling submit_encoding (setting
    ``deps.encoder_submission``), optionally raising afterwards."""

    def __init__(self, deps, submission, raise_after):
        self._deps = deps
        self._submission = submission
        self._raise_after = raise_after
        self.result = None  # the structured output no longer carries the result

    def __aiter__(self):
        return self

    async def __anext__(self):
        # the model "calls submit_encoding" → captured into per-run deps
        self._deps.encoder_submission = self._submission
        if self._raise_after:
            raise RuntimeError("would exceed the request_limit of 120")
        raise StopAsyncIteration

    def all_messages(self):
        return []

    def usage(self):
        return None


class _SubmittingCM:
    def __init__(self, deps, submission, raise_after):
        self._args = (deps, submission, raise_after)

    async def __aenter__(self):
        return _SubmittingRun(*self._args)

    async def __aexit__(self, *exc):
        return False


class _SubmittingAgent:
    """Fake agent whose iter() captures the deps it's handed and simulates a
    submit_encoding call on it."""

    def __init__(self, submission, *, raise_after=False):
        self._submission = submission
        self._raise_after = raise_after

    def iter(self, **kw):
        return _SubmittingCM(kw.get("deps"), self._submission, self._raise_after)


async def test_run_setup_encoder_returns_submitted_result(tmp_path):
    """When the model submits via submit_encoding, the runner returns that
    EncoderResult (read from deps, not run.output)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    submission = EncoderResult(kb_id=7, status="deferred", notes="needs a threshold")
    final = await setup_encoder._run_setup_encoder(
        agent=_SubmittingAgent(submission), kb_id=7, kb_body="KB 7 — body",
        deps_results=[], storage=object(), db="households", sessions_dir=tmp_path,
    )
    assert final.status == "deferred"
    assert final.kb_id == 7
    assert "threshold" in final.notes


# ---------------------------------------------------------------------------
# deps-block reflects the LIVE datasource, not a stale in-memory status
# (DEV-1454): the setup pass can leave a dep's column persisted + tagged while
# its EncoderResult.status is a (raced) "deferred". The old deps-block reported
# it as "NOT encoded", which deadlocked the dependent KB (KB 13). The block must
# instead report the live entity.
# ---------------------------------------------------------------------------


async def _storage_with_tagged_column(tmp_path, *, model="infrastructure",
                                      col="roadsurface_score", kb_id=4):
    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name=model, data_source=DB, sql_table=model,
        columns=[Column(name="id", primary_key=True),
                 Column(name=col, sql="1", meta={"kb_id": kb_id})],
    ))
    return storage


async def test_deps_block_reflects_live_storage_over_stale_status(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_deps_block,
    )

    storage = await _storage_with_tagged_column(tmp_path, kb_id=4)
    # A FALSE "deferred" with no entities (the verify torn-read race) …
    dep = EncoderResult(kb_id=4, status="deferred", entities=[], notes="raced")
    block = await _format_deps_block([dep], storage, DB)
    # … must NOT be reported as not-encoded; the live tagged column wins.
    assert "NOT encoded" not in block
    assert f"{DB}.infrastructure.roadsurface_score" in block
    assert "KB 4" in block


async def test_deps_block_marks_genuinely_missing_dep_not_encoded(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_deps_block,
    )

    storage = await _storage_with_tagged_column(tmp_path, kb_id=4)
    # KB 99 has no tagged entity anywhere → genuinely not encoded.
    dep = EncoderResult(kb_id=99, status="deferred", entities=[], notes="missing")
    block = await _format_deps_block([dep], storage, DB)
    assert "KB 99" in block
    assert "NOT encoded" in block


async def test_deps_block_includes_deferred_child_notes(tmp_path):
    """Codex follow-up (PR #4): the defensive-defer prompt tells the
    encoder to "quote the reason verbatim from the deps block" so a
    later per-task agent can distinguish "child deferred because
    ambiguous" from "child errored during encode". The block must
    surface the child's `notes` (when present) so the dependent KB has
    a concrete reason to quote — `status="deferred"` on its own is the
    same word for every clean defer and carries no signal."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_deps_block,
    )

    storage = await _storage_with_tagged_column(tmp_path, kb_id=4)
    dep = EncoderResult(
        kb_id=99, status="deferred", entities=[],
        notes="Sample-value mapping ambiguous: tier_label has no numeric "
              "encoding in any column meaning.",
    )
    block = await _format_deps_block([dep], storage, DB)
    assert "KB 99" in block
    assert "NOT encoded" in block
    # The child's notes — the actual ambiguity statement — must appear so
    # the dependent KB can quote it.
    assert "Sample-value mapping ambiguous" in block
    assert "tier_label" in block


async def test_deps_block_truncates_very_long_child_notes(tmp_path):
    """A long-winded child note must be truncated so it can't blow up the
    parent's prompt budget."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_deps_block,
    )

    storage = await _storage_with_tagged_column(tmp_path, kb_id=4)
    long_note = "x" * 1000
    dep = EncoderResult(
        kb_id=99, status="deferred", entities=[], notes=long_note,
    )
    block = await _format_deps_block([dep], storage, DB)
    assert "KB 99" in block
    assert "NOT encoded" in block
    # Truncated — the rendered block has nowhere near 1000 x's in a row.
    assert "x" * 250 not in block


async def test_deps_block_empty_is_none(tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_deps_block,
    )

    storage = await _storage_with_tagged_column(tmp_path)
    assert await _format_deps_block([], storage, DB) == "(none)"


async def test_run_setup_encoder_keeps_submission_when_run_raises_after_submit(tmp_path):
    """Codex finding: if the model submits a valid result and THEN the run raises
    (e.g. UsageLimitExceeded chasing a final text), the valid submission must NOT
    be discarded as an error."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    submission = EncoderResult(kb_id=9, status="deferred", notes="submitted then capped")
    final = await setup_encoder._run_setup_encoder(
        agent=_SubmittingAgent(submission, raise_after=True), kb_id=9,
        kb_body="KB 9 — body", deps_results=[], storage=object(),
        db="households", sessions_dir=tmp_path,
    )
    assert final.status == "deferred"   # NOT "error"
    assert "submitted then capped" in final.notes


# ---------------------------------------------------------------------------
# Reverse-dependency block (DEV-1466): a KB must see the KBs that REFERENCE it
# (its parents) + their type/knowledge/definition, so a value_illustration can
# defer an embedded scoring scheme to a calculation_knowledge parent that owns
# that score (KB-6 "Dwelling Type" → defer the score to KB-44 "Dwelling Type
# Score"), while still emitting component scores whose parent merely AVERAGES
# them (KB-3/4/5 → KB-13). Pure helper, mirrors `_format_deps_block`.
# ---------------------------------------------------------------------------


def test_reverse_deps_block_empty_is_none():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_reverse_deps_block,
    )

    assert _format_reverse_deps_block([]) == "(none)"
    assert _format_reverse_deps_block(None) == "(none)"


def test_reverse_deps_block_lists_parents_sorted_with_type_and_knowledge():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_reverse_deps_block,
    )

    parents = [
        {"id": 44, "type": "calculation_knowledge",
         "knowledge": "Dwelling Type Score",
         "definition": "Brickwork house 4, Apartment 3, else 1."},
        {"id": 20, "type": "calculation_knowledge",
         "knowledge": "Living Condition Score",
         "definition": "50/50 average of dwelling + infra score."},
    ]
    block = _format_reverse_deps_block(parents)
    lines = [ln for ln in block.splitlines() if ln.strip()]
    assert len(lines) == 2
    # sorted by kb id → KB 20 line precedes KB 44 line
    assert "KB 20" in lines[0]
    assert "KB 44" in lines[1]
    assert "calculation_knowledge" in block
    # knowledge AND definition both surface (the GUARD needs the definition to
    # judge whether the parent IS this score vs merely averages it)
    assert "Dwelling Type Score" in block
    assert "Living Condition Score" in block
    assert "Brickwork house 4" in block
    assert "50/50 average" in block


def test_reverse_deps_block_tolerates_missing_fields_and_dedups():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_reverse_deps_block,
    )

    parents = [
        {"id": 7},                                       # no type/knowledge/def
        {"id": 7, "type": "calc", "knowledge": "dup"},   # duplicate id 7
        {"id": 9, "type": "domain_knowledge", "knowledge": "X",
         "definition": "D" * 1000},                      # very long definition
    ]
    block = _format_reverse_deps_block(parents)  # must not raise
    lines = [ln for ln in block.splitlines() if ln.strip()]
    # duplicate id 7 collapses to ONE line; 9 is its own line
    assert sum("KB 7" in ln for ln in lines) == 1
    assert any("KB 9" in ln for ln in lines)
    # the 1000-char definition is present-but-TRUNCATED: a prefix of it appears,
    # but not the whole 1000-char string, and the block stays bounded.
    assert "DDDD" in block               # a prefix of the long definition shows
    assert ("D" * 1000) not in block     # but not the full untrimmed value
    assert len(block) < 1000


async def test_run_setup_encoder_threads_reverse_deps_into_prompt(tmp_path):
    """DEV-1466: a `reverse_deps` passed to `_run_setup_encoder` must reach the
    formatted SETUP_ENCODER_PROMPT (so the encoder can see its parents)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder

    captured: dict = {}

    class _CapRun:
        result = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        def all_messages(self):
            return []

        def usage(self):
            return None

    class _CapCM:
        def __init__(self, kw):
            self._kw = kw

        async def __aenter__(self):
            captured["instructions"] = self._kw.get("instructions")
            return _CapRun()

        async def __aexit__(self, *exc):
            return False

    class _CapAgent:
        def iter(self, **kw):
            return _CapCM(kw)

    # unique sentinel so the assertion can't pass on kb_body / deps content
    reverse = [{"id": 44, "type": "calculation_knowledge",
                "knowledge": "UNIQUE_REVERSE_PARENT_SENTINEL",
                "definition": "Brickwork 4, Apartment 3, else 1."}]
    await setup_encoder._run_setup_encoder(
        agent=_CapAgent(), kb_id=6, kb_body="KB 6 — Dwelling Type",
        deps_results=[], storage=object(), db="households",
        reverse_deps=reverse, sessions_dir=tmp_path,
    )
    instr = captured["instructions"]
    assert instr is not None
    assert "UNIQUE_REVERSE_PARENT_SENTINEL" in instr
    assert "KB 44" in instr


async def test_run_one_closure_forwards_reverse_deps(monkeypatch):
    """DEV-1466: the real `run_one` closure from `make_setup_build_encoder` must
    forward `reverse_deps` into `_run_setup_encoder` (the seam between
    `_encode_all` and the prompt)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    captured: dict = {}

    async def _capture_run(*, agent, kb_id, kb_body, deps_results, storage, db,
                           self_model_id="unknown", sessions_dir=None,
                           index_rows=None, reverse_deps=None):
        captured["reverse_deps"] = reverse_deps
        return EncoderResult(kb_id=kb_id, status="encoded", entities=[], notes="")

    monkeypatch.setattr(setup_encoder, "_build_setup_encoder", lambda **kw: object())
    monkeypatch.setattr(setup_encoder, "_run_setup_encoder", _capture_run)

    build_encoder = setup_encoder.make_setup_build_encoder(
        model=object(), model_settings=None, self_model_id="m",
        build_shared_slayer_server=lambda _dir: None,
    )
    run_one = build_encoder(storage=object(), build_dir="/x", db="households")
    reverse = [{"id": 44, "type": "calculation_knowledge",
                "knowledge": "Dwelling Type Score", "definition": "x"}]
    await run_one(6, {"id": 6, "knowledge": "Dwelling Type"}, [],
                  reverse_deps=reverse)

    assert captured["reverse_deps"] == reverse


# ---------------------------------------------------------------------------
# DEV-1478: PEER-KB DEDUP support — a new helper renders every meta.kb_id-
# tagged entity already present in storage as a block the prompt can show
# the agent. Mechanical contract test (the function is a pure
# storage-driven renderer; we're not testing prompt content here).
# ---------------------------------------------------------------------------


async def test_existing_kb_tagged_entities_block_empty_when_nothing_tagged(tmp_path):
    """No models / no kb-tagged entities → block renders `(none)` so the prompt
    has a deterministic substitution value even on a fresh build."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    assert await _format_existing_kb_tagged_entities_block(storage, DB) == "(none)"


async def test_existing_kb_tagged_entities_block_lists_tagged_columns(tmp_path):
    """Storage has a Column with meta.kb_id=9 → block names the column, its
    kind, and the kb_id."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = await _storage_with_tagged_column(
        tmp_path, model="service_types", col="socsupport", kb_id=9,
    )
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    assert "service_types" in block
    assert "socsupport" in block
    assert "kb_id=9" in block


async def test_existing_kb_tagged_entities_block_filters_to_lower_kb_id(tmp_path):
    """Codex follow-up (PR #4, second-pass review): setup encoders run
    concurrently via asyncio fan-out — completion order doesn't match
    topo-id order. The peer-KB block must filter to entities whose
    kb_id is strictly less than the current KB's id so the rendered
    "lower-id is canonical" claim in the prompt holds regardless of
    concurrent completion order."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    # Three tagged entities at kb_id ∈ {3, 10, 15}. When the current
    # encoder is at kb_id=10, ONLY kb_id=3 should appear (strictly less
    # than 10). kb_id=10 itself (same id, possibly a sibling write from
    # a concurrent encoder) and kb_id=15 (a higher id that finished
    # first under concurrency) must NOT appear.
    await storage.save_model(SlayerModel(
        name="m_lower", data_source=DB, sql_table="m_lower",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c_lower", sql="1", meta={"kb_id": 3})],
    ))
    await storage.save_model(SlayerModel(
        name="m_same", data_source=DB, sql_table="m_same",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c_same", sql="1", meta={"kb_id": 10})],
    ))
    await storage.save_model(SlayerModel(
        name="m_higher", data_source=DB, sql_table="m_higher",
        columns=[Column(name="id", primary_key=True),
                 Column(name="c_higher", sql="1", meta={"kb_id": 15})],
    ))

    # With current_kb_id=10 — only kb_id=3 qualifies.
    block = await _format_existing_kb_tagged_entities_block(
        storage, DB, current_kb_id=10,
    )
    assert "c_lower" in block
    assert "kb_id=3" in block
    assert "c_same" not in block, (
        f"same-kb_id entries (concurrent sibling) must NOT leak in. "
        f"block:\n{block}"
    )
    assert "c_higher" not in block, (
        f"higher-kb_id entries (concurrent earlier-finisher) must NOT "
        f"leak in. block:\n{block}"
    )

    # With current_kb_id=None (no filter) — all three appear.
    block_all = await _format_existing_kb_tagged_entities_block(
        storage, DB, current_kb_id=None,
    )
    assert "c_lower" in block_all
    assert "c_same" in block_all
    assert "c_higher" in block_all


async def test_existing_kb_tagged_entities_block_sorted_by_kb_id(tmp_path):
    """Multiple tagged entities → output is sorted by kb_id ascending so the
    canonical (lower-id) entity appears first."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="a", data_source=DB, sql_table="a",
        columns=[Column(name="id", primary_key=True),
                 Column(name="late", sql="1", meta={"kb_id": 37})],
    ))
    await storage.save_model(SlayerModel(
        name="b", data_source=DB, sql_table="b",
        columns=[Column(name="id", primary_key=True),
                 Column(name="early", sql="1", meta={"kb_id": 9})],
    ))
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    pos_9 = block.find("kb_id=9")
    pos_37 = block.find("kb_id=37")
    assert pos_9 != -1 and pos_37 != -1
    assert pos_9 < pos_37, (
        f"lower kb_id must appear first in the block; got block:\n{block}"
    )


async def test_existing_kb_tagged_entities_block_same_kb_id_deterministic(tmp_path):
    """CodeRabbit PR #4 nitpick: two entities sharing the same kb_id
    (e.g. R-FILTER's Column + ModelMeasure both tagged with the same id)
    must sort to a deterministic order across rebuilds — tie-broken on
    the rendered line so prompt text stays stable."""
    from slayer.core.models import ModelMeasure

    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    # Two entities sharing kb_id=5 on the same model (Column + Measure
    # is the R-FILTER recipe's natural output).
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="active_flag", sql="status='Y'",
                   meta={"kb_id": 5}),
        ],
        measures=[ModelMeasure(
            name="active_share", formula="active_flag:avg",
            meta={"kb_id": 5},
        )],
    ))
    block1 = await _format_existing_kb_tagged_entities_block(storage, DB)
    block2 = await _format_existing_kb_tagged_entities_block(storage, DB)
    # Rebuild stability: the block is byte-identical across two calls
    # against the same storage. This pins the property the nitpick was
    # about (storage-traversal order would otherwise leak into the
    # rendered string and produce drift across rebuilds).
    assert block1 == block2, (
        "rendering must be deterministic for entities sharing kb_id "
        "(CodeRabbit PR #4 nitpick on factories.py:799-800)"
    )
    # And we should sort alphabetically on the rendered line: 'measure'
    # entries come after 'column' alphabetically within the kb_id tier,
    # since the leading text is `  - {db}.m.{name} ({kind}) -> kb_id=5`
    # and "active_flag (column)" < "active_share (measure)" lexicographically.
    pos_col = block1.find("active_flag")
    pos_meas = block1.find("active_share")
    assert pos_col != -1 and pos_meas != -1
    assert pos_col < pos_meas


async def test_existing_kb_tagged_entities_block_lists_tagged_measures(tmp_path):
    """ModelMeasure with meta.kb_id must also appear (R-MEASURE / R-FILTER
    recipes write measures, not just columns)."""
    from slayer.core.models import ModelMeasure

    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True),
                 Column(name="status", sql="status")],
        measures=[ModelMeasure(
            name="active_count", formula="status:count",
            meta={"kb_id": 11},
        )],
    ))
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    assert "active_count" in block
    assert "kb_id=11" in block


async def test_existing_kb_tagged_entities_block_lists_tagged_aggregations(tmp_path):
    """Aggregation with meta.kb_id must appear (R-AGG recipe). Aggregations
    live on the model under `aggregations: List[Aggregation]`."""
    from slayer.core.models import Aggregation

    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[Column(name="id", primary_key=True)],
        aggregations=[Aggregation(
            name="weighted_score",
            formula="SUM({value} * w) / SUM(w)",
            meta={"kb_id": 13},
        )],
    ))
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    assert "weighted_score" in block
    assert "kb_id=13" in block


async def test_existing_kb_tagged_entities_block_ignores_untagged_columns(tmp_path):
    """Columns without meta.kb_id must not appear in the block (only
    canonical-kb-tagged entities qualify for PEER-KB DEDUP)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
        _format_existing_kb_tagged_entities_block,
    )

    storage = YAMLStorage(base_dir=str(tmp_path))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name="m", data_source=DB, sql_table="m",
        columns=[
            Column(name="id", primary_key=True),
            Column(name="untagged", sql="1"),   # no meta
            Column(name="tagged", sql="1", meta={"kb_id": 5}),
        ],
    ))
    block = await _format_existing_kb_tagged_entities_block(storage, DB)
    assert "tagged" in block and "kb_id=5" in block
    # The untagged column must not appear (substring `untagged` would match
    # `untagged` literally, so we check the unique column name).
    assert "untagged" not in block


async def test_existing_kb_tagged_entities_block_threaded_into_setup_prompt(tmp_path):
    """End-to-end seam (mirrors `test_run_setup_encoder_threads_reverse_deps_into_prompt`):
    the real `_run_setup_encoder` formats the prompt with a
    `existing_kb_tagged_entities_block` derived from the supplied storage."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import setup_encoder

    captured: dict = {}

    class _CapRun:
        def __init__(self):
            self._stream = self._stream_impl()

        def _stream_impl(self):
            if False:
                yield None
            return

        def __aiter__(self): return self

        async def __anext__(self): raise StopAsyncIteration

    class _CapCM:
        def __init__(self, run, deps, instructions):
            self._run = run
            self._instructions = instructions
            self._deps = deps

        async def __aenter__(self):
            captured["instructions"] = self._instructions
            return self._run

        async def __aexit__(self, *exc): return False

    class _CapAgent:
        def iter(self, *, user_prompt, instructions, deps, usage_limits):
            return _CapCM(_CapRun(), deps, instructions)

    # Storage has a canonical kb=9 entity present (the lower-id sibling that
    # PEER-KB DEDUP should warn about when KB 37 fires).
    storage = await _storage_with_tagged_column(
        tmp_path, model="service_types", col="socsupport", kb_id=9,
    )

    await setup_encoder._run_setup_encoder(
        agent=_CapAgent(), kb_id=37, kb_body="KB 37 — Social Assistance",
        deps_results=[], storage=storage, db=DB,
        sessions_dir=tmp_path,
    )
    instr = captured["instructions"]
    assert instr is not None
    assert "socsupport" in instr
    assert "kb_id=9" in instr
