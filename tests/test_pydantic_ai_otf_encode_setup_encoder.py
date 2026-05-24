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
                        index_rows=None):
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
