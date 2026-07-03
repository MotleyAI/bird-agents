"""DEV-1589: the claude_sdk-native build-time OTF reference encoder
(`agents.claude_sdk_otf_encode.setup_encoder`).

Per-KB, serial (a per-build asyncio.Lock), fresh hermetic ClaudeSDKClient + its
own SLayer stdio subprocess. The agent wires its own output; a <=3 corrective
re-prompt loop on the warm client enforces HC-present + HC-depuse, then
downgrades-to-deferred + purges partials. HC-desc + HC-mem are auto-completed
by the harness after the session. submit_encoding is the only native.

These tests stub the SDK entirely (monkeypatched `hermetic_claude_sdk_session`
→ a fake client whose per-cycle "behavior" writes entities to the build_dir
storage + calls the REAL `submit_encoding` native), so they exercise the run_one
control flow + verification with no LLM.
"""

from __future__ import annotations

import asyncio

import pytest
from slayer.core.models import Column, DatasourceConfig, ModelMeasure, SlayerModel
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.agents.claude_sdk.agent import _ctx_var
from bird_interact_agents.agents.claude_sdk_otf_encode import setup_encoder as se
from bird_interact_agents.slayer_otf.encoder_types import (
    EncodedEntity,
    EncoderResult,
)

DB = "tinydb"
MODEL = "orders"


# ---------------------------------------------------------------------------
# Fakes (mirrors tests/test_dev1581_discovery_channel.py)
# ---------------------------------------------------------------------------


class ResultMessage:
    """Name MUST be 'ResultMessage' — run_one breaks the receive loop on it AND
    SdkUsageTracker.observe() commits only this type name."""

    def __init__(self, usage=None):
        self.usage = usage or {
            "input_tokens": 10, "output_tokens": 2,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }


class _FakeClient:
    """One fake warm ClaudeSDKClient. `cycles[i]` drives the i-th query()/
    receive_response() pair: an async `behavior` (simulating the agent's storage
    writes + submit_encoding), a `usage` dict, and a `block` flag (for timeout)."""

    def __init__(self, cycles, *, default=None):
        self.cycles = cycles
        self.default = default or {}
        self.i = 0
        self.queries: list[str] = []
        self._cur = {}
        # Mimic the wrapper's recording proxy: the real _TranscriptClient
        # populates `.transcript`; here we append a serialized message per
        # streamed ResultMessage so the encoder's persistence wiring is testable.
        self.transcript: list[dict] = []

    async def query(self, msg):
        self.queries.append(msg)
        self._cur = self.cycles[self.i] if self.i < len(self.cycles) else self.default
        beh = self._cur.get("behavior")
        if beh is not None:
            await beh()

    def receive_response(self):
        cur = self._cur
        self.i += 1
        return self._gen(cur)

    async def _gen(self, cur):
        if cur.get("block"):
            await asyncio.sleep(100)  # never reaches a ResultMessage → times out
        m = ResultMessage(cur.get("usage"))
        self.transcript.append({"type": "ResultMessage", "data": {}})
        yield m


class _FakeSession:
    def __init__(self, client, inflight=None, raise_on_enter=False):
        self._client = client
        self._inflight = inflight
        self._raise = raise_on_enter

    async def __aenter__(self):
        if self._raise:
            raise RuntimeError("hermetic session enter failed")
        if self._inflight is not None:
            self._inflight["cur"] += 1
            self._inflight["max"] = max(self._inflight["max"], self._inflight["cur"])
            # Force a scheduling window: if run_one did NOT hold the per-build
            # lock across the whole session, a second concurrent run_one would
            # enter here during this await and drive max to 2 (Codex test #2).
            await asyncio.sleep(0.02)
        return self._client

    async def __aexit__(self, *exc):
        if self._inflight is not None:
            await asyncio.sleep(0.02)
            self._inflight["cur"] -= 1
        return False


def _patch_session(monkeypatch, clients, *, inflight=None, raise_on_enter=False,
                   captured=None):
    """Patch `setup_encoder.hermetic_claude_sdk_session` to hand out `clients`
    (a list, one per run_one invocation) wrapped in fake sessions. When
    `captured` is given, each call's kwargs (incl. mcp_servers) are appended."""
    q = list(clients)

    def _factory(model, **kwargs):
        if captured is not None:
            captured.append({"model": model, **kwargs})
        client = q.pop(0) if q else _FakeClient([])
        return _FakeSession(client, inflight=inflight, raise_on_enter=raise_on_enter)

    monkeypatch.setattr(se, "hermetic_claude_sdk_session", _factory)


# ---------------------------------------------------------------------------
# Storage seed + agent-behavior helpers
# ---------------------------------------------------------------------------


async def _seed(build_dir, *, kb_id=7, with_memory=True):
    storage = YAMLStorage(base_dir=str(build_dir))
    await storage.save_datasource(DatasourceConfig(
        name=DB, type="sqlite", connection_string="sqlite:///x.sqlite",
    ))
    await storage.save_model(SlayerModel(
        name=MODEL, data_source=DB, sql_table=MODEL,
        columns=[Column(name="id", primary_key=True), Column(name="amount", sql="amount")],
        measures=[ModelMeasure(name="premium_revenue", formula="amount:sum",
                               meta={"kb_id": 4})],
    ))
    if with_memory:
        await storage.save_memory(
            learning=f"KB {kb_id} — body", entities=[DB], query=None,
            id=f"{DB}_kb_{kb_id}", description="",
        )
    return storage


async def _write_col(build_dir, name, kb_id, *, sql="1", tagged=True):
    """Upsert a column (mirrors SLayer edit_model — unique names, replace in
    place), so re-writing the same entity across retries doesn't duplicate it."""
    s = YAMLStorage(base_dir=str(build_dir))
    m = await s.get_model(MODEL, data_source=DB)
    meta = {"kb_id": kb_id} if tagged else {}
    new = Column(name=name, sql=sql, description="agent wrote this", meta=meta)
    cols = [c for c in (m.columns or []) if c.name != name]
    cols.append(new)
    await s.save_model(m.model_copy(update={"columns": cols}))


def _ent(name, kind="column"):
    return EncodedEntity(kind=kind, host_model=MODEL, name=name,
                         entity_ref=f"{DB}.{MODEL}.{name}")


async def _submit(result: EncoderResult):
    await se.submit_encoding.handler({"result_json": result.model_dump_json()})


def _encoded(kb_id, names):
    return EncoderResult(kb_id=kb_id, status="encoded",
                         entities=[_ent(n) for n in names], notes="ok")


def _row(kb_id=7):
    return {"id": kb_id, "knowledge": "premium revenue per order",
            "children_knowledge": -1}


async def _run_one(build_dir, clients, monkeypatch, *, kb_id=7, row=None,
                   deps_results=None, inflight=None):
    _patch_session(monkeypatch, clients, inflight=inflight)
    build_encoder = se.make_claude_sdk_build_encoder(
        model="zai/glm-5.2", self_model_id="glm",
    )
    storage = YAMLStorage(base_dir=str(build_dir))
    run_one = build_encoder(storage, build_dir, DB, build_dir / "_sessions")
    await run_one.aopen()
    try:
        return await run_one(kb_id, row or _row(kb_id), deps_results or []), run_one
    finally:
        await run_one.aclose()


# ---------------------------------------------------------------------------
# submit_encoding native
# ---------------------------------------------------------------------------


async def test_submit_encoding_captures_valid_into_ctx():
    _ctx_var.set({"_encoder_submission": None})
    r = EncoderResult(kb_id=3, status="deferred", notes="x")
    await se.submit_encoding.handler({"result_json": r.model_dump_json()})
    assert _ctx_var.get()["_encoder_submission"] == r


async def test_submit_encoding_rejects_invalid_json():
    _ctx_var.set({"_encoder_submission": None})
    out = await se.submit_encoding.handler({"result_json": "{not json"})
    assert _ctx_var.get()["_encoder_submission"] is None
    # returns an error string (tool envelope), does not raise
    text = "".join(b["text"] for b in out["content"])
    assert "submit_encoding" in text.lower() or "invalid" in text.lower()


# ---------------------------------------------------------------------------
# build_encoder surface
# ---------------------------------------------------------------------------


def test_build_encoder_exposes_aopen_aclose_index_usage(tmp_path):
    from bird_interact_agents.usage import TokenUsage

    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    assert callable(getattr(run_one, "aopen", None))
    assert callable(getattr(run_one, "aclose", None))
    assert isinstance(run_one.index_rows, list)
    assert isinstance(run_one.usage, TokenUsage)


async def test_aopen_aclose_are_noops(tmp_path):
    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    await run_one.aopen()
    await run_one.aclose()  # must not raise


def test_exact_tool_surface():
    """Mechanical contract (NOT prompt content)."""
    slayer = set(se.SLAYER_MCP_TOOLS)
    assert slayer == {
        "create_model", "edit_model", "save_memory", "validate_models",
        "help", "search", "models_summary", "inspect_model", "inspect",
        "recommend_root_model", "query", "query_nested",
    }
    assert se.NATIVE_TOOL_NAME == "mcp__bird-interact-tools__submit_encoding"
    # save_memory is REQUIRED here (the agent wires its own memory) and must
    # NOT be in the disallowed set.
    assert "mcp__slayer__save_memory" not in set(se.DISALLOWED_TOOL_NAMES)
    assert "mcp__slayer__forget_memory" in set(se.DISALLOWED_TOOL_NAMES)


# ---------------------------------------------------------------------------
# run_one happy path + auto-wire
# ---------------------------------------------------------------------------


async def test_run_one_happy_path_encodes_and_autowires(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)

    async def behavior():
        await _write_col(tmp_path, "c", 7, sql="amount * 1.0")
        await _submit(_encoded(7, ["c"]))

    client = _FakeClient([{"behavior": behavior}])
    result, run_one = await _run_one(tmp_path, [client], monkeypatch)

    assert result.status == "encoded"
    assert {e.name for e in result.entities} == {"c"}
    assert client.queries == client.queries[:1]  # exactly one cycle (no retry)
    assert len(client.queries) == 1

    s = YAMLStorage(base_dir=str(tmp_path))
    model = await s.get_model(MODEL, data_source=DB)
    col = next(c for c in model.columns if c.name == "c")
    # HC-desc: verbatim KB row injected into the description
    assert "[kb=7]" in (col.description or "")
    assert "premium revenue per order" in (col.description or "")
    # HC-mem: backref appended
    mem = await s.get_memory_row(f"{DB}_kb_7")
    assert f"{DB}.{MODEL}.c" in mem.entities


# ---------------------------------------------------------------------------
# retry loop
# ---------------------------------------------------------------------------


async def test_run_one_retries_then_succeeds(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)

    async def fail_absent():
        # claims encoded but never writes the entity → HC-present fails
        await _submit(_encoded(7, ["c"]))

    async def succeed():
        await _write_col(tmp_path, "c", 7)
        await _submit(_encoded(7, ["c"]))

    client = _FakeClient([
        {"behavior": fail_absent},
        {"behavior": fail_absent},
        {"behavior": succeed},
    ])
    result, run_one = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "encoded"
    assert len(client.queries) == 3  # 1 + 2 retries before success
    # The corrective re-prompts (queries[1:]) must name the concrete failure so
    # the agent can act on it (Codex test #3) — the failing entity ref appears.
    assert any("c" in q and "tinydb" in q for q in client.queries[1:]), (
        f"corrective prompt must name the failing entity; got {client.queries[1:]}"
    )
    # One triage row appended per run_one (Codex test #7).
    assert len(run_one.index_rows) == 1


async def test_run_one_always_fails_present_downgrades(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)

    async def fail_absent():
        await _submit(_encoded(7, ["ghost"]))   # never written

    cycles = [{"behavior": fail_absent} for _ in range(se.MAX_ENCODE_ATTEMPTS)]
    client = _FakeClient(cycles)
    result, _ = await _run_one(tmp_path, [client], monkeypatch)

    assert result.status == "deferred"
    assert result.entities == []
    assert len(client.queries) == se.MAX_ENCODE_ATTEMPTS


async def test_run_one_downgrade_purges_tagged_partials(tmp_path, monkeypatch):
    """A tagged entity that fails HC-depuse on every attempt is purged from
    storage on downgrade (no orphan in the committed reference)."""
    await _seed(tmp_path, kb_id=7)
    dep = EncoderResult(
        kb_id=4, status="encoded",
        entities=[EncodedEntity(kind="measure", host_model=MODEL,
                                name="premium_revenue",
                                entity_ref=f"{DB}.{MODEL}.premium_revenue")],
    )
    # KB 7 declares KB 4 as a child, but inlines SUM(amount) instead of
    # referencing premium_revenue → HC-depuse fails every attempt.
    row = {"id": 7, "knowledge": "margin", "children_knowledge": 4}

    async def inline_dep():
        await _write_col(tmp_path, "margin", 7, sql="SUM(amount) / 2")
        await _submit(_encoded(7, ["margin"]))

    cycles = [{"behavior": inline_dep} for _ in range(se.MAX_ENCODE_ATTEMPTS)]
    client = _FakeClient(cycles)
    result, _ = await _run_one(
        tmp_path, [client], monkeypatch, row=row, deps_results=[dep],
    )
    assert result.status == "deferred"
    assert result.entities == []
    s = YAMLStorage(base_dir=str(tmp_path))
    model = await s.get_model(MODEL, data_source=DB)
    assert "margin" not in {c.name for c in (model.columns or [])}  # purged
    mem = await s.get_memory_row(f"{DB}_kb_7")
    assert f"{DB}.{MODEL}.margin" not in mem.entities


# ---------------------------------------------------------------------------
# error / no-submit / deferred-with-entities
# ---------------------------------------------------------------------------


async def test_no_resubmit_on_corrective_attempt_keeps_prompting(tmp_path, monkeypatch):
    """Codex review: a corrective attempt that fixes storage but does NOT call
    submit_encoding must not be accepted by re-verifying the PRIOR (stale)
    submission — the loop keeps prompting for an explicit re-submit."""
    await _seed(tmp_path, kb_id=7)

    async def fail_absent():
        await _submit(_encoded(7, ["c"]))     # claims c, never writes it → fail

    async def fix_no_submit():
        await _write_col(tmp_path, "c", 7)    # writes c but does NOT re-submit

    async def resubmit():
        await _submit(_encoded(7, ["c"]))     # re-confirm now that c exists

    client = _FakeClient([
        {"behavior": fail_absent},
        {"behavior": fix_no_submit},
        {"behavior": resubmit},
    ])
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "encoded"
    # Without the guard, the no-submit attempt would have short-circuited to
    # success at cycle 2 (queries==2); the guard forces the 3rd confirm.
    assert len(client.queries) == 3


async def test_run_one_heuristic_dep_not_required(tmp_path, monkeypatch):
    """HC-depuse applies ONLY to DECLARED children_knowledge deps. An encoded
    dep that reached run_one via a heuristic formula-token edge (NOT in
    children_knowledge) must NOT force a reference — inlining it stays encoded
    (DEV-1589-spec §1; Codex test #4)."""
    await _seed(tmp_path, kb_id=7)
    dep = EncoderResult(
        kb_id=4, status="encoded",
        entities=[EncodedEntity(kind="measure", host_model=MODEL,
                                name="premium_revenue",
                                entity_ref=f"{DB}.{MODEL}.premium_revenue")],
    )
    # children_knowledge == -1 → KB 4 is NOT a declared child, even though it's
    # passed in deps_results (heuristic edge).
    row = {"id": 7, "knowledge": "margin", "children_knowledge": -1}

    async def inline_dep():
        await _write_col(tmp_path, "margin", 7, sql="SUM(amount) / 2")
        await _submit(_encoded(7, ["margin"]))

    client = _FakeClient([{"behavior": inline_dep}])
    result, _ = await _run_one(
        tmp_path, [client], monkeypatch, row=row, deps_results=[dep],
    )
    assert result.status == "encoded"  # NOT downgraded
    assert len(client.queries) == 1    # no corrective retry


async def test_hermetic_session_gets_real_mcp_servers(tmp_path, monkeypatch):
    """Codex review (critical): the `slayer` stdio server (and bird-interact-
    tools) MUST be declared to hermetic_claude_sdk_session, else its parity
    assertion rejects the loaded slayer subprocess and every KB errors."""
    await _seed(tmp_path, kb_id=7)
    captured: list[dict] = []

    async def behavior():
        await _write_col(tmp_path, "c", 7)
        await _submit(_encoded(7, ["c"]))

    _patch_session(monkeypatch, [_FakeClient([{"behavior": behavior}])],
                   captured=captured)
    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    await run_one.aopen()
    try:
        await run_one(7, _row(7), [])
    finally:
        await run_one.aclose()
    assert captured, "hermetic session must be entered"
    servers = captured[0]["mcp_servers"]
    assert "slayer" in servers and "bird-interact-tools" in servers


async def test_encoder_registers_force_compact_search_hook(tmp_path, monkeypatch):
    """DEV-1591 (merge with DEV-1629): the build-time encoder hardwires SLayer
    ``search`` to ``compact=True`` via ``_force_compact_search_hook`` — parity
    with the on-the-fly task agents, so a broad encoder ``search`` cannot drag
    full per-entity renders into context. Assert the PreToolUse hook is wired on
    the ``mcp__slayer__search`` matcher alongside the write-normalization hook.
    """
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        _SLAYER_SEARCH_TOOL,
        _force_compact_search_hook,
    )

    await _seed(tmp_path, kb_id=7)
    captured: list[dict] = []

    async def behavior():
        await _write_col(tmp_path, "c", 7)
        await _submit(_encoded(7, ["c"]))

    _patch_session(monkeypatch, [_FakeClient([{"behavior": behavior}])],
                   captured=captured)
    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    await run_one.aopen()
    try:
        await run_one(7, _row(7), [])
    finally:
        await run_one.aclose()
    assert captured, "hermetic session must be entered"
    options = captured[0]["build_options"]({})
    pre_matchers = options.hooks["PreToolUse"]
    search_matchers = [
        m for m in pre_matchers if getattr(m, "matcher", None) == _SLAYER_SEARCH_TOOL
    ]
    assert search_matchers, "encoder must register a `search` PreToolUse matcher"
    assert any(
        _force_compact_search_hook in m.hooks for m in search_matchers
    ), "the search matcher must run _force_compact_search_hook"


def _dep_kb4_measure():
    return EncoderResult(
        kb_id=4, status="encoded",
        entities=[EncodedEntity(kind="measure", host_model=MODEL,
                                name="premium_revenue",
                                entity_ref=f"{DB}.{MODEL}.premium_revenue")],
        notes="Sum of order amounts for premium-tier orders.",
    )


async def test_deps_block_uses_inspect_compact_false(tmp_path):
    """The deps block hands the model SLayer's real inspect(compact=False)
    rendering of each created dependency entity (motley-slayer >= 0.8.3):
    formula + description + label, so it references by name without
    re-discovering via search/inspect."""
    await _seed(tmp_path, kb_id=7)  # seeds `orders` w/ measure premium_revenue (amount:sum, kb_id 4)
    block = await se._format_deps_block([_dep_kb4_measure()], tmp_path, DB)
    assert "KB 4" in block
    assert "premium_revenue" in block           # the entity name
    assert "amount:sum" in block                # inspect renders the measure formula
    assert "premium-tier orders" in block       # the dep's encode notes


async def test_deps_block_empty_is_none(tmp_path):
    assert await se._format_deps_block([], tmp_path, DB) == "(none)"


async def test_submit_or_defer_nudge_fires_after_budget():
    """The PostToolUse nudge stays silent under the soft budget, then fires on
    EVERY turn after it, telling the model to submit (encode) or defer."""
    hook = se._make_submit_or_defer_nudge(soft_budget=5)
    for _ in range(4):                       # calls 1..4 → silent
        assert await hook({}, None, None) == {}
    out = await hook({}, None, None)         # call 5 → nudge
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "submit_encoding" in ctx and "deferred" in ctx
    # sustained: still nudges on the next turn
    out2 = await hook({}, None, None)
    assert "submit_encoding" in out2["hookSpecificOutput"]["additionalContext"]


async def test_transcript_persisted_to_session_file(tmp_path, monkeypatch):
    """The per-KB transcript captured by the wrapper proxy (here simulated on
    the fake client) is written into the session `.json` file."""
    import json
    await _seed(tmp_path, kb_id=7)

    async def behavior():
        await _write_col(tmp_path, "c", 7)
        await _submit(_encoded(7, ["c"]))

    client = _FakeClient([{"behavior": behavior}])
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "encoded"
    j = json.loads((tmp_path / "_sessions" / "setup__kb_007.json").read_text())
    assert isinstance(j, list) and len(j) >= 1
    assert any(e.get("type") == "ResultMessage" for e in j)


async def test_encoded_with_no_entities_downgrades(tmp_path, monkeypatch):
    """Codex review: an 'encoded' submission listing NO entities is downgraded
    to deferred (legacy verifier parity)."""
    await _seed(tmp_path, kb_id=7)

    async def behavior():
        await _submit(EncoderResult(kb_id=7, status="encoded", entities=[], notes="x"))

    cycles = [{"behavior": behavior} for _ in range(se.MAX_ENCODE_ATTEMPTS)]
    result, _ = await _run_one(tmp_path, [_FakeClient(cycles)], monkeypatch)
    assert result.status == "deferred"


async def test_non_canonical_entity_ref_downgrades(tmp_path, monkeypatch):
    """Codex review: a present+tagged entity reported with a wrong entity_ref is
    a hard failure (else the bad ref leaks into memory backrefs)."""
    await _seed(tmp_path, kb_id=7)

    async def behavior():
        await _write_col(tmp_path, "c", 7)
        bad = EncodedEntity(kind="column", host_model=MODEL, name="c",
                            entity_ref=f"{DB}.WRONGMODEL.c")
        await _submit(EncoderResult(kb_id=7, status="encoded", entities=[bad]))

    cycles = [{"behavior": behavior} for _ in range(se.MAX_ENCODE_ATTEMPTS)]
    result, _ = await _run_one(tmp_path, [_FakeClient(cycles)], monkeypatch)
    assert result.status == "deferred"


async def test_index_row_appended_on_error(tmp_path, monkeypatch):
    """A triage row is appended even on the error path (Codex test #7)."""
    await _seed(tmp_path, kb_id=7)
    _patch_session(monkeypatch, [_FakeClient([])], raise_on_enter=True)
    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    await run_one.aopen()
    try:
        await run_one(7, _row(7), [])
    finally:
        await run_one.aclose()
    assert len(run_one.index_rows) == 1


async def test_run_one_no_submit_is_error_and_purges(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)

    async def write_but_no_submit():
        await _write_col(tmp_path, "c", 7)   # tagged write, but never submits

    client = _FakeClient([{"behavior": write_but_no_submit}])
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "error"
    assert result.entities == []
    s = YAMLStorage(base_dir=str(tmp_path))
    model = await s.get_model(MODEL, data_source=DB)
    assert "c" not in {c.name for c in (model.columns or [])}  # purged


async def test_run_one_deferred_with_entities_clears_and_purges(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)

    async def defer_with_writes():
        await _write_col(tmp_path, "c", 7)
        # defensively lists an entity on a deferred result
        await _submit(EncoderResult(kb_id=7, status="deferred",
                                    entities=[_ent("c")], notes="ambiguous"))

    client = _FakeClient([{"behavior": defer_with_writes}])
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "deferred"
    assert result.entities == []
    assert "ambiguous" in result.notes
    s = YAMLStorage(base_dir=str(tmp_path))
    model = await s.get_model(MODEL, data_source=DB)
    assert "c" not in {c.name for c in (model.columns or [])}


async def test_run_one_session_exception_is_error(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)
    _patch_session(monkeypatch, [_FakeClient([])], raise_on_enter=True)
    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    await run_one.aopen()
    try:
        result = await run_one(7, _row(7), [])
    finally:
        await run_one.aclose()
    assert result.status == "error"
    assert result.error


# ---------------------------------------------------------------------------
# per-attempt timeout (Codex r2 #1)
# ---------------------------------------------------------------------------


async def test_run_one_attempt_timeout_is_error(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)
    monkeypatch.setattr(se, "ENCODE_ATTEMPT_TIMEOUT_S", 0.05, raising=False)
    client = _FakeClient([{"block": True}])
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "error"


async def test_timeout_after_failed_encoded_is_error_not_deferred(tmp_path, monkeypatch):
    """Codex review: a timeout on a corrective attempt AFTER an earlier failed
    encoded submission surfaces as `error` (with the message), not a clean
    `deferred` that hides the interrupted build."""
    await _seed(tmp_path, kb_id=7)
    monkeypatch.setattr(se, "ENCODE_ATTEMPT_TIMEOUT_S", 0.05, raising=False)

    async def fail_absent():
        await _submit(_encoded(7, ["ghost"]))   # encoded but never written → fails

    client = _FakeClient([
        {"behavior": fail_absent},   # attempt 0: failed encoded submission
        {"block": True},             # attempt 1: hangs → timeout
    ])
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "error"
    assert result.error


# ---------------------------------------------------------------------------
# per-build lock serializes concurrent run_one (E1)
# ---------------------------------------------------------------------------


async def test_per_build_lock_serializes(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=1, with_memory=True)
    s = YAMLStorage(base_dir=str(tmp_path))
    await s.save_memory(learning="KB 2", entities=[DB], query=None,
                        id=f"{DB}_kb_2", description="")
    inflight = {"cur": 0, "max": 0}

    async def beh(kb_id, name):
        async def _b():
            await _write_col(tmp_path, name, kb_id)
            await _submit(_encoded(kb_id, [name]))
        return _b

    c1 = _FakeClient([{"behavior": await beh(1, "c1")}])
    c2 = _FakeClient([{"behavior": await beh(2, "c2")}])
    _patch_session(monkeypatch, [c1, c2], inflight=inflight)

    be = se.make_claude_sdk_build_encoder(model="zai/glm-5.2", self_model_id="glm")
    run_one = be(YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "s")
    await run_one.aopen()
    try:
        await asyncio.gather(
            run_one(1, {"id": 1, "knowledge": "k1", "children_knowledge": -1}, []),
            run_one(2, {"id": 2, "knowledge": "k2", "children_knowledge": -1}, []),
        )
    finally:
        await run_one.aclose()
    assert inflight["max"] == 1, "per-build lock must serialize the sessions"


# ---------------------------------------------------------------------------
# usage accumulates under scope="setup_encoder", summed across re-prompt cycles
# ---------------------------------------------------------------------------


async def test_usage_scope_setup_encoder_summed_over_cycles(tmp_path, monkeypatch):
    await _seed(tmp_path, kb_id=7)
    usage_dict = {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    async def fail_absent():
        await _submit(_encoded(7, ["c"]))

    async def succeed():
        await _write_col(tmp_path, "c", 7)
        await _submit(_encoded(7, ["c"]))

    client = _FakeClient([
        {"behavior": fail_absent, "usage": usage_dict},
        {"behavior": succeed, "usage": usage_dict},
    ])
    result, run_one = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "encoded"

    rows = [r for r in run_one.usage.breakdown if r.scope == "setup_encoder"]
    assert rows, "usage must be recorded under scope=setup_encoder"
    total_prompt = sum(r.prompt_tokens for r in rows)
    assert total_prompt == 200  # 2 cycles summed (not just the first)


# ---------------------------------------------------------------------------
# DEV-1609: the build encoder bounds the SDK session enter so a stalled cloud
# CLI/MCP handshake surfaces as a per-KB error instead of hanging forever.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_forwards_session_enter_timeout(tmp_path, monkeypatch):
    await _seed(tmp_path)
    captured: list[dict] = []

    async def _beh():
        await _write_col(tmp_path, "c", 7)
        await _submit(_encoded(7, ["c"]))

    client = _FakeClient([{"behavior": _beh, "usage": None}])
    _patch_session(monkeypatch, [client], captured=captured)

    build_encoder = se.make_claude_sdk_build_encoder(
        model="zai/glm-5.2", self_model_id="glm",
    )
    run_one = build_encoder(
        YAMLStorage(base_dir=str(tmp_path)), tmp_path, DB, tmp_path / "_s",
    )
    await run_one.aopen()
    try:
        await run_one(7, _row(7), [])
    finally:
        await run_one.aclose()

    assert captured, "hermetic session factory was never invoked"
    assert captured[0]["enter_timeout_s"] == se.SESSION_ENTER_TIMEOUT_S


# ---------------------------------------------------------------------------
# DEV-1609: _drive_with_timeout — a wedged SDK transport read must become a
# per-KB error (force-close the client), never a forever-hang.
# ---------------------------------------------------------------------------


class _DisconnectClient:
    def __init__(self):
        self.disconnects = 0

    async def disconnect(self):
        self.disconnects += 1


@pytest.mark.asyncio
async def test_drive_with_timeout_returns_on_completion():
    client = _DisconnectClient()

    async def drive():
        return None

    await se._drive_with_timeout(client, drive, 5.0)
    assert client.disconnects == 0  # healthy cycle never force-closes


@pytest.mark.asyncio
async def test_drive_with_timeout_force_closes_on_hang():
    client = _DisconnectClient()

    async def drive():
        await asyncio.sleep(10)  # >> the tiny timeout below

    with pytest.raises(TimeoutError):
        await se._drive_with_timeout(client, drive, 0.05)
    assert client.disconnects == 1  # force-closed to break the wedged read


@pytest.mark.asyncio
async def test_drive_with_timeout_propagates_inner_error():
    client = _DisconnectClient()

    async def drive():
        raise ValueError("boom in cycle")

    with pytest.raises(ValueError, match="boom in cycle"):
        await se._drive_with_timeout(client, drive, 5.0)
    assert client.disconnects == 0


@pytest.mark.asyncio
async def test_drive_with_timeout_bounds_a_wedged_teardown(monkeypatch):
    """CodeRabbit: even when BOTH disconnect() hangs AND the drive task ignores
    cancellation, _drive_with_timeout must still return (raise TimeoutError)
    within its budget — the teardown is itself capped by TEARDOWN_TIMEOUT_S."""
    monkeypatch.setattr(se, "TEARDOWN_TIMEOUT_S", 0.05, raising=False)
    released = asyncio.Event()

    class _WedgedClient:
        async def disconnect(self):
            await asyncio.sleep(10)  # disconnect itself hangs

    async def drive():
        # Ignore the first cancellation and keep waiting — a truly
        # cancellation-resistant hang (plain asyncio.sleep would exit on cancel).
        while not released.is_set():
            try:
                await released.wait()
            except asyncio.CancelledError:
                continue

    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                se._drive_with_timeout(_WedgedClient(), drive, 0.05),
                timeout=5.0,  # outer guard: the call must return well under this
            )
    finally:
        released.set()  # let the abandoned child task unwind
