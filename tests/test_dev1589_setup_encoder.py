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


class _Result:
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
        yield _Result(cur.get("usage"))


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
        return self._client

    async def __aexit__(self, *exc):
        if self._inflight is not None:
            self._inflight["cur"] -= 1
        return False


def _patch_session(monkeypatch, clients, *, inflight=None, raise_on_enter=False):
    """Patch `setup_encoder.hermetic_claude_sdk_session` to hand out `clients`
    (a list, one per run_one invocation) wrapped in fake sessions."""
    q = list(clients)

    def _factory(model, **kwargs):
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
    s = YAMLStorage(base_dir=str(build_dir))
    m = await s.get_model(MODEL, data_source=DB)
    cols = list(m.columns or [])
    meta = {"kb_id": kb_id} if tagged else {}
    cols.append(Column(name=name, sql=sql, description="agent wrote this", meta=meta))
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
        "help", "search", "models_summary", "inspect_model",
        "query", "query_nested",
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
    result, _ = await _run_one(tmp_path, [client], monkeypatch)
    assert result.status == "encoded"
    assert len(client.queries) == 3  # 1 + 2 retries before success


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
