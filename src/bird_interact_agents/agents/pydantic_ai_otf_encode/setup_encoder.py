"""The DEV-1454 per-DB **setup encoder** — the build-time encoder run once over
the whole KB by ``slayer_otf.reference_build``.

Unlike the task-time ``KB_ENCODER`` (factories.py) it has **no `ask_user`**:
there is no task to clarify against at build time. It encodes when confident
and DEFERS otherwise. Every claimed entity is verified to exist AND carry
``meta.kb_id`` (the sole HARD-8 deletion key); a missing/untagged entity
downgrades the KB to ``deferred``.

``make_setup_build_encoder`` returns the ``build_encoder(storage, build_dir) ->
run_one`` callable that ``ensure_db_reference`` consumes — keeping all model /
MCP construction in the agent layer (no circular import into ``slayer_otf``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from bird_interact_agents.agents._session_log import session_from_run, write_session
from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
    EncoderCaptureDeps,
    EncoderResult,
)
from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
    _format_deps_block,
    _format_existing_kb_tagged_entities_block,
    _format_reverse_deps_block,
    _make_prepare_tools,
    _meta_has_kb_id,
    _register_submit_encoding,
)
from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
    SETUP_ENCODER_PROMPT,
)
from bird_interact_agents.harness import MAX_MODEL_TURNS
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


def _build_setup_encoder(
    *,
    model: Any,
    model_settings: Any,
    shared_slayer_server: Any,
    self_model_id: str = "unknown",
) -> Agent:
    """Build the setup encoder agent. SLayer MCP read/write tools come via the
    shared toolset; the native function-toolset carries ONLY `submit_encoding` —
    no `ask_user`, no `submit_query`, no `spawn_subagent`, no `kb_to_slayer`.

    DEV-1454: NO structured ``output_type`` — the encoder reasons in text and
    delivers its ``EncoderResult`` via ``submit_encoding`` into per-run
    ``EncoderCaptureDeps`` (a structured output_type forces tool_choice='any',
    forbidding any reasoning text)."""
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=EncoderCaptureDeps, retries=2,
        prepare_tools=_make_prepare_tools(False),
    )
    if shared_slayer_server is not None:
        kwargs["toolsets"] = [shared_slayer_server]
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_submit_encoding(agent)
    return agent


async def _entity_present_and_tagged(
    ent: Any, storage: Any, db: str, kb_id: int,
) -> bool:
    """True iff ``ent`` exists in storage AND carries ``meta.kb_id == kb_id``."""
    try:
        if ent.kind == "model":
            model = await storage.get_model(ent.name)
            return (
                model is not None
                and model.data_source == db
                and _meta_has_kb_id(model.meta, kb_id)
            )
        if ent.host_model is None:
            return False
        model = await storage.get_model(ent.host_model)
        if model is None or model.data_source != db:
            return False
        bucket = {
            "column": model.columns,
            "measure": model.measures,
            "aggregation": model.aggregations,
        }.get(ent.kind)
        if bucket is None:
            return False
        for item in bucket or []:
            if item.name == ent.name:
                return _meta_has_kb_id(getattr(item, "meta", None), kb_id)
        return False
    except Exception:  # noqa: BLE001 — storage may raise on a missing model
        return False


async def _verify_setup_result(
    result: EncoderResult, storage: Any, db: str,
) -> EncoderResult:
    """Verify an ``encoded`` result: every claimed entity must exist in storage
    AND carry ``meta.kb_id``. Any failure downgrades to ``deferred`` (so a
    later per-task agent retries) — never silently ships a phantom/untagged
    entity. Non-encoded results pass through unchanged.

    Column / measure / create_model writes are SQL-validated BEFORE they persist
    (the agent.py write-gate), so for those kinds presence ⇒ valid. Aggregation
    SQL is NOT execution-validated (inline ModelExtension carries no aggregations
    field) — a documented limitation; here only presence + tagging is checked."""
    if result.status != "encoded":
        return result
    if not result.entities:
        return result.model_copy(update={
            "status": "deferred",
            "notes": (result.notes + " | " if result.notes else "")
            + "setup encoder reported 'encoded' but wrote no entities",
        })
    bad: list[str] = []
    for ent in result.entities:
        if not await _entity_present_and_tagged(ent, storage, db, result.kb_id):
            bad.append(ent.entity_ref)
    if bad:
        return result.model_copy(update={
            "status": "deferred",
            "entities": [],
            "notes": (result.notes + " | " if result.notes else "")
            + f"entities missing or untagged in storage: {', '.join(bad)}",
        })
    return result


async def _run_setup_encoder(
    *,
    agent: Agent,
    kb_id: int,
    kb_body: str,
    deps_results: list[EncoderResult],
    storage: Any,
    db: str,
    self_model_id: str = "unknown",
    sessions_dir: Any = None,
    index_rows: list | None = None,
    reverse_deps: list[dict] | None = None,
    usage: TokenUsage | None = None,
) -> EncoderResult:
    """Format the prompt, run the setup encoder agent, normalise + verify the
    result. Never raises — failures become ``status="error"``.

    Uses ``agent.iter()`` (not ``agent.run()``) so the partial trajectory is
    captured even when the run raises mid-flight (e.g. ``UsageLimitExceeded``) —
    those are exactly the sessions worth inspecting. When ``sessions_dir`` is
    given, the full per-kb session (header + step trace) is written there and the
    full triage row (``write_session``'s return — turns / tool calls / tool
    errors) is appended to ``index_rows`` for the INDEX.
    """
    deps_block = await _format_deps_block(deps_results, storage, db)
    reverse_deps_block = _format_reverse_deps_block(reverse_deps)
    existing_kb_tagged_entities_block = (
        await _format_existing_kb_tagged_entities_block(
            storage, db, current_kb_id=kb_id,
        )
    )
    prompt = SETUP_ENCODER_PROMPT.format(
        db_name=db, kb_id=kb_id, kb_body=kb_body,
        deps_block=deps_block, reverse_deps_block=reverse_deps_block,
        existing_kb_tagged_entities_block=existing_kb_tagged_entities_block,
    )
    started = time.monotonic()
    deps = EncoderCaptureDeps()
    agent_run: Any = None
    err: str | None = None
    try:
        async with agent.iter(
            user_prompt=f"Encode KB {kb_id} per the recipe book, or defer it.",
            instructions=prompt,
            deps=deps,
            usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
        ) as agent_run:
            try:
                async for _node in agent_run:
                    pass
            except Exception as exc:  # noqa: BLE001 — capture, don't abort the build
                err = f"{type(exc).__name__}: {exc}"
                logger.exception("setup encoder failed for kb_id=%s", kb_id)
            # Serialise the (possibly partial) trajectory while the run is live.
            session = session_from_run(agent_run)
    except Exception as exc:  # noqa: BLE001 — entering/exiting the run context
        err = err or f"{type(exc).__name__}: {exc}"
        logger.exception("setup encoder iter-context failed for kb_id=%s", kb_id)
        session = session_from_run(agent_run)

    # Capture the encode's token usage so the per-DB reference build's cost is
    # accounted (DEV-1478: setup-encode was previously uninstrumented). Records
    # under scope "setup_encoder" — distinct from the per-task "agent"/"user_sim"
    # scopes, so it's summable separately. Best-effort: never break the build.
    if usage is not None and agent_run is not None:
        try:
            ru = agent_run.usage()
            usage.add_call(
                scope="setup_encoder", model=self_model_id,
                prompt=getattr(ru, "input_tokens", 0) or 0,
                completion=getattr(ru, "output_tokens", 0) or 0,
                cache_read=getattr(ru, "cache_read_tokens", 0) or 0,
                cache_write=getattr(ru, "cache_write_tokens", 0) or 0,
            )
        except Exception:  # noqa: BLE001 — usage capture must not abort the build
            logger.debug("setup-encode usage capture failed for kb_id=%s", kb_id)

    # DEV-1454: the result is delivered via submit_encoding into per-run deps
    # (not structured output), so a valid submission survives a post-submit
    # failure (e.g. UsageLimitExceeded chasing the final text).
    submission = deps.encoder_submission
    if submission is not None:
        result = (
            submission if submission.kb_id == kb_id
            else submission.model_copy(update={"kb_id": kb_id})
        )
        final = await _verify_setup_result(result, storage, db)
    else:
        final = EncoderResult(
            kb_id=kb_id, status="error", entities=[], notes="",
            error=err or "agent never called submit_encoding",
        )

    if sessions_dir is not None:
        row = write_session(
            sessions_dir, f"setup__kb_{kb_id:03d}",
            messages=session.get("messages"),
            tool_call_stats=session.get("tool_call_stats"),
            n_turns=session.get("n_turns"),
            role="setup_encoder", meta={"kb_id": kb_id},
            status=final.status, output=final.model_dump(),
            error=final.error, duration_s=time.monotonic() - started,
        )
        if index_rows is not None:
            index_rows.append(row)
    return final


def make_setup_build_encoder(
    *,
    model: Any,
    model_settings: Any,
    self_model_id: str,
    build_shared_slayer_server,
) -> Any:
    """Return a ``build_encoder(storage, build_dir, db) -> run_one`` for
    :func:`reference_build.ensure_db_reference`.

    The returned ``run_one`` is awaited concurrently by the scheduler; all
    encoders share ONE MCP server. ``reference_build`` enters it via
    ``run_one.aopen`` and closes it via ``run_one.aclose`` — both from its own
    (parent) task, BEFORE/AFTER the concurrent ``_encode_all`` fan-out. This is
    mandatory: ``MCPServerStdio.__aenter__`` opens an anyio ``stdio_client``
    whose cancel scope is bound to the entering task, so entering inside a
    gather'd child task and exiting in the parent raises "Attempted to exit
    cancel scope in a different task than it was entered in". Entering in the
    parent before any fan-out also means exactly one ``__aenter__`` (no
    double-subprocess race), so no entry lock is needed. Cross-encoder write
    safety relies on the stdio server processing tool calls serially +
    ``edit_model``'s upsert-by-name + the post-build collision check
    (reference_build) — not a bespoke lock. ``build_shared_slayer_server`` is
    injected (``agent.py``'s ``_build_shared_slayer_server``) so this module
    needs no harness MCP-config import. The real ``db`` is passed in (NOT
    inferred from the tmp build-dir name) so the encoder prompts + entity
    verification use the true datasource id.
    """
    def build_encoder(storage: Any, build_dir, db: str, sessions_dir: Any = None):
        server = build_shared_slayer_server(str(build_dir))
        agent = _build_setup_encoder(
            model=model, model_settings=model_settings,
            shared_slayer_server=server, self_model_id=self_model_id,
        )
        state: dict[str, Any] = {"entered": False, "cm": None}
        # Per-DB session index rows, appended by each run_one. Single-threaded
        # asyncio + list.append (no await between read/modify) → race-free.
        index_rows: list[dict] = []
        # Per-DB setup-encode token usage/cost, accumulated across every KB
        # encode (scope "setup_encoder"). Exposed on run_one so reference_build
        # can persist it next to the reference.
        usage = TokenUsage()

        async def aopen() -> None:
            # Enter the shared MCP server in the CALLER's task (reference_build's
            # parent task), before the concurrent fan-out. Same task must later
            # call aclose() — anyio cancel scopes are task-bound.
            if server is None or state["entered"]:
                return
            state["cm"] = server
            await server.__aenter__()
            state["entered"] = True

        async def run_one(kb_id, row, deps_results, *, reverse_deps=None):
            # Server already entered by aopen() in the parent task; never
            # __aenter__ here (this runs inside a gather'd child task).
            # _run_setup_encoder appends the FULL triage row (turns / tool calls
            # / tool errors — Codex finding #4) to index_rows via write_session.
            # `reverse_deps` (DEV-1466): the rows of the KBs that REFERENCE this
            # one, surfaced to the prompt so a value_illustration can defer an
            # embedded score to a calculation_knowledge parent that owns it.
            kb_body = _row_to_body(row)
            return await _run_setup_encoder(
                agent=agent, kb_id=kb_id, kb_body=kb_body,
                deps_results=deps_results, storage=storage, db=db,
                self_model_id=self_model_id, sessions_dir=sessions_dir,
                index_rows=index_rows, reverse_deps=reverse_deps, usage=usage,
            )

        async def aclose() -> None:
            if state["entered"] and state["cm"] is not None:
                await state["cm"].__aexit__(None, None, None)
                state["entered"] = False

        run_one.aopen = aopen  # type: ignore[attr-defined]
        run_one.aclose = aclose  # type: ignore[attr-defined]
        run_one.index_rows = index_rows  # type: ignore[attr-defined]
        run_one.usage = usage  # type: ignore[attr-defined]
        return run_one

    return build_encoder


def _row_to_body(row: dict) -> str:
    """Render a raw KB row as the prompt body (knowledge + verbatim block) —
    the same shape the memory carries, so the setup and task encoders see
    equivalent content."""
    import yaml as _yaml

    kid = row.get("id")
    knowledge = row.get("knowledge", "")
    return (
        f"KB {kid} — {knowledge}\n\n"
        f"KB item (verbatim):\n"
        f"{_yaml.safe_dump(row, sort_keys=False).strip()}"
    )
