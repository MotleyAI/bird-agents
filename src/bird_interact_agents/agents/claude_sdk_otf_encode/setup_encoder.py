"""DEV-1589: the claude_sdk-native build-time OTF *reference* encoder.

One agent per KB item, run SERIALLY (a per-build ``asyncio.Lock``), each with a
fresh hermetic ``ClaudeSDKClient`` + its own SLayer **stdio** MCP subprocess
over the shared ``build_dir`` store. The agent wires its own output (writes the
entity, injects the verbatim KB row into its description, ``save_memory``-s the
backref); a <=3 corrective re-prompt loop on the warm client enforces HC-present
+ HC-depuse, then downgrades-to-deferred + purges partials. HC-desc + HC-mem are
auto-completed deterministically by the harness after the session closes (the
SLayer subprocess is then dead, so the parent process is the single writer).

Because builds are serial and each KB's subprocess dies at session end, the
parent-process auto-wire / purge between KBs is single-writer-safe, and the next
KB's fresh subprocess reads the committed YAML at startup.

``make_claude_sdk_build_encoder`` returns the
``build_encoder(storage, build_dir, db, sessions_dir) -> run_one`` callable that
``slayer_otf.reference_build.ensure_db_reference`` consumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    create_sdk_mcp_server,
    tool,
)
from pydantic import ValidationError
from slayer.inspect.service import InspectService
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.agents._session_log import write_session
from bird_interact_agents.agents._slayer_tool_surface import (
    derive_disallowed_slayer_tools,
)
from bird_interact_agents.agents.claude_sdk.agent import (
    SdkUsageTracker,
    _ctx,
    _ctx_var,
    _text,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
)
from bird_interact_agents.agents.claude_sdk_otf_encode.prompts import (
    ENCODER_PROMPT,
)
from bird_interact_agents.harness import MAX_MODEL_TURNS, slayer_mcp_stdio_config
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.slayer_otf import encoder_verify as ev
from bird_interact_agents.slayer_otf.encoder_types import EncoderResult
from bird_interact_agents.slayer_otf.kb_memory_encoder import _normalise_children
from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_tool_filters,
)
from bird_interact_agents.slayer_otf.timing import log_otf_event, otf_timer
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


# Encode-then-query is turn-expensive; give the same headroom as the OTF agents.
_MAX_TURNS = 2 * MAX_MODEL_TURNS

# 1 initial attempt + up to 3 corrective re-prompts.
MAX_ENCODE_ATTEMPTS = 4

# Per-attempt receive backstop. The per-build lock is held across the whole
# session, so a hung stream would otherwise block every later KB. A timeout
# converts the hang into a per-KB ``status="error"``. 300s (was 600s): the
# submit-or-defer nudge below converts most strugglers into a clean defer well
# before this, so the cap only catches genuine hangs (DEV-1589 smoke diagnosis).
ENCODE_ATTEMPT_TIMEOUT_S = 300.0

# Backstop for the SDK session ENTER (the bundled `claude` CLI + slayer stdio
# MCP handshake + parity probe), which the per-attempt timeout above does NOT
# cover. A healthy enter is seconds (DEV-1561, telemetry disabled); 240s is far
# above that yet bounds a genuine handshake hang. DEV-1609: the maiden cloud
# on-the-fly encode stalled forever in `__aenter__` (actor alive, zero output,
# killed only by the 2h VM self-delete timer) — this converts that into a clean
# per-KB `status="error"`.
SESSION_ENTER_TIMEOUT_S = 240.0

# Secondary cap on the force-close TEARDOWN inside _drive_with_timeout. The SDK's
# disconnect() has no internal timeout and a cancelled drive task may ignore
# cancellation, so disconnect()+drain could itself hang — wedging the very path
# meant to contain a hang (CodeRabbit). If teardown stalls past this, abandon it
# and raise anyway, so _drive_with_timeout ALWAYS returns within its budget.
TEARDOWN_TIMEOUT_S = 30.0

# Soft turn budget for the submit-or-defer nudge (NOT the hard ``max_turns``=120
# cap). The DEV-1589 glm-5.2 smoke showed a healthy encode needs ~10 tool calls,
# while the KBs that timed out looped 40-70+ tool calls WITHOUT ever calling
# submit_encoding (neither encode nor defer) — the hard turn cap (120) sits above
# even that, so a nudge keyed to it never fires before the wall-clock timeout.
# Once a KB crosses this soft budget, EVERY subsequent turn nudges it to submit
# (encode) or defer, converting dead-end loops into clean deferrals.
_NUDGE_TURN_BUDGET = 35

# SLayer MCP tools the encoder may call. `save_memory` is REQUIRED (the agent
# wires its own KB memory); `query_nested` lets it self-test a multi-stage DAG.
SLAYER_MCP_TOOLS = [
    "create_model", "edit_model", "save_memory", "validate_models",
    "help", "search", "models_summary", "inspect_model", "inspect",
    "recommend_root_model", "query", "query_nested",
]

NATIVE_TOOL_NAME = "mcp__bird-interact-tools__submit_encoding"

# Write tools whose backing-query `filters` must be normalised
# (`lower(trim(col)) = '<lower>'`) BEFORE SLayer persists them — otherwise the
# committed reference bakes case-sensitive raw text-equality predicates that
# break case-insensitive pre-encoded runs. Mirrors the pydantic encoder + the
# claude_sdk_otf agent (Codex review).
_WRITE_TOOLS_NEEDING_NORMALIZATION = (
    "mcp__slayer__create_model",
    "mcp__slayer__edit_model",
)
_NORMALIZE_WRITE_FILTERS_MATCHER = "|".join(_WRITE_TOOLS_NEEDING_NORMALIZATION)


async def _normalize_write_tool_filters_hook(input_data, tool_use_id, context):
    """PreToolUse: rewrite create_model/edit_model `filters` to
    `lower(trim(col)) = '<lower>'` before SLayer persists them. Returns the SDK's
    `updatedInput` directive; a no-op normalisation returns the (deep-copied)
    input unchanged."""
    tool_name = input_data.get("tool_name") or ""
    if tool_name not in _WRITE_TOOLS_NEEDING_NORMALIZATION:
        return {}
    bare = tool_name.split("__", 2)[-1]
    tool_input = input_data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return {}
    updated = normalize_tool_filters(bare, tool_input)
    if not isinstance(updated, dict):
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse", "updatedInput": updated,
        }
    }


# ---------------------------------------------------------------------------
# The single native: submit_encoding (report-only; captures into _ctx)
# ---------------------------------------------------------------------------


@tool(
    "submit_encoding",
    "Report what you encoded for this KB as an EncoderResult JSON. Call EXACTLY "
    "ONCE at the end with `result_json` = a JSON object: {\"kb_id\": <int>, "
    "\"status\": \"encoded\"|\"deferred\", \"entities\": [{\"kind\": "
    "\"column\"|\"measure\"|\"aggregation\"|\"model\", \"host_model\": <model "
    "name or null>, \"name\": <entity name>, \"entity_ref\": "
    "\"<db>.<model>[.<leaf>]\"}], \"notes\": <1-2 sentences>, "
    "\"clarifying_questions\": [<str>, ...]}.",
    {"result_json": str},
)
async def submit_encoding(args: dict) -> dict:
    raw = args.get("result_json")
    try:
        result = EncoderResult.model_validate_json(raw)
    except (ValidationError, ValueError, TypeError) as e:
        return _text(f"submit_encoding: invalid EncoderResult JSON: {e}")
    _ctx["_encoder_submission"] = result
    return _text("recorded")


def allowed_tool_names() -> list[str]:
    return [f"mcp__slayer__{t}" for t in SLAYER_MCP_TOOLS] + [NATIVE_TOOL_NAME]


def disallowed_tool_names() -> list[str]:
    """DEV-1644: the SLayer tools to HIDE from the model's per-turn context,
    DERIVED as the complement of `SLAYER_MCP_TOOLS` against the live SLayer
    surface. One allow-list to maintain; nothing leaks by construction (the
    old hand-written deny-list drifted and left `list_datasources` /
    `describe_datasource` / `edit_datasource` / `delete_model` visible)."""
    return derive_disallowed_slayer_tools(SLAYER_MCP_TOOLS)


def _make_submit_or_defer_nudge(soft_budget: int = _NUDGE_TURN_BUDGET):
    """PostToolUse hook: once a single KB has spent more than ``soft_budget``
    tool calls (far more than a normal encoding needs), nudge the model on EVERY
    subsequent turn to STOP exploring and call ``submit_encoding`` — encode if it
    can, otherwise DEFER. Counts its own invocations (≈ one per turn), so it is
    per-KB self-contained. Sustained (not a one-shot window) so the pressure
    holds until the model actually submits (DEV-1589 smoke diagnosis: the timed-
    out KBs looped without ever submitting)."""
    state = {"calls": 0}

    async def _hook(input_data, tool_use_id, context):
        state["calls"] += 1
        if state["calls"] < soft_budget:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": (
                    f"[BUDGET] You have made {state['calls']} tool calls on this "
                    "single KB — already far more than a normal encoding needs. "
                    "STOP exploring. EITHER (a) write the entity/entities now "
                    "(tagged with this KB's meta.kb_id) and call submit_encoding "
                    "with status='encoded'; OR (b) if you cannot confidently pin "
                    "this KB down, call submit_encoding with status='deferred' and "
                    "your clarifying_questions. A deferred KB is an ACCEPTABLE "
                    "outcome; an un-submitted KB is recorded as an error."
                ),
            }
        }

    return _hook


# ---------------------------------------------------------------------------
# Prompt-body helpers
# ---------------------------------------------------------------------------


def _row_to_body(row: dict) -> str:
    kid = row.get("id")
    knowledge = row.get("knowledge", "")
    return (
        f"KB {kid} — {knowledge}\n\n"
        f"KB item (verbatim):\n{yaml.safe_dump(row, sort_keys=False).strip()}"
    )


def _verbatim_kb_block(row: dict) -> str:
    """The `[kb=N]` + verbatim KB row injected into each entity's description."""
    kid = row.get("id")
    dump = yaml.safe_dump(row, sort_keys=False).rstrip()
    return f"[kb={kid}]\nKB item (verbatim):\n{dump}"


def _corrective_message(kb_id: int, failures: list[str]) -> str:
    joined = "\n".join(f"- {f}" for f in failures)
    return (
        f"Your encoding of KB {kb_id} is NOT yet acceptable:\n{joined}\n\n"
        f"Fix the entities in the store (write them, tag meta.kb_id={kb_id}, "
        f"reference dependencies by name rather than re-deriving), then call "
        f"submit_encoding again."
    )


# ---------------------------------------------------------------------------
# build_encoder factory
# ---------------------------------------------------------------------------


def make_claude_sdk_build_encoder(
    *,
    model: str,
    reasoning_effort: str | None = None,
    self_model_id: str,
) -> Any:
    """Return ``build_encoder(storage, build_dir, db, sessions_dir) -> run_one``.

    ``model`` is the RAW registry/provider model string (e.g. ``zai/glm-5.2``);
    the hermetic session layers its provider base-url/auth.
    """

    def build_encoder(storage: Any, build_dir: Any, db: str, sessions_dir: Any = None):
        build_dir = Path(build_dir)
        index_rows: list[dict] = []
        usage = TokenUsage()
        lock = asyncio.Lock()  # E1 — serialise per-build (one subprocess alive)

        native_server = create_sdk_mcp_server(
            name="bird-interact-tools", version="1.0.0", tools=[submit_encoding],
        )
        # Build the MCP server set ONCE and pass it BOTH to the options builder
        # AND to hermetic_claude_sdk_session: the hermetic parity assertion's
        # `expected` set is exactly the keys handed to the session, so the
        # `slayer` stdio server MUST be declared there or every KB raises
        # HermeticEnvError before the first query (Codex review, critical). A
        # fresh subprocess is still spawned per KB on each session enter; the
        # config dict is reusable.
        mcp_servers = {
            "bird-interact-tools": native_server,
            "slayer": slayer_mcp_stdio_config(
                str(build_dir), ingest_on_startup=False,
            ),
        }

        def _build_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
            return ClaudeAgentOptions(
                **_opt_kwargs,
                mcp_servers=mcp_servers,
                allowed_tools=allowed_tool_names(),
                disallowed_tools=disallowed_tool_names(),
                tools=[],
                setting_sources=[],
                model=native_model_id(model),
                effort=reasoning_effort,
                max_turns=_MAX_TURNS,
                hooks={
                    "PreToolUse": [HookMatcher(
                        matcher=_NORMALIZE_WRITE_FILTERS_MATCHER,
                        hooks=[_normalize_write_tool_filters_hook],
                    )],
                    # Fresh per-KB nudge state (build_options is invoked once per
                    # KB session, so the counter resets each KB).
                    "PostToolUse": [HookMatcher(
                        hooks=[_make_submit_or_defer_nudge()],
                    )],
                },
            )

        async def aopen() -> None:  # no-op: fresh client per KB inside run_one
            return None

        async def aclose() -> None:
            return None

        async def run_one(kb_id, row, deps_results, *, reverse_deps=None):
            async with lock:
                return await _encode_one_kb(
                    kb_id=kb_id, row=row, deps_results=deps_results,
                    reverse_deps=reverse_deps, db=db, build_dir=build_dir,
                    model=model, self_model_id=self_model_id, usage=usage,
                    index_rows=index_rows, sessions_dir=sessions_dir,
                    mcp_servers=mcp_servers, build_options=_build_options,
                )

        run_one.aopen = aopen  # type: ignore[attr-defined]
        run_one.aclose = aclose  # type: ignore[attr-defined]
        run_one.index_rows = index_rows  # type: ignore[attr-defined]
        run_one.usage = usage  # type: ignore[attr-defined]
        return run_one

    return build_encoder


# ---------------------------------------------------------------------------
# Per-KB encode (serial; runs inside the per-build lock)
# ---------------------------------------------------------------------------


async def _drive_with_timeout(client, drive, timeout_s: float) -> None:
    """Run one query/receive cycle under a hard wall-clock cap that a wedged SDK
    transport read cannot defeat.

    DEV-1609: plain ``asyncio.wait_for(drive(), t)`` can HANG indefinitely here.
    On timeout it cancels the inner coroutine and AWAITS its cancellation — but
    the Claude Agent SDK's transport read runs through an anyio-threaded
    subprocess, and when the CLI never responds (the cloud failure mode), the
    cancellation never returns, so ``wait_for`` itself never returns either.
    Instead, time the cycle with ``asyncio.wait`` (which does NOT cancel on
    timeout), then on expiry force-close the client — unblocking the stuck read
    — drain the task, and raise ``TimeoutError`` so the caller records a per-KB
    error rather than blocking forever."""
    task = asyncio.ensure_future(drive())
    # Retrieve the eventual result so an abandoned/late drive task (timeout path
    # below) never logs an "exception was never retrieved" warning. Harmless on
    # the success path — task.result() there retrieves first.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    done, _pending = await asyncio.wait({task}, timeout=timeout_s)
    if task not in done:
        # Force-close to break the wedged read, then drain the task — but BOTH
        # disconnect() and the drain can themselves hang (CodeRabbit). Run them
        # as a child task capped by asyncio.wait (which does NOT cancel on
        # timeout): if teardown stalls past TEARDOWN_TIMEOUT_S, abandon the
        # leaked child (the actor/process reaps it) and raise anyway, so this
        # function ALWAYS returns within timeout_s + TEARDOWN_TIMEOUT_S.
        async def _teardown() -> None:
            # Cancel the drive task FIRST (Codex): otherwise, if disconnect()
            # hangs, the cancel is never even requested and the drive coroutine
            # stays alive holding the SDK stream / MCP subprocess — free to
            # write into the build dir after _encode_one_kb finalizes. With the
            # cancel requested up front, the task dies the moment disconnect (or
            # process teardown) unblocks the read.
            task.cancel()
            with contextlib.suppress(Exception):
                await client.disconnect()  # unblock the read so the cancel lands
            with contextlib.suppress(BaseException):
                await task  # drain (bounded by the outer asyncio.wait below)

        td = asyncio.ensure_future(_teardown())
        # Retrieve the result once done so an abandoned/late child doesn't log
        # an "exception was never retrieved" warning.
        td.add_done_callback(lambda t: t.cancelled() or t.exception())
        await asyncio.wait({td}, timeout=TEARDOWN_TIMEOUT_S)
        raise TimeoutError(
            f"encode attempt exceeded {timeout_s}s; SDK client force-closed"
        )
    # Completed within budget — surface any exception raised inside the cycle.
    task.result()


async def _encode_one_kb(
    *, kb_id, row, deps_results, reverse_deps, db, build_dir, model,
    self_model_id, usage, index_rows, sessions_dir, mcp_servers, build_options,
) -> EncoderResult:
    started = time.monotonic()
    # Per-KB milestone on the (already INFO-configured, cloud-visible)
    # otf_timing logger so a healthy build shows KB-by-KB progress and a stall
    # is localised — root logging defaults to WARNING in the cloud actor, so a
    # bare module `logger.info` would not surface (DEV-1609 diagnosis).
    log_otf_event("setup_encoder.kb.start", instance_id=f"{db}_kb_{kb_id}")
    ctx = {"_encoder_submission": None}
    _ctx_var.set(ctx)

    kb_body = _row_to_body(row)
    verbatim_desc = _verbatim_kb_block(row)
    declared_ids = set(_normalise_children(row.get("children_knowledge")))
    encoded_deps = [
        d for d in (deps_results or [])
        if getattr(d, "status", None) == "encoded"
        and getattr(d, "entities", None)
        and d.kb_id in declared_ids
    ]
    confirmed: EncoderResult | None = None        # fresh submission that passed all checks
    last_submission: EncoderResult | None = None  # most recent submission (any status)
    failures: list[str] = []
    err: str | None = None
    transcript: list[dict] = []                   # captured by the wrapper's recording proxy
    try:
        # Inside the try so an inspect()/format failure surfaces as a per-KB
        # error rather than crashing the whole build.
        prompt = ENCODER_PROMPT.format(
            db_name=db, kb_id=kb_id, kb_body=kb_body,
            deps_block=await _format_deps_block(encoded_deps, build_dir, db),
            reverse_deps_block=_format_reverse_deps_block(reverse_deps),
        )
        async with hermetic_claude_sdk_session(
            model,
            mcp_servers=mcp_servers,
            build_options=build_options,
            enter_timeout_s=SESSION_ENTER_TIMEOUT_S,
            enter_cm_factory=lambda: otf_timer(
                "setup_encoder.sdk_client_enter", instance_id=f"{db}_kb_{kb_id}",
            ),
        ) as client:
            # Live reference to the wrapper's recording-proxy transcript list —
            # captured BEFORE the loop so a timeout/exception still leaves the
            # partial transcript (the cases most worth inspecting).
            transcript = getattr(client, "transcript", transcript)
            for attempt in range(MAX_ENCODE_ATTEMPTS):
                ctx["_encoder_submission"] = None
                msg = prompt if attempt == 0 else _corrective_message(kb_id, failures)
                # Pass the FULL provider-prefixed model string (NOT the stripped
                # self_model_id) so the registry/litellm pricing in
                # TokenUsage.add_call resolves it (same as the claude_sdk agents).
                cycle_tracker = SdkUsageTracker(
                    usage, model, scope="setup_encoder",
                )

                async def _drive(_msg=msg, _tracker=cycle_tracker):
                    await client.query(_msg)
                    agen = client.receive_response()
                    try:
                        async for m in agen:
                            _tracker.observe(m)
                            if type(m).__name__ == "ResultMessage":
                                break
                    finally:
                        aclose = getattr(agen, "aclose", None)
                        if aclose is not None:
                            await agen.aclose()

                try:
                    await _drive_with_timeout(
                        client, _drive, ENCODE_ATTEMPT_TIMEOUT_S,
                    )
                finally:
                    cycle_tracker.finalize()

                attempt_sub = ctx["_encoder_submission"]
                if attempt_sub is None:
                    # No fresh submission this attempt.
                    if attempt == 0:
                        break  # never submitted at all → error path
                    # A corrective attempt that edited storage WITHOUT re-calling
                    # submit_encoding is NOT accepted by re-verifying a prior
                    # submission — keep prompting for an explicit re-submit, so a
                    # timeout/exception after a silent fix can't finalize a stale
                    # encoding (Codex review).
                    failures = [
                        "You did not call submit_encoding this turn — call it to "
                        "confirm the entities you encoded for this KB."
                    ]
                    continue
                last_submission = attempt_sub
                if attempt_sub.status != "encoded":
                    break  # explicit deferred/error — nothing to verify
                failures = list(await ev.hard_failures(
                    build_dir, db, kb_id, attempt_sub.entities, encoded_deps,
                ))
                # An 'encoded' result that lists NO entities is not a real
                # encoding (legacy verifier parity).
                if not attempt_sub.entities:
                    failures.append("claimed 'encoded' but listed no entities")
                if not failures:
                    confirmed = attempt_sub   # passed IN THIS attempt — definitive
                    break
    except Exception as exc:  # noqa: BLE001 — never abort the build
        err = f"{type(exc).__name__}: {exc}"
        logger.exception("claude_sdk encoder failed for kb_id=%s", kb_id)

    # ---- client/subprocess CLOSED → parent storage is single-writer-safe ----
    result = await _finalize_kb(
        kb_id=kb_id, confirmed=confirmed, last_submission=last_submission,
        last_failures=failures, err=err, db=db, build_dir=build_dir,
        verbatim_desc=verbatim_desc,
    )

    _append_index_row(
        index_rows, sessions_dir, kb_id, result, time.monotonic() - started,
        transcript,
    )
    log_otf_event(
        "setup_encoder.kb.done", instance_id=f"{db}_kb_{kb_id}",
        status=result.status, elapsed_s=f"{time.monotonic() - started:.3f}",
    )
    return result


async def _finalize_kb(
    *, kb_id, confirmed, last_submission, last_failures, err, db, build_dir,
    verbatim_desc,
) -> EncoderResult:
    """Act on the loop's verdict WITHOUT re-verifying — only a submission that
    passed all hard checks IN-LOOP (``confirmed``) is accepted as encoded, so a
    stale prior submission can never be re-verified into success on the
    timeout/exception path (Codex review)."""
    if confirmed is not None:
        result = confirmed.model_copy(update={"kb_id": kb_id})
        # encoded + all hard checks passed → auto-wire the mechanical properties.
        await ev.autowire_descriptions(build_dir, db, result.entities, verbatim_desc)
        await ev.autowire_memory_backrefs(build_dir, db, kb_id, result.entities)
        return result

    # Not confirmed → purge ALL of this KB's tagged partial writes (not just any
    # reported entities) and force entities=[], so no orphan entity / dangling
    # backref survives into the committed reference.
    await ev.purge_kb_entities_and_backrefs(build_dir, db, kb_id)

    if err is not None:
        # A timeout / exception interrupted the encode — surface it as `error`
        # (with the message), NOT a clean `deferred`, even if an earlier attempt
        # had submitted a (failed) encoded result (Codex review).
        return EncoderResult(
            kb_id=kb_id, status="error", entities=[], error=err,
            notes=(last_submission.notes if last_submission else ""),
        )
    if last_submission is None:
        return EncoderResult(
            kb_id=kb_id, status="error", entities=[],
            error="agent never called submit_encoding",
        )
    if last_submission.status == "encoded":
        # encoded but failed the hard checks across all attempts → defer.
        extra = (
            " | downgraded: " + "; ".join(last_failures)
            if last_failures else " | downgraded: failed hard checks"
        )
        return last_submission.model_copy(update={
            "kb_id": kb_id, "status": "deferred", "entities": [],
            "notes": last_submission.notes + extra,
        })
    # explicit deferred / error submission — keep its status/notes, clear entities.
    return last_submission.model_copy(update={"kb_id": kb_id, "entities": []})


def _append_index_row(index_rows, sessions_dir, kb_id, result, duration_s,
                      transcript=None):
    if sessions_dir is None:
        index_rows.append({
            "session_id": f"setup__kb_{kb_id:03d}", "role": "setup_encoder",
            "kb_id": kb_id, "status": result.status, "error": result.error,
        })
        return
    row = write_session(
        sessions_dir, f"setup__kb_{kb_id:03d}", messages=transcript or [],
        role="setup_encoder", meta={"kb_id": kb_id}, status=result.status,
        output=result.model_dump(), error=result.error, duration_s=duration_s,
    )
    index_rows.append(row)


# ---------------------------------------------------------------------------
# Prompt-block renderers (declared-dep + reverse-dep context)
# ---------------------------------------------------------------------------


async def _format_deps_block(encoded_deps, build_dir, db) -> str:
    """Hand each task the FULL detail of its dependencies' CREATED ENTITIES via
    SLayer's ``inspect(compact=False)`` (DEV-1588, motley-slayer >= 0.8.3) — the
    canonical single-entity point-lookup: formula + full description + label,
    exactly what the agent would get by inspecting the entity itself.

    DEV-1589 smoke: the KBs that timed out looped on search/inspect of their own
    dependencies (SLayer's default compact search returns bare refs). Handing
    over the inspect rendering up-front removes that whole exploration round-trip.
    """
    if not encoded_deps:
        return "(none)"
    svc = InspectService(storage=YAMLStorage(base_dir=str(build_dir)))
    blocks: list[str] = []
    for dep in sorted(encoded_deps, key=lambda d: d.kb_id):
        parts = [
            f"### KB {dep.kb_id} is ALREADY ENCODED — reference these entities "
            f"BY NAME, do NOT re-derive them:"
        ]
        for e in dep.entities:
            detail = await svc.inspect(
                reference=e.entity_ref, entity_type=e.kind, compact=False,
            )
            parts.append(f"Entity `{e.entity_ref}` ({e.kind}):\n{detail}")
        notes = (dep.notes or "").strip()
        if notes:
            parts.append(f"(its encode notes: {notes[:240]})")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _format_reverse_deps_block(parents: list[dict] | None) -> str:
    """Render the KBs that REFERENCE this one (its parents) so a
    value_illustration can defer an embedded score to a calculation_knowledge
    parent that owns it (DEV-1466). Sorted by id, deduped, fields truncated so
    a long definition can't blow the prompt budget."""
    if not parents:
        return "(none)"
    seen: set[int] = set()
    lines: list[str] = []
    for p in sorted(parents, key=lambda r: int(r.get("id", 0))):
        pid = int(p.get("id", 0))
        if pid in seen:
            continue
        seen.add(pid)
        line = f"KB {pid}"
        typ = p.get("type")
        if typ:
            line += f" ({typ})"
        knowledge = str(p.get("knowledge", "") or "")[:200]
        if knowledge:
            line += f": {knowledge}"
        definition = str(p.get("definition", "") or "")[:200]
        if definition:
            line += f" | def: {definition}"
        lines.append(line)
    return "\n".join(lines)
