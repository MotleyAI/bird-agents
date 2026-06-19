"""DEV-1555 v0/v1 split — v0 agent structure + surgical edits.

The four v0 agents (``claude_sdk_otf``, ``claude_sdk_otf_raw``,
``claude_sdk_otf_ainteract``, ``claude_sdk_otf_ainteract_raw``) are
restored from origin/main with exactly THREE surgical edits each:

1. Imports ``SdkUsageTracker`` from ``claude_sdk.agent`` (replaces
   ``accumulate_assistant_usage``).
2. The receive-loop body uses ``tracker.observe(msg)`` instead of
   ``accumulate_assistant_usage(accum, msg, ...)``.
3. The ``except Exception`` block contains a ``tracker.finalize()``
   call (so error-path ``accum.model_dump()`` is populated — the
   tracker only auto-finalizes on terminal ``ResultMessage``).

For the ``query`` MCP tool, v0 agents import from
``agents.claude_sdk._query_v0`` (NOT from ``claude_sdk.agent``), so the
agent sees the origin/main ``query_json``-shaped schema that v0 prompts
were written against.

These tests inspect SOURCE BYTES rather than the live agent class —
the live class requires real SDK + slayer storage which is not set up
in unit tests. Source-byte inspection is precise enough for the
three-edit contract.
"""

from __future__ import annotations

import ast
import importlib
import re

import pytest


_V0_AGENT_PACKAGES = (
    "bird_interact_agents.agents.claude_sdk_otf",
    "bird_interact_agents.agents.claude_sdk_otf_raw",
    "bird_interact_agents.agents.claude_sdk_otf_ainteract",
    "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
)


def _agent_source(package: str) -> str:
    mod = importlib.import_module(f"{package}.agent")
    path = mod.__file__
    assert path is not None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _agent_ast(package: str) -> ast.Module:
    return ast.parse(_agent_source(package))


def _walk_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _tracker_var_names(tree: ast.AST) -> set[str]:
    """Names assigned `SdkUsageTracker(accum, self.model)` (or compatible)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        callee = node.value.func
        callee_name = (
            callee.id if isinstance(callee, ast.Name)
            else callee.attr if isinstance(callee, ast.Attribute)
            else None
        )
        if callee_name != "SdkUsageTracker":
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                names.add(tgt.id)
    return names


def _attr_call_inside(tree_or_node: ast.AST, var: str, method: str) -> list[ast.Call]:
    """Find calls of the form ``<var>.<method>(...)`` anywhere in the subtree."""
    matches: list[ast.Call] = []
    for node in ast.walk(tree_or_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == var
            and func.attr == method
        ):
            matches.append(node)
    return matches


def _except_or_finally_handlers(tree: ast.Module):
    """Yield every ExceptHandler body AND finalbody from the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                yield h
            if node.finalbody:
                # synthesize a pseudo-handler whose body is finalbody
                yield ast.copy_location(
                    ast.ExceptHandler(type=None, name=None, body=node.finalbody),
                    node,
                )


# ---------------------------------------------------------------------------
# Package existence + class exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg,cls_name",
    [
        ("bird_interact_agents.agents.claude_sdk_otf", "ClaudeSDKOtfAgent"),
        ("bird_interact_agents.agents.claude_sdk_otf_raw", "ClaudeSDKOtfRawAgent"),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract",
            "ClaudeSDKOtfAInteractAgent",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
            "ClaudeSDKOtfAInteractRawAgent",
        ),
    ],
)
def test_v0_package_exports_class(pkg: str, cls_name: str):
    """Each v0 package re-exports its agent class at the package root."""
    mod = importlib.import_module(pkg)
    assert hasattr(mod, cls_name), (
        f"{pkg} does not export {cls_name} "
        f"(public: {[n for n in dir(mod) if not n.startswith('_')]})"
    )


# ---------------------------------------------------------------------------
# Surgical edit 1+2: SdkUsageTracker is imported and called in the loop;
# `accumulate_assistant_usage(accum, msg, ...)` per-message call is GONE.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", _V0_AGENT_PACKAGES)
def test_v0_imports_sdk_usage_tracker(pkg: str):
    """v0 agent.py imports ``SdkUsageTracker`` from claude_sdk.agent."""
    src = _agent_source(pkg)
    assert re.search(
        r"from\s+bird_interact_agents\.agents\.claude_sdk\.agent\s+import\b[^)]*\bSdkUsageTracker\b",
        src,
        re.DOTALL,
    ), (
        f"{pkg}/agent.py must import SdkUsageTracker from claude_sdk.agent."
    )


@pytest.mark.parametrize("pkg", _V0_AGENT_PACKAGES)
def test_v0_uses_tracker_observe_in_loop(pkg: str):
    """Receive loop calls ``<tracker_var>.observe(msg)`` (Surgical edit 2).

    The variable name is whatever was assigned ``SdkUsageTracker(...)``.
    We use AST to find that assignment, then assert there's an
    ``<assigned_var>.observe(msg)`` call somewhere in the module body
    (typically inside the ``async for msg in client.receive_response()``
    loop).
    """
    tree = _agent_ast(pkg)
    var_names = _tracker_var_names(tree)
    assert var_names, (
        f"{pkg}/agent.py must assign `<var> = SdkUsageTracker(...)`."
    )
    found_observe = False
    for var in var_names:
        for call in _attr_call_inside(tree, var, "observe"):
            # `observe(msg)` — assert msg is the first positional arg.
            args = call.args
            if (
                args
                and isinstance(args[0], ast.Name)
                and args[0].id == "msg"
            ):
                found_observe = True
                break
        if found_observe:
            break
    assert found_observe, (
        f"{pkg}/agent.py: no `<tracker_var>.observe(msg)` call found "
        f"(checked tracker vars: {sorted(var_names)}). See Plan surgical edit 2."
    )


@pytest.mark.parametrize("pkg", _V0_AGENT_PACKAGES)
def test_v0_drops_per_message_accumulate(pkg: str):
    """``accumulate_assistant_usage(...)`` is NOT called per message anymore.

    The legacy accumulator is the very thing the tracker replaces; if a
    call survives the surgical edit, usage will be double-counted.
    """
    src = _agent_source(pkg)
    # Tolerate `from ... import accumulate_assistant_usage` (the legacy
    # symbol may stay importable), but no call site.
    matches = re.findall(r"accumulate_assistant_usage\s*\(", src)
    assert not matches, (
        f"{pkg}/agent.py still calls accumulate_assistant_usage(...) "
        f"({len(matches)} occurrences); replace with SdkUsageTracker."
    )


@pytest.mark.parametrize("pkg", _V0_AGENT_PACKAGES)
def test_v0_instantiates_tracker_with_correct_ctor_args(pkg: str):
    """``SdkUsageTracker(accum, self.model)`` — pin constructor args.

    The tracker takes ``(accum: TokenUsage, model: str)``. Per-task
    instantiation must pass the SAME ``accum`` the agent's row uses,
    and the SAME ``self.model`` the agent's class carries — otherwise
    usage rows land on the wrong scope/model and breakdowns get
    mis-attributed.
    """
    tree = _agent_ast(pkg)
    constructors: list[ast.Call] = []
    for call in _walk_calls(tree):
        func = call.func
        callee_name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else None
        )
        if callee_name == "SdkUsageTracker":
            constructors.append(call)
    assert constructors, (
        f"{pkg}/agent.py must instantiate SdkUsageTracker(...) — found "
        "the import but no constructor call."
    )
    # At least one constructor call must match the (accum, self.model)
    # signature. Tolerate keyword args.
    matched = False
    for ctor in constructors:
        pos = ctor.args
        kws = {kw.arg: kw.value for kw in ctor.keywords}
        first = (
            pos[0] if len(pos) >= 1 else kws.get("accum")
        )
        second = (
            pos[1] if len(pos) >= 2 else kws.get("model")
        )
        first_is_accum = (
            isinstance(first, ast.Name) and first.id == "accum"
        )
        second_is_self_model = (
            isinstance(second, ast.Attribute)
            and isinstance(second.value, ast.Name)
            and second.value.id == "self"
            and second.attr == "model"
        )
        if first_is_accum and second_is_self_model:
            matched = True
            break
    assert matched, (
        f"{pkg}/agent.py: no SdkUsageTracker(accum, self.model) constructor "
        "call found. Surgical edit 1 requires the tracker share the row's "
        "accum and the agent's self.model."
    )


# ---------------------------------------------------------------------------
# Surgical edit 3: error-path finalize.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", _V0_AGENT_PACKAGES)
def test_v0_finalize_inside_except_or_finally(pkg: str):
    """``<tracker_var>.finalize()`` must live inside an ``except`` or ``finally`` block.

    On origin/main, the exception handler reads ``accum.model_dump()``
    expecting per-message accumulation to have populated it. With
    SdkUsageTracker, ``observe()`` doesn't write to accum until
    ``finalize()`` is called — which only auto-fires on terminal
    ``ResultMessage``. An exception before ResultMessage leaves accum
    empty unless we explicitly finalize INSIDE the exception path.

    Plain ``tracker.finalize()`` outside an except/finally block does
    NOT satisfy this contract — the loop's success path already
    finalizes via ``observe(ResultMessage)``; the only place an
    additional finalize matters is the error path.

    (Plan: surgical edit 3.)
    """
    tree = _agent_ast(pkg)
    var_names = _tracker_var_names(tree)
    assert var_names, (
        f"{pkg}/agent.py must assign `<var> = SdkUsageTracker(...)`."
    )
    in_handler = False
    for handler in _except_or_finally_handlers(tree):
        for stmt in handler.body:
            for var in var_names:
                if _attr_call_inside(stmt, var, "finalize"):
                    in_handler = True
                    break
            if in_handler:
                break
        if in_handler:
            break
    assert in_handler, (
        f"{pkg}/agent.py: `<tracker_var>.finalize()` must appear inside "
        f"an `except` or `finally` block (checked tracker vars: "
        f"{sorted(var_names)}). See Plan surgical edit 3."
    )


# ---------------------------------------------------------------------------
# v0 ``query`` tool import points to _query_v0, not claude_sdk.agent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", _V0_AGENT_PACKAGES)
def test_v0_query_imported_from_query_v0_module(pkg: str):
    """v0 agent imports ``query`` from ``claude_sdk._query_v0``.

    Variants of the slayer flavour need the v0 query tool; the raw flavour
    doesn't use query at all (raw agents talk to the DB directly through
    SLAYER-free tools). For raw, the test simply asserts the agent does NOT
    import the v1 ``query`` symbol from claude_sdk.agent.
    """
    src = _agent_source(pkg)
    imports_v0 = bool(
        re.search(
            r"from\s+bird_interact_agents\.agents\.claude_sdk\._query_v0\s+import\b[^)]*\bquery\b",
            src,
            re.DOTALL,
        )
    )
    imports_v1 = bool(
        re.search(
            r"from\s+bird_interact_agents\.agents\.claude_sdk\.agent\s+import\b[^)]*\bquery\b(?!_)",
            src,
            re.DOTALL,
        )
    )
    is_raw_flavour = "raw" in pkg
    if is_raw_flavour:
        assert not imports_v1, (
            f"{pkg}/agent.py (raw flavour) must NOT import the v1 `query` "
            "from claude_sdk.agent; raw agents don't use SLayer tools."
        )
    else:
        assert imports_v0 and not imports_v1, (
            f"{pkg}/agent.py (slayer flavour) must import `query` from "
            "claude_sdk._query_v0 and NOT from claude_sdk.agent. "
            f"(v0 import: {imports_v0}, v1 import: {imports_v1})"
        )


# ---------------------------------------------------------------------------
# Sibling-inheritance is preserved on the v0 side: raw/ainteract/
# ainteract_raw still inherit from the v0 root claude_sdk_otf.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg",
    [
        "bird_interact_agents.agents.claude_sdk_otf_raw",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
    ],
)
def test_v0_siblings_import_from_v0_root(pkg: str):
    """v0 raw/ainteract/ainteract_raw import from v0 claude_sdk_otf.agent."""
    src = _agent_source(pkg)
    assert re.search(
        r"from\s+bird_interact_agents\.agents\.claude_sdk_otf\.agent\s+import\b",
        src,
        re.DOTALL,
    ), (
        f"{pkg}/agent.py must import from claude_sdk_otf.agent (the v0 root); "
        "the v1 sibling imports from claude_sdk_otf_v1.agent."
    )
    # And explicitly NOT from the v1 root.
    assert not re.search(
        r"from\s+bird_interact_agents\.agents\.claude_sdk_otf_v1\.agent\s+import\b",
        src,
        re.DOTALL,
    ), (
        f"{pkg}/agent.py must NOT import from claude_sdk_otf_v1.agent; "
        "that would break the v0/v1 isolation contract."
    )


# ---------------------------------------------------------------------------
# v0 prompts.py imports the *_V0 names from _shared_otf_prompts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pkg,expected_v0_const",
    [
        (
            "bird_interact_agents.agents.claude_sdk_otf",
            None,  # slayer one-shot doesn't reference a v0 SHARED const by
            # name in source (its constant builds via _shared imports
            # with no _V0 suffix necessary at source); the SHA check
            # in test_dev1555_v0_v1_shared_prompts catches the bytes.
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_raw",
            None,
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract",
            None,
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw",
            None,
        ),
    ],
)
def test_v0_prompts_module_loads(pkg: str, expected_v0_const: str | None):
    """v0 prompts module imports cleanly (smoke).

    Byte-identity to origin/main is asserted via SHA in
    ``test_dev1555_v0_v1_shared_prompts.py``; this is the import-time smoke.
    """
    mod = importlib.import_module(f"{pkg}.prompts")
    assert mod is not None
