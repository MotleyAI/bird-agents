"""Shim around BIRD-Interact's upstream graders so cascade tier N1 lines
up with the harness our reported numbers compare to.

Tier N1 ("phase1_against_original_gold") used to be a bag-equality on
``repr(cell)`` over the in-process pred / gold row sets. Upstream's
``test_case_default`` actually does more:

* strips comments / ``DISTINCT`` / ``ROUND`` from BOTH SQLs;
* runs ``preprocess_results`` (2-dp Decimal/float rounding, date
  normalisation, dict/list canonicalisation);
* compares ``set(...) == set(...)``, i.e. dedup.

For mini-interact (SQLite) the upstream lives at
``BIRD-Interact/mini_interact/knowledge_based/mini_interact_conv/
evaluation/test_utils.py``; for the livesqlbench family (Postgres + the
sqlite-shimmed lite variant) at
``livesqlbench/evaluation/src/test_utils.py``. Both expose the same
``ex_base`` / ``remove_*`` API — the only delta is the driver.

Root resolution (per-tree, in order):

1. ``$BIRD_BIRD_INTERACT_ROOT`` / ``$BIRD_LIVESQLBENCH_ROOT`` env var if set.
2. The in-image bake path under ``/app/upstream_graders/{bird-interact,
   livesqlbench}/`` (populated by ``Dockerfile.cloud`` via BuildKit
   ``--build-context``) — wins inside the cloud actor.
3. Sibling of the main checkout via ``paths.bird_interact_root()`` /
   ``paths.livesqlbench_root()`` — common local-dev layout.

NEVER bake author-private absolute paths into the defaults: if the cloud
image silently fell back to legacy ``_set_equal`` because the upstream
tree was unreachable, the N1 cascade tier would report fake numbers.

Deliberate deviation from upstream: when BOTH preprocessed result lists
come out empty, the shim returns ``True`` (matches our legacy
``_set_equal([], [])`` behaviour). Upstream returns ``0`` in that
branch; the deviation keeps zero-row gold/pred matches as passes the way
the existing cascade analysis was scored.

For the Postgres livesqlbench variants the caller is responsible for
passing a fresh psycopg2 connection; the shim issues
``conn.rollback()`` in ``try/finally`` so mutation-bearing prediction
SQL cannot leak into the next grade on the same conn.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class ExBaseUnavailableError(Exception):
    """The upstream grader could not be loaded or the benchmark is not
    in the ex_base-backed N1 supported set. Callers (the N1 dispatch in
    ``tolerant_grader``) catch this and fall back to the legacy
    ``_set_equal`` path so a missing upstream tree never crashes
    grading."""


# ---------------------------------------------------------------------------
# is_mutation_sql — regex on SQL keywords that imply DB state change.
# ---------------------------------------------------------------------------

_MUTATION_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
    "ALTER", "TRUNCATE", "REPLACE",
)

# Match a mutation keyword AT statement start: either after ``;``
# (with optional whitespace) or at the very beginning of the SQL.
# Keeps ``SELECT REPLACE(...)`` (function call inside a SELECT) from
# being misclassified as a mutation while still catching real
# ``INSERT INTO`` / ``UPDATE`` / multi-statement ``...; DELETE ...``.
_MUTATION_AT_STMT_START_RE = re.compile(
    r"(?:^|;)\s*(?:" + "|".join(_MUTATION_KEYWORDS) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# CTE-prefixed mutations: ``WITH x AS (...) DELETE FROM t WHERE ...``
# (SQLite + Postgres both accept this). The statement-start regex above
# only sees ``WITH``, so we also scan for a verb-target pair anywhere in
# the cleaned SQL. The verb-target shape is tight enough that ordinary
# SELECT clauses don't trip it (``SELECT INSERT_NUM FROM t`` doesn't
# contain ``INSERT INTO``). (Codex round 5.)
_MUTATION_VERB_TARGET_RE = re.compile(
    r"\b(?:"
    r"INSERT\s+INTO|"
    r"REPLACE\s+INTO|"
    r"UPDATE\s+[A-Za-z_][\w.]*\s+SET|"
    r"DELETE\s+FROM|"
    r"CREATE\s+(?:TEMP\s+|TEMPORARY\s+|OR\s+REPLACE\s+)?"
    r"(?:TABLE|VIEW|INDEX|TRIGGER|SCHEMA|DATABASE)|"
    r"DROP\s+(?:TABLE|VIEW|INDEX|TRIGGER|SCHEMA|DATABASE)|"
    r"ALTER\s+(?:TABLE|VIEW|INDEX|SCHEMA)|"
    r"TRUNCATE(?:\s+TABLE)?\s+[A-Za-z_]"
    r")\b",
    re.IGNORECASE,
)

# SQL comment forms upstream's `remove_comments` strips before exec:
# ``-- ... <EOL>`` (single-line) and ``/* ... */`` (multi-line, non-greedy
# across newlines). Strip both before the mutation regex match so a
# commented mutation (``-- explanation\nINSERT INTO ...``) is still
# caught (Codex round 3). Upstream's exec path also drops the comments,
# so without this strip the dispatcher would think the SQL is read-only
# but upstream's cleaned-up SQL would still execute the mutation.
_SQL_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_sql_comments(sql: str) -> str:
    no_block = _SQL_BLOCK_COMMENT_RE.sub("", sql)
    return _SQL_LINE_COMMENT_RE.sub("", no_block)


def is_mutation_sql(sql: str) -> bool:
    """Return True iff ``sql`` carries an SQL mutation.

    Detects two shapes:

    1. ``MUTATION_KEYWORD`` as the LEADING token of any statement
       (statements split on ``;``). A SELECT calling ``REPLACE(...)``
       as a function is NOT a mutation — only ``REPLACE INTO``
       (statement-leading) is.
    2. A verb-target pair (``DELETE FROM``, ``INSERT INTO``,
       ``UPDATE <ident> SET``, ``CREATE TABLE/VIEW/...``, etc.) anywhere
       in the cleaned SQL — catches CTE-prefixed mutations like
       ``WITH x AS (...) DELETE FROM t WHERE id IN (SELECT id FROM x)``
       that the statement-start regex misses because ``WITH`` itself
       isn't a mutation keyword (Codex round 5).

    Comments (``-- ...`` / ``/* ... */``) are stripped before matching
    so ``-- explanation\\nINSERT INTO ...`` is still recognised as a
    mutation. Mirrors upstream's ``remove_comments`` cleanup so the
    dispatcher and the exec path agree on what counts as a mutation.
    """
    if not sql:
        return False
    cleaned = _strip_sql_comments(sql)
    if _MUTATION_AT_STMT_START_RE.search(cleaned):
        return True
    return bool(_MUTATION_VERB_TARGET_RE.search(cleaned))


# ---------------------------------------------------------------------------
# Lazy upstream module loaders
# ---------------------------------------------------------------------------

# In-image bake locations: Dockerfile.cloud's `COPY --from=` lines
# (matched 1:1 by `cloud.image.build_and_push`'s BuildKit
# `--build-context` flags) populate these dirs at image build time. The
# host-tree directory structure is preserved (so `_MINI_INTERACT_REL` and
# `_LIVESQLBENCH_REL` resolve identically on cloud and on the developer
# laptop). Tests pin these against author-private paths via
# `tests/eval/test_upstream_ex_base.py::test_default_roots_*`.
_CLOUD_BIRD_INTERACT_ROOT = Path("/app/upstream_graders/bird-interact")
_CLOUD_LIVESQLBENCH_ROOT = Path("/app/upstream_graders/livesqlbench")

_MINI_INTERACT_REL = (
    "mini_interact/knowledge_based/mini_interact_conv/evaluation/test_utils.py"
)
_LIVESQLBENCH_REL = "evaluation/src/test_utils.py"


def _resolve_upstream_root(
    env_var: str,
    cloud_path: Path,
    sibling_name: str,
    *,
    marker_rel: str,
) -> Path:
    """Resolve the host root for one upstream tree.

    Order: env override → in-image bake path (cloud actor) → sibling of
    main checkout (local dev convention). The sibling-discovery branch
    imports :mod:`bird_interact_agents.paths` lazily so a missing
    upstream tree doesn't bring the package's path machinery into modules
    that don't need it.

    ``marker_rel`` is the path (relative to the root) of the marker file
    the upstream loader actually imports — used to validate the cloud
    bake. A partial bake that leaves the cloud root dir present but
    drops the deeper grader file would otherwise short-circuit the
    sibling fallback and silently downgrade cascade tier N1 to legacy
    ``_set_equal`` (Codex round 7). Verifying the marker at the rel
    path is the cheapest way to detect an incomplete bake — if the
    marker isn't there, fall through to the sibling discovery.
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    if (cloud_path / marker_rel).is_file():
        return cloud_path
    from bird_interact_agents import paths
    return (
        paths.bird_interact_upstream_root()
        if sibling_name == "BIRD-Interact"
        else paths.livesqlbench_upstream_root()
    )


# Sibling-helper module names the upstream files import via the bare
# (un-namespaced) name. Each upstream tree ships its own ``db_utils`` —
# mini-interact's wraps sqlite3, livesqlbench's wraps psycopg2 — so a
# shared ``sys.modules["db_utils"]`` cached from the first load would
# leak the wrong DB driver into whichever tree is loaded second.
_UPSTREAM_SIBLING_MODULES = ("db_utils",)


def _load_module_from_file(
    name: str, path: Path, *, sys_path_addition: Optional[Path] = None,
) -> ModuleType:
    """Load a Python file as a module by path. Optionally prepend
    ``sys_path_addition`` so the module can import its sibling
    ``db_utils.py`` (upstream files use ``from db_utils import ...``).

    Both upstream trees define ``db_utils`` with the same bare name —
    once Python caches the first tree's ``db_utils`` in ``sys.modules``,
    subsequent loads reuse it instead of re-importing from the second
    tree's path. We snapshot + temporarily evict the sibling cache
    entries during ``exec_module`` so each load re-resolves them
    relative to ``sys_path_addition``, then restore the snapshot. The
    upstream module itself keeps its own bound references to the right
    helpers in its module globals after exec — the cache restore only
    affects future bare imports.
    """
    if not path.is_file():
        raise FileNotFoundError(f"upstream module not found at {path}")
    if sys_path_addition is not None:
        # ALWAYS re-front the per-load directory, even if it's already
        # somewhere in sys.path. Otherwise the second tree we load
        # leaves its own dir at the front, and a subsequent reload of
        # the first tree still walks sys.path in [new_first_dir, ...,
        # mini_dir, ...] order — the bare `from db_utils import ...`
        # would then bind to the WRONG tree's sibling because the
        # sys.path search hits the second tree first. (Codex round 2.)
        path_str = str(sys_path_addition)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Snapshot + evict the sibling cache so the upcoming exec re-imports
    # them from the right tree (resolved via ``sys_path_addition`` we
    # just prepended). Restore unconditionally on the way out, so a
    # later loader against the OTHER tree starts from a clean cache too.
    sibling_snapshot: dict[str, Optional[ModuleType]] = {}
    for sibling in _UPSTREAM_SIBLING_MODULES:
        sibling_snapshot[sibling] = sys.modules.get(sibling)
        sys.modules.pop(sibling, None)
    try:
        spec.loader.exec_module(mod)
    finally:
        for sibling, prior in sibling_snapshot.items():
            if prior is not None:
                sys.modules[sibling] = prior
            else:
                sys.modules.pop(sibling, None)
    return mod


def _load_mini_interact_module() -> ModuleType:
    """Load the mini-interact upstream comparator module on demand."""
    root = _resolve_upstream_root(
        "BIRD_BIRD_INTERACT_ROOT", _CLOUD_BIRD_INTERACT_ROOT, "BIRD-Interact",
        marker_rel=_MINI_INTERACT_REL,
    )
    target = root / _MINI_INTERACT_REL
    return _load_module_from_file(
        "_bird_interact_mini_interact_test_utils",
        target,
        sys_path_addition=target.parent,
    )


def _load_livesqlbench_module() -> ModuleType:
    """Load the livesqlbench upstream comparator module on demand."""
    root = _resolve_upstream_root(
        "BIRD_LIVESQLBENCH_ROOT", _CLOUD_LIVESQLBENCH_ROOT, "livesqlbench",
        marker_rel=_LIVESQLBENCH_REL,
    )
    target = root / _LIVESQLBENCH_REL
    return _load_module_from_file(
        "_bird_interact_livesqlbench_test_utils",
        target,
        sys_path_addition=target.parent,
    )


# ---------------------------------------------------------------------------
# Benchmark dispatch
# ---------------------------------------------------------------------------

_MINI_INTERACT_BENCHMARKS = frozenset({"mini-interact"})
_LIVESQLBENCH_BENCHMARKS = frozenset({
    # The lite-sqlite variant uses upstream livesqlbench's comparator —
    # same algorithm, different driver. (We grade against our own local
    # SQLite copies, but the comparator code is in the livesqlbench tree.)
    "livesqlbench-base-lite-sqlite",
    "livesqlbench-base-lite",
    "livesqlbench-base-full",
    "livesqlbench-large",
})


def _benchmark_loader(benchmark: str):
    if benchmark in _MINI_INTERACT_BENCHMARKS:
        return _load_mini_interact_module, "sqlite"
    if benchmark in _LIVESQLBENCH_BENCHMARKS:
        return _load_livesqlbench_module, (
            "sqlite" if benchmark.endswith("-sqlite") else "postgres"
        )
    raise ExBaseUnavailableError(
        f"benchmark {benchmark!r} is not in the ex_base-backed N1 supported "
        f"set; caller must fall back to legacy _set_equal"
    )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def compare_pred_vs_gold_ex_base(
    *,
    benchmark: str,
    pred_sqls: Sequence[str],
    sol_sqls: Sequence[str],
    db_name: str,
    conn,
    conditions: Optional[dict] = None,
) -> bool:
    """Grade predicted SQL against gold via the upstream BIRD-Interact
    ``test_case_default`` pipeline.

    On any upstream-load failure (missing tree, ImportError,
    FileNotFoundError, ...) we re-raise as :class:`ExBaseUnavailableError`
    so the caller can fall back to legacy ``_set_equal``.

    For livesqlbench Postgres conns we ``conn.rollback()`` in
    ``try/finally`` to keep pred mutations from leaking into the next
    grade on the same connection.
    """
    try:
        loader, driver = _benchmark_loader(benchmark)
    except ExBaseUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExBaseUnavailableError(
            f"benchmark dispatch failed for {benchmark!r}: {exc}"
        ) from exc

    try:
        upstream = loader()
    except ExBaseUnavailableError:
        raise
    except (ImportError, FileNotFoundError) as exc:
        raise ExBaseUnavailableError(
            f"upstream comparator unavailable for {benchmark!r}: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ExBaseUnavailableError(
            f"loading upstream comparator for {benchmark!r} raised: {exc}"
        ) from exc

    # Apply upstream cleanup (matches `test_case_default`, NOT raw
    # `ex_base` — `ex_base` itself does not strip).
    cleaned_pred = upstream.remove_round(
        upstream.remove_distinct(
            upstream.remove_comments(list(pred_sqls))
        )
    )
    cleaned_sol = upstream.remove_round(
        upstream.remove_distinct(
            upstream.remove_comments(list(sol_sqls))
        )
    )

    needs_rollback = (
        benchmark in _LIVESQLBENCH_BENCHMARKS and driver == "postgres"
    )
    # Codex round 4 #1: when the caller passes ``conn=None``, upstream's
    # ``execute_queries`` opens a connection per call (sqlite3.connect /
    # pool.getconn) but never closes it — every N1 comparison leaks at
    # least one conn. Open one ourselves and close it in the finally.
    # This makes BOTH preprocessed-result executions reuse the same
    # conn AND the rollback/close path own the lifecycle.
    owned_conn = None
    if conn is None:
        try:
            if driver == "sqlite":
                import sqlite3 as _sqlite3
                conn = _sqlite3.connect(db_name, timeout=30)
                owned_conn = conn
            elif driver == "postgres":
                # Local import to avoid a hard dep on bird_interact_agents'
                # postgres helper when only mini-interact is in use.
                from bird_interact_agents.db_connection import (
                    _open_psycopg2_connection,
                )
                import os as _os
                host = _os.environ.get("BIRD_PG_HOST", "localhost")
                port = int(_os.environ.get("BIRD_PG_PORT", "5432"))
                user = _os.environ.get("BIRD_PG_USER", "bird_interact")
                password = _os.environ.get("BIRD_PG_PASSWORD", "bird_interact")
                stmt_timeout = int(
                    _os.environ.get("BIRD_PG_STATEMENT_TIMEOUT", "30000")
                )
                # `db_name` is the DB short name; upstream livesqlbench's
                # `perform_query_on_postgresql_databases` will issue
                # queries through this raw psycopg2 conn.
                conn = _open_psycopg2_connection(
                    db_name, host, port, user, password, stmt_timeout,
                )
                owned_conn = conn
        except Exception as exc:  # noqa: BLE001
            raise ExBaseUnavailableError(
                f"could not open conn for {benchmark!r} on {db_name!r}: {exc}"
            ) from exc

    # Upstream's SQLite `perform_query_on_sqlite_databases` runs
    # `PRAGMA synchronous = OFF` / `journal_mode = WAL` on the conn,
    # which SQLite rejects when a transaction is open ("Safety level
    # may not be changed inside a transaction"). Pre-commit any pending
    # tx so the upstream PRAGMAs land cleanly. No-op for psycopg2 since
    # the rollback in the finally block reclaims state either way.
    if driver == "sqlite":
        try:
            conn.commit()
        except Exception:  # noqa: BLE001
            # Conn may not support .commit() (e.g. some custom DB-API
            # impl); pressing on is fine, the upstream call will surface
            # the original failure if any.
            pass
    try:
        try:
            try:
                result = upstream.ex_base(
                    cleaned_pred, cleaned_sol, db_name, conn, conditions,
                )
            except Exception:
                # Re-raise after the rollback finally below; the outer
                # try/finally closes ``owned_conn`` either way.
                raise
        finally:
            if needs_rollback:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "[upstream_ex_base] conn.rollback() failed after "
                        "compare_pred_vs_gold_ex_base — next caller may "
                        "see leaked state",
                        exc_info=True,
                    )

        # Codex round-1 finding #3: legacy deviation from upstream —
        # both empty preprocessed results = pass.
        if result == 0 and _both_results_empty(
            upstream, cleaned_pred, cleaned_sol, db_name, conn,
        ):
            return True
        return bool(result == 1)
    finally:
        # Codex round 4 #1: close any conn we opened ourselves. The
        # outer finally fires on both the success and exception paths,
        # AFTER ``_both_results_empty`` reuses the same conn (which is
        # why the empty-check runs inside the same try block).
        if owned_conn is not None:
            try:
                owned_conn.close()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "[upstream_ex_base] owned-conn close failed",
                    exc_info=True,
                )


def _both_results_empty(
    upstream: ModuleType,
    pred_sqls: Sequence[str],
    sol_sqls: Sequence[str],
    db_name: str,
    conn,
) -> bool:
    """Best-effort check that BOTH sides produced an empty preprocessed
    result list (the only case where we deviate from upstream).

    ``ex_base`` returns ``0`` for any non-match. We need to disambiguate
    "both empty → kept as pass" from "real mismatch → fail". Re-run the
    two pred/gold queries through upstream's execute + preprocess_results
    and check explicitly.
    """
    try:
        execute_queries = upstream.execute_queries
        preprocess_results = upstream.preprocess_results
    except AttributeError:
        return False
    try:
        # The mini-interact and livesqlbench `execute_queries` signatures
        # accept ``(sqls, db, conn, ...)``; mini-interact has the SQLite
        # variant ``(sqls, db_path, conn)`` while livesqlbench has
        # ``(sqls, db_name, conn, None, "")``. Probe both shapes.
        try:
            pred_rows, p_err, p_to = execute_queries(
                pred_sqls, db_name, conn, None, "",
            )
        except TypeError:
            pred_rows, p_err, p_to = execute_queries(pred_sqls, db_name, conn)
        try:
            gold_rows, g_err, g_to = execute_queries(
                sol_sqls, db_name, conn, None, "",
            )
        except TypeError:
            gold_rows, g_err, g_to = execute_queries(sol_sqls, db_name, conn)
    except Exception:  # noqa: BLE001
        return False
    if any([p_err, p_to, g_err, g_to]):
        return False
    return (
        not preprocess_results(pred_rows or [])
        and not preprocess_results(gold_rows or [])
    )
