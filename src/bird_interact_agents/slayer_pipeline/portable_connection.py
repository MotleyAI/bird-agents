"""Portable SQLite ``connection_string`` for committed datasource YAMLs.

Live SLayer storage (``~/.local/share/slayer/``) carries absolute
SQLite paths in each datasource's ``connection_string`` —
``sqlite:////home/<user>/.../mini-interact/<db>/<db>.sqlite``. That
form is per-machine: committing it makes the YAML unusable for any
other developer / CI runner.

This module's two helpers translate between absolute and portable
forms so the committed YAML stays machine-agnostic while runtime
consumers still get an absolute path that SQLAlchemy can open.

* ``to_portable_connection_string`` — used by
  ``scripts/export_slayer_models.py`` before writing the committed
  datasource. Strips the mini-interact prefix and emits the relative
  form ``sqlite:///<db>/<db>.sqlite``.
* ``resolve_committed_connection_string`` — used by
  ``hard8_preprocessor.build_task_variant_storage`` before writing
  the variant datasource. Detects the relative form and re-anchors
  it against ``$BIRD_DB_PATH`` (or a sibling ``mini-interact/``
  next to ``slayer_models/``).

The split-slash convention follows SQLAlchemy: ``sqlite:///rel`` is
relative, ``sqlite:////abs`` is absolute.
"""

from __future__ import annotations

import os
from pathlib import Path

_SQLITE_PREFIX_ABSOLUTE = "sqlite:////"
_SQLITE_PREFIX_RELATIVE = "sqlite:///"


def to_portable_connection_string(
    connection_string: str, mini_interact_root: Path
) -> str:
    """Strip an absolute mini-interact prefix from a SQLite connection
    string, returning the path relative to ``mini_interact_root`` if
    the path is rooted there. Returns the input unchanged otherwise.
    """
    if not connection_string:
        return connection_string
    # Tolerate the malformed 5-slash form that some pipeline runs
    # emitted (``sqlite://///abs``) by normalising any run of 4+
    # slashes after ``sqlite:`` down to exactly 4.
    if connection_string.startswith("sqlite:"):
        idx = len("sqlite:")
        while idx < len(connection_string) and connection_string[idx] == "/":
            idx += 1
        slashes = idx - len("sqlite:")
        if slashes >= 4:
            connection_string = "sqlite:////" + connection_string[idx:]
    if not connection_string.startswith(_SQLITE_PREFIX_ABSOLUTE):
        return connection_string
    abs_path_str = "/" + connection_string[len(_SQLITE_PREFIX_ABSOLUTE):]
    try:
        rel = (
            Path(abs_path_str).resolve().relative_to(mini_interact_root.resolve())
        )
    except (ValueError, OSError):
        return connection_string
    return f"{_SQLITE_PREFIX_RELATIVE}{rel.as_posix()}"


def resolve_committed_connection_string(
    connection_string: str, mini_interact_root: Path,
    *, db_root: Path | None = None,
) -> str:
    """Re-anchor a portable (relative) connection_string at the local
    mini-interact root, returning an absolute SQLite URI.

    Precedence (DEV-1462 / second-round Codex review): an explicit
    ``db_root`` kwarg wins over ``$BIRD_DB_PATH``. Callers that know the
    authoritative root (e.g. the otf_encode adapter handling a
    LiveSQLBench run passes the harness's ``--db-path``) MUST be able to
    override the env, because conftest + day-to-day shells often set
    ``$BIRD_DB_PATH`` to the mini-interact root, which would otherwise
    mis-anchor a LiveSQLBench per-task variant resolve at runtime.

    Legacy precedence (no ``db_root``): ``$BIRD_DB_PATH`` wins over
    ``mini_interact_root`` — back-compat with existing callers.

    Absolute connection strings are returned unchanged (so live storage
    and migration-from-old-yamls both pass through cleanly).
    """
    if not connection_string:
        return connection_string
    if connection_string.startswith(_SQLITE_PREFIX_ABSOLUTE):
        return connection_string
    if not connection_string.startswith(_SQLITE_PREFIX_RELATIVE):
        return connection_string
    rel_path = connection_string[len(_SQLITE_PREFIX_RELATIVE):]
    if db_root is not None:
        root: Path = db_root
    else:
        env_root = os.environ.get("BIRD_DB_PATH")
        root = Path(env_root).expanduser() if env_root else mini_interact_root
    abs_path = (root / rel_path).resolve()
    return f"{_SQLITE_PREFIX_ABSOLUTE}{abs_path.as_posix().lstrip('/')}"


def expected_connection_string(
    db: str, mini_interact_root: Path, *, db_root: Path | None = None,
) -> str:
    """The absolute SQLite URL the datasource for ``db`` SHOULD carry in
    the CURRENT environment: ``<root>/<db>/<db>.sqlite`` where ``root`` is
    an explicit ``db_root`` (when given) else ``$BIRD_DB_PATH`` (when set)
    else ``mini_interact_root``.

    Precedence (DEV-1462 / second-round Codex review): an explicit
    ``db_root`` wins over ``$BIRD_DB_PATH`` — a LiveSQLBench run passes its
    ``--db-path`` here so conftest's / a day-to-day shell's
    ``$BIRD_DB_PATH=<mini-interact>`` can't mis-anchor a LiveSQLBench DB.

    Unlike :func:`resolve_committed_connection_string` — which passes an
    absolute connection_string through UNCHANGED — this forces the path
    to the current root. That is what makes a deterministic cache built
    on one machine (absolute local path baked in, e.g.
    ``sqlite:////home/<user>/.../mini-interact/<db>/<db>.sqlite``) usable
    after transport into a different filesystem layout (e.g. a cloud
    container where the DB lives at ``/data/mini-interact/<db>/...``):
    resolving alone leaves the foreign absolute path intact and SQLite
    then fails with "unable to open database file".
    """
    if db_root is not None:
        base: Path = db_root
    else:
        env_root = os.environ.get("BIRD_DB_PATH")
        base = Path(env_root).expanduser() if env_root else mini_interact_root
    abs_sqlite = (base / db / f"{db}.sqlite").resolve()
    return f"{_SQLITE_PREFIX_ABSOLUTE}{abs_sqlite.as_posix().lstrip('/')}"


def reanchor_connection_string(
    connection_string: str, db: str, mini_interact_root: Path,
    *, db_root: Path | None = None,
) -> str:
    """Re-anchor a ``connection_string`` to the current environment.

    * **Relative** form (``sqlite:///<rel>``) → resolve against the root,
      PRESERVING the path component. This is the committed/pre-encoded
      form and behaviour is unchanged from
      :func:`resolve_committed_connection_string`.
    * **Absolute** form (``sqlite:////abs`` or the malformed 5-slash
      ``sqlite://///abs``) → FORCE-rewrite to the canonical
      ``<root>/<db>/<db>.sqlite``. This is the on-the-fly cache form: a
      cache built on one machine bakes in an absolute local path that does
      NOT exist after transport (DEV-1478 cloud bug). The mini-interact
      convention places every DB at ``<root>/<db>/<db>.sqlite``, so the
      stale absolute path's host is irrelevant — we re-root it.

    Root precedence in BOTH branches: an explicit ``db_root`` wins over
    ``$BIRD_DB_PATH`` (DEV-1462 — a LiveSQLBench run threads its
    ``--db-path`` so the env can't mis-anchor it); else ``$BIRD_DB_PATH``
    wins over ``mini_interact_root``.

    A ``None`` / empty / non-sqlite connection_string is returned
    unchanged so non-sqlite datasources pass through untouched.
    """
    if not connection_string or not connection_string.startswith("sqlite:"):
        return connection_string
    # Count leading slashes after ``sqlite:`` — 3 = relative, 4+ = absolute
    # (the pipeline historically emitted a malformed 5-slash absolute form).
    idx = len("sqlite:")
    n_slashes = 0
    while idx < len(connection_string) and connection_string[idx] == "/":
        idx += 1
        n_slashes += 1
    if n_slashes >= 4:
        return expected_connection_string(
            db, mini_interact_root, db_root=db_root,
        )
    return resolve_committed_connection_string(
        connection_string, mini_interact_root, db_root=db_root,
    )
