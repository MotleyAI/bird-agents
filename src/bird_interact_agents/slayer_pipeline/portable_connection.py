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
    connection_string: str, mini_interact_root: Path
) -> str:
    """Re-anchor a portable (relative) connection_string at the local
    mini-interact root, returning an absolute SQLite URI. ``$BIRD_DB_PATH``
    overrides the supplied root when set.

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
    env_root = os.environ.get("BIRD_DB_PATH")
    root = Path(env_root).expanduser() if env_root else mini_interact_root
    abs_path = (root / rel_path).resolve()
    return f"{_SQLITE_PREFIX_ABSOLUTE}{abs_path.as_posix().lstrip('/')}"
