"""DEV-1822 (Codex C1): turn Cube's parameterized `/v1/sql` output into a
standalone, regrade-safe Postgres statement.

Cube returns ``[text_with_$N, params]``; the graded submission must be one
executable SQL string. `$N` tokens inside single-quoted strings / double-quoted
identifiers are left untouched. Lenient on surplus params (Cube always supplies
exactly-used params); a `$N` with no param raises.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Sequence


def render_literal(value: Any) -> str:
    """Render a Python value as a safe Postgres SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):  # before int — bool is a subclass of int
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot render non-finite float literal: {value!r}")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, dt.datetime):
        return "'" + value.isoformat(sep=" ") + "'"
    if isinstance(value, dt.date):
        return "'" + value.isoformat() + "'"
    if isinstance(value, (list, tuple)):
        return "ARRAY[" + ", ".join(render_literal(v) for v in value) + "]"
    raise TypeError(f"unsupported SQL literal type {type(value).__name__}: {value!r}")


def materialize_sql(sql: str, params: Sequence[Any]) -> str:
    """Substitute node-postgres ``$N`` placeholders in *sql* with rendered
    literals from *params*, skipping quoted strings/identifiers."""
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):  # copy a quoted region verbatim (handles doubled quotes)
            out.append(ch)
            i += 1
            while i < n:
                out.append(sql[i])
                if sql[i] == ch:
                    if i + 1 < n and sql[i + 1] == ch:  # escaped quote → stay inside
                        out.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == "$" and i + 1 < n and sql[i + 1].isdigit():
            j = i + 1
            while j < n and sql[j].isdigit():
                j += 1
            idx = int(sql[i + 1:j])
            if idx < 1 or idx > len(params):
                raise ValueError(
                    f"SQL placeholder ${idx} has no corresponding parameter "
                    f"(got {len(params)} param(s))"
                )
            out.append(render_literal(params[idx - 1]))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)
