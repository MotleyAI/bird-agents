"""DEV-1822 (Codex C9): validate a final Cube query for regrade-safety, then
compile it to standalone SQL via `/v1/sql` + `materialize_sql`.

Only "regular" query shapes are accepted: anything Cube would post-process in
Node (compareDateRange, total, blending) or that carries inline member
expressions is refused BEFORE any budget/state mutation, so the generated
`/v1/sql` faithfully represents the answer.
"""

from __future__ import annotations

from typing import Any

from bird_interact_agents.cube_local.sql_render import materialize_sql

ALLOWED_QUERY_KEYS = {
    "measures", "dimensions", "filters", "timeDimensions", "segments",
    "order", "limit", "offset", "timezone", "ungrouped",
}
_STRING_MEMBER_KEYS = ("measures", "dimensions", "segments")


class CubeQueryRefused(Exception):
    """The submitted Cube query is not regrade-safe."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_cube_query(query: Any) -> None:
    """Raise :class:`CubeQueryRefused` if *query* is not a regular, regrade-safe
    single-query object."""
    if not isinstance(query, dict):
        raise CubeQueryRefused(
            "query must be a single Cube query object; data blending / "
            "multi-query arrays are not supported."
        )
    extra = set(query) - ALLOWED_QUERY_KEYS
    if extra:
        raise CubeQueryRefused(
            f"unsupported query key(s) {sorted(extra)}; allowed: "
            f"{sorted(ALLOWED_QUERY_KEYS)}. (post-processed shapes like `total` "
            "are refused.)"
        )
    for key in _STRING_MEMBER_KEYS:
        for member in query.get(key) or []:
            if not isinstance(member, str):
                raise CubeQueryRefused(
                    f"{key} entries must be member names (strings), not inline "
                    f"member expressions; got {member!r}."
                )
    for td in query.get("timeDimensions") or []:
        if not isinstance(td, dict):
            raise CubeQueryRefused("timeDimensions entries must be objects.")
        if "compareDateRange" in td:
            raise CubeQueryRefused(
                "compareDateRange is post-processed and not regrade-safe."
            )
        if not isinstance(td.get("dimension"), str):
            raise CubeQueryRefused("timeDimension.dimension must be a member name.")


def cube_query_to_sql(query: Any, client) -> str:
    """Validate *query*, fetch its generated SQL via `/v1/sql`, and materialize
    the parameters into a standalone Postgres statement."""
    validate_cube_query(query)
    text, params = client.sql(query)
    return materialize_sql(text, params)
