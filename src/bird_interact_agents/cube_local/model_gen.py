"""DEV-1822: deterministic Cube model generation from an introspected Postgres
schema + the BIRD `<db>_column_meaning_base.json` meanings.

One cube per table; every non-json column a typed dimension
(numeric/string/time/boolean). json/jsonb columns contribute ONLY their
documented leaf dimensions (a raw `::text` blob dim is invalid to Cube and
near-useless), built via the reused `slayer_pipeline.jsonb` null-safe extract
(C5); `sum/avg/min/max` for every number dimension incl. JSON-derived; FK → join
(first FK per target wins, self-FK skipped). Identifiers are sanitized into
valid Cube members with collision suffixes; SQL always quotes the originals.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml
from pydantic import BaseModel, Field

from slayer.core.models import DataType

from bird_interact_agents import paths
from bird_interact_agents.slayer_pipeline.casts import quote_ident
from bird_interact_agents.slayer_pipeline.jsonb import (
    jsonb_meaning_entries,
    walk_fields_meaning,
)

# Bump to force regeneration of every cached model when the generator changes.
MODEL_GEN_VERSION = 2


# --- introspected schema (inputs) ------------------------------------------

class FKRef(BaseModel):
    table: str
    column: str
    schema_name: str = "public"


class ColumnInfo(BaseModel):
    name: str
    pg_type: str
    is_pk: bool = False
    fk: Optional[FKRef] = None


class TableInfo(BaseModel):
    table_name: str
    columns: list[ColumnInfo]
    schema_name: str = "public"


# --- generated cube model (outputs) ----------------------------------------

class DimDef(BaseModel):
    name: str
    sql: str
    type: str
    primary_key: bool = False
    description: Optional[str] = None


class MeasureDef(BaseModel):
    name: str
    type: str
    sql: Optional[str] = None


class JoinDef(BaseModel):
    name: str
    sql: str
    relationship: str


class CubeDef(BaseModel):
    name: str
    sql_table: str
    dimensions: list[DimDef]
    measures: list[MeasureDef]
    joins: list[JoinDef] = Field(default_factory=list)


# --- type mapping -----------------------------------------------------------

_NUMBER_PG = {
    "smallint", "integer", "bigint", "int", "int2", "int4", "int8",
    "numeric", "decimal", "real", "double precision", "float", "float4",
    "float8", "money", "smallserial", "serial", "bigserial",
}
_STRING_PG = {
    "text", "varchar", "character varying", "char", "character", "bpchar",
    "name", "uuid", "citext",
}
_TIME_PG = {
    "date", "timestamp", "timestamp without time zone",
    "timestamp with time zone", "timestamptz", "time",
    "time without time zone", "time with time zone", "timetz",
}
_BOOL_PG = {"boolean", "bool"}
# json/jsonb columns are NOT emitted as a raw dimension (a `::text` blob dim is
# both invalid to Cube and near-useless); only their documented leaves are.
_JSON_PG = {"json", "jsonb"}

_DATATYPE_TO_CUBE = {
    DataType.TEXT: "string",
    DataType.INT: "number",
    DataType.DOUBLE: "number",
    DataType.BOOLEAN: "boolean",
    DataType.DATE: "time",
    DataType.TIMESTAMP: "time",
}


def _resolve_pg_type(pg_type: str) -> tuple[str, bool]:
    """Return ``(cube_type, needs_text_cast)`` for a Postgres type string."""
    t = pg_type.strip().lower()
    if t in _NUMBER_PG:
        return "number", False
    if t in _STRING_PG:
        return "string", False
    if t in _TIME_PG:
        return "time", False
    if t in _BOOL_PG:
        return "boolean", False
    return "string", True  # jsonb/json/arrays/enums/unknown → string via ::text


def _col_type_sql(col: ColumnInfo) -> tuple[str, str]:
    cube_type, needs_cast = _resolve_pg_type(col.pg_type)
    ident = quote_ident(col.name)
    if needs_cast:
        return "string", f"({ident})::text"
    return cube_type, ident


# --- identifier sanitizing --------------------------------------------------

def sanitize_member_name(raw: str, taken: set[str]) -> str:
    """Turn an arbitrary identifier into a valid, unique Cube member name."""
    s = re.sub(r"[^0-9a-zA-Z_]", "_", raw).lower()
    if not s or s[0].isdigit():
        s = "_" + s
    if s not in taken:
        return s
    i = 2
    while f"{s}_{i}" in taken:
        i += 1
    return f"{s}_{i}"


def _sql_table(t: TableInfo) -> str:
    return f"{quote_ident(t.schema_name)}.{quote_ident(t.table_name)}"


def _compute_cube_names(tables: Sequence[TableInfo]) -> dict[tuple[str, str], str]:
    taken: set[str] = set()
    names: dict[tuple[str, str], str] = {}
    for t in tables:
        raw = t.table_name if t.schema_name == "public" else f"{t.schema_name}__{t.table_name}"
        name = sanitize_member_name(raw, taken)
        taken.add(name)
        names[(t.schema_name, t.table_name.lower())] = name
    return names


# --- meanings ---------------------------------------------------------------

def _plain_desc_map(meanings: dict) -> dict[tuple[str, str], str]:
    """`(table_lower, col_lower) -> description` for plain (non-jsonb) columns."""
    out: dict[tuple[str, str], str] = {}
    for key, value in meanings.items():
        parts = key.split("|")
        if len(parts) != 3 or not isinstance(value, str):
            continue
        out[(parts[1].lower(), parts[2].lower())] = value
    return out


# --- build ------------------------------------------------------------------

def build_cube_defs(
    tables: Sequence[TableInfo], meanings: dict, *,
    leaf_sampler: Optional[Callable[[str, str], list[str]]] = None,
) -> list[CubeDef]:
    """Build the deterministic list of :class:`CubeDef` for *tables*."""
    cube_names = _compute_cube_names(tables)
    desc_map = _plain_desc_map(meanings)
    jsonb_map = {(t, c): entry for t, c, entry in jsonb_meaning_entries(meanings)}

    defs: list[CubeDef] = []
    for t in tables:
        cube_name = cube_names[(t.schema_name, t.table_name.lower())]
        taken: set[str] = {"count"}
        dims: list[DimDef] = []
        measures: list[MeasureDef] = [MeasureDef(name="count", type="count")]
        number_dims: list[tuple[str, str]] = []
        joins: list[JoinDef] = []
        join_targets: set[str] = set()

        for col in t.columns:
            # json/jsonb columns contribute only their documented leaf dims
            # (below), never a raw `::text` blob dimension.
            if col.pg_type.strip().lower() not in _JSON_PG:
                cube_type, sql = _col_type_sql(col)
                dim_name = sanitize_member_name(col.name, taken)
                taken.add(dim_name)
                dims.append(DimDef(
                    name=dim_name, sql=sql, type=cube_type, primary_key=col.is_pk,
                    description=desc_map.get((t.table_name.lower(), col.name.lower())),
                ))
                if cube_type == "number":
                    number_dims.append((dim_name, sql))
            if col.fk is not None:
                tgt = (col.fk.schema_name, col.fk.table.lower())
                if col.fk.table.lower() == t.table_name.lower():
                    continue  # self-FK: skip
                if tgt[1] in join_targets or tgt not in cube_names:
                    continue  # already joined this target, or target not modeled
                join_targets.add(tgt[1])
                jname = cube_names[tgt]
                joins.append(JoinDef(
                    name=jname, relationship="many_to_one",
                    sql=f"{{CUBE}}.{quote_ident(col.name)} = "
                        f"{{{jname}}}.{quote_ident(col.fk.column)}",
                ))

        # documented JSON leaves → extra dimensions (reused jsonb helper = C5)
        for col in t.columns:
            entry = jsonb_map.get((t.table_name.lower(), col.name.lower()))
            if not entry:
                continue
            for leaf in walk_fields_meaning(
                col.name, entry.get("fields_meaning") or {}, backend="postgres",
                table=t.table_name, pg_extract_sampler=leaf_sampler,
            ):
                dim_name = sanitize_member_name(leaf.column_name, taken)
                taken.add(dim_name)
                cube_type = _DATATYPE_TO_CUBE.get(leaf.type, "string")
                desc = leaf.description + (f" [{leaf.warning}]" if leaf.warning else "")
                dims.append(DimDef(name=dim_name, sql=leaf.sql, type=cube_type,
                                   description=desc))
                if cube_type == "number":
                    number_dims.append((dim_name, leaf.sql))

        # sum/avg/min/max for every number dimension (incl. JSON-derived)
        for dim_name, sql in number_dims:
            for agg in ("sum", "avg", "min", "max"):
                mname = sanitize_member_name(f"{dim_name}_{agg}", taken)
                taken.add(mname)
                measures.append(MeasureDef(name=mname, type=agg, sql=sql))

        defs.append(CubeDef(name=cube_name, sql_table=_sql_table(t),
                            dimensions=dims, measures=measures, joins=joins))
    return defs


# --- fingerprint (C7) -------------------------------------------------------

def model_fingerprint(tables: Sequence[TableInfo], meanings: dict) -> str:
    payload = {
        "version": MODEL_GEN_VERSION,
        "tables": [t.model_dump() for t in tables],
        "meanings": meanings,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# --- YAML rendering ---------------------------------------------------------

def _dim_to_dict(d: DimDef) -> dict:
    out: dict = {"name": d.name, "sql": d.sql, "type": d.type}
    if d.primary_key:
        out["primary_key"] = True
        out["public"] = True
    if d.description:
        out["description"] = d.description
    return out


def _cube_to_dict(c: CubeDef) -> dict:
    out: dict = {"name": c.name, "sql_table": c.sql_table}
    if c.joins:
        out["joins"] = [{"name": j.name, "sql": j.sql,
                         "relationship": j.relationship} for j in c.joins]
    out["dimensions"] = [_dim_to_dict(d) for d in c.dimensions]
    out["measures"] = [
        ({"name": m.name, "type": m.type, "sql": m.sql} if m.sql is not None
         else {"name": m.name, "type": m.type})
        for m in c.measures
    ]
    return out


def render_model_files(cube_defs: Sequence[CubeDef]) -> dict[str, str]:
    """Return ``{"<cube>.yml": yaml_text}`` for each cube."""
    return {
        f"{c.name}.yml": yaml.safe_dump({"cubes": [_cube_to_dict(c)]}, sort_keys=False)
        for c in cube_defs
    }


# --- introspection + orchestration (integration paths) ---------------------

def _connect(pg_env: dict, db: str):
    import psycopg2  # project ships psycopg2, not psycopg3
    return psycopg2.connect(
        host=pg_env["BIRD_PG_HOST"], port=int(pg_env.get("BIRD_PG_PORT", 5432)),
        dbname=db, user=pg_env.get("BIRD_PG_USER", "bird_interact"),
        password=pg_env.get("BIRD_PG_PASSWORD"),
    )


def introspect_schema(conn, db: str | None = None) -> list[TableInfo]:
    """Introspect base tables + columns + PK/FK from a live connection."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_schema, table_name FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY table_schema, table_name
        """)
        table_order = cur.fetchall()

        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type, ordinal_position
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY table_schema, table_name, ordinal_position
        """)
        col_rows = cur.fetchall()

        cur.execute("""
            SELECT tc.table_schema, tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name=kcu.constraint_name
             AND tc.table_schema=kcu.table_schema
            WHERE tc.constraint_type='PRIMARY KEY'
        """)
        pk_set = {(s, t, c) for s, t, c in cur.fetchall()}

        cur.execute("""
            SELECT tc.table_schema, tc.table_name, kcu.column_name,
                   ccu.table_schema, ccu.table_name, ccu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name=kcu.constraint_name
             AND tc.table_schema=kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name=tc.constraint_name
             AND ccu.table_schema=tc.table_schema
            WHERE tc.constraint_type='FOREIGN KEY'
        """)
        fk_map: dict[tuple[str, str, str], FKRef] = {}
        for s, t, c, fs, ft, fc in cur.fetchall():
            fk_map.setdefault((s, t, c), FKRef(table=ft, column=fc, schema_name=fs))

    cols_by_table: dict[tuple[str, str], list[ColumnInfo]] = {}
    for s, t, name, data_type, _ord in col_rows:
        cols_by_table.setdefault((s, t), []).append(ColumnInfo(
            name=name, pg_type=data_type,
            is_pk=(s, t, name) in pk_set, fk=fk_map.get((s, t, name)),
        ))
    return [
        TableInfo(schema_name=s, table_name=t, columns=cols_by_table.get((s, t), []))
        for s, t in table_order
    ]


def _load_meanings(benchmark: str, db: str) -> dict:
    path = paths.benchmark_data_root(benchmark) / db / f"{db}_column_meaning_base.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _models_dir(benchmark: str) -> Path:
    return paths.cube_local_root(benchmark=benchmark) / "conf" / "model"


def list_generated_dbs(benchmark: str) -> set[str]:
    """DBs with a committed model dir (`_model_fp.txt`) — the tenant allow-list."""
    root = _models_dir(benchmark)
    if not root.exists():
        return set()
    return {d.name for d in root.iterdir() if (d / "_model_fp.txt").exists()}


def ensure_models(benchmark: str, dbs: Sequence[str], pg_env: dict) -> dict[str, Path]:
    """Generate (or refresh on fingerprint drift) the per-DB Cube model dirs."""
    out: dict[str, Path] = {}
    for db in dbs:
        meanings = _load_meanings(benchmark, db)
        conn = _connect(pg_env, db)  # psycopg2: `with conn` is a txn, not close
        try:
            tables = introspect_schema(conn, db)
        finally:
            conn.close()
        fp = model_fingerprint(tables, meanings)
        db_dir = _models_dir(benchmark) / db
        out[db] = db_dir
        fp_file = db_dir / "_model_fp.txt"
        if fp_file.exists() and fp_file.read_text().strip() == fp:
            continue
        files = render_model_files(build_cube_defs(tables, meanings))
        db_dir.mkdir(parents=True, exist_ok=True)
        for existing in db_dir.glob("*.yml"):
            existing.unlink()
        for fname, text in files.items():
            (db_dir / fname).write_text(text)
        fp_file.write_text(fp)
    return out
