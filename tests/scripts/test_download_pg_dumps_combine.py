"""Tests for the per-table dump combiner in ``scripts/download_pg_dumps.py``.

The large-v1 zip ships one dump file per table (CREATE TABLE + INSERTs + the
table's own FK constraints) plus partial ``*_full.sql`` aggregates. The combiner
must reassemble a single dump that loads in ONE clean pass:

* pick the SINGLE-table files (the complete set), not the biggest aggregate;
* emit ``CREATE TYPE`` before the tables that use them;
* defer ``FOREIGN KEY`` constraints to the very end;

all while correctly classifying statements despite pg_dump's ``-- Name: …``
comment prefixes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "download_pg_dumps", _SCRIPTS / "download_pg_dumps.py")
dpd = importlib.util.module_from_spec(_spec)
sys.modules["download_pg_dumps"] = dpd
_spec.loader.exec_module(dpd)


# A per-table file carries the pg_dump comment banners that fooled the first
# implementation into never deferring FKs.
_CHILD_SQL = """\
--
-- Name: child; Type: TABLE; Schema: public; Owner: root
--
CREATE TABLE public."child" (
    id integer NOT NULL,
    status public.status_enum
);
INSERT INTO public."child" VALUES (1, 'A');
--
-- Name: child pk; Type: CONSTRAINT; Schema: public; Owner: root
--
ALTER TABLE ONLY public."child"
    ADD CONSTRAINT child_pkey PRIMARY KEY (id);
--
-- Name: child fk; Type: FK CONSTRAINT; Schema: public; Owner: root
--
ALTER TABLE ONLY public."child"
    ADD CONSTRAINT child_parent_fkey FOREIGN KEY (id) REFERENCES public."parent"(id);
"""

_PARENT_SQL = """\
--
-- Name: status_enum; Type: TYPE; Schema: public; Owner: root
--
CREATE TYPE public.status_enum AS ENUM ('A', 'B');
--
-- Name: parent; Type: TABLE; Schema: public; Owner: root
--
CREATE TABLE public."parent" (
    id integer NOT NULL
);
INSERT INTO public."parent" VALUES (1);
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_classifiers_ignore_comment_prefix():
    fk_stmt = ('--\n-- Name: child fk; Type: FK CONSTRAINT\n--\n'
               'ALTER TABLE ONLY public."child"\n'
               '    ADD CONSTRAINT c FOREIGN KEY (id) REFERENCES public."parent"(id);')
    type_stmt = ('-- Name: e; Type: TYPE\n'
                 "CREATE TYPE public.e AS ENUM ('A');")
    assert dpd._is_fk_statement(fk_stmt)
    assert dpd._is_type_statement(type_stmt)
    assert not dpd._is_fk_statement(type_stmt)
    # A primary-key ALTER is not a foreign key.
    pk = 'ALTER TABLE ONLY public."child"\n    ADD CONSTRAINT p PRIMARY KEY (id);'
    assert not dpd._is_fk_statement(pk)


def test_combine_orders_types_first_fks_last(tmp_path):
    # Deliberately name the child file first so sort order puts the FK-bearing
    # file BEFORE its parent/type — the combiner must still fix the ordering.
    files = [_write(tmp_path, "a_child.sql", _CHILD_SQL),
             _write(tmp_path, "z_parent.sql", _PARENT_SQL)]
    out = dpd.combine_per_table_dumps(files)

    i_type = out.index("CREATE TYPE public.status_enum")
    i_child_tbl = out.index('CREATE TABLE public."child"')
    i_parent_tbl = out.index('CREATE TABLE public."parent"')
    i_fk = out.index("child_parent_fkey")

    # Type before every table; FK after every table.
    assert i_type < i_child_tbl
    assert i_type < i_parent_tbl
    assert i_fk > i_child_tbl
    assert i_fk > i_parent_tbl
    # Extensions are prepended.
    assert out.index("CREATE EXTENSION IF NOT EXISTS hstore") < i_type


def test_combine_prefers_single_table_files_over_aggregate(tmp_path):
    # An aggregate with 2 tables should be ignored when single-table files exist.
    agg = _write(tmp_path, "everything_full.sql",
                 'CREATE TABLE public."child" (id int);\n'
                 'CREATE TABLE public."parent" (id int);\n')
    files = [_write(tmp_path, "child.sql", _CHILD_SQL),
             _write(tmp_path, "parent.sql", _PARENT_SQL),
             agg]
    out = dpd.combine_per_table_dumps(files)
    # Exactly the two single-table CREATE TABLEs (not the aggregate's copies):
    assert out.count('CREATE TABLE public."child"') == 1
    assert out.count('CREATE TABLE public."parent"') == 1
    # The INSERT data from the per-table files made it in.
    assert 'INSERT INTO public."parent"' in out


def test_iter_statements_handles_copy_and_drops_meta(tmp_path):
    text = (
        "\\restrict TOKEN\n"
        "SET client_encoding = 'UTF8';\n"
        "COPY public.t (a) FROM stdin;\n"
        "1\n"
        "2\n"
        "\\.\n"
        "INSERT INTO public.t VALUES (3);\n"
        "\\unrestrict TOKEN\n"
    )
    stmts = [s for s in dpd._iter_statements(text) if s.strip()]
    # \restrict / \unrestrict dropped; COPY block kept as ONE statement.
    assert not any("restrict" in s for s in stmts)
    copy_stmts = [s for s in stmts if s.startswith("COPY ")]
    assert len(copy_stmts) == 1
    assert "1\n2\n\\." in copy_stmts[0]
    assert any(s.startswith("INSERT") for s in stmts)
