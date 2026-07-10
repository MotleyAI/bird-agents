"""Tests for the HARD-8 preprocessor.

Covers extraction of deleted KB ids from a task record and the
``build_task_variant_storage`` happy paths + edge cases against a
real ``YAMLStorage`` round-trip.
"""

from pathlib import Path

import pytest

from slayer.core.models import (
    Aggregation,
    AggregationParam,
    Column,
    DatasourceConfig,
    ModelMeasure,
    SlayerModel,
)
from slayer.storage.yaml_storage import YAMLStorage

from bird_interact_agents.hard8_preprocessor import (
    build_task_variant_storage,
    extract_deleted_kb_ids,
)
from bird_interact_agents.memory_store_io import (
    read_memories,
    write_memories_files,
)


# ---------------------------------------------------------------------------
# extract_deleted_kb_ids
# ---------------------------------------------------------------------------

def test_extract_deleted_kb_ids_int_form():
    task = {
        "knowledge_ambiguity": [
            {"term": "x", "deleted_knowledge": 7},
            {"term": "y", "deleted_knowledge": 12},
        ],
    }
    assert extract_deleted_kb_ids(task) == {7, 12}


def test_extract_deleted_kb_ids_list_form():
    task = {
        "knowledge_ambiguity": [
            {"term": "x", "deleted_knowledge": [3, 4]},
            {"term": "y", "deleted_knowledge": 9},
        ],
    }
    assert extract_deleted_kb_ids(task) == {3, 4, 9}


def test_extract_deleted_kb_ids_missing_or_empty():
    assert extract_deleted_kb_ids({}) == set()
    assert extract_deleted_kb_ids({"knowledge_ambiguity": []}) == set()
    assert extract_deleted_kb_ids({"knowledge_ambiguity": None}) == set()
    assert extract_deleted_kb_ids({
        "knowledge_ambiguity": [{"term": "x", "deleted_knowledge": None}],
    }) == set()


# ---------------------------------------------------------------------------
# build_task_variant_storage — fixture
# ---------------------------------------------------------------------------

DB_NAME = "tinydb"


async def _seed_canonical(canonical_root: Path) -> None:
    """Write a 3-model canonical YAMLStorage with mixed meta.kb_id coverage.

    - ``alpha``: model-level ``meta.kb_id=1`` → deletion of 1 drops the model.
    - ``beta``: column with ``kb_id=2``, measure with ``kb_id=3``,
      aggregation with ``kb_id=4``, plus an unmarked column.
    - ``gamma``: ``meta.kb_ids=[5,6]`` on the model → deletion of either drops it.
    """
    storage = YAMLStorage(base_dir=str(canonical_root / DB_NAME))
    await storage.save_datasource(
        DatasourceConfig(
            name=DB_NAME,
            type="sqlite",
            connection_string=f"sqlite:///{DB_NAME}.db",
        )
    )
    await storage.save_model(
        SlayerModel(
            name="alpha",
            data_source=DB_NAME,
            sql_table="alpha",
            columns=[Column(name="id", primary_key=True)],
            meta={"kb_id": 1},
        )
    )
    await storage.save_model(
        SlayerModel(
            name="beta",
            data_source=DB_NAME,
            sql_table="beta",
            columns=[
                Column(name="id", primary_key=True),
                Column(name="amount", meta={"kb_id": 2}),
                Column(name="plain"),
            ],
            measures=[
                ModelMeasure(name="total", formula="amount:sum", meta={"kb_id": 3}),
                ModelMeasure(name="rows", formula="*:count"),
            ],
            aggregations=[
                Aggregation(
                    name="weighted_amt",
                    formula="SUM({sql} * {weight}) / NULLIF(SUM({weight}), 0)",
                    params=[AggregationParam(name="weight", sql="amount")],
                    meta={"kb_id": 4},
                ),
            ],
        )
    )
    await storage.save_model(
        SlayerModel(
            name="gamma",
            data_source=DB_NAME,
            sql_table="gamma",
            columns=[Column(name="id", primary_key=True)],
            meta={"kb_ids": [5, 6]},
        )
    )


@pytest.fixture
async def canonical_root(tmp_path: Path) -> Path:
    root = tmp_path / "canonical"
    root.mkdir()
    await _seed_canonical(root)
    return root


# ---------------------------------------------------------------------------
# build_task_variant_storage — behaviors
# ---------------------------------------------------------------------------

async def test_two_tasks_get_isolated_copies(
    canonical_root: Path, tmp_path: Path
):
    """Concurrent tasks on the same DB must not share storage. Each task's
    `work_dir` produces an independent on-disk copy; mutating one (via the
    SLayer storage API) leaves the other untouched."""
    work_a = tmp_path / "task_a"
    work_b = tmp_path / "task_b"
    work_a.mkdir()
    work_b.mkdir()

    out_a = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=work_a,
    )
    out_b = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=work_b,
    )
    assert out_a != out_b
    assert out_a.exists() and out_b.exists()

    # Mutate task A's copy: drop a model. Task B's copy + the canonical
    # reference must be unaffected.
    storage_a = YAMLStorage(base_dir=str(out_a))
    await storage_a.delete_model("alpha")

    a_after = sorted(await storage_a.list_models())
    b_after = sorted(await YAMLStorage(base_dir=str(out_b)).list_models())
    canonical_after = sorted(
        await YAMLStorage(base_dir=str(canonical_root / DB_NAME)).list_models()
    )
    assert "alpha" not in a_after
    assert "alpha" in b_after
    assert "alpha" in canonical_after


async def test_no_deletions_still_materialises_per_task_copy(
    canonical_root: Path, tmp_path: Path
):
    """Even with empty `deleted_kb_ids`, the function must materialise a
    fresh per-task copy under `work_dir`. This isolation lets each task's
    SLayer MCP server safely write back type-refinement metadata without
    mutating the committed `slayer_models/` reference, and confines any
    agent `create_model` / `edit_model` calls to the scratch dir."""
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=work,
    )
    # Output points at a fresh per-task copy under work_dir (a unique subdir),
    # not the canonical path.
    assert out.parent == work
    assert out.exists()

    # The copy contains every model from the canonical store.
    src = YAMLStorage(base_dir=str(canonical_root / DB_NAME))
    dst = YAMLStorage(base_dir=str(out))
    canonical_names = sorted(await src.list_models())
    variant_names = sorted(await dst.list_models())
    assert canonical_names == variant_names

    # And the datasource carries through.
    ds = await dst.get_datasource(DB_NAME)
    assert ds is not None and ds.name == DB_NAME


async def test_deletion_drops_model_by_kb_id(canonical_root: Path, tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids={1},
        work_dir=work,
    )
    variant = YAMLStorage(base_dir=str(out))
    names = await variant.list_models()
    assert "alpha" not in names
    assert {"beta", "gamma"} <= set(names)
    # Datasource is preserved.
    ds = await variant.get_datasource(DB_NAME)
    assert ds is not None and ds.name == DB_NAME


async def test_deletion_drops_model_by_kb_ids_list(
    canonical_root: Path, tmp_path: Path
):
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids={6},  # gamma has kb_ids=[5,6] — match on either
        work_dir=work,
    )
    variant = YAMLStorage(base_dir=str(out))
    names = await variant.list_models()
    assert "gamma" not in names
    assert "alpha" in names
    assert "beta" in names


async def test_deletion_drops_column_measure_aggregation(
    canonical_root: Path, tmp_path: Path
):
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids={2, 3, 4},
        work_dir=work,
    )
    variant = YAMLStorage(base_dir=str(out))
    beta = await variant.get_model("beta")
    assert beta is not None
    # The unmarked column + the PK column survive; "amount" was kb_id=2.
    col_names = {c.name for c in beta.columns}
    assert col_names == {"id", "plain"}
    # Only the unmarked measure survives; "total" was kb_id=3.
    assert {m.name for m in beta.measures} == {"rows"}
    # The custom aggregation was kb_id=4.
    assert beta.aggregations == []


async def test_unmarked_model_is_returned_unchanged_object(
    canonical_root: Path, tmp_path: Path
):
    """A model with no meta-matching entities should serialize identically."""
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids={2},  # touches only beta.amount; alpha + gamma untouched
        work_dir=work,
    )
    variant = YAMLStorage(base_dir=str(out))
    src = YAMLStorage(base_dir=str(canonical_root / DB_NAME))
    alpha_v = await variant.get_model("alpha")
    alpha_s = await src.get_model("alpha")
    assert alpha_v is not None and alpha_s is not None
    assert alpha_v.model_dump() == alpha_s.model_dump()


# ---------------------------------------------------------------------------
# build_task_variant_storage — memories + embeddings copy
# ---------------------------------------------------------------------------

async def _seed_memories_and_embeddings(canonical_root: Path) -> None:
    """Drop a ``memories.yaml`` + ``embeddings.db`` into the canonical
    store so the copy paths have something to copy. The memory bodies
    follow the kb-to-slayer-models skill convention (``KB <n> — ``).
    """
    import sqlite3

    db_root = canonical_root / DB_NAME

    # Two memories: one for KB 2 (will be deleted in the filter test),
    # one for KB 99 (always survives — not in any task's deleted set).
    # DEV-1668: slayer 0.9.6 stores per-id ``memories/<id>.md``; the int-suffix
    # id (``memory:1`` / ``memory:2``) keys the embedding rows below.
    write_memories_files(
        db_root,
        [
            {
                "version": 1,
                "id": "1",
                "learning": "KB 2 — Important value bands\n\nKB body...",
                "entities": [f"{DB_NAME}.beta.amount"],
                "query": None,
                "created_at": "2026-05-14T00:00:00Z",
            },
            {
                "version": 1,
                "id": "2",
                "learning": "KB 99 — Unrelated note\n\nKB body...",
                "entities": [f"{DB_NAME}.gamma"],
                "query": None,
                "created_at": "2026-05-14T00:00:00Z",
            },
        ],
    )

    # Three embedding rows:
    # - memory:1   (drops alongside KB 2 deletion)
    # - memory:2   (survives)
    # - tinydb.beta.amount  (entity row; unrelated to the memory filter)
    db_path = db_root / "embeddings.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                canonical_id TEXT NOT NULL,
                embedding_model_name TEXT NOT NULL,
                entity_kind TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (canonical_id, embedding_model_name)
            )
            """
        )
        rows = [
            ("memory:1", "test-model", "memory", "h1", "[]", "2026-05-14"),
            ("memory:2", "test-model", "memory", "h2", "[]", "2026-05-14"),
            (f"{DB_NAME}.beta.amount", "test-model", "column", "h3", "[]", "2026-05-14"),
        ]
        con.executemany(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)", rows
        )
        con.commit()
    finally:
        con.close()


async def test_variant_copies_memories_yaml(canonical_root: Path, tmp_path: Path):
    """Memories from canonical land in the variant's per-id memory store."""
    await _seed_memories_and_embeddings(canonical_root)
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=work,
    )
    assert {m.id for m in read_memories(out)} == {"1", "2"}


async def test_variant_copies_embeddings_db(canonical_root: Path, tmp_path: Path):
    """`embeddings.db` is copied verbatim when no deletions apply."""
    import sqlite3

    await _seed_memories_and_embeddings(canonical_root)
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=work,
    )
    con = sqlite3.connect(out / "embeddings.db")
    cids = {r[0] for r in con.execute("SELECT canonical_id FROM embeddings")}
    con.close()
    assert cids == {"memory:1", "memory:2", f"{DB_NAME}.beta.amount"}


async def test_variant_filters_memory_and_embedding_for_deleted_kb(
    canonical_root: Path, tmp_path: Path
):
    """When KB 2 is in `deleted_kb_ids`, the matching memory AND its
    `memory:1` embedding both drop. The KB-99 memory and its embedding
    survive, and non-memory embedding rows are left alone."""
    import sqlite3

    await _seed_memories_and_embeddings(canonical_root)
    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids={2},
        work_dir=work,
    )
    # KB 2's memory has id "1"; filtering by KB id 2 drops it, leaving KB 99 (id "2").
    assert {m.id for m in read_memories(out)} == {"2"}, "KB-2 memory should be filtered"

    con = sqlite3.connect(out / "embeddings.db")
    cids = {r[0] for r in con.execute("SELECT canonical_id FROM embeddings")}
    con.close()
    assert "memory:1" not in cids
    assert "memory:2" in cids
    # Non-memory entity row stays even when its kb is in the deleted set —
    # _apply_deletions filters at the YAML/model level; the embedding row
    # is inert (not referenced by the rebuilt corpus) so we leave it.
    assert f"{DB_NAME}.beta.amount" in cids


async def test_variant_prunes_string_id_memory_embedding(
    canonical_root: Path, tmp_path: Path
):
    """DEV-1668 (Codex): memories keyed with `<db>_kb_<n>` STRING ids (the
    encode_kb_as_memories scheme) must have their `memory:<db>_kb_<n>`
    embedding row pruned on KB deletion — the prune is id-scheme-agnostic and
    mirrors runtime._prune_deleted_memory_embeddings, not just the int-id case.
    """
    import sqlite3

    db_root = canonical_root / DB_NAME
    write_memories_files(db_root, [
        {"version": 1, "id": f"{DB_NAME}_kb_2",
         "learning": "KB 2 — Important value bands\n\nbody",
         "entities": [f"{DB_NAME}.beta.amount"],
         "created_at": "2026-05-14T00:00:00Z"},
        {"version": 1, "id": f"{DB_NAME}_kb_99",
         "learning": "KB 99 — Unrelated note\n\nbody",
         "entities": [f"{DB_NAME}.gamma"],
         "created_at": "2026-05-14T00:00:00Z"},
    ])
    db_path = db_root / "embeddings.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (canonical_id TEXT NOT NULL, "
        "embedding_model_name TEXT NOT NULL, entity_kind TEXT NOT NULL, "
        "content_hash TEXT NOT NULL, embedding TEXT NOT NULL, "
        "created_at TEXT NOT NULL, PRIMARY KEY (canonical_id, embedding_model_name))"
    )
    con.executemany("INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)", [
        (f"memory:{DB_NAME}_kb_2", "test-model", "memory", "h1", "[]", "2026-05-14"),
        (f"memory:{DB_NAME}_kb_99", "test-model", "memory", "h2", "[]", "2026-05-14"),
        (f"{DB_NAME}.beta.amount", "test-model", "column", "h3", "[]", "2026-05-14"),
    ])
    con.commit()
    con.close()

    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids={2}, work_dir=work,
    )
    assert {m.id for m in read_memories(out)} == {f"{DB_NAME}_kb_99"}

    con = sqlite3.connect(out / "embeddings.db")
    cids = {r[0] for r in con.execute("SELECT canonical_id FROM embeddings")}
    con.close()
    # The dropped string-id memory's embedding is pruned; the survivor + the
    # non-memory entity row stay.
    assert f"memory:{DB_NAME}_kb_2" not in cids
    assert f"memory:{DB_NAME}_kb_99" in cids
    assert f"{DB_NAME}.beta.amount" in cids


async def test_variant_omits_memories_when_canonical_lacks_them(
    canonical_root: Path, tmp_path: Path
):
    """No memories in canonical → variant doesn't get spurious ones.

    `embeddings.db` is created lazily by YAMLStorage's init regardless;
    if canonical's was absent the variant's stays empty (0 rows), which
    is functionally equivalent for the search service."""
    import sqlite3

    work = tmp_path / "work"
    work.mkdir()
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=work,
    )
    assert read_memories(out) == []
    if (out / "embeddings.db").exists():
        con = sqlite3.connect(out / "embeddings.db")
        try:
            (count,) = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        finally:
            con.close()
        assert count == 0


# ---------------------------------------------------------------------------
# build_task_variant_storage — datasource connection re-anchoring (DEV-1454)
# The portable connection string must resolve against the run's --db-path
# (mini_interact_root), not just the default sibling mini-interact/ — so a
# non-default dataset is queried consistently with how the reference was built.
# ---------------------------------------------------------------------------


async def test_variant_resolves_connection_against_mini_interact_root(
    canonical_root: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    custom = tmp_path / "custom_data"
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=tmp_path / "wd",
        mini_interact_root=custom,
    )
    ds = await YAMLStorage(base_dir=str(out)).get_datasource(DB_NAME)
    assert ds is not None
    assert ds.connection_string.startswith("sqlite:////")
    # anchored at the custom root …
    assert (custom / f"{DB_NAME}.db").resolve().as_posix().lstrip("/") \
        in ds.connection_string
    # … and NOT the sibling default
    sibling = (canonical_root.parent.parent / "mini-interact").resolve()
    assert sibling.as_posix().lstrip("/") not in ds.connection_string


async def test_variant_defaults_to_sibling_mini_interact_root(
    canonical_root: Path, tmp_path: Path, monkeypatch
):
    """No mini_interact_root → the sibling default is used (preserves the
    pre-existing recursive/pre-encoded behaviour)."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=tmp_path / "wd2",
    )
    ds = await YAMLStorage(base_dir=str(out)).get_datasource(DB_NAME)
    assert ds is not None
    sibling = (canonical_root.parent.parent / "mini-interact").resolve()
    assert sibling.as_posix().lstrip("/") in ds.connection_string


async def test_variant_bird_db_path_env_overrides_root(
    canonical_root: Path, tmp_path: Path, monkeypatch
):
    """$BIRD_DB_PATH still wins over mini_interact_root when set."""
    env_root = tmp_path / "env_data"
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))
    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root,
        db_name=DB_NAME,
        deleted_kb_ids=set(),
        work_dir=tmp_path / "wd3",
        mini_interact_root=tmp_path / "ignored_root",
    )
    ds = await YAMLStorage(base_dir=str(out)).get_datasource(DB_NAME)
    assert ds is not None
    assert (env_root / f"{DB_NAME}.db").resolve().as_posix().lstrip("/") \
        in ds.connection_string


async def test_variant_uses_fresh_unique_subdir_per_build(
    canonical_root: Path, tmp_path: Path
):
    """Each build materialises a UNIQUE scratch dir under work_dir (Codex
    finding): a prior run's leftovers can't leak in, AND two builds sharing a
    work_dir get distinct dirs so neither deletes the other's storage."""
    work = tmp_path / "work"
    # Leftover from a "prior run" directly under work_dir — must be ignored.
    (work / DB_NAME / "models" / DB_NAME).mkdir(parents=True)
    (work / DB_NAME / "models" / DB_NAME / "ghost.yaml").write_text("name: ghost\n")

    out1 = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=work,
    )
    out2 = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=work,
    )
    # fresh, distinct dirs under work_dir — neither clobbers the other
    assert out1 != out2
    assert out1.parent == work and out2.parent == work
    assert out1.exists() and out2.exists()
    # no stale ghost model leaked into either; canonical models present
    for out in (out1, out2):
        names = set(await YAMLStorage(base_dir=str(out)).list_models())
        assert "ghost" not in names
        assert {"alpha", "beta", "gamma"} <= names


# ---------------------------------------------------------------------------
# DEV-1478 cloud bug: the OTF deterministic cache bakes in an ABSOLUTE local
# sqlite path. `build_task_variant_storage` must FORCE-rewrite that path to
# the current root (mirroring the recursive path's `prepare_task_storage`),
# otherwise the foreign absolute path survives into the cloud container and
# every agent query fails "unable to open database file".
# ---------------------------------------------------------------------------


async def _seed_canonical_with_connection(
    canonical_root: Path, connection_string: str
) -> None:
    storage = YAMLStorage(base_dir=str(canonical_root / DB_NAME))
    await storage.save_datasource(
        DatasourceConfig(
            name=DB_NAME, type="sqlite",
            connection_string=connection_string,
        )
    )
    await storage.save_model(
        SlayerModel(
            name="alpha", data_source=DB_NAME, sql_table="alpha",
            columns=[Column(name="id", primary_key=True)],
        )
    )


async def test_variant_reanchors_stale_foreign_absolute_connection(
    tmp_path: Path, monkeypatch
):
    """A cache built on another machine carries an absolute path that does
    NOT exist in this environment. The variant's datasource must be
    re-anchored to `<mini_interact_root>/<db>/<db>.sqlite`."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    await _seed_canonical_with_connection(
        canonical_root,
        "sqlite:////home/someone/Dropbox/SLayer/mini-interact/"
        f"{DB_NAME}/{DB_NAME}.sqlite",
    )
    mini_root = tmp_path / "data" / "mini-interact"

    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=tmp_path / "work",
        mini_interact_root=mini_root,
    )
    ds = await YAMLStorage(base_dir=str(out)).get_datasource(DB_NAME)
    want = (
        "sqlite:////"
        + (mini_root / DB_NAME / f"{DB_NAME}.sqlite").resolve()
        .as_posix().lstrip("/")
    )
    assert ds is not None
    assert ds.connection_string == want, (
        f"stale foreign absolute path must be re-anchored to the current "
        f"root; got {ds.connection_string!r}"
    )
    assert "/home/someone/" not in (ds.connection_string or "")


async def test_variant_reanchors_honors_bird_db_path(tmp_path: Path, monkeypatch):
    """`$BIRD_DB_PATH` wins — mirrors the cloud container where
    BIRD_DB_PATH=/data/mini-interact even though the supplied root differs."""
    env_root = tmp_path / "data" / "mini-interact"
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    await _seed_canonical_with_connection(
        canonical_root, f"sqlite:////home/someone/x/{DB_NAME}/{DB_NAME}.sqlite",
    )

    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=tmp_path / "work",
        mini_interact_root=tmp_path / "stale-root",
    )
    ds = await YAMLStorage(base_dir=str(out)).get_datasource(DB_NAME)
    want = (
        "sqlite:////"
        + (env_root / DB_NAME / f"{DB_NAME}.sqlite").resolve()
        .as_posix().lstrip("/")
    )
    assert ds is not None and ds.connection_string == want


async def test_variant_reanchors_relative_connection_noop_equivalent(
    tmp_path: Path, monkeypatch
):
    """The pre-encoded relative form re-anchors to the same place the old
    resolve-only logic produced — confirms the fix is behaviour-preserving
    for the pre-encoded path."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    await _seed_canonical_with_connection(
        canonical_root, f"sqlite:///{DB_NAME}/{DB_NAME}.sqlite",
    )
    mini_root = tmp_path / "mini-interact"

    out = await build_task_variant_storage(
        canonical_storage_root=canonical_root, db_name=DB_NAME,
        deleted_kb_ids=set(), work_dir=tmp_path / "work",
        mini_interact_root=mini_root,
    )
    ds = await YAMLStorage(base_dir=str(out)).get_datasource(DB_NAME)
    want = (
        "sqlite:////"
        + (mini_root / DB_NAME / f"{DB_NAME}.sqlite").resolve()
        .as_posix().lstrip("/")
    )
    assert ds is not None and ds.connection_string == want
