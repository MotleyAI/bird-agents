#!/usr/bin/env python
"""Batch-build the LLM-encoded SLayer references for every DB in a benchmark
(DEV-1586).

The pre-encoded ``claude_sdk`` agents (``--pre-encoded-models otf``) consume
an ALREADY-encoded reference at
``slayer_models_otf/<benchmark>/<db>/`` (marker ``_reference_fp.txt``). That
reference is produced by RUNNING the LLM setup-encoder — there was previously
no single entry point to build it for a whole benchmark (``build_local_otf_
cache.sh`` only builds the deterministic, LLM-free cache that is the encoder's
*input*). This script fills that gap: it enumerates the benchmark's DBs and,
for each, ensures the deterministic cache then runs the setup-encoder to
materialise the reference (idempotent — a DB whose ``_reference_fp.txt`` is
present is skipped unless ``--force``).

It reuses the exact encoder wiring of the ``pydantic_ai_otf_encode`` agent
(``make_setup_build_encoder`` + ``reference_build.ensure_db_reference``), so
the references it builds are byte-identical to the ones a cloud
``pydantic_ai_otf_encode`` submit would merge back.

For postgres benchmarks, start the user-space postgres first with
``scripts/build_local_otf_cache.sh <benchmark>`` (this script reuses that
running instance via the same ``BIRD_PG_*`` env vars).

Usage:
    uv run python scripts/build_otf_references.py <benchmark> \
        --agent-model anthropic/claude-opus-4-7 [--force] [--only db1,db2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import (
    build_pydantic_ai_model,
    encoder_version_slug,
    is_anthropic,
    native_model_id,
)
from bird_interact_agents.slayer_otf import ensure_db_reference
from bird_interact_agents.slayer_otf.encoder_types import EncoderMetaSettings


def resolve_build_version(
    *, agent_model: str, version_arg: str | None,
) -> tuple[str, bool]:
    """DEV-1605: resolve the version label for a build + whether it was
    operator-supplied. Default = the encoder-model slug.

    An explicit ``--version ""`` (empty string, e.g. an unset env expansion)
    is preserved as operator-supplied — it is NOT silently treated as "flag
    omitted", so it fails loudly at the path-label validator instead of
    building into the default slot with the wrong provenance (CodeRabbit)."""
    if version_arg is not None:
        return version_arg, True
    return encoder_version_slug(agent_model), False

# The two encoder frameworks live behind SEPARATE optional extras
# (`claude-sdk` vs `pydantic-ai`). Guard the imports so building with one
# framework does not require the OTHER framework's extra to be installed
# (Codex review). `make_build_encoder` raises a clear error if the chosen
# framework's dependency is missing. Imports stay at module top per repo style.
try:
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        make_claude_sdk_build_encoder,
    )
except ImportError as _e:  # pragma: no cover - exercised only without the extra
    make_claude_sdk_build_encoder = None
    _CLAUDE_SDK_IMPORT_ERROR = _e
else:
    _CLAUDE_SDK_IMPORT_ERROR = None

try:
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _build_shared_slayer_server,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.setup_encoder import (
        make_setup_build_encoder,
    )
except ImportError as _e:  # pragma: no cover - exercised only without the extra
    _build_shared_slayer_server = None
    make_setup_build_encoder = None
    _PYDANTIC_AI_IMPORT_ERROR = _e
else:
    _PYDANTIC_AI_IMPORT_ERROR = None

logger = logging.getLogger("build_otf_references")


def _dbs_for_benchmark(benchmark_name: str) -> list[str]:
    """Unique ``selected_database`` values from the benchmark's data file."""
    bench = get_benchmark(benchmark_name)
    data_file = paths.benchmark_data_root(benchmark_name) / bench.data_file
    dbs = {
        json.loads(line)["selected_database"]
        for line in data_file.read_text().splitlines()
        if line.strip()
    }
    return sorted(dbs)


def _build_model(agent_model: str):
    """Construct the pydantic-ai model + cache settings + native id for the
    setup-encoder, mirroring ``PydanticAIOtfEncodeAgent.__init__``."""
    model_settings = None
    if is_anthropic(agent_model):
        try:
            from bird_interact_agents.agents.pydantic_ai.agent import (
                _anthropic_cache_settings,
            )
            model_settings = _anthropic_cache_settings()
        except Exception:  # noqa: BLE001 — caching is best-effort
            model_settings = None
    return build_pydantic_ai_model(agent_model), model_settings


def make_build_encoder(framework: str, agent_model: str):
    """Select the setup-encoder for ``framework`` (DEV-1589).

    ``claude_sdk`` (default) drives ANY registry/open-weight model end-to-end —
    it gets the RAW model string (e.g. ``zai/glm-5.2``), NOT a built pydantic-ai
    model (pydantic_ai can't reach z.ai). ``pydantic_ai`` keeps the legacy path.
    """
    if framework == "claude_sdk":
        if make_claude_sdk_build_encoder is None:
            raise SystemExit(
                "--encoder-framework claude_sdk needs the `claude-sdk` extra "
                f"(uv sync --extra claude-sdk). Import failed: "
                f"{_CLAUDE_SDK_IMPORT_ERROR}"
            )
        return make_claude_sdk_build_encoder(
            model=agent_model,
            self_model_id=native_model_id(agent_model),
        )
    if framework == "pydantic_ai":
        if make_setup_build_encoder is None:
            raise SystemExit(
                "--encoder-framework pydantic_ai needs the `pydantic-ai` extra "
                f"(uv sync --extra pydantic-ai). Import failed: "
                f"{_PYDANTIC_AI_IMPORT_ERROR}"
            )
        model, model_settings = _build_model(agent_model)
        return make_setup_build_encoder(
            model=model,
            model_settings=model_settings,
            self_model_id=native_model_id(agent_model),
            build_shared_slayer_server=_build_shared_slayer_server,
        )
    raise ValueError(f"unknown encoder framework: {framework!r}")


async def _build_one(
    db: str, *, benchmark_name: str, build_encoder, force: bool,
    encoder_model: str, encoder_framework: str, version: str,
    version_was_explicit: bool,
) -> None:
    bench = get_benchmark(benchmark_name)
    data_root = paths.benchmark_data_root(benchmark_name)
    entry = await ensure_db_reference(
        db,
        # DEV-1605: build into the version-scoped root
        # slayer_models_otf/<benchmark>/<version>/<db>.
        reference_root=paths.slayer_models_otf_root(
            benchmark=benchmark_name, version=version,
        ),
        cache_root=paths.slayer_otf_cache_root(benchmark=benchmark_name),
        mini_interact_root=data_root,
        build_encoder=build_encoder,
        force=force,
        db_root=data_root,
        benchmark=bench,
        encoder_model=encoder_model,
        encoder_framework=encoder_framework,
        version=version,
        encoder_settings=EncoderMetaSettings(
            version_was_explicit=version_was_explicit,
        ),
    )
    logger.info("  %s -> %s", db, entry.reference_dir)


async def _main_async(args: argparse.Namespace) -> int:
    benchmark_name = get_benchmark(args.benchmark).name
    dbs = _dbs_for_benchmark(benchmark_name)
    if args.only:
        wanted = {d.strip() for d in args.only.split(",") if d.strip()}
        dbs = [d for d in dbs if d in wanted]
        missing = wanted - set(dbs)
        if missing:
            raise SystemExit(
                f"--only names DBs not in {benchmark_name}: {sorted(missing)}"
            )
    if not dbs:
        raise SystemExit(f"no databases found for benchmark {benchmark_name!r}")

    build_encoder = make_build_encoder(args.encoder_framework, args.agent_model)
    version, version_was_explicit = resolve_build_version(
        agent_model=args.agent_model, version_arg=args.version,
    )

    logger.info(
        "Building OTF references for %d DB(s) in %s "
        "(encoder=%s, version=%s, force=%s): %s",
        len(dbs), benchmark_name, args.encoder_framework, version,
        args.force, dbs,
    )
    for db in dbs:
        logger.info("encoding %s ...", db)
        await _build_one(
            db, benchmark_name=benchmark_name,
            build_encoder=build_encoder, force=args.force,
            encoder_model=args.agent_model,
            encoder_framework=args.encoder_framework,
            version=version, version_was_explicit=version_was_explicit,
        )
    logger.info(
        "Done. References at %s",
        paths.slayer_models_otf_root(benchmark=benchmark_name, version=version),
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark", help="Benchmark registry token.")
    p.add_argument(
        "--agent-model", required=True,
        help="Encoder model (LiteLLM-style provider/model_id).",
    )
    p.add_argument(
        "--encoder-framework", choices=["claude_sdk", "pydantic_ai"],
        default="claude_sdk",
        help="Which setup-encoder builds the reference (DEV-1589). Default "
             "claude_sdk drives any registry/open-weight model; pydantic_ai is "
             "the legacy Anthropic-only path.",
    )
    p.add_argument(
        "--version", default=None,
        help="DEV-1605: version label for this build (the subdir under "
             "slayer_models_otf/<benchmark>/). Default = the --agent-model "
             "slug (e.g. opus-4-7, glm-5.2). Pass an explicit label for two "
             "builds with the same model but different settings.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild even if this version's _reference_fp.txt marker is "
             "present (rebuilds ONLY the targeted version slot).",
    )
    p.add_argument(
        "--only", default=None,
        help="Comma-separated subset of DB names to build (default: all).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
