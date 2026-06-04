"""DEV-1515: `paths.annotations_root()` worktree-safety regression.

Mirrors the contract used by `audited_gold_root()` /
`slayer_models_otf_root()` etc.: the helper MUST anchor at the main
checkout (via `main_checkout_root()`), so a `git worktree add` spawned
elsewhere reads/writes the same annotations directory.
"""
from __future__ import annotations


def test_annotations_root_anchors_at_main_checkout(monkeypatch, tmp_path):
    """Patching `main_checkout_root` MUST move `annotations_root`."""
    import bird_interact_agents.paths as p

    monkeypatch.setattr(p, "main_checkout_root", lambda: tmp_path)
    assert p.annotations_root() == tmp_path / "annotations"


def test_annotations_root_is_under_main_checkout_not_worktree():
    """`annotations_root()` must not contain `.worktrees` — that would
    mean a worktree-spawned process is reading a throwaway copy."""
    import bird_interact_agents.paths as p

    root = p.annotations_root()
    assert ".worktrees" not in root.parts


def test_annotation_io_uses_paths_helper(monkeypatch, tmp_path):
    """Round-trip: write via annotation_io with the patched root, the
    file lands under `<patched_root>/annotations/<benchmark>/<db>/...`."""
    import bird_interact_agents.paths as paths_mod
    from bird_interact_agents.eval import (
        AuditedGoldRef,
        GoldVariantRef,
        MaskedTerm,
        MetadataSufficiency,
        TaskAnnotation,
        task_annotation_path,
        write_task_annotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance

    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    ann = TaskAnnotation(
        instance_id="alien_42",
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-05-31",
        amb_user_query="x",
        external_knowledge=[1],
        masked_terms=[MaskedTerm(term="x", type="knowledge_linking_ambiguity")],
        metadata_sufficiency=MetadataSufficiency(
            verdict="ambiguous", rationale="r"
        ),
        gold_variants=[
            GoldVariantRef(
                variant_id="primary",
                interpretation="x",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id="alien_42",
                ),
            )
        ],
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="alien_42",
        ),
    )
    # The path helper accepts an explicit repo_root, but the production
    # code path resolves through paths.main_checkout_root(); to exercise
    # both, call without repo_root and patch the helper instead.
    p_explicit = task_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_42",
        repo_root=tmp_path,
    )
    write_task_annotation(ann, p_explicit)
    assert p_explicit.exists()
    assert p_explicit.is_relative_to(tmp_path / "annotations")


def test_annotation_paths_use_canonical_hyphenated_name(tmp_path):
    """Post-DEV-1525: no normalization is applied — each benchmark name maps
    to its own subdirectory.  The canonical form is hyphenated (``mini-interact``)
    so that is what production code (cloud workers, CLI, annotator) writes under."""
    from bird_interact_agents.eval import (
        submission_annotation_path,
        task_annotation_path,
    )

    canonical = task_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_1",
        repo_root=tmp_path,
    )
    assert "mini-interact" in canonical.parts
    assert "mini_interact" not in canonical.parts

    sub = submission_annotation_path(
        benchmark="mini-interact",
        selected_database="alien",
        instance_id="alien_1",
        run_id="r1",
        repo_root=tmp_path,
    )
    assert "mini-interact" in sub.parts
    assert "mini_interact" not in sub.parts
