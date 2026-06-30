"""Framework-family predicates shared across the local runner and the cloud.

DEV-1609: a single source of truth for "is this framework an OTF *reference
encoder*?" — i.e. a run whose only durable output is the per-DB
``slayer_models_otf/<benchmark>/<db>`` reference that the cloud merge-back
uploads home. Both encode frameworks produce the identical artifact set
(``slayer_otf_cache`` immutable input + ``slayer_models_otf`` mutable
merge-back), so the cloud keys all of upload-selection, download-selection,
artifact-naming, the seed snapshot, and the merge-back guard off this one
predicate instead of scattering the literal framework string.

``claude_sdk_otf_encode`` is the default encoder going forward (DEV-1589/1609);
``pydantic_ai_otf_encode`` is retained ONLY so existing cloud manifests and
``resubmit`` keep resolving — it is deprecated as a code path.
"""

from __future__ import annotations

# Default (claude_sdk) + legacy (pydantic_ai, back-compat only).
OTF_ENCODE_FRAMEWORKS = frozenset(
    {
        "pydantic_ai_otf_encode",
        "claude_sdk_otf_encode",
    }
)


def is_otf_encode_framework(framework: str | None) -> bool:
    """True iff ``framework`` is an OTF *reference encoder* (its run merges a
    freshly built ``slayer_models_otf/<benchmark>/<db>`` reference back)."""
    return framework in OTF_ENCODE_FRAMEWORKS
