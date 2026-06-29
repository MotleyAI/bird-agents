"""DEV-1589: the claude_sdk-native build-time OTF *reference* encoder.

Encodes the per-DB KB into a durable ``slayer_models_otf/<benchmark>/<db>/``
reference using ANY registry / open-weight model (e.g. ``zai/glm-5.2``) that
pydantic_ai cannot drive. The default builder used by
``scripts/build_otf_references.py``.
"""

from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
    make_claude_sdk_build_encoder,
)

__all__ = ["make_claude_sdk_build_encoder"]
