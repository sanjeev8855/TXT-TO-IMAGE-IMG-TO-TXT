"""Provider Abstraction Layer for Multimodal Research Architecture."""
from backend.providers.factory import (
    get_vision_provider,
    get_primary_i2t_provider,
    get_fallback_i2t_provider,
    get_llm_provider,
    get_embedding_provider,
    get_image_generation_provider,
    get_forensic_provider,
    get_all_provider_statuses
)

__all__ = [
    "get_vision_provider",
    "get_primary_i2t_provider",
    "get_fallback_i2t_provider",
    "get_llm_provider",
    "get_embedding_provider",
    "get_image_generation_provider",
    "get_forensic_provider",
    "get_all_provider_statuses",
]
