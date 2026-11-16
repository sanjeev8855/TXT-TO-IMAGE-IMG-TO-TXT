"""
Central Provider Factory (Section 14 & Model Provider Abstraction)
Provides unified access to Vision, Primary Image->Text (Llama), Fallback (Kimi),
LLM, Embeddings, Image Generation, and Forensics.
"""
from typing import Dict, Any
from backend.config import config
from backend.providers.gemini_provider import GeminiVisionProvider, GeminiLLMProvider, GeminiImageGenProvider
from backend.providers.llama_provider import llama_vision_provider
from backend.providers.kimi_provider import kimi_provider
from backend.forensic.detector import forensic_detector

def get_vision_provider():
    """Returns Primary Vision Provider: Gemini Vision."""
    return GeminiVisionProvider(api_key=config.vision_api_key, model=config.vision_model)

def get_primary_i2t_provider():
    """Returns Primary Image -> Text Provider: Llama 3.2 90B Vision Instruct."""
    return llama_vision_provider

def get_fallback_i2t_provider():
    """Returns Fallback Image -> Text & Reasoning Provider: Kimi K3."""
    return kimi_provider

def get_llm_provider():
    """Returns default LLM / Prompt Engineering Provider."""
    return GeminiLLMProvider(api_key=config.llm_api_key, model=config.llm_model)

def get_forensic_provider():
    """Returns Independent Forensic Detector."""
    return forensic_detector

def get_embedding_provider():
    """Returns Text Embedding Provider."""
    from backend.rag.embeddings import embedding_engine
    return embedding_engine

def get_image_generation_provider():
    """Returns Text-to-Image Generation Provider."""
    from backend.text_to_image.generator import image_generator
    return image_generator

def get_all_provider_statuses() -> Dict[str, Any]:
    """Returns live status of all models and providers for UI status badges and research audit."""
    return {
        "primary_vision": {
            "model": config.vision_model,
            "provider": "Gemini Vision",
            "configured": bool(config.vision_api_key),
            "role": "Primary Visual Observation Stage"
        },
        "primary_image_to_text": {
            "model": config.llama_model,
            "provider": "Llama 3.2 90B Vision Instruct",
            "configured": bool(config.llama_api_key),
            "role": "Primary Image-to-Text Model"
        },
        "fallback_image_to_text": {
            "model": config.kimi_model,
            "provider": "Kimi K3",
            "configured": bool(config.kimi_api_key),
            "role": "Automatic Fallback & Prompt Reasoning"
        },
        "image_generation": {
            "model": config.image_generation_model,
            "provider": "NVIDIA GenAI (FLUX.1-dev)" if config.image_generation_api_key else "Latent Diffusion / Imagen 3",
            "configured": bool(config.image_generation_api_key or config.image_generation_provider == "local"),
            "role": "Iterative Latent Synthesis"
        },
        "forensic_detector": {
            "model": config.forensic_model,
            "provider": "Multi-Stream Spatial / 2D FFT / Noise",
            "configured": True,
            "role": "Independent Forensic AI Detector"
        }
    }
