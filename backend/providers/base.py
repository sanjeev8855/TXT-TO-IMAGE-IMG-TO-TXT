from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

class BaseVisionProvider(ABC):
    """Abstract interface for Vision / Image Understanding providers."""
    @abstractmethod
    def analyze_image(self, pil_image: Image.Image, np_array: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Performs vision-language analysis and returns structured visual descriptors."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class BaseLLMProvider(ABC):
    """Abstract interface for LLM / RAG / Prompt Engineering providers."""
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2
    ) -> str:
        """Generates text or structured JSON from prompt."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class BaseEmbeddingProvider(ABC):
    """Abstract interface for Embedding providers."""
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds single text string into dense vector representation."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

class BaseImageGenerationProvider(ABC):
    """Abstract interface for Text-to-Image generation providers."""
    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        seed: int,
        steps: int = 30,
        resolution: Tuple[int, int] = (512, 512),
        guidance_scale: float = 7.5
    ) -> Tuple[Image.Image, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Synthesizes an RGB image from prompt.
        Returns: (PIL.Image, metadata_dict, steps_telemetry_list)
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class BaseForensicProvider(ABC):
    """Abstract interface for Forensic AI image detectors."""
    @abstractmethod
    def evaluate_image(
        self,
        np_image: np.ndarray,
        patch_size: int = 64,
        stride: int = 32,
        include_frequency: bool = True,
        include_noise_residual: bool = True
    ) -> Dict[str, Any]:
        """
        Performs independent patch-level forensic evaluation.
        Returns score matrix, patch records, statistics, and frequency/residual arrays.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
