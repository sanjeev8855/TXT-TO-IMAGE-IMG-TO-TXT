from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.image_to_text.vlm import VLMStructuredOutput
from backend.rag.prompt_builder import PromptValidationResult

class SystemStatusResponse(BaseModel):
    models: Dict[str, str]
    gemini_api_configured: bool
    reproducibility: Dict[str, Any]
    rag_document_count: int
    forensic_config: Dict[str, Any]
    model_providers: Optional[Dict[str, Any]] = None

class ImageUploadResponse(BaseModel):
    filename: str
    file_url: str
    original_dimensions: List[int]
    processed_dimensions: List[int]
    format: str
    mean_intensity: float
    std_intensity: float
    exif_stripped: bool

class VLMAnalysisRequest(BaseModel):
    filename: str
    force_llama_failure: Optional[bool] = False
    force_kimi_failure: Optional[bool] = False

class RAGRetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    category_filter: Optional[str] = None

class RAGRetrieveResponse(BaseModel):
    query: str
    top_k: int
    retrieved_documents: List[Dict[str, Any]]
    context_summary: str

class PromptBuildRequest(BaseModel):
    vlm_output: VLMStructuredOutput
    retrieved_documents: List[Dict[str, Any]]
    user_instructions: Optional[str] = None

class TextToImageRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    steps: Optional[int] = None

class ForensicAnalysisRequest(BaseModel):
    filename: str  # Can be uploaded or generated
    alpha: Optional[float] = 0.45

class FullPipelineRequest(BaseModel):
    filename: str
    user_instructions: Optional[str] = None
    rag_top_k: Optional[int] = 5
    generation_seed: Optional[int] = None
    generation_steps: Optional[int] = None
    force_llama_failure: Optional[bool] = False
    force_kimi_failure: Optional[bool] = False

class BidirectionalCompareRequest(BaseModel):
    original_filename: str
    generated_filename: str
    original_description: Optional[str] = None
    generated_prompt: Optional[str] = None
